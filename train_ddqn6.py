import argparse
import random
import time
from datetime import datetime
import os
from collections import deque
from dataclasses import dataclass
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class DuelingDQN(nn.Module):
    def __init__(self, in_dim=24, out_dim=5):
        super().__init__()
        self.out_dim = out_dim
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        v = self.value_stream(x)
        adv = self.adv_stream(x)
        return v + (adv - adv.mean(dim=1, keepdim=True))


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, cap=500_000):
        self.buffer = deque(maxlen=cap)
    
    def add(self, t: Transition):
        self.buffer.append(t)
    
    def sample(self, batch_size):
        items = random.sample(self.buffer, batch_size)
        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)

class EpsilonGreedyWrapper:
    """
    A wrapper to handle the stateful decay of epsilon.
    """
    def __init__(self, epsilon_start, epsilon_end, max_episodes, decay_type='exponential'):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.max_episodes = max_episodes
        self.epsilon = epsilon_start
        self.decay_type = decay_type

    def get_epsilon(self):
        return self.epsilon

    def decay(self):
        if self.decay_type == 'linear':
            self.epsilon += (self.epsilon_end - self.epsilon_start) / self.max_episodes
            self.epsion = max(self.epsilon_end, self.epsilon)
        else:
            self.epsilon *= (self.epsilon_end / self.epsilon_start) ** (1 / self.max_episodes)
            self.epsilon = max(self.epsilon_end, self.epsilon)
        return self.epsilon

    def select_action(self, state, net, device=None):
        if np.random.random() < self.epsilon:
            return np.random.randint(net[-1].out_features if isinstance(net, nn.Sequential) else net.out_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0) 
            if device:
                state = state.to(device)
            with torch.no_grad():
                q_values = net(state)
            return q_values.argmax(dim=1).item()

class PrioritizedReplay:
    def __init__(self, bufferSize: int = 500_000, alpha = 0.6, beta = 0.4, prioritized=True, **kwargs):
        self.buffer = deque(maxlen=bufferSize)
        self.bufferSize = bufferSize
        # self.priorities = deque(maxlen=bufferSize)
        self.prioritized = kwargs.get('prioritized', True)
        self.alpha = kwargs.get('alpha', 0.6)
        self.beta = kwargs.get('beta', 0.4)
        self.beta_rate = kwargs.get('beta_rate', 1e-5)
        if self.prioritized:
            self.priorities = deque(maxlen=bufferSize)
        self.pos = 0
        return
        
    def store(self, experience):
        
        self.buffer.append(experience)
        if self.prioritized:
            max_priority = max(self.priorities, default=1.0)
            self.priorities.append(max_priority)
        
        return
    
    def update(self, indices, priorities):
        
        if not self.prioritized:
            return

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5
        return

    def collectExperiences(self, env, state, explorationStrategy, countExperiences, net = None):
        
        for _ in range(countExperiences):
            if isinstance(explorationStrategy, EpsilonGreedyWrapper):
                action = explorationStrategy.select_action(state, net)
            else:
                action = explorationStrategy(net, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            experience = (state, action, reward, next_state, done)
            self.store(experience)
            state = next_state

            if done:
                break

    def sample(self, batchSize, **kwargs):
        if self.prioritized:

            priorities = np.array(self.priorities)
            probs = priorities ** self.alpha
            probs = probs / probs.sum()

            indices = np.random.choice(len(self.buffer), batchSize, p=probs)

            experiences = [self.buffer[i] for i in indices]

            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights = weights / weights.max()
            self.beta = min(1.0, self.beta + self.beta_rate)
            return experiences, indices, weights
        else:
            indices = np.random.choice(len(self.buffer), batchSize, replace=False)
            experiences = [self.buffer[i] for i in indices]
            return experiences, indices, None

    def splitExperiences(self, experiences):

        states, actions, rewards, next_states, dones = zip(*experiences)

        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.uint8))

    def length(self):
        return len(self.buffer)

step_counter = 0

def get_inference_obs(raw_s, prev_raw_s, step_counter, consecutive_fw, consecutive_turns, 
                       ir_latch_count, last_action, wide_arc_counters):
    """
    State Dimension: 38
    """
    front_arc_p = min(1.0, wide_arc_counters[0] / 15.0)
    left_arc_p  = min(1.0, wide_arc_counters[1] / 15.0)
    right_arc_p = min(1.0, wide_arc_counters[2] / 15.0)
    
    is_turning = 1.0 if last_action in [0, 1, 3, 4] else -1.0
    is_gliding = 1.0 if (left_arc_p > 0.5 or right_arc_p > 0.5) else -1.0

    right_near = np.sum(raw_s[0:4][1::2]) / 2.0
    left_near  = np.sum(raw_s[12:16][1::2]) / 2.0
    front_near = np.sum(raw_s[4:12][1::2]) / 4.0
    front_far  = np.sum(raw_s[4:12][::2]) / 4.0
    
    path_is_clear = 1.0 if (np.sum(raw_s[4:12]) < 0.1) else -1.0

    latch_confidence = min(1.0, ir_latch_count / 10.0)
    beacon_signal = (0.5 + 0.5 * latch_confidence) if raw_s[16] > 0.5 else -1.0

    t_s_phase = (step_counter % 80) / 80.0
    t_s_sin, t_s_cos = np.sin(2*np.pi*t_s_phase), np.cos(2*np.pi*t_s_phase)
    
    t_l_phase = (step_counter % 400) / 400.0
    t_l_sin, t_l_cos = np.sin(2*np.pi*t_l_phase), np.cos(2*np.pi*t_l_phase)

    rot_fatigue = min(1.0, consecutive_turns / 20.0) # Scale down from 5.0 to 1.0
    fw_momentum = min(1.0, consecutive_fw / 20.0)
    signal_delta = np.clip(np.sum(raw_s[4:12]) - np.sum(prev_raw_s[4:12]), -1.0, 1.0)

    lat_pressure = (left_near + left_arc_p*0.5) - (right_near + right_arc_p*0.5)
    lat_pressure = np.clip(lat_pressure, -2.0, 2.0)

    return np.concatenate([
        raw_s,               
        [lat_pressure],      
        [front_near],        
        [front_far],         
        [path_is_clear],     
        [beacon_signal],     
        [rot_fatigue],       
        [latch_confidence],  
        [t_s_sin], [t_s_cos],
        [t_l_sin], [t_l_cos],
        [front_arc_p],       
        [left_arc_p],        
        [right_arc_p],       
        [is_gliding],        
        [is_turning],        
        [fw_momentum],       
        [signal_delta],      
        [np.sum(raw_s[12:16][::2]) / 2.0], 
        [np.sum(raw_s[0:4][::2]) / 2.0]    
    ]).astype(np.float32)

def calculate_shaped_reward(env_reward, next_aug_state, action, latch, fw, turns):

    if next_aug_state[16] > 0.5:
        latch = min(20, latch + 2)
    else:
        latch = max(0, latch - 1)

    if action == 2: # Forward
        fw += 1; turns = 0
    else:
        turns += 1; fw = 0

    is_facing = next_aug_state[16] > 0.5
    shaped = 0.0

    if is_facing:
        shaped += 2.5 if action == 2 else -0.5
    else:
        if action == 2:
            shaped -= 0.1
        else:
            shaped += 0.02

    if abs(next_aug_state[35]) < 0.02 and action != 2:
        shaped -= 0.5

    if env_reward <= -100: final_r = -5.0
    elif env_reward >= 100: final_r = 15.0
    else: final_r = (env_reward / 10.0) + shaped

    return float(final_r), latch, fw, turns

def record_video(env_params, model, device, episode, filename="d3qn", episodes=20):
    """Generates an MP4 video of the trained agent."""
    filename = f"{filename}{episode}.mp4"
    print(f"Recording to {filename}...")
    temp_env = OBELIX(**env_params)
    raw_s = temp_env.reset()
    h, w, _ = temp_env.frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
    prev_raw_s = np.zeros(18)
    
    model.eval()
    for ep in range(episodes):
        print(f"Recording Episode {ep+1}/{episodes}")
        raw_s = temp_env.reset()
        done = False
        steps = 0
        ir_latch_count, consecutive_fw, consecutive_turns = 0, 0, 0
        arc_counters = [0, 0, 0]
        last_actions = deque(2*[10], maxlen=10)
        while not done:
            steps += 1
            s = get_inference_obs(raw_s, prev_raw_s, step_counter, consecutive_fw, 
                                        consecutive_turns, ir_latch_count, 2, arc_counters)
            st = torch.FloatTensor(s).unsqueeze(0).to(device)
            with torch.no_grad():
                a = model(st).argmax(1).item()
                is_turning = a in [0,1,3,4]
                is_fw = a == 2
                if is_turning:
                    consecutive_fw = 0
                    consecutive_turns += 1
                if is_fw:
                    consecutive_fw += 1
                    consecutive_turns = 0
            last_actions.append(a)
            raw_s2, _, done = temp_env.step(ACTIONS[a], render=False)
            if raw_s2[16] > 0.5:
                ir_latch_count = min(20, ir_latch_count + 2)
            else:
                ir_latch_count = max(0, ir_latch_count - 1)
                
            if np.sum(raw_s2[4:12][::2]) > 0.8 and a != 2 and not raw_s2[16] > 0.5:
                arc_counters[0] += 1
            else: arc_counters[0] = 0
            arc_counters[1] = arc_counters[1] + 1 if (np.sum(raw_s2[12:16][::2]) > 0.8 and a != 2) else 0
            arc_counters[2] = arc_counters[2] + 1 if (np.sum(raw_s2[0:4][::2]) > 0.8 and a != 2) else 0

            if a == 2:
                consecutive_fw += 1
                consecutive_turns = 0
            else:
                consecutive_turns += 1
                consecutive_fw = 0
            prev_raw_s = raw_s
            raw_s = raw_s2
            out.write(temp_env.frame)
            
    out.release()
    print("Video saved successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "record"], default="train")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default="d3qn_pth")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log folder")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--tau", type=float, default=0.005)
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    env_config = {
        "scaling_factor": 5,
        "arena_size": 500,
        "wall_obstacles": True,
        "difficulty": 1,
        "box_speed": 10,
    }
    env = OBELIX(**env_config)


    q_net = DuelingDQN(38, 5).to(device)
    # q_net.load_state_dict(torch.load("ob5weights899.pth", map_location=device))
    # prio_replay = pickle.load(open("ob4weights499_replay.pkl", "rb"))
    t_net = DuelingDQN(38, 5).to(device)
    t_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    
    if args.mode == "record":
        if os.path.exists(args.out):
            # q_net.load_state_dict(torch.load(args.out))
            q_net.load_state_dict(torch.load('ob6weights49.pth', map_location=device))
            record_video(env_config, q_net, device, episode=1)
        else:
            print("No weights found. Train first!")
        return

    # eps_greedy_action = EpsilonGreedyWrapper(epsilon_start=1.0, epsilon_end=0.01, max_episodes=400)
    prio_replay = PrioritizedReplay()
    # prio_replay.buffer = pickle.load(open("ob5weights899_buffer.pkl", "rb"))
    # prio_replay.priorities = pickle.load(open("ob5weights899_priorities.pkl", "rb"))
    # print(f"Loaded replay buffer with {len(prio_replay.buffer)} experiences.")
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join("runs", f"PPO_{current_time}")
    writer = SummaryWriter(log_dir=log_dir)
    steps = 0
    wall_clock = time.time()
    prev_raw_s = np.zeros(18)
    last_action = 2
    consecutive_fw = 0
    consecutive_turns = 0
    BASE_EPS = 0.01
    for ep in range(args.episodes):
        ep_start_time = time.time()
        ep_loss = torch.tensor(0.0)
        eps = BASE_EPS
        ir_latch_count, consecutive_fw, consecutive_turns = 0, 0, 0
        arc_counters = [0, 0, 0]
        raw_s = env.reset()
        actions_list = deque([2]*10, maxlen=10)        
        s = state = get_inference_obs(raw_s, prev_raw_s, step_counter, consecutive_fw, 
                                        consecutive_turns, ir_latch_count, 2, arc_counters)
        ep_ret = 0
        consecutive_turns = 0
        consecutive_fw = 0
 
        stuck_2 = False
        for step in range(args.max_steps):
            is_stuck = (raw_s[17] > 0.5) if step == 0 else stuck_2
            eps = max(BASE_EPS, 1 - steps / 20000)
            if np.random.random() < eps:
                if np.random.random() < 0.75:
                    a = 2
                else:
                    a = np.random.choice([0,1,3,4])
            else :
                with torch.no_grad():
                    st = torch.FloatTensor(s).unsqueeze(0).to(device)
                    a = q_net(st).argmax(1).item()
            action_str = ACTIONS[a]
            actions_list.append(a)
            is_forward = (action_str == "FW")
            is_turning = action_str in ["L45", "L22", "R22", "R45"]

            if is_forward:
                consecutive_fw += 1
                consecutive_turns = 0
            else:
                consecutive_fw = 0
                consecutive_turns += 1
            raw_s2, r_env, done = env.step(action_str)
            s2 = get_inference_obs(raw_s2, prev_raw_s, step_counter, consecutive_fw, 
                                        consecutive_turns, ir_latch_count, 2, arc_counters) 

            if raw_s2[16] > 0.5:
                ir_latch_count = min(20, ir_latch_count + 2)
            else:
                ir_latch_count = max(0, ir_latch_count - 1)
                
            if np.sum(raw_s2[4:12][::2]) > 0.8 and a != 2 and not raw_s2[16] > 0.5:
                arc_counters[0] += 1
            else: arc_counters[0] = 0
            arc_counters[1] = arc_counters[1] + 1 if (np.sum(raw_s2[12:16][::2]) > 0.8 and a != 2) else 0
            arc_counters[2] = arc_counters[2] + 1 if (np.sum(raw_s2[0:4][::2]) > 0.8 and a != 2) else 0

            if a == 2:
                consecutive_fw += 1
                consecutive_turns = 0
            else:
                consecutive_turns += 1
                consecutive_fw = 0
            last_action = a
            total_reward = r_env

            # if step % 50 == 0:
            #     print(f"Dist: {d_pre:.1f} | Beacon: {s2[-1]:.2f} | Action: {ACTIONS[a]}")
            prio_replay.store((s, a, total_reward, s2, done))

            steps += 1
            exp = prio_replay.buffer[-1]
            ep_ret += exp[2]
            s = exp[3]
            done = exp[4]
            

            if len(prio_replay.buffer) > args.batch:
                experiences, indices, weights = prio_replay.sample(args.batch)
                sb, ab, rb, s2b, db = prio_replay.splitExperiences(experiences)
                # sb, ab, rb, s2b, db = prio_replay.sample(args.batch)

                sb_t = torch.FloatTensor(sb).to(device)
                ab_t = torch.LongTensor(ab).to(device)
                rb_t = torch.FloatTensor(rb).to(device)
                s2b_t = torch.FloatTensor(s2b).to(device)
                db_t = torch.FloatTensor(db).to(device)

                with torch.no_grad():
                    next_actions = q_net(s2b_t).argmax(1, keepdim=True)
                    next_q = t_net(s2b_t).gather(1, next_actions).squeeze()
                    targets = rb_t + 0.99 * (1 - db_t) * next_q

                current_q = q_net(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze()
                # loss = F.smooth_l1_loss(current_q, targets)
                td_errors = torch.abs(current_q - targets).detach().cpu().numpy()
                prio_replay.update(indices, td_errors)
                
                loss = (torch.FloatTensor(weights).to(device) * F.smooth_l1_loss(current_q, targets, reduction='none')).mean()
                
                ep_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
                optimizer.step()

                # print(env.get_feedback())
                writer.add_scalar("Loss", loss.item(), steps)
                writer.add_scalar("Reward", total_reward, steps)
                writer.add_scalar("Ep_Return", ep_ret, steps)
                if steps % 20 == 0:
                    # t_net.load_state_dict(q_net.state_dict())
                    for target_param, online_param in zip(q_net.parameters(), t_net.parameters()):
                        target_param.data.copy_(args.tau * online_param.data + (1.0 - args.tau) * target_param.data)
            if done: 
                break
        
        ep_end_time = time.time()
        ep_time = ep_end_time - ep_start_time
        wall_time = ep_end_time - wall_clock

        if (ep + 1) % 1 == 0:
            # print(f"Ep: {ep+1} | Return: {ep_ret:.2f} | Loss: {ep_loss:.2f} | Eps: {eps:.2f} | Ep Time: {ep_time:.2f}")

        # if (ep + 1) % 10 == 0:
        #     save_heatmap(ep+1, env_config)
            print(f"Ep: {ep+1}/{args.episodes} | Return: {ep_ret:.2f} | Eps: {eps:.2f} | Ep Time: {ep_time:.2f} | Wall Time: {wall_time/60:.2f} min | Replay: {len(prio_replay.buffer)}")
        if (ep + 1) % 100 == 0:
            torch.save(q_net.state_dict(), f"d3qn_{ep}.pth")
            record_video(env_config, q_net, device, episode=ep+1)
            pickle.dump(prio_replay.buffer, open(f"d3qn_{ep}_buffer.pkl", "wb"))
            pickle.dump(prio_replay.priorities, open(f"d3qn_{ep}_priorities.pkl", "wb"))


    torch.save(q_net.state_dict(), args.out)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()