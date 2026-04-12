from collections import deque
import time
import os
import cv2
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from obelix import OBELIX

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_EPOCHS = 4
ENTROPY_COEF = 0.08 
LEARNING_RATE = 2e-4

class RobotMemory:
    def __init__(self):
        self.step_counter = 0
        self.consecutive_fw = 0
        self.consecutive_turns = 0
        self.ir_latch_count = 0
        self.arc_counters = [0, 0, 0]  # [Front, Left, Right]
        self.last_action_idx = 2       # Forward
        self.prev_raw_obs = np.zeros(18)  
        self.est_x, self.est_y, self.est_angle = 0.0, 0.0, 0.0
        self.current_dist = 0.0
        self.dist_delta = 0.0
        self.wiggle_timer = 0
        self.prev_signal = 0.0

    def reset_if_new_episode(self, current_raw_obs):
        if self.prev_raw_obs is None:
            self.prev_raw_obs = current_raw_obs.copy()
        
    def update_odometry(self, action_idx, next_raw_s):
        # Update Angle
        if action_idx == 0: self.est_angle -= np.radians(45)
        elif action_idx == 1: self.est_angle -= np.radians(22)
        elif action_idx == 3: self.est_angle += np.radians(22)
        elif action_idx == 4: self.est_angle += np.radians(45)
        
        # (DR)
        is_stuck = next_raw_s[17] > 0.5
        if action_idx == 2 and not is_stuck:
            self.est_x += np.cos(self.est_angle) * 1.5
            self.est_y += np.sin(self.est_angle) * 1.5
        
        prev_dist = self.current_dist
        self.current_dist = np.sqrt(self.est_x**2 + self.est_y**2)
        self.dist_delta = self.current_dist - prev_dist
        return self.dist_delta
    
    def update_robot_memory(self, next_raw_s, action):
        if next_raw_s[16] > 0.1: self.ir_latch_count = min(20, self.ir_latch_count + 2)
        else: self.ir_latch_count = max(0, self.ir_latch_count - 1)
        
        t_long = self.step_counter * 0.01 
        if self.wiggle_timer > 0:
            self.wiggle_timer -= 1
        self.tl_x = np.sin(2 * t_long)
        self.tl_y = np.cos(1 * t_long)
        
        t_short = (self.step_counter % 2000) * 0.005
        self.ts_sin = np.sin(t_short)
        self.ts_cos = np.cos(t_short)

        is_near_front = np.sum(next_raw_s[4:12][::2]) > 0.8
        self.arc_counters[0] = self.arc_counters[0] + 1 if (is_near_front and action != 2) else 0
        self.arc_counters[1] = self.arc_counters[1] + 1 if (np.sum(next_raw_s[12:16][::2]) > 0.8 and action != 2) else 0
        self.arc_counters[2] = self.arc_counters[2] + 1 if (np.sum(next_raw_s[0:4][::2]) > 0.8 and action != 2) else 0

        if action == 2: self.consecutive_fw, self.consecutive_turns = self.consecutive_fw + 1, 0
        else: self.consecutive_turns, self.consecutive_fw = self.consecutive_turns + 1, 0
            
        return self.ir_latch_count, self.arc_counters, self.consecutive_fw, self.consecutive_turns
    
    def get_inference_obs(self, current_raw_s):
        front_arc_p = min(1.0, self.arc_counters[0] / 15.0)
        left_arc_p  = min(1.0, self.arc_counters[1] / 15.0)
        right_arc_p = min(1.0, self.arc_counters[2] / 15.0)
        
        if self.last_action_idx in [3, 4]:
            is_turning = 1.0
        elif self.last_action_idx in [0, 1]:
            is_turning = -1.0
        else:
            is_turning = 0.0

        is_gliding = 1.0 if (left_arc_p > 0.5 or right_arc_p > 0.5) else -1.0

        right_near = np.sum(current_raw_s[0:4][1::2]) / 2.0
        left_near  = np.sum(current_raw_s[12:16][1::2]) / 2.0
        front_near = np.sum(current_raw_s[4:12][1::2]) / 4.0
        front_far  = np.sum(current_raw_s[4:12][::2]) / 4.0
        
        path_is_clear = 1.0 if (np.sum(current_raw_s[4:12]) < 0.1) else -1.0

        latch_confidence = min(1.0, self.ir_latch_count / 10.0)
        if current_raw_s[16] > 0.5 or latch_confidence > 0.6 or front_near > 0.5:
            beacon_signal = (0.5 + 0.5 * (latch_confidence + front_near))
        else:
            beacon_signal = -1.0

        h_period = 600.0 
        t_phase = (self.step_counter % h_period) / h_period
        t_l_sin = np.cos(1 * 2 * np.pi * t_phase) 
        t_l_cos = np.sin(2 * 2 * np.pi * t_phase)
        
        t_s_period = 40.0
        t_s_phase = (self.step_counter % t_s_period) / t_s_period
        t_s_sin = np.sin(2 * np.pi * t_s_phase)
        t_s_cos = np.cos(2 * np.pi * t_s_phase)
        t_velocity = np.clip(-np.sin(2 * np.pi * t_phase), -1.0, 1.0)

        rot_fatigue = min(1.0, self.consecutive_turns / 12.0)
        fw_momentum = min(1.0, self.consecutive_fw / 20.0)
        signal_delta = np.clip(np.sum(current_raw_s[4:12]) - np.sum(self.prev_raw_obs[4:12]), -1.0, 1.0)

        lat_pressure = (left_near + left_arc_p*0.5) - (right_near + right_arc_p*0.5)
        lat_pressure = np.clip(lat_pressure, -2.0, 2.0) 
        
        self.dist_delta = self.update_odometry(self.last_action_idx, current_raw_s)
        movement_velocity = np.clip(self.dist_delta * 5.0 , -1.0, 1.0)

        return np.concatenate([
            current_raw_s,
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
            [np.sum(current_raw_s[12:16][::2]) / 2.0],
            [np.sum(current_raw_s[0:4][::2]) / 2.0],  
            [movement_velocity], 
            [t_velocity]         
        ]).astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, in_dim=40, out_dim=5):
        super(ActorCritic, self).__init__()
        
        self.base = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.Softmax(dim=-1) 
        )
        
        self.critic = nn.Linear(256, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        nn.init.orthogonal_(self.actor[0].weight, gain=0.01)
        nn.init.constant_(self.actor[0].bias, 0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

        with torch.no_grad():
            self.actor[0].bias[2] += 0.01

    def get_action_distribution(self, state, step_counter, decay_factor=1.0):
        features = self.base(state)
        logits = self.actor(features)
        device = logits.device

        front_near = state[:, 19]     
        ir_active = state[:, 22] > 0.1     
        is_stuck_flag = state[:, 17] > 0.5 
        rot_fatigue = state[:, 23] 
        tl_x = state[:, 28]                
        turn_momentum = state[:, 33]       
        arc_f, arc_l, arc_r = state[:, 29], state[:, 30], state[:, 31]

        is_panic = is_stuck_flag
        is_too_dizzy = (rot_fatigue > 0.25) 
        is_hard_jammed = is_panic & (rot_fatigue > 0.80)
        
        is_ir_lock = ir_active & (~is_panic) & (~is_too_dizzy)
        is_wiggling = (arc_f > 0.3) & (arc_f < 0.7) & (front_near < 0.45) & (~is_ir_lock)
        is_avoiding_wall = (front_near > 0.8) & (~is_hard_jammed) & (~is_ir_lock)
        is_cruising = ~(is_panic | is_avoiding_wall | is_ir_lock | is_too_dizzy | is_wiggling)

        h_bias = torch.zeros_like(logits)

        if is_ir_lock.any():
            h_bias[:, 2] = torch.where(is_ir_lock, torch.tensor(6.0, device=device), h_bias[:, 2])
            h_bias[:, 0] = torch.where(is_ir_lock & (arc_l > arc_r + 0.05), torch.tensor(5.0, device=device), h_bias[:, 0])
            h_bias[:, 4] = torch.where(is_ir_lock & (arc_r > arc_l + 0.05), torch.tensor(5.0, device=device), h_bias[:, 4])

        active_avoidance = (is_panic | is_avoiding_wall) & (~is_hard_jammed)
        if active_avoidance.any():
            h_bias[:, 2] = torch.where(is_panic, torch.tensor(-6.0, device=device), torch.tensor(2.0, device=device))
            
            t_step = torch.as_tensor(step_counter, device=device, dtype=torch.float32)
            phase_drift = torch.sin(t_step * 0.1) * 0.4
            random_nudge = torch.randn(active_avoidance.size(0), device=device) * 0.2
            commit_right = (turn_momentum > 0.1) | ((turn_momentum.abs() < 0.1) & (tl_x + phase_drift + random_nudge > 0.0))
            
            turn_val = 7.0
            h_bias[:, 0] = torch.where(active_avoidance & (~commit_right), turn_val, h_bias[:, 0])
            h_bias[:, 4] = torch.where(active_avoidance & commit_right, turn_val, h_bias[:, 4])

        if is_wiggling.any():
            h_bias[:, 2] = torch.where(is_wiggling, torch.tensor(3.0, device=device), h_bias[:, 2])
            wiggle_mask = torch.as_tensor((step_counter % 2 == 0), device=device, dtype=torch.bool).expand(is_wiggling.size(0))
            h_bias[:, 1] = torch.where(is_wiggling & wiggle_mask, torch.tensor(4.0, device=device), h_bias[:, 1])
            h_bias[:, 3] = torch.where(is_wiggling & (~wiggle_mask), torch.tensor(4.0, device=device), h_bias[:, 3])

        if is_too_dizzy.any():
            h_bias[:, 2] = torch.where(is_too_dizzy, torch.tensor(10.0, device=device), h_bias[:, 2])
            h_bias[:, [0, 1, 3, 4]] -= 10.0 # Suppress all turns

        if is_cruising.any():
            h_bias[:, 2] = torch.where(is_cruising, torch.tensor(3.0, device=device), h_bias[:, 2]) 
            h_bias[:, 3] = torch.where(is_cruising & (tl_x > 0.15), torch.tensor(2.0, device=device), h_bias[:, 3]) 
            h_bias[:, 1] = torch.where(is_cruising & (tl_x < -0.15), torch.tensor(2.0, device=device), h_bias[:, 1])

        probs = F.softmax(logits + (h_bias * decay_factor), dim=-1)
        probs = (probs + 1e-9) / (probs + 1e-9).sum(dim=-1, keepdim=True)
        return probs, self.critic(features)

    def act(self, state):
        probs, state_value = self.get_action_distribution(state, step_counter=0)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach(), state_value.detach()

    def evaluate(self, state, action):
        probs, state_value = self.get_action_distribution(state, step_counter=0)
        dist = Categorical(probs)
        return dist.log_prob(action), state_value, dist.entropy()

def update_ppo(model, optimizer, buffer, device):
    s = torch.stack(buffer['states']).to(device)
    a = torch.stack(buffer['actions']).to(device)
    lp = torch.stack(buffer['log_probs']).detach().to(device)
    ret = torch.stack(buffer['returns']).detach().to(device)
    adv = torch.stack(buffer['advantages']).detach().to(device)
    steps = torch.tensor(buffer['step_counts'], device=device)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        probs, values = model.get_action_distribution(s, steps)
        dist = Categorical(probs)
        
        v_pred = values.view(-1)
        returns_target = ret.view(-1)
        
        new_lp = dist.log_prob(a)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_lp - lp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv
        
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(v_pred, returns_target)
        loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return actor_loss.item(), critic_loss.item(), entropy.item()

def record_video(model, env_config, device, filename="eval_video", episodes=1):
    """
    Records a video of the agent's performance.
    """
    eval_env = OBELIX(**env_config) # Initialize
    eval_mem = RobotMemory()
    video_path = f"{filename}.mp4"
    
    _ = eval_env.reset()
    h, w, _ = eval_env.frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
    
    model.eval()
    rewards = []
    for ep in range(episodes):
        total_reward = 0.0
        obs = eval_env.reset(seed=ep)
        eval_mem.step_counter = 0
        print(f"Recording Episode {ep+1}/{episodes} for video...")
        for t in range(2000):
            
            out.write(cv2.cvtColor(eval_env.frame, cv2.COLOR_RGB2BGR))

            with torch.no_grad():
                aug_obs = eval_mem.get_inference_obs(obs)
                state_t = torch.as_tensor(aug_obs, dtype=torch.float32).to(device).unsqueeze(0)
                probs, _ = model.get_action_distribution(state_t, eval_mem.step_counter)
                action = torch.argmax(probs).item()
            action_str = ACTIONS[action]
            next_obs, reward, done= eval_env.step(action_str, render=False)
            eval_mem.update_robot_memory(next_obs, action)
            obs = next_obs
            total_reward += reward
            
            if done:
                break
        rewards.append(total_reward)
                
    if out:
        out.release()
    model.train()
    print(f"Average Reward over {episodes} episodes: {np.mean(rewards):.2f}, {rewards}")
    print(f"Video saved to {video_path}")

def train():
    gamma = 0.99
    lam = 0.95
    learning_rate = 1e-4
    episodes = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env_config = {
        "scaling_factor": 5,
        "arena_size": 500,
        "wall_obstacles": True,
        "difficulty": 2,
        "box_speed": 10,
        "seed": 0
    }
    env = OBELIX(**env_config)
    memory = RobotMemory()
    model = ActorCritic(in_dim=40, out_dim=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=f"runs/gae_ra2_{int(time.time())}")

    past_ep_rewards = deque(maxlen=10)
    for ep in range(episodes):
        obs = env.reset(env_config['seed'] + ep)
        memory.prev_raw_obs = obs.copy()
        
        values, log_probs, rewards, masks = [], [], [], []
        entropies = 0
        # Buffer
        buffer = {'states': [], 'actions': [], 'log_probs': [], 
                  'rewards': [], 'masks': [], 'values': [], 
                  'step_counts': [], 'advantages': [], 'returns': []}
        
        ep_reward = 0

        for t in range(2000): # Max steps
            aug_obs = memory.get_inference_obs(obs)
            state_t = torch.as_tensor(aug_obs, dtype=torch.float32).to(device).unsqueeze(0)
            
            probs, val = model.get_action_distribution(state_t, memory.step_counter)
            dist = Categorical(probs)
            action_idx = dist.sample()
            
            action_str = ACTIONS[action_idx.item()]
            next_obs, reward, done = env.step(action_str, render=True)
            
            current_signal = next_obs[0] # Assuming first raw sensor or use aug_obs[22]
            if current_signal > 0 and memory.prev_signal <= 0:
                reward += 2.0
            if current_signal > memory.prev_signal and current_signal > 0:
                reward += 0.5
            memory.prev_signal = current_signal

            memory.update_robot_memory(next_obs, int(action_idx))

            buffer['states'].append(state_t.squeeze(0))
            buffer['actions'].append(action_idx)
            buffer['log_probs'].append(dist.log_prob(action_idx))
            buffer['values'].append(val.flatten())
            buffer['rewards'].append(torch.tensor(reward, device=device))
            buffer['masks'].append(torch.tensor(1 - done, device=device))
            buffer['step_counts'].append(memory.step_counter)
            
            obs = next_obs
            ep_reward += reward
            past_ep_rewards.append(ep_reward)
            memory.step_counter += 1
            if done: break

        with torch.no_grad():
            final_aug = memory.get_inference_obs(obs)
            _, next_val = model.get_action_distribution(
                torch.as_tensor(final_aug, dtype=torch.float32).to(device).unsqueeze(0),
                memory.step_counter
            )
        
        advs, rets = [], []
        gae = 0
        last_val = next_val.flatten()
        for i in reversed(range(len(buffer['rewards']))):
            delta = buffer['rewards'][i] + GAMMA * last_val * buffer['masks'][i] - buffer['values'][i]
            gae = delta + GAMMA * GAE_LAMBDA * buffer['masks'][i] * gae
            last_val = buffer['values'][i]
            advs.insert(0, gae)
            rets.insert(0, gae + buffer['values'][i])

        buffer['advantages'], buffer['returns'] = advs, rets
        
        a_loss, c_loss, e_loss = update_ppo(model, optimizer, buffer, device)

        writer.add_scalar('Reward/Episode', ep_reward, ep)
        writer.add_scalar('Loss/Actor', a_loss, ep)
        writer.add_scalar('Loss/Critic', c_loss, ep)
        writer.add_scalar('Loss/Entropy', e_loss, ep)
        if ep % 10 == 0:
            print(f"Ep {ep} | Reward: {np.mean(past_ep_rewards):.2f} | Ent: {e_loss:.6f} | A_Loss: {a_loss:.4f}")

        if (ep+1) % 500 == 0:
            timestamp = int(time.time())
            vid_name = f"ppo_eval_ep{ep+1}_{timestamp}"
            record_video(model, env_config, device, filename=vid_name, episodes=10)
            torch.save(model.state_dict(), f"gae_ra2_ep{ep+1}.pth")
    # torch.save(model.state_dict(), "gae_ra2_final.pth")

if __name__ == "__main__":
    train()