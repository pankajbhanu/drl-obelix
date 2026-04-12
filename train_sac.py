import argparse
import sys
from datetime import datetime
import time
import os
import random
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    Used for O(log N) prioritized sampling.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write_idx = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, n_step=3, gamma=0.99, alpha=0.6, beta_start=0.4, beta_frames=100000):        
        self.tree = SumTree(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta_start
        self.epsilon = 1e-6
        self.max_priority = 1.0
        self.beta_frames = beta_frames
        self.frame_counter = 0
        
        self.n_step_buffer = deque(maxlen=n_step)

    # def current_beta(self):
    #     # Linearly anneal beta from beta_start to 1.0
    #     return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def _get_n_step_info(self):
        reward = 0
        for i, transition in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * transition[2]
            if transition[4]:
                return reward, transition[3], True
                
        return reward, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return

        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]
        
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = []
        indices = []
        weights = []
        
        segment = self.tree.total_priority() / batch_size
        # beta = self.current_beta()
        beta = min(1.0, self.beta + self.frame_counter * (1.0 - self.beta) / self.beta_frames)        
        self.frame_counter += 1

        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority()
        if p_min == 0: p_min = self.epsilon
        max_weight = (p_min * self.n_entries) ** (-beta)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            sampling_prob = priority / self.tree.total_priority()
            weight = (sampling_prob * self.n_entries) ** (-beta)
            
            indices.append(idx)
            weights.append(weight / max_weight) 
            batch.append(data)

        state, action, reward, next_state, done = zip(*batch)
        
        return (torch.FloatTensor(np.array(state)), 
                torch.LongTensor(action), 
                torch.FloatTensor(reward), 
                torch.FloatTensor(np.array(next_state)), 
                torch.FloatTensor(done),
                indices,
                torch.FloatTensor(weights).unsqueeze(1))

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = min(float((abs(error) + self.epsilon) ** self.alpha), 1000.0)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    @property
    def n_entries(self):
        return self.tree.n_entries

    def __len__(self):
        return self.n_entries

class Actor(nn.Module):
    def __init__(self, in_dim=38, out_dim=5):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, state):
        logits = self.net(state)
        device = logits.device
        
        ir_active     = state[:, 16] > 0.5
        is_stuck_flag = state[:, 17] > 0.5
        rot_fatigue   = state[:, 23]      
        
        tl_x, tl_y = state[:, 27], state[:, 28] 
        
        in_recovery = is_stuck_flag | ir_active
        
        h_bias = torch.zeros_like(logits)

        h_bias[:, 2] += torch.where(in_recovery, torch.tensor(0.0, device=device), torch.tensor(4.5, device=device))
        
        h_bias[:, 3] = torch.where(tl_x > 0.15, torch.tensor(5.0, device=device), h_bias[:, 3])
        h_bias[:, 1] = torch.where(tl_x < -0.15, torch.tensor(5.0, device=device), h_bias[:, 1])
        
        in_chamber = torch.abs(tl_x) > 0.5
        h_bias[:, 1] = torch.where(in_chamber & (tl_y > 0.3), torch.tensor(4.0, device=device), h_bias[:, 1])
        h_bias[:, 3] = torch.where(in_chamber & (tl_y < -0.3), torch.tensor(4.0, device=device), h_bias[:, 3])

        h_bias[:, 4] = torch.where(in_recovery & (tl_x > 0), torch.tensor(16.0, device=device), h_bias[:, 4])
        h_bias[:, 0] = torch.where(in_recovery & (tl_x <= 0), torch.tensor(16.0, device=device), h_bias[:, 0])

        commitment_mask = torch.zeros_like(logits)
        
        commitment_mask[:, 2] = torch.where(in_recovery, torch.tensor(-1e9, device=device), 0.0)
        
        is_too_dizzy = rot_fatigue > 0.8
        commitment_mask[:, [0, 1, 3, 4]] = torch.where(
            is_too_dizzy.unsqueeze(1), 
            torch.tensor(-1e9, device=device), 
            commitment_mask[:, [0, 1, 3, 4]]
        )

        return F.softmax(logits + commitment_mask + h_bias, dim=-1)

class SoftQNetwork(nn.Module):
    def __init__(self, in_dim=38, out_dim=5):
        super(SoftQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, state):
        return self.net(state)

class SACAgent:
    def __init__(self, in_dim, out_dim, device, actor_lr=8e-5, 
                 critic_lr=3e-4, 
                 alpha_lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, writer=None):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha 
        self.target_entropy = -np.log(1.0 / out_dim) * 0.001 # Heuristic target
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().item()
        # Actor
        self.actor = Actor(in_dim, out_dim).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.policy_update_freq = 2 
        self.update_step_counter = 0
        # Twin QNetworks
        self.q1 = SoftQNetwork(in_dim, out_dim).to(device)
        self.q2 = SoftQNetwork(in_dim, out_dim).to(device)
        self.q1_target = SoftQNetwork(in_dim, out_dim).to(device)
        self.q2_target = SoftQNetwork(in_dim, out_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)
        self.writer = writer

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, replay_buffer, batch_size, global_step):
        s, a, r, s_next, done, indices, weights = replay_buffer.sample(batch_size)
        s, a, r, s_next, done, weights = s.to(self.device), a.to(self.device), r.to(self.device), s_next.to(self.device), done.to(self.device), weights.to(self.device)
        self.update_step_counter += 1
        with torch.no_grad():
            next_probs = self.actor(s_next)
            next_log_probs = torch.log(next_probs + 1e-10)
            # q1_next = self.q1_target(s_next)
            # q2_next = self.q2_target(s_next)
            # min_q_next = torch.min(q1_next, q2_next)
            # # Soft State Value Calculation: V = min(Q) - alpha * log_pi
            # v_next = torch.sum(next_probs * (min_q_next - self.alpha * next_log_probs), dim=1)
            # q_target = r + (1 - done) * self.gamma * v_next

            q_next = torch.min(self.q1_target(s_next), self.q2_target(s_next))
            v_next = torch.sum(next_probs * (q_next - self.alpha * next_log_probs), dim=1)
            # n-step target
            gamma_n = self.gamma ** replay_buffer.n_step
            q_target = r + (1 - done) * gamma_n * v_next

        curr_q1 = self.q1(s).gather(1, a.unsqueeze(1)).squeeze(-1)
        curr_q2 = self.q2(s).gather(1, a.unsqueeze(1)).squeeze(-1)
        
        td_errors = (torch.abs(q_target - curr_q1) + torch.abs(q_target - curr_q2)) / 2.0
        
        replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        q1_loss = (weights * F.mse_loss(curr_q1, q_target, reduction='none')).mean()
        q2_loss = (weights * F.mse_loss(curr_q2, q_target, reduction='none')).mean()
        q_loss = q1_loss + q2_loss

        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=5.0)
        self.q_opt.step()

        if self.update_step_counter % self.policy_update_freq == 0:
            probs = self.actor(s)
            log_probs = torch.log(probs + 1e-10)
            with torch.no_grad():
                q1_val = self.q1(s)
                q2_val = self.q2(s)
                min_q_val = torch.min(q1_val, q2_val)
            
            actor_loss = torch.sum(probs * (self.alpha * log_probs - min_q_val), dim=1).mean()
            if self.writer:
                self.writer.add_scalar('Loss/Actor', actor_loss.item(), global_step)
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
            self.actor_opt.step()

            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self.alpha = self.log_alpha.exp().detach().item()

            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def get_inference_obs(raw_s, prev_raw_s, step_counter, consecutive_fw, consecutive_turns, 
                       ir_latch_count, last_action, wide_arc_counters, dist_delta):
    """
    State Dimension: 38
    wide_arc_counters: List [front_c, left_c, right_c]
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
    if raw_s[16] > 0.5 or latch_confidence > 0.6 or front_near > 0.5:
        beacon_signal = (0.5 + 0.5 * (latch_confidence + front_near))
    else:
        beacon_signal = -1.0

    h_period = 600.0 
    t_phase = (step_counter % h_period) / h_period
    
    t_l_sin = np.cos(1 * 2 * np.pi * t_phase) 
    
    t_l_cos = np.sin(2 * 2 * np.pi * t_phase)
    t_s_period = 40.0
    t_s_phase = (step_counter % t_s_period) / t_s_period
    
    t_s_sin = np.sin(2 * np.pi * t_s_phase)
    t_s_cos = np.cos(2 * np.pi * t_s_phase)
    t_velocity = np.clip(-np.sin(2 * np.pi * t_phase), -1.0, 1.0)

    rot_fatigue = min(1.0, consecutive_turns / 12.0)
    fw_momentum = min(1.0, consecutive_fw / 20.0)
    signal_delta = np.clip(np.sum(raw_s[4:12]) - np.sum(prev_raw_s[4:12]), -1.0, 1.0)

    lat_pressure = (left_near + left_arc_p*0.5) - (right_near + right_arc_p*0.5)
    lat_pressure = np.clip(lat_pressure, -2.0, 2.0) 
    movement_velocity = np.clip(dist_delta * 5.0 , -1.0, 1.0)

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
        [np.sum(raw_s[0:4][::2]) / 2.0],   
        [movement_velocity],
        [t_velocity]
    ]).astype(np.float32)

def calculate_ppo_reward(env_reward, next_augmented_state, action, ir_latch_count, 
                         consecutive_fw, consecutive_turns, current_dist, prev_dist, 
                         prev_front_near, old_t_l_sin, old_t_l_cos):
    ir_val = next_augmented_state[16]
    front_near = next_augmented_state[19] 
    tl_sin = next_augmented_state[27] 
    tl_cos = next_augmented_state[28] 
    movement_vel = next_augmented_state[38]
    is_stuck = next_augmented_state[17] > 0.5
    ir_improvement = (ir_val > 0.1)
    
    dist_delta = current_dist - prev_dist
    shaped_reward = 0.0

    v_dist = np.sqrt((tl_sin - old_t_l_sin)**2 + (tl_cos - old_t_l_cos)**2)
    
    if action == 2:
        shaped_reward += 2.0 + (v_dist * 50.0)
        
        if abs(tl_cos) < 0.2 and front_near < 0.5:
            shaped_reward += 5.0

    if ir_val > 0.3:
        if action == 2:
            shaped_reward += 10.0 + (ir_val * 5.0)
        elif action in [1, 3]:
            shaped_reward += 2.0

    if ir_val > 0.1:
        if action == 2:
            return 12.0 + (ir_val * 3.0)
        else:
            return -2.0

    if action == 2:
        if is_stuck or (movement_vel < 0.1 and front_near > 0.7):
            return -15.0

        is_in_vertical_leg = abs(tl_sin) > 0.7
        is_on_bridge = abs(tl_cos) < 0.2 

        if is_in_vertical_leg or is_on_bridge:
            shaped_reward += 6.0 + np.clip(dist_delta * 5.0, 0, 5.0)
        else:
            shaped_reward += 1.0

    elif action in [0, 1, 3, 4]:
        if abs(tl_cos) > 0.8:
            shaped_reward += 4.0
        else:
            shaped_reward -= 5.0

    if env_reward >= 100: return 15.0
    if env_reward <= -100: return -15.0
    
    existence = 1.0 if movement_vel > 0.2 else -1.0
    combined = (env_reward / 20.0) + shaped_reward + existence
    return float(np.clip(combined, -15.0, 15.0))

def update_robot_memory(next_raw_s, action, ir_latch, arc_counts, fw_count, turn_count):

    if next_raw_s[16] > 0.1:
        ir_latch = min(20, ir_latch + 2)
    else:
        ir_latch = max(0, ir_latch - 1)
        
    is_near_front = np.sum(next_raw_s[4:12][::2]) > 0.8
    arc_counts[0] = arc_counts[0] + 1 if (is_near_front and action != 2 and not next_raw_s[16] > 0.1) else 0
    arc_counts[1] = arc_counts[1] + 1 if (np.sum(next_raw_s[12:16][::2]) > 0.8 and action != 2) else 0
    arc_counts[2] = arc_counts[2] + 1 if (np.sum(next_raw_s[0:4][::2]) > 0.8 and action != 2) else 0

    if action == 2:
        fw_count += 1
        turn_count = 0
    else:
        turn_count += 1
        fw_count = 0
        
    return ir_latch, arc_counts, fw_count, turn_count

def evaluate_agent(env_params, agent, device, episodes=10):
    print(f"\n--- Starting Evaluation: {episodes} Episodes ---")
    temp_env = OBELIX(**env_params)
    
    _ = temp_env.reset()
    total_eval_rewards = []
    agent.actor.eval()
    
    for ep in range(episodes):
        state_raw = temp_env.reset()
        prev_raw_s = np.copy(state_raw)
        
        step_counter = 0
        consecutive_fw = 0
        consecutive_turns = 0
        ir_latch_count = 0
        arc_counters = [0, 0, 0]
        last_action = 2         
        dist_delta = 0.0
        prev_dist = 0.0
        current_dist = 0.0
        est_x, est_y, est_angle = 0.0, 0.0, 0.0
        done = False
        ep_reward = 0
        
        while not done:
            state_vec = get_inference_obs(
                state_raw, prev_raw_s, step_counter, consecutive_fw, 
                consecutive_turns, ir_latch_count, last_action, arc_counters, dist_delta
            )
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # features = agent.policy.base(state_t)
                # logits = agent.policy.act(features)
                # probs = torch.softmax(logits, dim=-1)
                # action = torch.argmax(probs, dim=-1).item()
                prob_tensor = agent.actor(state_t)
                    
                action = prob_tensor.argmax(dim=-1).item()
                # log_prob = log_prob_tensor.item()
            next_raw_s, reward, done = temp_env.step(ACTIONS[action], render=True)
            ep_reward += reward
            is_physically_stuck = (action == 2 and next_raw_s[17] == 1) # Penalty usually implies a hit/grind

            if action == 2 and not is_physically_stuck:
                est_x += np.cos(est_angle) * 1.5
                est_y += np.sin(est_angle) * 1.5
            else:
                pass

            prev_dist = current_dist
            current_dist = np.sqrt(est_x**2 + est_y**2)
            dist_delta = current_dist - prev_dist
            ir_latch_count, arc_counters, consecutive_fw, consecutive_turns = update_robot_memory(
                next_raw_s, action, ir_latch_count, arc_counters, consecutive_fw, consecutive_turns
            )
                
            prev_raw_s = np.copy(state_raw)
            state_raw = next_raw_s
            last_action = action
            step_counter += 1
            
            if step_counter > 2000:
                break
        total_eval_rewards.append(ep_reward)

    avg_eval_reward = np.mean(total_eval_rewards)
    print(f"Evaluation Complete. Avg Reward: {avg_eval_reward:.2f} | {total_eval_rewards}")
    
    agent.actor.train() # Return to training mode
    return avg_eval_reward

def record_video(env_params, agent, device, filename="ppo", episodes=10):
    """Generates an MP4 video of the trained agent using PPO2 logic."""

    filename = f"{filename}.mp4"
    print(f"Recording to {filename}...")
    temp_env = OBELIX(**env_params)
    
    _ = temp_env.reset()
    h, w, _ = temp_env.frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
    
    agent.actor.eval()
    total_eval_rewards = []
    for ep in range(episodes):
        print(f"Recording Episode {ep+1}/{episodes}")
        state_raw = temp_env.reset()
        prev_raw_s = np.copy(state_raw)
        
        step_counter = 0
        consecutive_fw = 0
        consecutive_turns = 0
        ir_latch_count = 0
        arc_counters = [0, 0, 0]  # [Front, Left, Right]
        last_action = 2           # Forward
        dist_delta = 0.0
        prev_dist = 0.0
        current_dist = 0.0
        est_x, est_y, est_angle = 0.0, 0.0, 0.0
        done = False
        ep_reward = 0
        
        while not done:
            state_vec = get_inference_obs(
                state_raw, prev_raw_s, step_counter, consecutive_fw, 
                consecutive_turns, ir_latch_count, last_action, arc_counters, dist_delta
            )
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # features = agent.policy.base(state_t)
                # logits = agent.policy.act(features)
                # probs = torch.softmax(logits, dim=-1)
                # action = torch.argmax(probs, dim=-1).item()
                prob_tensor = agent.actor(state_t)
                    
                action = prob_tensor.argmax(dim=-1).item()
            next_raw_s, reward, done = temp_env.step(ACTIONS[action], render=False)
            
            out.write(temp_env.frame)
            ep_reward += reward
            is_physically_stuck = (action == 2 and next_raw_s[17] == 1) 

            if action == 2 and not is_physically_stuck:
                est_x += np.cos(est_angle) * 1.5
                est_y += np.sin(est_angle) * 1.5
            else:
                # If we are stuck, we don't gain distance. 
                pass

            prev_dist = current_dist
            current_dist = np.sqrt(est_x**2 + est_y**2)
            dist_delta = current_dist - prev_dist
            ir_latch_count, arc_counters, consecutive_fw, consecutive_turns = update_robot_memory(
                next_raw_s, action, ir_latch_count, arc_counters, consecutive_fw, consecutive_turns
            )
                
            prev_raw_s = np.copy(state_raw)
            state_raw = next_raw_s
            last_action = action
            step_counter += 1
            
            if step_counter > 2000:
                break
        total_eval_rewards.append(ep_reward)
    avg_eval_reward = np.mean(total_eval_rewards)
    print(f"Recording Complete. Avg Reward: {avg_eval_reward:.2f} | {total_eval_rewards}")
    agent.actor.train()     
    out.release()
    print(f"Video saved successfully as {filename}")

class GlobalCoverageTracker:
    def __init__(self, arena_size=500, save_dir="heatmaps"):
        self.arena_size = arena_size
        self.save_dir = save_dir
        self.grid = np.zeros((arena_size, arena_size), dtype=np.float32)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def update(self, x, y):
        ix = int(np.clip(x, 0, self.arena_size - 1))
        iy = int(np.clip(y, 0, self.arena_size - 1))
        self.grid[iy, ix] += 1

    def save_heatmap(self, episode_num):
        if self.grid.max() == 0:
            return

        log_grid = np.log1p(self.grid)
        
        norm_grid = cv2.normalize(log_grid, None, 0, 255, cv2.NORM_MINMAX)
        norm_grid = norm_grid.astype(np.uint8)

        heatmap_img = cv2.applyColorMap(norm_grid, cv2.COLORMAP_JET)

        file_path = os.path.join(self.save_dir, f"sac_cov_ep_{episode_num}.png")
        cv2.imwrite(file_path, heatmap_img)
        print(f"--- Saved cumulative heatmap: {file_path} ---")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "record"], default="train")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default="ppo_pth")
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
        "difficulty": 3,
        "box_speed": 10,
    }
    env = OBELIX(**env_config)
    tracker = GlobalCoverageTracker(arena_size=env_config["arena_size"])
    state_dim = 40
    action_dim = 5
    buffer_size = 100000
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join("runs", f"SAC_{current_time}")
    writer = SummaryWriter(log_dir=log_dir)
    # agent = PPOAgent(state_dim, action_dim,lr_actor=1e-4, lr_critic=3e-4, device=device, writer=writer)
    agent = SACAgent(in_dim=state_dim, out_dim=action_dim, device=device, gamma=0.99, tau=0.005, alpha=0.2, writer=writer)
    replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
    start_steps = 6000
    # try:
    #     path = f"{args.out}.pth"
    #     agent.policy.load_state_dict(torch.load(path, map_location=device))
    #     print(f"Successfully loaded weights from {path}")
    # except FileNotFoundError:
    #     print("No saved weights found. Starting from scratch.")
    
    if args.mode == "record":
        args.out = 'ppoweights.ep300.pth'
        if os.path.exists(args.out):
            # q_net.load_state_dict(torch.load(args.out))
            agent.actor.load_state_dict(torch.load('ppoweights.ep300.pth', map_location=device))
            # agent.policy_old.load_state_dict(agent.policy.state_dict())
            record_video(env_config, agent, device)
        else:
            print("No weights found. Train first!")
        return
    # print_detailed_navigation_flow(agent, env, 8) # Initial navigation flow before training
    MAX_STEPS_PER_BATCH = 1024
    stats_history = {'actor': [], 'critic': [], 'entropy': [], 'reward': []}

    threshold_entropy = 0.3 
    raw_s = env.reset()
    prev_raw_s = np.copy(raw_s)
    consecutive_fw = 0
    consecutive_turns = 0
    ir_latch_count = 0
    arc_counters = [0, 0, 0] # [Front, Left, Right]
    last_action = 2          # 'Forward'
    step_counter = 0
    iter_counter = 0
    update_timestep = 2000

    wall_time = time.time()
    episode = 0
    current_total_steps = 0
    episode_rewards = deque(maxlen=10) 
    raw_rewards = deque(maxlen=10) 
    pos_history_x = deque(maxlen=10000)
    pos_history_y = deque(maxlen=10000)
    try:
        print("Obelix Training Started. Press Ctrl+C to save and exit.")
        for ep in range(args.episodes): 
            episode += 1
            episode_step_counter = 0
            state_raw = env.reset()
            ir_latch_count, consecutive_fw, consecutive_turns = 0, 0, 0
            arc_counters = [0, 0, 0]
            est_x, est_y, est_angle = 0.0, 0.0, 0.0
            current_dist = 0.0
            dist_delta = 0.0
            ep_reward = 0
            ep_ppo_reward = 0
            raw_reward = 0
            done = False
            
            for step in range(args.max_steps):
                state = get_inference_obs(state_raw, prev_raw_s, step_counter, consecutive_fw, 
                          consecutive_turns, ir_latch_count, last_action, arc_counters, dist_delta)
                
                if step_counter < start_steps:
                    action = random.randint(0, 4)
                else:
                    action, _ = agent.select_action(state)
                
                old_front_near = state[19]
                next_raw_s, raw_reward, done = env.step(ACTIONS[action], render=True)
                step_counter += 1
                action_str = ACTIONS[action]
                if action == 0: est_angle -= np.radians(45)
                elif action == 1: est_angle -= np.radians(22)
                elif action == 3: est_angle += np.radians(22)
                elif action == 4: est_angle += np.radians(45)
                
                front_obstacle = np.max(next_raw_s[4:12][1::2]) 
                
                is_physically_stuck = (action == 2 and next_raw_s[17] == 1) 

                if action == 2 and not is_physically_stuck:
                    est_x += np.cos(est_angle) * 1.5
                    est_y += np.sin(est_angle) * 1.5
                else:
                    pass

                prev_dist = current_dist
                current_dist = np.sqrt(est_x**2 + est_y**2)
                dist_delta = current_dist - prev_dist
                ir_latch_count, arc_counters, consecutive_fw, consecutive_turns = update_robot_memory(
                    next_raw_s, action, ir_latch_count, arc_counters, consecutive_fw, consecutive_turns
                )
                is_physically_stuck = (action == 2 and next_raw_s[17] == 1) 

                if is_physically_stuck:
                    consecutive_fw = 0 
                    dist_delta = -1.0  
                    current_dist = prev_dist 
                
                old_t_l_sin = state[27]
                old_t_l_cos = state[28]
                next_state_aug = get_inference_obs(next_raw_s, state_raw, step_counter, consecutive_fw, 
                                                consecutive_turns, ir_latch_count, action, arc_counters, dist_delta)
                
                final_reward = calculate_ppo_reward(
                    raw_reward, next_state_aug, action, ir_latch_count, consecutive_fw, consecutive_turns, current_dist, prev_dist
                , old_front_near, old_t_l_sin, old_t_l_cos)

                replay_buffer.push(state, action, final_reward, next_state_aug, done)
                prev_raw_s = state_raw.copy()
                state_raw = next_raw_s
                last_action = action
                ep_reward += raw_reward
                ep_ppo_reward += final_reward
                episode_step_counter += 1
                pos_history_x.append(next_state_aug[27]) 
                pos_history_y.append(next_state_aug[28]) 
                if step_counter >= start_steps and len(replay_buffer) >= args.batch:
                    agent.update(replay_buffer, args.batch, step_counter)

                tracker.update(env.bot_center_x, env.bot_center_y)
                if done:
                    break

            episode_rewards.append(ep_ppo_reward)
            raw_rewards.append(ep_reward)
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_raw_reward = np.mean(raw_rewards) if raw_rewards else 0
            # avg_actor_loss = np.mean(a_loss)
            # avg_critic_loss = np.mean(c_loss)
            # avg_entropy = np.mean(ent)

            writer.add_scalar('Reward/Average', avg_reward, step_counter)
            writer.add_scalar('Reward/Raw_Average', avg_raw_reward, step_counter)
            # writer.add_scalar('Loss/Actor', avg_actor_loss, step_counter)
            # writer.add_scalar('Loss/Critic', avg_critic_loss, step_counter)
            # writer.add_scalar('Policy/Entropy', avg_entropy, step_counter)

            # writer.add_scalar('Params/Learning_Rate', agent.optimizer.param_groups[0]['lr'], step_counter)

            # for name, param in agent.policy.named_parameters():
            #     writer.add_histogram(f"Weights/{name}", param, step_counter)

            for i, val in enumerate(state):
                if i % 5 == 0: # Log every 5th feature to keep the chart clean
                    agent.writer.add_scalar(f'Observation/Feature_{i}', val, step_counter)
            if episode % 100 == 0:
                filename = f"sac_{current_time}_ep{episode}"
                tracker.save_heatmap(episode)
                torch.save(agent.actor.state_dict(), f"{filename}.pth")
                record_video(env_config, agent, device, filename=filename, episodes=10)


            # if episode % 5 == 0:
            #     writer.add_figure('State/Vector_Visual', plt.figure(figsize=(10,2)))
            #     # plt.bar(range(len(state)), state)
            #     writer.add_figure('State/Vector_Visual', plt.gcf(), step_counter)
            #     log_2d_scatter(writer, pos_history_x, pos_history_y, step_counter)
            elapsed_min = (time.time() - wall_time) / 60
            writer.add_scalar("Time/Total_Elapsed_Minutes", elapsed_min, step_counter)
            writer.add_scalar("Reward/Episode", ep_reward, step_counter)
            if episode % 10 == 0:
                print(f"Ep {episode}/{args.episodes}, Total Steps: {step_counter}, Avg Raw Reward: {np.mean(raw_rewards):.2f},  Average Reward: {np.mean(episode_rewards):.2f}, Time Elapsed: {(time.time() - wall_time)/60:.2f} mins")

            if episode % 50 == 0:
                avg_eval_reward = evaluate_agent(env_config, agent, device, episodes=10)
                print(f"Evaluation after Episode {episode}: Average Reward = {avg_eval_reward:.2f}")
                writer.add_scalar('Reward/Evaluation', avg_eval_reward, step_counter)

    except KeyboardInterrupt:
        print("\n[!] Manual Interrupt. Saving...")
        # torch.save(agent.policy.state_dict(), f"ppo_interrupted_{episode}.pth")
        writer.close()
        sys.exit(0)

if __name__ == "__main__":
    main()