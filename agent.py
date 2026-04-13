import os
from typing import List, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from obelix import OBELIX
from datetime import datetime
ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

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
        if action_idx == 0: self.est_angle -= np.radians(45)
        elif action_idx == 1: self.est_angle -= np.radians(22)
        elif action_idx == 3: self.est_angle += np.radians(22)
        elif action_idx == 4: self.est_angle += np.radians(45)
        
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
        rot_fatigue = state[:, 23] 
        tl_x = state[:, 28]                
        arc_f = state[:, 29]

        is_too_dizzy = (rot_fatigue > 0.20)
        # is_emergency = (state[:, 17] > 0.5) | (front_near > 0.85)
        is_emergency = (state[:, 17] > 0.5)
        
        is_wiggling = (arc_f > 0.2) & (arc_f < 0.8) & (~is_emergency) & (~is_too_dizzy)

        h_bias = torch.zeros_like(logits)

        if is_too_dizzy.any():
            h_bias = torch.where(is_too_dizzy.unsqueeze(1), 
                                torch.tensor([-30., -30., 20.0, -30., -30.], device=device), h_bias)
            return self._finalize_probs(logits, h_bias, decay_factor, features)

        if is_emergency.any():
            h_bias[:, 2] = torch.where(is_emergency, torch.tensor(-10.0, device=device), h_bias[:, 2])
            turn_idx = torch.where(tl_x <= 0, torch.tensor(0, device=device), torch.tensor(4, device=device))
            h_bias.scatter_(1, turn_idx.unsqueeze(1), 12.0)
            return self._finalize_probs(logits, h_bias, decay_factor, features)

        h_bias[:, 2] = 6.0 

        if is_wiggling.any():
            wiggle_intensity = 10.0
            
            wiggle_clock = (step_counter // 2) % 2 == 0
            
            h_bias[:, 0] = torch.where(is_wiggling & wiggle_clock, 
                                    torch.tensor(wiggle_intensity, device=device), h_bias[:, 0])
            h_bias[:, 4] = torch.where(is_wiggling & (~wiggle_clock), 
                                    torch.tensor(wiggle_intensity, device=device), h_bias[:, 4])
        
        if ir_active.any():
            h_bias[:, 2] += 2.0 
            h_bias[:, 0] += torch.where(state[:, 30] > state[:, 31], torch.tensor(4.0), torch.tensor(0.))
            h_bias[:, 4] += torch.where(state[:, 31] > state[:, 30], torch.tensor(4.0), torch.tensor(0.))
        else:
            h_bias[:, 1] += torch.where(tl_x < -0.2, torch.tensor(2.0), torch.tensor(0.))
            h_bias[:, 3] += torch.where(tl_x > 0.2, torch.tensor(2.0), torch.tensor(0.))

        return self._finalize_probs(logits, h_bias, decay_factor, features)

    def _finalize_probs(self, logits, h_bias, decay_factor, features):
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


def record_video(model, env_config, device, filename="eval_video", episodes=1):
    """
    Records a video of the agent's performance.
    """
    env_config = {
        "scaling_factor": 5,
        "arena_size": 500,
        "wall_obstacles": True,
        "difficulty": 2,
        "box_speed": 10,
        "seed": 0
    }
    eval_env = OBELIX(**env_config) #
    eval_mem = RobotMemory()
    video_path = f"{filename}.mp4"
    os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)
    _ = eval_env.reset()
    h, w, _ = eval_env.frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))

    if not out.isOpened():
        print("VideoWriter failed to open. Trying fallback AVI/XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = f"{filename}.avi"
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
    model.eval()
    rewards = []
    for ep in range(episodes):
        total_reward = 0.0
        obs = eval_env.reset(seed=ep)
        eval_mem.step_counter = 0
        print(f"Recording Episode {ep+1}/{episodes} for video...")
        for t in range(2000):
            
            out.write(cv2.cvtColor(eval_env.frame, cv2.COLOR_RGB2BGR))

            eval_mem.update_odometry(eval_mem.last_action_idx, obs)
            
            with torch.no_grad():
                aug_obs = eval_mem.get_inference_obs(obs)
                state_t = torch.as_tensor(aug_obs, dtype=torch.float32).to(device).unsqueeze(0)
                probs, _ = model.get_action_distribution(state_t, eval_mem.step_counter) # Removed t from here if handled inside
                action = torch.argmax(probs).item()
            
            action_str = ACTIONS[action]
            next_obs, reward, done = eval_env.step(action_str, render=True)
            
            eval_mem.update_robot_memory(next_obs, action)
            eval_mem.last_action_idx = int(action)
            eval_mem.step_counter += 1
            
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

# Global instances
_model: Optional[ActorCritic] = None
_mem = RobotMemory()

def _load_once():
    global _model
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    _model = ActorCritic(in_dim=40, out_dim=5)
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    _model.load_state_dict(sd, strict=True)
    _model.eval()

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _mem
    _load_once()
    
    if _mem.prev_raw_obs is None:
        _mem.prev_raw_obs = obs.copy()

    _mem.update_odometry(_mem.last_action_idx, obs)
    
    _mem.update_robot_memory(obs, _mem.last_action_idx)

    aug_obs = _mem.get_inference_obs(obs)
    state_t = torch.as_tensor(aug_obs, dtype=torch.float32).unsqueeze(0).to(torch.device("cpu"))
    
    probs, _ = _model.get_action_distribution(state_t, _mem.step_counter, decay_factor=0.5)
    
    probs = probs.squeeze(0).cpu().numpy()
    action_idx = int(np.argmax(probs))

    _mem.last_action_idx = action_idx 
    _mem.step_counter += 1
    _mem.prev_raw_obs = obs.copy()
    
    return ACTIONS[action_idx]