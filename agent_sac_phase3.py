import os
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class RobotMemory:
    def __init__(self):
        self.step_counter = 0
        self.consecutive_fw = 0
        self.consecutive_turns = 0
        self.ir_latch_count = 0
        self.arc_counters = [0, 0, 0]  # [Front, Left, Right]
        self.last_action_idx = 2       # Forward
        self.prev_raw_obs = None
        self.est_x, self.est_y, self.est_angle = 0.0, 0.0, 0.0
        self.current_dist = 0.0

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
        return self.current_dist - prev_dist

class Actor(nn.Module):
    def __init__(self, in_dim=40, out_dim=5):
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

# Global instances
_model: Optional[Actor] = None
_mem = RobotMemory()

def get_inference_obs(raw_s, prev_raw_s, step_counter, consecutive_fw, consecutive_turns, 
                       ir_latch_count, last_action, wide_arc_counters, dist_delta):
    """Augments raw 18-dim observation to 40-dim"""
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
    beacon_signal = (0.5 + 0.5 * (latch_confidence + front_near)) if (raw_s[16] > 0.5 or latch_confidence > 0.6) else -1.0

    h_period = 600.0 
    t_phase = (step_counter % h_period) / h_period
    t_l_sin = np.cos(1 * 2 * np.pi * t_phase) 
    t_l_cos = np.sin(2 * 2 * np.pi * t_phase)
    
    t_s_period = 40.0
    t_s_phase = (step_counter % t_s_period) / t_s_period
    t_s_sin, t_s_cos = np.sin(2 * np.pi * t_s_phase), np.cos(2 * np.pi * t_s_phase)
    t_velocity = np.clip(-np.sin(2 * np.pi * t_phase), -1.0, 1.0)

    rot_fatigue = min(1.0, consecutive_turns / 12.0)
    fw_momentum = min(1.0, consecutive_fw / 20.0)
    signal_delta = np.clip(np.sum(raw_s[4:12]) - np.sum(prev_raw_s[4:12]), -1.0, 1.0)
    lat_pressure = np.clip((left_near + left_arc_p*0.5) - (right_near + right_arc_p*0.5), -2.0, 2.0)
    movement_velocity = np.clip(dist_delta * 5.0 , -1.0, 1.0)

    return np.concatenate([
        raw_s, [lat_pressure], [front_near], [front_far], [path_is_clear], [beacon_signal],
        [rot_fatigue], [latch_confidence], [t_s_sin], [t_s_cos], [t_l_sin], [t_l_cos],
        [front_arc_p], [left_arc_p], [right_arc_p], [is_gliding], [is_turning],
        [fw_momentum], [signal_delta], [np.sum(raw_s[12:16][::2]) / 2.0],
        [np.sum(raw_s[0:4][::2]) / 2.0], [movement_velocity], [t_velocity]
    ]).astype(np.float32)

def update_robot_memory(next_raw_s, action, ir_latch, arc_counts, fw_count, turn_count):
    """Updates internal counters"""
    if next_raw_s[16] > 0.1: ir_latch = min(20, ir_latch + 2)
    else: ir_latch = max(0, ir_latch - 1)
        
    is_near_front = np.sum(next_raw_s[4:12][::2]) > 0.8
    arc_counts[0] = arc_counts[0] + 1 if (is_near_front and action != 2) else 0
    arc_counts[1] = arc_counts[1] + 1 if (np.sum(next_raw_s[12:16][::2]) > 0.8 and action != 2) else 0
    arc_counts[2] = arc_counts[2] + 1 if (np.sum(next_raw_s[0:4][::2]) > 0.8 and action != 2) else 0

    if action == 2: fw_count, turn_count = fw_count + 1, 0
    else: turn_count, fw_count = turn_count + 1, 0
        
    return ir_latch, arc_counts, fw_count, turn_count

def update_odometry(self, last_action_idx, current_raw_obs):
    """
    Updates internal coordinates and returns the change in distance from origin.
   
    """
    if last_action_idx == 0: 
        self.est_angle -= np.radians(45)
    elif last_action_idx == 1: 
        self.est_angle -= np.radians(22)
    elif last_action_idx == 3: 
        self.est_angle += np.radians(22)
    elif last_action_idx == 4: 
        self.est_angle += np.radians(45)
    
    is_physically_stuck = current_raw_obs[17] > 0.5
    
    if last_action_idx == 2 and not is_physically_stuck:
        self.est_x += np.cos(self.est_angle) * 1.5
        self.est_y += np.sin(self.est_angle) * 1.5
    
    prev_dist = self.current_dist
    self.current_dist = np.sqrt(self.est_x**2 + self.est_y**2)
    
    return self.current_dist - prev_dist

def _load_once():
    global _model
    if _model is not None: return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    _model = Actor(in_dim=40, out_dim=5)
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    _model.load_state_dict(sd, strict=True)
    # _model.eval()
    _model.eval()

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _mem
    _load_once()
    
    if _mem.prev_raw_obs is None:
        _mem.prev_raw_obs = obs.copy()

    dist_delta = _mem.update_odometry(_mem.last_action_idx, obs)

    aug_obs = get_inference_obs(
        obs, 
        _mem.prev_raw_obs, 
        _mem.step_counter, 
        _mem.consecutive_fw,
        _mem.consecutive_turns, 
        _mem.ir_latch_count, 
        _mem.last_action_idx, 
        _mem.arc_counters, 
        dist_delta
    )
    
    state_t = torch.as_tensor(aug_obs, dtype=torch.float32).unsqueeze(0)
    probs = _model(state_t).squeeze(0).cpu().numpy()
    action_idx = int(np.argmax(probs))

    _mem.ir_latch_count, _mem.arc_counters, _mem.consecutive_fw, _mem.consecutive_turns = update_robot_memory(
        obs, action_idx, _mem.ir_latch_count, _mem.arc_counters, _mem.consecutive_fw, _mem.consecutive_turns
    )
    
    _mem.prev_raw_obs = obs.copy()
    _mem.last_action_idx = action_idx
    _mem.step_counter += 1
    
    return ACTIONS[action_idx]