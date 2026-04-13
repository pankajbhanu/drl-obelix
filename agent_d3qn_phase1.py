"""Better DQN-style agent scaffold for OBELIX (CPU).

This agent is *evaluation-only*: it loads pretrained weights from a file
placed next to agent.py inside the submission zip (weights.pth).

Why your STD is huge:
- if the policy is stochastic (epsilon > 0) during evaluation, scores vary a lot.
Fix:
- greedy action selection (epsilon=0), model.eval(), torch.no_grad().
- optional action smoothing to reduce oscillation when Q-values are close.

Submission ZIP structure:
  submission.zip
    agent.py
    weights.pth
"""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class DuelingDQN(nn.Module):
    def __init__(self, in_dim=38, out_dim=5):
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
    
def get_inference_obs(raw_s, prev_raw_s, step_counter, consecutive_fw, consecutive_turns, 
                    ir_latch_count, last_action, wide_arc_counters):
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

_model: Optional[DuelingDQN] = None
_last_action: Optional[int] = None
_repeat_count: int = 0
_step_counter: int = 0  

ir_latch_count = 0
consecutive_fw = 0 
consecutive_turns = 0
arc_counters = [0, 0, 0]
prev_obs: Optional[np.ndarray] = np.zeros(18, dtype=np.float32)

_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05

def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    m = DuelingDQN(in_dim=38, out_dim=5)
    
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _step_counter, ir_latch_count, consecutive_fw, consecutive_turns, arc_counters, prev_obs
    _load_once()
    _step_counter += 1

    action_str = ACTIONS[_last_action] if _last_action is not None else "FW"
    
    is_forward = (action_str == "FW")
    if np.sum(obs[4:12][::2]) > 0.8 and _last_action != 2 and not obs[16] > 0.5:
        arc_counters[0] += 1
    else: arc_counters[0] = 0
    arc_counters[1] = arc_counters[1] + 1 if (np.sum(obs[12:16][::2]) > 0.8 and _last_action != 2) else 0
    arc_counters[2] = arc_counters[2] + 1 if (np.sum(obs[0:4][::2]) > 0.8 and _last_action != 2) else 0

    if is_forward:
        consecutive_fw += 1
        consecutive_turns = 0
    else:
        consecutive_turns += 1
        consecutive_fw = 0
    
    

    obs_aug = get_inference_obs(obs, prev_obs, _step_counter, consecutive_fw, 
                                consecutive_turns, ir_latch_count, _last_action, arc_counters)
    
    x = torch.tensor(obs_aug, dtype=torch.float32).unsqueeze(0)
    q = _model(x).squeeze(0).cpu().numpy()
    best = int(np.argmax(q))
    prev_obs = obs.copy()  

    if _last_action is not None:
        order = np.argsort(-q)
        best_q, second_q = float(q[order[0]]), float(q[order[1]])
        if (best_q - second_q) < _CLOSE_Q_DELTA:
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_action = best
    return ACTIONS[best]
