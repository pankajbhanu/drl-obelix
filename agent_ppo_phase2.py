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
from torch.distributions import Categorical
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, in_dim=38, out_dim=5):
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
            self.actor[0].bias[2] += 0.001

    def act(self, state):
        features = self.base(state)
        logits = self.actor[0](features)
        
  
        probs = torch.softmax(logits, dim=-1)
        probs = probs + 1e-9
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.detach(), dist.log_prob(action).detach(), self.critic(features).detach()

    def evaluate(self, state, action):
        features = self.base(state)
        logits = self.actor[0](features)
        
        probs = torch.softmax(logits, dim=-1) + 1e-9
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        dist = Categorical(probs)
        return dist.log_prob(action), self.critic(features), dist.entropy()

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, device=torch.device('cpu'), writer=None):
        self.device = device
        self.gamma = gamma
        self.eps_clip = 0.2          
        self.gamma = 0.98            
        self.K_epochs = 8            
        self.mini_batch_size = 64    
        self.vf_coef = 0.5           
        self.ent_coef = 0.05        

        self.buffer = RolloutBuffer()
        self.writer = writer

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.base.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.MseLoss = nn.MSELoss()

    def get_action(self, state, evaluate=True):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            features = self.policy_old.base(state)
            probs = self.policy_old.actor(features)
            
            if evaluate:
                action = torch.argmax(probs, dim=1) 
            else:
                dist = Categorical(probs)
                action = dist.sample()              
                
        return action.item()

    def select_action(self, state):
        """Used ONLY in the main training loop."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            action, action_logprob, state_value = self.policy_old.act(state)
        
        # Save to buffer
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_value)
        
        return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        old_states = torch.stack(self.buffer.states).to(self.device).detach().squeeze()
        old_actions = torch.stack(self.buffer.actions).to(self.device).detach().squeeze()
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device).detach().squeeze()
        old_state_values = torch.stack(self.buffer.state_values).to(self.device).detach().squeeze()

        advantages = rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        total_entropy = 0
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            total_entropy += dist_entropy.mean().item()
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.05 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer.clear()
        return total_entropy / self.K_epochs  
    
    
    def update_ppo(self, b_states, b_actions, b_log_probs, b_returns, b_advantages, global_step):
        """
        Updates the Actor and Critic networks using the GAE data.
        All inputs should already be PyTorch Tensors on the correct device.
        """
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_entropies = []
        batch_size = b_states.size(0)
        
        for _ in range(self.K_epochs):
            
            indices = torch.randperm(batch_size).to(b_states.device)
            
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]
                
                mb_states = b_states[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_old_log_probs = b_log_probs[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_advantages = b_advantages[mb_indices]

                mb_features = self.policy.base(mb_states)

                action_probs = self.policy.actor(mb_features)
                action_probs = action_probs + 1e-9
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(mb_actions)
                dist_entropy = dist.entropy().mean()

                state_values = self.policy.critic(mb_features).squeeze(-1)

                ratios = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = self.MseLoss(state_values, mb_returns)

                loss = actor_loss - (self.ent_coef * dist_entropy) + (self.vf_coef * critic_loss)

                self.optimizer.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=0.5)
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_norm=0.5)
                
                self.optimizer.step()

                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())
                epoch_entropies.append(dist_entropy.item())

        self.policy_old.load_state_dict(self.policy.state_dict())

        # self.writer.add_scalar('Loss/Total_Loss', loss.mean().item(), global_step)
        # self.writer.add_scalar('Loss/Entropy', dist_entropy.mean().item(), global_step)
        return epoch_actor_losses, epoch_critic_losses, epoch_entropies

def compute_gae(rewards, values, next_value, masks, gamma=0.98, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - masks[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - masks[t]
            next_values = values[t+1]
            
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]

        advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


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
    if raw_s[16] > 0.5 or latch_confidence > 0.6 or front_near > 0.5:
        beacon_signal = (0.5 + 0.5 * (latch_confidence + front_near))
    else:
        beacon_signal = -1.0

    t_s_phase = (step_counter % 80) / 80.0
    t_s_sin, t_s_cos = np.sin(2*np.pi*t_s_phase), np.cos(2*np.pi*t_s_phase)
    
    t_l_phase = (step_counter % 200) / 200.0
    t_l_sin, t_l_cos = np.sin(2*np.pi*t_l_phase), np.cos(2*np.pi*t_l_phase)

    rot_fatigue = min(1.0, consecutive_turns / 30.0)
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

_model: Optional[PPOAgent] = None
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
    m = PPOAgent(38, 5, lr_actor=1e-4, lr_critic=3e-4, device=torch.device("cpu"), writer=None)
    
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.policy.load_state_dict(sd, strict=True)
    m.policy.eval()
    m.policy_old.load_state_dict(m.policy.state_dict())
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
    q = _model.policy.act(x)[0][0].cpu().numpy()
    best = int(np.argmax(q))
    prev_obs = obs.copy() 

    # if _last_action is not None:
    #     order = np.argsort(-q)
    #     best_q, second_q = float(q[order[0]]), float(q[order[1]])
    #     if (best_q - second_q) < _CLOSE_Q_DELTA:
    #         if _repeat_count < _MAX_REPEAT:
    #             best = _last_action
    #             _repeat_count += 1
    #         else:
    #             _repeat_count = 0
    #     else:
    #         _repeat_count = 0

    # _last_action = best
    return ACTIONS[best]