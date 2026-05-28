import torch
import torch.nn.functional as F
from typing import Tuple, Dict

# 從獨立模組導入 Runner
from runner import SelfPlayRunner


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    廣義優勢估計 (GAE)
    """
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    device = values.device
    for t in reversed(range(len(rewards))):
        nextvalues = torch.tensor(0.0, device=device) if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalues - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * lastgaelam
    returns = advantages + values          
    return advantages, returns


def train_ppo_epoch(model, runner: SelfPlayRunner, optimizer, epochs: int = 4) -> Dict[str, float]:
    """
    管理訓練調度與 PPO 損失計算（動作遮罩完美對齊安全版）
    """
    model.prepare_for_ppo()
    model.eval()
    
    trajectories = runner.run_match()      
    device = next(model.parameters()).device
    
    total_metrics = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "avg_reward": 0.0, "trajectory_length": 0
    }
    valid_trajs = 0
    total_episodes_reward = 0.0
    total_episodes_length = 0
                          
    for pid, traj in trajectories.items():
        if len(traj) == 0: continue
        valid_trajs += 1
        
        total_episodes_reward += sum(step["reward"] for step in traj)
        total_episodes_length += len(traj)
        
        states = torch.stack([step["obs"] for step in traj]).to(device)            
        actions = torch.tensor([step["action"] for step in traj]).to(device)       
        log_probs = torch.tensor([step["log_prob"] for step in traj]).to(device)   
        rewards = torch.tensor([step["reward"] for step in traj]).to(device, dtype=torch.float32)       
        timesteps = torch.tensor([step["timestep"] for step in traj]).to(device)   
        
        # 📌 修正行 1：從軌跡提取當時環境決策的動作遮罩
        masks = torch.stack([step["mask"] for step in traj]).to(device)
        
        seq_len = states.size(0)
        start_token = torch.tensor([180], dtype=torch.long, device=device)
        input_actions = torch.cat([start_token, actions[:-1]])
        
        input_rtgs = torch.zeros(seq_len, 1, dtype=torch.float32, device=device)
        running_rtg = 0.0
        for t in reversed(range(seq_len)):
            running_rtg += rewards[t].item()
            input_rtgs[t, 0] = running_rtg
        
        states_seq = states.unsqueeze(0)       
        timesteps_seq = timesteps.unsqueeze(0)
        input_actions_seq = input_actions.unsqueeze(0)
        input_rtgs_seq = input_rtgs.unsqueeze(0)
        
        with torch.no_grad():
            _, values, _, _ = model(input_rtgs_seq, states_seq, input_actions_seq, timesteps_seq)
            values = values.squeeze(0).squeeze(-1) 
            
        advantages, returns = compute_gae(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        model.train()
        traj_policy_loss, traj_value_loss, traj_entropy = 0, 0, 0
        
        for epoch in range(epochs):
            actor_logits, new_values, _, _ = model(input_rtgs_seq, states_seq, input_actions_seq, timesteps_seq)
            actor_logits_seq = actor_logits.squeeze(0) 
            
            # 📌 修正行 2：建立對齊矩陣，並利用布林索引將非法動作重置為 -inf
            masked_logits = torch.full_like(actor_logits_seq, float('-inf'))
            legal_indices = (masks != float('-inf'))
            masked_logits[legal_indices] = actor_logits_seq[legal_indices]
            
            # 📌 修正行 3：使用精準對齊後的遮罩進行 log_softmax 與隨後的分佈熵計算
            log_probs_dist = F.log_softmax(masked_logits, dim=-1)
            new_log_probs = log_probs_dist.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            probs = F.softmax(masked_logits, dim=-1)
            entropy = -(probs * log_probs_dist).sum(dim=-1).mean()
            if torch.isnan(entropy):
                entropy = torch.tensor(0.0, device=device)
            
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(new_values.squeeze(0).squeeze(-1), returns)
            loss = policy_loss + 1.0 * value_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
            optimizer.step()
            
            traj_policy_loss += policy_loss.item()
            traj_value_loss += value_loss.item()
            traj_entropy += entropy.item()
            
        total_metrics["policy_loss"] += traj_policy_loss / epochs
        total_metrics["value_loss"] += traj_value_loss / epochs
        total_metrics["entropy"] += traj_entropy / epochs

    if valid_trajs == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "avg_reward": 0.0, "trajectory_length": 0}
    
    total_metrics["avg_reward"] = total_episodes_reward / valid_trajs
    total_metrics["trajectory_length"] = total_episodes_length / valid_trajs
    
    return {
        k: v / valid_trajs if k != "avg_reward" and k != "trajectory_length" else v 
        for k, v in total_metrics.items()
    }