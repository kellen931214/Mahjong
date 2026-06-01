"""
強化學習訓練函數 - 逐步獎勵版
管理 PPO 訓練循環，包含軌跡收集、GAE 計算與 PPO 更新
"""
import torch
import numpy as np
from collections import defaultdict

# 數值安全常數：用於替代 float('-inf')，避免 NaN 傳播
NEG_INF = -1e9
EPS = 1e-8

def train_ppo_epoch(
    model,
    runner,
    optimizer,
    epochs=1,
    gamma=0.99,
    lam=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.05,
    max_grad_norm=0.5,
    temperature=2.0,
    device="cuda",
):
    """
    訓練一個 PPO epoch。軌跡收集與反向傳播全部 inline，確保每一步都在 with torch.no_grad() 外。

    Args:
        model: DecisionMamba 模型
        runner: SelfPlayRunner 實例
        optimizer: PyTorch Optimizer
        epochs: PPO 更新 epoch 數（LoRA 建議 1，因數據只洗禮一次）
        gamma: GAE 衰減係數
        lam: GAE λ 參數
        clip_epsilon: PPO 剪裁範圍 ε（安全閥，防止極端獎勵拉偏 Policy）
        value_coef: Value Loss 權重 c1（平衡 Critic 與 Actor 學習速度）
        entropy_coef: 策略熵係數 c2（控制探索強度）
        max_grad_norm: 梯度裁剪閾值（防止 GAE 劇烈變動導致梯度爆炸）
        temperature: Logit 採樣溫度（Softmax 前除以係數，拉平極端分佈）
        device: 計算設備

    Returns:
        dict: {'policy_loss': float, 'value_loss': float, 'entropy': float,
               'avg_reward': float, 'trajectory_length': float, 'game_result': dict}
    """
    import torch.nn.functional as F
    import torch
    import numpy as np
    import random
    from rewards import create_default_calculator
    from torch.distributions import Categorical
    import mjx

    model.train()
    model.prepare_for_ppo()

    all_trajectory_data = defaultdict(list)
    total_reward = 0.0
    max_len = 0

    trajectories_dict, game_result = runner.run_match(temperature=temperature)
    for pid, pid_trajectories in trajectories_dict.items():
        if len(pid_trajectories) == 0:
            continue

        for step_data in pid_trajectories:
            all_trajectory_data["obs"].append(step_data["obs"].cpu().unsqueeze(0))
            all_trajectory_data["action"].append(torch.tensor(step_data["action"], dtype=torch.long).unsqueeze(0))
            all_trajectory_data["reward"].append(torch.tensor(step_data["reward"], dtype=torch.float32).unsqueeze(0))
            all_trajectory_data["log_prob"].append(torch.tensor(step_data["log_prob"], dtype=torch.float32).unsqueeze(0))
            all_trajectory_data["timestep"].append(torch.tensor(step_data["timestep"], dtype=torch.long).unsqueeze(0))
            # 🚀 直接使用 runner 提供的 boolean legal_mask
            legal_mask_i = step_data["mask"]
            all_trajectory_data["legal_mask"].append(legal_mask_i.cpu().unsqueeze(0))

        trajectory_len = len(pid_trajectories)
        total_reward += pid_trajectories[-1]["reward"] if trajectory_len > 0 else 0.0
        max_len = max(max_len, trajectory_len)

    # 將每個 step 堆疊成完整軌跡張量
    obs_sequence = torch.cat(all_trajectory_data["obs"]).unsqueeze(0)      # (1, seq_len, state_dim)
    act_sequence = torch.cat(all_trajectory_data["action"]).unsqueeze(0)     # (1, seq_len)
    reward_sequence = torch.cat(all_trajectory_data["reward"]).unsqueeze(0)  # (1, seq_len)
    timestep_sequence = torch.cat(all_trajectory_data["timestep"]).unsqueeze(0)  # (1, seq_len)
    log_prob_sequence = torch.cat(all_trajectory_data["log_prob"]).unsqueeze(0)  # (1, seq_len)

    # 🚀 legal_mask 使用 boolean，形狀為 (1, seq_len, action_dim)
    legal_mask_sequence = torch.cat(all_trajectory_data["legal_mask"]).unsqueeze(0).bool()  # (1, seq_len, action_dim)

    # 🔧【修正】rtg_sequence 需要最後一維為 1，與 BC 訓練時的形狀 (batch, seq_len, 1) 對齊
    rtg_sequence = torch.zeros_like(reward_sequence).unsqueeze(-1)  # (1, seq_len, 1)
    running_rtg = 1.0
    for t in range(reward_sequence.shape[1] - 1, -1, -1):
        running_rtg += reward_sequence[0, t].item()
        rtg_sequence[0, t, 0] = running_rtg

    # ======================================================================
    # 🚀 模型前向傳播：構造 Mamba 訓練所需的 Padded 輸入 (T+1 步)
    # ======================================================================
    # 1. act_sequence 形狀為 (1, T) -> padded 後變為 (1, T+1)
    padded_act = torch.cat([act_sequence, torch.zeros(1, 1, dtype=torch.long)], dim=1) 
    
    # 2. rtg_sequence 形狀為 (1, T, 1) -> padded 後變為 (1, T+1, 1)
    # 🔧【修正】rtg_padding_token 也必須是 (1, 1, 1) 以維持三維結構
    rtg_padding_token = torch.ones(1, 1, 1) * 1.0  
    padded_rtg = torch.cat([rtg_sequence, rtg_padding_token], dim=1)  
    
    # 3. timestep_sequence 形狀為 (1, T) -> padded 後變為 (1, T+1)
    padded_timestep = torch.cat([timestep_sequence, torch.zeros(1, 1, dtype=torch.long)], dim=1) 
    
    # 4. obs_sequence 形狀為 (1, T, 1380) -> padded 後變為 (1, T+1, 1380)
    obs_padding_token = torch.zeros(1, 1, obs_sequence.shape[-1])  # (1, 1, 1380)
    padded_obs = torch.cat([obs_sequence, obs_padding_token], dim=1) 

    # ========================
    # 搬移至設備與分配優化目標
    # ========================
    obs_sequence = obs_sequence.to(device)
    act_sequence = act_sequence.to(device)
    reward_sequence = reward_sequence.to(device)
    timestep_sequence = timestep_sequence.to(device)
    log_prob_sequence = log_prob_sequence.to(device)
    legal_mask_sequence = legal_mask_sequence.to(device)
    rtg_sequence = rtg_sequence.to(device)

    padded_act = padded_act.to(device)
    padded_rtg = padded_rtg.to(device)
    padded_timestep = padded_timestep.to(device)
    padded_obs = padded_obs.to(device)

    target_actions = act_sequence          # (1, seq_len)
    target_log_probs = log_prob_sequence   # (1, seq_len)
    target_rewards = reward_sequence       # (1, seq_len)
    target_legal_mask = legal_mask_sequence # (1, seq_len, action_dim)

    # ========================
    # Value 計算（使用完整的 T+1 步 padded 輸入）
    # ========================
    with torch.no_grad():
        model.eval()
        _, values_all, _, _ = model(padded_rtg, padded_obs, padded_act, padded_timestep)
        model.train()
        values_all = values_all.squeeze(0).squeeze(-1)  # (seq_len + 1,)

    # ========================
    # GAE 計算（精準利用第 seq_len 個位置進行 Bootstrapping）
    # ========================
    seq_len = reward_sequence.shape[1] # 取得實體軌跡長度 T
    advantages = torch.zeros_like(reward_sequence) # (1, seq_len)
    gae = 0.0
    
    for t in reversed(range(seq_len)):
        # 🚀 完美的自舉：t = seq_len-1 時，next_val = values_all[t + 1]，成功拿到了 Padding 步的預測價值！
        next_val = values_all[t + 1] 
        delta = reward_sequence[0, t] + gamma * next_val - values_all[t]
        gae = delta + gamma * lam * gae
        advantages[0, t] = gae
        
    # returns 的基準應該對齊前 seq_len 步 the 實體價值
    returns = advantages + values_all[:seq_len]

    # ========================
    # 🛡️ 標準化（含數量守衛，防止單元素 std() 返回 NaN）
    # ========================
    adv_mean = advantages.mean()
    adv_std = advantages.std() if advantages.numel() > 1 else torch.tensor(0.0, device=device)
    advantages = (advantages - adv_mean) / (adv_std + EPS)

    ret_mean = returns.mean()
    ret_std = returns.std() if returns.numel() > 1 else torch.tensor(0.0, device=device)
    returns_norm = (returns - ret_mean) / (ret_std + EPS)

    # ========================
    # PPO 更新迴圈
    # ========================
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0

    for epoch in range(epochs):
        # 前向：取得當前策略的 logits 和 values
        actor_logits, critic_values, _, _ = model(padded_rtg, padded_obs, padded_act, padded_timestep)

        # 只取前 T 步的輸出（最後一個位置僅用於 value bootstrapping）
        logits = actor_logits[:, :seq_len, :].squeeze(0)  # (T, action_dim)
        new_values = critic_values[:, :seq_len, :].squeeze(0).squeeze(-1)  # (T,)

        # 🚀 使用大有限負數 NEG_INF 取代 float('-inf') 來做 action masking
        legal_mask = target_legal_mask.squeeze(0)  # (T, action_dim), boolean
        masked_logits = torch.where(legal_mask, logits, torch.tensor(NEG_INF, device=device))

        # 策略分佈與對數機率（套用 temperature 拉平極端 logit 分佈）
        log_probs = F.log_softmax(masked_logits / temperature, dim=-1)
        new_log_probs = log_probs.gather(1, target_actions.squeeze(0).unsqueeze(1)).squeeze(1)  # (T,)

        probs = F.softmax(masked_logits, dim=-1)

        # 0.0 * (-1e9) = 0.0（不是 NaN！），正向反向都乾淨
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # PPO 損失
        ratio = torch.exp(new_log_probs - target_log_probs.squeeze(0))  # (T,)

        # 🛡️ Ratio clamping：防止數值 overflow
        ratio = torch.clamp(ratio, 0.0, 10.0)

        adv = advantages.squeeze(0)[:seq_len]
        surr1 = ratio * adv  # (T,)
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value 損失
        value_loss = F.mse_loss(new_values, returns_norm.squeeze(0)[:seq_len])

        # 總損失
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()

        # 🛡️ 梯度裁剪前先檢查是否有 NaN 梯度
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()

    avg_policy_loss = total_policy_loss / epochs
    avg_value_loss = total_value_loss / epochs
    avg_entropy = total_entropy / epochs
    avg_reward = total_reward / max(1, len(trajectories_dict))

    # ======================================================================
    # 📊 🚀【新增列印段落】每個 Epoch 結束時將數據格式化輸出到終端機
    # ======================================================================
    print(f"📈 [Epoch Metrics] "
          f"Policy Loss: {avg_policy_loss:8.4f} | "
          f"Value Loss: {avg_value_loss:8.4f} | "
          f"Entropy: {avg_entropy:6.4f} | "
          f"Avg Reward: {avg_reward:7.2f} | "
          f"Steps: {max_len:4d}")

    return {
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "entropy": avg_entropy,
        "avg_reward": avg_reward,
        "trajectory_length": max_len,
        "game_result": game_result,
    }