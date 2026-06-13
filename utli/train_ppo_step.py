"""
強化學習訓練函數 - 逐步獎勵版（Batch 化：Mamba 隱狀態在局間完全隔離）
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
    num_trajectories=32,
):
    """
    訓練一個 PPO epoch。收集 N 局軌跡，Batch 化輸入 Mamba 以確保跨局隱狀態隔離。

    🆕 Batch 化設計：
      - 每局獨立收集軌跡
      - Padding 到 N 局中的 max_seq_len
      - Stack 成 (B, max_seq_len, ...) 送入 Mamba
      - Mamba 在 batch 維度間 SSM 隱狀態互不干擾
      - 用 valid_mask (B, max_seq_len) 排除 padding 位置的 loss

    Args:
        model: DecisionMamba 模型
        runner: SelfPlayRunner 實例
        optimizer: PyTorch Optimizer
        epochs: PPO 更新 epoch 數
        gamma: GAE 衰減係數
        lam: GAE λ 參數
        clip_epsilon: PPO 剪裁範圍 ε
        value_coef: Value Loss 權重 c1
        entropy_coef: 策略熵係數 c2
        max_grad_norm: 梯度裁剪閾值
        temperature: Logit 採樣溫度
        device: 計算設備
        num_trajectories: 每 iter 收集的軌跡數

    Returns:
        dict: {'policy_loss', 'value_loss', 'entropy', 'avg_reward', 'trajectory_length', 'game_result'}
    """
    import torch.nn.functional as F
    import torch
    import numpy as np
    import random

    model.train()
    model.prepare_for_ppo()

    # ── 🆕 收集 N 局軌跡，每局獨立儲存 ──
    game_data = []  # list of dicts, each: {"obs": [(D,)], "action": [int], ...}
    total_reward = 0.0
    max_len = 0
    all_game_results = []

    for traj_idx in range(num_trajectories):
        trajectories_dict, game_result = runner.run_match(temperature=temperature)
        all_game_results.append(game_result)

        game_entry = {
            "obs": [],
            "action": [],
            "reward": [],
            "log_prob": [],
            "timestep": [],
            "legal_mask": [],
        }
        for pid, pid_trajectories in trajectories_dict.items():
            if len(pid_trajectories) == 0:
                continue
            for step_data in pid_trajectories:
                game_entry["obs"].append(step_data["obs"].cpu())
                game_entry["action"].append(step_data["action"])
                game_entry["reward"].append(step_data["reward"])
                game_entry["log_prob"].append(step_data["log_prob"])
                game_entry["timestep"].append(step_data["timestep"])
                game_entry["legal_mask"].append(step_data["mask"].cpu())

        if len(game_entry["obs"]) == 0:
            continue

        T = len(game_entry["obs"])
        max_len = max(max_len, T)
        total_reward += sum(game_entry["reward"])  # 🆕 整局總獎勵（非最後一步）
        game_data.append(game_entry)

    B = len(game_data)
    if B == 0:
        return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "avg_reward": 0, "trajectory_length": 0, "game_results": []}

    # ── 🆕 Padding + Stack 成 (B, max_seq_len, ...) ──
    obs_pad = torch.zeros(B, max_len, 1380)
    action_pad = torch.zeros(B, max_len, dtype=torch.long)
    reward_pad = torch.zeros(B, max_len)
    log_prob_pad = torch.zeros(B, max_len)
    timestep_pad = torch.zeros(B, max_len, dtype=torch.long)
    legal_mask_pad = torch.zeros(B, max_len, 181, dtype=torch.bool)
    valid_mask = torch.zeros(B, max_len, dtype=torch.bool)  # True = 有效位置

    for i, g in enumerate(game_data):
        T = len(g["obs"])
        valid_mask[i, :T] = True
        for t in range(T):
            obs_pad[i, t] = g["obs"][t]
            action_pad[i, t] = g["action"][t]
            reward_pad[i, t] = g["reward"][t]
            log_prob_pad[i, t] = g["log_prob"][t]
            timestep_pad[i, t] = g["timestep"][t]
            legal_mask_pad[i, t] = g["legal_mask"][t]

    # ── RTG 計算（每局獨立計算，padding 位置留 0）──
    rtg_pad = torch.zeros(B, max_len, 1)
    for i, g in enumerate(game_data):
        T = len(g["obs"])
        running_rtg = 1.0
        for t in range(T - 1, -1, -1):
            running_rtg += g["reward"][t]
            rtg_pad[i, t, 0] = running_rtg

    # ── Padded 輸入（T+1 步，最後一位為 padding token）──
    padded_act = torch.cat([action_pad, torch.zeros(B, 1, dtype=torch.long)], dim=1)  # (B, T+1)
    padded_rtg = torch.cat([rtg_pad, torch.ones(B, 1, 1) * 1.0], dim=1)  # (B, T+1, 1)
    padded_timestep = torch.cat([timestep_pad, torch.zeros(B, 1, dtype=torch.long)], dim=1)  # (B, T+1)
    padded_obs = torch.cat([obs_pad, torch.zeros(B, 1, 1380)], dim=1)  # (B, T+1, 1380)

    # ── 搬移至設備 ──
    obs_pad = obs_pad.to(device)
    action_pad = action_pad.to(device)
    reward_pad = reward_pad.to(device)
    log_prob_pad = log_prob_pad.to(device)
    timestep_pad = timestep_pad.to(device)
    legal_mask_pad = legal_mask_pad.to(device)
    valid_mask = valid_mask.to(device)
    rtg_pad = rtg_pad.to(device)

    padded_act = padded_act.to(device)
    padded_rtg = padded_rtg.to(device)
    padded_timestep = padded_timestep.to(device)
    padded_obs = padded_obs.to(device)

    target_actions = action_pad          # (B, T)
    target_log_probs = log_prob_pad      # (B, T)
    target_rewards = reward_pad          # (B, T)
    target_legal_mask = legal_mask_pad   # (B, T, 181)

    # ── Value 計算（T+1 步 padded 輸入，最後一位用於 bootstrapping）──
    with torch.no_grad():
        model.eval()
        _, values_all, _, _ = model(padded_rtg, padded_obs, padded_act, padded_timestep)
        model.train()
        values_all = values_all.squeeze(-1)  # (B, T+1)

    # ── 🆕 GAE 逐局計算（Mamba 隱狀態不跨局，GAE 也不跨局）──
    advantages = torch.zeros(B, max_len, device=device)
    for i, g in enumerate(game_data):
        T = len(g["obs"])
        gae = 0.0
        for t in range(T - 1, -1, -1):
            next_val = values_all[i, t + 1]
            delta = reward_pad[i, t] + gamma * next_val - values_all[i, t]
            gae = delta + gamma * lam * gae
            advantages[i, t] = gae

    # returns = advantages + values（僅前 T 步）
    returns = advantages + values_all[:, :max_len]  # (B, T)

    # ── 標準化（僅在有效位置上計算）──
    adv_flat = advantages[valid_mask]
    adv_mean = adv_flat.mean()
    adv_std = adv_flat.std() if adv_flat.numel() > 1 else torch.tensor(1.0, device=device)
    advantages = (advantages - adv_mean) / (adv_std + EPS)

    # 🆕 Return Z-score normalization：讓 Critic 預測目標在 ~[-3, +3] 穩定範圍
    #    取代 per-batch ret_std 除法，消除 Value Loss 因 batch 組成不同造成的劇烈跳動
    ret_valid = returns[valid_mask]
    ret_mean = ret_valid.mean()
    ret_std = ret_valid.std() if ret_valid.numel() > 1 else torch.tensor(1.0, device=device)
    returns_norm = (returns - ret_mean) / (ret_std + EPS)

    # ── PPO 更新迴圈 ──
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0

    for epoch in range(epochs):
        actor_logits, critic_values, _, _ = model(padded_rtg, padded_obs, padded_act, padded_timestep)

        # 只取前 T 步的輸出
        logits = actor_logits[:, :max_len, :]      # (B, T, 181)
        new_values = critic_values[:, :max_len, :].squeeze(-1)  # (B, T)

        # Action masking
        masked_logits = torch.where(target_legal_mask, logits, torch.tensor(NEG_INF, device=device))

        # Log probs
        log_probs = F.log_softmax(masked_logits / temperature, dim=-1)
        new_log_probs = log_probs.gather(2, target_actions.unsqueeze(-1)).squeeze(-1)  # (B, T)

        probs = F.softmax(masked_logits, dim=-1)

        # Entropy（僅有效位置）
        legal_mask_float = target_legal_mask.float()
        legal_probs = probs * legal_mask_float
        legal_log_probs = torch.where(target_legal_mask, log_probs, torch.tensor(0.0, device=device))
        ent_per_step = -(legal_probs * legal_log_probs).sum(dim=-1)  # (B, T)
        entropy = (ent_per_step * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

        # PPO ratio
        ratio = torch.exp(new_log_probs - target_log_probs)  # (B, T)
        ratio = torch.clamp(ratio, 0.0, 10.0)

        adv = advantages  # (B, T)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
        policy_loss_per_step = -torch.min(surr1, surr2)  # (B, T)
        policy_loss = (policy_loss_per_step * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

        # Value loss（returns_norm 已做 z-score 標準化，目標值穩定在 ~[-3,+3]）
        value_diff = (new_values - returns_norm) ** 2  # (B, T)
        value_loss = (value_diff * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

        # 總損失
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()

    avg_policy_loss = total_policy_loss / epochs
    avg_value_loss = total_value_loss / epochs
    avg_entropy = total_entropy / epochs
    avg_reward = total_reward / max(1, B)

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
        "game_results": all_game_results,
    }
