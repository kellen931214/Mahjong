import torch
import torch.nn.functional as F
from mjx.env import MjxEnv
import numpy as np
from torch.distributions import Categorical
import copy
import random
from typing import List, Dict

# 導入本地獎勵計算模組
from rewards import create_default_calculator

class SelfPlayRunner:
    def __init__(self, model, device: str = "cuda", opponent_pool_size: int = 5):
        """
        初始化自我博弈環境與對手池
        """
        self.model = model.to(device)
        self.device = device
        self.env = MjxEnv()        
        self.opponent_pool = [copy.deepcopy(self.model).eval()]
        self.opponent_pool_size = opponent_pool_size
        self.reward_calculator = create_default_calculator()

    def update_opponent_pool(self):
        """
        更新對手池：將當前最新模型克隆後加入池中
        """
        new_opponent = copy.deepcopy(self.model).eval()
        self.opponent_pool.append(new_opponent)
        if len(self.opponent_pool) > self.opponent_pool_size:
            self.opponent_pool.pop(1)

    def extract_model_input(self, obs) -> torch.Tensor:
        """
        將 mjx 的 Observation 轉換為 Mamba 模型特徵張量
        直接使用 mjx 內建的 decision-mamba-v0 特徵（1380 維）
        """
        features = obs.to_features("decision-mamba-v0")  # np.ndarray shape=(1380,), dtype=float32
        return torch.from_numpy(features).float().to(self.device)

    def run_match(self) -> Dict[int, List[Dict]]:    
        """
        使用 MjxEnv 進行一局完整的遊戲對弈並收集軌跡
        """
        obs_dict = self.env.reset()
        agent_pid = random.choice([0, 1, 2, 3])
        agent_key = f"player_{agent_pid}" # 📌 預先定義主 Agent 的 Key 方便辨識
        
        assigned_models = {
            agent_pid: self.model,
            **{pid: random.choice(self.opponent_pool) for pid in [0, 1, 2, 3] if pid != agent_pid}
        }
        
        trajectories = {agent_pid: []}
        obs_histories = {pid: [] for pid in range(4)}
        timesteps_histories = {pid: [] for pid in range(4)}
        
        act_histories = {pid: [180] for pid in range(4)} # 180 爲 mjx Pass Token
        rtg_histories = {pid: [1.0] for pid in range(4)}
        step_counts = {pid: 0 for pid in range(4)}
        
        MAX_CONTEXT_LEN = 128
        step_log_counter = 0
        
        while not self.env.done():
            current_player_key = list(obs_dict.keys())[0]  
            current_pid = int(current_player_key.split('_')[1])
            obs = obs_dict[current_player_key]
            
            legal_actions = obs.legal_actions()
            if len(legal_actions) == 0:
                break
            
            state_tensor = self.extract_model_input(obs)
            obs_histories[current_pid].append(state_tensor)
            timesteps_histories[current_pid].append(step_counts[current_pid])
            
            hist_len = min(len(obs_histories[current_pid]), MAX_CONTEXT_LEN)
            
            seq_state = torch.stack(obs_histories[current_pid][-hist_len:]).unsqueeze(0).to(self.device)
            seq_time = torch.tensor(timesteps_histories[current_pid][-hist_len:], dtype=torch.long, device=self.device).unsqueeze(0)
            seq_act = torch.tensor(act_histories[current_pid][-hist_len:], dtype=torch.long, device=self.device).unsqueeze(0)
            seq_rtg = torch.tensor(rtg_histories[current_pid][-hist_len:], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)
            
            legal_indices = [act.to_idx() if hasattr(act, 'to_idx') else int(act) for act in legal_actions]
            
            with torch.no_grad():
                actor_logits, _, _, _ = assigned_models[current_pid](seq_rtg, seq_state, seq_act, seq_time)
                logits = actor_logits[:, -1, :].squeeze(0)
                
                if len(legal_indices) > 0:
                    mask = torch.full_like(logits, float('-inf'), dtype=torch.float32, device=self.device)
                    for idx in legal_indices:
                        if 0 <= idx < len(mask):
                            mask[idx] = logits[idx]
                    probs = F.softmax(mask, dim=-1)
                else:
                    probs = torch.ones_like(logits) / len(logits)
                
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    probs = torch.ones(len(legal_indices), device=self.device) / len(legal_indices)
            
            action_idx = torch.multinomial(probs, 1).item()
            mjx_action = next((a for a in legal_actions if hasattr(a, 'to_idx') and a.to_idx() == action_idx), legal_actions[0])
            
            act_histories[current_pid].append(action_idx)
            rtg_histories[current_pid].append(1.0)
            
            if current_pid == agent_pid:
                try:
                    dist = Categorical(probs=probs)
                    log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device))
                except:
                    log_prob = torch.tensor(0.0, device=self.device)
                
                step_reward = self.reward_calculator.calculate_potential_reward(obs)
                
                trajectories[agent_pid].append({
                    "obs": state_tensor,
                    "action": action_idx,
                    "log_prob": log_prob.item(),
                    "reward": step_reward, 
                    "timestep": step_counts[current_pid],
                    "mask": mask,
                    "obs_raw": obs  
                })
            
            step_counts[current_pid] += 1
            obs_dict = self.env.step({current_player_key: mjx_action})
            
            step_log_counter += 1
            if step_log_counter % 50 == 0:
                print(f"Match progress: {step_log_counter} steps simulated...")
        
        # 終局分數獲取
        try:
            final_rewards_raw = self.env.rewards()
            final_rewards = final_rewards_raw if isinstance(final_rewards_raw, dict) else {i: final_rewards_raw[i] for i in range(len(final_rewards_raw))}
        except:
            final_rewards = {i: 0 for i in range(4)}
        
        # 📌 更好地解決方法：從最後一步跳出時的 obs_dict 精準截取真正的終局觀測
        if agent_pid in trajectories and len(trajectories[agent_pid]) > 0:
            final_obs = None
            if agent_key in obs_dict:
                final_obs = obs_dict[agent_key]  # ✨ 成功攔截含 TSUMO/RON 事件的終局 Observation
            else:
                # 降級防禦線：若極端特例未回傳，才使用最後歷史步
                final_obs = trajectories[agent_pid][-1]["obs_raw"]
                
            final_score = final_rewards.get(agent_pid, 0)
            
            # 1. 檢查是否放銃 (r_penalty)
            is_houjuu = self.reward_calculator.check_houjuu(final_obs)
            if is_houjuu:
                opponent_score = max(final_rewards.values())
                penalty = self.reward_calculator.calculate_penalty_reward(final_obs, opponent_score)
                trajectories[agent_pid][-1]["reward"] += penalty
            
            # 2. 檢查是否胡牌型，實施密集分數回溯 (r_backward)
            final_hand_info = self.reward_calculator.compute_winning_hand_info(final_obs)
            if final_hand_info is not None and final_score > 0:
                for step in trajectories[agent_pid]:
                    current_hand_34 = self.reward_calculator._get_current_hand_34(step["obs_raw"])
                    r_back = self.reward_calculator.calculate_backward_reward(
                        final_hand_info, final_score, current_hand_34
                    )
                    step["reward"] += r_back
            elif final_score == 0:
                for step in trajectories[agent_pid]:
                    step["reward"] += -0.001
            
            # 清理原始物件引用，防止記憶體洩漏
            for step in trajectories[agent_pid]:
                if "obs_raw" in step: del step["obs_raw"]
                    
        return trajectories