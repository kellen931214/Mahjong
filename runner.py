import torch
import torch.nn.functional as F
from mjx.env import MjxEnv
from mjx.const import ActionType
import numpy as np
from torch.distributions import Categorical
import copy
import random
from typing import List, Dict, Tuple

# 導入本地獎勵計算模組
from rewards import create_default_calculator

# 🚀 數值安全常數：用於替代 float('-inf')，避免 NaN 傳播
NEG_INF = -1e9

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
        將 mjx 的 Observation 轉換為 Mamba 模型特徵張量（1380 維）
        """
        features = obs.to_features("decision-mamba-v0")  
        return torch.from_numpy(features).float().to(self.device)

    def run_match(self, temperature: float = 2.0) -> Tuple[Dict[int, List[Dict]], Dict]:    
        """
        使用 MjxEnv 進行一局完整的遊戲對弈並收集軌跡（時序與 PPO 完全對齊版）
        
        Args:
            temperature: Logit 採樣溫度（Softmax 前除以係數，拉平極端分佈）
        
        Returns:
            (trajectories, game_result): 軌跡字典與遊戲結果摘要
                game_result 包含: final_scores (4人分數), agent_rank, is_win, is_agari
        """
        obs_dict = self.env.reset()
        agent_pid = random.choice([0, 1, 2, 3])
        agent_key = f"player_{agent_pid}" 
        
        assigned_models = {
            agent_pid: self.model,
            **{pid: random.choice(self.opponent_pool) for pid in [0, 1, 2, 3] if pid != agent_pid}
        }
        
        trajectories = {agent_pid: []}
        obs_histories = {pid: [] for pid in range(4)}
        timesteps_histories = {pid: [] for pid in range(4)}
        
        # 🚀【核心修正】移除初始化的 [180]，使歷史增長步伐與狀態完全一致，留給前向傳播動態處理
        act_histories = {pid: [] for pid in range(4)} 
        rtg_histories = {pid: [] for pid in range(4)}
        step_counts = {pid: 0 for pid in range(4)}
        
        MAX_CONTEXT_LEN = 128
        step_log_counter = 0
        prev_shanten = None  # 🆕 追蹤前一步向聽數（用於 progression reward）
        agent_has_won = False  # 🚀 追蹤 agent 是否本局曾胡牌
        
        while not self.env.done():
            current_player_key = list(obs_dict.keys())[0]  
            current_pid = int(current_player_key.split('_')[1])
            obs = obs_dict[current_player_key]
            
            legal_actions = obs.legal_actions()
            if len(legal_actions) == 0:
                break
            
            state_tensor = self.extract_model_input(obs)
            
            # 🚀【時序對齊】Decision Mamba 語義：action[i] = 第 i-1 步的動作（首位補 [180]）
            # 因此 act/rtg context 永遠需要比 obs context 多一個前置 token
            # 預留 1 個 slot：act_context 總長 = 1 + last (MAX_CONTEXT_LEN-1) actions
            max_act_len = MAX_CONTEXT_LEN - 1
            current_action_context = [180] + act_histories[current_pid][-max_act_len:]
            current_rtg_context = [1.0] + rtg_histories[current_pid][-max_act_len:]

            # 🚀 當前步的 obs 尚未壓入歷史，先用現有的 obs 歷史計算 hist_len
            # obs/act/rtg/timestep 四者的歷史長度在此時全部一致（皆為 N-1，首步時 obs 為 0）
            hist_len = min(len(obs_histories[current_pid]) + 1, MAX_CONTEXT_LEN)

            # 🚀【嚴格截取】先取現有 obs 歷史再補上當前 state_tensor，確保 dim=1 長度 = hist_len
            # obs 歷史：取 [-hist_len+1:]（即現有歷史的尾部），再 append 當前 state
            obs_context = obs_histories[current_pid][-(hist_len - 1):] if hist_len > 1 else []
            obs_context = obs_context + [state_tensor]
            seq_state = torch.stack(obs_context).unsqueeze(0).to(self.device)

            timestep_context = timesteps_histories[current_pid][-(hist_len - 1):] if hist_len > 1 else []
            timestep_context = timestep_context + [step_counts[current_pid]]
            seq_time = torch.tensor(timestep_context, dtype=torch.long, device=self.device).unsqueeze(0)

            seq_act = torch.tensor(current_action_context[-hist_len:], dtype=torch.long, device=self.device).unsqueeze(0)
            seq_rtg = torch.tensor(current_rtg_context[-hist_len:], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)
            
            legal_indices = [act.to_idx() if hasattr(act, 'to_idx') else int(act) for act in legal_actions]
            
            with torch.no_grad():
                actor_logits, _, _, _ = assigned_models[current_pid](seq_rtg, seq_state, seq_act, seq_time)
                logits = actor_logits[:, -1, :].squeeze(0)
                
                legal_mask_bool = torch.zeros_like(logits, dtype=torch.bool, device=self.device)
                
                if len(legal_indices) > 0:
                    mask = torch.full_like(logits, NEG_INF, dtype=torch.float32, device=self.device)
                    for idx in legal_indices:
                        if 0 <= idx < len(mask):
                            mask[idx] = logits[idx]
                            legal_mask_bool[idx] = True
                    # 🆕 套用 temperature 拉平極端分佈再 softmax
                    probs = F.softmax(mask / temperature, dim=-1)
                else:
                    probs = torch.ones_like(logits) / len(logits)
                
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    probs = torch.ones(len(legal_indices), device=self.device) / len(legal_indices)
            
            action_idx = torch.multinomial(probs, 1).item()
            mjx_action = None
            for a in legal_actions:
                if hasattr(a, 'to_idx') and a.to_idx() == action_idx:
                    mjx_action = a
                    break
            if mjx_action is None:
                print(f"⚠️ 警告：action_idx {action_idx} 無法對應到任何合法動作，fallback 至 legal_actions[0]")
                mjx_action = legal_actions[0]
            
            # 🚀 動作做出後，才正式登記進歷史清單，供下一手使用
            # 所有歷史（obs/act/rtg/timestep）同步增長，確保維度永遠一致
            obs_histories[current_pid].append(state_tensor)
            timesteps_histories[current_pid].append(step_counts[current_pid])
            act_histories[current_pid].append(action_idx)
            rtg_histories[current_pid].append(1.0)
            
            # 🚀 追蹤 agent 是否本局曾胡牌（不依賴 round_terminal.wins）
            if mjx_action.type() == ActionType.TSUMO and current_pid == agent_pid:
                agent_has_won = True
            elif mjx_action.type() == ActionType.RON and current_pid == agent_pid:
                agent_has_won = True
            
            if current_pid == agent_pid:
                try:
                    dist = Categorical(probs=probs)
                    log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device))
                except:
                    log_prob = torch.tensor(0.0, device=self.device)
                
                step_reward = self.reward_calculator.calculate_potential_reward(obs)
                
                # 🆕 方案四：寶牌潛力獎勵（即時鼓勵保留寶牌追求大牌）
                r_dora = self.reward_calculator.calculate_dora_potential_reward(obs)
                step_reward += r_dora
                
                # 🆕 計算進展獎勵（比較前後向聽數的變化）
                curr_shanten = obs.curr_hand().shanten_number()
                r_prog = self.reward_calculator.calculate_progression_reward(prev_shanten, curr_shanten)
                step_reward += r_prog
                prev_shanten = curr_shanten
                
                # 🚀【接口與記憶體優化】直接回傳 .cpu() 資料以節省 Rollout 顯存，並將 Key 命名為 "mask" 傳出布林值
                trajectories[agent_pid].append({
                    "obs": state_tensor.cpu(),
                    "action": action_idx,
                    "log_prob": log_prob.item(),
                    "reward": step_reward, 
                    "timestep": step_counts[current_pid],
                    "mask": legal_mask_bool.cpu(),   # 🚀 完美咬合 PPO 主程式的 step_data["mask"].bool()
                    "obs_raw": obs  
                })
            
            step_counts[current_pid] += 1
            obs_dict = self.env.step({current_player_key: mjx_action})
            
            step_log_counter += 1
            if step_log_counter % 50 == 0:
                print(f"Match progress: {step_log_counter} steps simulated...")
        
        # 終局分數獲取
        # 🔧 env.rewards() 回傳 Dict[str, int]，key 為 "player_0".."player_3"
        try:
            final_rewards = self.env.rewards()
        except:
            final_rewards = {f"player_{i}": 0 for i in range(4)}
        
        # ========================
        # 🚀【修正】從 env.state() proto 取得真實終局分數（含胡牌贏的點數）
        #    obs.tens() 只記錄初始分-供託，不含胡牌贏的點數
        #    round_terminal.final_score.tens 才是真正的終局分
        # ========================
        try:
            final_state_proto = self.env.state().to_proto()
            if (final_state_proto.HasField("round_terminal") and
                final_state_proto.round_terminal.HasField("final_score")):
                real_tens = list(final_state_proto.round_terminal.final_score.tens)
            else:
                real_tens = [25000, 25000, 25000, 25000]
        except Exception:
            real_tens = [25000, 25000, 25000, 25000]
            final_state_proto = None
        
        if agent_pid in trajectories and len(trajectories[agent_pid]) > 0:
            final_obs = None
            if agent_key in obs_dict:
                final_obs = obs_dict[agent_key]  
            else:
                final_obs = trajectories[agent_pid][-1]["obs_raw"]
            
            # 🚀【修正】使用真實終局點數作為 r_backward 的基礎分
            #    大牌範例：real_tens=48000 → delta=23000
            #    小牌範例：real_tens=26000 → delta=1000
            agent_score_delta = real_tens[agent_pid] - 25000
            
            # 1. 檢查是否放銃 (r_penalty)
            #    🚀【修正】opponent_score 改用實際和牌方的終局點數差
            is_houjuu = self.reward_calculator.check_houjuu(final_obs)
            if is_houjuu:
                # 找出贏最多分的人（即和牌者）的點數差
                best_opponent_delta = max(real_tens[i] for i in range(4) if i != agent_pid) - 25000
                penalty = self.reward_calculator.calculate_penalty_reward(final_obs, best_opponent_delta)
                trajectories[agent_pid][-1]["reward"] += penalty
            
            # 2. 檢查是否胡牌型，實施密集分數回溯 (r_backward)
            #    🚀【修正】使用真實點數差 agent_score_delta 作為 r_backward 的基礎分
            final_hand_info = self.reward_calculator.compute_winning_hand_info(final_obs)
            if final_hand_info is not None and agent_score_delta > 0:
                for step in trajectories[agent_pid]:
                    current_hand_34 = self.reward_calculator._get_current_hand_34(step["obs_raw"])
                    r_back = self.reward_calculator.calculate_backward_reward(
                        final_hand_info, agent_score_delta, current_hand_34
                    )
                    step["reward"] += r_back
            elif agent_score_delta <= 0:
                for step in trajectories[agent_pid]:
                    step["reward"] += -0.001
            
            # 清理原始物件引用，防止記憶體洩漏
            for step in trajectories[agent_pid]:
                if "obs_raw" in step: del step["obs_raw"]
        
        final_scores = [real_tens[i] for i in range(4)]
        agent_score = real_tens[agent_pid]
        
        # 排名：分數越高排名越前（1 = 第一名），同分時以較小 pid 為 tie-breaker
        sorted_pids = sorted(range(4), key=lambda i: (real_tens[i], -i), reverse=True)
        agent_rank = sorted_pids.index(agent_pid) + 1  # 1-indexed
        
        # 🚀【修正】is_agari = agent 本局曾胡牌
        #    mjx 在 round over 時 legal_actions 只回傳 DUMMY (action=99)，
        #    TSUMO(5)/RON(10) 不會出現在 action-level，因此必須從 proto wins 讀取。
        wins_pids = []
        if (final_state_proto is not None and
            final_state_proto.HasField("round_terminal") and
            len(final_state_proto.round_terminal.wins) > 0):
            wins_pids = [w.who for w in final_state_proto.round_terminal.wins]
        is_agari = agent_has_won or (agent_pid in wins_pids)
        
        game_result = {
            "final_scores": final_scores,
            "agent_score": agent_score,
            "agent_rank": agent_rank,
            "agent_pid": agent_pid,
            "is_win": (agent_rank == 1),
            "is_agari": is_agari,
            "is_houjuu": is_houjuu,
            "anyone_agari": (len(wins_pids) > 0),
        }
        
        return trajectories, game_result