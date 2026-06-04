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
from utli.rewards import create_default_calculator

# 🚀 數值安全常數：用於替代 float('-inf')，避免 NaN 傳播
NEG_INF = -1e9


class SelfPlayRunner:
    def __init__(self, model, device: str = "cuda", opponent_pool_size: int = 5, train_mode: str = "attack", opponent_base_model=None, external_agents=None):
        """
        初始化自我博弈環境與對手池

        Args:
            model:                 DecisionMamba 模型（已載入權重）
            device:                計算設備（"cuda" 或 "cpu"）
            opponent_pool_size:    對手池上限（保留最近 N 代歷史模型用於對抗訓練）
            train_mode:            "attack" = 使用進攻 reward 訓練 / "defense" = 使用防守 reward 訓練
            opponent_base_model:   對手基底模型（選填）。若提供，對手池以此模型為基底；
                                   若為 None，對手池以 self.model 的複本為基底（自我對弈）。
                                   用於 PPO vs BC 對比評估：agent 用 model，對手用 BC baseline。
            external_agents:       選填。dict[pid] = agent_object，用於注入外部 AI（如 MortalAgent）。
                                   agent 物件必須實作 act(obs) → mjx.Action 和 reset() 方法。
                                   當 player pid 有對應的外部 agent 時，使用外部 agent 而非 DecisionMamba 推論。
        """
        if train_mode not in ("attack", "defense"):
            raise ValueError(f"train_mode 必須是 'attack' 或 'defense'，收到: {train_mode}")
        self.model = model.to(device)
        self.device = device
        self.env = MjxEnv()
        # 🆕 對手池基底：agent 永遠是 self.model，對手池可用外部模型（如 BC baseline）
        self._opponent_base = opponent_base_model if opponent_base_model is not None else self.model
        self.opponent_pool = [copy.deepcopy(self._opponent_base).eval()]
        self.opponent_pool_size = opponent_pool_size
        self.train_mode = train_mode
        self.reward_calculator = create_default_calculator()
        # 全程固定模式，不需要動態切換
        self.reward_calculator.set_mode(train_mode)
        # 🆕 外部 agent 對照表（如 MortalAgent）
        self.external_agents = external_agents or {}

    def update_opponent_pool(self):
        """
        更新對手池：將當前最新模型克隆後加入池中，
        超出容量上限時移除最舊的非當前模型。

        🆕 若設有 opponent_base_model（PPO vs BC 模式），
        對手池更新仍以 opponent_base_model 為基底（保持對手一致）。
        """
        new_opponent = copy.deepcopy(self._opponent_base).eval()
        self.opponent_pool.append(new_opponent)
        if len(self.opponent_pool) > self.opponent_pool_size:
            self.opponent_pool.pop(1)

    def extract_model_input(self, obs) -> torch.Tensor:
        """
        將 mjx 的 Observation 轉換為 Mamba 模型特徵張量（1380 維）。
        """
        features = obs.to_features("decision-mamba-v0")
        return torch.from_numpy(features).float().to(self.device)

    def run_match(self, temperature: float = 2.0) -> Tuple[Dict[int, List[Dict]], Dict]:
        """
        使用 MjxEnv 進行一局完整的遊戲對弈並收集軌跡。

        全程使用 train_mode 指定的 reward 權重（攻擊 or 防守），不切換。

        🆕 每局牌（round）結束時獨立擷取 round_terminal，以 per-hand 語義
        累加 hand_results，供 MahjongMetricTracker 計算 Suphx 規範的和了率/放銃率

        Args:
            temperature: Logit 採樣溫度

        Returns:
            (trajectories, game_result): 軌跡字典與遊戲結果摘要
                game_result 新增 "hand_results" (List[Dict]) 和 "total_hands" (int)
        """
        obs_dict = self.env.reset()

        # 🆕 重置外部 agent 狀態（如 MortalAgent）
        for pid, ext_agent in self.external_agents.items():
            if hasattr(ext_agent, 'reset'):
                ext_agent.reset()

        agent_pid = random.choice([0, 1, 2, 3])
        agent_key = f"player_{agent_pid}"

        # 🆕 決定每位玩家使用 DecisionMamba 還是外部 agent
        # external_agents 優先：若外部 agent 存在則使用它，否則用 DecisionMamba
        # agent_pid（我們訓練的模型）永遠使用 self.model（PPO agent）
        assigned_models = {}
        for pid in range(4):
            if pid == agent_pid:
                assigned_models[pid] = ("mamba", self.model)
            elif pid in self.external_agents:
                assigned_models[pid] = ("external", self.external_agents[pid])
            else:
                assigned_models[pid] = ("mamba", random.choice(self.opponent_pool))

        trajectories = {agent_pid: []}
        obs_histories = {pid: [] for pid in range(4)}
        timesteps_histories = {pid: [] for pid in range(4)}

        act_histories = {pid: [] for pid in range(4)}
        rtg_histories = {pid: [] for pid in range(4)}
        step_counts = {pid: 0 for pid in range(4)}

        MAX_CONTEXT_LEN = 128
        step_log_counter = 0
        prev_shanten = None
        agent_has_won = False

        # ── 🆕 Per-hand（每局牌）統計追蹤 ──
        # 每個 round 獨立計數，用於 Suphx 論文規範的 per-hand 和了率/放銃率
        # 分母 = 總局數（total_hands），而非總半莊數（total_games）
        hand_results: List[Dict] = []
        total_hands = 0

        while not self.env.done("game"):
            current_player_key = list(obs_dict.keys())[0]
            current_pid = int(current_player_key.split("_")[1])
            obs = obs_dict[current_player_key]

            legal_actions = obs.legal_actions()
            if len(legal_actions) == 0:
                break

            # 提取 1380 維狀態（不做任何拼接）
            state_tensor = self.extract_model_input(obs)  # (1380,)

            # ── 時序對齊 ──
            max_act_len = MAX_CONTEXT_LEN - 1
            current_action_context = [180] + act_histories[current_pid][-max_act_len:]
            current_rtg_context = [1.0] + rtg_histories[current_pid][-max_act_len:]

            hist_len = min(len(obs_histories[current_pid]) + 1, MAX_CONTEXT_LEN)

            obs_context = (
                obs_histories[current_pid][-(hist_len - 1):]
                if hist_len > 1
                else []
            )
            obs_context = obs_context + [state_tensor]
            seq_state = torch.stack(obs_context).unsqueeze(0).to(self.device)

            timestep_context = (
                timesteps_histories[current_pid][-(hist_len - 1):]
                if hist_len > 1
                else []
            )
            timestep_context = timestep_context + [step_counts[current_pid]]
            seq_time = torch.tensor(
                timestep_context, dtype=torch.long, device=self.device
            ).unsqueeze(0)

            seq_act = torch.tensor(
                current_action_context[-hist_len:],
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            seq_rtg = (
                torch.tensor(
                    current_rtg_context[-hist_len:],
                    dtype=torch.float32,
                    device=self.device,
                )
                .unsqueeze(0)
                .unsqueeze(-1)
            )

            legal_indices = [
                act.to_idx() if hasattr(act, "to_idx") else int(act)
                for act in legal_actions
            ]

            # ── 模型推論 ──
            agent_type, agent_model = assigned_models[current_pid]

            if agent_type == "external":
                # 🆕 使用外部 agent（如 MortalAgent）進行決策
                mjx_action = agent_model.act(obs)
                action_idx = mjx_action.to_idx() if hasattr(mjx_action, "to_idx") else 0

                # 簡化：外部 agent 的 log_prob / probs 設為均勻分佈
                # 因為我們不需要對 extern agent 收集訓練軌跡
                probs = torch.ones(len(legal_indices), device=self.device) / len(legal_indices)
                legal_mask_bool = torch.zeros(181, dtype=torch.bool, device=self.device)
                for idx in legal_indices:
                    if 0 <= idx < 181:
                        legal_mask_bool[idx] = True

            else:
                # 🧠 使用 DecisionMamba 模型推論（原有邏輯不變）
                with torch.no_grad():
                    actor_logits, _, _, _ = agent_model(
                        seq_rtg, seq_state, seq_act, seq_time
                    )
                    logits = actor_logits[:, -1, :].squeeze(0)

                    legal_mask_bool = torch.zeros_like(
                        logits, dtype=torch.bool, device=self.device
                    )

                    if len(legal_indices) > 0:
                        mask = torch.full_like(
                            logits, NEG_INF, dtype=torch.float32, device=self.device
                        )
                        for idx in legal_indices:
                            if 0 <= idx < len(mask):
                                mask[idx] = logits[idx]
                                legal_mask_bool[idx] = True

                        # 溫度 ≤ 0 → 貪婪決策 (argmax)；溫度 > 0 → 隨機抽樣
                        if temperature <= 0:
                            action_idx = mask.argmax().item()
                            probs = torch.zeros_like(mask)
                            probs[action_idx] = 1.0
                        else:
                            probs = F.softmax(mask / temperature, dim=-1)
                            if torch.isnan(probs).any() or torch.isinf(probs).any():
                                probs = (
                                    torch.ones(len(legal_indices), device=self.device)
                                    / len(legal_indices)
                                )
                            action_idx = torch.multinomial(probs, 1).item()
                    else:
                        probs = torch.ones_like(logits) / len(logits)
                        action_idx = 0

                mjx_action = None
                for a in legal_actions:
                    if hasattr(a, "to_idx") and a.to_idx() == action_idx:
                        mjx_action = a
                        break
                if mjx_action is None:
                    print(
                        f"⚠️ 警告：action_idx {action_idx} 無法對應到任何合法動作，fallback 至 legal_actions[0]"
                    )
                    mjx_action = legal_actions[0]

            obs_histories[current_pid].append(state_tensor)
            timesteps_histories[current_pid].append(step_counts[current_pid])
            act_histories[current_pid].append(action_idx)
            rtg_histories[current_pid].append(1.0)

            if mjx_action.type() == ActionType.TSUMO and current_pid == agent_pid:
                agent_has_won = True
            elif mjx_action.type() == ActionType.RON and current_pid == agent_pid:
                agent_has_won = True

            # ── 收集 agent 軌跡 ──
            if current_pid == agent_pid:
                try:
                    dist = Categorical(probs=probs)
                    log_prob = dist.log_prob(
                        torch.tensor(action_idx, device=self.device)
                    )
                except:
                    log_prob = torch.tensor(0.0, device=self.device)

                step_reward = self.reward_calculator.calculate_potential_reward(obs)
                r_dora = self.reward_calculator.calculate_dora_potential_reward(obs)
                step_reward += r_dora

                curr_shanten = obs.curr_hand().shanten_number()
                r_prog = self.reward_calculator.calculate_progression_reward(
                    prev_shanten, curr_shanten
                )
                step_reward += r_prog
                prev_shanten = curr_shanten

                trajectories[agent_pid].append(
                    {
                        "obs": state_tensor.cpu(),
                        "action": action_idx,
                        "log_prob": log_prob.item(),
                        "reward": step_reward,
                        "timestep": step_counts[current_pid],
                        "mask": legal_mask_bool.cpu(),
                        "obs_raw": obs,
                    }
                )

            step_counts[current_pid] += 1
            obs_dict = self.env.step({current_player_key: mjx_action})

            # ── 🆕 Per-round 邊界偵測 ──
            # 當 mjx 環境回報 done("round") 時，表示一局牌（hand）已結束。
            # 從當前 observation 的 round_terminal proto 提取本局胡牌/放銃/流局狀態，
            # 獨立打包為 hand_result，供 MahjongMetricTracker 以 per-hand 語義統計。
            if self.env.done("round"):
                # 擷取任一玩家的 observation 以讀取 round_terminal（四人共享同一份終局資訊）
                first_obs = next(iter(obs_dict.values()))
                obs_proto = first_obs.to_proto()

                if obs_proto.HasField("round_terminal"):
                    rt = obs_proto.round_terminal

                    # 流局判定：優先使用 proto 原生 no_winner 欄位
                    # RoundTerminal.no_winner 非 None 即為流局（含荒牌流局、九種九牌等）
                    is_draw = rt.HasField("no_winner")

                    # 胡牌者清單（Win.who）
                    wins_this_hand = [w.who for w in rt.wins] if len(rt.wins) > 0 else []

                    # agent 本局是否胡牌
                    agent_won = agent_pid in wins_this_hand

                    # agent 本局是否放銃
                    # 從 agent 視角的 observation events 中回溯 RON 事件來源
                    agent_obs = obs_dict.get(agent_key)
                    agent_dealt_in = False
                    if agent_obs is not None:
                        agent_dealt_in = self.reward_calculator.check_houjuu(agent_obs)

                    hand_results.append({
                        "is_draw": is_draw,
                        "agent_won": agent_won,
                        "agent_deal_in": agent_dealt_in,
                    })
                    total_hands += 1

                # 新局開始：重置向聽追蹤（避免跨局殘留 prev_shanten）
                prev_shanten = None



        # ── 終局結算 ──
        try:
            final_rewards = self.env.rewards()
        except:
            final_rewards = {f"player_{i}": 0 for i in range(4)}

        try:
            final_state_proto = self.env.state().to_proto()
            if (
                final_state_proto.HasField("round_terminal")
                and final_state_proto.round_terminal.HasField("final_score")
            ):
                real_tens = list(
                    final_state_proto.round_terminal.final_score.tens
                )
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

            agent_score_delta = real_tens[agent_pid] - 25000

            is_houjuu = self.reward_calculator.check_houjuu(final_obs)
            if is_houjuu:
                best_opponent_delta = (
                    max(real_tens[i] for i in range(4) if i != agent_pid) - 25000
                )
                penalty = self.reward_calculator.calculate_penalty_reward(
                    final_obs, best_opponent_delta
                )
                trajectories[agent_pid][-1]["reward"] += penalty

            final_hand_info = self.reward_calculator.compute_winning_hand_info(
                final_obs
            )
            if final_hand_info is not None and agent_score_delta > 0:
                for step in trajectories[agent_pid]:
                    current_hand_34 = self.reward_calculator._get_current_hand_34(
                        step["obs_raw"]
                    )
                    r_back = self.reward_calculator.calculate_backward_reward(
                        final_hand_info, agent_score_delta, current_hand_34
                    )
                    step["reward"] += r_back
            elif agent_score_delta <= 0:
                for step in trajectories[agent_pid]:
                    step["reward"] += -0.001

            for step in trajectories[agent_pid]:
                if "obs_raw" in step:
                    del step["obs_raw"]

        final_scores = [real_tens[i] for i in range(4)]
        agent_score = real_tens[agent_pid]

        sorted_pids = sorted(
            range(4), key=lambda i: (real_tens[i], -i), reverse=True
        )
        agent_rank = sorted_pids.index(agent_pid) + 1

        # ── 終局胡牌清單提取 ──
        wins_pids = []
        if (
            final_state_proto is not None
            and final_state_proto.HasField("round_terminal")
            and len(final_state_proto.round_terminal.wins) > 0
        ):
            wins_pids = [w.who for w in final_state_proto.round_terminal.wins]

        # 🚀 互斥鎖定修正：當 final_state_proto 為 None（proto 解析失敗）時，
        # 若 agent_has_won 已為 True，強制推斷 wins_pids 包含 agent_pid，
        # 杜絕 wins += 1 與 draw_games += 1 同時觸發的統計矛盾。
        if final_state_proto is None and agent_has_won:
            wins_pids = [agent_pid]

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
            # 🆕 Per-hand 統計（用於 Suphx 論文規範的 per-hand 和了率/放銃率）
            "hand_results": hand_results,
            "total_hands": total_hands,
        }

        return trajectories, game_result