import torch
import torch.nn.functional as F
from mjx.env import MjxEnv
from mjx.const import EventType
from torch.distributions import Categorical
import copy
import random
from typing import List, Dict, Tuple

# 導入本地獎勵計算模組
from utli.rewards import create_default_calculator

# 🚀 數值安全常數：用於替代 float('-inf')，避免 NaN 傳播
NEG_INF = -1e9


class SelfPlayRunner:
    def __init__(self, model, device: str = "cuda", opponent_pool_size: int = 5, train_mode: str = "attack", opponent_base_model=None, external_agents=None, reward_mode: str = "sparse"):
        """
        初始化自我博弈環境與對手池

        Args:
            model:                 DecisionMamba 模型（已載入權重）
            device:                計算設備（"cuda" 或 "cpu"）
            opponent_pool_size:    對手池上限（保留最近 N 代歷史模型用於對抗訓練）
            train_mode:            保留 attack / defense CLI 相容性，兩者皆使用 unified reward
            opponent_base_model:   對手基底模型（選填）。若提供，對手池以此模型為基底；
                                   若為 None，對手池以 self.model 的複本為基底（自我對弈）。
                                   用於 PPO vs BC 對比評估：agent 用 model，對手用 BC baseline。
            external_agents:       選填。dict[pid] = agent_object，用於注入外部 AI（如 MortalAgent）。
                                   agent 物件必須實作 act(obs) → mjx.Action 和 reset() 方法。
                                   當 player pid 有對應的外部 agent 時，使用外部 agent 而非 DecisionMamba 推論。
            reward_mode:           保留 sparse / dense CLI 相容性，兩者皆使用 unified reward
        """
        if train_mode not in ("attack", "defense"):
            raise ValueError(f"train_mode 必須是 'attack' 或 'defense'，收到: {train_mode}")
        if reward_mode not in ("sparse", "dense"):
            raise ValueError(f"reward_mode 必須是 'sparse' 或 'dense'，收到: {reward_mode}")
        self.model = model.to(device)
        self.device = device
        self.env = MjxEnv()
        self._opponent_base = opponent_base_model if opponent_base_model is not None else self.model
        self.opponent_pool = [copy.deepcopy(self._opponent_base).eval()]
        self.opponent_pool_size = opponent_pool_size
        self.train_mode = train_mode
        self.reward_mode = reward_mode
        self.reward_calculator = create_default_calculator()
        self.reward_calculator.set_mode(train_mode)
        self.external_agents = external_agents or {}

    def _add_reward_components(self, step: Dict, **components: float) -> None:
        reward_components = step["reward_components"]
        for name, value in components.items():
            reward_components[name] += float(value)
        step["reward"] = self.reward_calculator.combine_rewards(
            **reward_components
        )

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

        train_mode 與 reward_mode 僅保留參數相容性。所有 rollout 都使用
        unified shape + score delta + hand event + game end reward。

        🆕 每局牌（round）結束時獨立擷取 round_terminal，以 per-hand 語義
        累加 hand_results，供 MahjongMetricTracker 計算 Suphx 規範的和了率/放銃率

        Args:
            temperature: Logit 採樣溫度

        Returns:
            (trajectories, game_result): 軌跡字典與遊戲結果摘要
        """
        obs_dict = self.env.reset()

        for pid, ext_agent in self.external_agents.items():
            if hasattr(ext_agent, 'reset'):
                ext_agent.reset()

        agent_pid = random.choice([0, 1, 2, 3])

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
        last_agent_obs = None
        hand_start_step = 0

        hand_results: List[Dict] = []
        total_hands = 0
        anyone_agari = False

        while not self.env.done("game"):
            current_player_key = list(obs_dict.keys())[0]
            current_pid = int(current_player_key.split("_")[1])
            obs = obs_dict[current_player_key]

            legal_actions = obs.legal_actions()
            if len(legal_actions) == 0:
                break

            # Round terminal observations only carry a DUMMY action used to
            # advance the environment. They are not agent decision steps.
            if obs.to_proto().HasField("round_terminal"):
                obs_dict = self.env.step(
                    {current_player_key: legal_actions[0]}
                )
                continue

            state_tensor = self.extract_model_input(obs)

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
                mjx_action = agent_model.act(obs)
                action_idx = mjx_action.to_idx() if hasattr(mjx_action, "to_idx") else 0
                probs = torch.ones(len(legal_indices), device=self.device) / len(legal_indices)
                legal_mask_bool = torch.zeros(181, dtype=torch.bool, device=self.device)
                for idx in legal_indices:
                    if 0 <= idx < 181:
                        legal_mask_bool[idx] = True
            else:
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

            # ── 收集 agent 軌跡 ──
            if current_pid == agent_pid:
                if last_agent_obs is not None:
                    same_hand = self.reward_calculator.observations_in_same_hand(
                        last_agent_obs, obs
                    )
                    transition = self.reward_calculator.compute_transition_reward(
                        last_agent_obs, obs, same_hand
                    )
                    self._add_reward_components(
                        trajectories[agent_pid][-1],
                        shape=transition["shape"],
                        score=transition["score"],
                    )

                try:
                    dist = Categorical(probs=probs)
                    log_prob = dist.log_prob(
                        torch.tensor(action_idx, device=self.device)
                    )
                except Exception:
                    log_prob = torch.tensor(0.0, device=self.device)

                trajectories[agent_pid].append(
                    {
                        "obs": state_tensor.cpu(),
                        "action": action_idx,
                        "log_prob": log_prob.item(),
                        "reward": 0.0,
                        "reward_components": (
                            self.reward_calculator.empty_components()
                        ),
                        "timestep": step_counts[current_pid],
                        "mask": legal_mask_bool.cpu(),
                        "obs_raw": obs,
                    }
                )
                last_agent_obs = obs

            step_counts[current_pid] += 1
            obs_dict = self.env.step({current_player_key: mjx_action})

            # ── Per-round 邊界偵測 ──
            if self.env.done("round"):
                first_obs = next(iter(obs_dict.values()))
                obs_proto = first_obs.to_proto()

                if obs_proto.HasField("round_terminal"):
                    rt = obs_proto.round_terminal
                    is_draw = rt.HasField("no_winner")
                    wins_this_hand = [w.who for w in rt.wins] if len(rt.wins) > 0 else []
                    agent_won = agent_pid in wins_this_hand
                    event_info = (
                        self.reward_calculator.build_hand_end_event_info(
                            rt,
                            list(obs_proto.public_observation.events),
                            agent_pid,
                        )
                    )
                    agent_dealt_in = event_info["deal_in"]
                    agent_tenpai = event_info[
                        "exhaustive_draw_and_tenpai"
                    ]
                    exhaustive_draw = (
                        event_info["exhaustive_draw_and_tenpai"]
                        or event_info["exhaustive_draw_and_noten"]
                    )
                    events = list(obs_proto.public_observation.events)
                    agent_riichi = any(
                        event.type == int(EventType.RIICHI)
                        and event.who == agent_pid
                        for event in events
                    )
                    agent_called = any(
                        event.type
                        in (
                            int(EventType.CHI),
                            int(EventType.PON),
                            int(EventType.OPEN_KAN),
                        )
                        and event.who == agent_pid
                        for event in events
                    )

                    has_agent_step = len(trajectories[agent_pid]) > hand_start_step
                    if has_agent_step and last_agent_obs is not None:
                        final_score = int(rt.final_score.tens[agent_pid])
                        terminal_score_reward = (
                            self.reward_calculator.compute_score_delta_reward(
                                self.reward_calculator.extract_player_score(
                                    last_agent_obs, agent_pid
                                ),
                                final_score,
                            )
                        )
                        event_reward = (
                            self.reward_calculator.compute_hand_end_event_reward(
                                event_info
                            )
                        )
                        self._add_reward_components(
                            trajectories[agent_pid][-1],
                            score=terminal_score_reward,
                            event=event_reward,
                        )

                        if rt.is_game_over:
                            final_scores_now = list(rt.final_score.tens)
                            sorted_pids = sorted(
                                range(4),
                                key=lambda i: (final_scores_now[i], -i),
                                reverse=True,
                            )
                            final_rank = sorted_pids.index(agent_pid) + 1
                            game_reward = (
                                self.reward_calculator.compute_game_end_reward(
                                    final_rank, final_score
                                )
                            )
                            self._add_reward_components(
                                trajectories[agent_pid][-1],
                                game=game_reward,
                            )

                    hand_results.append({
                        "is_draw": is_draw,
                        "agent_won": agent_won,
                        "agent_deal_in": agent_dealt_in,
                        "agent_riichi": agent_riichi,
                        "agent_called": agent_called,
                        "agent_tenpai": agent_tenpai,
                        "tenpai_opportunity": exhaustive_draw,
                    })
                    total_hands += 1
                    anyone_agari = anyone_agari or bool(wins_this_hand)

                last_agent_obs = None
                hand_start_step = len(trajectories[agent_pid])

        # ── 終局結算 ──
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

        # 清理 obs_raw
        if agent_pid in trajectories:
            for step in trajectories[agent_pid]:
                if "obs_raw" in step:
                    del step["obs_raw"]

        final_scores = [real_tens[i] for i in range(4)]
        agent_score = real_tens[agent_pid]

        sorted_pids = sorted(
            range(4), key=lambda i: (real_tens[i], -i), reverse=True
        )
        agent_rank = sorted_pids.index(agent_pid) + 1

        is_agari = any(
            result.get("agent_won", False) for result in hand_results
        )
        is_houjuu = any(
            result.get("agent_deal_in", False) for result in hand_results
        )

        game_result = {
            "final_scores": final_scores,
            "agent_score": agent_score,
            "point_delta": agent_score - 25000,
            "agent_rank": agent_rank,
            "agent_pid": agent_pid,
            "is_win": (agent_rank == 1),
            "is_agari": is_agari,
            "is_houjuu": is_houjuu,
            "anyone_agari": anyone_agari,
            "hand_results": hand_results,
            "total_hands": total_hands,
        }

        return trajectories, game_result
