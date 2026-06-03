"""
evaluation_metrics.py — 離線準確率計算 + 線上對局統計追蹤

基於 Suphx 論文評估指標體系：
  1. 離線模型準確率（Offline Action Accuracy）：按動作類別分別計算
  2. 線上對局統計器（Online Tracker）：以 per-hand（每局牌）語義追蹤和了率、放銃率、順位分佈等

🆕 v2 重大變更（符合 Suphx 學術標準）：
  - 和了率 / 放銃率分母從 total_games（半莊數）改為 total_hands（每局牌數）
  - 流局判定優先使用 proto round_terminal.no_winner 原生欄位
  - Fallback 分數閾值從 ±1000 放寬到 ±4000（容納不聽罰符＋立直棒）
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


# ============================================================================
#  動作 ID 分區常數（參照 action_converter.py 的編碼表）
# ============================================================================

ACTION_BINS = {
    "dahai":     (0, 73),     # DISCARD 0~36 + TSUMOGIRI 37~73
    "chow":      (74, 103),   # CHI
    "pong":      (104, 140),  # PON
    "kong":      (141, 174),  # CLOSED_KAN / OPEN_KAN / ADDED_KAN
    "riichi":    (177, 177),  # RIICHI
    "tsumo":     (175, 175),  # TSUMO
    "ron":       (176, 176),  # RON
    "nine_term": (178, 178),  # ABORTIVE_DRAW_NINE_TERMINALS
    "no":        (179, 179),  # PASS
    "dummy":     (180, 180),  # DUMMY
}

# 評估時關注的五個主要類別（對接 evaluate.md 規格）
EVAL_CATEGORIES = ["dahai", "chow", "pong", "kong", "riichi"]


# ============================================================================
#  1. 離線模型準確率 (Offline Action Accuracy)
# ============================================================================

def compute_offline_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    計算模型預測 logits 與真實行為標籤的類別準確率。

    Args:
        logits: 模型輸出 (batch_size, 181)，原始 logits（未經 softmax）
        targets: 真實標籤 (batch_size,)，每個元素為 0~180 的動作 ID
        mask: 合法動作遮罩 (batch_size, 181)，True=該動作合法。
              若為 None，則所有動作視為合法。

    Returns:
        dict: {
            "dahai_accuracy":  ...,
            "chow_accuracy":   ...,
            "pong_accuracy":   ...,
            "kong_accuracy":   ...,
            "riichi_accuracy": ...,
            "overall_accuracy": ...,
        }
    """
    if logits.dim() != 2 or logits.size(1) != 181:
        raise ValueError(f"logits 形狀必須為 (batch, 181)，實際: {logits.shape}")
    if targets.dim() != 1:
        raise ValueError(f"targets 形狀必須為 (batch,)，實際: {targets.shape}")
    if logits.size(0) != targets.size(0):
        raise ValueError(f"logits 與 targets 的 batch 大小不一致: {logits.size(0)} vs {targets.size(0)}")

    batch_size = logits.size(0)
    device = logits.device

    # 合法動作遮罩處理
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(f"mask 形狀必須與 logits 相同 (batch, 181)，實際: {mask.shape}")
        # 確保 mask 是 bool 型別
        legal_bool = mask.bool()
    else:
        legal_bool = torch.ones_like(logits, dtype=torch.bool, device=device)

    # 對 logits 應用遮罩：非法動作設為 -inf，再做 argmax
    masked_logits = logits.clone()
    masked_logits[~legal_bool] = float("-inf")
    predictions = masked_logits.argmax(dim=-1)  # (batch,)

    # 計算總體準確率（只算至少有一個合法動作可選的樣本）
    has_any_legal = legal_bool.any(dim=-1)  # (batch,)
    correct = (predictions == targets) & has_any_legal
    overall_acc = correct.float().sum().item() / max(has_any_legal.sum().item(), 1)

    # 按類別計算
    results = {"overall_accuracy": overall_acc}

    for cat_name in EVAL_CATEGORIES:
        lo, hi = ACTION_BINS[cat_name]
        # 找出目標落在此類別範圍內的樣本
        in_category = (targets >= lo) & (targets <= hi)  # (batch,)
        # 且該類別中至少存在一個合法動作
        category_legal = legal_bool[:, lo:hi+1].any(dim=-1)  # (batch,)
        valid = in_category & category_legal

        if valid.sum().item() == 0:
            results[f"{cat_name}_accuracy"] = float("nan")
        else:
            cat_correct = (predictions == targets) & valid
            results[f"{cat_name}_accuracy"] = cat_correct[valid].float().mean().item()

    return results


def compute_detailed_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    top_k: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """
    計算更詳細的準確率，含 Top-K 指標。

    Args:
        logits: (batch, 181)
        targets: (batch,)
        mask: (batch, 181) bool，合法動作遮罩
        top_k: 要計算的 Top-K 值

    Returns:
        dict: 含各類別的 Top-1/Top-3/Top-5 準確率
    """
    if mask is not None:
        legal_bool = mask.bool()
    else:
        legal_bool = torch.ones_like(logits, dtype=torch.bool)

    masked_logits = logits.clone()
    masked_logits[~legal_bool] = float("-inf")

    has_any_legal = legal_bool.any(dim=-1)
    batch_size = logits.size(0)

    results = {}

    for k in top_k:
        # Top-K 預測
        _, topk_indices = masked_logits.topk(k, dim=-1)  # (batch, k)

        # 總體 Top-K
        match = (topk_indices == targets.unsqueeze(-1)).any(dim=-1) & has_any_legal
        results[f"top{k}_accuracy"] = match.float().sum().item() / max(has_any_legal.sum().item(), 1)

        # 各類別 Top-K
        for cat_name in EVAL_CATEGORIES:
            lo, hi = ACTION_BINS[cat_name]
            in_category = (targets >= lo) & (targets <= hi)
            category_legal = legal_bool[:, lo:hi+1].any(dim=-1)
            valid = in_category & category_legal
            if valid.sum().item() == 0:
                results[f"{cat_name}_top{k}"] = float("nan")
            else:
                cmatch = (topk_indices == targets.unsqueeze(-1)).any(dim=-1) & valid
                results[f"{cat_name}_top{k}"] = cmatch[valid].float().mean().item()

    return results


# ============================================================================
#  1.5 流式準確率追蹤器 (O(1) 記憶體，避免 OOM)
# ============================================================================

class StreamingAccuracyTracker:
    """
    Per-batch 流式累加計數器：不保留任何大張量，只存純量 correct/total。

    適用於超大驗證集（數千萬樣本）場景，記憶體複雜度 O(1)，避免 torch.cat()
    時因同時持有分散碎片 + 連續分配空間而觸發 OOM Killer。

    使用方式:
        tracker = StreamingAccuracyTracker()
        for batch in val_loader:
            logits = model(state, action, ...)
            targets = batch["target_action"]
            # 過濾 padding / NO / DUMMY 後
            tracker.update(logits, targets)
        results = tracker.compute()  # 回傳與 compute_offline_accuracy 相同格式
    """

    def __init__(self, categories: Optional[List[str]] = None):
        """
        Args:
            categories: 要追蹤的類別列表。預設為 EVAL_CATEGORIES。
        """
        cats = categories or list(EVAL_CATEGORIES)
        self.categories = cats
        self.reset()

    def reset(self):
        """重置所有計數器。"""
        self._counters = {}
        for cat in self.categories:
            self._counters[cat] = {"correct": 0, "total": 0}
        self._counters["overall"] = {"correct": 0, "total": 0}
        self._total_samples = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        對單一 batch 計算預測結果，並按類別累加 correct / total。

        Args:
            logits: (B, 181) 模型原始 logits
            targets: (B,) 真實標籤，0~180，不得含 padding (-100) / NO / DUMMY
            mask: (B, 181) bool 合法動作遮罩。若 None 則全部合法。
        """
        B = logits.size(0)
        if B == 0:
            return

        device = logits.device

        # 合法動作遮罩
        if mask is not None and mask.shape == logits.shape:
            legal_bool = mask.bool()
        else:
            legal_bool = torch.ones_like(logits, dtype=torch.bool, device=device)

        # argmax 預測
        masked_logits = logits.clone()
        masked_logits[~legal_bool] = float("-inf")
        predictions = masked_logits.argmax(dim=-1)  # (B,)

        has_legal = legal_bool.any(dim=-1)
        correct = (predictions == targets) & has_legal

        # 總體
        self._counters["overall"]["correct"] += correct.sum().item()
        self._counters["overall"]["total"] += has_legal.sum().item()
        self._total_samples += B

        # 各類別
        for cat in self.categories:
            lo, hi = ACTION_BINS[cat]
            in_cat = (targets >= lo) & (targets <= hi)
            cat_legal = legal_bool[:, lo:hi + 1].any(dim=-1)
            valid = in_cat & cat_legal

            if valid.sum().item() > 0:
                cat_correct = (predictions == targets) & valid
                self._counters[cat]["correct"] += cat_correct.sum().item()
                self._counters[cat]["total"] += valid.sum().item()

    def compute(self) -> Dict[str, float]:
        """
        從累加的計數器計算最終準確率。

        Returns:
            dict: 與 compute_offline_accuracy 相同格式的準確率字典
        """
        results = {}
        for cat in ["overall"] + self.categories:
            c = self._counters[cat]["correct"]
            t = self._counters[cat]["total"]
            if t == 0:
                results[f"{cat}_accuracy" if cat != "overall" else "overall_accuracy"] = float("nan")
            else:
                results[f"{cat}_accuracy" if cat != "overall" else "overall_accuracy"] = c / t
        return results

    def summary(self) -> Dict:
        """回傳完整計數摘要（含原始 correct/total 數值）。"""
        return {
            cat: dict(self._counters[cat])
            for cat in ["overall"] + self.categories
        }


# ============================================================================
#  2. 線上對局統計器 (Online Tracker) — 以 per-hand 語義計算學術指標
# ============================================================================

class MahjongMetricTracker:
    """
    追蹤模型在數千場自主對弈（Self-Play）中的核心指標。

    🆕 v2：以 per-hand（每局牌）語義計算和了率/放銃率。
    分母 = total_hands（總局數），而非 total_games（總半莊數）。

    符合 Suphx 論文規範：
      - 和了率 (Win Rate)：agent 胡牌局數 / 總局數
      - 放銃率 (Deal-in Rate)：agent 放銃局數 / 總局數
      - 順位分佈 (Placement Distribution)：1/2/3/4 名的精確百分比

    使用方式:
        tracker = MahjongMetricTracker()
        for _ in range(num_games):
            _, game_result = runner.run_match()
            tracker.record_game(game_result)
        tracker.report()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有統計計數器。"""
        # ── Game-level 計數器（半莊維度） ──
        self.total_games = 0
        self.placements = {1: 0, 2: 0, 3: 0, 4: 0}  # 排名計數
        self.cumulative_scores: List[float] = []  # 累積分數用於計算均分

        # 🆕 Hand-level 計數器（Per-hand 和了率/放銃率分母）
        self.total_hands = 0
        self.wins = 0          # agent 胡牌局數（per-hand）
        self.deal_ins = 0      # agent 放銃局數（per-hand）
        self.draw_hands = 0    # 流局局數

    def record_hand(self, hand_result: Dict):
        """
        記錄一局牌（hand）的結果，以 per-hand 語義累加。

        Args:
            hand_result: dict，由 runner.py 在每局結束時產生
                預期包含以下 key：
                - "is_draw" (bool): 是否為流局
                - "agent_won" (bool): agent 是否胡牌
                - "agent_deal_in" (bool): agent 是否放銃
        """
        self.total_hands += 1

        if hand_result.get("agent_won", False):
            self.wins += 1

        if hand_result.get("agent_deal_in", False):
            self.deal_ins += 1

        if hand_result.get("is_draw", False):
            self.draw_hands += 1

    def record_game(self, game_result: Dict):
        """
        記錄一局完整遊戲（半莊）的結果。

        🆕 若 game_result 包含 "hand_results"，則逐手呼叫 record_hand()，
        以 per-hand 語義計算和了率/放銃率。
        若 game_result 不含 "hand_results"（向後相容舊版 runner），
        則 fallback 為舊有的 game-level 邏輯。

        Args:
            game_result: 來自 SelfPlayRunner.run_match() 的 game_result dict。
                預期包含以下 key：
                - "agent_rank" (int): agent 排名 (1~4)
                - "is_win" (bool): agent 是否為第一名
                - "final_scores" (List[int]): 四人最終分數
                - "agent_score" (int): agent 最終分數
                - "is_agari" (bool, optional): agent 是否胡牌（舊版 fallback）
                - "is_houjuu" (bool, optional): agent 是否放銃（舊版 fallback）
                - "anyone_agari" (bool, optional): 本局是否有人胡牌（舊版 fallback）
                🆕 "hand_results" (List[Dict], optional): per-hand 統計
                🆕 "total_hands" (int, optional): 總局數
        """
        self.total_games += 1

        # ── 🆕 Per-hand 處理 ──
        hand_results = game_result.get("hand_results", None)
        if hand_results is not None and len(hand_results) > 0:
            for hand in hand_results:
                self.record_hand(hand)
        else:
            # ── 向後相容：舊版 game_result 不含 hand_results ──
            # 使用 game-level 的 is_agari / is_houjuu / anyone_agari 作為 fallback。
            # 注意：這是 per-game 語義，精確度不如 per-hand。
            if game_result.get("is_agari", False):
                self.wins += 1
                self.total_hands += 1
            if game_result.get("is_houjuu", False):
                self.deal_ins += 1
                self.total_hands += 1

            # 流局推斷（舊版 fallback 邏輯）
            anyone_agari = game_result.get("anyone_agari", None)
            if anyone_agari is None:
                anyone_agari = not is_draw_game(
                    game_result,
                    round_terminal=None,  # 舊版沒有 round_terminal
                )
            if not anyone_agari:
                self.draw_hands += 1
                if self.total_hands == 0 or (not game_result.get("is_agari", False) and not game_result.get("is_houjuu", False)):
                    # 確保流局也被計入分母
                    self.total_hands += 1

        # ── Game-level 排名 ──
        rank = game_result.get("agent_rank", None)
        if rank is not None and 1 <= rank <= 4:
            self.placements[rank] += 1

        # ── 累積分數 ──
        agent_score = game_result.get("agent_score", None)
        if agent_score is not None:
            self.cumulative_scores.append(float(agent_score))

    @property
    def win_rate(self) -> float:
        """
        🆕 和了率（Per-hand 語義）：agent 胡牌局數 / 總局數

        符合 Suphx 論文規範，分母 = total_hands。
        """
        if self.total_hands == 0:
            return 0.0
        return self.wins / self.total_hands

    @property
    def deal_in_rate(self) -> float:
        """
        🆕 放銃率（Per-hand 語義）：agent 放銃局數 / 總局數

        符合 Suphx 論文規範，分母 = total_hands。
        """
        if self.total_hands == 0:
            return 0.0
        return self.deal_ins / self.total_hands

    @property
    def placement_distribution(self) -> Dict[int, float]:
        """順位分佈：{1: pct, 2: pct, 3: pct, 4: pct}（百分比）"""
        if self.total_games == 0:
            return {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        return {
            rank: (count / self.total_games) * 100.0
            for rank, count in self.placements.items()
        }

    @property
    def avg_score(self) -> float:
        """agent 平均終局分數"""
        if len(self.cumulative_scores) == 0:
            return 0.0
        return sum(self.cumulative_scores) / len(self.cumulative_scores)

    def summary(self) -> Dict:
        """回傳完整統計摘要 dict。"""
        return {
            "total_games": self.total_games,
            "total_hands": self.total_hands,
            "win_rate": self.win_rate,
            "deal_in_rate": self.deal_in_rate,
            "placement_distribution": self.placement_distribution,
            "avg_score": self.avg_score,
            "wins": self.wins,
            "deal_ins": self.deal_ins,
            "draw_hands": self.draw_hands,
            "placements": dict(self.placements),
        }

    def report(self) -> str:
        """
        輸出格式化的統計報告字串。

        🆕 同時顯示 game-level（半莊）與 hand-level（每局牌）指標。

        Returns:
            str: 多行統計報告
        """
        d = self.placement_distribution
        lines = [
            "=" * 60,
            "           麻將 AI 自我對弈統計報告",
            "=" * 60,
            f"  半莊數 (Games)        : {self.total_games}",
            f"  總局數 (Hands)         : {self.total_hands}",
            "",
            "  ── Per-Hand 指標（Suphx 學術標準）──",
            f"  和了率 (Win Rate)      : {self.win_rate * 100:.2f}%  ({self.wins}/{self.total_hands})",
            f"  放銃率 (Deal-in Rate)  : {self.deal_in_rate * 100:.2f}%  ({self.deal_ins}/{self.total_hands})",
            f"  流局率 (Draw Rate)     : {self.draw_hands / max(self.total_hands, 1) * 100:.2f}%  ({self.draw_hands}/{self.total_hands})",
            "",
            "  ── Game-Level 指標 ──",
            f"  平均終局分數           : {self.avg_score:.1f}",
            "",
            "  順位分佈:",
            f"    1 位 (一位): {d[1]:.2f}%  ({self.placements[1]})",
            f"    2 位 (二位): {d[2]:.2f}%  ({self.placements[2]})",
            f"    3 位 (三位): {d[3]:.2f}%  ({self.placements[3]})",
            f"    4 位 (四位): {d[4]:.2f}%  ({self.placements[4]})",
            "=" * 60,
        ]
        return "\n".join(lines)

    def print_report(self):
        """直接印出統計報告。"""
        print(self.report())


# ============================================================================
#  輔助：從 game_result 或 proto round_terminal 判斷是否為和局
# ============================================================================

def is_draw_game(
    game_result: Dict,
    round_terminal: object = None,
) -> bool:
    """
    判斷一局是否為和局（流局，無人和牌）。

    🆕 優先使用 proto 原生欄位 round_terminal.no_winner，
    若不可用則 fallback 至分數變化閾值（放寬至 ±4000）。

    判斷優先級：
        1. 若 round_terminal 可用且 HasField("no_winner") → True（流局）
        2. 若 round_terminal 可用且有 wins → False（有人胡牌）
        3. Fallback：終局分數最大變化 < 4000 點 → True（可能為流局）

    Args:
        game_result: runner 回傳的 game_result dict（含 "final_scores"）
        round_terminal: 可選的 proto RoundTerminal 物件（None 表示不可用）

    Returns:
        bool: True 表示流局
    """
    # ── 優先級 1 & 2：使用 proto 原生欄位 ──
    if round_terminal is not None:
        try:
            # HasField("no_winner") 表示流局（荒牌流局、九種九牌、四風連打等）
            if round_terminal.HasField("no_winner"):
                return True
            # 有 wins 表示有人胡牌
            if len(round_terminal.wins) > 0:
                return False
        except Exception:
            pass  # proto 存取失敗時 fallback 到分數判斷

    # ── 優先級 3：Fallback 分數閾值（放寬至 4000） ──
    # 日麻荒牌流局中，不聽罰符變化範圍為 ±1000～±3000，
    # 若加上立直棒沉底（每根 1000）極端情況可能達 ±3000 + 1000 = ±4000
    scores = game_result.get("final_scores", [25000, 25000, 25000, 25000])
    max_delta = max(abs(s - 25000) for s in scores)
    return max_delta < 4000