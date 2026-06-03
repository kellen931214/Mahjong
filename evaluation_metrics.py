"""
evaluation_metrics.py — 離線準確率計算 + 線上對局統計追蹤

基於 Suphx 論文評估指標體系：
  1. 離線模型準確率（Offline Action Accuracy）：按動作類別分別計算
  2. 線上對局統計器（Online Tracker）：追蹤和了率、放銃率、順位分佈等
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
#  2. 線上對局統計器 (Online Tracker)
# ============================================================================

class MahjongMetricTracker:
    """
    追蹤模型在數千場自主對弈（Self-Play）中的核心指標。

    動態維護並計算：
      - 和了率 (Win Rate)：自己胡牌的次數 / 總局數
      - 放銃率 (Deal-in Rate)：點炮給對手的次數 / 總局數
      - 順位分佈 (Placement Distribution)：1/2/3/4 名的精確百分比
      - 每局是和局還是有贏牌

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
        self.total_games = 0
        self.wins = 0                    # agent 胡牌次數（is_agari=True）
        self.deal_ins = 0                # agent 放銃次數
        self.placements = {1: 0, 2: 0, 3: 0, 4: 0}  # 排名計數
        self.draw_games = 0              # 和局（無人和牌）次數
        self.agari_games = 0             # 有人胡牌的局數
        self.cumulative_scores: List[float] = []  # 累積分數用於計算均分

    def record_game(self, game_result: Dict):
        """
        記錄一局遊戲的結果。

        Args:
            game_result: 來自 SelfPlayRunner.run_match() 的 game_result dict。
                預期包含以下 key：
                - "is_agari" (bool): agent 是否胡牌
                - "agent_rank" (int): agent 排名 (1~4)
                - "is_win" (bool): agent 是否為第一名
                - "final_scores" (List[int]): 四人最終分數
                - "agent_score" (int): agent 最終分數
                - "is_houjuu" (bool, optional): agent 是否放銃。
                  若未提供則預設為 False。
                - "anyone_agari" (bool, optional): 本局是否有人胡牌。
                  若未提供則用 is_draw_game() 從 final_scores 反推。
        """
        self.total_games += 1

        # 和了
        if game_result.get("is_agari", False):
            self.wins += 1

        # 放銃
        if game_result.get("is_houjuu", False):
            self.deal_ins += 1

        # 排名
        rank = game_result.get("agent_rank", None)
        if rank is not None and 1 <= rank <= 4:
            self.placements[rank] += 1

        # 和局 vs 有人胡牌
        anyone_agari = game_result.get("anyone_agari", None)
        if anyone_agari is None:
            # 🚀【修正】不能 fallback 到 is_agari：若對手胡牌但 agent 沒胡，
            # is_agari=False 會把「有人胡牌」誤判為和局，造成 draw_games 被放大 3~4 倍。
            # 正確做法：用 is_draw_game() 從 final_scores 反推是否為和局。
            anyone_agari = not is_draw_game(game_result)

        if anyone_agari:
            self.agari_games += 1
        else:
            self.draw_games += 1

        # 累積分數
        agent_score = game_result.get("agent_score", None)
        if agent_score is not None:
            self.cumulative_scores.append(float(agent_score))

    @property
    def win_rate(self) -> float:
        """和了率：自己胡牌次數 / 總局數"""
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games

    @property
    def deal_in_rate(self) -> float:
        """放銃率：點炮次數 / 總局數"""
        if self.total_games == 0:
            return 0.0
        return self.deal_ins / self.total_games

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
            "win_rate": self.win_rate,
            "deal_in_rate": self.deal_in_rate,
            "placement_distribution": self.placement_distribution,
            "avg_score": self.avg_score,
            "wins": self.wins,
            "deal_ins": self.deal_ins,
            "agari_games": self.agari_games,
            "draw_games": self.draw_games,
            "placements": dict(self.placements),
        }

    def report(self) -> str:
        """
        輸出格式化的統計報告字串。

        Returns:
            str: 多行統計報告
        """
        d = self.placement_distribution
        lines = [
            "=" * 60,
            "           麻將 AI 自我對弈統計報告",
            "=" * 60,
            f"  總局數              : {self.total_games}",
            f"  和了率 (Win Rate)    : {self.win_rate * 100:.2f}%  ({self.wins}/{self.total_games})",
            f"  放銃率 (Deal-in Rate): {self.deal_in_rate * 100:.2f}%  ({self.deal_ins}/{self.total_games})",
            f"  平均終局分數         : {self.avg_score:.1f}",
            "",
            "  順位分佈:",
            f"    1 位 (一位): {d[1]:.2f}%  ({self.placements[1]})",
            f"    2 位 (二位): {d[2]:.2f}%  ({self.placements[2]})",
            f"    3 位 (三位): {d[3]:.2f}%  ({self.placements[3]})",
            f"    4 位 (四位): {d[4]:.2f}%  ({self.placements[4]})",
            "",
            f"  有人胡牌局數 : {self.agari_games}",
            f"  和局（流局） : {self.draw_games}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def print_report(self):
        """直接印出統計報告。"""
        print(self.report())


# ============================================================================
#  輔助：從 runner 的 game_result 判斷是否為和局
# ============================================================================

def is_draw_game(game_result: Dict) -> bool:
    """
    判斷一局是否為和局（流局，無人和牌）。

    判斷邏輯：如果沒有任何玩家胡牌（is_agari 全部為 False），
    且沒有人分數顯著偏離 25000 起始分，則為和局。

    Args:
        game_result: runner 回傳的 game_result

    Returns:
        bool: True 表示和局
    """
    if game_result.get("is_agari", False):
        return False
    # 檢查是否有人胡牌引發的分數變動
    scores = game_result.get("final_scores", [25000, 25000, 25000, 25000])
    max_delta = max(abs(s - 25000) for s in scores)
    # 若所有人分數變動都在 ±1000 以內，極可能是和局
    return max_delta < 1000