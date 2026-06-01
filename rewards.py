"""
複合獎勵函數 (Composite Reward Function)
實作 Readme.md 中的三段式獎勵：
  R_total = r_potential + r_backward + r_penalty

使用 mjx 的 Observation / Hand / Event API。
"""

import sys
from pathlib import Path

import numpy as np
from typing import List, Dict, Optional, Tuple

# 📌 確保本地 mjx 模組可導入（需先編譯 mjx C++ 擴展 _mjx）
_mjx_path = Path(__file__).resolve().parent / "mjx"
if str(_mjx_path) not in sys.path:
    sys.path.insert(0, str(_mjx_path))

import mjx
from mjx.const import EventType, TileType


class MahjongRewardCalculator:
    """
    麻將複合獎勵計算機

    三段獎勵：
    1. r_potential  — 進攻潛力（每一步都計算）
    2. r_backward  — 回溯番數（round 結束後，分配給每一步）
    3. r_penalty   — 防守懲罰（放銃時觸發）
    """

    def __init__(
        self,
        base_exp_score: float = 2000.0,
        score_norm_factor: float = 10000.0,
        penalty_weight: float = 2.0,
        potential_weight: float = 0.3,
    ):
        """
        Args:
            base_exp_score: 預設期望打點（用於 r_potential 的基準分）
            score_norm_factor: 正規化因子，避免 PPO 梯度爆炸
            penalty_weight: 放銃懲罰倍率（方案四：預設 2.0，加重防守意識）
            potential_weight: r_potential 權重（v5: 0.3，降低沿途雞牌雜音）
        """
        self.base_exp_score = base_exp_score
        self.score_norm_factor = score_norm_factor
        self.penalty_weight = penalty_weight
        self.potential_weight = potential_weight

    # ==================== 共用輔助方法 ====================

    @staticmethod
    def _tile_list_to_34_count(tiles) -> np.ndarray:
        """
        將 Tile 列表轉為 34 維張數分佈。

        Args:
            tiles: List[Tile] 或任何 iterable of mjx.Tile
        Returns: np.ndarray shape=(34,), dtype=int32
        """
        counts = np.zeros(34, dtype=np.int32)
        for t in tiles:
            counts[t.type()] += 1
        return counts

    def count_visible_tiles(self, obs: mjx.Observation, tile_type: int) -> int:
        """
        計算某種牌在場上已可見的數量。
        範圍：自己手牌 + 自己副露 + 所有人捨牌 + 所有人副露 + 寶牌指示牌
        """
        count = 0

        # 1. 自己手牌（closed tiles）
        hand = obs.curr_hand()
        count += self._tile_list_to_34_count(hand.closed_tiles())[tile_type]

        # 2. 自己副露
        for open_meld in hand.opens():
            for t in open_meld.tiles():
                if t.type() == tile_type:
                    count += 1

        # 3. 從 events 解析所有人捨牌與副露
        for event in obs.events():
            evt_type = event.type()
            if evt_type in (EventType.DISCARD, EventType.TSUMOGIRI):
                tile = event.tile()
                if tile is not None and tile.type() == tile_type:
                    count += 1
            elif evt_type in (
                EventType.CHI,
                EventType.PON,
                EventType.ADDED_KAN,
                EventType.OPEN_KAN,
                EventType.CLOSED_KAN,
            ):
                open_obj = event.open()
                if open_obj is not None:
                    for t in open_obj.tiles():
                        if t.type() == tile_type:
                            count += 1

        # 4. 寶牌指示牌
        for dora_type in obs.doras():
            if dora_type == tile_type:
                count += 1

        return min(count, 4)  # 一種牌最多 4 張

    def _get_current_hand_34(self, obs: mjx.Observation) -> np.ndarray:
        """
        取得當前手牌（含副露）的 34 維張數分佈。
        Returns: np.ndarray shape=(34,), dtype=int32
        """
        hand = obs.curr_hand()
        counts = self._tile_list_to_34_count(hand.closed_tiles())

        for open_meld in hand.opens():
            for t in open_meld.tiles():
                counts[t.type()] += 1

        return counts

    # ==================== 🆕 方案四：寶牌相關方法 ====================

    def count_dora_in_hand(self, obs: mjx.Observation) -> int:
        """
        計算手牌（含副露）中的寶牌張數。
        
        寶牌來源：
        1. 赤寶牌（tile.is_red()）
        2. 寶牌指示牌對應的寶牌（obs.doras() → 指示牌種類，寶牌為指示牌+1）

        對齊 mjx 的 dora_num_in_hand feature 計算邏輯。
        
        Returns:
            dora_count: 手牌中寶牌總張數（上限 13）
        """
        dora_indicators = list(obs.doras())  # 寶牌指示牌種類列表
        dora_count = 0

        # 檢查手牌中的 closed tiles
        for tile in obs.curr_hand().closed_tiles():
            if tile.is_red():
                dora_count += 1
            for dora_type in dora_indicators:
                if tile.type() == dora_type:
                    dora_count += 1

        # 檢查副露中的 tiles
        for meld in obs.curr_hand().opens():
            for tile in meld.tiles():
                if tile.is_red():
                    dora_count += 1
                for dora_type in dora_indicators:
                    if tile.type() == dora_type:
                        dora_count += 1

        return min(dora_count, 13)

    def calculate_dora_potential_reward(self, obs: mjx.Observation) -> float:
        """
        r_dora = dora_count × 0.05 / 1000

        即時獎勵，鼓勵模型保留寶牌以追求高價值牌型。
        每多一張寶牌，獎勵增加 0.05/1000。
        
        Returns:
            正規化後的寶牌潛力獎勵
        """
        dora_count = self.count_dora_in_hand(obs)
        return dora_count * 0.05 / 1000.0

    # ==================== 1. r_potential — 進攻潛力 ====================

    def calculate_ukeire(self, obs: mjx.Observation) -> int:
        """
        計算 Ukeire（有效進張數）。
        對每個 effective_draw_type，計算剩餘張數 = 4 - 已可見數量。
        """
        hand = obs.curr_hand()
        effective_draws = hand.effective_draw_types()

        total_ukeire = 0
        for tile_type in effective_draws:
            visible = self.count_visible_tiles(obs, int(tile_type))
            remaining = 4 - visible
            total_ukeire += max(0, remaining)

        return total_ukeire

    def calculate_potential_reward(self, obs: mjx.Observation) -> float:
        """
        r_potential = base_exp_score * Ukeire(s) / (Shanten(s) + 1)

        對齊 Readme.md 的進攻潛力公式。
        向聽數愈低、有效進張愈多 → 獎勵愈高。
        """
        hand = obs.curr_hand()
        shanten = hand.shanten_number()
        ukeire = self.calculate_ukeire(obs)

        if shanten <= 0:
            # 已聽牌或已和牌，給予基準獎勵
            return self.base_exp_score / self.score_norm_factor

        potential = self.base_exp_score * ukeire / (shanten + 1.0)
        return potential / self.score_norm_factor * self.potential_weight

    # ==================== 2. r_backward — Han Backward ====================

    def compute_winning_hand_info(
        self, obs: mjx.Observation
    ) -> Optional[np.ndarray]:
        """
        從最終 observation 擷取和牌型資訊。
        只有在 player 以 TSUMO 或 RON 和牌時才有意義。

        ⚠️ 注意：obs.tens() 在非終局 observation 中僅回傳初始分，不可用於 reward 計算。
        此方法只返回手牌分佈，結算得分應由外部（env.rewards()）傳入。

        Returns:
            hand_34: 最終手牌的 34 維張數分佈，或 None（非和牌狀態）
        """
        my_idx = obs.who()
        events = obs.events()

        for event in events:
            if event.type() in (EventType.TSUMO, EventType.RON):
                if event.who() == my_idx:
                    return self._get_current_hand_34(obs)

        return None

    def calculate_backward_reward(
        self,
        final_hand_34: np.ndarray,
        final_score: int,
        current_hand_34: np.ndarray,
    ) -> float:
        """
        r_backward = sum_i min(final_hand_i, current_hand_i) * unit_score_i

        對齊 Readme.md Eq.12。
        將最終得分按牌型比例均分，回饋給每一步中與最終牌型重合的牌。

         🆕 v5：Score² 非線性獎勵
             unit_score = final_score² / (total_tiles × score_norm_factor)
             效果：30分→0.09, 90分→0.81（9倍差距，鼓勵做大牌和牌）

        Args:
            final_hand_34: 最終和牌型的 34 維張數分佈
            final_score: 最終得分（正整數）
            current_hand_34: 當前步的手牌 34 維張數分佈
        """
        total_tiles = int(final_hand_34.sum())
        if total_tiles == 0:
            return 0.0

        # 🆕 v5：Score² 非線性獎勵，大幅放大高分牌的回饋
        # 30分→0.09, 90分→0.81（9倍差距），鼓勵模型追求大牌和牌
        unit_score = (final_score * final_score) / (total_tiles * self.score_norm_factor)

        overlap = np.minimum(final_hand_34, current_hand_34)
        raw_reward = np.sum(overlap * unit_score)

        return raw_reward

    # ==================== 3. r_penalty — 防守懲罰 ====================

    def check_houjuu(
        self, obs: mjx.Observation
    ) -> bool:
        """
        檢查目前 observation 中，自己是否放銃。

        放銃判斷邏輯：
        - 事件中出現 RON，且 RON 的執行者不是自己
        - RON 事件之前的最後一個 DISCARD/TSUMOGIRI 是自己

        ⚠️ 返回 bool 而非分數；分數應由外部 env.rewards() 取得後傳入 calculate_penalty_reward。

        Returns:
            is_houjuu: True 表示自己放銃
        """
        my_idx = obs.who()
        events = obs.events()

        for i, event in enumerate(events):
            if event.type() == EventType.RON and event.who() != my_idx:
                for j in range(i - 1, -1, -1):
                    prev = events[j]
                    if prev.type() in (EventType.DISCARD, EventType.TSUMOGIRI):
                        return prev.who() == my_idx

        return False

    def calculate_penalty_reward(
        self, obs: mjx.Observation, opponent_score: Optional[float] = None
    ) -> float:
        """
        r_penalty = -對手得分 × penalty_weight / norm_factor（如果自己放銃）
        對齊 Readme.md Eq.13
        
        🆕 方案四：penalty_weight=2.0，放銃懲罰加倍，鼓勵防守

        Args:
            obs: 當前 observation
            opponent_score: 對手結算得分（應來自 env.rewards()，非 obs.tens()）
        """
        is_houjuu = self.check_houjuu(obs)
        if is_houjuu and opponent_score is not None:
            return -abs(opponent_score) * self.penalty_weight / self.score_norm_factor
        return 0.0

    # ==================== 4. r_progression — 進步獎勵 ====================

    def calculate_progression_reward(
        self,
        prev_shanten: Optional[int],
        curr_shanten: int,
    ) -> float:
        """
        r_progression = (prev_shanten - curr_shanten) * progression_scale / norm_factor
        
        鼓勵向聽數下降（進展），懲罰向聽數上升（倒退）。
        
        Args:
            prev_shanten: 前一步的向聽數（第一步為 None）
            curr_shanten: 當前步的向聽數
        Returns:
            正規化後的進步獎勵
        """
        if prev_shanten is None:
            return 0.0
        delta = prev_shanten - curr_shanten  # 正值=進步, 負值=退步
        progression_scale = 1.0
        return delta * progression_scale / self.score_norm_factor

    # ==================== 整合方法 ====================

    def compute_step_reward(
        self,
        obs: mjx.Observation,
        final_hand_34: Optional[np.ndarray] = None,
        final_score: Optional[int] = None,
        opponent_score: Optional[float] = None,
    ) -> float:
        """
        計算單步完整獎勵。

        🆕 方案四：R_total = r_potential + r_dora + r_backward + r_penalty

        Args:
            obs: 當前 observation
            final_hand_34: 最終和牌型（round 結束後，用於 r_backward）
            final_score: 最終得分（應來自 env.rewards()，用於 r_backward）
            opponent_score: 對手結算得分（應來自 env.rewards()，用於 r_penalty）
        Returns:
            正規化後的總獎勵
        """
        r_potential = self.calculate_potential_reward(obs)
        r_dora = self.calculate_dora_potential_reward(obs)  # 🆕 方案四
        r_penalty = self.calculate_penalty_reward(obs, opponent_score)

        r_backward = 0.0
        if final_hand_34 is not None and final_score is not None:
            current_hand = self._get_current_hand_34(obs)
            r_backward = self.calculate_backward_reward(
                final_hand_34, final_score, current_hand
            )

        return r_potential + r_dora + r_backward + r_penalty

    def compute_trajectory_rewards(
        self,
        observations: List[mjx.Observation],
        final_hand_34: Optional[np.ndarray] = None,
        final_score: Optional[int] = None,
        opponent_score: Optional[float] = None,
    ) -> List[float]:
        """
        計算整條軌跡每步的獎勵（用於 PPO rollout 結束後的 reward 分配）。

        Args:
            observations: 軌跡中的所有 observation
            final_hand_34: 最終和牌型（若有和牌）
            final_score: 最終得分（若有和牌）
            opponent_score: 對手結算得分（若有）
        Returns:
            每步獎勵列表
        """
        return [
            self.compute_step_reward(
                obs, final_hand_34, final_score, opponent_score
            )
            for obs in observations
        ]


def create_default_calculator() -> MahjongRewardCalculator:
    """建立預設參數的獎勵計算機。"""
    # 🆕 方案四：score_norm_factor=10000（v2 為 1000）
    # r_backward 非線性放大後（score^1.5），reward 範圍從 0.003~0.03 升到 0.03~0.68，
    # 需要更大的正規化因子才能控制 rtg 累積不至於讓 Value Head 爆炸
    return MahjongRewardCalculator(base_exp_score=2000.0, score_norm_factor=10000.0, penalty_weight=2.0, potential_weight=0.3)
