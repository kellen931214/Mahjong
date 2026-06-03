import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple

_mjx_path = Path(__file__).resolve().parent / "mjx"
if str(_mjx_path) not in sys.path:
    sys.path.insert(0, str(_mjx_path))

import mjx
from mjx.const import EventType, TileType


class MahjongRewardCalculator:
    def __init__(
        self,
        base_exp_score: float = 2000.0,
        score_norm_factor: float = 10000.0,
        penalty_weight: float = 2.0,
        potential_weight: float = 0.3,
    ):
        self.base_exp_score = base_exp_score
        self.score_norm_factor = score_norm_factor
        self.penalty_weight = penalty_weight
        self.potential_weight = potential_weight

    @staticmethod
    def _tile_list_to_34_count(tiles) -> np.ndarray:
        counts = np.zeros(34, dtype=np.int32)
        for t in tiles:
            counts[t.type()] += 1
        return counts

    # ==================== 🛠️ 修正 1：obs.doras() 已返回真實寶牌，無需二次映射 ====================
    #   C++ 底層 observation.cpp L94-101 已執行 internal::IndicatorToDora()，
    #   因此 obs.doras() 返回的是真實的寶牌 TileType，無需 Python 層再次轉換。

    # ==================== 🛠️ 修正 2：過濾被吃碰走的時間步，防止重複計數 ====================
    def count_visible_tiles(self, obs: mjx.Observation, tile_type: int) -> int:
        """計算某種牌在場上已可見的數量（嚴格過濾被副露截胡的捨牌）"""
        count = 0
        hand = obs.curr_hand()
        count += self._tile_list_to_34_count(hand.closed_tiles())[tile_type]

        for open_meld in hand.opens():
            for t in open_meld.tiles():
                if t.type() == tile_type: count += 1

        # 為了防止捨牌被吃碰後重複計算，我們用不重複的 tile id 集合或精確事件過濾
        # 在這裡，最安全的做法是直接統計所有事件中「最終留下來的物理牌」
        stolen_tile_ids = set()
        events = obs.events()
        
        # 先搜集所有被副露消耗掉的物理牌 ID
        # 🚀【修正】mjx C++ 中 CHI/PON/OPEN_KAN 的 event.tile() 返回 None（見 event.cpp L42-50），
        #   必須透過 open().stolen_tile() 取得被吃/碰/大明槓走的牌 ID。
        #   ADDED_KAN（加槓）是從手牌補入已有 PON，不從捨牌河取牌，因此不在此處處理。
        for event in events:
            if event.type() in (EventType.CHI, EventType.PON, EventType.OPEN_KAN):
                open_obj = event.open()
                if open_obj is not None:
                    stolen_tile_ids.add(open_obj.stolen_tile().id())

        # 重新統計捨牌河，已被副露拿走的物理牌不計入捨牌河分母
        for event in events:
            evt_type = event.type()
            if evt_type in (EventType.DISCARD, EventType.TSUMOGIRI):
                t = event.tile()
                if t and t.id() not in stolen_tile_ids and t.type() == tile_type:
                    count += 1
            elif evt_type == EventType.CLOSED_KAN: # 暗槓是自己手牌，獨立計算
                open_obj = event.open()
                if open_obj:
                    for t in open_obj.tiles():
                        if t.type() == tile_type: count += 1

        # 寶牌指示牌本身不消耗物理牌，但將真實寶牌計入可見以保守估計可用牌數
        # obs.doras() 返回已轉換的真實寶牌 TileType（非指示牌）
        for dora_type in obs.doras():
            if int(dora_type) == tile_type:
                count += 1

        return min(count, 4)

    def _get_current_hand_34(self, obs: mjx.Observation) -> np.ndarray:
        hand = obs.curr_hand()
        counts = self._tile_list_to_34_count(hand.closed_tiles())
        for open_meld in hand.opens():
            for t in open_meld.tiles():
                counts[t.type()] += 1
        return counts

    # ==================== 🛠️ 修正 3：對齊真實寶牌計算 ====================
    def count_dora_in_hand(self, obs: mjx.Observation) -> int:
        """計算手牌與副露中真正的寶牌數量（含赤寶牌與指示牌轉換）"""
        # obs.doras() 已返回真實寶牌 TileType（C++ 層已做 IndicatorToDora 轉換）
        real_dora_types = [int(d) for d in obs.doras()]
        dora_count = 0

        for tile in obs.curr_hand().closed_tiles():
            if tile.is_red(): dora_count += 1
            if tile.type() in real_dora_types: dora_count += 1

        for meld in obs.curr_hand().opens():
            for tile in meld.tiles():
                if tile.is_red(): dora_count += 1
                if tile.type() in real_dora_types: dora_count += 1

        return min(dora_count, 13)

    def calculate_dora_potential_reward(self, obs: mjx.Observation) -> float:
        """寶牌留存獎勵：每張寶牌給予 0.01 獎勵（最大 0.13），與 r_potential 量級對齊"""
        dora_count = self.count_dora_in_hand(obs)
        return dora_count * 0.01

    def calculate_ukeire(self, obs: mjx.Observation) -> int:
        hand = obs.curr_hand()
        effective_draws = hand.effective_draw_types()
        total_ukeire = 0
        for tile_type in effective_draws:
            visible = self.count_visible_tiles(obs, int(tile_type))
            total_ukeire += max(0, 4 - visible)
        return total_ukeire

    def calculate_potential_reward(self, obs: mjx.Observation) -> float:
        hand = obs.curr_hand()
        shanten = hand.shanten_number()
        ukeire = self.calculate_ukeire(obs)

        if shanten <= 0:
            return self.base_exp_score / self.score_norm_factor

        potential = self.base_exp_score * ukeire / (shanten + 1.0)
        return potential / self.score_norm_factor * self.potential_weight

    def compute_winning_hand_info(self, obs: mjx.Observation) -> Optional[np.ndarray]:
        my_idx = obs.who()
        for event in obs.events():
            if event.type() in (EventType.TSUMO, EventType.RON):
                if event.who() == my_idx:
                    return self._get_current_hand_34(obs)
        return None

    # ==================== 🛠️ 修正 4：修正平方縮放範疇，防止 Critic 爆炸 ====================
    def calculate_backward_reward(
        self,
        final_hand_34: np.ndarray,
        score_delta: int,
        current_hand_34: np.ndarray,
    ) -> float:
        """
        將最終得分進行非線性二次縮放。
        為防止 raw score (如 8000) 平方導致數值暴走，
        在此處將點數先轉為學術級基底（點數 / 1000），再執行平方優化。
        """
        total_tiles = int(final_hand_34.sum())
        if total_tiles == 0 or score_delta <= 0:
            return 0.0

        # 將真實點數縮小為點數基底 (例如 8000點 -> 基底為 8)
        score_basis = score_delta / 1000.0
        
        # 實作限制：(8^2) / (14 * 10000) 這樣數值範圍會極度安全且保有非線性拉開大牌回報的效果
        unit_score = (score_basis * score_basis) / (total_tiles * self.score_norm_factor)

        overlap = np.minimum(final_hand_34, current_hand_34)
        raw_reward = np.sum(overlap * unit_score)
        return raw_reward

    # ==================== 🛠️ 修正 5：補上加槓（槍槓）事件防護 ====================
    def check_houjuu(self, obs: mjx.Observation) -> bool:
        """檢查自己是否放銃（完美相容普通放銃與槍槓放銃）"""
        my_idx = obs.who()
        events = obs.events()

        for i, event in enumerate(events):
            if event.type() == EventType.RON and event.who() != my_idx:
                # 往回尋找導致這個 RON 的物理動作
                for j in range(i - 1, -1, -1):
                    prev = events[j]
                    # 包含普通打牌、摸切打牌、以及加槓（防止槍槓漏抓）
                    # 🚀 回溯 DISCARD（手切）、TSUMOGIRI（摸切）、ADDED_KAN（槍槓）、OPEN_KAN（大明槓槍槓）
                    if prev.type() in (EventType.DISCARD, EventType.TSUMOGIRI, EventType.ADDED_KAN, EventType.OPEN_KAN):
                        return prev.who() == my_idx
        return False

    def calculate_penalty_reward(self, obs: mjx.Observation, opponent_score: Optional[float] = None) -> float:
        is_houjuu = self.check_houjuu(obs)
        if is_houjuu and opponent_score is not None:
            return -abs(opponent_score) * self.penalty_weight / self.score_norm_factor
        return 0.0

    def calculate_progression_reward(self, prev_shanten: Optional[int], curr_shanten: int) -> float:
        if prev_shanten is None: return 0.0
        delta = prev_shanten - curr_shanten
        # 🚀【修正】原縮放 1.0/10000=0.0001 過小，提升至 0.05 以提供有效梯度信號
        return delta * 0.05

    def compute_step_reward(
        self,
        obs: mjx.Observation,
        final_hand_34: Optional[np.ndarray] = None,
        score_delta: Optional[int] = None,
        opponent_score: Optional[float] = None,
    ) -> float:
        r_potential = self.calculate_potential_reward(obs)
        r_dora = self.calculate_dora_potential_reward(obs)
        r_penalty = self.calculate_penalty_reward(obs, opponent_score)

        r_backward = 0.0
        if final_hand_34 is not None and score_delta is not None:
            current_hand = self._get_current_hand_34(obs)
            r_backward = self.calculate_backward_reward(final_hand_34, score_delta, current_hand)

        return r_potential + r_dora + r_backward + r_penalty


def create_default_calculator() -> MahjongRewardCalculator:
    return MahjongRewardCalculator(base_exp_score=2000.0, score_norm_factor=10000.0, penalty_weight=2.0, potential_weight=0.3)