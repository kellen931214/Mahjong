import sys
from pathlib import Path
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

_mjx_path = Path(__file__).resolve().parent.parent / "mjx"  # utli/ → ../mjx
if str(_mjx_path) not in sys.path:
    sys.path.insert(0, str(_mjx_path))

import mjx
from mjx.const import EventType, TileType


class MahjongRewardCalculator:
    """
    🧠 條件式離散攻守雙模式獎勵計算器（Conditional Discrete Dual-Mode Reward Calculator）

    採用「條件特徵注入（Conditional Feature Injection）」範式：
    - 獎勵函數仍為 MDP 的靜態組成（不中途換 reward），保證 Critic 平穩性
    - 攻守人格訊號透過 2 維 One-hot 特徵注入 Decision Mamba 骨幹，由 SSM 動態學習人格切換

    ⚔️ 攻擊模式（Attack）：  油門全開——高向聽權重、高寶牌獎勵、標準放銃懲罰
    🛡️ 防守模式（Defense）： 油門斷電——極低向聽權重、低寶牌獎勵、重罰放銃（6.0x）
    """

    # ------------------------------------------------------------------
    # 雙模式參數表（所有權重在此集中管理，消融實驗時只需改這裡）
    # ------------------------------------------------------------------
    MODE_PARAMS: Dict[str, Dict[str, float]] = {
        "attack": {
            "penalty_weight": 1.8,       # 標準放銃懲罰倍率
            "potential_weight": 0.4,      # 向聽潛力「油門」權重
            "dora_weight": 0.01,          # 每張寶牌的即時獎勵
            "progression_weight": 0.05,   # 向聽數改善 Δ 的單位獎勵
        },
        "defense": {
            "penalty_weight": 6.0,        # 重度放銃懲罰，逼迫絕對防守
            "potential_weight": 0.08,     # 攻擊的 20%，保留微弱進攻意識
            "dora_weight": 0.005,         # 攻擊的 50%
            "progression_weight": 0.025,  # 攻擊的 50%，強化向聽進展梯度
        },
    }

    def __init__(
        self,
        base_exp_score: float = 2000.0,
        score_norm_factor: float = 10000.0,
    ):
        """
        Args:
            base_exp_score:    聽牌期望得點基準（用於 r_potential 聽牌上限計算）
            score_norm_factor: 得點歸一化因子（將點數統一壓縮到合理數值範圍）
        """
        # ── 全域常數（與模式無關）──
        self.base_exp_score = base_exp_score
        self.score_norm_factor = score_norm_factor

        # ── 模式狀態機 ──
        self.current_mode: str = "attack"  # 預設開局為進攻模式

    # ==================================================================
    # 🎛️ 模式控制介面
    # ==================================================================

    def set_mode(self, mode_str: str) -> None:
        """
        動態切換攻守模式。
        
        Args:
            mode_str: "attack" 或 "defense"
        
        Raises:
            ValueError: 若傳入不支援的模式字串
        """
        if mode_str not in self.MODE_PARAMS:
            raise ValueError(
                f"不支援的模式 '{mode_str}'，可用模式: {list(self.MODE_PARAMS.keys())}"
            )
        self.current_mode = mode_str

    def get_mode_tensor(self) -> torch.Tensor:
        """
        取得當前模式的 2 維 One-hot 條件特徵張量。

        此張量將被 torch.cat 拼接到原始 1380 維狀態向量後方，
        形成 1382 維的增廣狀態（Augmented State），餵入 Decision Mamba 骨幹。

        Returns:
            torch.Tensor: shape (2,)
                [1.0, 0.0] → 攻擊模式
                [0.0, 1.0] → 防守模式
        """
        if self.current_mode == "attack":
            return torch.tensor([1.0, 0.0], dtype=torch.float32)
        else:
            return torch.tensor([0.0, 1.0], dtype=torch.float32)

    # ==================================================================
    # 🔍 內部輔助：動態讀取當前模式參數
    #   所有獎勵核心函數透過此 property 取得當前的有效權重，
    #   避免在每個函數中重複索引字典。
    # ==================================================================

    @property
    def _pw(self) -> float:
        """當前模式的 penalty_weight（放銃懲罰倍率）"""
        return self.MODE_PARAMS[self.current_mode]["penalty_weight"]

    @property
    def _ptw(self) -> float:
        """當前模式的 potential_weight（向聽潛力權重）"""
        return self.MODE_PARAMS[self.current_mode]["potential_weight"]

    @property
    def _dw(self) -> float:
        """當前模式的 dora_weight（寶牌留存單位獎勵）"""
        return self.MODE_PARAMS[self.current_mode]["dora_weight"]

    @property
    def _pgw(self) -> float:
        """當前模式的 progression_weight（向聽進展 Δ 單位獎勵）"""
        return self.MODE_PARAMS[self.current_mode]["progression_weight"]

    # ==================================================================
    # 🧱 靜態工具函數（與模式無關，保持原樣）
    # ==================================================================

    @staticmethod
    def _tile_list_to_34_count(tiles) -> np.ndarray:
        """將 mjx Tile 清單轉換為 34 類牌的計數向量"""
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
                if t.type() == tile_type:
                    count += 1

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
            elif evt_type == EventType.CLOSED_KAN:  # 暗槓是自己手牌，獨立計算
                open_obj = event.open()
                if open_obj:
                    for t in open_obj.tiles():
                        if t.type() == tile_type:
                            count += 1

        # 寶牌指示牌本身不消耗物理牌，但將真實寶牌計入可見以保守估計可用牌數
        # obs.doras() 返回已轉換的真實寶牌 TileType（非指示牌）
        for dora_type in obs.doras():
            if int(dora_type) == tile_type:
                count += 1

        return min(count, 4)

    def _get_current_hand_34(self, obs: mjx.Observation) -> np.ndarray:
        """將當前手牌（含副露）轉換為 34 類計數向量"""
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
            if tile.is_red():
                dora_count += 1
            if tile.type() in real_dora_types:
                dora_count += 1

        for meld in obs.curr_hand().opens():
            for tile in meld.tiles():
                if tile.is_red():
                    dora_count += 1
                if tile.type() in real_dora_types:
                    dora_count += 1

        return min(dora_count, 13)

    # ==================================================================
    # ⚔️🛡️ 雙模式動態獎勵核心（四個函數透過 self._dw / self._pw / self._ptw / self._pgw
    #         讀取當前 effective weight，實現攻守分流）
    # ==================================================================

    def calculate_dora_potential_reward(self, obs: mjx.Observation) -> float:
        """
        🀄 寶牌留存獎勵（Mode-Aware）

        攻擊模式：dora_weight=0.01 → 積極保留寶牌追求大牌
        防守模式：dora_weight=0.002 → 即使手上有寶牌也可為安全考量化切出
        """
        dora_count = self.count_dora_in_hand(obs)
        return dora_count * self._dw  # 動態讀取當前模式的 dora_weight

    def calculate_ukeire(self, obs: mjx.Observation) -> int:
        """
        計算有效進張數（有効牌枚数 / Ukeire）

        走查所有 effective_draw_types()，扣除場上已可見的牌，得到實際可摸進的有效牌張數。
        （此函數不涉及模式權重，純為 potential_reward 的因子）
        """
        hand = obs.curr_hand()
        effective_draws = hand.effective_draw_types()
        total_ukeire = 0
        for tile_type in effective_draws:
            visible = self.count_visible_tiles(obs, int(tile_type))
            total_ukeire += max(0, 4 - visible)
        return total_ukeire

    def calculate_potential_reward(self, obs: mjx.Observation) -> float:
        """
        📐 向聽潛力獎勵 r_potential（Mode-Aware ——「油門」）

        攻擊模式：potential_weight=0.4 → 全力推進向聽數，積極做大牌
        防守模式：potential_weight=0.01 → 油門被物理斷電，避免貪攻而放銃

        公式：
            if shanten <= 0: return base_exp_score / score_norm_factor  (聽牌上限)
            else:            return base_exp_score * ukeire / (shanten+1) / score_norm_factor * potential_weight
        """
        hand = obs.curr_hand()
        shanten = hand.shanten_number()
        ukeire = self.calculate_ukeire(obs)

        if shanten <= 0:
            # 已聽牌：給予固定的高分基底（但仍在攻擊/防守模式中保持不變）
            return self.base_exp_score / self.score_norm_factor

        potential = self.base_exp_score * ukeire / (shanten + 1.0)
        return potential / self.score_norm_factor * self._ptw  # 動態讀取 potential_weight

    # ==================================================================
    # 🏆 終局後向得分（Mode-Invariant —— 只取決於是否胡牌）
    # ==================================================================

    def compute_winning_hand_info(self, obs: mjx.Observation) -> Optional[np.ndarray]:
        """
        若當前觀測中包含自身的 TSUMO 或 RON 事件（即本局胡牌），
        回傳最終胡牌面的 34 類計數向量；否則回傳 None。
        """
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
        🏆 終局後向得分分佈 r_backward（Mode-Invariant）

        將終局獲得的延遲點數（Delayed Final Score）依照結構重要性拆分，
        評估當前手牌與最終胡牌面（hand_hu）的重疊度，讓模型在每一步能規律累積稠密獎勵。

        為防止 raw score (如 8000) 平方導致數值暴走，
        在此處將點數先轉為學術級基底（點數 / 1000），再執行平方優化。
        """
        total_tiles = int(final_hand_34.sum())
        if total_tiles == 0 or score_delta <= 0:
            return 0.0

        # 將真實點數縮小為點數基底 (例如 8000點 → 基底為 8)
        score_basis = score_delta / 1000.0

        # 實作限制：(8^2) / (14 * 10000) 這樣數值範圍會極度安全且保有非線性拉開大牌回報的效果
        unit_score = (score_basis * score_basis) / (total_tiles * self.score_norm_factor)

        overlap = np.minimum(final_hand_34, current_hand_34)
        raw_reward = np.sum(overlap * unit_score)
        return raw_reward

    # ==================== 🛠️ 修正 5：補上加槓（槍槓）事件防護 ====================
    def check_houjuu(self, obs: mjx.Observation) -> bool:
        """
        檢查自己是否放銃（完美相容普通放銃與槍槓放銃）

        回溯邏輯：從 RON 事件向前搜索，找最近一次由自己所執行的
        DISCARD / TSUMOGIRI / ADDED_KAN / OPEN_KAN 動作，
        若找到則確認為放銃。
        """
        my_idx = obs.who()
        events = obs.events()

        for i, event in enumerate(events):
            if event.type() == EventType.RON and event.who() != my_idx:
                # 往回尋找導致這個 RON 的物理動作
                for j in range(i - 1, -1, -1):
                    prev = events[j]
                    # 包含普通打牌、摸切打牌、以及加槓（防止槍槓漏抓）
                    # 🚀 回溯 DISCARD（手切）、TSUMOGIRI（摸切）、ADDED_KAN（槍槓）、OPEN_KAN（大明槓槍槓）
                    if prev.type() in (
                        EventType.DISCARD,
                        EventType.TSUMOGIRI,
                        EventType.ADDED_KAN,
                        EventType.OPEN_KAN,
                    ):
                        return prev.who() == my_idx
        return False

    def calculate_penalty_reward(
        self, obs: mjx.Observation, opponent_score: Optional[float] = None
    ) -> float:
        """
        💀 放銃防守懲罰 r_penalty（Mode-Aware ——「安全氣囊」）

        攻擊模式：penalty_weight=1.8 → 標準避險警覺
        防守模式：penalty_weight=6.0 → 威懾力暴增，強烈逼迫 Policy 選安全牌

        公式：
            if 放銃:  return -|opponent_score| * penalty_weight / score_norm_factor
            else:     return 0.0
        """
        is_houjuu = self.check_houjuu(obs)
        if is_houjuu and opponent_score is not None:
            return -abs(opponent_score) * self._pw / self.score_norm_factor  # 動態讀取 penalty_weight
        return 0.0

    def calculate_progression_reward(
        self, prev_shanten: Optional[int], curr_shanten: int
    ) -> float:
        """
        📉 向聽進展獎勵 r_progression（Mode-Aware）

        攻擊模式：progression_weight=0.05 → 每次向聽數 -1 給予顯著正向梯度信號
        防守模式：progression_weight=0.005 → 大幅衰減，防止因進展誘惑而冒險

        公式：
            delta = prev_shanten - curr_shanten
            return delta * progression_weight
        """
        if prev_shanten is None:
            return 0.0
        delta = prev_shanten - curr_shanten  # >0 表示向聽改善；<0 表示退後
        return delta * self._pgw  # 動態讀取 progression_weight

    # ==================================================================
    # 🧮 複合獎勵聚合（保留完整五項 reward 元件，僅前四項受模式影響）
    # ==================================================================

    def compute_step_reward(
        self,
        obs: mjx.Observation,
        final_hand_34: Optional[np.ndarray] = None,
        score_delta: Optional[int] = None,
        opponent_score: Optional[float] = None,
    ) -> float:
        """
        將所有即時與延遲獎勵聚合為單步獎勵標量。

        五項獎勵元件：
            1. r_potential   — 向聽潛力（Mode-Aware）
            2. r_dora        — 寶牌留存（Mode-Aware）
            3. r_penalty     — 放銃懲罰（Mode-Aware，僅終局觸發）
            4. r_backward    — 終局後向得分（Mode-Invariant，僅胡牌觸發）
            5. r_progression — 由 runner.py 在迴圈中單獨計算

        注意：此函數目前保留完整性，但 runner.py 實際上是分開呼叫四個核心函數
              以便靈活組合。compute_step_reward 用於向後相容。
        """
        r_potential = self.calculate_potential_reward(obs)
        r_dora = self.calculate_dora_potential_reward(obs)
        r_penalty = self.calculate_penalty_reward(obs, opponent_score)

        r_backward = 0.0
        if final_hand_34 is not None and score_delta is not None:
            current_hand = self._get_current_hand_34(obs)
            r_backward = self.calculate_backward_reward(
                final_hand_34, score_delta, current_hand
            )

        return r_potential + r_dora + r_backward + r_penalty


# ======================================================================
# 🏭 工廠函數（向後相容：現有程式碼無需修改 import 方式）
# ======================================================================

def create_default_calculator() -> MahjongRewardCalculator:
    """
    建立預設的 MahjongRewardCalculator 實例。

    預設開局為攻擊模式（"attack"），在 runner.py 的 run_match()
    迴圈中會根據局勢（RIICHI 事件）動態切換至防守模式。
    """
    return MahjongRewardCalculator(
        base_exp_score=2000.0,
        score_norm_factor=10000.0,
    )