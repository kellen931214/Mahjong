"""
mortal_agent.py — Mortal 模型包裝器（用於 mjx 自我對弈評估）

將 Mortal（Equim-chan/Mortal）包裝成可注入 SelfPlayRunner 的外部 agent。
透過將 mjx Observation 直接轉換為 mjai 事件流，同步 libriichi PlayerState。

架構（重寫版 - 穩健方案）：
  mjx Observation → 提取公開事件 + 自身摸牌記錄 → mjai JSON 事件
  → PlayerState.update() → encode_obs() → MortalEngine.react_batch()
  → Mortal 46-dim action → mjx Action

關鍵設計決策：
  - 只發送此玩家能合法看到的 mjai 事件（符合實際牌局資訊）
  - 自己摸牌：從 obs.draws()[-1] 提取
  - 他人摸牌：不發送 tsumo（我們不知道他摸什麼牌）
  - 碰/吃/槓：從 evt.open() 提取完整 tile 資訊
  - 每局牌（kyoku）開始時重建 PlayerState 並重播歷史事件
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import traceback

import numpy as np
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import mjx

# Mortal Python 模組路徑（mortal/model.py, mortal/engine.py 所在）
_MORTAL_PKG_DIR = Path(__file__).resolve().parent.parent / "Mortal" / "mortal"
if str(_MORTAL_PKG_DIR.parent) not in sys.path:
    sys.path.append(str(_MORTAL_PKG_DIR.parent))

# ── mjai tile index ↔ string mapping ──
_TILE_IDX_TO_MJAI = [
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    "E","S","W","N","P","F","C",
]
_TILE_TYPE_TO_MJAI = _TILE_IDX_TO_MJAI  # same mapping for mjx TileType 0-33


def _mjx_tile_to_mjai(tile) -> str:
    """將 mjx Tile 物件轉為 mjai tile 字串，如 '1m', '5sr'"""
    try:
        tt = tile.type() if hasattr(tile, 'type') else tile
    except Exception:
        tt = tile
    if isinstance(tt, int) and 0 <= tt < len(_TILE_IDX_TO_MJAI):
        return _TILE_IDX_TO_MJAI[tt]
    return "?"


def _mjx_open_to_mjai_tiles(open_obj) -> tuple:
    """
    從 mjx Open 物件提取 mjai 所需的 tile 資訊。
    回傳 (pai: str, consumed: List[str], target_relative: int)
    
    mjx Open API:
      - tiles() → 全部牌（含手牌和被吃/碰/槓的牌）
      - tiles_from_hand() → 來自手上的牌（消耗的牌）
      - stolen_tile() → 被吃/碰/大明槓的那張牌
      - last_tile() → 加槓的最後一張牌
      - steal_from() → RelativePlayerIdx（從誰那裏拿的）
      - event_type() → CHI/PON/CLOSED_KAN/OPEN_KAN/ADDED_KAN
    """
    pai = "?"
    consumed = []
    target_rel = None
    
    if open_obj is None:
        return pai, consumed, target_rel

    try:
        # 被偷的那張牌（chi/pon/daiminkan 的目標牌）
        stolen = open_obj.stolen_tile() if hasattr(open_obj, 'stolen_tile') else None
        if stolen:
            pai = _mjx_tile_to_mjai(stolen)
    except Exception:
        pass

    try:
        # 從手上消耗的牌
        hand_tiles = open_obj.tiles_from_hand() if hasattr(open_obj, 'tiles_from_hand') else []
        consumed = [_mjx_tile_to_mjai(t) for t in hand_tiles]
    except Exception:
        pass

    try:
        target_rel = open_obj.steal_from() if hasattr(open_obj, 'steal_from') else None
    except Exception:
        pass

    return pai, consumed, target_rel


class MortalAgent:
    """
    Mortal AI 代理，可直接嵌入 mjx 自我對弈流程。
    每局牌（hand）開始時重建 PlayerState 並同步公開事件。
    """

    def __init__(
        self,
        weights_path: str,
        player_id: int = 0,
        device: str = "cuda",
        enable_amp: bool = True,
        enable_quick_eval: bool = True,
        enable_rule_based_agari_guard: bool = True,
    ):
        assert 0 <= player_id <= 3, f"player_id 必須在 0-3，收到: {player_id}"
        self.player_id = player_id
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        if str(self.device) != device:
            print(f"[MortalAgent] 警告：CUDA 不可用，改用 {self.device}")

        # ── 載入 Mortal 模型 ──
        from mortal.model import Brain, DQN
        from mortal.engine import MortalEngine

        wp = Path(weights_path)
        if not wp.exists():
            raise FileNotFoundError(f"Mortal 權重檔不存在: {wp}")
        print(f"[MortalAgent p{player_id}] 載入權重: {wp}")
        state = torch.load(wp, map_location=torch.device("cpu"), weights_only=False)
        cfg = state["config"]
        version = cfg["control"].get("version", 4)
        conv_channels = cfg["resnet"]["conv_channels"]
        num_blocks = cfg["resnet"]["num_blocks"]

        mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        dqn = DQN(version=version).eval()
        mortal.load_state_dict(state["mortal"])
        dqn.load_state_dict(state["current_dqn"])

        self.engine = MortalEngine(
            mortal, dqn,
            is_oracle=False,
            version=version,
            device=self.device,
            enable_amp=enable_amp,
            enable_quick_eval=enable_quick_eval,
            enable_rule_based_agari_guard=enable_rule_based_agari_guard,
            name=f"mortal_p{player_id}",
        )

        # ── 初始化 libriichi PlayerState ──
        import libriichi as _lib
        self._lib = _lib
        self._state = _lib.state.PlayerState(player_id)

        # ── 事件追蹤 ──
        # 記錄「已處理過的 mjx event 總數」用於增量
        self._processed_mjx_event_count = 0
        # 記錄「每個 kyoku 開頭的事件計數」用於判斷是否新局
        self._kyoku_start_event_idx = 0
        self._game_started = False
        self._kyoku_started = False

        # 當前局的 mjai 事件歷史（每局重建時重播用）
        self._hand_mjai_events: List[str] = []
        # 上一個 observation 用於判斷是否新局
        self._prev_round = -1

        print(f"[MortalAgent p{player_id}] 初始化完成 (v{version}, device={self.device})")

    # ========================================================================
    #  公開介面
    # ========================================================================

    def reset(self) -> None:
        """重置內部狀態。"""
        self._state = self._lib.state.PlayerState(self.player_id)
        self._processed_mjx_event_count = 0
        self._kyoku_start_event_idx = 0
        self._game_started = False
        self._kyoku_started = False
        self._hand_mjai_events = []
        self._prev_round = -1

    def act(self, obs: "mjx.Observation") -> "mjx.Action":
        """
        根據 mjx observation 返回 Mortal 的動作。
        """
        # ── 1. 確保同步到當前局的最新狀態 ──
        self._sync_state(obs)

        # ── 2. 編碼 observation → 呼叫 MortalEngine ──
        obs_arr, mask_arr = self._state.encode_obs(self.engine.version, at_kan_select=False)
        action_list, _, _, _ = self.engine.react_batch([obs_arr], [mask_arr], None)
        mortal_action = action_list[0]

        # ── 3. 轉換為 mjx Action ──
        return self._convert_action(mortal_action, obs)

    # ========================================================================
    #  狀態同步核心
    # ========================================================================

    def _sync_state(self, obs: "mjx.Observation") -> None:
        """
        將 mjx Observation 轉換為 mjai 事件並同步 PlayerState。

        策略：
          - 偵測新局（round 變化）→ 重建 PlayerState + 重播歷史事件
          - 否則 → 增量更新
        """
        mjx_events = list(obs.events()) if hasattr(obs, "events") and obs.events() else []

        # 偵測新局：round 改變 或 events 從頭開始
        try:
            current_round = obs.round()
        except Exception:
            current_round = -1

        is_new_kyoku = (current_round != self._prev_round and self._prev_round >= 0) or \
                       len(mjx_events) == 0 or \
                       (len(mjx_events) > 0 and self._processed_mjx_event_count == 0)

        if is_new_kyoku:
            self._rebuild_state(obs, mjx_events)
        else:
            self._incremental_update(obs, mjx_events)

        self._prev_round = current_round

    def _rebuild_state(self, obs: "mjx.Observation", mjx_events: list) -> None:
        """
        重建 PlayerState：從頭建立 start_game + start_kyoku，
        然後重播所有已發生的公開事件。
        """
        # ── 1. 發送 start_game + start_kyoku ──
        self._state = self._lib.state.PlayerState(self.player_id)
        self._game_started = False
        start_events = self._build_start_events(obs)
        for evt_json in start_events:
            self._state.update(evt_json)
        if not self._game_started:
            self._game_started = True

        # ── 2. 收集本局所有可轉換的公開事件 ──
        self._hand_mjai_events = []
        self._processed_mjx_event_count = 0

        mjai_batch = []
        for evt in mjx_events:
            evt_json = self._mjx_event_to_mjai(evt, obs)
            if evt_json:
                mjai_batch.append(evt_json)
                self._hand_mjai_events.append(evt_json)
            self._processed_mjx_event_count += 1

        # ── 3. 重播所有公開事件 ──
        for evt_json in mjai_batch:
            try:
                self._state.update(evt_json)
            except Exception:
                pass  # 重播時忽略個別事件錯誤

        self._kyoku_started = True

    def _incremental_update(self, obs: "mjx.Observation", mjx_events: list) -> None:
        """
        增量更新：處理自上次同步後新增的 mjx events。
        """
        new_events = mjx_events[self._processed_mjx_event_count:]
        if not new_events:
            return

        for evt in new_events:
            evt_json = self._mjx_event_to_mjai(evt, obs)
            if evt_json:
                try:
                    self._state.update(evt_json)
                    self._hand_mjai_events.append(evt_json)
                except Exception:
                    pass  # 單個事件失敗不中斷
            self._processed_mjx_event_count += 1

    # ========================================================================
    #  start_game / start_kyoku 事件構建
    # ========================================================================

    def _build_start_events(self, obs: "mjx.Observation") -> List[str]:
        """產生 start_game 和 start_kyoku mjai JSON。"""
        from mjx.const import TileType

        events = []

        # start_game
        events.append(json.dumps({
            "type": "start_game",
            "names": [f"player_{i}" for i in range(4)],
        }))

        # ── 提取局資訊 ──
        try:
            scores = list(obs.tens())
        except Exception:
            scores = [25000] * 4

        try:
            oya = obs.dealer()
        except Exception:
            oya = 0

        try:
            round_num = obs.round()
        except Exception:
            round_num = 0

        try:
            bakaze = TileType.EW if round_num < 4 else TileType.SW
        except Exception:
            bakaze = TileType.EW

        kyoku = (round_num % 4) + 1

        try:
            honba = obs.honba()
        except Exception:
            honba = 0

        try:
            kyotaku = obs.kyotaku()
        except Exception:
            kyotaku = 0

        try:
            doras = obs.doras()
            dora_marker = doras[0] if len(doras) > 0 else TileType.M1
        except Exception:
            dora_marker = TileType.M1

        # ── 構建 tehais ──
        # 自己手牌從 curr_hand() 提取，其他玩家用 "?"（Mortal 能處理）
        tehais = [["?"] * 13 for _ in range(4)]
        try:
            hand = obs.curr_hand()
            my_tiles = [_mjx_tile_to_mjai(t) for t in hand.closed_tiles()]
            while len(my_tiles) < 13:
                my_tiles.append("?")
            tehais[self.player_id] = my_tiles[:13]
        except Exception:
            pass

        bakaze_str = _TILE_TYPE_TO_MJAI[bakaze] if isinstance(bakaze, int) and 0 <= bakaze < 34 else "E"
        dora_marker_str = _TILE_TYPE_TO_MJAI[dora_marker] if isinstance(dora_marker, int) and 0 <= dora_marker < 34 else "1m"

        start_kyoku = {
            "type": "start_kyoku",
            "bakaze": bakaze_str,
            "dora_marker": dora_marker_str,
            "kyoku": kyoku,
            "honba": honba,
            "kyotaku": kyotaku,
            "oya": oya,
            "scores": scores,
            "tehais": tehais,
        }
        events.append(json.dumps(start_kyoku))

        return events

    # ========================================================================
    #  mjx Event → mjai JSON 事件轉換（核心修正）
    # ========================================================================

    def _mjx_event_to_mjai(self, evt, obs) -> Optional[str]:
        """
        將單個 mjx Event 轉換為 mjai JSON 字串。
        
        核心修正：
          - DRAW: 使用 obs.draws()[-1] 取得摸到的牌
          - CHI/PON/OPEN_KAN: 從 evt.open() 提取完整 consumed/stolen tile 資訊
          - target: 從 evt.open().steal_from() 計算絕對玩家 ID
          - DISCARD/TSUMOGIRI: 使用 evt.tile()（已正確）
        """
        from mjx.const import EventType as ET

        try:
            evt_type = evt.type()
            actor = evt.who() if hasattr(evt, "who") else None
            if actor is None:
                return None

            if evt_type == ET.DRAW:
                # DRAW 事件自己摸牌時可從 draws 取得；別人摸牌時跳過
                if actor == self.player_id:
                    try:
                        draws = obs.draws()
                        tile = draws[-1] if draws else None
                    except Exception:
                        tile = None
                    pai = _mjx_tile_to_mjai(tile) if tile else "?"
                else:
                    # 別人的摸牌：跳過（我們不知道他摸什麼）
                    # 但為了 PlayerState 正確性，仍記錄摸牌事件
                    # 使用 "?" 在某些情況可能導致後續 discard 出錯
                    # 因此採用「記錄事件但 tile 設為 unknown」
                    # PlayerState 收到 "?" 的 tsumo 後，會等待該玩家的 discard 來辨識 tile
                    return None  # 跳過未知的他人摸牌

                return json.dumps({"type": "tsumo", "actor": actor, "pai": pai})

            elif evt_type == ET.DISCARD:
                tile = evt.tile()
                pai = _mjx_tile_to_mjai(tile) if tile else "?"
                return json.dumps({
                    "type": "dahai", "actor": actor, "pai": pai, "tsumogiri": False,
                })

            elif evt_type == ET.TSUMOGIRI:
                tile = evt.tile()
                pai = _mjx_tile_to_mjai(tile) if tile else "?"
                return json.dumps({
                    "type": "dahai", "actor": actor, "pai": pai, "tsumogiri": True,
                })

            elif evt_type == ET.RIICHI:
                return json.dumps({"type": "reach", "actor": actor})

            elif evt_type in (ET.CHI, ET.PON, ET.OPEN_KAN):
                open_obj = evt.open() if hasattr(evt, "open") else None
                pai, consumed, rel_target = _mjx_open_to_mjai_tiles(open_obj)

                # 計算絕對 target（被拿牌的玩家）
                # steal_from 返回 RelativePlayerIdx: 0=self, 1=right, 2=center, 3=left
                if rel_target is not None:
                    # 相對位置 → 絕對位置
                    # RelativePlayerIdx: 0=self, 1=下家(right), 2=對面(center), 3=上家(left)
                    rel_map = {0: actor, 1: (actor + 1) % 4, 2: (actor + 2) % 4, 3: (actor + 3) % 4}
                    try:
                        target = rel_map[int(rel_target)]
                    except (ValueError, TypeError):
                        target = actor  # fallback
                else:
                    target = actor  # fallback

                if evt_type == ET.CHI:
                    return json.dumps({
                        "type": "chi", "actor": actor, "target": target,
                        "pai": pai, "consumed": consumed if consumed else ["?", "?"],
                    })
                elif evt_type == ET.PON:
                    return json.dumps({
                        "type": "pon", "actor": actor, "target": target,
                        "pai": pai, "consumed": consumed if consumed else ["?", "?"],
                    })
                else:  # OPEN_KAN
                    return json.dumps({
                        "type": "daiminkan", "actor": actor, "target": target,
                        "pai": pai, "consumed": consumed if consumed else ["?", "?", "?"],
                    })

            elif evt_type == ET.CLOSED_KAN:
                open_obj = evt.open() if hasattr(evt, "open") else None
                pai, consumed, _ = _mjx_open_to_mjai_tiles(open_obj)
                if consumed and len(consumed) == 4:
                    return json.dumps({"type": "ankan", "actor": actor, "consumed": consumed})
                # fallback: 從 tiles() 提取
                try:
                    all_tiles = [_mjx_tile_to_mjai(t) for t in open_obj.tiles()]
                    return json.dumps({"type": "ankan", "actor": actor, "consumed": all_tiles[:4]})
                except Exception:
                    return json.dumps({"type": "ankan", "actor": actor, "consumed": ["?","?","?","?"]})

            elif evt_type == ET.ADDED_KAN:
                open_obj = evt.open() if hasattr(evt, "open") else None
                pai, consumed, _ = _mjx_open_to_mjai_tiles(open_obj)
                return json.dumps({
                    "type": "kakan", "actor": actor,
                    "pai": pai, "consumed": consumed if consumed else ["?", "?", "?"],
                })

            elif evt_type == ET.TSUMO:
                return json.dumps({"type": "hora", "actor": actor, "target": actor})

            elif evt_type == ET.RON:
                # RON 的 target 是被榮和的玩家。從 events history 推斷
                # 由於 mjx event 不直接提供 target，假設 target 是上一輪出牌的玩家
                return json.dumps({"type": "hora", "actor": actor, "target": (actor + 1) % 4})

            elif evt_type == ET.NEW_DORA:
                # 嘗試從 obs 提取新 dora
                try:
                    all_doras = obs.doras()
                    dora = all_doras[-1] if all_doras else None
                except Exception:
                    dora = None
                return json.dumps({
                    "type": "dora",
                    "dora_marker": _mjx_tile_to_mjai(dora) if dora else "?",
                })

            elif evt_type in (ET.ABORTIVE_DRAW_NORMAL, ET.ABORTIVE_DRAW_NAGASHI_MANGAN,
                              ET.ABORTIVE_DRAW_FOUR_RIICHIS, ET.ABORTIVE_DRAW_THREE_RONS,
                              ET.ABORTIVE_DRAW_FOUR_KANS, ET.ABORTIVE_DRAW_FOUR_WINDS,
                              ET.ABORTIVE_DRAW_NINE_TERMINALS,):
                return json.dumps({"type": "ryukyoku"})

            elif evt_type in (ET.RIICHI_SCORE_CHANGE,):
                return None  # Mortal 不需要

            else:
                return None

        except Exception as e:
            print(f"[MortalAgent] _mjx_event_to_mjai 錯誤: {e}")
            return None

    # ========================================================================
    #  Mortal 46-dim action → mjx 181-dim action 轉換
    # ========================================================================

    def _convert_action(self, mortal_action: int, obs: "mjx.Observation") -> "mjx.Action":
        """
        將 Mortal 的 46 維動作索引轉換為 mjx Action 物件。

        Mortal 動作空間（libriichi/src/agent/mortal.rs）:
          0-36:  切牌（tile_id 0-33 + 3 種赤牌特殊處理）
          37:    立直
          38-40: 吃（低/中/高）
          41:    碰
          42:    槓（決定）
          43:    和了（自摸/榮和）
          44:    流局（九種九牌）
          45:    pass

        mjx 動作空間（evaluate.py 的 ACTION_BINS）:
          0-73:   切牌（DISCARD 0-36 + TSUMOGIRI 37-73）
          74-103: 吃
          104-140:碰
          141-174:槓
          175:    自摸
          176:    榮和
          177:    立直
          178:    九種九牌
          179:    pass
          180:    dummy
        """
        from mjx.action import Action

        legal_actions = obs.legal_actions()

        if mortal_action <= 36:
            # 切牌：Mortal 用 tile_id (0-33)，mjx 用 discard action (0-36 或 tsumogiri 37-73)
            tile_id = mortal_action if mortal_action < 34 else (mortal_action - 34)
            # 先在 DISCARD (0-36) 中找匹配的 tile
            for act in legal_actions:
                act_idx = act.to_idx()
                if 0 <= act_idx <= 36:
                    act_tile = act.tile()
                    if act_tile is not None and hasattr(act_tile, 'type'):
                        if act_tile.type() == tile_id:
                            return act
            # fallback: 任意切牌動作
            for act in legal_actions:
                if 0 <= act.to_idx() <= 73:
                    return act

        elif mortal_action == 37:  # 立直
            for act in legal_actions:
                if act.to_idx() == 177:
                    return act

        elif 38 <= mortal_action <= 40:  # 吃
            for act in legal_actions:
                if 74 <= act.to_idx() <= 103:
                    return act

        elif mortal_action == 41:  # 碰
            for act in legal_actions:
                if 104 <= act.to_idx() <= 140:
                    return act

        elif mortal_action == 42:  # 槓
            for act in legal_actions:
                if 141 <= act.to_idx() <= 174:
                    return act

        elif mortal_action == 43:  # 和了
            for act in legal_actions:
                if act.to_idx() in (175, 176):
                    return act

        elif mortal_action == 44:  # 流局
            for act in legal_actions:
                if act.to_idx() == 178:
                    return act

        elif mortal_action == 45:  # pass
            for act in legal_actions:
                if act.to_idx() == 179:
                    return act

        # ── fallback ──
        print(f"[MortalAgent] 警告：無法映射 Mortal action {mortal_action}，使用 fallback")
        return legal_actions[0]

    # ========================================================================
    #  輔助
    # ========================================================================

    def close(self) -> None:
        self.engine = None
        self._state = None