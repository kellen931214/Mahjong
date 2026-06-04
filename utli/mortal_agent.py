"""
mortal_agent.py — Mortal 模型包裝器（用於 mjx 自我對弈評估）

將 Mortal（Equim-chan/Mortal）包裝成可注入 SelfPlayRunner 的外部 agent。
透過 mjai 事件流橋接 mjx 環境與 Mortal 的 Rust libriichi 遊戲引擎。

架構：
  mjx Observation → mjai 事件流轉換 → libriichi.PlayerState.update()
  → PlayerState.encode_obs() → MortalEngine.react_batch()
  → Mortal 46-dim action → mjx Action

使用方式:
  agent = MortalAgent(weights_path="mortal.pth", player_id=0, device="cuda")
  action = agent.act(obs)  # obs 為 mjx.Observation
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import traceback

import numpy as np
import torch

# Mortal Python 模組路徑（mortal/model.py, mortal/engine.py 所在）
_MORTAL_PKG_DIR = Path(__file__).resolve().parent.parent / "Mortal" / "mortal"
# libriichi 已透過 maturin 安裝到 site-packages，無需特殊路徑處理
# 確保 Mortal 套件可被導入，但不遮蔽已安裝的 libriichi
if str(_MORTAL_PKG_DIR.parent) not in sys.path:
    sys.path.append(str(_MORTAL_PKG_DIR.parent))


class MortalAgent:
    """
    Mortal AI 代理，可直接嵌入 mjx 自我對弈流程。

    注意事項：
    - 需要 Mortal 預訓練權重檔（.pth），內含 "mortal" (Brain) 與 "current_dqn" (DQN) 兩個 key
    - 使用 libriichi 的 PlayerState 來維護遊戲狀態，與 mjx 平行運行
    - 每局開始時需呼叫 reset()；每步收到 observation 時呼叫 act()
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
        """
        Args:
            weights_path: Mortal 權重檔路徑（.pth）
            player_id: 玩家編號 0-3
            device: 推理裝置
            enable_amp: 啟用 AMP 加速
            enable_quick_eval: 啟用快速評估（單一合法動作時直接選取，不跑模型）
            enable_rule_based_agari_guard: 啟用規則基礎的和了守衛
        """
        assert 0 <= player_id <= 3, f"player_id 必須在 0-3，收到: {player_id}"
        self.player_id = player_id
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        if str(self.device) != device:
            print(f"[MortalAgent] 警告：CUDA 不可用，改用 {self.device}")

        # ── 延遲導入 Mortal 模組（避免汙染全域）──
        from mortal.model import Brain, DQN
        from mortal.engine import MortalEngine

        # ── 載入權重 ──
        wp = Path(weights_path)
        if not wp.exists():
            raise FileNotFoundError(f"Mortal 權重檔不存在: {wp}")
        print(f"[MortalAgent] 載入權重: {wp}")
        state = torch.load(wp, map_location=torch.device("cpu"), weights_only=False)
        cfg = state["config"]
        version = cfg["control"].get("version", 4)
        conv_channels = cfg["resnet"]["conv_channels"]
        num_blocks = cfg["resnet"]["num_blocks"]
        print(f"[MortalAgent] model v{version}, blocks={num_blocks}, channels={conv_channels}")

        # 建立模型
        mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        dqn = DQN(version=version).eval()
        mortal.load_state_dict(state["mortal"])
        dqn.load_state_dict(state["current_dqn"])

        # 建立 MortalEngine
        self.engine = MortalEngine(
            mortal,
            dqn,
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
        self._state = _lib.state.PlayerState(player_id)

        # 遊戲事件緩衝區（累積 mjai 事件以同步狀態）
        self._events: List[str] = []
        self._kyoku_started = False
        self._game_started = False

        # 前一步 observation（用於 delta 事件推斷）
        self._prev_obs = None

        print(f"[MortalAgent] 初始化完成 (pid={player_id}, device={self.device})")

    # ========================================================================
    #  公開介面
    # ========================================================================

    def reset(self) -> None:
        """重置內部狀態，在新一局開始時呼叫。"""
        self._state = type(self._state)(self.player_id)  # 重建 PlayerState
        self._events = []
        self._kyoku_started = False
        self._game_started = False
        self._prev_obs = None

    def act(self, obs: "mjx.Observation") -> "mjx.Action":
        """
        根據 mjx observation 返回 Mortal 的動作。

        Args:
            obs: mjx 的 Observation 物件

        Returns:
            mjx.Action: Mortal 選擇的合法動作
        """
        # ── 1. 取得合法動作列表 ──
        legal_actions = obs.legal_actions()
        if len(legal_actions) == 0:
            raise RuntimeError(f"[MortalAgent] player {self.player_id} 沒有合法動作")

        # ── 2. 同步 mjai 事件 → 更新 PlayerState ──
        self._sync_state(obs)

        # ── 3. 呼叫 MortalEngine.react_batch ──
        # 編碼 observation
        import libriichi as _lib
        PS = _lib.state.PlayerState
        obs_arr, mask_arr = self._state.encode_obs(self.engine.version, at_kan_select=False)

        # Mortal 46 維動作空間的 mask
        # 轉換成 list 格式給 react_batch
        action, q_values, masks_recv, is_greedy = self.engine.react_batch(
            [obs_arr], [mask_arr], None
        )
        mortal_action_idx = action[0]  # 0~45

        # ── 4. Mortal 46-dim action → mjx Action ──
        mjx_action = self._convert_action(mortal_action_idx, obs)

        # 記錄前一步 obs
        self._prev_obs = obs

        return mjx_action

    # ========================================================================
    #  內部方法：狀態同步
    # ========================================================================

    def _sync_state(self, obs: "mjx.Observation") -> None:
        """
        從 mjx Observation 重建 mjai 事件流，同步 libriichi PlayerState。

        策略：
        - 首步：產生 start_game + start_kyoku 事件（從 obs 提取初始資訊）
        - 後續：根據 events 增量產生 tsumo / dahai / chi / pon / kan / reach / hora / dora 等事件
        """
        from mjx.const import EventType, ActionType

        mjx_events = obs.events() if hasattr(obs, "events") else []

        # ── 起始事件 ──
        if not self._kyoku_started:
            self._handle_start_events(obs)
            self._kyoku_started = True
            self._prev_obs = obs
            return

        # ── 增量事件 ──
        # 從 mjx events 中提取新發生的事件（比較新舊 events 列表）
        prev_event_count = len(self._events)
        self._process_mjx_events(obs, mjx_events)

    def _handle_start_events(self, obs: "mjx.Observation") -> None:
        """產生 start_game / start_kyoku 事件。"""
        from mjx.const import TileType

        obs_proto = obs.to_proto()

        # 從 observation 提取局資訊
        # mjx observation proto: public_observation, private_observation 等
        public = obs_proto.public_observation if obs_proto.HasField("public_observation") else None
        private = obs_proto.private_observation if obs_proto.HasField("private_observation") else None
        if public is None and obs_proto.HasField("publicObservation"):
            public = obs_proto.publicObservation  # 相容不同 proto 命名

        # 提取初始手牌（從 curr_hand 提取，但 start_kyoku 需要 tehais[4][13]）
        # 由於 mjx 不直接暴露初始手牌，我們用 observation 中可取得的資訊近似
        try:
            hand = obs.curr_hand()
            closed_tiles = hand.closed_tiles()  # 手牌列表
        except Exception:
            closed_tiles = []

        # ── 嘗試從 proto 提取局資訊 ──
        # init_score = 25000
        # kyoku (0=東一, ..., 7=南四)
        # honba, kyotaku, oya, bakaze, dora_marker

        try:
            # 從 proto 提取分數
            if public is not None:
                scores = list(public.scores) if hasattr(public, "scores") else [25000] * 4
            else:
                try:
                    scores = list(obs.tens())
                except Exception:
                    scores = [25000] * 4
        except Exception:
            scores = [25000] * 4

        # 提取 oya（親家）
        try:
            oya = obs_proto.public_observation.init_score.tens  # 這是錯的，用 dealer()
            # 正確方式
            from mjx.observation import Observation
            oya = obs.dealer()
        except Exception:
            oya = 0

        # 提取 round number
        try:
            round_num = obs.round()
        except Exception:
            round_num = 0

        # 提取 doras
        try:
            doras = obs.doras()
            dora_marker = doras[0] if len(doras) > 0 else TileType.M1
        except Exception:
            dora_marker = TileType.M1

        # 提取 bakaze（場風）
        bakaze_map = {
            0: TileType.EW, 1: TileType.EW, 2: TileType.EW, 3: TileType.EW,
            4: TileType.SW, 5: TileType.SW, 6: TileType.SW, 7: TileType.SW,
        }
        bakaze = bakaze_map.get(round_num, TileType.EW)

        # 提取 kyoku (1-indexed)
        kyoku = (round_num % 4) + 1

        # 提取 honba
        try:
            honba = obs.honba()
        except Exception:
            honba = 0

        # 提取 kyotaku
        try:
            kyotaku = obs.kyotaku()
        except Exception:
            kyotaku = 0

        # ── 構建 mjai tile 字串 ──
        def _tile_type_to_mjai_str(tt) -> str:
            """將 mjx TileType (0-33) 轉為 mjai tile 字串（如 '1m', '5sr'）"""
            mjai_map = [
                "1m","2m","3m","4m","5m","6m","7m","8m","9m",
                "1p","2p","3p","4p","5p","6p","7p","8p","9p",
                "1s","2s","3s","4s","5s","6s","7s","8s","9s",
                "E","S","W","N","P","F","C",
            ]
            if isinstance(tt, int) and 0 <= tt < len(mjai_map):
                return mjai_map[tt]
            elif hasattr(tt, 'to_char') or hasattr(tt, 'as_char'):
                return str(tt)
            return "?"

        def _tile_to_mjai_str(tile) -> str:
            """將 mjx Tile 物件轉為 mjai tile 字串"""
            if hasattr(tile, 'type'):
                tt = tile.type()
            elif isinstance(tile, int):
                tt = tile
            else:
                try:
                    tt = int(tile)
                except Exception:
                    return "?"
            return _tile_type_to_mjai_str(tt)

        # ── 產生 mjai JSON 事件 ──
        events = []

        # start_game
        if not self._game_started:
            events.append(json.dumps({
                "type": "start_game",
                "names": [f"player_{i}" for i in range(4)],
            }))
            self._game_started = True

        # 構建初始手牌（tehais）
        # 注意：mjx observation 不直接給出四個人的初始手牌
        # 我們只能給出目前已知的手牌（該玩家的手牌、其他人的公開資訊）
        # Mortal 需要 tehais[4][13] 但不一定需要完整（它會忽略非自己手牌的資訊）
        tehais = [["?"] * 13 for _ in range(4)]
        # 填入自己手牌
        my_tiles = []
        for tile in closed_tiles:
            my_tiles.append(_tile_to_mjai_str(tile))
        # 如果手牌不足 13 張，用 ? 補齊
        while len(my_tiles) < 13:
            my_tiles.append("?")
        tehais[self.player_id] = my_tiles[:13]

        start_kyoku = {
            "type": "start_kyoku",
            "bakaze": _tile_type_to_mjai_str(bakaze),
            "dora_marker": _tile_type_to_mjai_str(dora_marker),
            "kyoku": kyoku,
            "honba": honba,
            "kyotaku": kyotaku,
            "oya": oya,
            "scores": scores,
            "tehais": tehais,
        }
        events.append(json.dumps(start_kyoku))

        # 將事件送入 PlayerState
        for evt_json in events:
            self._state.update(evt_json)
            self._events.append(evt_json)

    def _process_mjx_events(self, obs: "mjx.Observation", mjx_events) -> None:
        """
        處理增量 mjx events，轉換為 mjai JSON 事件並更新 PlayerState。
        """
        from mjx.const import EventType, ActionType

        prev_count = len(self._events)
        new_events = []

        for evt in mjx_events[prev_count:] if len(mjx_events) > prev_count else []:
            try:
                evt_type = evt.type()
                evt_json = self._mjx_event_to_mjai(evt, obs)
                if evt_json:
                    new_events.append(evt_json)
            except Exception as e:
                print(f"[MortalAgent] 警告：無法轉換 mjx event: {e}")
                continue

        for evt_json in new_events:
            try:
                self._state.update(evt_json)
                self._events.append(evt_json)
            except Exception as e:
                print(f"[MortalAgent] 警告：PlayerState.update 失敗: {e}, event={evt_json}")

    def _mjx_event_to_mjai(self, evt, obs) -> Optional[str]:
        """將單個 mjx Event 轉換為 mjai JSON 字串。"""
        from mjx.const import EventType as ET

        def _tile_str(tile) -> str:
            """將 mjx Tile 轉為 mjai tile 字串"""
            mjai_map = [
                "1m","2m","3m","4m","5m","6m","7m","8m","9m",
                "1p","2p","3p","4p","5p","6p","7p","8p","9p",
                "1s","2s","3s","4s","5s","6s","7s","8s","9s",
                "E","S","W","N","P","F","C",
            ]
            try:
                tt = evt.tile().type() if hasattr(evt, "tile") and evt.tile() else None
                if tt is not None and 0 <= tt < len(mjai_map):
                    return mjai_map[tt]
                # fallback: 從 tile 直接轉
                if tile is not None:
                    if hasattr(tile, 'type') and 0 <= tile.type() < 34:
                        return mjai_map[tile.type()]
            except Exception:
                pass
            return "?"

        try:
            evt_type = evt.type()
            actor = evt.who() if hasattr(evt, "who") else None

            if evt_type == ET.DRAW:
                tile = evt.tile() if hasattr(evt, "tile") else None
                return json.dumps({
                    "type": "tsumo",
                    "actor": actor,
                    "pai": _tile_str(tile) if tile else "?",
                })
            elif evt_type == ET.DISCARD:
                tile = evt.tile() if hasattr(evt, "tile") else None
                return json.dumps({
                    "type": "dahai",
                    "actor": actor,
                    "pai": _tile_str(tile) if tile else "?",
                    "tsumogiri": False,
                })
            elif evt_type == ET.TSUMOGIRI:
                tile = evt.tile() if hasattr(evt, "tile") else None
                return json.dumps({
                    "type": "dahai",
                    "actor": actor,
                    "pai": _tile_str(tile) if tile else "?",
                    "tsumogiri": True,
                })
            elif evt_type == ET.RIICHI:
                return json.dumps({"type": "reach", "actor": actor})
            elif evt_type == ET.CHI:
                return json.dumps({
                    "type": "chi",
                    "actor": actor,
                    "pai": "?",
                    "target": getattr(evt, "target", actor),
                    "consumed": ["?", "?"],
                })
            elif evt_type == ET.PON:
                return json.dumps({
                    "type": "pon",
                    "actor": actor,
                    "pai": "?",
                    "target": getattr(evt, "target", actor),
                    "consumed": ["?", "?"],
                })
            elif evt_type == ET.CLOSED_KAN:
                return json.dumps({
                    "type": "ankan",
                    "actor": actor,
                    "consumed": ["?", "?", "?", "?"],
                })
            elif evt_type == ET.OPEN_KAN:
                return json.dumps({
                    "type": "daiminkan",
                    "actor": actor,
                    "pai": "?",
                    "target": getattr(evt, "target", actor),
                    "consumed": ["?", "?", "?"],
                })
            elif evt_type == ET.ADDED_KAN:
                return json.dumps({
                    "type": "kakan",
                    "actor": actor,
                    "pai": "?",
                    "consumed": ["?", "?", "?"],
                })
            elif evt_type == ET.TSUMO:
                return json.dumps({
                    "type": "hora",
                    "actor": actor,
                    "target": actor,
                })
            elif evt_type == ET.RON:
                return json.dumps({
                    "type": "hora",
                    "actor": actor,
                    "target": getattr(evt, "target", 0),
                })
            elif evt_type in (ET.RIICHI_SCORE_CHANGE,):
                return None  # Mortal 不需要此事件
            elif evt_type == ET.NEW_DORA:
                return json.dumps({
                    "type": "dora",
                    "dora_marker": _tile_str(None),
                })
            elif evt_type in (ET.ABORTIVE_DRAW_NORMAL, ET.ABORTIVE_DRAW_NAGASHI_MANGAN,
                              ET.ABORTIVE_DRAW_FOUR_RIICHIS, ET.ABORTIVE_DRAW_THREE_RONS,
                              ET.ABORTIVE_DRAW_FOUR_KANS, ET.ABORTIVE_DRAW_FOUR_WINDS,
                              ET.ABORTIVE_DRAW_NINE_TERMINALS):
                return json.dumps({"type": "ryukyoku"})
            else:
                return None
        except Exception as e:
            print(f"[MortalAgent] _mjx_event_to_mjai 錯誤: {e}")
            return None

    # ========================================================================
    #  Mortal 46-dim action → mjx 181-dim action 轉換
    # ========================================================================

    # Mortal action space (46 dims):
    #   0-36:  discard / kan select  (37 dims, tile_id 0-33 + 3 kan-related)
    #   37:    riichi
    #   38-40: chi (low, mid, high)
    #   41:    pon
    #   42:    kan (decide)
    #   43:    agari (tsumo/ron)
    #   44:    ryukyoku
    #   45:    pass (none)

    def _convert_action(self, mortal_action: int, obs: "mjx.Observation") -> "mjx.Action":
        """
        將 Mortal 的 46 維動作索引轉換為 mjx Action 物件。

        Mortal 動作空間對應（參考 libriichi/src/agent/mortal.rs）:
          0-36: 切牌（含赤牌）/ 暗槓/加槓選擇
          37:   立直
          38-40: 吃（低/中/高）
          41:   碰
          42:   槓（決定）
          43:   和了
          44:   流局（九種九牌）
          45:   pass
        """
        from mjx.action import Action
        from mjx.const import ActionType

        legal_actions = obs.legal_actions()
        legal_indices = [
            a.to_idx() if hasattr(a, "to_idx") else int(a)
            for a in legal_actions
        ]

        if mortal_action <= 36:
            # 切牌 (0-33 為 tile_id, 34-36 可能為特殊赤牌處理)
            tile_id = mortal_action if mortal_action < 34 else mortal_action  # approximate
            # 在 mjx 的 181 維中找對應的切牌動作（ID 0-73）
            for act in legal_actions:
                act_idx = act.to_idx()
                if 0 <= act_idx <= 73:
                    # 檢查 tile 是否匹配
                    act_tile = act.tile()
                    if act_tile is not None and hasattr(act_tile, 'type'):
                        if act_tile.type() == tile_id:
                            return act
            # fallback：返回第一個合法的切牌動作
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

        elif mortal_action == 43:  # 和了（自摸/榮和）
            # mjx: 175=tsumo, 176=ron
            for act in legal_actions:
                if act.to_idx() in (175, 176):
                    return act

        elif mortal_action == 44:  # 流局（九種九牌）
            for act in legal_actions:
                if act.to_idx() == 178:
                    return act

        elif mortal_action == 45:  # pass
            for act in legal_actions:
                if act.to_idx() == 179:
                    return act

        # ── fallback：返回第一個合法動作 ──
        print(f"[MortalAgent] 警告：無法映射 Mortal action {mortal_action}，使用 fallback")
        return legal_actions[0]

    # ========================================================================
    #  輔助方法
    # ========================================================================

    def close(self) -> None:
        """釋放資源。"""
        self.engine = None
        self._state = None