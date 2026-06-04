"""
mortal_agent.py — Mortal 模型包裝器

每次決策時建立全新 PlayerState，餵入本局至今的所有 mjai 兼容事件，
呼叫 encode_obs() → MortalEngine.react_batch() → 轉回 mjx Action。

關鍵原則：
  - 每次 act() 使用全新的 PlayerState（避免累積狀態毀損）
  - 所有 mjai JSON 中的 tile 必須是合法字串（"1m" ~ "C"），絕不發送 "?"
  - 本局歷史事件累積，重播時只餵入轉換成功的事件
"""

import json
import sys
import copy
from pathlib import Path
from typing import List, Optional

import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import mjx

_MORTAL_DIR = Path(__file__).resolve().parent.parent / "Mortal"
if str(_MORTAL_DIR) not in sys.path:
    sys.path.append(str(_MORTAL_DIR))

# mjai tile strings for tile type 0-33
_T = [
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    "E","S","W","N","P","F","C",
]


def _tt(tt) -> str:
    """tile type → mjai string, always valid"""
    if isinstance(tt, int) and 0 <= tt < 34:
        return _T[tt]
    try:
        tt = tt.type()
        if 0 <= tt < 34:
            return _T[tt]
    except Exception:
        pass
    return "1m"  # fallback: never return unknown


class MortalAgent:
    """使用全新 PlayerState 每次決策，從 mjx Observation 重建局資訊"""

    def __init__(self, weights_path: str, player_id: int = 0, device: str = "cuda"):
        assert 0 <= player_id <= 3
        self.player_id = player_id
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        import libriichi as _lib
        from mortal.model import Brain, DQN
        from mortal.engine import MortalEngine

        self._lib = _lib
        wp = Path(weights_path)
        state = torch.load(wp, map_location="cpu", weights_only=False)
        cfg = state["config"]
        version = cfg["control"]["version"]
        c = cfg["resnet"]["conv_channels"]
        b = cfg["resnet"]["num_blocks"]

        mortal = Brain(version=version, conv_channels=c, num_blocks=b).eval()
        dqn = DQN(version=version).eval()
        mortal.load_state_dict(state["mortal"])
        dqn.load_state_dict(state["current_dqn"])
        self.engine = MortalEngine(
            mortal, dqn, version=version, is_oracle=False, device=self.device,
            enable_amp=False, enable_quick_eval=True,
            enable_rule_based_agari_guard=True, name=f"m{player_id}",
        )
        self.version = version

        self._events: list = []  # mjx events this hand
        self._mjai_events: List[str] = []  # converted mjai JSON strings
        self._last_round = -1
        self._sent_start = False
        print(f"[MortalAgent p{player_id}] init v{version}")

    def reset(self):
        self._events = []
        self._mjai_events = []
        self._last_round = -1
        self._sent_start = False

    # ── public API ──

    def act(self, obs: "mjx.Observation") -> "mjx.Action":
        legals = obs.legal_actions()
        if not legals:
            raise RuntimeError(f"[MortalAgent p{self.player_id}] no legal actions")

        # ── detect new hand ──
        try: r = obs.round()
        except: r = -1
        if r != self._last_round:
            self._events = []
            self._mjai_events = []
            self._last_round = r

        # ── accumulate new events, convert to mjai ──
        mjx_evts = list(obs.events()) if hasattr(obs, "events") and obs.events() else []
        new = mjx_evts[len(self._events):]
        self._events = mjx_evts
        for evt in new:
            line = self._to_mjai(evt, obs)
            if line:
                self._mjai_events.append(line)

        # ── build fresh PlayerState, replay all valid events ──
        state = self._lib.state.PlayerState(self.player_id)
        try:
            state.update(self._start_game_json())
        except Exception:
            pass
        try:
            state.update(self._start_kyoku_json(obs))
        except Exception:
            pass
        for line in self._mjai_events:
            try:
                state.update(line)
            except Exception:
                pass

        # ── encode + react ──
        try:
            obs_arr, mask = state.encode_obs(self.version, False)
            actions, _, _, _ = self.engine.react_batch([obs_arr], [mask], None)
            mortal_act = actions[0]
        except Exception:
            mortal_act = 0  # fallback

        return self._to_mjx(mortal_act, legals)

    # ── event conversion ──

    def _to_mjai(self, evt, obs) -> Optional[str]:
        from mjx.const import EventType as ET
        try:
            t = evt.type()
            a = evt.who() if hasattr(evt, "who") else None
            if a is None: return None

            if t == ET.DRAW:
                pai = "1m"
                if a == self.player_id:
                    try:
                        draws = obs.draws()
                        if draws: pai = _tt(draws[-1])
                    except: pass
                return json.dumps({"type":"tsumo","actor":a,"pai":pai})

            if t == ET.DISCARD:
                return json.dumps({"type":"dahai","actor":a,"pai":_tt(evt.tile()),"tsumogiri":False})
            if t == ET.TSUMOGIRI:
                return json.dumps({"type":"dahai","actor":a,"pai":_tt(evt.tile()),"tsumogiri":True})
            if t == ET.RIICHI:
                return json.dumps({"type":"reach","actor":a})
            if t == ET.TSUMO:
                return json.dumps({"type":"hora","actor":a,"target":a})
            if t == ET.RON:
                return json.dumps({"type":"hora","actor":a,"target":(a+1)%4})
            if t == ET.NEW_DORA:
                try: d = obs.doras()[-1]
                except: d = 0
                return json.dumps({"type":"dora","dora_marker":_tt(d)})

            if t in (ET.ABORTIVE_DRAW_NORMAL, ET.ABORTIVE_DRAW_NAGASHI_MANGAN,
                     ET.ABORTIVE_DRAW_NINE_TERMINALS, ET.ABORTIVE_DRAW_FOUR_RIICHIS,
                     ET.ABORTIVE_DRAW_THREE_RONS, ET.ABORTIVE_DRAW_FOUR_KANS,
                     ET.ABORTIVE_DRAW_FOUR_WINDS):
                return json.dumps({"type":"ryukyoku"})

            # meld events
            open_obj = evt.open() if hasattr(evt, "open") else None
            if open_obj is None: return None

            try: pai = _tt(open_obj.stolen_tile())
            except: pai = _tt(0)

            consumed = []
            try:
                for tile in open_obj.tiles_from_hand():
                    consumed.append(_tt(tile))
            except: pass

            try:
                rel = int(open_obj.steal_from())
                target = (a + rel) % 4 if rel else a
            except:
                target = a

            if t == ET.CHI:
                if len(consumed) < 2: consumed = [_tt(0), _tt(1)]
                return json.dumps({"type":"chi","actor":a,"target":target,"pai":pai,"consumed":consumed[:2]})
            if t == ET.PON:
                if len(consumed) < 2: consumed = [_tt(0), _tt(0)]
                return json.dumps({"type":"pon","actor":a,"target":target,"pai":pai,"consumed":consumed[:2]})
            if t == ET.OPEN_KAN:
                if len(consumed) < 3: consumed = [_tt(0), _tt(0), _tt(0)]
                return json.dumps({"type":"daiminkan","actor":a,"target":target,"pai":pai,"consumed":consumed[:3]})
            if t == ET.CLOSED_KAN:
                try: consumed = [_tt(t) for t in open_obj.tiles()]
                except: consumed = [_tt(0)]*4
                return json.dumps({"type":"ankan","actor":a,"consumed":consumed[:4]})
            if t == ET.ADDED_KAN:
                if len(consumed) < 3: consumed = [_tt(0), _tt(0), _tt(0)]
                try: pai = _tt(open_obj.last_tile())
                except: pass
                return json.dumps({"type":"kakan","actor":a,"pai":pai,"consumed":consumed[:3]})

            return None
        except Exception:
            return None

    # ── start events ──

    def _start_game_json(self) -> str:
        return json.dumps({"type":"start_game","names":[f"p{i}" for i in range(4)]})

    def _start_kyoku_json(self, obs) -> str:
        try: scores = list(obs.tens())
        except: scores = [25000]*4
        try: oya = obs.dealer()
        except: oya = 0
        try: rn = obs.round()
        except: rn = 0
        try: honba = obs.honba()
        except: honba = 0
        try: kyotaku = obs.kyotaku()
        except: kyotaku = 0
        try: dora = _tt(obs.doras()[0])
        except: dora = _tt(0)
        bakaze = _T[27] if rn < 4 else _T[28]

        # tehais — always valid tiles
        tehais = [[_tt(0)]*13 for _ in range(4)]
        try:
            hand = obs.curr_hand()
            my_tiles = [_tt(t) for t in hand.closed_tiles()]
            while len(my_tiles) < 13:
                my_tiles.append(_tt(0))
            tehais[self.player_id] = my_tiles[:13]
        except:
            pass

        kyoku = (rn % 4) + 1
        return json.dumps({
            "type":"start_kyoku","bakaze":bakaze,"dora_marker":dora,
            "kyoku":kyoku,"honba":honba,"kyotaku":kyotaku,
            "oya":oya,"scores":scores,"tehais":tehais,
        })

    # ── Mortal 46-dim → mjx 181-dim ──

    def _to_mjx(self, action, legals):
        if action <= 36:  # discard
            tid = action if action < 34 else (action % 34)
            for a in legals:
                if 0 <= a.to_idx() <= 73:
                    at = a.tile()
                    if at and hasattr(at, 'type') and at.type() == tid:
                        return a
            for a in legals:
                if 0 <= a.to_idx() <= 73: return a

        if action == 37:  # riichi
            for a in legals:
                if a.to_idx() == 177: return a
        if 38 <= action <= 40:  # chi
            for a in legals:
                if 74 <= a.to_idx() <= 103: return a
        if action == 41:  # pon
            for a in legals:
                if 104 <= a.to_idx() <= 140: return a
        if action == 42:  # kan
            for a in legals:
                if 141 <= a.to_idx() <= 174: return a
        if action == 43:  # agari
            for a in legals:
                if a.to_idx() in (175,176): return a
        if action == 44:  # ryukyoku
            for a in legals:
                if a.to_idx() == 178: return a
        if action == 45:  # pass
            for a in legals:
                if a.to_idx() == 179: return a

        for a in legals:
            if 0 <= a.to_idx() <= 73: return a
        return legals[0]

    def close(self):
        pass