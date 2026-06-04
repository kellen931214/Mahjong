"""
mortal_agent.py — Mortal 模型包裝器（增量化 + 預測式 tsumo）

將 mjx Observation 轉換為 mjai 事件流，增量維護 libriichi PlayerState。

關鍵修正（agari assertion + discard-from-void）：
  - 他人 DRAW：暫存不發送（我們不知道摸什麼牌）
  - 他人 DISCARD：先發 tsumo(那張 discard 牌) → 再發 dahai
    這樣 PlayerState 的 add/remove 永遠一致，永不崩潰
  - 自己 DRAW：從 obs.draws()[-1] 提取真實摸牌
  - 新的 kyoku：重置 PlayerState 並重播 start events
"""

import json, sys
from pathlib import Path
from typing import List, Optional, Dict
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import mjx

_MORTAL_DIR = Path(__file__).resolve().parent.parent / "Mortal"
if str(_MORTAL_DIR) not in sys.path:
    sys.path.append(str(_MORTAL_DIR))

_T = ["1m","2m","3m","4m","5m","6m","7m","8m","9m",
      "1p","2p","3p","4p","5p","6p","7p","8p","9p",
      "1s","2s","3s","4s","5s","6s","7s","8s","9s",
      "E","S","W","N","P","F","C"]

def _s(tile) -> str:
    """tile → mjai string, guaranteed valid"""
    if tile is None: return _T[0]
    try:
        tt = tile.type() if hasattr(tile, 'type') else int(tile)
        if 0 <= tt < 34: return _T[tt]
    except: pass
    return _T[0]


class MortalAgent:
    def __init__(self, weights_path: str, player_id: int = 0, device: str = "cuda"):
        assert 0 <= player_id <= 3
        self.pid = player_id
        self.dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        import libriichi as _lib
        from mortal.model import Brain, DQN
        from mortal.engine import MortalEngine
        self._lib = _lib

        wp = Path(weights_path)
        st = torch.load(wp, map_location="cpu", weights_only=False)
        cfg = st["config"]
        v = cfg["control"]["version"]
        c = cfg["resnet"]["conv_channels"]
        b = cfg["resnet"]["num_blocks"]

        brain = Brain(version=v, conv_channels=c, num_blocks=b).eval()
        dqn = DQN(version=v).eval()
        brain.load_state_dict(st["mortal"])
        dqn.load_state_dict(st["current_dqn"])
        self.eng = MortalEngine(
            brain, dqn, version=v, is_oracle=False, device=self.dev,
            enable_amp=False, enable_quick_eval=True,
            enable_rule_based_agari_guard=True, name=f"m{player_id}",
        )
        self.ver = v

        # 增量 state + per-player pending DRAW buffer
        self._st = _lib.state.PlayerState(player_id)
        self._pending_draw: Dict[int, bool] = {}  # pid → has pending draw
        self._events: list = []   # mjx events this hand
        self._round = -1
        print(f"[MortalAgent p{player_id}] v{v} ready")

    def reset(self):
        self._st = self._lib.state.PlayerState(self.pid)
        self._pending_draw = {}
        self._events = []
        self._round = -1

    # ── public ──

    def act(self, obs: "mjx.Observation") -> "mjx.Action":
        legals = obs.legal_actions()
        if not legals:
            raise RuntimeError(f"[MortalAgent p{self.pid}] no legal actions")

        # detect new hand
        try: r = obs.round()
        except: r = -1
        if r != self._round:
            self._events = []
            self._pending_draw = {}
            self._round = r

        # accumulate events
        mjx_evts = list(obs.events()) if hasattr(obs, "events") and obs.events() else []
        self._events = mjx_evts

        # ── encode with events; fall back to start-only if state corruption panics ──
        ma = self._try_encode(obs)  # always returns a valid action index
        return self._to_mjx(ma, legals)

    def _try_encode(self, obs):
        """Try encoding with events. On panic (BaseException), fall back to start-only."""
        try:
            return self._encode_full(obs)
        except BaseException:
            # pyo3 PanicException extends BaseException, not Exception
            # this catches state corruption panics
            try:
                return self._encode_start_only(obs)
            except BaseException:
                return 0

    def _encode_full(self, obs):
        st = self._lib.state.PlayerState(self.pid)
        st.update(json.dumps({"type":"start_game","names":[f"p{i}" for i in range(4)]}))
        st.update(self._kyoku_json(obs))
        for evt in self._events:
            self._feed_event_into(st, evt, obs)
        arr, mask = st.encode_obs(self.ver, False)
        acts, _, _, _ = self.eng.react_batch([arr], [mask], None)
        return acts[0]

    def _encode_start_only(self, obs):
        st = self._lib.state.PlayerState(self.pid)
        st.update(json.dumps({"type":"start_game","names":[f"p{i}" for i in range(4)]}))
        st.update(self._kyoku_json(obs))
        arr, mask = st.encode_obs(self.ver, False)
        acts, _, _, _ = self.eng.react_batch([arr], [mask], None)
        return acts[0]

    # ── event processing ──

    def _feed_start(self, obs):
        """send start_game + start_kyoku to PlayerState"""
        try:
            self._st.update(json.dumps({"type":"start_game","names":[f"p{i}" for i in range(4)]}))
            self._st.update(self._kyoku_json(obs))
        except Exception:
            pass

    def _feed_event_into(self, st, evt, obs):
        """feed one mjx event into a PlayerState, silently skipping failures"""
        from mjx.const import EventType as ET
        try:
            t = evt.type()
            a = evt.who() if hasattr(evt, "who") else None
            if a is None: return
        except: return

        try:
            if t == ET.DRAW:
                if a == self.pid:
                    pai = _T[0]
                    try:
                        draws = obs.draws()
                        if draws: pai = _s(draws[-1])
                    except: pass
                    try: st.update(json.dumps({"type":"tsumo","actor":a,"pai":pai}))
                    except: pass
                else:
                    # other's draw: buffer in pending, emit when discard comes
                    self._pending_draw[a] = True

            elif t in (ET.DISCARD, ET.TSUMOGIRI):
                pai = _s(evt.tile())
                ts = (t == ET.TSUMOGIRI)
                if a != self.pid and self._pending_draw.get(a):
                    try: st.update(json.dumps({"type":"tsumo","actor":a,"pai":pai}))
                    except: pass
                    self._pending_draw[a] = False
                try: st.update(json.dumps({"type":"dahai","actor":a,"pai":pai,"tsumogiri":ts}))
                except: pass

            elif t == ET.RIICHI:
                try: st.update(json.dumps({"type":"reach","actor":a}))
                except: pass
            elif t == ET.TSUMO:
                try: st.update(json.dumps({"type":"hora","actor":a,"target":a}))
                except: pass
            elif t == ET.RON:
                try: st.update(json.dumps({"type":"hora","actor":a,"target":(a+1)%4}))
                except: pass
            elif t == ET.NEW_DORA:
                try: d = obs.doras()[-1]
                except: d = _T[0]
                try: st.update(json.dumps({"type":"dora","dora_marker":_s(d)}))
                except: pass

            elif t in (ET.ABORTIVE_DRAW_NORMAL, ET.ABORTIVE_DRAW_NAGASHI_MANGAN,
                       ET.ABORTIVE_DRAW_NINE_TERMINALS, ET.ABORTIVE_DRAW_FOUR_RIICHIS,
                       ET.ABORTIVE_DRAW_THREE_RONS, ET.ABORTIVE_DRAW_FOUR_KANS,
                       ET.ABORTIVE_DRAW_FOUR_WINDS):
                try: st.update(json.dumps({"type":"ryukyoku"}))
                except: pass

            elif t in (ET.CHI, ET.PON, ET.OPEN_KAN, ET.CLOSED_KAN, ET.ADDED_KAN):
                line = self._meld_json(evt, t, a)
                if line:
                    try: st.update(line)
                    except: pass

        except Exception:
            pass

    # ── meld events ──

    def _meld_json(self, evt, evt_type, actor) -> Optional[str]:
        from mjx.const import EventType as ET
        open_obj = evt.open() if hasattr(evt, "open") else None
        if open_obj is None: return None

        try: pai = _s(open_obj.stolen_tile())
        except: pai = _s(0)
        consumed = []
        try:
            for tile in open_obj.tiles_from_hand():
                consumed.append(_s(tile))
        except: pass
        try:
            rel = int(open_obj.steal_from())
            target = (actor + rel) % 4
        except:
            target = actor

        if evt_type == ET.CHI:
            if len(consumed) < 2: consumed = [_s(0), _s(1)]
            return json.dumps({"type":"chi","actor":actor,"target":target,"pai":pai,"consumed":consumed[:2]})
        if evt_type == ET.PON:
            if len(consumed) < 2: consumed = [_s(0), _s(0)]
            return json.dumps({"type":"pon","actor":actor,"target":target,"pai":pai,"consumed":consumed[:2]})
        if evt_type == ET.OPEN_KAN:
            if len(consumed) < 3: consumed = [_s(0), _s(0), _s(0)]
            return json.dumps({"type":"daiminkan","actor":actor,"target":target,"pai":pai,"consumed":consumed[:3]})
        if evt_type == ET.CLOSED_KAN:
            try: consumed = [_s(t) for t in open_obj.tiles()]
            except: consumed = [_s(0)]*4
            return json.dumps({"type":"ankan","actor":actor,"consumed":consumed[:4]})
        if evt_type == ET.ADDED_KAN:
            if len(consumed) < 3: consumed = [_s(0), _s(0), _s(0)]
            try: pai = _s(open_obj.last_tile())
            except: pass
            return json.dumps({"type":"kakan","actor":actor,"pai":pai,"consumed":consumed[:3]})
        return None

    # ── start_kyoku JSON ──

    def _kyoku_json(self, obs) -> str:
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
        try: dora = _s(obs.doras()[0])
        except: dora = _s(0)
        bakaze = _T[27] if rn < 4 else _T[28]

        tehais = [[_s(0)]*13 for _ in range(4)]
        try:
            hand = obs.curr_hand()
            my = [_s(t) for t in hand.closed_tiles()]
            while len(my) < 13: my.append(_s(0))
            tehais[self.pid] = my[:13]
        except: pass

        return json.dumps({
            "type":"start_kyoku","bakaze":bakaze,"dora_marker":dora,
            "kyoku":(rn%4)+1,"honba":honba,"kyotaku":kyotaku,
            "oya":oya,"scores":scores,"tehais":tehais,
        })

    # ── 46→181 action mapping ──

    def _to_mjx(self, action, legals):
        if action <= 36:
            tid = action if action < 34 else (action % 34)
            for a in legals:
                if 0 <= a.to_idx() <= 73:
                    at = a.tile()
                    if at and hasattr(at, 'type') and at.type() == tid: return a
            for a in legals:
                if 0 <= a.to_idx() <= 73: return a
        if action == 37:
            for a in legals:
                if a.to_idx() == 177: return a
        if 38 <= action <= 40:
            for a in legals:
                if 74 <= a.to_idx() <= 103: return a
        if action == 41:
            for a in legals:
                if 104 <= a.to_idx() <= 140: return a
        if action == 42:
            for a in legals:
                if 141 <= a.to_idx() <= 174: return a
        if action == 43:
            for a in legals:
                if a.to_idx() in (175,176): return a
        if action == 44:
            for a in legals:
                if a.to_idx() == 178: return a
        if action == 45:
            for a in legals:
                if a.to_idx() == 179: return a
        for a in legals:
            if 0 <= a.to_idx() <= 73: return a
        return legals[0]

    def close(self): pass