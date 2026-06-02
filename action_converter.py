"""
action_converter.py — Decision Mamba 模型輸出 (0~180) → Akagi mjai JSON 動作

實作 mjx/include/mjx/internal/action.cpp::Action::Encode() 的反向轉換。
參考 feature_action_reference.md 的編碼表與 Akagi/mjai_bot/example/bot.py 的 State 格式。

用法：
    from action_converter import decode
    mjai_json = decode(action_code, actor_id, hand34, melds, aka_in_hand, last_event)
"""

from __future__ import annotations
from typing import List, Dict, Optional

# =========================
# 常數：TileType (0~33) → mjai 字串
# =========================
_TILE_NAMES: List[str] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
]

# 赤 5 對應資訊
_RED_TIDS = {4, 13, 22}  # 5m, 5p, 5s 的 TileType id
_RED_SUFFIX = {4: "m", 13: "p", 22: "s"}

# mjai 字串 → TileType id（含赤牌）
_HONOR_MAP = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}


def _mjai_to_tid(tile: str) -> int:
    """mjai tile string → TileType id (0~33), strip 'r' suffix."""
    t = tile[:-1] if tile.endswith("r") else tile
    if t in _HONOR_MAP:
        return _HONOR_MAP[t]
    if t[0] == "0":
        # 赤牌記法 0m/0p/0s → 5m/5p/5s
        n = 5
    else:
        n = int(t[0])
    base = {"m": 0, "p": 9, "s": 18}[t[1]]
    return base + n - 1


def _tid_to_mjai(tid: int, is_aka: bool = False) -> str:
    """TileType id → mjai string."""
    if is_aka and tid in _RED_TIDS:
        return f"5{_RED_SUFFIX[tid]}r"
    return _TILE_NAMES[tid]


# =========================
# 各動作解碼函式
# =========================

def _decode_discard(code: int, actor: int, hand34: List[int],
                    aka_in_hand: Dict[str, int],
                    tsumogiri: bool, last_self_tsumo: str = "") -> dict:
    """
    DISCARD (0~36) / TSUMOGIRI
    code: 0~33=普通牌, 34=赤5m, 35=赤5p, 36=赤5s
    """
    if tsumogiri and last_self_tsumo:
        # 摸切：牌即為上次摸到的牌
        return {
            "type": "dahai",
            "actor": actor,
            "pai": last_self_tsumo,
            "tsumogiri": True,
        }

    if 0 <= code <= 33:
        tid = code
        is_aka = False
    elif code == 34:
        tid, is_aka = 4, True
    elif code == 35:
        tid, is_aka = 13, True
    elif code == 36:
        tid, is_aka = 22, True
    else:
        return {"type": "none"}

    pai = _tid_to_mjai(tid, is_aka)
    return {
        "type": "dahai",
        "actor": actor,
        "pai": pai,
        "tsumogiri": tsumogiri,
    }


def _decode_chi(code: int, actor: int, hand34: List[int],
                aka_in_hand: Dict[str, int],
                last_event: dict) -> dict:
    """
    CHI (74~103)
    74~94: 無赤牌 → (code - 74) 對應花色+起始牌
    95~103: 含赤牌

    需從 hand34 中找出 consumed 的兩張牌。
    pai 來自 last_event（對手的捨牌）。
    """
    target = last_event.get("actor", -1)
    discarded = last_event.get("pai", "")

    if 74 <= code <= 94:
        idx = code - 74
        suit_idx = idx // 7  # 0=m, 1=p, 2=s
        start_num = (idx % 7) + 1  # 1~7
        suit_char = "mps"[suit_idx]
        has_aka = False
    elif 95 <= code <= 103:
        idx = code - 95
        suit_idx = idx // 3  # 0=m, 1=p, 2=s
        pos = idx % 3     # 0→m3, 1→m4, 2→m5(赤)
        suit_char = "mps"[suit_idx]
        has_aka = True
        if pos == 0:
            start_num = 3
        elif pos == 1:
            start_num = 4
        else:
            start_num = 5  # 赤5 為最小值時
    else:
        return {"type": "none"}

    # 構建三牌序列
    seq = [start_num, start_num + 1, start_num + 2]
    seq_tids = [suit_idx * 9 + (n - 1) for n in seq]

    # 決定哪張是 pai（來自對手），哪兩張是 consumed（來自手牌）
    discarded_tid = _mjai_to_tid(discarded)
    discarded_is_aka = discarded.endswith("r")

    consumed: List[str] = []
    pai = discarded  # default

    for i, tid in enumerate(seq_tids):
        num = seq[i]
        # 這張牌是否為赤5（當 has_aka=True 且是中間那張時）
        is_red = (has_aka and num == 5 and i == (2 if start_num == 3 else 1 if start_num == 4 else 0))
        tile_str = _tid_to_mjai(tid, is_red)

        if tid == discarded_tid:
            # 這張是對手打的
            if discarded_is_aka:
                pai = _tid_to_mjai(tid, True)
            else:
                pai = tile_str
        else:
            consumed.append(tile_str)

    # 若未找到 pai（可能是赤牌匹配問題），用 discarded
    if len(consumed) != 2:
        # fallback: 把非 discarded 的兩張當 consumed
        consumed = []
        for i, tid in enumerate(seq_tids):
            if tid != discarded_tid:
                consumed.append(_TILE_NAMES[tid])
        if len(consumed) != 2:
            return {"type": "none"}

    return {
        "type": "chi",
        "actor": actor,
        "target": target,
        "pai": pai,
        "consumed": consumed,
    }


def _decode_pon(code: int, actor: int, hand34: List[int],
                aka_in_hand: Dict[str, int],
                last_event: dict) -> dict:
    """
    PON (104~140)
    104~137: 無赤牌 → code - 104 = TileType id
    138: 赤5m, 139: 赤5p, 140: 赤5s
    """
    target = last_event.get("actor", -1)
    discarded = last_event.get("pai", "")

    if 104 <= code <= 137:
        tid = code - 104
        has_aka = False
    elif code == 138:
        tid, has_aka = 4, True
    elif code == 139:
        tid, has_aka = 13, True
    elif code == 140:
        tid, has_aka = 22, True
    else:
        return {"type": "none"}

    pai = discarded

    # 從手牌中取 2 張 consumed
    suit_char = _RED_SUFFIX.get(tid, "")
    aka_count = aka_in_hand.get(suit_char, 0)

    if has_aka and aka_count >= 1 and hand34[tid] >= 2:
        consumed = [f"5{suit_char}r", _TILE_NAMES[tid]]
    else:
        consumed = [_TILE_NAMES[tid], _TILE_NAMES[tid]]

    return {
        "type": "pon",
        "actor": actor,
        "target": target,
        "pai": pai,
        "consumed": consumed,
    }


def _decode_kan(code: int, actor: int, hand34: List[int],
                aka_in_hand: Dict[str, int],
                melds: List[dict],
                last_event: dict) -> dict:
    """
    KAN (141~174) — 三種槓共用此區間
    CLOSED_KAN / OPEN_KAN (Daiminkan) / ADDED_KAN (Kakan)

    透過 last_event 區分：
    - last_event.type == "tsumo" → 自家回合 → 可能是 ANKAN 或 KAKAN
    - last_event.type == "dahai" → 別家回合 → DAIMINKAN
    """
    tid = code - 141
    if not (0 <= tid <= 33):
        return {"type": "none"}

    last_type = last_event.get("type", "")

    # --- 情況 1: 自家回合 (tsumo) → ANKAN 或 KAKAN ---
    if last_type == "tsumo":
        # 檢查是否有現有 pon 可以升級為 kakan
        for m in melds:
            if m.get("type") == "pon":
                meld_tid = _mjai_to_tid(m["tiles"][0])
                if meld_tid == tid:
                    # KAKAN
                    suit_char = _RED_SUFFIX.get(tid, "")
                    if suit_char and aka_in_hand.get(suit_char, 0) > 0:
                        pai = f"5{suit_char}r"
                    else:
                        pai = _TILE_NAMES[tid]
                    return {
                        "type": "kakan",
                        "actor": actor,
                        "pai": pai,
                        "consumed": list(m["tiles"]),
                    }
        # ANKAN
        suit_char = _RED_SUFFIX.get(tid, "")
        if suit_char and aka_in_hand.get(suit_char, 0) >= 1 and hand34[tid] >= 4:
            consumed = [f"5{suit_char}r"] + [_TILE_NAMES[tid]] * 3
        else:
            consumed = [_TILE_NAMES[tid]] * 4
        return {
            "type": "ankan",
            "actor": actor,
            "consumed": consumed,
        }

    # --- 情況 2: 別家回合 (dahai) → DAIMINKAN ---
    target = last_event.get("actor", -1)
    discarded = last_event.get("pai", "")

    suit_char = _RED_SUFFIX.get(tid, "")
    if suit_char and aka_in_hand.get(suit_char, 0) >= 1 and hand34[tid] >= 3:
        consumed = [f"5{suit_char}r"] + [_TILE_NAMES[tid]] * 2
    else:
        consumed = [_TILE_NAMES[tid]] * 3

    return {
        "type": "daiminkan",
        "actor": actor,
        "target": target,
        "pai": discarded,
        "consumed": consumed,
    }


def _decode_agari(actor: int, state_hand34: List[int],
                  last_event: dict, is_tsumo: bool) -> dict:
    """
    TSUMO (175) / RON (176)

    TSUMO: actor=target=self, pai=最後摸的牌
    RON: actor=self, target=放銃者, pai=放銃牌
    """
    if is_tsumo:
        pai = last_event.get("pai", "")
        return {
            "type": "hora",
            "actor": actor,
            "target": actor,
            "pai": pai,
        }
    else:
        target = last_event.get("actor", -1)
        pai = last_event.get("pai", "")
        return {
            "type": "hora",
            "actor": actor,
            "target": target,
            "pai": pai,
        }


# =========================
# 主入口
# =========================

def decode(
    action_code: int,
    actor_id: int,
    hand34: List[int],
    melds: List[dict],
    aka_in_hand: Dict[str, int],
    last_event: dict,
    last_self_tsumo: str = "",
) -> dict:
    """
    將模型輸出的 0~180 action index 轉換為 Akagi mjai JSON 動作。

    Args:
        action_code: 模型輸出 (0~180)，對應 mjx Action::Encode()
        actor_id: Bot 的座位號碼 (0~3)
        hand34: 手牌 34-array，hand34[i] = 牌 i 的持有張數
        melds: 現有副露列表 [{"type":"pon","tiles":[...], ...}, ...]
        aka_in_hand: 手牌中赤牌的數量 {"m": N, "p": N, "s": N}
        last_event: 最後一個 mjai 事件 dict
        last_self_tsumo: 上次自家摸到的牌 (mjai string)，供 tsumogiri 使用

    Returns:
        mjai 動作 dict，可直接 json.dumps() 輸出給 Akagi
    """
    # ----- DISCARD (0~36) -----
    if 0 <= action_code <= 36:
        return _decode_discard(action_code, actor_id, hand34, aka_in_hand,
                               tsumogiri=False)

    # ----- TSUMOGIRI (37~73) -----
    if 37 <= action_code <= 73:
        return _decode_discard(action_code - 37, actor_id, hand34, aka_in_hand,
                               tsumogiri=True, last_self_tsumo=last_self_tsumo)

    # ----- CHI (74~103) -----
    if 74 <= action_code <= 103:
        return _decode_chi(action_code, actor_id, hand34, aka_in_hand, last_event)

    # ----- PON (104~140) -----
    if 104 <= action_code <= 140:
        return _decode_pon(action_code, actor_id, hand34, aka_in_hand, last_event)

    # ----- KAN (141~174) -----
    if 141 <= action_code <= 174:
        return _decode_kan(action_code, actor_id, hand34, aka_in_hand, melds, last_event)

    # ----- TSUMO (175) -----
    if action_code == 175:
        return _decode_agari(actor_id, hand34, last_event, is_tsumo=True)

    # ----- RON (176) -----
    if action_code == 176:
        return _decode_agari(actor_id, hand34, last_event, is_tsumo=False)

    # ----- RIICHI (177) -----
    if action_code == 177:
        return {"type": "reach", "actor": actor_id}

    # ----- NINE_TERMINALS (178) -----
    if action_code == 178:
        return {"type": "kyushu", "actor": actor_id}

    # ----- NO (179) / DUMMY (180) -----
    if action_code in (179, 180):
        return {"type": "none"}

    # 未知編碼，fallback
    return {"type": "none"}


# =========================
# 測試
# =========================

if __name__ == "__main__":
    import json

    hand = [0] * 34
    # 手牌: 1m, 2m, 3m, 4m (各1), 5mr, 6m, 7m, 8m, 9m, 1p, 2p, 3p, E
    for t in ["1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "E"]:
        hand[_mjai_to_tid(t)] += 1
    hand[4] += 1  # 5m
    aka = {"m": 1, "p": 0, "s": 0}

    print("=== 測試 DISCARD ===")
    print(json.dumps(decode(0, 0, hand, [], aka, {})))
    print(json.dumps(decode(34, 0, hand, [], aka, {})))

    print("\n=== 測試 TSUMOGIRI ===")
    print(json.dumps(decode(37, 0, hand, [], aka, {}, last_self_tsumo="9p")))

    print("\n=== 測試 CHI (無赤) ===")
    evt = {"type": "dahai", "actor": 3, "pai": "3m"}
    print(f"last_event: {evt}")
    print(json.dumps(decode(74, 0, hand, [], aka, evt)))

    print("\n=== 測試 CHI (含赤) ===")
    evt = {"type": "dahai", "actor": 3, "pai": "5mr"}
    # code 96 = m4, 456 含赤 (tiles: m4, m5r, m6)
    print(f"last_event: {evt}")
    print(json.dumps(decode(96, 0, hand, [], aka, evt)))

    print("\n=== 測試 PON ===")
    evt = {"type": "dahai", "actor": 3, "pai": "1m"}
    print(f"last_event: {evt}")
    print(json.dumps(decode(104, 0, hand, [], aka, evt)))

    print("\n=== 測試 KAN (ANKAN) ===")
    hand_k = [0] * 34
    hand_k[0] = 4  # 4 張 1m
    evt = {"type": "tsumo", "actor": 0, "pai": "1m"}
    print(json.dumps(decode(141, 0, hand_k, [], {"m": 0, "p": 0, "s": 0}, evt)))

    print("\n=== 測試 TSUMO ===")
    evt = {"type": "tsumo", "actor": 0, "pai": "5m"}
    print(json.dumps(decode(175, 0, hand, [], aka, evt)))

    print("\n=== 測試 RON ===")
    evt = {"type": "dahai", "actor": 2, "pai": "E"}
    print(json.dumps(decode(176, 0, hand, [], aka, evt)))

    print("\n=== 測試 RIICHI ===")
    print(json.dumps(decode(177, 0, hand, [], aka, {})))

    print("\n=== 測試 NO ===")
    print(json.dumps(decode(179, 0, hand, [], aka, {})))

    print("\n=== 測試 DUMMY ===")
    print(json.dumps(decode(180, 0, hand, [], aka, {})))

    print("\n✅ 所有測試完成")