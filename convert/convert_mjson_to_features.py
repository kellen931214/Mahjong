"""Convert MJAI mjson logs to Decision Mamba NPY chunks.

Output contract is unchanged:
  features.npy: (N, 1380) float32
  actions.npy: (N,) int64
  rtgs.npy: (N, 1) float32
  trajectory_boundaries.npy: (num_trajectories,) int64

The reward/RTG calculation uses the unified reward from utli/rewards.py.
Install the pure-Python shanten dependency on the conversion machine:

    pip install mahjong
"""

from __future__ import annotations

import argparse
import copy
import glob
import gzip
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utli.rewards import MahjongRewardCalculator, discounted_returns


TILE_STR_TO_TYPE = {
    "1m": 0, "2m": 1, "3m": 2, "4m": 3, "5m": 4,
    "6m": 5, "7m": 6, "8m": 7, "9m": 8,
    "1p": 9, "2p": 10, "3p": 11, "4p": 12, "5p": 13,
    "6p": 14, "7p": 15, "8p": 16, "9p": 17,
    "1s": 18, "2s": 19, "3s": 20, "4s": 21, "5s": 22,
    "6s": 23, "7s": 24, "8s": 25, "9s": 26,
    "E": 27, "S": 28, "W": 29, "N": 30,
    "P": 31, "F": 32, "C": 33,
}
for tile_name in list(TILE_STR_TO_TYPE):
    TILE_STR_TO_TYPE[tile_name + "r"] = TILE_STR_TO_TYPE[tile_name]

DECISION_EVENTS = {
    "dahai",
    "chi",
    "pon",
    "kan",
    "daiminkan",
    "ankan",
    "kakan",
    "reach",
    "hora",
    "ryukyoku",
    "none",
}
CALL_EVENTS = {"chi", "pon", "kan", "daiminkan"}
KAN_EVENTS = {"kan", "daiminkan", "ankan", "kakan"}
EXHAUSTIVE_DRAW_REASONS = {"", "fanpai", "normal", "exhaustive", "nm"}
DEBUG_DTYPE = np.dtype(
    [
        ("total", np.float32),
        ("shape", np.float32),
        ("score", np.float32),
        ("event", np.float32),
        ("game", np.float32),
        ("potential", np.float32),
    ]
)
UNIFIED_REWARD = MahjongRewardCalculator()


class MjxActionEncoder:
    @staticmethod
    def encode_discard(tile_type: int, is_red: bool = False) -> int:
        if is_red:
            if tile_type == 4:
                return 34
            if tile_type == 13:
                return 35
            if tile_type == 22:
                return 36
        return tile_type

    @staticmethod
    def encode_tsumogiri(tile_type: int, is_red: bool = False) -> int:
        if is_red:
            if tile_type == 4:
                return 71
            if tile_type == 13:
                return 72
            if tile_type == 22:
                return 73
        return tile_type + 37

    @staticmethod
    def encode_chi(base_tile: int, has_red: bool = False) -> int:
        if not has_red:
            if base_tile <= 8:
                return (base_tile % 9) + 74
            if base_tile <= 17:
                return (base_tile % 9) + 81
            return (base_tile % 9) + 88
        return {
            2: 95, 3: 96, 4: 97,
            11: 98, 12: 99, 13: 100,
            20: 101, 21: 102, 22: 103,
        }.get(base_tile, 74)

    @staticmethod
    def encode_pon(tile_type: int, is_red: bool = False) -> int:
        if is_red:
            if tile_type == 4:
                return 138
            if tile_type == 13:
                return 139
            if tile_type == 22:
                return 140
        return tile_type + 104

    @staticmethod
    def encode_kan(tile_type: int) -> int:
        return tile_type + 141


@dataclass
class GameState:
    round_num: int
    honba: int
    kyotaku: int
    hands: List[List[int]]
    discards: List[List[int]]
    melds: List[List[List[int]]]
    scores: List[int]
    dora_indicators: List[int]
    dealer: int
    prevalent_wind: str
    reach_status: List[int] = field(default_factory=lambda: [0, 0, 0, 0])


@dataclass
class RewardState:
    hands: List[List[str]]
    melds: List[List[List[str]]]
    discards: List[List[str]]
    stolen_discard_indices: List[Set[int]]
    scores: List[int]
    dora_markers: List[str]


class FeatureExtractor:
    """Original 1380-dimensional feature definition. Do not change."""

    def __init__(self, observer_idx: int):
        self.observer_idx = observer_idx

    def extract_spatial_features(self, state: GameState) -> np.ndarray:
        features = np.zeros((40, 34), dtype=np.uint8)

        hand_counts = [0] * 34
        for tile in state.hands[self.observer_idx]:
            if 0 <= tile < 34:
                hand_counts[tile] += 1
        for tile, count in enumerate(hand_counts):
            for index in range(min(count, 4)):
                features[index][tile] = 1

        for meld_idx, meld in enumerate(state.melds[self.observer_idx][:4]):
            for tile in set(meld):
                if 0 <= tile < 34:
                    features[4 + meld_idx][tile] = 1

        for relative_pos in range(1, 4):
            abs_pos = (self.observer_idx + relative_pos) % 4
            for meld_idx, meld in enumerate(state.melds[abs_pos][:4]):
                channel_idx = 8 + (relative_pos - 1) * 4 + meld_idx
                for tile in set(meld):
                    if 0 <= tile < 34:
                        features[channel_idx][tile] = 1

        for relative_pos in range(4):
            abs_pos = (self.observer_idx + relative_pos) % 4
            discards = state.discards[abs_pos]
            for group_idx in range(4):
                channel_idx = 20 + relative_pos * 4 + group_idx
                start_idx = group_idx * 6
                end_idx = min(start_idx + 6, len(discards))
                for tile in discards[start_idx:end_idx]:
                    if 0 <= tile < 34:
                        features[channel_idx][tile] = 1

        for dora_idx, dora in enumerate(state.dora_indicators[:4]):
            if 0 <= dora < 34:
                features[36 + dora_idx][dora] = 1
        return features

    def extract_scalar_features(self, state: GameState) -> np.ndarray:
        features = np.zeros(20, dtype=np.float32)
        for index in range(4):
            abs_pos = (self.observer_idx + index) % 4
            features[index] = max(
                0, (state.scores[abs_pos] + 50000) / 100000.0
            )

        used_tiles = sum(len(discards) for discards in state.discards)
        remaining_wall = (
            136
            - used_tiles
            - sum(len(hand) for hand in state.hands)
        )
        features[4] = max(0, remaining_wall / 70.0)

        if state.prevalent_wind == "E":
            features[5] = 1.0
        elif state.prevalent_wind == "S":
            features[6] = 1.0

        round_idx = state.round_num % 4
        if 0 <= round_idx < 4:
            features[7 + round_idx] = 1.0

        features[11] = min(state.honba / 30.0, 1.0)
        features[12] = min(state.kyotaku / 4.0, 1.0)

        for index in range(4):
            abs_pos = (self.observer_idx + index) % 4
            features[13 + index] = float(state.reach_status[abs_pos])

        relative_dealer = (state.dealer - self.observer_idx + 4) % 4
        if relative_dealer == 0:
            features[17] = 1.0
        elif relative_dealer == 1:
            features[18] = 1.0
        elif relative_dealer == 2:
            features[19] = 1.0
        return features

    def extract_features(self, state: GameState) -> np.ndarray:
        return np.concatenate(
            [
                self.extract_spatial_features(state).flatten(),
                self.extract_scalar_features(state),
            ]
        )


class MahjongPotentialCalculator:
    def __init__(self):
        try:
            from mahjong.shanten import Shanten
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The converter requires the 'mahjong' package. "
                "Install it with: pip install mahjong"
            ) from exc
        self._shanten = Shanten()

    @staticmethod
    def _counts(tiles: Iterable[str]) -> List[int]:
        counts = [0] * 34
        for tile in tiles:
            tile_type = TILE_STR_TO_TYPE.get(tile)
            if tile_type is not None:
                counts[tile_type] += 1
        return counts

    def shanten(self, state: RewardState, player_idx: int) -> int:
        counts = self._counts(state.hands[player_idx])
        closed_hand = len(state.melds[player_idx]) == 0
        return self._calculate_counts_shanten(counts, closed_hand)

    def _calculate_counts_shanten(
        self, counts: List[int], closed_hand: bool
    ) -> int:
        tile_count = sum(counts)
        if tile_count % 3 != 0:
            return int(
                self._shanten.calculate_shanten(
                    counts,
                    closed_hand,
                    closed_hand,
                )
            )

        # MJX permits the temporary 3n count created while checking whether
        # one more draw improves a 3n+2 decision hand. The Python mahjong
        # package rejects that count, so evaluate every valid subset obtained
        # by dropping one surplus tile.
        candidates = []
        for tile_type, count in enumerate(counts):
            if count == 0:
                continue
            reduced = counts.copy()
            reduced[tile_type] -= 1
            candidates.append(
                int(
                    self._shanten.calculate_shanten(
                        reduced,
                        closed_hand,
                        closed_hand,
                    )
                )
            )
        if not candidates:
            raise ValueError("Cannot calculate shanten for an empty hand")
        return min(candidates)

    @staticmethod
    def _dora_type(marker: str) -> Optional[int]:
        marker_type = TILE_STR_TO_TYPE.get(marker)
        if marker_type is None:
            return None
        if marker_type < 27:
            suit_start = (marker_type // 9) * 9
            return suit_start + ((marker_type - suit_start + 1) % 9)
        if marker_type <= 30:
            return 27 + ((marker_type - 27 + 1) % 4)
        return 31 + ((marker_type - 31 + 1) % 3)

    def dora_count(self, state: RewardState, player_idx: int) -> int:
        dora_types = [
            dora_type
            for dora_type in (
                self._dora_type(marker) for marker in state.dora_markers
            )
            if dora_type is not None
        ]
        tiles = list(state.hands[player_idx])
        for meld in state.melds[player_idx]:
            tiles.extend(meld)
        count = sum(tile.endswith("r") for tile in tiles)
        count += sum(
            TILE_STR_TO_TYPE.get(tile) in dora_types for tile in tiles
        )
        return count

    @staticmethod
    def visible_counts(state: RewardState, player_idx: int) -> List[int]:
        counts = [0] * 34

        def add(tile: str) -> None:
            tile_type = TILE_STR_TO_TYPE.get(tile)
            if tile_type is not None:
                counts[tile_type] += 1

        for tile in state.hands[player_idx]:
            add(tile)
        for player_melds in state.melds:
            for meld in player_melds:
                for tile in meld:
                    add(tile)
        for discarder, discards in enumerate(state.discards):
            stolen = state.stolen_discard_indices[discarder]
            for index, tile in enumerate(discards):
                if index not in stolen:
                    add(tile)
        for marker in state.dora_markers:
            add(marker)
        return [min(count, 4) for count in counts]

    def ukeire(self, state: RewardState, player_idx: int) -> int:
        current_shanten = self.shanten(state, player_idx)
        visible = self.visible_counts(state, player_idx)
        hand_counts = self._counts(state.hands[player_idx])
        closed_hand = len(state.melds[player_idx]) == 0
        total = 0
        for tile_type in range(34):
            if visible[tile_type] >= 4:
                continue
            candidate = hand_counts.copy()
            candidate[tile_type] += 1
            new_shanten = self._calculate_counts_shanten(
                candidate, closed_hand
            )
            if new_shanten < current_shanten:
                total += 4 - visible[tile_type]
        return total

    def compute_potential(self, state: RewardState, player_idx: int) -> float:
        shanten = self.shanten(state, player_idx)
        ukeire = self.ukeire(state, player_idx)
        dora_count = self.dora_count(state, player_idx)
        f_shanten = 1.0 - min(float(shanten), 6.0) / 6.0
        f_ukeire = math.log1p(min(float(ukeire), 20.0)) / math.log(21.0)
        f_dora = min(float(dora_count), 3.0) / 3.0
        return 0.60 * f_shanten + 0.30 * f_ukeire + 0.10 * f_dora


def parse_mjson_file(file_path: str) -> List[Dict]:
    events = []
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    events.append(json.loads(line))
    except (gzip.BadGzipFile, OSError):
        with open(file_path, "rt", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    events.append(json.loads(line))
    return events


def _encode_action_mjx(evt_type: str, evt: Dict) -> int:
    actor = evt.get("actor", -1)
    if evt_type == "dahai":
        tile_name = evt.get("pai", "")
        tile_type = TILE_STR_TO_TYPE.get(tile_name, 0)
        if evt.get("tsumogiri", False):
            return MjxActionEncoder.encode_tsumogiri(
                tile_type, tile_name.endswith("r")
            )
        return MjxActionEncoder.encode_discard(
            tile_type, tile_name.endswith("r")
        )
    if evt_type == "chi":
        tiles = [evt.get("pai", "")] + list(evt.get("consumed", []))
        tile_types = [
            TILE_STR_TO_TYPE[tile] for tile in tiles if tile in TILE_STR_TO_TYPE
        ]
        return MjxActionEncoder.encode_chi(
            min(tile_types) if tile_types else 0,
            any(tile.endswith("r") for tile in tiles),
        )
    if evt_type == "pon":
        tiles = [evt.get("pai", "")] + list(evt.get("consumed", []))
        return MjxActionEncoder.encode_pon(
            TILE_STR_TO_TYPE.get(evt.get("pai", ""), 0),
            any(tile.endswith("r") for tile in tiles),
        )
    if evt_type in KAN_EVENTS:
        tile_name = evt.get("pai", "")
        if not tile_name and evt.get("consumed"):
            tile_name = evt["consumed"][0]
        return MjxActionEncoder.encode_kan(
            TILE_STR_TO_TYPE.get(tile_name, 0)
        )
    if evt_type == "hora":
        return 175 if evt.get("target") == actor else 176
    if evt_type == "reach" and evt.get("step") == 1:
        return 177
    if (
        evt_type == "ryukyoku"
        and evt.get("reason") in {"kyushukyuhai", "yao9"}
    ):
        return 178
    if evt_type == "none":
        return 179
    return 180


def _remove_raw_tile(hand: List[str], tile: str) -> None:
    if tile in hand:
        hand.remove(tile)
        return
    tile_type = TILE_STR_TO_TYPE.get(tile)
    for index, candidate in enumerate(hand):
        if TILE_STR_TO_TYPE.get(candidate) == tile_type:
            hand.pop(index)
            return


def _update_feature_state(state: GameState, evt: Dict) -> None:
    evt_type = evt.get("type")
    actor = evt.get("actor", -1)
    if not 0 <= actor < 4:
        return

    if evt_type == "tsumo":
        tile_type = TILE_STR_TO_TYPE.get(evt.get("pai", ""), -1)
        if tile_type != -1:
            state.hands[actor].append(tile_type)
    elif evt_type == "reach" and evt.get("step") == 1:
        state.reach_status[actor] = 1
        state.kyotaku += 1
    elif evt_type == "dahai":
        tile_type = TILE_STR_TO_TYPE.get(evt.get("pai", ""), -1)
        if tile_type != -1:
            state.discards[actor].append(tile_type)
            if tile_type in state.hands[actor]:
                state.hands[actor].remove(tile_type)
    elif evt_type in {"chi", "pon", *KAN_EVENTS}:
        pai = evt.get("pai", "")
        consumed = list(evt.get("consumed", []))
        tile_names = ([pai] if pai else []) + consumed
        meld = [
            TILE_STR_TO_TYPE[tile]
            for tile in tile_names
            if tile in TILE_STR_TO_TYPE
        ]
        if meld:
            state.melds[actor].append(meld)
            for tile_name in consumed:
                tile_type = TILE_STR_TO_TYPE.get(tile_name, -1)
                if tile_type in state.hands[actor]:
                    state.hands[actor].remove(tile_type)


def _mark_stolen_discard(state: RewardState, evt: Dict) -> None:
    target = evt.get("target", -1)
    if not 0 <= target < 4 or not state.discards[target]:
        return
    expected_type = TILE_STR_TO_TYPE.get(evt.get("pai", ""))
    for index in range(len(state.discards[target]) - 1, -1, -1):
        if index in state.stolen_discard_indices[target]:
            continue
        if TILE_STR_TO_TYPE.get(state.discards[target][index]) == expected_type:
            state.stolen_discard_indices[target].add(index)
            return


def _update_reward_tiles(state: RewardState, evt: Dict) -> None:
    evt_type = evt.get("type")
    actor = evt.get("actor", -1)
    if evt_type == "dora":
        marker = evt.get("dora_marker", evt.get("pai", ""))
        if marker in TILE_STR_TO_TYPE:
            state.dora_markers.append(marker)
        return
    if not 0 <= actor < 4:
        return

    if evt_type == "tsumo":
        tile = evt.get("pai", "")
        if tile in TILE_STR_TO_TYPE:
            state.hands[actor].append(tile)
    elif evt_type == "dahai":
        tile = evt.get("pai", "")
        if tile in TILE_STR_TO_TYPE:
            state.discards[actor].append(tile)
            _remove_raw_tile(state.hands[actor], tile)
    elif evt_type == "kakan":
        tile = evt.get("pai", "")
        tile_type = TILE_STR_TO_TYPE.get(tile)
        for meld in state.melds[actor]:
            if (
                len(meld) == 3
                and TILE_STR_TO_TYPE.get(meld[0]) == tile_type
            ):
                meld.append(tile)
                break
        else:
            state.melds[actor].append(
                [tile] + list(evt.get("consumed", []))
            )
        _remove_raw_tile(state.hands[actor], tile)
    elif evt_type in {"chi", "pon", "kan", "daiminkan", "ankan"}:
        pai = evt.get("pai", "")
        consumed = list(evt.get("consumed", []))
        meld = ([pai] if pai else []) + consumed
        meld = [tile for tile in meld if tile in TILE_STR_TO_TYPE]
        if meld:
            state.melds[actor].append(meld)
        for tile in consumed:
            _remove_raw_tile(state.hands[actor], tile)
        if evt_type in CALL_EVENTS:
            _mark_stolen_discard(state, evt)


def _apply_score_event(scores: List[int], evt: Dict) -> None:
    evt_type = evt.get("type")
    if evt_type not in {"reach", "hora", "ryukyoku", "end_game"}:
        return
    if "scores" in evt and len(evt["scores"]) == 4:
        scores[:] = [int(score) for score in evt["scores"]]
        return
    if "deltas" in evt and len(evt["deltas"]) == 4:
        scores[:] = [
            int(scores[index] + evt["deltas"][index])
            for index in range(4)
        ]
        return
    if evt_type == "reach" and evt.get("step") == 2:
        actor = evt.get("actor", -1)
        if 0 <= actor < 4:
            scores[actor] -= 1000


def _reset_states(start_evt: Dict) -> tuple[GameState, RewardState]:
    scores = list(start_evt.get("scores", [25000] * 4))
    raw_hands = [
        [
            tile
            for tile in start_evt.get("tehais", [[], [], [], []])[index]
            if tile in TILE_STR_TO_TYPE
        ]
        for index in range(4)
    ]
    feature_state = GameState(
        round_num=start_evt.get("kyoku", 0) - 1,
        honba=start_evt.get("honba", 0),
        kyotaku=start_evt.get("kyotaku", 0),
        hands=[
            [TILE_STR_TO_TYPE[tile] for tile in hand] for hand in raw_hands
        ],
        discards=[[] for _ in range(4)],
        melds=[[] for _ in range(4)],
        scores=scores.copy(),
        dora_indicators=[
            TILE_STR_TO_TYPE.get(start_evt.get("dora_marker", ""), 0)
        ],
        dealer=start_evt.get("oya", 0),
        prevalent_wind=start_evt.get("bakaze", "E"),
    )
    reward_state = RewardState(
        hands=copy.deepcopy(raw_hands),
        melds=[[] for _ in range(4)],
        discards=[[] for _ in range(4)],
        stolen_discard_indices=[set() for _ in range(4)],
        scores=scores.copy(),
        dora_markers=[start_evt.get("dora_marker", "")],
    )
    return feature_state, reward_state


def _split_hands(events: Sequence[Dict]) -> List[List[Dict]]:
    hands = []
    current = None
    for evt in events:
        if evt.get("type") == "start_kyoku":
            if current:
                hands.append(current)
            current = [evt]
        elif current is not None:
            current.append(evt)
    if current:
        hands.append(current)
    return hands


def _event_info(
    hand_events: Sequence[Dict],
    reward_state: RewardState,
    player_idx: int,
    potential_calculator: MahjongPotentialCalculator,
) -> Dict[str, bool]:
    horas = [evt for evt in hand_events if evt.get("type") == "hora"]
    self_win = any(evt.get("actor") == player_idx for evt in horas)
    deal_in = any(
        evt.get("actor") != player_idx
        and evt.get("target") == player_idx
        for evt in horas
    )
    tsumo_loss = any(
        evt.get("actor") != player_idx
        and evt.get("target") == evt.get("actor")
        for evt in horas
    )

    draws = [evt for evt in hand_events if evt.get("type") == "ryukyoku"]
    exhaustive_draw = any(
        str(evt.get("reason", "")).lower() in EXHAUSTIVE_DRAW_REASONS
        for evt in draws
    )
    tenpai = (
        exhaustive_draw
        and potential_calculator.shanten(reward_state, player_idx) <= 0
    )
    return {
        "self_win": self_win,
        "deal_in": deal_in,
        "tsumo_loss": tsumo_loss,
        "exhaustive_draw_and_tenpai": tenpai,
        "exhaustive_draw_and_noten": exhaustive_draw and not tenpai,
    }


def _hand_event_reward(event_info: Dict[str, bool]) -> float:
    return UNIFIED_REWARD.compute_hand_end_event_reward(event_info)


def _score_reward(previous: int, current: int) -> float:
    return UNIFIED_REWARD.compute_score_delta_reward(previous, current)


def _game_reward(final_rank: int, final_score: int) -> float:
    return UNIFIED_REWARD.compute_game_end_reward(
        final_rank, final_score
    )


def _refresh_total(step: Dict) -> None:
    step["reward"] = UNIFIED_REWARD.combine_rewards(
        shape=step["shape_reward"],
        score=step["score_reward"],
        event=step["event_reward"],
        game=step["game_reward"],
    )


def _ranks(final_scores: Sequence[int]) -> List[int]:
    order = sorted(
        range(4),
        key=lambda player_idx: (final_scores[player_idx], -player_idx),
        reverse=True,
    )
    ranks = [0] * 4
    for rank, player_idx in enumerate(order, start=1):
        ranks[player_idx] = rank
    return ranks


def extract_game_trajectories(
    events: Sequence[Dict],
    potential_calculator: Optional[MahjongPotentialCalculator] = None,
) -> List[List[Dict]]:
    calculator = potential_calculator or MahjongPotentialCalculator()
    trajectories: List[List[Dict]] = [[] for _ in range(4)]
    final_scores: Optional[List[int]] = None

    hands = _split_hands(events)
    for hand_index, hand_events in enumerate(hands):
        feature_state, reward_state = _reset_states(hand_events[0])
        hand_start = [len(trajectory) for trajectory in trajectories]
        last_potential: List[Optional[float]] = [None] * 4
        last_score: List[Optional[int]] = [None] * 4

        for evt in hand_events[1:]:
            evt_type = evt.get("type")
            actor = evt.get("actor", -1)
            is_decision = evt_type in DECISION_EVENTS and 0 <= actor < 4
            if evt_type == "reach" and evt.get("step") != 1:
                is_decision = False
            if evt_type == "ryukyoku" and evt.get("reason") not in {
                "kyushukyuhai",
                "yao9",
            }:
                is_decision = False

            if is_decision:
                potential = calculator.compute_potential(reward_state, actor)
                current_score = reward_state.scores[actor]
                if last_potential[actor] is not None:
                    previous_step = trajectories[actor][-1]
                    previous_step["shape_reward"] += float(
                        np.clip(
                            0.10
                            * (
                                0.99 * potential
                                - float(last_potential[actor])
                            ),
                            -0.05,
                            0.05,
                        )
                    )
                    previous_step["score_reward"] += _score_reward(
                        int(last_score[actor]),
                        current_score,
                    )
                    _refresh_total(previous_step)

                trajectories[actor].append(
                    {
                        "features": FeatureExtractor(actor).extract_features(
                            feature_state
                        ),
                        "action": _encode_action_mjx(evt_type, evt),
                        "reward": 0.0,
                        "shape_reward": 0.0,
                        "score_reward": 0.0,
                        "event_reward": 0.0,
                        "game_reward": 0.0,
                        "potential": potential,
                    }
                )
                last_potential[actor] = potential
                last_score[actor] = current_score

            _update_feature_state(feature_state, evt)
            _update_reward_tiles(reward_state, evt)
            _apply_score_event(reward_state.scores, evt)

        if hand_index + 1 < len(hands):
            next_scores = hands[hand_index + 1][0].get("scores")
            if next_scores is not None and len(next_scores) == 4:
                reward_state.scores[:] = [
                    int(score) for score in next_scores
                ]
        final_scores = reward_state.scores.copy()
        for player_idx in range(4):
            if len(trajectories[player_idx]) == hand_start[player_idx]:
                continue
            step = trajectories[player_idx][-1]
            step["score_reward"] += _score_reward(
                int(last_score[player_idx]),
                final_scores[player_idx],
            )
            step["event_reward"] += _hand_event_reward(
                _event_info(
                    hand_events,
                    reward_state,
                    player_idx,
                    calculator,
                )
            )
            _refresh_total(step)

    if final_scores is None:
        return trajectories

    ranks = _ranks(final_scores)
    for player_idx, trajectory in enumerate(trajectories):
        if not trajectory:
            continue
        trajectory[-1]["game_reward"] += _game_reward(
            ranks[player_idx],
            final_scores[player_idx],
        )
        _refresh_total(trajectory[-1])

        rtgs = discounted_returns(
            [step["reward"] for step in trajectory],
            gamma=0.99,
        )
        for step, rtg in zip(trajectory, rtgs):
            step["rtg"] = float(rtg)
    return trajectories


def _debug_array(trajectories: Sequence[Sequence[Dict]]) -> np.ndarray:
    debug = np.zeros(
        sum(len(trajectory) for trajectory in trajectories),
        dtype=DEBUG_DTYPE,
    )
    index = 0
    for trajectory in trajectories:
        for step in trajectory:
            debug[index] = (
                step["reward"],
                step["shape_reward"],
                step["score_reward"],
                step["event_reward"],
                step["game_reward"],
                step["potential"],
            )
            index += 1
    return debug


def _save_chunk(
    output_dir: str,
    chunk_index: int,
    trajectories: Sequence[Sequence[Dict]],
    write_debug: bool,
) -> int:
    chunk_folder = os.path.join(output_dir, f"chunk_{chunk_index:03d}")
    os.makedirs(chunk_folder, exist_ok=True)
    steps = [step for trajectory in trajectories for step in trajectory]
    boundaries = np.cumsum(
        [len(trajectory) for trajectory in trajectories],
        dtype=np.int64,
    )

    np.save(
        os.path.join(chunk_folder, "features.npy"),
        np.asarray([step["features"] for step in steps], dtype=np.float32),
    )
    np.save(
        os.path.join(chunk_folder, "actions.npy"),
        np.asarray([step["action"] for step in steps], dtype=np.int64),
    )
    np.save(
        os.path.join(chunk_folder, "rtgs.npy"),
        np.asarray([step["rtg"] for step in steps], dtype=np.float32).reshape(
            -1, 1
        ),
    )
    np.save(
        os.path.join(chunk_folder, "trajectory_boundaries.npy"),
        boundaries,
    )
    if write_debug:
        np.save(
            os.path.join(chunk_folder, "rewards_debug.npy"),
            _debug_array(trajectories),
        )
    return len(steps)


def convert_mjson_directory(
    data_dir: str,
    output_dir: str,
    max_files: int = -1,
    files_per_chunk: int = 5000,
    write_debug: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    existing_chunks = glob.glob(os.path.join(output_dir, "chunk_*"))
    if existing_chunks:
        raise FileExistsError(
            f"{output_dir} already contains chunk_* directories. "
            "Use a new output directory to avoid mixing RTG versions."
        )
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.mjson")))
    files = all_files[:max_files] if max_files > 0 else all_files
    calculator = MahjongPotentialCalculator()
    print(f"Converting {len(files)}/{len(all_files)} mjson files...")

    chunk_index = 0
    total_steps = 0
    failed_files = 0
    for batch_start in range(0, len(files), files_per_chunk):
        batch_files = files[batch_start:batch_start + files_per_chunk]
        batch_trajectories = []
        print(
            f"\n[Chunk {chunk_index}] Processing files "
            f"{batch_start + 1}-{batch_start + len(batch_files)}..."
        )

        for file_index, file_path in enumerate(
            batch_files, start=batch_start + 1
        ):
            basename = os.path.basename(file_path)
            print(
                f"  [{file_index}/{len(files)}] {basename}...",
                end=" ",
                flush=True,
            )
            try:
                events = parse_mjson_file(file_path)
                trajectories = extract_game_trajectories(
                    events, calculator
                )
                valid = [
                    trajectory
                    for trajectory in trajectories
                    if trajectory
                ]
                if not valid:
                    failed_files += 1
                    print("NO ACTIONS")
                    continue
                batch_trajectories.extend(valid)
                print(f"OK ({sum(map(len, valid))} steps)")
            except Exception as exc:
                failed_files += 1
                print(f"ERROR: {exc}")

        if batch_trajectories:
            steps = _save_chunk(
                output_dir,
                chunk_index,
                batch_trajectories,
                write_debug,
            )
            total_steps += steps
            print(
                f"  Saved chunk_{chunk_index:03d}: "
                f"{steps} steps, cumulative={total_steps}"
            )
            chunk_index += 1

    print(
        f"\nDone: files={len(files)}, failed={failed_files}, "
        f"chunks={chunk_index}, steps={total_steps}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MJAI mjson to unified-reward NPY chunks"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-files", type=int, default=-1)
    parser.add_argument("--files-per-chunk", type=int, default=5000)
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Do not write rewards_debug.npy",
    )
    args = parser.parse_args()
    convert_mjson_directory(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_files=args.max_files,
        files_per_chunk=args.files_per_chunk,
        write_debug=not args.no_debug,
    )


if __name__ == "__main__":
    main()
