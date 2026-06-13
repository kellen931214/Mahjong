from __future__ import annotations

import math
import sys
from enum import IntEnum
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

_mjx_path = Path(__file__).resolve().parent.parent / "mjx"
if str(_mjx_path) not in sys.path:
    sys.path.insert(0, str(_mjx_path))

try:
    import mjx
    from mjx.const import EventType
except ModuleNotFoundError as import_error:
    if import_error.name != "_mjx":
        raise
    mjx = None

    class EventType(IntEnum):
        DISCARD = 0
        TSUMOGIRI = 1
        RIICHI = 2
        CLOSED_KAN = 3
        ADDED_KAN = 4
        CHI = 7
        PON = 8
        OPEN_KAN = 9
        RIICHI_SCORE_CHANGE = 13
        ABORTIVE_DRAW_NORMAL = 19


REWARD_COMPONENTS = ("shape", "score", "event", "game")


class MahjongRewardCalculator:
    """Unified reward used by offline RTG generation and PPO rollout."""

    def __init__(self) -> None:
        # Kept for CLI/checkpoint compatibility. Both modes use this reward.
        self.current_mode = "unified"

    def set_mode(self, mode_str: str) -> None:
        if mode_str not in ("attack", "defense", "unified"):
            raise ValueError(
                f"Unsupported mode '{mode_str}'; expected attack, defense, or unified"
            )
        self.current_mode = "unified"

    def get_mode_tensor(self) -> torch.Tensor:
        """Compatibility helper; reward mode no longer changes the objective."""
        if torch is None:
            raise RuntimeError("PyTorch is required for get_mode_tensor()")
        return torch.tensor([1.0, 0.0], dtype=torch.float32)

    @staticmethod
    def _tile_list_to_34_count(tiles) -> np.ndarray:
        counts = np.zeros(34, dtype=np.int32)
        for tile in tiles:
            counts[tile.type()] += 1
        return counts

    def count_visible_tiles(self, obs: mjx.Observation, tile_type: int) -> int:
        visible_tile_ids = set()
        hand = obs.curr_hand()
        visible_tile_ids.update(tile.id() for tile in hand.closed_tiles())

        for open_meld in hand.opens():
            visible_tile_ids.update(tile.id() for tile in open_meld.tiles())

        for event in obs.events():
            if event.type() in (
                EventType.CHI,
                EventType.PON,
                EventType.OPEN_KAN,
                EventType.CLOSED_KAN,
                EventType.ADDED_KAN,
            ):
                open_obj = event.open()
                if open_obj is not None:
                    visible_tile_ids.update(
                        tile.id() for tile in open_obj.tiles()
                    )
            elif event.type() in (EventType.DISCARD, EventType.TSUMOGIRI):
                tile = event.tile()
                if tile is not None:
                    visible_tile_ids.add(tile.id())

        proto = obs.to_proto()
        visible_tile_ids.update(
            int(tile_id)
            for tile_id in proto.public_observation.dora_indicators
        )
        return min(
            sum(tile_id // 4 == tile_type for tile_id in visible_tile_ids),
            4,
        )

    def count_dora_in_hand(self, obs: mjx.Observation) -> int:
        real_dora_types = [int(dora) for dora in obs.doras()]
        dora_count = 0
        for tile in obs.curr_hand().closed_tiles():
            dora_count += int(tile.is_red())
            dora_count += int(tile.type() in real_dora_types)
        for meld in obs.curr_hand().opens():
            for tile in meld.tiles():
                dora_count += int(tile.is_red())
                dora_count += int(tile.type() in real_dora_types)
        return dora_count

    def calculate_ukeire(self, obs: mjx.Observation) -> int:
        total_ukeire = 0
        for tile_type in obs.curr_hand().effective_draw_types():
            visible = self.count_visible_tiles(obs, int(tile_type))
            total_ukeire += max(0, 4 - visible)
        return total_ukeire

    def compute_potential(self, obs: mjx.Observation) -> float:
        shanten = obs.curr_hand().shanten_number()
        ukeire = self.calculate_ukeire(obs)
        dora_count = self.count_dora_in_hand(obs)

        f_shanten = 1.0 - min(float(shanten), 6.0) / 6.0
        f_ukeire = math.log1p(min(float(ukeire), 20.0)) / math.log(21.0)
        f_dora = min(float(dora_count), 3.0) / 3.0
        return 0.60 * f_shanten + 0.30 * f_ukeire + 0.10 * f_dora

    def compute_shape_reward(
        self,
        prev_obs: Optional[mjx.Observation],
        obs: Optional[mjx.Observation],
        same_hand: bool,
    ) -> float:
        if prev_obs is None or obs is None or not same_hand:
            return 0.0
        reward = 0.10 * (
            0.99 * self.compute_potential(obs) - self.compute_potential(prev_obs)
        )
        return float(np.clip(reward, -0.05, 0.05))

    @staticmethod
    def compute_score_delta_reward(prev_score: float, curr_score: float) -> float:
        return 0.35 * math.asinh((float(curr_score) - float(prev_score)) / 4000.0)

    @staticmethod
    def compute_hand_end_event_reward(event_info: Mapping[str, bool]) -> float:
        if event_info.get("self_win", False):
            return 0.25
        if event_info.get("deal_in", False):
            return -0.60
        if event_info.get("tsumo_loss", False):
            return -0.30
        if event_info.get("exhaustive_draw_and_tenpai", False):
            return 0.08
        if event_info.get("exhaustive_draw_and_noten", False):
            return -0.08
        return 0.0

    @staticmethod
    def compute_game_end_reward(final_rank: int, final_score: float) -> float:
        rank_bonus = {1: 1.00, 2: 0.30, 3: -0.30, 4: -1.00}
        if final_rank not in rank_bonus:
            raise ValueError(f"final_rank must be in [1, 4], got {final_rank}")
        return rank_bonus[final_rank] + 0.20 * math.asinh(
            (float(final_score) - 25000.0) / 12000.0
        )

    @staticmethod
    def combine_rewards(
        shape: float = 0.0,
        score: float = 0.0,
        event: float = 0.0,
        game: float = 0.0,
    ) -> float:
        return float(np.clip(shape + score + event + game, -2.0, 2.0))

    @staticmethod
    def empty_components() -> Dict[str, float]:
        return {name: 0.0 for name in REWARD_COMPONENTS}

    def compute_transition_reward(
        self,
        prev_obs: mjx.Observation,
        obs: mjx.Observation,
        same_hand: bool,
    ) -> Dict[str, float]:
        components = self.empty_components()
        components["shape"] = self.compute_shape_reward(prev_obs, obs, same_hand)
        if same_hand:
            player_idx = int(prev_obs.who())
            components["score"] = self.compute_score_delta_reward(
                self.extract_player_score(prev_obs, player_idx),
                self.extract_player_score(obs, player_idx),
            )
        components["total"] = self.combine_rewards(**components)
        return components

    @staticmethod
    def hand_key(obs: mjx.Observation) -> tuple:
        proto = obs.to_proto()
        public = proto.public_observation
        init_score = public.init_score
        return (public.game_id, init_score.round, init_score.honba)

    @classmethod
    def observations_in_same_hand(
        cls, prev_obs: Optional[mjx.Observation], obs: Optional[mjx.Observation]
    ) -> bool:
        if prev_obs is None or obs is None:
            return False
        return cls.hand_key(prev_obs) == cls.hand_key(obs)

    @staticmethod
    def extract_player_score(obs: mjx.Observation, player_idx: Optional[int] = None) -> int:
        proto = obs.to_proto()
        if player_idx is None:
            player_idx = int(obs.who())
        if proto.HasField("round_terminal"):
            tens = proto.round_terminal.final_score.tens
            if len(tens) == 4:
                return int(tens[player_idx])

        scores = list(proto.public_observation.init_score.tens)
        if len(scores) != 4:
            return 25000
        for event in proto.public_observation.events:
            if event.type == int(EventType.RIICHI_SCORE_CHANGE):
                scores[event.who] -= 1000
        return int(scores[player_idx])

    @staticmethod
    def build_hand_end_event_info(
        round_terminal,
        public_events: Sequence,
        player_idx: int,
    ) -> Dict[str, bool]:
        wins = list(round_terminal.wins)
        self_win = any(win.who == player_idx for win in wins)
        deal_in = any(
            win.who != player_idx and win.from_who == player_idx for win in wins
        )
        tsumo_loss = any(
            win.who != player_idx and win.from_who == win.who for win in wins
        )

        exhaustive_draw = bool(public_events) and (
            public_events[-1].type == int(EventType.ABORTIVE_DRAW_NORMAL)
        )
        tenpai_players = (
            {tenpai.who for tenpai in round_terminal.no_winner.tenpais}
            if round_terminal.HasField("no_winner")
            else set()
        )
        return {
            "self_win": self_win,
            "deal_in": deal_in,
            "tsumo_loss": tsumo_loss,
            "exhaustive_draw_and_tenpai": (
                exhaustive_draw and player_idx in tenpai_players
            ),
            "exhaustive_draw_and_noten": (
                exhaustive_draw and player_idx not in tenpai_players
            ),
        }


def discounted_returns(rewards: Sequence[float], gamma: float = 0.99) -> np.ndarray:
    rtgs = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        rtgs[index] = running
    return rtgs


def create_default_calculator() -> MahjongRewardCalculator:
    return MahjongRewardCalculator()
