import tempfile
import unittest
from pathlib import Path

import numpy as np

from convert.convert_mjson_to_features import (
    _save_chunk,
    extract_game_trajectories,
)


class _FakePotentialCalculator:
    def compute_potential(self, state, player_idx):
        return len(state.hands[player_idx]) / 20.0

    def shanten(self, state, player_idx):
        return 1


def _start_kyoku(kyoku, scores):
    base_hand = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m",
        "8m", "9m", "1p", "2p", "3p", "4p",
    ]
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": kyoku,
        "honba": 0,
        "kyotaku": 0,
        "oya": kyoku - 1,
        "dora_marker": "1s",
        "scores": scores,
        "tehais": [base_hand.copy() for _ in range(4)],
    }


class ConvertMjsonTest(unittest.TestCase):
    def test_reward_rtg_and_output_contract(self):
        events = [
            _start_kyoku(1, [25000, 25000, 25000, 25000]),
            {"type": "tsumo", "actor": 0, "pai": "5p"},
            {
                "type": "dahai",
                "actor": 0,
                "pai": "5p",
                "tsumogiri": True,
            },
            {
                "type": "ryukyoku",
                "reason": "fanpai",
                "deltas": [0, 0, 0, 0],
            },
            _start_kyoku(2, [25000, 25000, 25000, 25000]),
            {"type": "tsumo", "actor": 0, "pai": "6p"},
            {
                "type": "dahai",
                "actor": 0,
                "pai": "6p",
                "tsumogiri": True,
            },
            {"type": "tsumo", "actor": 0, "pai": "7p"},
            {
                "type": "hora",
                "actor": 0,
                "target": 0,
                "pai": "7p",
                "deltas": [6000, -2000, -2000, -2000],
            },
            {"type": "end_game"},
        ]

        trajectories = extract_game_trajectories(
            events,
            potential_calculator=_FakePotentialCalculator(),
        )
        player_zero = trajectories[0]

        self.assertEqual(len(player_zero), 3)
        self.assertEqual(player_zero[0]["features"].shape, (1380,))
        self.assertEqual(player_zero[0]["action"], 37 + 13)
        self.assertEqual(player_zero[1]["action"], 37 + 14)
        self.assertEqual(player_zero[2]["action"], 175)

        # No potential comparison is made across the hand boundary.
        self.assertEqual(player_zero[0]["shape_reward"], 0.0)
        self.assertNotEqual(player_zero[1]["shape_reward"], 0.0)
        self.assertEqual(player_zero[2]["event_reward"], 0.25)
        self.assertGreater(player_zero[2]["score_reward"], 0.0)
        self.assertGreater(player_zero[2]["game_reward"], 1.0)

        expected_first_rtg = (
            player_zero[0]["reward"]
            + 0.99 * player_zero[1]["reward"]
            + 0.99**2 * player_zero[2]["reward"]
        )
        self.assertAlmostEqual(player_zero[0]["rtg"], expected_first_rtg)

        with tempfile.TemporaryDirectory() as temp_dir:
            step_count = _save_chunk(
                temp_dir,
                chunk_index=0,
                trajectories=[player_zero],
                write_debug=True,
            )
            chunk = Path(temp_dir) / "chunk_000"
            features = np.load(chunk / "features.npy")
            actions = np.load(chunk / "actions.npy")
            rtgs = np.load(chunk / "rtgs.npy")
            boundaries = np.load(chunk / "trajectory_boundaries.npy")
            debug = np.load(chunk / "rewards_debug.npy")

        self.assertEqual(step_count, 3)
        self.assertEqual(features.shape, (3, 1380))
        self.assertEqual(actions.shape, (3,))
        self.assertEqual(rtgs.shape, (3, 1))
        np.testing.assert_array_equal(boundaries, np.asarray([3]))
        self.assertEqual(
            debug.dtype.names,
            ("total", "shape", "score", "event", "game", "potential"),
        )


if __name__ == "__main__":
    unittest.main()
