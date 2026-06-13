import math
import unittest

import numpy as np

from utli.rewards import MahjongRewardCalculator, discounted_returns


class _FakeHand:
    def __init__(self, shanten):
        self._shanten = shanten

    def shanten_number(self):
        return self._shanten


class _FakeObservation:
    def __init__(self, shanten=3, ukeire=10, dora=1, potential=None):
        self.hand = _FakeHand(shanten)
        self.ukeire = ukeire
        self.dora = dora
        self.potential = potential

    def curr_hand(self):
        return self.hand


class _FakeCalculator(MahjongRewardCalculator):
    def calculate_ukeire(self, obs):
        return obs.ukeire

    def count_dora_in_hand(self, obs):
        return obs.dora


class _PotentialCalculator(MahjongRewardCalculator):
    def compute_potential(self, obs):
        return obs.potential


class RewardFormulaTest(unittest.TestCase):
    def test_potential_formula(self):
        calculator = _FakeCalculator()
        obs = _FakeObservation(shanten=3, ukeire=10, dora=1)
        expected = (
            0.60 * (1.0 - 3.0 / 6.0)
            + 0.30 * math.log(11.0) / math.log(21.0)
            + 0.10 / 3.0
        )
        self.assertAlmostEqual(calculator.compute_potential(obs), expected)

    def test_shape_reward_and_hand_boundary(self):
        calculator = _PotentialCalculator()
        prev_obs = _FakeObservation(potential=0.2)
        obs = _FakeObservation(potential=0.3)
        self.assertAlmostEqual(
            calculator.compute_shape_reward(prev_obs, obs, True),
            0.10 * (0.99 * 0.3 - 0.2),
        )
        clipped_obs = _FakeObservation(potential=0.8)
        self.assertEqual(
            calculator.compute_shape_reward(prev_obs, clipped_obs, True),
            0.05,
        )
        self.assertEqual(
            calculator.compute_shape_reward(prev_obs, obs, False),
            0.0,
        )

    def test_score_delta_reward(self):
        expected = 0.35 * math.asinh(1.0)
        self.assertAlmostEqual(
            MahjongRewardCalculator.compute_score_delta_reward(25000, 29000),
            expected,
        )

    def test_hand_end_event_rewards(self):
        calculator = MahjongRewardCalculator()
        cases = [
            ({"self_win": True}, 0.25),
            ({"deal_in": True}, -0.60),
            ({"tsumo_loss": True}, -0.30),
            ({"exhaustive_draw_and_tenpai": True}, 0.08),
            ({"exhaustive_draw_and_noten": True}, -0.08),
            ({}, 0.0),
        ]
        for event_info, expected in cases:
            with self.subTest(event_info=event_info):
                self.assertEqual(
                    calculator.compute_hand_end_event_reward(event_info),
                    expected,
                )

    def test_game_end_reward(self):
        calculator = MahjongRewardCalculator()
        self.assertEqual(calculator.compute_game_end_reward(1, 25000), 1.0)
        self.assertEqual(calculator.compute_game_end_reward(4, 25000), -1.0)
        expected = 0.30 + 0.20 * math.asinh(12000.0 / 12000.0)
        self.assertAlmostEqual(
            calculator.compute_game_end_reward(2, 37000),
            expected,
        )

    def test_total_reward_clip(self):
        calculator = MahjongRewardCalculator()
        self.assertEqual(
            calculator.combine_rewards(
                shape=0.05, score=1.0, event=0.25, game=1.5
            ),
            2.0,
        )
        self.assertEqual(
            calculator.combine_rewards(
                shape=-0.05, score=-1.0, event=-0.60, game=-1.5
            ),
            -2.0,
        )

    def test_discounted_return_does_not_reset_at_hand_boundary(self):
        rewards = [1.0, 2.0, 3.0]
        expected = np.asarray(
            [
                1.0 + 0.99 * 2.0 + 0.99**2 * 3.0,
                2.0 + 0.99 * 3.0,
                3.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            discounted_returns(rewards, gamma=0.99),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
