"""Recompute unified rewards and RTGs from terminal MJX state logs.

The state log format is the JSON-lines output produced by Mjx EnvRunner:
one terminal State JSON per hand and one file per hanchan.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MJX_ROOT = PROJECT_ROOT / "mjx"
for import_path in (PROJECT_ROOT, MJX_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import mjx

from utli.rewards import create_default_calculator, discounted_returns


DEBUG_DTYPE = np.dtype(
    [
        ("total", np.float32),
        ("shape", np.float32),
        ("score", np.float32),
        ("event", np.float32),
        ("game", np.float32),
    ]
)


def _add_components(calculator, step: Dict, **values: float) -> None:
    for name, value in values.items():
        step["components"][name] += float(value)
    step["reward"] = calculator.combine_rewards(**step["components"])


def _rank_for_player(final_scores: List[int], player_idx: int) -> int:
    sorted_players = sorted(
        range(4),
        key=lambda index: (final_scores[index], -index),
        reverse=True,
    )
    return sorted_players.index(player_idx) + 1


def build_game_trajectories(state_log_path: Path) -> List[Dict]:
    calculator = create_default_calculator()
    trajectories = {player_idx: [] for player_idx in range(4)}
    last_obs = {player_idx: None for player_idx in range(4)}
    saw_game_end = False

    with state_log_path.open("r", encoding="utf-8") as state_file:
        state_lines = [line.strip() for line in state_file if line.strip()]

    if not state_lines:
        raise ValueError(f"Empty state log: {state_log_path}")

    for line_number, state_json in enumerate(state_lines, start=1):
        state = mjx.State(state_json)
        state_proto = state.to_proto()
        if not state_proto.HasField("round_terminal"):
            raise ValueError(
                f"{state_log_path}:{line_number} is not a terminal hand state"
            )

        round_terminal = state_proto.round_terminal
        hand_players = set()
        for obs, action in state.past_decisions():
            player_idx = int(obs.who())
            hand_players.add(player_idx)
            previous = last_obs[player_idx]
            if previous is not None:
                transition = calculator.compute_transition_reward(
                    previous,
                    obs,
                    calculator.observations_in_same_hand(previous, obs),
                )
                _add_components(
                    calculator,
                    trajectories[player_idx][-1],
                    shape=transition["shape"],
                    score=transition["score"],
                )

            trajectories[player_idx].append(
                {
                    "feature": obs.to_features("decision-mamba-v0"),
                    "action": action.to_idx(),
                    "reward": 0.0,
                    "components": calculator.empty_components(),
                }
            )
            last_obs[player_idx] = obs

        final_scores = list(round_terminal.final_score.tens)
        if len(final_scores) != 4:
            raise ValueError(
                f"{state_log_path}:{line_number} has no four-player final score"
            )

        for player_idx in hand_players:
            previous = last_obs[player_idx]
            terminal_score_reward = calculator.compute_score_delta_reward(
                calculator.extract_player_score(previous, player_idx),
                final_scores[player_idx],
            )
            event_info = calculator.build_hand_end_event_info(
                round_terminal,
                list(state_proto.public_observation.events),
                player_idx,
            )
            event_reward = calculator.compute_hand_end_event_reward(event_info)
            game_reward = 0.0
            if round_terminal.is_game_over:
                game_reward = calculator.compute_game_end_reward(
                    _rank_for_player(final_scores, player_idx),
                    final_scores[player_idx],
                )
            _add_components(
                calculator,
                trajectories[player_idx][-1],
                score=terminal_score_reward,
                event=event_reward,
                game=game_reward,
            )
            last_obs[player_idx] = None

        saw_game_end = saw_game_end or round_terminal.is_game_over

    if not saw_game_end:
        raise ValueError(f"State log has no hanchan terminal: {state_log_path}")

    output = []
    for player_idx in range(4):
        steps = trajectories[player_idx]
        if not steps:
            continue
        rewards = [step["reward"] for step in steps]
        output.append(
            {
                "source": str(state_log_path),
                "player_idx": player_idx,
                "features": np.asarray(
                    [step["feature"] for step in steps], dtype=np.float32
                ),
                "actions": np.asarray(
                    [step["action"] for step in steps], dtype=np.int64
                ),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "rtgs": discounted_returns(rewards, gamma=0.99),
                "components": [
                    step["components"] for step in steps
                ],
            }
        )
    return output


def _load_state_paths(
    states_dir: Path, manifest_path: Optional[Path]
) -> List[Path]:
    if manifest_path is not None:
        paths = []
        with manifest_path.open("r", encoding="utf-8") as manifest:
            for line in manifest:
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = states_dir / path
                paths.append(path)
        return paths

    return sorted(path for path in states_dir.rglob("*") if path.is_file())


def _iter_trajectories(
    state_paths: Iterable[Path],
) -> Iterable[Dict]:
    for state_path in state_paths:
        yield from build_game_trajectories(state_path)


def _debug_array(trajectory: Dict) -> np.ndarray:
    debug = np.zeros(len(trajectory["rewards"]), dtype=DEBUG_DTYPE)
    debug["total"] = trajectory["rewards"]
    for index, components in enumerate(trajectory["components"]):
        for name in ("shape", "score", "event", "game"):
            debug[name][index] = components[name]
    return debug


def recompute_chunks(
    data_dir: Path,
    states_dir: Path,
    manifest_path: Optional[Path] = None,
    write_debug: bool = False,
    dry_run: bool = False,
    verify_features: bool = True,
) -> Dict[str, int]:
    state_paths = _load_state_paths(states_dir, manifest_path)
    if not state_paths:
        raise FileNotFoundError(f"No state logs found under {states_dir}")

    trajectory_iter = iter(_iter_trajectories(state_paths))
    chunk_folders = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith("chunk_")
    )
    if not chunk_folders:
        raise FileNotFoundError(f"No chunk_* folders found under {data_dir}")

    trajectory_count = 0
    step_count = 0
    for chunk_folder in chunk_folders:
        features = np.load(chunk_folder / "features.npy", mmap_mode="r")
        actions = np.load(chunk_folder / "actions.npy", mmap_mode="r")
        boundaries = np.load(chunk_folder / "trajectory_boundaries.npy")
        chunk_rtgs = np.zeros((len(actions), 1), dtype=np.float32)
        chunk_debug = np.zeros(len(actions), dtype=DEBUG_DTYPE)

        start = 0
        for boundary in boundaries:
            end = int(boundary)
            try:
                trajectory = next(trajectory_iter)
            except StopIteration as exc:
                raise ValueError(
                    "State logs contain fewer trajectories than the dataset"
                ) from exc

            expected_length = end - start
            actual_length = len(trajectory["actions"])
            if actual_length != expected_length:
                raise ValueError(
                    "Trajectory length mismatch at "
                    f"{chunk_folder.name}[{trajectory_count}]: "
                    f"dataset={expected_length}, log={actual_length}, "
                    f"source={trajectory['source']}, "
                    f"player={trajectory['player_idx']}"
                )
            if not np.array_equal(actions[start:end], trajectory["actions"]):
                raise ValueError(
                    "Action mismatch at "
                    f"{chunk_folder.name}[{trajectory_count}] from "
                    f"{trajectory['source']} player={trajectory['player_idx']}"
                )
            if verify_features and not np.array_equal(
                features[start:end], trajectory["features"]
            ):
                raise ValueError(
                    "Feature mismatch at "
                    f"{chunk_folder.name}[{trajectory_count}] from "
                    f"{trajectory['source']} player={trajectory['player_idx']}"
                )

            chunk_rtgs[start:end, 0] = trajectory["rtgs"]
            chunk_debug[start:end] = _debug_array(trajectory)
            trajectory_count += 1
            step_count += expected_length
            start = end

        if start != len(actions):
            raise ValueError(
                f"Last boundary in {chunk_folder} does not equal action count"
            )

        if not dry_run:
            rtg_tmp = chunk_folder / "rtgs.npy.tmp"
            with rtg_tmp.open("wb") as output:
                np.save(output, chunk_rtgs)
            os.replace(rtg_tmp, chunk_folder / "rtgs.npy")

            if write_debug:
                debug_tmp = chunk_folder / "rewards_debug.npy.tmp"
                with debug_tmp.open("wb") as output:
                    np.save(output, chunk_debug)
                os.replace(
                    debug_tmp, chunk_folder / "rewards_debug.npy"
                )

    try:
        extra = next(trajectory_iter)
    except StopIteration:
        extra = None
    if extra is not None:
        raise ValueError(
            "State logs contain more trajectories than the dataset; "
            f"first extra source={extra['source']} "
            f"player={extra['player_idx']}"
        )

    return {
        "state_logs": len(state_paths),
        "trajectories": trajectory_count,
        "steps": step_count,
        "chunks": len(chunk_folders),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute unified rewards and rtgs.npy from MJX state logs"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--states-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional ordered list of state-log paths relative to states-dir",
    )
    parser.add_argument("--write-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-feature-verification",
        action="store_true",
        help="Verify trajectory lengths/actions only",
    )
    args = parser.parse_args()

    summary = recompute_chunks(
        data_dir=args.data_dir,
        states_dir=args.states_dir,
        manifest_path=args.manifest,
        write_debug=args.write_debug,
        dry_run=args.dry_run,
        verify_features=not args.skip_feature_verification,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
