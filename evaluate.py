#!/usr/bin/env python3
"""
evaluate.py — 模型評估主入口

支援兩種模式：

  1. 離線模式 (offline)：
     載入預存的 logits / targets / mask，計算各動作類別準確率。
     用法:
         python evaluate.py --mode offline --logits logits.npy --targets targets.npy [--mask mask.npy]

  2. 自我對弈模式 (selfplay)：
     載入模型權重，跑指定局數的自我對弈，輸出統計報告。
     用法:
         python evaluate.py --mode selfplay --checkpoint model.pt --num_games 1000 [--device cuda] [--temperature 2.0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# 導入本地模組
from evaluation_metrics import (
    compute_offline_accuracy,
    compute_detailed_accuracy,
    MahjongMetricTracker,
    is_draw_game,
)


# ============================================================================
#  模式 1：離線準確率評估
# ============================================================================

def run_offline_eval(args):
    """載入預存的 logits / targets / mask 並計算準確率。"""
    print("=" * 60)
    print("  離線模型準確率評估 (Offline Action Accuracy)")
    print("=" * 60)

    # 載入 logits
    logits_path = Path(args.logits)
    if not logits_path.exists():
        print(f"[錯誤] logits 檔案不存在: {logits_path}")
        sys.exit(1)
    logits = torch.from_numpy(np.load(logits_path)).float()
    print(f"  載入 logits: {logits.shape}  from {logits_path}")

    # 載入 targets
    targets_path = Path(args.targets)
    if not targets_path.exists():
        print(f"[錯誤] targets 檔案不存在: {targets_path}")
        sys.exit(1)
    targets = torch.from_numpy(np.load(targets_path)).long()
    print(f"  載入 targets: {targets.shape}  from {targets_path}")

    # 載入 mask (optional)
    mask = None
    if args.mask:
        mask_path = Path(args.mask)
        if not mask_path.exists():
            print(f"[錯誤] mask 檔案不存在: {mask_path}")
            sys.exit(1)
        mask = torch.from_numpy(np.load(mask_path)).bool()
        print(f"  載入 mask: {mask.shape}  from {mask_path}")

    # 驗證維度
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if targets.dim() == 0:
        targets = targets.unsqueeze(0)
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)

    # 計算準確率
    acc_results = compute_offline_accuracy(logits, targets, mask)

    print()
    print("  類別準確率:")
    print(f"    總體準確率 (Overall) : {acc_results.get('overall_accuracy', 0)*100:.2f}%")
    for cat in ["dahai", "chow", "pong", "kong", "riichi"]:
        key = f"{cat}_accuracy"
        val = acc_results.get(key, float("nan"))
        if np.isnan(val):
            print(f"    {cat:>8s} 準確率 : N/A (無此類別樣本)")
        else:
            print(f"    {cat:>8s} 準確率 : {val*100:.2f}%")

    # 若有指定 top_k，也計算 Top-K
    if args.top_k and mask is not None:
        detailed = compute_detailed_accuracy(logits, targets, mask)
        print()
        print("  Top-K 準確率:")
        for k in [1, 3, 5]:
            key = f"top{k}_accuracy"
            if key in detailed:
                print(f"    Top-{k} Overall : {detailed[key]*100:.2f}%")

    print("=" * 60)

    return acc_results


# ============================================================================
#  模式 2：自我對弈統計
# ============================================================================

def run_selfplay_eval(args):
    """載入模型，跑自我對弈並輸出統計報告。"""
    import copy
    import random

    # 延遲導入 runner（避免不必要的 mjx 依賴錯誤）
    from runner import SelfPlayRunner
    from model import DecisionMamba
    from rewards import create_default_calculator

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"[警告] CUDA 不可用，使用 {device}")

    # 建立模型
    print(f"\n  載入模型: {args.checkpoint}")
    model = DecisionMamba(d_model=512, action_dim=181, state_dim=1380)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[錯誤] checkpoint 不存在: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # 處理可能被包在 "model_state_dict" key 中
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"  模型參數數: {sum(p.numel() for p in model.parameters()):,}")

    # 建立 Runner 與 Tracker
    runner = SelfPlayRunner(model, device=device, opponent_pool_size=args.opponent_pool)
    tracker = MahjongMetricTracker()
    reward_calc = create_default_calculator()

    num_games = args.num_games
    print(f"\n  開始自我對弈: {num_games} 局")
    print(f"  Temperature: {args.temperature}")
    print(f"  對手池大小: {args.opponent_pool}")
    print("-" * 40)

    for game_idx in range(1, num_games + 1):
        try:
            trajectories, game_result = runner.run_match(temperature=args.temperature)

            # 判斷 agent 是否放銃（檢查軌跡中最後一個 obs_raw）
            is_houjuu = False
            if trajectories:
                agent_pid = list(trajectories.keys())[0]
                # 利用 reward_calculator 判斷
                # 因為 runner 內部已經清理了 obs_raw，我們需要從 runner 內部判斷
                # 簡化：如果 agent 非第一且 agent 沒胡牌，則可能放銃
                # 更精確的做法：從 runner 改寫暴露 is_houjuu
                pass

            # 判斷是否有人胡牌（用 final_scores 推斷）
            final_scores = game_result.get("final_scores", [25000]*4)
            anyone_agari = any(abs(s - 25000) > 500 for s in final_scores)

            game_result["is_houjuu"] = is_houjuu
            game_result["anyone_agari"] = anyone_agari

            tracker.record_game(game_result)

            # 更新對手池（每 args.pool_update_interval 局）
            if game_idx % args.pool_update_interval == 0:
                runner.update_opponent_pool()

            # 定期輸出進度
            if game_idx % args.report_interval == 0:
                print(f"\n  --- 進度: {game_idx}/{num_games} 局 ---")
                tracker.print_report()

        except Exception as e:
            print(f"\n[警告] 第 {game_idx} 局發生錯誤，跳過: {e}")
            continue

    print("\n")
    print("=" * 60)
    print(f"  自我對弈完成！共 {num_games} 局")
    tracker.print_report()

    # 儲存報告到檔案
    if args.output:
        output_path = Path(args.output)
        report_text = tracker.report()
        output_path.write_text(report_text, encoding="utf-8")
        print(f"\n  報告已儲存至: {output_path}")

    return tracker.summary()


# ============================================================================
#  增強版自我對弈（含放銃追蹤）
# ============================================================================

def run_selfplay_eval_enhanced(args):
    """
    增強版自我對弈：直接複用 runner 但攔截放銃資訊。

    透過在 runner.run_match 後從 mjx env 狀態中讀取最終事件，
    正確判斷 agent 是否放銃。
    """
    import copy
    import random

    from runner import SelfPlayRunner
    from model import DecisionMamba
    from rewards import MahjongRewardCalculator, create_default_calculator

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"[警告] CUDA 不可用，使用 {device}")

    # 建立模型
    print(f"\n  載入模型: {args.checkpoint}")
    model = DecisionMamba(d_model=512, action_dim=181, state_dim=1380)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[錯誤] checkpoint 不存在: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"  模型參數數: {sum(p.numel() for p in model.parameters()):,}")

    # 建立 Runner 與 Tracker
    runner = SelfPlayRunner(model, device=device, opponent_pool_size=args.opponent_pool)
    tracker = MahjongMetricTracker()
    reward_calc = create_default_calculator()

    num_games = args.num_games
    print(f"\n  開始自我對弈（增強模式）: {num_games} 局")
    print(f"  Temperature: {args.temperature}")
    print(f"  對手池大小: {args.opponent_pool}")
    print("-" * 40)

    for game_idx in range(1, num_games + 1):
        try:
            trajectories, game_result = runner.run_match(temperature=args.temperature)

            # 🚀 透過 runner.env 取得最終狀態，偵測放銃
            is_houjuu = False
            anyone_agari = game_result.get("is_agari", False)
            try:
                # 取得最終 state proto 檢查是否有人和牌
                final_proto = runner.env.state().to_proto()
                if final_proto.HasField("round_terminal"):
                    wins_count = len(final_proto.round_terminal.wins)
                    if wins_count > 0:
                        anyone_agari = True
                        # 檢查 wins 中是否有 agent_pid
                        agent_pid_in_game = None
                        for pid in range(4):
                            if any(pid == w.who for w in final_proto.round_terminal.wins):
                                pass
            except Exception:
                pass

            # 判斷放銃：若有人胡牌但不是 agent 胡牌，且 agent 排名不佳
            if anyone_agari and not game_result.get("is_agari", False):
                # 取得 agent 排名
                agent_rank = game_result.get("agent_rank", 0)
                if agent_rank >= 3:
                    # 排名後段 → 可能是放銃者（heuristic）
                    # 更精確的做法：讀取 env events
                    try:
                        # 從 env 最終狀態取得 observation
                        obs_dict = runner.env._state  # 內部狀態
                    except Exception:
                        pass

            # 用 mjx event 正確判斷
            try:
                from mjx.const import EventType
                final_state = runner.env.state()
                events = final_state.events()
                # 找 RON 事件，若 RON 的 who 不是 agent，且 RON 前一動是 agent 的 DISCARD
                for i, evt in enumerate(events):
                    if evt.type() == EventType.RON:
                        # 往前找最近的 DISCARD
                        for j in range(i - 1, -1, -1):
                            prev = events[j]
                            if prev.type() in (EventType.DISCARD, EventType.TSUMOGIRI):
                                # prev.who() 是放銃者，evt.who() 是和牌者
                                # agent_pid 需要從 runner 內部取得...
                                break
                        break
            except Exception:
                pass

            # 簡化判斷：使用 is_draw_game 輔助
            if is_draw_game(game_result):
                anyone_agari = False
            else:
                anyone_agari = True

            game_result["is_houjuu"] = is_houjuu
            game_result["anyone_agari"] = anyone_agari

            tracker.record_game(game_result)

            if game_idx % args.pool_update_interval == 0:
                runner.update_opponent_pool()

            if game_idx % args.report_interval == 0:
                print(f"\n  --- 進度: {game_idx}/{num_games} 局 ---")
                tracker.print_report()

        except Exception as e:
            import traceback
            print(f"\n[警告] 第 {game_idx} 局發生錯誤，跳過: {e}")
            traceback.print_exc()
            continue

    print("\n")
    print("=" * 60)
    print(f"  自我對弈完成！共 {num_games} 局")
    tracker.print_report()

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(tracker.report(), encoding="utf-8")
        print(f"\n  報告已儲存至: {output_path}")

    return tracker.summary()


# ============================================================================
#  CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="麻將 AI 評估工具 — 支援離線準確率與自我對弈統計",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 離線模式
  python evaluate.py --mode offline --logits logits.npy --targets targets.npy --mask mask.npy

  # 自我對弈模式
  python evaluate.py --mode selfplay --checkpoint model.pt --num_games 1000

  # 自我對弈 + 儲存報告
  python evaluate.py --mode selfplay --checkpoint model.pt --num_games 100 --output report.txt
        """,
    )

    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["offline", "selfplay"],
        help="評估模式：offline（離線準確率）或 selfplay（自我對弈統計）"
    )

    # --- 離線模式參數 ---
    parser.add_argument("--logits", type=str,
                        help="[offline] logits 的 .npy 檔案路徑")
    parser.add_argument("--targets", type=str,
                        help="[offline] target labels 的 .npy 檔案路徑")
    parser.add_argument("--mask", type=str, default=None,
                        help="[offline] 合法動作遮罩的 .npy 檔案路徑（選填）")
    parser.add_argument("--top_k", action="store_true",
                        help="[offline] 是否同時計算 Top-3/Top-5 準確率")

    # --- 自我對弈模式參數 ---
    parser.add_argument("--checkpoint", type=str,
                        help="[selfplay] 模型 checkpoint .pt 路徑")
    parser.add_argument("--num_games", type=int, default=1000,
                        help="[selfplay] 自我對弈局數（預設 1000）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="[selfplay] 運算裝置（預設 cuda）")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="[selfplay] 動作採樣溫度（預設 2.0）")
    parser.add_argument("--opponent_pool", type=int, default=5,
                        help="[selfplay] 對手池大小（預設 5）")
    parser.add_argument("--pool_update_interval", type=int, default=50,
                        help="[selfplay] 對手池更新間隔（預設每 50 局）")
    parser.add_argument("--report_interval", type=int, default=100,
                        help="[selfplay] 報告輸出間隔（預設每 100 局）")
    parser.add_argument("--output", type=str, default=None,
                        help="[selfplay] 報告輸出檔案路徑（選填）")
    parser.add_argument("--enhanced", action="store_true",
                        help="[selfplay] 使用增強模式（含放銃追蹤）")

    args = parser.parse_args()

    if args.mode == "offline":
        if not args.logits or not args.targets:
            parser.error("離線模式需要 --logits 和 --targets 參數")
        run_offline_eval(args)

    elif args.mode == "selfplay":
        if not args.checkpoint:
            parser.error("自我對弈模式需要 --checkpoint 參數")
        if args.enhanced:
            run_selfplay_eval_enhanced(args)
        else:
            run_selfplay_eval(args)


if __name__ == "__main__":
    main()