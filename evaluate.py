#!/usr/bin/env python3
"""
evaluate.py — 模型評估主入口

支援兩種模式：

  1. 離線模式 (offline)：
     方式 A — 從預存 .npy 載入:
         python evaluate.py --mode offline --logits logits.npy --targets targets.npy [--mask mask.npy]
     方式 B — 從模型 + 資料集直接跑推論（使用與 train_bc.py 相同的驗證集分割）:
         python evaluate.py --mode offline --checkpoint model.pt --data-path /data/converted_features_npy
                            [--val-split 0.2] [--seed 42] [--batch-size 256] [--device cuda]

  2. 自我對弈模式 (selfplay)：
     載入模型權重，跑指定局數的自我對弈，輸出統計報告。
     用法:
         python evaluate.py --mode selfplay --checkpoint model.pt --num_games 1000 [--device cuda]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# 導入本地模組
from evaluation_metrics import (
    compute_offline_accuracy,
    compute_detailed_accuracy,
    MahjongMetricTracker,
    StreamingAccuracyTracker,
    is_draw_game,
)


# ============================================================================
#  模式 1a：從預存 .npy 檔案評估
# ============================================================================

def run_offline_eval(args):
    """載入預存的 logits / targets / mask 並計算準確率。"""
    print("=" * 60)
    print("  離線模型準確率評估 (Offline Action Accuracy)")
    print("  [來源: 預存 .npy 檔案]")
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
    _print_accuracy_results(logits, targets, mask, args.top_k)


# ============================================================================
#  模式 1b：從模型 + 資料集跑推論評估（使用 BC 驗證集分割）
# ============================================================================

def run_offline_eval_from_dataset(args):
    """
    載入模型 checkpoint 與原始資料集，用與 train_bc.py 相同的 random_split
    (val_split=0.2, seed=42) 取出驗證集，逐 batch 跑推論後計算各類別準確率。
    """
    from dataset import BehavioralCloningDataset, bc_collate_fn
    from model import DecisionMamba

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"[警告] CUDA 不可用，使用 {device}")

    print("=" * 60)
    print("  離線模型準確率評估 (Offline Action Accuracy)")
    print("  [來源: 模型推論 + BC 驗證集分割]")
    print("=" * 60)

    # ── 1. 載入資料集 ──
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"[錯誤] 資料路徑不存在: {data_path}")
        sys.exit(1)
    print(f"\n📂 載入資料集: {data_path}")
    full_dataset = BehavioralCloningDataset(str(data_path))

    # 與 train_bc.py 完全相同的隨機分割
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"   總軌跡數: {dataset_size}")
    print(f"   驗證集: {val_size} 條 ({args.val_split*100:.0f}%) | seed={args.seed}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=bc_collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
    )

    # ── 2. 載入模型 ──
    print(f"\n🔧 載入模型: {args.checkpoint}")
    model = DecisionMamba(d_model=512, action_dim=181, state_dim=1380)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[錯誤] checkpoint 不存在: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"   模型參數數: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. 逐 batch 推論，流式計數（O(1) 記憶體，避免 OOM）──
    tracker = StreamingAccuracyTracker()
    total_valid_samples = 0

    print(f"\n🚀 開始推論 ({len(val_loader)} batches)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            rtg = batch["rtg"].to(device)
            state = batch["state"].to(device)
            input_action = batch["input_action"].to(device)
            target_action = batch["target_action"]  # 保留在 CPU
            timesteps = batch["timesteps"].to(device)

            pred_action, _, _, _ = model(
                rtg=rtg, state=state, action=input_action, timesteps=timesteps
            )
            # pred_action: (B, T, 181), target_action: (B, T)

            # 提取有效樣本（過濾 padding / NO / DUMMY）
            B, T = target_action.shape
            batch_logits = []
            batch_targets = []

            for b in range(B):
                for t in range(T):
                    tid = target_action[b, t].item()
                    if tid < 0 or tid in (179, 180):
                        continue
                    batch_logits.append(pred_action[b, t])
                    batch_targets.append(tid)

            if len(batch_targets) == 0:
                continue

            # 🔥 關鍵：per-batch 立即計數，不保留 tensor
            batch_logits_t = torch.stack(batch_logits)  # 最多 B*T 個，記憶體可控
            batch_targets_t = torch.tensor(batch_targets, dtype=torch.long, device=device)
            tracker.update(batch_logits_t, batch_targets_t)
            total_valid_samples += len(batch_targets)

            # 🔥 立即釋放本 batch 的 GPU/CPU tensor
            del batch_logits, batch_targets, batch_logits_t, batch_targets_t

            if (batch_idx + 1) % max(1, len(val_loader) // 10) == 0:
                print(f"  進度: {batch_idx + 1}/{len(val_loader)} batches, "
                      f"已處理 {total_valid_samples} 有效樣本")

    print(f"\n  推論完成，共處理 {total_valid_samples} 個有效動作樣本")

    if total_valid_samples == 0:
        print("[錯誤] 沒有有效樣本可供評估")
        sys.exit(1)

    # ── 4. 計算準確率（從流式計數器直接相除）──
    acc_results = tracker.compute()
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
    print("=" * 60)


def _print_accuracy_results(logits, targets, mask, top_k=False):
    """共用的準確率計算與輸出。"""
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

    if top_k:
        detailed = compute_detailed_accuracy(logits, targets, mask)
        print()
        print("  Top-K 準確率:")
        for k in [1, 3, 5]:
            key = f"top{k}_accuracy"
            if key in detailed:
                print(f"    Top-{k} Overall : {detailed[key]*100:.2f}%")

    print("=" * 60)


# ============================================================================
#  模式 2：自我對弈統計
# ============================================================================

def run_selfplay_eval(args):
    """
    載入模型，跑指定局數的自我對弈，輸出完整統計報告。

    game_result 已由 runner.run_match() 直接提供：
      - is_agari, is_houjuu, anyone_agari, agent_rank, agent_score, final_scores
    無需再做額外推斷。
    """
    import traceback

    # 延遲導入（避免無 mjx 環境時直接報錯）
    from runner import SelfPlayRunner
    from model import DecisionMamba

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
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"  模型參數數: {sum(p.numel() for p in model.parameters()):,}")

    # 建立 Runner 與 Tracker
    runner = SelfPlayRunner(model, device=device, opponent_pool_size=args.opponent_pool)
    tracker = MahjongMetricTracker()

    num_games = args.num_games
    print(f"\n  開始自我對弈: {num_games} 局")
    print(f"  Temperature: {args.temperature}")
    print(f"  對手池大小: {args.opponent_pool}")
    print("-" * 40)

    for game_idx in range(1, num_games + 1):
        try:
            _, game_result = runner.run_match(temperature=args.temperature)
            tracker.record_game(game_result)

            # 定期更新對手池
            if game_idx % args.pool_update_interval == 0:
                runner.update_opponent_pool()

            # 定期輸出進度
            if game_idx % args.report_interval == 0:
                print(f"\n  --- 進度: {game_idx}/{num_games} 局 ---")
                tracker.print_report()

        except Exception as e:
            print(f"\n[警告] 第 {game_idx} 局發生錯誤，跳過: {e}")
            if args.verbose:
                traceback.print_exc()
            continue

    print("\n")
    print("=" * 60)
    print(f"  自我對弈完成！共 {num_games} 局")
    tracker.print_report()

    # 儲存報告到檔案
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

    # --- 離線模式參數（方式 A: 預存 .npy） ---
    parser.add_argument("--logits", type=str, default=None,
                        help="[offline-A] logits 的 .npy 檔案路徑")
    parser.add_argument("--targets", type=str, default=None,
                        help="[offline-A] target labels 的 .npy 檔案路徑")
    parser.add_argument("--mask", type=str, default=None,
                        help="[offline-A] 合法動作遮罩的 .npy 檔案路徑（選填）")

    # --- 離線模式參數（方式 B: 模型推論 + 資料集） ---
    parser.add_argument("--data-path", type=str, default=None,
                        help="[offline-B] 轉換後的資料集目錄路徑（如 /data/converted_features_npy）")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="[offline-B] 驗證集比例（預設 0.2，與 train_bc.py 一致）")
    parser.add_argument("--seed", type=int, default=42,
                        help="[offline-B] 隨機種子（預設 42，與 train_bc.py 一致）")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="[offline-B] 推論批次大小（預設 256）")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="[offline-B] DataLoader worker 數量（預設 4）")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="[offline-B] 每個 worker 預先載入的 batch 數量（預設 2）")

    # --- 共用參數 ---
    parser.add_argument("--device", type=str, default="cuda",
                        help="運算裝置（預設 cuda）")
    parser.add_argument("--top_k", action="store_true",
                        help="[offline] 是否同時計算 Top-3/Top-5 準確率")

    # --- 自我對弈模式參數 ---
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型 checkpoint .pt 路徑")
    parser.add_argument("--num_games", type=int, default=1000,
                        help="[selfplay] 自我對弈局數（預設 1000）")
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
    parser.add_argument("--verbose", action="store_true",
                        help="顯示詳細錯誤追蹤")

    args = parser.parse_args()

    if args.mode == "offline":
        # 判斷使用方式 A（.npy）或方式 B（模型+資料集）
        if args.checkpoint and args.data_path:
            run_offline_eval_from_dataset(args)
        elif args.logits and args.targets:
            run_offline_eval(args)
        else:
            parser.error(
                "離線模式需要以下其中一組參數：\n"
                "  方式 A (預存 .npy):  --logits LOGITS --targets TARGETS\n"
                "  方式 B (模型推論):   --checkpoint MODEL.pt --data-path /path/to/data"
            )

    elif args.mode == "selfplay":
        if not args.checkpoint:
            parser.error("自我對弈模式需要 --checkpoint 參數")
        run_selfplay_eval(args)


if __name__ == "__main__":
    main()