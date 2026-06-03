"""
第二階段：多頭分權微調（Multi-Head Fine-tuning）

載入 BC 預訓練權重 → 替換為多頭輸出層 → 凍結骨幹 → 僅訓練五個專職 Linear Heads

用法：
    python train_bc_multhead.py --pretrained checkpoints/bc_model/best_bc_model.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

from dataset import BehavioralCloningDataset, bc_collate_fn
from model import DecisionMambaMultiHead
from train_bc_step import train_bc_step


# ==================== 微調初始化（Setup） ====================

def setup_multihead_finetune(
    pretrained_path: str,
    d_model: int = 512,
    action_dim: int = 181,
    state_dim: int = 1380,
    max_ep_len: int = 2048,
    finetune_lr: float = 5e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
):
    """
    兩階段微調初始化邏輯：

    1. 實例化全新的 DecisionMambaMultiHead 模型
    2. 載入舊有的 BC 預訓練權重（strict=False，自動忽略舊的單一 Linear 輸出層）
    3. 凍結所有 Mamba 骨幹參數（Backbone Freezing）
    4. 僅將多頭輸出層（五個 Linear Heads）送入 AdamW 優化器

    Args:
        pretrained_path: 舊 BC 預訓練權重檔案路徑（best_bc_model.pt）
        d_model: 隱層維度
        action_dim: 動作空間維度
        state_dim: 狀態特徵維度
        max_ep_len: 最大 episode 長度
        finetune_lr: 微調專用學習率（預設 5e-4）
        weight_decay: 權重衰減
        device: 計算設備

    Returns:
        model: 初始化完成的多頭模型
        optimizer: 僅包含多頭參數的 AdamW 優化器
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ========== Step 1: 實例化全新的多頭模型 ==========
    print("\n" + "=" * 70)
    print("🏗️  初始化 DecisionMambaMultiHead 多頭分權模型...")
    print("=" * 70)

    model = DecisionMambaMultiHead(
        d_model=d_model,
        action_dim=action_dim,
        state_dim=state_dim,
        max_ep_len=max_ep_len,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   總參數數量: {total_params:,}")

    # ========== Step 2: 載入預訓練權重（手動清洗 + strict=False） ==========
    print(f"\n📂 載入預訓練權重: {pretrained_path}")

    checkpoint = torch.load(pretrained_path, map_location=device)
    pretrained_state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 🧹 手動清洗：刪除舊的單一 head_action 權重（shape [181, 512]）
    # 因為新模型中 head_action 已被替換為 MahjongMultiHeadOutput（包含五個子 Linear），
    # 舊的 head_action.weight / head_action.bias 名稱與形狀皆不匹配。
    # 若不移除，某些 PyTorch 版本/邊界情況可能觸發 RuntimeError。
    new_multihead_prefixes = ("head_action.head_discard", "head_action.head_chow",
                              "head_action.head_pong", "head_action.head_kong",
                              "head_action.head_special")
    keys_to_remove = [
        k for k in list(pretrained_state_dict.keys())
        if k.startswith("head_action.") and not any(k.startswith(p) for p in new_multihead_prefixes)
    ]
    for k in keys_to_remove:
        del pretrained_state_dict[k]
        print(f"   🧹 已清除舊權重: {k} (shape: 不匹配的多頭結構)")

    # strict=False：自動忽略形狀不匹配的鍵，僅注入骨幹參數
    missing_keys, unexpected_keys = model.load_state_dict(
        pretrained_state_dict, strict=False
    )

    # 分析 missing keys — 只應包含新的多頭參數（將從頭訓練）
    multihead_keys = [k for k in missing_keys if k.startswith("head_action.")]
    other_missing = [k for k in missing_keys if not k.startswith("head_action.")]

    print(f"   ✅ 成功載入骨幹權重（strict=False，已手動清洗舊 Action Head）")
    print(f"   多頭層缺失鍵（正常，將從頭訓練）: {len(multihead_keys)} 個")
    for k in multihead_keys:
        print(f"      - {k}")
    if other_missing:
        print(f"   ⚠️  其他缺失鍵（非預期，請檢查）: {other_missing}")

    # 檢查 unexpected keys — 理論上應為空（已手動清除）
    if unexpected_keys:
        print(f"   ⚠️  仍存在未預期的鍵（非預期）: {len(unexpected_keys)} 個")
        for k in unexpected_keys:
            print(f"      - {k}")
    else:
        print(f"   ✅ 無殘留的未預期鍵：清洗完全成功")

    # ========== Step 3: 骨幹凍結（Backbone Freezing） ==========
    print(f"\n🔒 凍結 Mamba 骨幹網路...")

    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        if name.startswith("head_action."):
            # 多頭輸出層：保持可訓練
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            # 骨幹（embedding / input_proj / block / head_rtg / head_state）：凍結
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"   凍結參數: {frozen_count:,}")
    print(f"   可訓練參數（僅多頭輸出層）: {trainable_count:,}")
    print(f"   可訓練比例: {trainable_count / total_params * 100:.2f}%")

    # ========== Step 4: 設定優化器（過濾機制） ==========
    print(f"\n⚙️  設定 AdamW 優化器（lr={finetune_lr}, weight_decay={weight_decay}）...")

    # 過濾：只收集 requires_grad=True 的參數
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=finetune_lr, weight_decay=weight_decay)

    # 驗證優化器中的參數組數量
    opt_param_count = sum(
        p.numel() for group in optimizer.param_groups for p in group["params"]
    )
    print(f"   優化器管理的參數數量: {opt_param_count:,}")
    assert opt_param_count == trainable_count, (
        f"優化器參數數量 ({opt_param_count}) 與可訓練參數 ({trainable_count}) 不一致！"
    )
    print(f"   ✅ 過濾驗證通過")

    # ========== 模型移至設備 ==========
    model = model.to(device)
    print(f"\n🚀 模型已移至 {device}")
    print("=" * 70 + "\n")

    return model, optimizer


# ==================== 微調訓練函數 ====================

def train_multhead_finetune(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    device: str = "cuda",
    checkpoint_dir: str = "/workspace/Mahjong/checkpoints/bc_multhead",
):
    """
    多頭分權微調訓練迴圈

    與原 BC 訓練迴圈結構相同，可直接複用 train_bc_step。
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_acc = 0.0

    print(f"\n{'=' * 70}")
    print(f"🔥 開始多頭分權微調 | 設備: {device} | 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}\n")

    for epoch in range(1, num_epochs + 1):
        # ========== 訓練階段 ==========
        model.train()
        train_loss = 0.0
        train_metrics = {"action_loss": 0, "rtg_loss": 0, "state_loss": 0, "accuracy": 0}
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            loss, metrics = train_bc_step(model, batch, optimizer)

            train_loss += loss
            for key in train_metrics:
                train_metrics[key] += metrics[key]
            train_steps += 1

            print_every = max(1, min(10, len(train_loader) // 20))
            if (batch_idx + 1) % print_every == 0 or batch_idx == 0:
                print(
                    f"  Epoch {epoch}/{num_epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss:.4f} | "
                    f"Acc: {metrics['accuracy']:.4f}"
                )

        train_loss /= train_steps
        for key in train_metrics:
            train_metrics[key] /= train_steps

        # ========== 驗證階段 ==========
        val_loss = 0.0
        val_metrics = {"action_loss": 0, "rtg_loss": 0, "state_loss": 0, "accuracy": 0}
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                loss, metrics = train_bc_step(model, batch, optimizer=None)
                val_loss += loss
                for key in val_metrics:
                    val_metrics[key] += metrics[key]
                val_steps += 1

        val_loss /= val_steps
        for key in val_metrics:
            val_metrics[key] /= val_steps

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n📊 Epoch {epoch}/{num_epochs}")
        print(f"   訓練損失: {train_loss:.4f} | 準確率: {train_metrics['accuracy']:.4f}")
        print(f"   驗證損失: {val_loss:.4f} | 準確率: {val_metrics['accuracy']:.4f}")
        print(f"   學習率: {current_lr:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_metrics["accuracy"]

            best_ckpt_path = checkpoint_dir / "best_multhead_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_metrics["accuracy"],
                },
                best_ckpt_path,
            )
            print(f"   ✅ 保存最佳模型: {best_ckpt_path}")

        # 定期保存檢查點
        if epoch % 10 == 0:
            ckpt_path = checkpoint_dir / f"multhead_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_metrics["accuracy"],
                },
                ckpt_path,
            )

    print(f"\n{'=' * 70}")
    print(f"✅ 多頭分權微調完成")
    print(f"最佳驗證損失: {best_val_loss:.4f}")
    print(f"最佳驗證準確率: {best_val_acc:.4f}")
    print(f"{'=' * 70}\n")

    return model


# ==================== 主程式 ====================

def main():
    parser = argparse.ArgumentParser(description="多頭分權 BC 微調腳本")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="/workspace/Mahjong/checkpoints/bc_model_one_head/best_bc_model.pt",
        help="舊 BC 預訓練權重檔案路徑",
    )
    parser.add_argument(
        "--npz-file",
        type=str,
        default="/data/converted_features_npy",
        help="轉換後的特徵文件路徑",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="批次大小")
    parser.add_argument("--num-epochs", type=int, default=30, help="微調輪數")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="微調學習率")
    parser.add_argument("--val-split", type=float, default=0.2, help="驗證集比例")
    parser.add_argument("--device", type=str, default="cuda", help="計算設備")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument("--d-model", type=int, default=512, help="隱層維度")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker 數量"
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=2, help="每個 worker 預先載入的 batch 數量"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="限制使用的總軌跡數量（快速測試用）"
    )

    args = parser.parse_args()

    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ========== 微調初始化 ==========
    model, optimizer = setup_multihead_finetune(
        pretrained_path=args.pretrained,
        d_model=args.d_model,
        finetune_lr=args.learning_rate,
        device=args.device,
    )

    # ========== 加載數據 ==========
    print("📂 開始加載數據 (mmap 安全模式)...")
    dataset = BehavioralCloningDataset(args.npz_file)

    if args.max_samples is not None and args.max_samples < len(dataset):
        print(f"⚡ 快速測試模式：只使用前 {args.max_samples} 條軌跡")
        indices = list(range(args.max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    dataset_size = len(dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=bc_collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
    )

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

    print(f"🚀 數據加載完成")
    print(f"   訓練樣本: {train_size}")
    print(f"   驗證樣本: {val_size}\n")

    # ========== 開始微調 ==========
    train_multhead_finetune(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        device=args.device,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()