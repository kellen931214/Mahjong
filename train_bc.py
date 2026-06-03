"""
第一階段：行為模仿（Behavioral Cloning）訓練
使用 DecisionMamba 模型和 mjx 的 181 種動作編碼
"""

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# 📌 確保從你的本地模組正確導入
from dataset import BehavioralCloningDataset, bc_collate_fn
from model import DecisionMamba
from train_bc_step import train_bc_step


# ==================== 訓練函數 ====================

def train_bc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = 'cuda',
    checkpoint_dir: str = '/workspace/Mahjong/checkpoints/'
):
    """
    BC訓練循環
    
    Args:
        model: BC模型 (DecisionMamba)
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        num_epochs: 訓練輪數
        learning_rate: 學習率
        device: 計算設備
        checkpoint_dir: 模型保存路徑
    """
    # 設置設備
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 優化器和調度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 創建檢查點目錄
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 訓練日誌
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accs = []
    
    print(f"\n{'='*70}")
    print(f"開始BC訓練 | 設備: {device} | 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, num_epochs + 1):
        # ============ 訓練階段 ============
        model.train()
        train_loss = 0.0
        train_metrics = {'action_loss': 0, 'rtg_loss': 0, 'state_loss': 0, 'accuracy': 0}
        train_steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 使用 train_bc_step 進行訓練
            loss, metrics = train_bc_step(model, batch, optimizer)
            
            train_loss += loss
            for key in train_metrics:
                train_metrics[key] += metrics[key]
            train_steps += 1
            
            # 進度信息（每 10 個 batch 或每 5% 的進度印一次，取較頻繁者）
            print_every = max(1, min(10, len(train_loader) // 20))
            if (batch_idx + 1) % print_every == 0 or batch_idx == 0:
                print(f"  Epoch {epoch}/{num_epochs} | "
                      f"Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss:.4f} | "
                      f"Acc: {metrics['accuracy']:.4f}")
        
        # 計算平均訓練指標
        train_loss /= train_steps
        for key in train_metrics:
            train_metrics[key] /= train_steps
        train_losses.append(train_loss)
        
        # ============ 驗證階段 ============
        val_loss = 0.0
        val_metrics = {'action_loss': 0, 'rtg_loss': 0, 'state_loss': 0, 'accuracy': 0}
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 直接呼叫 train_bc_step
                loss, metrics = train_bc_step(model, batch, optimizer=None)
                
                # 累加損失與指標
                val_loss += loss
                for key in val_metrics:
                    val_metrics[key] += metrics[key]
                val_steps += 1
        
        # 計算平均驗證指標 
        val_loss /= val_steps
        for key in val_metrics:
            val_metrics[key] /= val_steps
        val_losses.append(val_loss)
        val_accs.append(val_metrics['accuracy'])
        
        # 學習率調度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 進度報告
        print(f"\n📊 Epoch {epoch}/{num_epochs}")
        print(f"   訓練損失: {train_loss:.4f} | 準確率: {train_metrics['accuracy']:.4f}")
        print(f"   驗證損失: {val_loss:.4f} | 準確率: {val_metrics['accuracy']:.4f}")
        print(f"   學習率: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_metrics['accuracy']
            
            best_ckpt_path = checkpoint_dir / 'best_bc_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_metrics['accuracy']
            }, best_ckpt_path)
            
            print(f"   ✅ 保存最佳模型: {best_ckpt_path}")
        
        # 定期保存檢查點
        if epoch % 10 == 0:
            ckpt_path = checkpoint_dir / f'bc_model_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_metrics['accuracy']
            }, ckpt_path)
    
    # 訓練完成
    print(f"\n{'='*70}")
    print(f"✅ BC訓練完成")
    print(f"最佳驗證損失: {best_val_loss:.4f}")
    print(f"最佳驗證準確率: {best_val_acc:.4f}")
    print(f"{'='*70}\n")
    
    # 保存訓練日誌
    log_path = checkpoint_dir / 'training_log.npz'
    np.savez(
        log_path,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        val_accs=np.array(val_accs)
    )
    print(f"訓練日誌已保存: {log_path}\n")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='BC訓練腳本')
    parser.add_argument('--npz-file', type=str, 
                       default='/data/converted_features_npy',
                       help='轉換後的特徵文件路徑或包含 Chunks 的目錄（預設位於 NVMe 高速儲存）')
    parser.add_argument('--batch-size', type=int, default=256, help='批次大小')
    parser.add_argument('--num-epochs', type=int, default=100, help='訓練輪數')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='學習率')
    parser.add_argument('--val-split', type=float, default=0.2, help='驗證集比例')
    parser.add_argument('--device', type=str, default='cuda', help='計算設備')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--d-model', type=int, default=512, help='隐层维度')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader 的 worker 數量（NVMe/SSD 建議 4~8，HDD 建議 1~2）')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='每個 worker 預先載入的 batch 數量')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='限制使用的總軌跡數量（用於快速測試，預設使用全部）')
    
    args = parser.parse_args()
    
    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加載數據
    print("📂 開始加載與掃描數據 (mmap 安全模式)...")
    dataset = BehavioralCloningDataset(args.npz_file)
    
    # 如果指定 max_samples，只取前 N 條
    if args.max_samples is not None and args.max_samples < len(dataset):
        print(f"⚡ 快速測試模式：只使用前 {args.max_samples} 條軌跡")
        indices = list(range(args.max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # 分割訓練集和驗證集
    dataset_size = len(dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # ✨ NVMe 環境下可安全使用較多 worker，prefetch 與 pin_memory 進一步加速
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
    
    print(f"🚀 數據加載與 Dataloader 建立完成")
    print(f"   訓練樣本: {train_size}")
    print(f"   驗證樣本: {val_size}\n")
    
    # 創建模型
    model = DecisionMamba(
        d_model=args.d_model,
        action_dim=181,  # mjx 的完整動作編碼
        state_dim=1380,  # 我們的特徵維度
        max_ep_len=2048
    )
    
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # 訓練
    model = train_bc(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir='/workspace/Mahjong/checkpoints/bc_model'
    )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()