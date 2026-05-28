"""
第二階段：PPO 自我博弈線上微調 (LoRA + Decision Mamba)
========================================================
使用 LoRA (Low-Rank Adaptation) 技術凍結 BC 預訓練的 Decision Mamba backbone，
僅訓練 LoRA 注入層 + Actor Head + Critic Head，以防止專家策略崩塌並節省顯存。

訓練流程：
  1. 載入 BC 預訓練權重
  2. 初始化 DecisionMamba → 呼叫 prepare_for_ppo() 啟用 LoRA
  3. 建立 SelfPlayRunner 進行自我博弈收集軌跡
  4. 建立 Optimizer，只優化 requires_grad=True 的參數（LoRA + 雙頭）
  5. 主訓練迴圈：train_ppo_epoch() → 記錄指標 → 更新對手池 → 保存 checkpoint

PPO 超參數：
  - 剪裁範圍 ε = 0.2
  - Value loss 係數 c1 = 1.0
  - Entropy bonus 係數 c2 = 0.01
  - GAE γ = 0.99, λ = 0.95
  - 每局軌跡更新 epochs = 4
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys

# 確保本地模組可導入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import DecisionMamba
from runner import SelfPlayRunner
from train_ppo_step import train_ppo_epoch


# ==================== PPO 主訓練函數 ====================

def train_ppo(
    bc_checkpoint_path: str = "/workspace/Mahjong/checkpoints/bc_model/best_bc_model.pt",
    num_iterations: int = 1000,
    ppo_epochs: int = 4,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    checkpoint_dir: str = "/workspace/Mahjong/checkpoints/ppo_model",
    opponent_pool_size: int = 5,
    update_opponent_every: int = 10,
    save_every: int = 50,
    log_every: int = 1,
    d_model: int = 512,
    action_dim: int = 181,
    state_dim: int = 1380,
    max_ep_len: int = 2048,
):
    """
    LoRA + PPO 自我博弈線上微調主訓練迴圈。

    Args:
        bc_checkpoint_path: BC 預訓練模型權重路徑
        num_iterations: PPO 訓練迭代次數（每迭代 = 一局自我博弈 + PPO 更新）
        ppo_epochs: 每條軌跡的 PPO 更新 epoch 數
        learning_rate: 學習率（僅作用於 LoRA + Actor/Critic head）
        device: 計算設備
        checkpoint_dir: PPO 模型保存路徑
        opponent_pool_size: 對手池大小
        update_opponent_every: 每 N 次迭代更新一次對手池
        save_every: 每 N 次迭代保存一次 checkpoint
        log_every: 每 N 次迭代輸出一次日誌
        d_model: 隱藏層維度
        action_dim: 動作空間維度（mjx 使用 181 種動作）
        state_dim: 狀態特徵維度（decision-mamba-v0 = 1380 維）
        max_ep_len: 最大軌跡長度
    """
    # ========== 1. 設備設置 ==========
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用設備: {device}")

    # ========== 2. 載入 BC 預訓練模型 ==========
    print(f"\n📂 載入 BC 預訓練權重: {bc_checkpoint_path}")
    model = DecisionMamba(
        d_model=d_model,
        action_dim=action_dim,
        state_dim=state_dim,
        max_ep_len=max_ep_len,
    )

    checkpoint = torch.load(bc_checkpoint_path, map_location=device)
    # 支援直接載入 state_dict 或包裝後的 checkpoint dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"   ✅ 載入 BC checkpoint (epoch {checkpoint.get('epoch', '?')})")
        print(f"   BC val_loss: {checkpoint.get('val_loss', '?'):.4f}, val_acc: {checkpoint.get('val_acc', '?'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("   ✅ 載入原始 state_dict")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型總參數量: {total_params:,}")

    # ========== 3. 啟用 LoRA：凍結 backbone，只訓練 LoRA + Actor/Critic head ==========
    print("\n🔒 啟用 LoRA 模式（凍結 Backbone，只訓練 LoRA 注入層 + Actor/Critic Head）...")
    model.prepare_for_ppo()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"   可訓練參數 (LoRA + Heads): {trainable_params:,}")
    print(f"   凍結參數 (Backbone):       {frozen_params:,}")
    print(f"   可訓練比例: {trainable_params / total_params * 100:.2f}%")

    # ========== 4. 建立 Optimizer（只優化 requires_grad=True 的參數）==========
    # 📌 嚴格遵循 Readme 要求：optimizer 必須只針對可訓練參數
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    print(f"\n⚡ Optimizer: AdamW (lr={learning_rate}, weight_decay=1e-4)")
    print(f"   Optimizer 管理參數數量: {sum(p.numel() for group in optimizer.param_groups for p in group['params']):,}")

    # ========== 5. 建立 SelfPlayRunner ==========
    print(f"\n🎮 初始化 SelfPlayRunner（對手池大小={opponent_pool_size}）...")
    runner = SelfPlayRunner(
        model=model,
        device=str(device),
        opponent_pool_size=opponent_pool_size,
    )
    print("   ✅ Runner 初始化完成")

    # ========== 6. 準備 Checkpoint 目錄與日誌 ==========
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 訓練指標歷史記錄
    history = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "avg_reward": [],
        "trajectory_length": [],
    }

    # ========== 7. 主訓練迴圈 ==========
    print(f"\n{'='*70}")
    print(f"🚀 開始 PPO + LoRA 自我博弈訓練")
    print(f"   迭代次數: {num_iterations}")
    print(f"   PPO epochs/局: {ppo_epochs}")
    print(f"   對手池更新頻率: 每 {update_opponent_every} 局")
    print(f"   開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    best_avg_reward = -float("inf")

    for iteration in range(1, num_iterations + 1):
        # -------- 7a. 執行一局自我博弈 + PPO 更新 --------
        # 📌 train_ppo_epoch 內部會：
        #     1. 呼叫 model.prepare_for_ppo() 確保 LoRA 模式
        #     2. runner.run_match() 收集軌跡
        #     3. 計算 GAE 優勢函數
        #     4. 執行 PPO 損失計算與反向傳播（inline）
        metrics = train_ppo_epoch(
            model=model,
            runner=runner,
            optimizer=optimizer,
            epochs=ppo_epochs,
        )

        # -------- 7b. 記錄指標 --------
        for key in history:
            history[key].append(metrics[key])

        # -------- 7c. 定期輸出日誌 --------
        if iteration % log_every == 0:
            print(
                f"📊 Iter {iteration:>5d}/{num_iterations} | "
                f"Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Value Loss: {metrics['value_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f} | "
                f"Avg Reward: {metrics['avg_reward']:.4f} | "
                f"Traj Len: {metrics['trajectory_length']:.1f}"
            )

        # -------- 7d. 更新對手池 --------
        if iteration % update_opponent_every == 0:
            print(f"🔄 Iter {iteration}: 更新對手池（當前大小={len(runner.opponent_pool)}）...")
            runner.update_opponent_pool()

        # -------- 7e. 定期保存 Checkpoint --------
        if iteration % save_every == 0:
            ckpt_path = checkpoint_dir / f"ppo_lora_iter_{iteration}.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": {k: v[-1] for k, v in history.items()},
                    "history": history,
                },
                ckpt_path,
            )
            print(f"💾 Checkpoint 已保存: {ckpt_path}")

        # -------- 7f. 追蹤最佳模型（按 avg_reward）--------
        if metrics["avg_reward"] > best_avg_reward:
            best_avg_reward = metrics["avg_reward"]
            best_ckpt_path = checkpoint_dir / "best_ppo_lora_model.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_avg_reward": best_avg_reward,
                    "history": history,
                },
                best_ckpt_path,
            )
            print(f"🏆 新最佳模型！Avg Reward: {best_avg_reward:.4f} → {best_ckpt_path}")

    # ========== 8. 訓練完成，保存最終模型與日誌 ==========
    print(f"\n{'='*70}")
    print(f"✅ PPO + LoRA 訓練完成")
    print(f"   最佳 Avg Reward: {best_avg_reward:.4f}")
    print(f"   結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # 保存最終模型
    final_ckpt_path = checkpoint_dir / "ppo_lora_final.pt"
    torch.save(
        {
            "iteration": num_iterations,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_avg_reward": best_avg_reward,
            "history": history,
        },
        final_ckpt_path,
    )
    print(f"💾 最終模型已保存: {final_ckpt_path}")

    # 保存訓練歷史日誌（numpy 格式方便後續繪圖分析）
    log_path = checkpoint_dir / "ppo_training_log.npz"
    np.savez(log_path, **{k: np.array(v) for k, v in history.items()})
    print(f"📈 訓練日誌已保存: {log_path}\n")

    return model, history


# ==================== 命令列入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="PPO + LoRA 自我博弈線上微調 (Decision Mamba)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  # 使用預設參數訓練
  python train_ppo.py

  # 自訂迭代次數與學習率
  python train_ppo.py --num-iterations 500 --learning-rate 5e-5

  # 從特定 BC checkpoint 開始
  python train_ppo.py --bc-checkpoint ./checkpoints/bc_model/best_bc_model.pt
        """,
    )

    # ---- 路徑參數 ----
    parser.add_argument(
        "--bc-checkpoint", type=str,
        default="/workspace/Mahjong/checkpoints/bc_model/best_bc_model.pt",
        help="BC 預訓練模型權重路徑",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str,
        default="/workspace/Mahjong/checkpoints/ppo_model",
        help="PPO checkpoint 輸出目錄",
    )

    # ---- 訓練超參數 ----
    parser.add_argument(
        "--num-iterations", type=int, default=1000,
        help="PPO 訓練迭代次數（每迭代 = 一局自我博弈 + PPO 更新）",
    )
    parser.add_argument(
        "--ppo-epochs", type=int, default=4,
        help="每條軌跡的 PPO 更新 epoch 數",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="學習率（僅作用於 LoRA + Actor/Critic head）",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="計算設備 (cuda / cpu)",
    )

    # ---- 對手池與保存頻率 ----
    parser.add_argument(
        "--opponent-pool-size", type=int, default=5,
        help="對手池大小",
    )
    parser.add_argument(
        "--update-opponent-every", type=int, default=10,
        help="每 N 次迭代更新一次對手池",
    )
    parser.add_argument(
        "--save-every", type=int, default=50,
        help="每 N 次迭代保存一次 checkpoint",
    )
    parser.add_argument(
        "--log-every", type=int, default=1,
        help="每 N 次迭代輸出一次日誌",
    )

    # ---- 模型架構參數（需與 BC 預訓練一致）----
    parser.add_argument(
        "--d-model", type=int, default=512,
        help="隱藏層維度",
    )
    parser.add_argument(
        "--action-dim", type=int, default=181,
        help="動作空間維度（mjx 使用 181）",
    )
    parser.add_argument(
        "--state-dim", type=int, default=1380,
        help="狀態特徵維度（decision-mamba-v0 = 1380）",
    )
    parser.add_argument(
        "--max-ep-len", type=int, default=2048,
        help="最大軌跡長度（用於 timestep embedding）",
    )

    # ---- 隨機種子 ----
    parser.add_argument(
        "--seed", type=int, default=42,
        help="隨機種子",
    )

    args = parser.parse_args()

    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    # 啟動訓練
    train_ppo(
        bc_checkpoint_path=args.bc_checkpoint,
        num_iterations=args.num_iterations,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        opponent_pool_size=args.opponent_pool_size,
        update_opponent_every=args.update_opponent_every,
        save_every=args.save_every,
        log_every=args.log_every,
        d_model=args.d_model,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        max_ep_len=args.max_ep_len,
    )


if __name__ == "__main__":
    main()