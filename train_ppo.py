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
from collections import deque
import argparse
import sys

# 確保本地模組可導入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import DecisionMamba
from utli.runner import SelfPlayRunner
from utli.train_ppo_step import train_ppo_epoch


# ==================== PPO 主訓練函數 ====================

def train_ppo(
    bc_checkpoint_path: str = "/workspace/Mahjong/checkpoints/bc_model/best_bc_model.pt",
    num_iterations: int = 1000,
    ppo_epochs: int = 1,
    learning_rate: float = 5e-5,
    device: str = "cuda",
    checkpoint_dir: str = "/workspace/Mahjong/checkpoints/ppo_lora_v3",
    opponent_pool_size: int = 5,
    update_opponent_every: int = 10,
    save_every: int = 50,
    log_every: int = 1,
    game_stats_window: int = 50,  # 🆕 滑動窗口大小（計算贏牌率等）
    d_model: int = 512,
    action_dim: int = 181,
    train_mode: str = "attack",
    state_dim: int = 1380,
    max_ep_len: int = 2048,
    # 🆕 LoRA PPO 超參數
    temperature: float = 2.0,
    entropy_coef: float = 0.05,
    value_coef: float = 0.5,
    clip_epsilon: float = 0.2,
    max_grad_norm: float = 0.5,
):
    """
    LoRA + PPO 自我博弈線上微調主訓練迴圈。

    Args:
        bc_checkpoint_path: BC 預訓練模型權重路徑
        num_iterations: PPO 訓練迭代次數（每迭代 = 一局自我博弈 + PPO 更新）
        ppo_epochs: 每條軌跡的 PPO 更新 epoch 數（LoRA 建議 1）
        learning_rate: 學習率（5e-5，LoRA 建議小步伐）
        device: 計算設備
        checkpoint_dir: PPO 模型保存路徑
        opponent_pool_size: 對手池大小
        update_opponent_every: 每 N 次迭代更新一次對手池
        save_every: 每 N 次迭代保存一次 checkpoint
        log_every: 每 N 次迭代輸出一次日誌
        game_stats_window: 遊戲指標滑動窗口大小
        d_model: 隱藏層維度
        action_dim: 動作空間維度（mjx 使用 181 種動作）
        state_dim: 狀態特徵維度（decision-mamba-v0 = 1380 維）
        max_ep_len: 最大軌跡長度
        temperature: Logit 採樣溫度（2.0~2.5，拉平極端分佈重新打開梯度通道）
        entropy_coef: 策略熵係數（0.03~0.05，控制探索強度）
        value_coef: Value Loss 權重（0.5，平衡 Critic 與 Actor）
        clip_epsilon: PPO 裁剪閾值（0.2，安全閥）
        max_grad_norm: 梯度裁剪閾值（0.5，防止梯度爆炸）
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

    checkpoint = torch.load(bc_checkpoint_path, map_location=device, weights_only=False)
    # 支援直接載入 state_dict 或包裝後的 checkpoint dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"   ✅ 載入 BC checkpoint (epoch {checkpoint.get('epoch', '?')})")
        print(f"   BC val_loss: {checkpoint.get('val_loss', '?'):.4f}, val_acc: {checkpoint.get('val_acc', '?'):.4f}")
    else:
        state_dict = checkpoint
        print("   ✅ 載入原始 state_dict")

    model.load_state_dict(state_dict, strict=False)

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
        train_mode=train_mode,
    )
    print(f"   ✅ Runner 初始化完成（train_mode={train_mode}）")

    # ========== 6. 準備 Checkpoint 目錄與日誌 ==========
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 訓練指標歷史記錄（原有欄位 + 🆕 遊戲結果）
    history = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "avg_reward": [],
        "trajectory_length": [],
        # 🆕 遊戲結果指標
        "agent_rank": [],
        "agent_score": [],
        "is_win": [],
        "is_agari": [],
    }

    # 🆕 滑動窗口緩衝區（用於計算近 N 局的贏牌率等統計）
    game_window = deque(maxlen=game_stats_window)

    # ========== 7. 主訓練迴圈 ==========
    print(f"\n{'='*70}")
    print(f"🚀 開始 PPO + LoRA 自我博弈訓練")
    print(f"   迭代次數: {num_iterations}")
    print(f"   PPO epochs/局: {ppo_epochs}")
    print(f"   Temperature: {temperature}")
    print(f"   Entropy Coef: {entropy_coef}")
    print(f"   Value Coef: {value_coef}")
    print(f"   Clip Epsilon: {clip_epsilon}")
    print(f"   Max Grad Norm: {max_grad_norm}")
    print(f"   對手池更新頻率: 每 {update_opponent_every} 局")
    print(f"   遊戲指標窗口: 最近 {game_stats_window} 局")
    print(f"   開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    best_avg_rank = float("inf")  # 🆕 用窗口均排名選最佳模型（越低越好）

    for iteration in range(1, num_iterations + 1):
        # -------- 7a. 執行一局自我博弈 + PPO 更新 --------
        metrics = train_ppo_epoch(
            model=model,
            runner=runner,
            optimizer=optimizer,
            epochs=ppo_epochs,
            temperature=temperature,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            clip_epsilon=clip_epsilon,
            max_grad_norm=max_grad_norm,
            num_trajectories=8,
        )

        # -------- 7b. 記錄指標 --------
        for key in history:
            if key in metrics:
                history[key].append(metrics[key])
        # 🆕 遊戲結果欄位可能不在 old history keys 中，但已加入上方 dict

        # -------- 7c. 🆕 累積遊戲結果到滑動窗口 --------
        game_result = metrics.get("game_result", {})
        if game_result:
            history["agent_rank"].append(game_result.get("agent_rank", 3))
            history["agent_score"].append(game_result.get("agent_score", 0))
            history["is_win"].append(1 if game_result.get("is_win", False) else 0)
            history["is_agari"].append(1 if game_result.get("is_agari", False) else 0)
            game_window.append(game_result)

        # -------- 7d. 定期輸出日誌 --------
        if iteration % log_every == 0:
            # 🆕 計算滑動窗口統計
            if len(game_window) > 0:
                window_win_rate = sum(1 for g in game_window if g.get("is_win", False)) / len(game_window) * 100
                window_agari_rate = sum(1 for g in game_window if g.get("is_agari", False)) / len(game_window) * 100
                window_avg_rank = sum(g.get("agent_rank", 3) for g in game_window) / len(game_window)
                window_avg_score = sum(g.get("agent_score", 0) for g in game_window) / len(game_window)
            else:
                window_win_rate = 0.0
                window_agari_rate = 0.0
                window_avg_rank = 3.0
                window_avg_score = 0.0

            print(
                f"📊 Iter {iteration:>5d}/{num_iterations} | "
                f"Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Value Loss: {metrics['value_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f} | "
                f"Avg Reward: {metrics['avg_reward']:.4f} | "
                f"Traj Len: {metrics['trajectory_length']:.1f}"
            )
            # 🆕 遊戲結果摘要（滑動窗口）
            print(
                f"🎮 近{min(len(game_window), game_stats_window)}局 | "
                f"贏牌率: {window_win_rate:5.1f}% | "
                f"和了率: {window_agari_rate:5.1f}% | "
                f"均排名: {window_avg_rank:.2f} | "
                f"均分數: {window_avg_score:7.0f} | "
                f"上一局: 排名={game_result.get('agent_rank','?')} "
                f"分數={game_result.get('agent_score','?'):.0f}pt "
                f"{'🏆' if game_result.get('is_win') else ''}"
            )

        # -------- 7e. 更新對手池 --------
        if iteration % update_opponent_every == 0:
            print(f"🔄 Iter {iteration}: 更新對手池（當前大小={len(runner.opponent_pool)}）...")
            runner.update_opponent_pool()

        # -------- 7f. 定期保存 Checkpoint --------
        if iteration % save_every == 0:
            ckpt_path = checkpoint_dir / f"ppo_lora_iter_{iteration}.pt"
            # 🆕 計算保存時的窗口統計
            if len(game_window) > 0:
                save_win_rate = sum(1 for g in game_window if g.get("is_win", False)) / len(game_window) * 100
                save_agari_rate = sum(1 for g in game_window if g.get("is_agari", False)) / len(game_window) * 100
            else:
                save_win_rate = 0.0
                save_agari_rate = 0.0
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": {k: v[-1] for k, v in history.items()},
                    "history": history,
                    "window_win_rate": save_win_rate,
                    "window_agari_rate": save_agari_rate,
                    "window_size": len(game_window),
                },
                ckpt_path,
            )
            print(f"💾 Checkpoint 已保存: {ckpt_path}")

        # -------- 7g. 🆕 追蹤最佳模型（按窗口均排名）--------
        # 用均排名取代 avg_reward：排名直接反映牌力，不受 reward hacking 影響
        if len(game_window) >= 50:
            current_avg_rank = sum(g.get("agent_rank", 3) for g in game_window) / len(game_window)
            if current_avg_rank < best_avg_rank:
                best_avg_rank = current_avg_rank
                current_win_rate = sum(1 for g in game_window if g.get("is_win", False)) / len(game_window) * 100
                best_ckpt_path = checkpoint_dir / "best_ppo_lora_model.pt"
                torch.save(
                    {
                        "iteration": iteration,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_avg_rank": best_avg_rank,
                        "current_avg_rank": current_avg_rank,
                        "window_win_rate": current_win_rate,
                        "history": history,
                    },
                    best_ckpt_path,
                )
                print(f"🏆 新最佳模型！均排名: {best_avg_rank:.2f} (近{len(game_window)}局贏牌率: {current_win_rate:.1f}%) → {best_ckpt_path}")

    # ========== 8. 訓練完成，保存最終模型與日誌 ==========
    print(f"\n{'='*70}")
    print(f"✅ PPO + LoRA 訓練完成")
    print(f"   最佳均排名: {best_avg_rank:.2f}")
    if len(game_window) > 0:
        final_win_rate = sum(1 for g in game_window if g.get("is_win", False)) / len(game_window) * 100
        print(f"   最終窗口贏牌率 ({len(game_window)}局): {final_win_rate:.1f}%")
    print(f"   結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # 保存最終模型
    final_ckpt_path = checkpoint_dir / "ppo_lora_final.pt"
    torch.save(
        {
            "iteration": num_iterations,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_avg_rank": best_avg_rank,
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
        default="/workspace/Mahjong/checkpoints/ppo_lora_v4",
        help="PPO checkpoint 輸出目錄",
    )

    # ---- 訓練超參數 ----
    parser.add_argument(
        "--num-iterations", type=int, default=1000,
        help="PPO 訓練迭代次數（每迭代 = 一局自我博弈 + PPO 更新）",
    )
    parser.add_argument(
        "--ppo-epochs", type=int, default=1,
        help="每條軌跡的 PPO 更新 epoch 數（LoRA 建議 1）",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5,
        help="學習率（僅作用於 LoRA + Actor/Critic head）",
    )
    # 🆕 LoRA PPO 超參數
    parser.add_argument(
        "--temperature", type=float, default=2.0,
        help="Logit 採樣溫度（2.0~2.5，拉平極端分佈重新打開梯度通道）",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.05,
        help="策略熵係數（0.03~0.05，控制探索強度）",
    )
    parser.add_argument(
        "--value-coef", type=float, default=0.5,
        help="Value Loss 權重（平衡 Critic 與 Actor 學習速度）",
    )
    parser.add_argument(
        "--clip-epsilon", type=float, default=0.2,
        help="PPO 裁剪閾值（安全閥，防止極端獎勵拉偏 Policy）",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5,
        help="梯度裁剪閾值（防止 GAE 劇烈變動導致梯度爆炸）",
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
    parser.add_argument(
        "--game-stats-window", type=int, default=50,
        help="遊戲指標滑動窗口大小（計算贏牌率等）",
    )

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
        "--mode", type=str, default="attack", choices=["attack", "defense"],
        help="訓練模式: attack=進攻 reward（預設）, defense=防守 reward",
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
        game_stats_window=args.game_stats_window,
        d_model=args.d_model,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        train_mode=args.mode,
        max_ep_len=args.max_ep_len,
        temperature=args.temperature,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        clip_epsilon=args.clip_epsilon,
        max_grad_norm=args.max_grad_norm,
    )


if __name__ == "__main__":
    main()