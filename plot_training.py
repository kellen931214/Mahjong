"""
訓練曲線繪圖腳本
=================================
讀取 train_ppo.py 產生的 ppo_training_log.npz，
畫出兩張 Demo 用圖：
  1. Loss v.s. Iteration 曲線（Policy Loss / Value Loss / Entropy）
  2. Reward & 遊戲表現 v.s. Iteration 曲線（Avg Reward / 贏牌率 / 均排名）

用法：
  python plot_training.py <ppo_training_log.npz> [--output <output_dir>]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 風格設定 ──
plt.rcParams.update(
    {
        "figure.figsize": (14, 5),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 1.6,
        "grid.alpha": 0.3,
    }
)


def smooth(y, window=5):
    """簡單移動平均平滑（保留陣列長度不變，邊界不做平滑）"""
    y = np.asarray(y, dtype=np.float64)
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="same")
    # 頭尾保持原值
    smoothed[: window // 2] = y[: window // 2]
    smoothed[-(window // 2) :] = y[-(window // 2) :]
    return smoothed


def plot_training(npz_path: str, output_dir: str = "."):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        print(f"❌ 找不到檔案: {npz_path}")
        sys.exit(1)

    data = np.load(npz_path)

    # ── 檢查必要欄位 ──
    required = ["policy_loss", "value_loss", "avg_reward"]
    for k in required:
        if k not in data.files:
            print(f"❌ npz 缺少必要欄位: {k}，可用欄位: {list(data.files)}")
            sys.exit(1)

    iterations = np.arange(1, len(data["policy_loss"]) + 1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 圖 1：Loss v.s. Iteration
    # ==================================================================
    fig1, ax1a = plt.subplots(figsize=(14, 5))

    # Policy Loss（藍色）
    ax1a.plot(iterations, data["policy_loss"], color="#2196F3", alpha=0.25, linewidth=0.8, label="Policy Loss (raw)")
    ax1a.plot(iterations, smooth(data["policy_loss"], 10), color="#2196F3", linewidth=2.2, label="Policy Loss (平滑)")
    ax1a.set_ylabel("Policy Loss", color="#2196F3")
    ax1a.tick_params(axis="y", labelcolor="#2196F3")

    # Value Loss（紅色，右軸）
    ax1b = ax1a.twinx()
    ax1b.plot(iterations, data["value_loss"], color="#F44336", alpha=0.25, linewidth=0.8, label="Value Loss (raw)")
    ax1b.plot(iterations, smooth(data["value_loss"], 10), color="#F44336", linewidth=2.2, label="Value Loss (平滑)")
    ax1b.set_ylabel("Value Loss", color="#F44336")
    ax1b.tick_params(axis="y", labelcolor="#F44336")

    # Entropy（綠色虛線，若有）
    if "entropy" in data.files:
        ax1c = ax1a.twinx()
        ax1c.spines["right"].set_position(("axes", 1.12))
        ax1c.plot(iterations, data["entropy"], color="#4CAF50", alpha=0.25, linewidth=0.8, label="Entropy (raw)")
        ax1c.plot(iterations, smooth(data["entropy"], 10), color="#4CAF50", linestyle=":", linewidth=2.0, label="Entropy (平滑)")
        ax1c.set_ylabel("Entropy", color="#4CAF50")
        ax1c.tick_params(axis="y", labelcolor="#4CAF50")

    ax1a.set_xlabel("Iteration")
    ax1a.set_title("PPO Training: Loss vs Iteration")
    ax1a.grid(True)

    # 合併圖例
    lines1, labels1 = ax1a.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    if "entropy" in data.files:
        lines3, labels3 = ax1c.get_legend_handles_labels()
        ax1a.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="upper right")
    else:
        ax1a.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig1.tight_layout()
    loss_path = output_dir / "training_loss.png"
    fig1.savefig(loss_path)
    plt.close(fig1)
    print(f"📈 Loss 曲線已保存: {loss_path}")

    # ==================================================================
    # 圖 2：Reward & 遊戲表現 v.s. Iteration
    # ==================================================================
    fig2, ax2a = plt.subplots(figsize=(14, 5))

    # Avg Reward（藍色條）
    ax2a.bar(iterations, data["avg_reward"], color="#2196F3", alpha=0.5, width=1.0, label="Avg Reward (per iter)")
    ax2a.plot(iterations, smooth(data["avg_reward"], 10), color="#2196F3", linewidth=2.2, label="Avg Reward (平滑)")
    ax2a.set_ylabel("Avg Reward", color="#2196F3")
    ax2a.tick_params(axis="y", labelcolor="#2196F3")

    # 贏牌率 & 均排名（右軸）
    ax2b = ax2a.twinx()

    # 優先使用 window 平滑數據（新版 train_ppo.py 會記錄），否則 fallback 到 per-game
    if "window_win_rate" in data.files and len(data["window_win_rate"]) > 0:
        win_rate = data["window_win_rate"]
        avg_rank = data["window_avg_rank"]
    elif "is_win" in data.files and len(data["is_win"]) > 0:
        # Fallback: 用 per-game 數據手動計算 50-game window
        is_win = data["is_win"]
        agent_rank = data["agent_rank"]
        win = 50
        win_rate = np.array(
            [np.mean(is_win[max(0, i - win + 1): i + 1]) * 100 for i in range(len(is_win))]
        )
        avg_rank = np.array(
            [np.mean(agent_rank[max(0, i - win + 1): i + 1]) for i in range(len(agent_rank))]
        )
        # 對齊 iterations 長度
        if len(win_rate) > len(iterations):
            win_rate = win_rate[-len(iterations):]
            avg_rank = avg_rank[-len(iterations):]
    else:
        win_rate = None
        avg_rank = None

    if win_rate is not None:
        ax2b.plot(iterations, win_rate, color="#F44336", linewidth=2.2, label="贏牌率 (Win Rate %)")
        ax2b.plot(iterations, avg_rank, color="#FF9800", linewidth=2.2, label="均排名 (Avg Rank)")
        ax2b.set_ylabel("Win Rate % / Avg Rank", color="#333333")
        ax2b.axhline(y=25.0, color="#F44336", linestyle="--", alpha=0.5, label="Random=25%")
        ax2b.axhline(y=2.5, color="#FF9800", linestyle="--", alpha=0.5, label="Random=2.5")

    ax2a.set_xlabel("Iteration")
    ax2a.set_title("PPO Training: Reward & Game Performance vs Iteration")
    ax2a.grid(True)

    lines1, labels1 = ax2a.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig2.tight_layout()
    reward_path = output_dir / "training_reward.png"
    fig2.savefig(reward_path)
    plt.close(fig2)
    print(f"📊 Reward 曲線已保存: {reward_path}")

    print("\n✅ 繪圖完成！")


# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser(description="PPO 訓練曲線繪圖工具")
    parser.add_argument("npz_path", help="ppo_training_log.npz 路徑")
    parser.add_argument("--output", "-o", default=".", help="圖片輸出目錄（預設: 當前目錄）")
    args = parser.parse_args()
    plot_training(args.npz_path, args.output)


if __name__ == "__main__":
    main()