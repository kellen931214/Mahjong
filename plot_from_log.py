"""從 train log 畫 Loss vs Iteration + Reward vs Iteration"""
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def parse_log(path):
    policy_loss, value_loss, avg_reward = [], [], []
    with open(path) as f:
        for line in f:
            m = re.search(r"Iter\s+(\d+)/\d+\s*\|\s*Policy Loss:\s*([\d.]+)\s*\|\s*Value Loss:\s*([\d.]+)\s*\|\s*Entropy:\s*([\d.]+)\s*\|\s*Avg Reward:\s*([-\d.]+)", line)
            if m:
                policy_loss.append(float(m.group(2)))
                value_loss.append(float(m.group(3)))
                avg_reward.append(float(m.group(5)))
    return np.array(policy_loss), np.array(value_loss), np.array(avg_reward)

def smooth(y, w=10):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    k = np.ones(w)/w
    s = np.convolve(y, k, mode="same")
    s[:w//2] = y[:w//2]
    s[-w//2:] = y[-w//2:]
    return s

log_path = sys.argv[1] if len(sys.argv) > 1 else "train_ppo_attack_bc_sparser.log"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "plots"

pl, vl, ar = parse_log(log_path)
x = np.arange(1, len(pl)+1)
print(f"Loaded {len(pl)} iterations from {log_path}")

# ── 圖 1: Loss vs Iteration ──
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(x, pl, color="#2196F3", alpha=0.2, lw=0.6, label="Policy Loss (raw)")
ax.plot(x, smooth(pl), color="#2196F3", lw=2.2, label="Policy Loss (smoothed)")
ax.plot(x, vl, color="#F44336", alpha=0.2, lw=0.6, label="Value Loss (raw)")
ax.plot(x, smooth(vl), color="#F44336", lw=2.2, label="Value Loss (smoothed)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("PPO Training: Policy & Value Loss vs Iteration")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{out_dir}/loss_vs_iteration.png", dpi=150)
plt.close()
print(f"Saved {out_dir}/loss_vs_iteration.png")

# ── 圖 2: Reward vs Iteration ──
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x, ar, color="#4CAF50", alpha=0.35, width=1.0, label="Avg Reward (per iter)")
ax.plot(x, smooth(ar), color="#4CAF50", lw=2.2, label="Avg Reward (smoothed)")
ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Avg Reward")
ax.set_title("PPO Training: Avg Reward vs Iteration")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{out_dir}/reward_vs_iteration.png", dpi=150)
plt.close()
print(f"Saved {out_dir}/reward_vs_iteration.png")
print("Done!")