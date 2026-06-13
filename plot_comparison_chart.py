"""比較不同 checkpoint 的 placement distribution（排名數）"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def extract_counts(log_path):
    """從 eval log 中擷取最終的排名次數"""
    with open(log_path) as f:
        content = f.read()
    parts = content.split("自我對弈完成")
    last = parts[-1] if len(parts) > 1 else content
    counts = {}
    for m in re.findall(r"(\d) 位.*?(\d+\.\d+)%\s*\((\d+)\)", last):
        counts[int(m[0])] = int(m[2])
    if len(counts) < 4:
        all_m = re.findall(r"(\d) 位.*?: (\d+\.\d+)%\s*\((\d+)\)", content)
        for rank_s, _, cnt_s in all_m[-4:]:
            counts[int(rank_s)] = int(cnt_s)
    return counts

logs = [
    "eval_vs_bc_defense_iter550.log",
    "eval_vs_bc_dense_8b.log",
]
labels = ["iter 550", "iter 800"]
colors = ["#FF9800", "#2196F3"]

fig, ax = plt.subplots(figsize=(10, 6))
total_games = 500
x = np.arange(4)
width = 0.22

for i, log in enumerate(logs):
    c = extract_counts(log)
    cnts = [c.get(r, 0) for r in [1, 2, 3, 4]]
    bars = ax.bar(x + i * width, cnts, width, color=colors[i], alpha=0.85, label=labels[i])
    # Print for verification
    print(f"{labels[i]}: {cnts}")
    # Add count labels on top of each bar
    for bar, cnt in zip(bars, cnts):
        pct = cnt / total_games * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{cnt}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlabel("Placement (Rank)")
ax.set_ylabel("Game Count (out of 500)")
ax.set_title("PPO vs BC: Placement Distribution at Different Training Stages")
ax.set_xticks(x + width)
ax.set_xticklabels(["1st", "2nd", "3rd", "4th"])
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(ax.get_ylim()[1] * 1.08, 160))
fig.tight_layout()
fig.savefig("plots/placement_comparison.png", dpi=150)
plt.close()
print("Saved plots/placement_comparison.png")