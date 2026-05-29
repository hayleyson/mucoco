import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Contradiction avoidance data
df = pd.DataFrame({
    "thresholds": [0.997154, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.48, 0.4],
    "original_contradiction_proba": [0.2502415459] * 10,
    "contradiction_proba": [0.049, 0.052, 0.047, 0.046, 0.048, 0.054, 0.066, 0.071, 0.072, 0.073],
    "original_total_ppl_qwen": [12.93199294] * 10,
    "total_ppl_qwen": [17.214, 17.175, 16.568, 16.179, 15.53, 14.492, 13.999, 13.897, 13.858, 13.713],
})

df = df.sort_values("thresholds").reset_index(drop=True)
df = df.iloc[1:-1,:].copy()

# Invert contradictory response rate to control accuracy and keep probability scale
df["control_accuracy"] = 1 - df["contradiction_proba"]
baseline_x = 1 - df["original_contradiction_proba"].iloc[0]
baseline_y = df["original_total_ppl_qwen"].iloc[0]

# Tighter display transform:
# Place the original data tick closer to 0.95, roughly one tick interval away.
baseline_display_x = 0.92

def transform_x(x):
    x = np.asarray(x, dtype=float)
    return np.where(x < 0.90, baseline_display_x + (x - baseline_x), x)

x_tradeoff = transform_x(df["control_accuracy"].values)
x_baseline = transform_x(baseline_x)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 17,
})

fig, ax = plt.subplots(figsize=(11, 7.5))

# Edited results curve
ax.plot(
    x_tradeoff,
    df["total_ppl_qwen"],
    color="orange",
    marker="o",
    linewidth=2.5,
    markersize=7,
    label="Edited Results for Each Threshold",
)

# Original data point with outlined marker
ax.scatter(
    [x_baseline],
    [baseline_y],
    marker="D",
    s=95,
    facecolor="orange",
    edgecolor="black",
    linewidth=1.0,
    label="Original Data",
    zorder=5,
)

# Threshold annotations
for _, row in df.iterrows():
    t = float(row["thresholds"])
    x = transform_x(row["control_accuracy"])
    y = row["total_ppl_qwen"]

    offsets = {
        # 0.4: (0, -20),
        0.48: (0, 7),
        0.5: (6, -20),
        0.6: (-8, 7),
        0.7: (-6, 8),
        0.8: (-20, -6),
        0.9: (-20, -6),
        0.95: (-26, -6),
        0.99: (-26, -10),
        # 0.997154: (50, -6),
    }.get(t, (7, 7))

    ax.annotate(
        f"{t:g}",
        (x, y),
        textcoords="offset points",
        xytext=offsets,
        ha="center",
        fontsize=17,
    )

# Annotate original point
ax.annotate(
    "Original Data",
    (x_baseline, baseline_y),
    textcoords="offset points",
    xytext=(0, 8),
    ha="left",
    fontsize=16,
)

# Title and labels
# ax.set_title("Threshold Ablation Results for Toxicity Avoidance", pad=18, fontsize=20)
ax.set_xlabel("Control Accuracy ↑", labelpad=10, fontsize=22)
ax.set_ylabel("Perplexity ↓", labelpad=10, fontsize=22)

# Ticks: original tick plus edited-result tick range on probability scale
xtick_values = [baseline_display_x, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955]
xticks = [transform_x(v) for v in xtick_values]
xticklabels = [f"{baseline_x:.3f}"] + [f"{v:.3f}" for v in xtick_values[1:]]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

# Limits
ax.set_xlim(0.918, 0.958)
ax.set_ylim(12.5, 17.5)

# Broken-axis markers placed in the tighter gap between original tick and 0.95
break_center = 0.925
for y0, y1 in [(-0.018, 0.018), (0.982, 1.018)]:
    ax.plot(
        [break_center - 0.0017, break_center - 0.00045],
        [y0, y1],
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        linewidth=1.5,
        color="black",
    )
    ax.plot(
        [break_center - 0.0008 + 0.00045, break_center - 0.0008 + 0.0017],
        [y0, y1],
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        linewidth=1.5,
        color="black",
    )

ax.grid(True, which="major", linewidth=1, alpha=0.6)
ax.legend(loc="upper left", frameon=False)

plt.tight_layout()

output_path = "/home/hyeryung/data/mucoco/laser_edit/analyses/charts/outputs/contradiction_avoidance_threshold_ablation.pdf"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(output_path)
