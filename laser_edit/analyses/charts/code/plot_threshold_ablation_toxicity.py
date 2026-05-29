import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Toxicity avoidance data
df = pd.DataFrame({
    "thresholds": [0.39, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    "original_total_ppl_qwen": [5.242829854] * 9,
    "total_ppl_qwen": [5.351, 5.35, 5.362, 5.423, 5.449, 5.541, 5.905, 6.468, 6.551],
    "original_toxic_probability_s": [0.1247826087] * 9,
    "toxic_probability_s": [0.049, 0.049, 0.04, 0.031, 0.029, 0.024, 0.017, 0.0, 0.0],
})

df = df.sort_values("thresholds").reset_index(drop=True)

# Invert toxic response rate to control accuracy and keep probability scale
df["control_accuracy"] = 1 - df["toxic_probability_s"]
baseline_x = 1 - df["original_toxic_probability_s"].iloc[0]
baseline_y = df["original_total_ppl_qwen"].iloc[0]

# Tighter display transform:
# Place the original data tick closer to 0.95, roughly one tick interval away.
baseline_display_x = 0.94

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

    if t == 0.39:
        ax.annotate(
            "0.39 / 0.4",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=17,
        )
    elif t == 0.4:
        continue
    else:
        offsets = {
            0.5: (0, 7),
            0.6: (-8, 7),
            0.7: (0, 8),
            0.8: (-8, 8),
            0.9: (-6, 8),
            0.95: (-22, 3),
            0.99: (-22, 3),
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
xtick_values = [baseline_x, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
xticks = [transform_x(v) for v in xtick_values]
xticklabels = [f"{baseline_x:.3f}"] + [f"{v:.2f}" for v in xtick_values[1:]]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

# Limits
ax.set_xlim(0.936, 1.003)
ax.set_ylim(5.1, 6.75)

# Broken-axis markers placed in the tighter gap between original tick and 0.95
break_center = 0.945
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
        [break_center + 0.00045, break_center + 0.0017],
        [y0, y1],
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        linewidth=1.5,
        color="black",
    )

ax.grid(True, which="major", linewidth=1, alpha=0.6)
ax.legend(loc="upper left", frameon=False)

plt.tight_layout()

output_path = "/home/hyeryung/data/mucoco/laser_edit/analyses/charts/outputs/toxicity_avoidance_threshold_ablation.pdf"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(output_path)
