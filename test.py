import matplotlib.pyplot as plt

# Updated data
thresholds = [
    "0-255",
    "15-255",
    "35-255",
    "55-255",
    "75-255",
    "95-255",
    "115-255",
    "135-255",
    "155-255",
    "175-255",
    "195-255",
    "215-255",
    "235-255",
]
accuracy_train = [
    0.9877,
    0.9910,
    0.9943,
    0.9910,
    0.9862,
    0.9943,
    0.9848,
    0.9858,
    0.9862,
    0.9905,
    0.9825,
    0.9654,
    0.9355,
]
accuracy_validation = [
    0.9849,
    0.9754,
    0.9735,
    0.9773,
    0.9848,
    0.9792,
    0.9734,
    0.9735,
    0.9772,
    0.9620,
    0.9545,
    0.9450,
    0.8898,
]
f1_score = [
    0.9847,
    0.9759,
    0.9734,
    0.9771,
    0.9848,
    0.9792,
    0.9738,
    0.9741,
    0.9773,
    0.9621,
    0.9548,
    0.9449,
    0.8900,
]

x = range(len(thresholds))

fig, ax = plt.subplots(figsize=(10, 8))

bar_width = 0.2
opacity = 0.8

# Bar for Accuracy Train
ax.bar(
    x,
    accuracy_train,
    width=bar_width,
    label="Accuracy Train",
    color="steelblue",
    alpha=opacity,
)
for i, v in enumerate(accuracy_train):
    ax.text(
        i - 0.06, v + 0.002, f"{v:.4f}", color="black", fontweight="bold", rotation=45
    )

# Bar for Accuracy Validation
ax.bar(
    [i + bar_width for i in x],
    accuracy_validation,
    width=bar_width,
    label="Accuracy Validation",
    color="darkorange",
    alpha=opacity,
)
for i, v in enumerate(accuracy_validation):
    ax.text(
        i + bar_width - 0.06,
        v + 0.002,
        f"{v:.4f}",
        color="black",
        fontweight="bold",
        rotation=45,
    )

# Bar for F1-score
ax.bar(
    [i + 2 * bar_width for i in x],
    f1_score,
    width=bar_width,
    label="F1-score",
    color="darkgreen",
    alpha=opacity,
)
for i, v in enumerate(f1_score):
    ax.text(
        i + 2 * bar_width - 0.06,
        v + 0.002,
        f"{v:.4f}",
        color="black",
        fontweight="bold",
        rotation=45,
    )

ax.set_xlabel("Thresholds")
ax.set_xticks([i + bar_width for i in x])
ax.set_xticklabels(thresholds)
ax.legend()
ax.set_ylim(0.87, 1.0)

# Lưu biểu đồ dưới dạng file PNG
plt.savefig("model_performance.png", dpi=300)

plt.tight_layout()
plt.show()
