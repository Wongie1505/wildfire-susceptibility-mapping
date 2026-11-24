"""Script 2: ROC curve of the final model and the summary pie chart.

ROC uses the pooled 2016 to 2024 test rows from test_predictions.csv, so the
AUC should match the 0.6029 reported in Section 4.3.

The pie chart shows how the burned cell-seasons split across the three
susceptibility classes (the 57.4% High share used in Chapter 5). Legend labels
give each class's share of the forest area.

Writes to outputs/:
  roc_curve.png
  summary_pie.png
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

from common import CLASS_COLOURS, DATA, OUT

# ---------- ROC ----------
pred = pd.read_csv(DATA / "test_predictions.csv")
fpr, tpr, _ = roc_curve(pred["burned"], pred["score"])
roc_auc = auc(fpr, tpr)
print(f"Pooled test AUC: {roc_auc:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(fpr, tpr, color="#f03b20", linewidth=2,
        label=f"Random Forest (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance (AUC = 0.5)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve, final model, 2016 to 2024 test rows")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT / "roc_curve.png", dpi=300)
plt.close(fig)

# ---------- Pie: burned cell-seasons by class ----------
burn = pd.read_csv(DATA / "class_validation.csv")
area = pd.read_csv(DATA / "area_table.csv")

labels = [
    f"{row.label} (forest area {area.loc[i, 'share_pct']}%)"
    for i, row in burn.iterrows()
]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
wedges, _, _ = ax.pie(
    burn["burned_cell_seasons"],
    colors=CLASS_COLOURS,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
    wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    pctdistance=0.7,
)
ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
          fontsize=8, title="Class")
ax.set_title("Share of burned cell-seasons by susceptibility class\n(2016 to 2024)")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "summary_pie.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved roc_curve.png and summary_pie.png")
