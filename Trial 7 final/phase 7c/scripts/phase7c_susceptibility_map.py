"""Phase 7c: susceptibility map, class areas and map checks.

The map is the average score of the 6a model over the nine test seasons
(2016 to 2024), taken from the Phase 7a predictions. These seasons were not
used to train the model, so the map is out of sample. No model is run here.

Classes: five classes by natural breaks (Fisher-Jenks) on the mean score.

Writes to outputs/:
  susceptibility_mean_score.tif   mean score per cell
  susceptibility_class.tif        class 1 (very low) to 5 (very high)
  class_breaks.csv                score thresholds between the classes
  area_table.csv                  cells, hectares and km2 per class
  class_validation.csv            how much of each class actually burned
  map_summary.csv                 overall agreement between map and burns
  susceptibility_map.png          quick map figure
  class_validation.png            burned share by class
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

# Paths, relative to the Phase 7c folder
BASE = Path(__file__).resolve().parents[1]
STATIC = BASE / "data" / "static_stack.tif"
PREDICTIONS = BASE / "data" / "test_predictions.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

N_CLASSES = 5
NODATA = -9999
LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
COLOURS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]

# ---------- Grid and forest mask from the Phase 2 stack ----------
with rasterio.open(STATIC) as src:
    transform = src.transform
    crs = src.crs
    height, width = src.height, src.width
    inside = src.read(1) != src.nodata

cell_ha = abs(transform.a * transform.e) / 10000

# ---------- Mean score and burn count for each cell ----------
pred = pd.read_csv(PREDICTIONS)
n_seasons = pred["year"].nunique()

cells = pred.groupby("cell_id").agg(
    mean_score=("score", "mean"),
    burn_count=("burned", "sum"),
).reset_index()

cells["row"] = cells["cell_id"] // width
cells["col"] = cells["cell_id"] % width

# Every scored cell must lie inside the forest mask
if not inside[cells["row"], cells["col"]].all():
    raise ValueError("Some scored cells fall outside the forest mask. Check the inputs.")


def jenks_breaks(values, k):
    """Lower bound of classes 2 to k from Fisher-Jenks natural breaks."""
    x = np.sort(values)
    n = len(x)
    s1 = np.concatenate([[0], np.cumsum(x)])
    s2 = np.concatenate([[0], np.cumsum(x ** 2)])

    def sse(i, j):
        """Squared error of one class made of x[i] to x[j]."""
        return (s2[j + 1] - s2[i]) - (s1[j + 1] - s1[i]) ** 2 / (j - i + 1)

    cost = np.full((k, n), np.inf)
    split = np.zeros((k, n), dtype=int)
    cost[0] = (s2[1:] - s2[0]) - (s1[1:] - s1[0]) ** 2 / np.arange(1, n + 1)

    for m in range(1, k):
        for j in range(m, n):
            i = np.arange(m, j + 1)
            total = cost[m - 1][i - 1] + sse(i, j)
            best = np.argmin(total)
            cost[m][j] = total[best]
            split[m][j] = i[best]

    # Walk back from the end to find where each class starts
    breaks = []
    j = n - 1

    for m in range(k - 1, 0, -1):
        start = split[m][j]
        breaks.append(x[start])
        j = start - 1

    return sorted(breaks)


# ---------- Classes ----------
breaks = jenks_breaks(cells["mean_score"].to_numpy(), N_CLASSES)
cells["class"] = np.digitize(cells["mean_score"], breaks) + 1

pd.DataFrame({
    "from_class": LABELS[:-1],
    "to_class": LABELS[1:],
    "score_threshold": np.round(breaks, 4),
}).to_csv(OUT / "class_breaks.csv", index=False)

# ---------- Rasters ----------
score_grid = np.full((height, width), NODATA, dtype="float32")
class_grid = np.full((height, width), NODATA, dtype="int16")
score_grid[cells["row"], cells["col"]] = cells["mean_score"]
class_grid[cells["row"], cells["col"]] = cells["class"]

profile = dict(driver="GTiff", height=height, width=width, count=1,
               crs=crs, transform=transform, nodata=NODATA)

with rasterio.open(OUT / "susceptibility_mean_score.tif", "w", dtype="float32", **profile) as dst:
    dst.write(score_grid, 1)

with rasterio.open(OUT / "susceptibility_class.tif", "w", dtype="int16", **profile) as dst:
    dst.write(class_grid, 1)

# ---------- Area table ----------
area = cells.groupby("class").size().reindex(range(1, N_CLASSES + 1), fill_value=0)

area_table = pd.DataFrame({
    "class": range(1, N_CLASSES + 1),
    "label": LABELS,
    "cells": area.values,
    "area_ha": (area.values * cell_ha).round(1),
    "area_km2": (area.values * cell_ha / 100).round(2),
    "share_pct": (100 * area.values / area.sum()).round(1),
})
area_table.to_csv(OUT / "area_table.csv", index=False)

# ---------- Check: did each class actually burn? ----------
overall_share = 100 * cells["burn_count"].sum() / (len(cells) * n_seasons)
rows = []

for k in range(1, N_CLASSES + 1):
    part = cells[cells["class"] == k]
    share = 100 * part["burn_count"].sum() / (len(part) * n_seasons)

    rows.append({
        "class": k,
        "label": LABELS[k - 1],
        "cells": len(part),
        "burned_cell_seasons": int(part["burn_count"].sum()),
        "burned_share_pct": round(share, 1),
        "ratio_to_overall": round(share / overall_share, 2),
        "cells_burned_at_least_once_pct": round(100 * (part["burn_count"] > 0).mean(), 1),
    })

validation = pd.DataFrame(rows)
validation.to_csv(OUT / "class_validation.csv", index=False)

# ---------- Overall agreement between map and burns ----------
rho = cells["mean_score"].corr(cells["burn_count"], method="spearman")
auc = roc_auc_score(cells["burn_count"] > 0, cells["mean_score"])

pd.DataFrame([
    {"measure": "seasons averaged", "value": n_seasons},
    {"measure": "cells", "value": len(cells)},
    {"measure": "overall burned share of cell-seasons (%)", "value": round(overall_share, 1)},
    {"measure": "Spearman: mean score against number of seasons burned", "value": round(rho, 3)},
    {"measure": "AUC: mean score against burned at least once", "value": round(auc, 3)},
]).to_csv(OUT / "map_summary.csv", index=False)

# ---------- Figure 1: the map ----------
extent = (
    transform.c, transform.c + width * transform.a,
    transform.f + height * transform.e, transform.f,
)
shown = np.ma.masked_equal(class_grid, NODATA)

fig, ax = plt.subplots(figsize=(5.5, 9))
ax.imshow(shown, cmap=ListedColormap(COLOURS), vmin=1, vmax=N_CLASSES, extent=extent)
ax.set_xlabel("Easting (m), UTM 36S")
ax.set_ylabel("Northing (m)")
ax.tick_params(labelsize=7)
ax.ticklabel_format(style="plain", useOffset=False)
ax.set_title("Wildfire susceptibility, Chikangawa\nMean of 2016 to 2024 (relative scores)", fontsize=10)

handles = [
    Patch(color=COLOURS[k], label=f"{LABELS[k]} ({area_table.loc[k, 'share_pct']}%)")
    for k in range(N_CLASSES)
]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, title="Class (share of area)")
fig.tight_layout()
fig.savefig(OUT / "susceptibility_map.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 2: burned share by class ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(validation["label"], validation["burned_share_pct"], color=COLOURS, edgecolor="black")
ax.axhline(overall_share, color="black", linestyle="--", label=f"All cells {overall_share:.1f}%")

for i, value in enumerate(validation["burned_share_pct"]):
    ax.text(i, value + 0.4, f"{value:.1f}%", ha="center", fontsize=9)

ax.set_ylabel("Share of cell-seasons that burned (%)")
ax.set_title("Did the classes burn as ranked? (2016 to 2024)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "class_validation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Print ----------
print(area_table.to_string(index=False))
print()
print(validation.to_string(index=False))
print()
print(pd.read_csv(OUT / "map_summary.csv").to_string(index=False))