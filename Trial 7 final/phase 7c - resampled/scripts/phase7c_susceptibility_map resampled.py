"""Phase 7c: susceptibility map, class areas, map checks and the final figure.

The map is the average score of the 6a model over the nine test seasons
(2016 to 2024), taken from the Phase 7a predictions. These seasons were not
used to train the model, so the map is out of sample. No model is run here.

Two versions of the map are made:
  500 m   the real model grid. All areas and checks use this version.
  30.6 m  the same map smoothed onto the DEM grid (bilinear) and clipped to the
          forest boundary. It is for display only and adds no information.

Classes: Low, Medium, High, by natural breaks (Fisher-Jenks) on the 500 m score.

Writes to outputs/:
  susceptibility_500m_score.tif / susceptibility_500m_class.tif
  susceptibility_30m_score.tif  / susceptibility_30m_class.tif
  class_breaks.csv, area_table.csv, class_validation.csv, map_summary.csv
  susceptibility_map.png        the two-panel map (Trial 5 design)
  class_validation.png          burned share by class
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from rasterio import features
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt, gaussian_filter
from sklearn.metrics import roc_auc_score

# ---------- Paths, relative to the Phase 7c folder ----------
BASE = Path(__file__).resolve().parents[1]
STATIC = BASE / "data" / "static_stack.tif"
PREDICTIONS = BASE / "data" / "test_predictions.csv"
DEM = BASE / "data" / "Viphya_DEM.tif"
BOUNDARY = BASE / "data" / "Chikangawa_forest" / "Chikangawa_forest.shp"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------- Settings ----------
N_CLASSES = 3
NODATA = -9999
LABELS = ["Low", "Medium", "High"]
CLASS_COLOURS = ["#00b050", "#e2e24a", "#f5333a"]
RAMP = ["#239b3f", "#7ad02b", "#e8f72a", "#fdae1a", "#f0451e", "#8c1010"]

# The model scores are relative susceptibility, not calibrated probabilities.
# Change this label only if the scores have been calibrated.
SCORE_LABEL = "Susceptibility score"

# Display only: light smoothing of the 30 m surface, in pixels (8 px is about 245 m).
# It removes single-cell speckle. Set to 0 to switch it off.
DISPLAY_SMOOTHING = 8

# ---------- 500 m grid and forest mask from the Phase 2 stack ----------
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


# ---------- Classes on the 500 m score ----------
breaks = jenks_breaks(cells["mean_score"].to_numpy(), N_CLASSES)
cells["class"] = np.digitize(cells["mean_score"], breaks) + 1

pd.DataFrame({
    "from_class": LABELS[:-1],
    "to_class": LABELS[1:],
    "score_threshold": np.round(breaks, 4),
}).to_csv(OUT / "class_breaks.csv", index=False)

# ---------- 500 m rasters ----------
score_grid = np.full((height, width), NODATA, dtype="float32")
class_grid = np.full((height, width), NODATA, dtype="int16")
score_grid[cells["row"], cells["col"]] = cells["mean_score"]
class_grid[cells["row"], cells["col"]] = cells["class"]


def write_raster(path, array, dtype, grid_transform, grid_crs):
    """Write one single-band GeoTIFF."""
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype=dtype, crs=grid_crs, transform=grid_transform, nodata=NODATA,
    ) as dst:
        dst.write(array.astype(dtype), 1)


write_raster(OUT / "susceptibility_500m_score.tif", score_grid, "float32", transform, crs)
write_raster(OUT / "susceptibility_500m_class.tif", class_grid, "int16", transform, crs)

# ---------- Area table (500 m, the analysis grid) ----------
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
    {"measure": "lowest and highest mean score", "value": f"{cells['mean_score'].min():.3f} to {cells['mean_score'].max():.3f}"},
]).to_csv(OUT / "map_summary.csv", index=False)

# ---------- Display surface: 500 m map smoothed onto the 30.6 m DEM grid ----------
with rasterio.open(DEM) as src:
    fine_transform, fine_crs = src.transform, src.crs
    fine_h, fine_w = src.height, src.width

boundary = gpd.read_file(BOUNDARY).to_crs(fine_crs)
forest = features.rasterize(
    [(geom, 1) for geom in boundary.geometry],
    out_shape=(fine_h, fine_w), transform=fine_transform, fill=0, dtype="uint8",
).astype(bool)

# Bilinear: each fine pixel blends the nearest 500 m cell centres, NaN ignored
coarse = np.where(score_grid == NODATA, np.nan, score_grid).astype("float32")
fine = np.full((fine_h, fine_w), np.nan, dtype="float32")

reproject(
    source=coarse, destination=fine,
    src_transform=transform, src_crs=crs,
    dst_transform=fine_transform, dst_crs=fine_crs,
    src_nodata=np.nan, dst_nodata=np.nan,
    resampling=Resampling.bilinear,
)

# Fill any forest pixel the blend could not reach with the nearest valid value
nearest = distance_transform_edt(np.isnan(fine), return_distances=False, return_indices=True)
fine = np.where(forest, fine[nearest[0], nearest[1]], np.nan)

# Light smoothing that stays inside the forest (edge pixels are not dragged down)
if DISPLAY_SMOOTHING > 0:
    inside_weight = gaussian_filter(forest.astype("float32"), DISPLAY_SMOOTHING)
    smooth = gaussian_filter(np.where(forest, fine, 0).astype("float32"), DISPLAY_SMOOTHING)
    fine = np.where(forest, smooth / np.maximum(inside_weight, 1e-6), np.nan)

# Classify the smooth surface with the thresholds found on the 500 m map
fine_class = np.where(np.isnan(fine), NODATA, np.digitize(fine, breaks) + 1)

write_raster(OUT / "susceptibility_30m_score.tif", np.nan_to_num(fine, nan=NODATA), "float32", fine_transform, fine_crs)
write_raster(OUT / "susceptibility_30m_class.tif", fine_class, "int16", fine_transform, fine_crs)

# ---------- Figure: two panels, Trial 5 design ----------
PX_W, PX_H = 3508, 2480          # layout was measured on a 300 dpi A4 sheet
FIG_W, FIG_H = PX_W / 300, PX_H / 300


def fx(px):
    return px / PX_W


def fy(px):
    return 1 - px / PX_H


# Crop to the forest
rows_any = np.where(forest.any(axis=1))[0]
cols_any = np.where(forest.any(axis=0))[0]
r0, r1, c0, c1 = rows_any[0], rows_any[-1] + 1, cols_any[0], cols_any[-1] + 1

map_left = fine_transform.c + c0 * fine_transform.a
map_right = fine_transform.c + c1 * fine_transform.a
map_top = fine_transform.f + r0 * fine_transform.e
map_bottom = fine_transform.f + r1 * fine_transform.e
image_extent = (map_left, map_right, map_bottom, map_top)

continuous = np.ma.masked_invalid(fine[r0:r1, c0:c1])
classes = np.ma.masked_equal(fine_class[r0:r1, c0:c1], NODATA)

# One scale for both maps: k inches per km, so the scale bar is true
frame_h_in = (1495 - 202) / 300
k = 0.95 * frame_h_in / ((map_top - map_bottom) / 1000)

# Colour scale for the continuous map. It is pinned to the classes, so Low areas
# are green, Medium areas yellow and High areas red in BOTH panels.
low_value, high_value = np.nanmin(fine), np.nanmax(fine)
colour_min = np.floor(low_value * 20) / 20
colour_max = np.ceil(high_value * 20) / 20
t1, t2 = breaks


def hex_to_rgb(code):
    return np.array([int(code[i:i + 2], 16) for i in (1, 3, 5)])


def blend(a, b):
    mixed = (hex_to_rgb(a) + hex_to_rgb(b)) / 2
    return "#%02x%02x%02x" % tuple(int(v) for v in mixed)


anchors = [
    (colour_min, "#127a36"),
    ((colour_min + t1) / 2, CLASS_COLOURS[0]),
    (t1, blend(CLASS_COLOURS[0], CLASS_COLOURS[1])),
    ((t1 + t2) / 2, CLASS_COLOURS[1]),
    (t2, blend(CLASS_COLOURS[1], CLASS_COLOURS[2])),
    ((t2 + colour_max) / 2, CLASS_COLOURS[2]),
    (colour_max, "#8c1010"),
]
ramp = LinearSegmentedColormap.from_list(
    "susceptibility",
    [((value - colour_min) / (colour_max - colour_min), colour) for value, colour in anchors],
)

fig = plt.figure(figsize=(FIG_W, FIG_H))


def map_frame(x0_px, x1_px, y0_px, y1_px, pad):
    """A framed axes whose limits give exactly k inches per km."""
    ax = fig.add_axes([fx(x0_px), fy(y1_px), fx(x1_px) - fx(x0_px), fy(y0_px) - fy(y1_px)])
    w_m = (x1_px - x0_px) / 300 / k * 1000
    h_m = (y1_px - y0_px) / 300 / k * 1000
    left = map_left - pad * w_m
    centre_y = (map_top + map_bottom) / 2
    ax.set_xlim(left, left + w_m)
    ax.set_ylim(centre_y - h_m / 2, centre_y + h_m / 2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    return ax


left_ax = map_frame(16, 1143, 202, 1495, pad=0.043)
right_ax = map_frame(1293, 2272, 204, 1495, pad=0.035)

image = left_ax.imshow(continuous, cmap=ramp, vmin=colour_min, vmax=colour_max,
                       extent=image_extent, interpolation="nearest")
right_ax.imshow(classes, cmap=ListedColormap(CLASS_COLOURS), vmin=1, vmax=N_CLASSES,
                extent=image_extent, interpolation="nearest")

# Titles
title = fig.text(fx(1203), fy(72), "Viphya Plantation - Wildfire Susceptibility Mapping",
                 ha="center", va="center", fontsize=14, fontweight="bold")

# Underline exactly as wide as the title text
fig.canvas.draw()
box = title.get_window_extent().transformed(fig.transFigure.inverted())
fig.add_artist(Line2D([box.x0, box.x1], [fy(100), fy(100)], color="black", linewidth=1.5))
fig.text(fx(1222), fy(150), "Random Forest Model", ha="center", va="center", fontsize=15, fontweight="bold")

# Colour bar, left panel
colour_ax = fig.add_axes([fx(717), fy(1470), fx(771) - fx(717), fy(403) - fy(1470)])
ticks = np.arange(np.ceil(colour_min * 10) / 10, colour_max + 1e-9, 0.1)
bar = fig.colorbar(image, cax=colour_ax, ticks=ticks)
bar.set_ticklabels([f"{t:.1f}" for t in ticks])
bar.outline.set_visible(False)

# Thin lines mark where the class thresholds fall on the colour bar
for threshold in breaks:
    bar.ax.axhline(threshold, color="black", linewidth=0.8)
bar.ax.tick_params(length=6, labelsize=10)
fig.text(fx(865), fy(300), "Susceptibility:\nContinuous", ha="center", va="center",
         fontsize=12, fontweight="bold")
fig.text(fx(985), fy(905), SCORE_LABEL, rotation=90, ha="center", va="center",
         fontsize=12, fontweight="bold")

# Text, right panel
fig.text(fx(2255), fy(330), "Susceptibility:\nClassification",
         ha="right", va="center", fontsize=11, fontweight="bold")
fig.text(fx(2255), fy(420), "(Low/Medium/High)", ha="right", va="center", fontsize=9, fontweight="bold")

# Legend box, right panel. It slides up from the bottom corner until it no
# longer touches the forest, whatever the shape of the forest is.
def touches_forest(x0_px, x1_px, y0_px, y1_px, margin=25):
    """True if a box in layout pixels (right panel) overlaps a forest pixel."""
    xlim, ylim = right_ax.get_xlim(), right_ax.get_ylim()

    def to_x(px):
        return xlim[0] + (px - 1293) / (2272 - 1293) * (xlim[1] - xlim[0])

    def to_y(py):
        return ylim[1] - (py - 204) / (1495 - 204) * (ylim[1] - ylim[0])

    col_a = int((to_x(x0_px - margin) - fine_transform.c) / fine_transform.a)
    col_b = int((to_x(x1_px + margin) - fine_transform.c) / fine_transform.a)
    row_a = int((fine_transform.f - to_y(y0_px - margin)) / abs(fine_transform.e))
    row_b = int((fine_transform.f - to_y(y1_px + margin)) / abs(fine_transform.e))

    block = forest[max(row_a, 0):max(row_b, 0) + 1, max(col_a, 0):max(col_b, 0) + 1]
    return block.any()


LEG_X0, LEG_X1, LEG_H = 1845, 2250, 300
leg_bottom = 1465

while leg_bottom - LEG_H > 520 and touches_forest(LEG_X0, LEG_X1, leg_bottom - LEG_H, leg_bottom):
    leg_bottom -= 10

legend_ax = fig.add_axes([fx(LEG_X0), fy(leg_bottom), fx(LEG_X1) - fx(LEG_X0), fy(leg_bottom - LEG_H) - fy(leg_bottom)])
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.axis("off")
legend_ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, linewidth=1.5, clip_on=False))
legend_ax.text(0.08, 0.92, "Random Forest Model", fontsize=7, va="center")
legend_ax.text(0.08, 0.82, "Susceptibility Ranges", fontsize=7, va="center", color="#555555")

edges = [0.0] + [round(b, 2) for b in breaks] + [1.0]

for i in range(N_CLASSES):
    y = 0.60 - i * 0.24
    legend_ax.add_patch(Rectangle((0.07, y - 0.08), 0.17, 0.16, color=CLASS_COLOURS[i]))
    legend_ax.text(0.31, y + 0.05, LABELS[i], fontsize=8, va="center", color="#222222")
    legend_ax.text(0.31, y - 0.05, f"({edges[i]:g} - {edges[i + 1]:g})", fontsize=7,
                   va="center", color="#555555")


def north_arrow(x0_px, y0_px):
    """Outline arrow with a hatched left half."""
    ax = fig.add_axes([fx(x0_px), fy(y0_px + 90), fx(68), fy(0) - fy(90)])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Polygon([(0.5, 1), (0, 0), (0.5, 0.25), (1, 0)], closed=True,
                         fill=False, linewidth=1.3, edgecolor="black"))
    ax.add_patch(Polygon([(0.5, 1), (0, 0), (0.5, 0.25)], closed=True,
                         fill=False, hatch="////", linewidth=0, edgecolor="#555555"))


north_arrow(50, 238)
north_arrow(1335, 225)

# Scale bar, true to the map scale
bar_left = fx(578)
bar_y = fy(1675)

for km in range(0, 81, 20):
    x = bar_left + km * k / FIG_W
    fig.add_artist(Line2D([x, x], [bar_y, bar_y + 0.012], color="black", linewidth=1.3))
    fig.text(x, bar_y + 0.018, f"{km} km" if km == 80 else f"{km}",
             ha="center", va="bottom", fontsize=12)

fig.add_artist(Line2D([bar_left, bar_left + 80 * k / FIG_W], [bar_y, bar_y], color="black", linewidth=1.3))

fig.text(fx(16), fy(1790),
         "Mean of the 2016 to 2024 model scores (relative susceptibility, not probabilities). "
         "Model grid 500 m, smoothed to the 30.6 m DEM grid for display only. "
         "Classes by natural breaks.",
         ha="left", va="center", fontsize=7, style="italic", color="#555555")

fig.savefig(OUT / "susceptibility_map.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Figure: burned share by class ----------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.bar(validation["label"], validation["burned_share_pct"], color=CLASS_COLOURS, edgecolor="black")
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
print("Class thresholds:", [round(b, 3) for b in breaks])
print(area_table.to_string(index=False))
print()
print(validation.to_string(index=False))
print()
print(pd.read_csv(OUT / "map_summary.csv").to_string(index=False))