"""Phase 9a: validation of the susceptibility map against raw FIRMS fire points.

Same design as the Trial 5 validation:
  1. Fire points (fire = 1): every raw MODIS detection with confidence 75 or
     more in the validation period (not clustered into events).
  2. Non-fire points (fire = 0): random points inside the forest, at least
     BUFFER_M from every fire point, as many as there are fire points.
  3. Each point is labelled with the class of the map cell it falls in.
  4. Two pie charts (where fire points fall, where non-fire points fall) and
     two overlay maps (side by side, and with hotspot outlines).

The map is the Phase 7c susceptibility map (mean of the 2016 to 2024 scores).
FIRMS points were not used as labels or predictors in the model, so this check
does not use the training labels. FIRMS and the burned area product both come
from MODIS, so they share some blind spots.

Writes to outputs/:
  validation_points.csv            every point with fire = 1 or 0 and its class
  class_distribution.csv           counts and shares per class for each group
  validation_summary.csv           AUC of the score, fire against non-fire points
  RF_Fire_1_Distribution.png       pie: where the fire points fall
  RF_Fire_0_Distribution.png       pie: where the non-fire points fall
  Validation_A_SideBySide.png/pdf  map of classes next to fire hotspot density
  Validation_B_Overlay.png/pdf     map of classes with hotspot outlines
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

# ---------- Paths, relative to the Phase 9a folder ----------
BASE = Path(__file__).resolve().parents[1]
DETECTIONS = BASE / "data" / "fire_detections_filtered.csv"
CLASS_500 = BASE / "data" / "susceptibility_500m_class.tif"
SCORE_500 = BASE / "data" / "susceptibility_500m_score.tif"
CLASS_30 = BASE / "data" / "susceptibility_30m_class.tif"
BREAKS = BASE / "data" / "class_breaks.csv"
BOUNDARY = BASE / "data" / "Chikangawa_forest" / "Chikangawa_forest.shp"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------- Settings ----------
YEAR_FROM, YEAR_TO = 2016, 2024   # validation period, the years the map is built from
BUFFER_M = 500                    # non-fire points stay this far from any fire point
SEED = 42
HOT = 0.6                         # density above this counts as a hotspot
SCALE_KM = 30
TEST_AUC = 0.603                  # pooled test AUC from Phase 7a, for the data box
NODATA = -9999

LABELS = ["Low", "Medium", "High"]
COLOURS = ["#00b050", "#e2e24a", "#f5333a"]

thresholds = pd.read_csv(BREAKS)["score_threshold"].to_numpy()
edges = [0.0] + [round(t, 2) for t in thresholds] + [1.0]
RANGES = [f"{LABELS[i]} ({edges[i]:.2f}-{edges[i + 1]:.2f})" for i in range(3)]
RANGES_SHORT = [f"{LABELS[i]} ({edges[i]:g} - {edges[i + 1]:g})" for i in range(3)]

# ---------- Map and boundary ----------
with rasterio.open(CLASS_500) as src:
    transform = src.transform
    height, width = src.height, src.width
    class_grid = src.read(1)

with rasterio.open(SCORE_500) as src:
    score_grid = src.read(1)

boundary = gpd.read_file(BOUNDARY).to_crs(32736)
forest = boundary.geometry.unary_union


def lookup(x, y):
    """Class and mean score of the 500 m cell under each point (nodata if none)."""
    col = np.floor((x - transform.c) / transform.a).astype(int)
    row = np.floor((y - transform.f) / transform.e).astype(int)
    ok = (row >= 0) & (row < height) & (col >= 0) & (col < width)

    classes = np.full(len(x), NODATA, dtype=int)
    scores = np.full(len(x), np.nan)
    classes[ok] = class_grid[row[ok], col[ok]]
    scores[ok] = score_grid[row[ok], col[ok]]
    return classes, scores


# ---------- Fire points: raw detections, confidence 75 or more ----------
fires = pd.read_csv(DETECTIONS, parse_dates=["acq_datetime"])
fires = fires[fires["acq_datetime"].dt.year.between(YEAR_FROM, YEAR_TO)].copy()

to_utm = Transformer.from_crs(4326, 32736, always_xy=True)
fires["x"], fires["y"] = to_utm.transform(fires["LONGITUDE"].to_numpy(), fires["LATITUDE"].to_numpy())
fires["class"], fires["mean_score"] = lookup(fires["x"].to_numpy(), fires["y"].to_numpy())

n_detections = len(fires)
fires = fires[fires["class"] != NODATA].reset_index(drop=True)
print(f"Fire points: {n_detections} detections, {len(fires)} fall on a map cell")

# ---------- Non-fire points ----------
rng = np.random.RandomState(SEED)
fire_tree = cKDTree(fires[["x", "y"]].to_numpy())
min_x, min_y, max_x, max_y = forest.bounds


def sample_non_fire(n, buffer_m):
    """Random points inside the forest, away from fire points, on a map cell."""
    kept = []

    while sum(len(k) for k in kept) < n:
        px = rng.uniform(min_x, max_x, 20000)
        py = rng.uniform(min_y, max_y, 20000)
        keep = shapely.contains_xy(forest, px, py)

        if buffer_m > 0:
            distance, _ = fire_tree.query(np.column_stack([px, py]))
            keep &= distance >= buffer_m

        px, py = px[keep], py[keep]
        classes, scores = lookup(px, py)
        on_map = classes != NODATA
        kept.append(pd.DataFrame({"x": px[on_map], "y": py[on_map],
                                  "class": classes[on_map], "mean_score": scores[on_map]}))

    return pd.concat(kept, ignore_index=True).iloc[:n]


non_fire = sample_non_fire(len(fires), BUFFER_M)
non_fire_open = sample_non_fire(len(fires), 0)   # no buffer, for the AUC check only

# ---------- Points table ----------
to_lonlat = Transformer.from_crs(32736, 4326, always_xy=True)


def table(frame, fire_value, date=None):
    lon, lat = to_lonlat.transform(frame["x"].to_numpy(), frame["y"].to_numpy())
    return pd.DataFrame({
        "fire": fire_value,
        "x": frame["x"].round(1), "y": frame["y"].round(1),
        "lon": np.round(lon, 5), "lat": np.round(lat, 5),
        "date": date if date is not None else "",
        "mean_score": frame["mean_score"].round(4),
        "class": frame["class"],
        "class_label": [LABELS[c - 1] for c in frame["class"]],
    })


points = pd.concat([
    table(fires, 1, fires["acq_datetime"].dt.strftime("%Y-%m-%d").to_numpy()),
    table(non_fire, 0),
], ignore_index=True)
points.insert(0, "point_id", np.arange(1, len(points) + 1))
points.to_csv(OUT / "validation_points.csv", index=False)

# ---------- Class distribution ----------
area_share = [100 * (class_grid == k).sum() / (class_grid != NODATA).sum() for k in (1, 2, 3)]
rows = []

for name, frame in (("fire = 1", fires), ("fire = 0", non_fire)):
    counts = frame["class"].value_counts().reindex([1, 2, 3], fill_value=0)

    for k in (1, 2, 3):
        rows.append({
            "group": name, "class": k, "label": LABELS[k - 1],
            "points": int(counts[k]),
            "share_pct": round(100 * counts[k] / counts.sum(), 1),
            "forest_area_share_pct": round(area_share[k - 1], 1),
        })

distribution = pd.DataFrame(rows)
distribution.to_csv(OUT / "class_distribution.csv", index=False)

# ---------- AUC: map score, fire points against non-fire points ----------
def point_auc(other):
    label = np.r_[np.ones(len(fires)), np.zeros(len(other))]
    score = np.r_[fires["mean_score"], other["mean_score"]]
    return round(roc_auc_score(label, score), 3)


pd.DataFrame([
    {"measure": "fire points used", "value": len(fires)},
    {"measure": "non-fire points used", "value": len(non_fire)},
    {"measure": "validation period", "value": f"{YEAR_FROM} to {YEAR_TO}"},
    {"measure": f"non-fire buffer (m)", "value": BUFFER_M},
    {"measure": f"AUC, fire vs non-fire points ({BUFFER_M} m buffer)", "value": point_auc(non_fire)},
    {"measure": "AUC, fire vs non-fire points (no buffer)", "value": point_auc(non_fire_open)},
]).to_csv(OUT / "validation_summary.csv", index=False)

# ---------- Pie charts ----------
def pie_chart(frame, title, name):
    counts = frame["class"].value_counts().reindex([1, 2, 3], fill_value=0).to_numpy()
    shares = 100 * counts / counts.sum()

    fig, ax = plt.subplots(figsize=(8.5, 9))
    wedges, _, autotexts = ax.pie(
        counts, colors=COLOURS, startangle=140, counterclock=False,
        autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
        pctdistance=0.62, wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=14, fontweight="bold"),
    )

    for text, colour in zip(autotexts, ["white", "black", "white"]):
        text.set_color(colour)

    # Small slices get their label outside, with the share
    for wedge, label, share in zip(wedges, RANGES, shares):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        text = label if share >= 5 else f"{label}\n{share:.1f}%"
        ax.text(1.18 * np.cos(angle), 1.18 * np.sin(angle), text, ha="center", va="center", fontsize=13)

    handles = [Patch(fc=c, label=f"{r}  (n={n:,})") for c, r, n in zip(COLOURS, RANGES, counts)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=13)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=30)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


pie_chart(fires, "Susceptibility Class Distribution\nFire Occurrence Points (Fire = 1)", "RF_Fire_1_Distribution.png")
pie_chart(non_fire, "Susceptibility Class Distribution\nNon-Fire Points (Fire = 0)", "RF_Fire_0_Distribution.png")

# ---------- Maps: hotspot density ----------
with rasterio.open(CLASS_30) as src:
    display = src.read(1)
    rb = src.bounds

display = np.ma.masked_where(~np.isin(display, [1, 2, 3]), display)

bounds = boundary.total_bounds
pad = 2000
xlim = (bounds[0] - pad, bounds[2] + pad)
ylim = (bounds[1] - pad, bounds[3] + pad)

gx = np.arange(*xlim, 500)
gy = np.arange(*ylim, 500)
xx, yy = np.meshgrid(gx, gy)
kde = gaussian_kde(fires[["x", "y"]].to_numpy().T, bw_method=0.1)
density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
density = (density - density.min()) / (density.max() - density.min())

susc_cmap = ListedColormap(COLOURS)
susc_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3)
fire_cmap = LinearSegmentedColormap.from_list(
    "fire", ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#e31a1c", "#800026"])
extent = [rb.left, rb.right, rb.bottom, rb.top]


def forest_path():
    """The forest outline as a matplotlib path, used to clip the density."""
    polygons = forest.geoms if forest.geom_type == "MultiPolygon" else [forest]
    verts, codes = [], []

    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            xy = np.asarray(ring.coords)
            verts.extend(xy)
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY]

    return MplPath(verts, codes)


FOREST_PATH = forest_path()


def clip(artist, ax):
    patch = PathPatch(FOREST_PATH, transform=ax.transData, fc="none", ec="none")
    ax.add_patch(patch)

    try:
        artist.set_clip_path(patch)
    except Exception:
        for part in artist.collections:
            part.set_clip_path(patch)


def map_ax(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def frame(fig):
    fig.add_artist(Rectangle((0.003, 0.003), 0.994, 0.994, transform=fig.transFigure,
                             fill=False, ec="black", lw=1.5))


def north_arrow(fig, x, y, w, h):
    a = fig.add_axes([x, y, w, h])
    a.set_xlim(0, 0.7)
    a.set_ylim(-0.25, 1)
    a.axis("off")
    a.add_patch(Polygon([(0.35, 1), (0, 0), (0.35, 0.25)], fc="black", ec="black", lw=1.2))
    a.add_patch(Polygon([(0.35, 1), (0.7, 0), (0.35, 0.25)], fc="white", ec="black", lw=1.2))
    a.text(0.35, -0.22, "N", ha="center", va="bottom", fontsize=11, fontweight="bold")


def scale_bar(fig, ax, x0, yb):
    ax.apply_aspect()
    pos = ax.get_position()
    fraction_per_m = pos.width / (ax.get_xlim()[1] - ax.get_xlim()[0])
    fig.add_artist(Line2D([x0, x0 + SCALE_KM * 1000 * fraction_per_m], [yb, yb],
                          color="black", lw=1.5, transform=fig.transFigure))

    for km in range(0, SCALE_KM + 1, 10):
        xt = x0 + km * 1000 * fraction_per_m
        fig.add_artist(Line2D([xt, xt], [yb, yb - 0.012], color="black", lw=1.5, transform=fig.transFigure))
        fig.text(xt, yb + 0.012, f"{km} km" if km == SCALE_KM else str(km),
                 ha="center", va="bottom", fontsize=11)


def data_box(fig, x, y, text):
    fig.text(x, y, text, fontsize=10, ha="left", va="bottom", linespacing=1.4,
             bbox=dict(boxstyle="square,pad=0.6", fc="white", ec="black", lw=1.2))


def draw_classes(ax):
    ax.imshow(display, cmap=susc_cmap, norm=susc_norm, extent=extent, origin="upper",
              interpolation="nearest", zorder=1)


source_text = (f"Data Source:  Random Forest model\n(test AUC {TEST_AUC}); MODIS Fire\n"
               f"Detections (n = {len(fires):,}, {YEAR_FROM}-{YEAR_TO},\nconfidence 75 or more)\n\nCRS: EPSG 32736")

# Option A: side by side
fig = plt.figure(figsize=(16, 9), facecolor="white")
frame(fig)

ax1 = map_ax(fig, [0.02, 0.04, 0.29, 0.82])
draw_classes(ax1)
boundary.plot(ax=ax1, facecolor="none", edgecolor="black", linewidth=1.2, zorder=3)
ax1.set_title("A) Susceptibility (Random Forest)", fontsize=13, fontweight="bold", pad=8)

ax2 = map_ax(fig, [0.32, 0.04, 0.29, 0.82])
image = ax2.contourf(xx, yy, density, levels=12, cmap=fire_cmap, zorder=1)
clip(image, ax2)
ax2.scatter(fires["x"], fires["y"], s=3, c="darkred", alpha=0.35, zorder=2)
boundary.plot(ax=ax2, facecolor="none", edgecolor="black", linewidth=1.2, zorder=3)
ax2.set_title("B) Fire Hotspot Density", fontsize=13, fontweight="bold", pad=8)

north_arrow(fig, 0.012, 0.87, 0.05, 0.11)
fig.text(0.81, 0.90, "Susceptibility vs.\nFire Occurrence", fontsize=24, fontweight="bold",
         ha="center", va="center", linespacing=1.3)

legend_ax = fig.add_axes([0.66, 0.58, 0.14, 0.2])
legend_ax.axis("off")
legend_ax.legend(handles=[Patch(fc=c, ec="black", label=l) for c, l in zip(COLOURS, RANGES_SHORT)],
                 loc="upper left", title="Susceptibility", title_fontsize=12, fontsize=11,
                 frameon=False, alignment="left")

colour_ax = fig.add_axes([0.85, 0.30, 0.02, 0.45])
bar = fig.colorbar(image, cax=colour_ax)
bar.set_ticks([0, 0.5, 1])
bar.set_ticklabels(["0 (Low)", "0.5", "1 (High)"])
bar.outline.set_visible(False)
bar.ax.tick_params(labelsize=11)
fig.text(0.85, 0.765, "Fire Density", fontsize=12, ha="left", va="bottom")

scale_bar(fig, ax1, 0.66, 0.22)
data_box(fig, 0.66, 0.03, source_text)

fig.savefig(OUT / "Validation_A_SideBySide.png", dpi=300, facecolor="white")
fig.savefig(OUT / "Validation_A_SideBySide.pdf", facecolor="white")
plt.close(fig)

# Option B: overlay with hotspot outlines
fig = plt.figure(figsize=(11, 9.5), facecolor="white")
frame(fig)

ax = map_ax(fig, [0.02, 0.03, 0.42, 0.94])
ax.set_anchor("W")
draw_classes(ax)

plt.rcParams["hatch.linewidth"] = 0.8
hot = ax.contourf(xx, yy, density, levels=[HOT, 1.01], colors="none", hatches=["////"], zorder=2)
clip(hot, ax)
outline = ax.contour(xx, yy, density, levels=[HOT], colors="black", linewidths=1.3, zorder=2)
clip(outline, ax)
boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.2, zorder=3)

north_arrow(fig, 0.012, 0.87, 0.06, 0.11)
fig.text(0.76, 0.86, "Model Validation:\nSusceptibility + Fire Hotspots",
         fontsize=22, fontweight="bold", ha="center", va="center", linespacing=1.3)

legend_ax = fig.add_axes([0.5, 0.52, 0.3, 0.24])
legend_ax.axis("off")
handles = [Patch(fc=c, ec="black", label=l) for c, l in zip(COLOURS, RANGES_SHORT)]
handles.append(Patch(fc="white", ec="black", hatch="////", label=f"Fire hotspot (density > {HOT})"))
legend_ax.legend(handles=handles, loc="upper left", title="Legend", title_fontsize=13,
                 fontsize=12, frameon=False, alignment="left")

scale_bar(fig, ax, 0.52, 0.42)
data_box(fig, 0.52, 0.05, source_text)

fig.savefig(OUT / "Validation_B_Overlay.png", dpi=300, facecolor="white")
fig.savefig(OUT / "Validation_B_Overlay.pdf", facecolor="white")
plt.close(fig)

# ---------- Print ----------
print(distribution.to_string(index=False))
print()
print(pd.read_csv(OUT / "validation_summary.csv").to_string(index=False))