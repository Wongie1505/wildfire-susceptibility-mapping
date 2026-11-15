"""Phase 5: build the modelling table.

One row per forest cell per fire season (2002 to 2024), with the label, the
14 predictors, a 5 km spatial block ID and the train or test flag.

Writes to outputs/:
  modelling_table.csv       the frozen table used by every later phase
  table_summary.csv         rows and burned share by split
  collinearity.csv          predictor pairs with Spearman |rho| of 0.7 or more
  single_predictor_auc.csv  AUC of each predictor alone, training rows only
  table_settings.csv        split, block size and seed
  figures/                  collinearity_heatmap.png, single_predictor_auc.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

# Paths, relative to the Phase 5 folder
BASE = Path(__file__).resolve().parents[1]
STATIC = BASE / "data" / "static_stack" / "static_stack.tif"
BURNED = BASE / "data" / "burned"
PREDICTORS = BASE / "data" / "predictors"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

YEARS = list(range(2002, 2025))
TRAIN_LAST = 2015  # training 2002 to 2015, test 2016 to 2024
BLOCK_M = 5000     # spatial block size for cross-validation
SEED = 42
NODATA = -9999

# Band order of static_stack.tif (Phase 2)
STATIC_BANDS = [
    "dem", "slope", "aspect_east", "aspect_north",
    "rivers_dist", "roads_dist", "settlement_dist",
]

# Seasonal predictors (Phase 4), one folder each
SEASONAL = [
    "rain_anomaly", "ndvi_peak", "ndmi_anomaly", "lst_anomaly",
    "lulc", "burn_count_3yr", "years_since_burn",
]

FEATURES = STATIC_BANDS + SEASONAL

# ---------- Static stack: grid, forest mask and static predictors ----------
with rasterio.open(STATIC) as src:
    transform = src.transform
    height, width = src.height, src.width
    static = src.read()

    if list(src.descriptions) != STATIC_BANDS:
        raise ValueError(f"Static band order is {src.descriptions}, expected {STATIC_BANDS}")

inside = static[0] != NODATA


def read_one(path):
    """Read a single-band file and stop if it is not on the static grid."""
    with rasterio.open(path) as src:
        same = np.allclose(tuple(src.transform)[:6], tuple(transform)[:6], atol=1e-3)

        if not same or (src.height, src.width) != (height, width):
            raise ValueError(f"{path.name} is not on the Phase 2 grid.")

        return src.read(1)


def find_file(predictor, year):
    """Find the file for a predictor and season. Folder may carry a P4_ prefix."""
    for folder in (predictor, f"P4_{predictor}"):
        matches = list((PREDICTORS / folder).glob(f"*_{year}.tif"))

        if matches:
            return matches[0]

    raise FileNotFoundError(f"No file for {predictor} {year} in {PREDICTORS}")


# ---------- Build one block of rows per season ----------
tables = []
dropped = 0

for year in YEARS:
    label = read_one(BURNED / f"burned_{year}.tif")

    seasonal = {name: read_one(find_file(name, year)) for name in SEASONAL}

    # Cells with a label and a value in every predictor
    valid = inside & (label != NODATA)

    for name in STATIC_BANDS:
        valid &= static[STATIC_BANDS.index(name)] != NODATA

    for name in SEASONAL:
        valid &= seasonal[name] != NODATA

    dropped += int((inside & (label != NODATA)).sum() - valid.sum())

    rows, cols = np.where(valid)
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e

    # 5 km block ID, for example "115_1744"
    block_x = pd.Series(np.floor(x / BLOCK_M).astype(int)).astype(str)
    block_y = pd.Series(np.floor(y / BLOCK_M).astype(int)).astype(str)

    frame = pd.DataFrame({
        "cell_id": rows * width + cols,
        "x": x,
        "y": y,
        "year": year,
        "split": "train" if year <= TRAIN_LAST else "test",
        "block_id": block_x + "_" + block_y,
        "burned": label[valid],
    })

    for name in STATIC_BANDS:
        frame[name] = static[STATIC_BANDS.index(name)][valid]

    for name in SEASONAL:
        frame[name] = seasonal[name][valid]

    tables.append(frame)

table = pd.concat(tables, ignore_index=True)
table.to_csv(OUT / "modelling_table.csv", index=False, float_format="%.4f")

# ---------- Summary by split ----------
summary = table.groupby("split").agg(
    rows=("burned", "size"),
    burned=("burned", "sum"),
    first_year=("year", "min"),
    last_year=("year", "max"),
    blocks=("block_id", "nunique"),
).reset_index()
# Cast to float first: the burned column is int16, and 100 x count overflows it
summary["burned_pct"] = (100 * summary["burned"].astype("float64") / summary["rows"]).round(2)
summary.to_csv(OUT / "table_summary.csv", index=False)

# ---------- Collinearity, training rows only ----------
train = table[table["split"] == "train"]
corr = train[FEATURES].corr(method="spearman")

pairs = []

for i, a in enumerate(FEATURES):
    for b in FEATURES[i + 1:]:
        rho = corr.loc[a, b]

        if abs(rho) >= 0.7:
            pairs.append({"predictor_a": a, "predictor_b": b, "spearman": round(rho, 3)})

pairs = pd.DataFrame(pairs, columns=["predictor_a", "predictor_b", "spearman"])
pairs.to_csv(OUT / "collinearity.csv", index=False)

# ---------- Leakage check: each predictor alone, training rows only ----------
# A very high AUC from one predictor would suggest it knows about the fire.
auc_rows = []

for name in FEATURES:
    auc = roc_auc_score(train["burned"], train[name])
    auc_rows.append({"predictor": name, "auc_alone": round(max(auc, 1 - auc), 3)})

single = pd.DataFrame(auc_rows).sort_values("auc_alone", ascending=False)
single.to_csv(OUT / "single_predictor_auc.csv", index=False)

# ---------- Figures for the dissertation ----------
LABELS = {
    "dem": "Elevation",
    "slope": "Slope",
    "aspect_east": "Aspect (eastness)",
    "aspect_north": "Aspect (northness)",
    "rivers_dist": "Distance to rivers",
    "roads_dist": "Distance to roads",
    "settlement_dist": "Distance to settlements",
    "rain_anomaly": "Rainfall anomaly",
    "ndvi_peak": "NDVI peak",
    "ndmi_anomaly": "NDMI anomaly",
    "lst_anomaly": "LST anomaly",
    "lulc": "Land cover",
    "burn_count_3yr": "Burns in last 3 seasons",
    "years_since_burn": "Years since last burn",
}

FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

# Figure 1: Spearman correlation heatmap (lower triangle only)
n = len(FEATURES)
names = [LABELS[f] for f in FEATURES]
values = corr.values

# Drop the empty first row and last column, keep the lower triangle
hidden = np.triu(np.ones((n - 1, n - 1), dtype=bool), k=1)
shown = np.ma.masked_array(values[1:, :-1], mask=hidden)

fig, ax = plt.subplots(figsize=(10, 8))
image = ax.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1)

ax.set_xticks(range(n - 1))
ax.set_xticklabels(names[:-1], rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(n - 1))
ax.set_yticklabels(names[1:])

# Write each value, in bold where |rho| is 0.7 or more
for r in range(n - 1):
    for c in range(r + 1):
        rho = values[r + 1, c]
        ax.text(
            c, r, f"{rho:.2f}",
            ha="center", va="center", fontsize=8,
            color="white" if abs(rho) > 0.6 else "black",
            fontweight="bold" if abs(rho) >= 0.7 else "normal",
        )

colourbar = fig.colorbar(image, ax=ax, shrink=0.8)
colourbar.set_label("Spearman correlation", labelpad=14)
ax.set_title("Correlation between predictors (training years 2002 to 2015)\nBold = |rho| of 0.7 or more")
fig.tight_layout()
fig.savefig(FIG / "collinearity_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 2: each predictor alone, AUC against the label
ranked = single.sort_values("auc_alone")
colours = ["#2b6cb0" if f in STATIC_BANDS else "#dd6b20" for f in ranked["predictor"]]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(
    [LABELS[f] for f in ranked["predictor"]],
    ranked["auc_alone"] - 0.5,
    left=0.5,
    color=colours,
)

for i, auc in enumerate(ranked["auc_alone"]):
    ax.text(auc + 0.004, i, f"{auc:.3f}", va="center", fontsize=9)

ax.axvline(0.5, color="grey", linestyle="--", label="Chance (0.5)")
ax.axvline(0.8, color="red", linestyle=":", label="Leakage warning (0.8)")
ax.set_xlim(0.45, 0.85)
ax.set_xlabel("AUC of the predictor on its own (direction ignored)")
ax.set_title("Single-predictor check for leakage (training years 2002 to 2015)")

# Legend: the two lines plus the two predictor groups
handles, _ = ax.get_legend_handles_labels()
handles += [
    Patch(color="#2b6cb0", label="Static predictor"),
    Patch(color="#dd6b20", label="Seasonal predictor"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)

fig.tight_layout()
fig.savefig(FIG / "single_predictor_auc.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Settings ----------
pd.DataFrame([
    {"setting": "training years", "value": f"{YEARS[0]} to {TRAIN_LAST}"},
    {"setting": "test years", "value": f"{TRAIN_LAST + 1} to {YEARS[-1]}"},
    {"setting": "spatial block size (m)", "value": BLOCK_M},
    {"setting": "random seed", "value": SEED},
    {"setting": "predictors", "value": len(FEATURES)},
]).to_csv(OUT / "table_settings.csv", index=False)

# ---------- Print ----------
print("Rows:", len(table), "| dropped for missing values:", dropped)
print(summary.to_string(index=False))
print("\nPairs with |Spearman| >= 0.7:")
print(pairs.to_string(index=False) if len(pairs) else "none")
print("\nStrongest single predictors (training):")
print(single.head(6).to_string(index=False))