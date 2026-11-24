"""Phase 3: build Sentinel-2 labels per cell-season, 2018 to 2024.

Rules fixed before any scoring (see log.txt):
  headline AUC        continuous burned_fraction (no threshold)
  binary label        s2_burned = burned_fraction >= 0.10 (kappa, confusion matrix)
  sensitivity labels  >= 0.05 and >= 0.25, reported in Phase 4
  coverage rule       rows with valid_fraction < 0.5 are kept but flagged
                      as not usable for scoring

Writes outputs/s2_labels.csv with one row per cell per season.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

BASE = Path(__file__).resolve().parents[1]
S2 = BASE / "data" / "s2"
TABLE = BASE / "data" / "modelling_table.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

YEARS = range(2018, 2025)
COVER_MIN = 0.5
CUT_PRIMARY = 0.10
CUT_LOW, CUT_HIGH = 0.05, 0.25

table = pd.read_csv(TABLE)
table = table[table["year"].isin(YEARS)][["x", "y", "year", "cell_id", "split"]]

frames = []
for year in YEARS:
    with rasterio.open(S2 / f"S2_burn_{year}.tif") as src:
        burned = src.read(1)
        valid = src.read(2)
        t = src.transform
    rows, cols = np.indices(burned.shape)
    xs = t.c + (cols + 0.5) * t.a
    ys = t.f + (rows + 0.5) * t.e
    grid = pd.DataFrame({
        "x": xs.ravel(), "y": ys.ravel(),
        "burned_fraction": burned.ravel(), "valid_fraction": valid.ravel(),
    })
    sub = table[table["year"] == year]
    merged = sub.merge(grid, on=["x", "y"], how="left")
    frames.append(merged)

labels = pd.concat(frames, ignore_index=True)
# Unusable or unmatched rows get no label (NaN), not zero
labels["usable"] = labels["valid_fraction"] >= COVER_MIN
burned_frac = labels["burned_fraction"].where(labels["usable"])
labels["s2_burned"] = (burned_frac >= CUT_PRIMARY).astype("float").where(labels["usable"])
labels["s2_burned_low"] = (burned_frac >= CUT_LOW).astype("float").where(labels["usable"])
labels["s2_burned_high"] = (burned_frac >= CUT_HIGH).astype("float").where(labels["usable"])

labels.to_csv(OUT / "s2_labels.csv", index=False)
summary = labels.groupby("year").agg(
    rows=("cell_id", "size"),
    matched=("burned_fraction", lambda s: int(s.notna().sum())),
    usable=("usable", "sum"),
    burned_share_usable=("s2_burned", "mean"),
)
summary.to_csv(OUT / "s2_labels_summary.csv")
print(summary.to_string())
print("done")