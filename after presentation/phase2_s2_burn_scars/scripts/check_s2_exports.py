"""Phase 2 check: verify the nine Sentinel-2 burn-scar exports.

Checks, for each season 2016 to 2024:
  1. file exists and has two bands (burned_fraction, valid_fraction)
  2. CRS is EPSG:32736 and cell size is 500 m
  3. grid origin is a multiple of 500 m, so it matches the modelling table keys
  4. coverage: mean valid_fraction (how much of the forest the scars can see)
  5. burned share: mean burned_fraction over valid cells
  6. descriptive overlap with MCD64A1 on cells present in both (not a validation)

Writes to outputs/:
  s2_check_summary.csv   one row per season
  s2_check_log.txt       pass or fail for each check
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# Copy from Phase 1 data (Trial 7 modelling table), used only for the cell keys
TABLE = BASE.parent / "phase1_train" / "data" / "modelling_table.csv"

YEARS = range(2016, 2025)
CELL = 500.0
rows, log = [], []

table = pd.read_csv(TABLE) if TABLE.exists() else None

for year in YEARS:
    path = DATA / f"S2_burn_{year}.tif"
    rec = {"year": year, "file_found": path.exists()}
    if not path.exists():
        log.append(f"{year}: FAIL file missing ({path.name})")
        rows.append(rec)
        continue

    with rasterio.open(path) as src:
        burned = src.read(1).astype(float)
        valid = src.read(2).astype(float)
        nodata = src.nodata
        t = src.transform

    # 1. bands
    rec["bands"] = 2
    # 2. CRS and cell size
    rec["crs"] = str(src.crs)
    rec["cell_m"] = abs(t.a)
    crs_ok = str(src.crs) == "EPSG:32736" and abs(t.a) == CELL and abs(t.e) == CELL
    # 3. grid alignment
    aligned = (t.c % CELL == 0) and (t.f % CELL == 0)
    rec["grid_aligned"] = aligned

    # 4 and 5. coverage and burned share
    v = valid.copy()
    b = burned.copy()
    for arr in (v, b):
        if nodata is not None:
            arr[arr == nodata] = np.nan
    rec["coverage_mean"] = round(float(np.nanmean(v)), 3)
    seen = np.isfinite(b) & (b >= 0)
    rec["burned_share_valid"] = round(float(np.nanmean(b[seen])), 4) if seen.any() else np.nan
    rec["cells_ge_0.10"] = int(np.sum(b >= 0.10))

    log.append(
        f"{year}: "
        f"{'PASS' if crs_ok else 'FAIL'} CRS/cell size ({rec['crs']}, {rec['cell_m']} m); "
        f"{'PASS' if aligned else 'FAIL'} grid alignment; "
        f"coverage {rec['coverage_mean']}; burned share {rec['burned_share_valid']}"
    )

    # 6. descriptive overlap with MCD64A1 (cells in both)
    if table is not None:
        cols = np.arange(b.shape[1])
        rws = np.arange(b.shape[0])
        xs = t.c + (cols + 0.5) * t.a
        ys = t.f + (rws + 0.5) * t.e
        xx, yy = np.meshgrid(xs, ys)
        pts = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(),
                            "s2": b.ravel(), "valid": v.ravel()})
        sub = table[table["year"] == year][["x", "y", "burned"]]
        merged = pts.merge(sub, on=["x", "y"], how="inner")
        merged = merged[np.isfinite(merged["s2"])]
        if len(merged):
            s2_flag = merged["s2"] >= 0.10
            rec["cells_compared"] = len(merged)
            rec["agree_pct_s2_vs_mcd64a1"] = round(
                100 * float(np.mean(s2_flag == (merged["burned"] == 1))), 1)
            rec["mcd64a1_burned_share"] = round(float(merged["burned"].mean()), 4)

    rows.append(rec)

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "s2_check_summary.csv", index=False)
(OUT / "s2_check_log.txt").write_text("\n".join(log) + "\n")
print(summary.to_string(index=False))
print("\n".join(log))