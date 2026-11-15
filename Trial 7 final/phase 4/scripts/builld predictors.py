"""Phase 4: check and clean the seasonal predictor files from GEE.

For each predictor and season (2002 to 2024) this reads the GEE file, checks
it sits on the Phase 2 grid, blanks everything outside the forest, and writes
a clean copy. It also builds two fire history layers from the Phase 3 burned
maps. Output folders, one file per season in each:

  rain_anomaly, ndvi_peak, ndmi_anomaly, lst_anomaly, lulc
  burn_count_3yr      seasons burned out of the previous 3
  years_since_burn    years since the cell last burned, capped at 5

Also writes predictor_check.csv with one row per file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# Paths, relative to the Phase 4 folder
BASE = Path(__file__).resolve().parents[1]
STATIC = BASE / "data" / "static_stack" / "static_stack.tif"
BURNED = BASE / "data" / "burned"
EXPORTS = BASE / "data" / "gee_exports"
OUT = BASE / "outputs"

YEARS = list(range(2002, 2025))
NODATA = -9999
HISTORY_CAP = 5

# Predictor folder name -> output data type
PREDICTORS = {
    "P4_rain_anomaly": "float32",
    "P4_ndvi_peak": "float32",
    "P4_ndmi_anomaly": "float32",
    "P4_lst_anomaly": "float32",
    "P4_lulc": "int16",
}

# ---------- Phase 2 grid and forest mask ----------
with rasterio.open(STATIC) as src:
    transform = src.transform
    crs = src.crs
    height, width = src.height, src.width
    inside = src.read(1) != src.nodata  # DEM band: valid only inside the forest


def read_file(path):
    """Read one GEE file and stop if it is not on the Phase 2 grid."""
    with rasterio.open(path) as src:
        same_grid = np.allclose(tuple(src.transform)[:6], tuple(transform)[:6], atol=1e-3)

        if not same_grid or (src.height, src.width) != (height, width):
            raise ValueError(
                f"{path.name} is not on the Phase 2 grid.\n"
                f"  Phase 2: {tuple(transform)[:6]}, {height} x {width}\n"
                f"  This file: {tuple(src.transform)[:6]}, {src.height} x {src.width}\n"
                "Fix X0, Y1, NCOL, NROW in gee_predictors.js and export again."
            )

        return src.read(1)


def write_file(folder, name, array, dtype):
    """Write one clean single-band file, nodata outside the forest."""
    (OUT / folder).mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        OUT / folder / name,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        crs=crs,
        transform=transform,
        dtype=dtype,
        nodata=NODATA,
    ) as dst:
        dst.write(array.astype(dtype), 1)


check_rows = []


def log_check(folder, year, array):
    """One row per file: how many forest cells are empty, and the value range."""
    in_forest = array[inside]
    good = in_forest[in_forest != NODATA]

    check_rows.append({
        "predictor": folder,
        "year": year,
        "cells_in_forest": int(inside.sum()),
        "cells_empty": int((in_forest == NODATA).sum()),
        "min": round(float(good.min()), 3) if good.size else np.nan,
        "max": round(float(good.max()), 3) if good.size else np.nan,
        "mean": round(float(good.mean()), 3) if good.size else np.nan,
    })


# ---------- GEE predictors ----------
for folder, dtype in PREDICTORS.items():
    for year in YEARS:
        raw = read_file(EXPORTS / folder / f"{folder[3:]}_{year}.tif")
        clean = np.where(inside, raw, NODATA)

        write_file(folder, f"{folder}_{year}.tif", clean, dtype)
        log_check(folder, year, clean)

# ---------- Fire history from the burned maps ----------
# Burned map for each season. 2001 comes from GEE, 2002 onwards from Phase 3.
burned = {2001: read_file(EXPORTS / "fire_history" / "burned_2001.tif")}

for year in YEARS[:-1]:  # 2024 is never needed as history
    burned[year] = read_file(BURNED / f"burned_{year}.tif")

for year in YEARS:
    count = np.zeros((height, width), dtype="int16")
    since = np.full((height, width), HISTORY_CAP, dtype="int16")

    # Previous seasons, most distant first, so a nearer burn overwrites it
    for k in range(HISTORY_CAP, 0, -1):
        past = burned.get(year - k)

        if past is None:
            continue  # before 2001 there is no record

        hit = past == 1
        since = np.where(hit, k, since)

        if k <= 3:
            count = count + hit.astype("int16")

    count = np.where(inside, count, NODATA)
    since = np.where(inside, since, NODATA)

    write_file("burn_count_3yr", f"burn_count_3yr_{year}.tif", count, "int16")
    write_file("years_since_burn", f"years_since_burn_{year}.tif", since, "int16")
    log_check("burn_count_3yr", year, count)
    log_check("years_since_burn", year, since)

# ---------- Save and summarise ----------
checks = pd.DataFrame(check_rows)
checks.to_csv(OUT / "predictor_check.csv", index=False)

problems = checks[checks["cells_empty"] > 0]

print("Files checked:", len(checks))
print("Files with empty forest cells:", len(problems))

if len(problems):
    print(problems[["predictor", "year", "cells_empty"]].to_string(index=False))

print(checks.groupby("predictor")[["min", "max", "mean"]].agg(
    {"min": "min", "max": "max", "mean": "mean"}).round(2).to_string())