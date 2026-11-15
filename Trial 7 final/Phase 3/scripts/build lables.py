"""Phase 3: check and clean the burned-area files from GEE.

Reads burned_YYYY.tif and burn_day_YYYY.tif (one pair per fire season),
checks they sit on the Phase 2 grid, blanks everything outside the forest,
and writes:
  outputs/burned/burned_YYYY.tif       cleaned copy of each burned map
  outputs/burn_day/burn_day_YYYY.tif   cleaned copy of each burn day map
  outputs/label_counts.csv             cell counts per season
  outputs/firms_agreement.csv          check against the old FIRMS fire events
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import maximum_filter

# Paths, relative to the Phase 3 folder
BASE = Path(__file__).resolve().parents[1]
STATIC = BASE / "data" / "static_stack" / "static_stack.tif"
EXPORTS = BASE / "data" / "gee_exports"
FIRMS = BASE / "data" / "fires_old" / "fire_events.csv"
OUT = BASE / "outputs"
(OUT / "burned").mkdir(parents=True, exist_ok=True)
(OUT / "burn_day").mkdir(parents=True, exist_ok=True)

YEARS = list(range(2002, 2025))
NODATA = -9999

# ---------- Phase 2 grid and forest mask ----------
with rasterio.open(STATIC) as src:
    transform = src.transform
    crs = src.crs
    height, width = src.height, src.width
    inside = src.read(1) != src.nodata  # DEM band: valid only inside the forest

profile = dict(
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    crs=crs,
    transform=transform,
    dtype="int16",
    nodata=NODATA,
)


def read_year(prefix, year):
    """Read one GEE file and stop if it is not on the Phase 2 grid."""
    path = EXPORTS / f"{prefix}_{year}.tif"

    with rasterio.open(path) as src:
        same_grid = np.allclose(tuple(src.transform)[:6], tuple(transform)[:6], atol=1e-3)

        if not same_grid or (src.height, src.width) != (height, width):
            raise ValueError(
                f"{path.name} is not on the Phase 2 grid.\n"
                f"  Phase 2: {tuple(transform)[:6]}, {height} x {width}\n"
                f"  This file: {tuple(src.transform)[:6]}, {src.height} x {src.width}\n"
                "Fix X0, Y1, NCOL, NROW in gee_burned_area.js and export again."
            )

        array = src.read(1)

    # Everything outside the forest becomes nodata
    return np.where(inside, array, NODATA).astype("int16")


def write_one(path, array):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


# ---------- Clean and save each season ----------
burned = {}
counts = []

for year in YEARS:
    b = read_year("burned", year)
    d = read_year("burn_day", year)
    burned[year] = b

    write_one(OUT / "burned" / f"burned_{year}.tif", b)
    write_one(OUT / "burn_day" / f"burn_day_{year}.tif", d)

    labelled = b != NODATA
    n_burned = int((b == 1).sum())

    counts.append({
        "year": year,
        "cells_in_forest": int(inside.sum()),
        "cells_with_label": int(labelled.sum()),
        "cells_no_label": int(inside.sum() - labelled.sum()),
        "burned": n_burned,
        "not_burned": int(labelled.sum()) - n_burned,
        "burned_pct": round(100 * n_burned / labelled.sum(), 2),
    })

counts = pd.DataFrame(counts)
counts.to_csv(OUT / "label_counts.csv", index=False)

# ---------- Check: do the old FIRMS events fall in burned cells? ----------
ev = pd.read_csv(FIRMS, parse_dates=["rep_datetime"])
ev["year"] = ev["rep_datetime"].dt.year

pts = gpd.GeoSeries.from_xy(ev["rep_longitude"], ev["rep_latitude"], crs=4326).to_crs(crs)
ev["col"] = np.floor((pts.x - transform.c) / transform.a).astype(int)
ev["row"] = np.floor((pts.y - transform.f) / transform.e).astype(int)

# Keep events that land on the grid
on_grid = ev["row"].between(0, height - 1) & ev["col"].between(0, width - 1)
ev = ev[on_grid]

agree = []

for year in YEARS:
    b = burned[year]
    hit_cell = b == 1
    hit_near = maximum_filter(hit_cell.astype("uint8"), size=3).astype(bool)

    e = ev[ev["year"] == year]
    e_valid = e[b[e["row"], e["col"]] != NODATA]

    agree.append({
        "year": year,
        "events_total": int((ev["year"] == year).sum()),
        "events_on_labelled_cells": len(e_valid),
        "in_burned_cell": int(hit_cell[e_valid["row"], e_valid["col"]].sum()),
        "within_1_cell": int(hit_near[e_valid["row"], e_valid["col"]].sum()),
        "burned_cell_share_pct": round(100 * hit_cell.sum() / (b != NODATA).sum(), 1),
    })

agree = pd.DataFrame(agree)

# Total row, then percentages
total = {
    "year": "TOTAL",
    "events_total": agree["events_total"].sum(),
    "events_on_labelled_cells": agree["events_on_labelled_cells"].sum(),
    "in_burned_cell": agree["in_burned_cell"].sum(),
    "within_1_cell": agree["within_1_cell"].sum(),
    "burned_cell_share_pct": round(counts["burned"].sum() / counts["cells_with_label"].sum() * 100, 1),
}
agree = pd.concat([agree, pd.DataFrame([total])], ignore_index=True)

agree["in_burned_cell_pct"] = (100 * agree["in_burned_cell"] / agree["events_on_labelled_cells"]).round(1)
agree["within_1_cell_pct"] = (100 * agree["within_1_cell"] / agree["events_on_labelled_cells"]).round(1)
agree.to_csv(OUT / "firms_agreement.csv", index=False)

# ---------- Summary ----------
print("Seasons written:", len(YEARS))
print("Burned share of all labelled cells:", agree.iloc[-1]["burned_cell_share_pct"], "%")
print("Cells with no label per season (min, max):",
      counts["cells_no_label"].min(), counts["cells_no_label"].max())
print(agree.tail(1).to_string(index=False))