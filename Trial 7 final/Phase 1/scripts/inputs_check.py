"""Phase 1: confirm every input opens and record its properties.

Run from Spyder. Paths are relative to the Phase 1 folder, so it works
wherever the project sits on disk.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

# Phase 1 folder = parent of the scripts folder
BASE = Path(__file__).resolve().parents[1]
PREDICTORS = BASE / "data" / "predictors"
BOUNDARY = BASE / "data" / "raw" / "Chikangawa_forest" / "Chikangawa_forest.shp"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# The DEM is the reference grid every other raster must match
DEM_PATH = PREDICTORS / "Viphya_DEM.tif"

with rasterio.open(DEM_PATH) as dem:
    ref = {
        "crs": dem.crs,
        "res": dem.res,
        "bounds": dem.bounds,
        "shape": (dem.height, dem.width),
    }

rows = []

# Check every raster in the predictors folder
for tif in sorted(PREDICTORS.glob("*.tif")):
    with rasterio.open(tif) as src:
        rows.append({
            "file": tif.name,
            "type": "raster",
            "crs": str(src.crs),
            "res_x": src.res[0],
            "res_y": src.res[1],
            "xmin": src.bounds.left,
            "ymin": src.bounds.bottom,
            "xmax": src.bounds.right,
            "ymax": src.bounds.top,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            # True only if CRS, cell size, extent and shape all equal the DEM
            "matches_dem": (
                src.crs == ref["crs"]
                and src.res == ref["res"]
                and src.bounds == ref["bounds"]
                and (src.height, src.width) == ref["shape"]
            ),
        })

# Check the boundary shapefile
gdf = gpd.read_file(BOUNDARY)
xmin, ymin, xmax, ymax = gdf.total_bounds

rows.append({
    "file": BOUNDARY.name,
    "type": f"vector ({len(gdf)} features, {gdf.geom_type.iloc[0]})",
    "crs": str(gdf.crs),
    "xmin": xmin,
    "ymin": ymin,
    "xmax": xmax,
    "ymax": ymax,
    # For the boundary, only the CRS needs to match
    "matches_dem": gdf.crs == ref["crs"],
})

# Save the table
table = pd.DataFrame(rows)
table.to_csv(OUT / "inputs_check.csv", index=False)

# Save the reference grid description
with open(OUT / "reference_grid.txt", "w") as f:
    f.write("Reference grid: Viphya_DEM.tif\n")
    f.write(f"CRS: {ref['crs']}\n")
    f.write(f"Cell size: {ref['res'][0]} x {ref['res'][1]}\n")
    f.write(f"Extent: {tuple(ref['bounds'])}\n")
    f.write(f"Rows x cols: {ref['shape']}\n")

print(table[["file", "crs", "res_x", "nodata", "matches_dem"]].to_string(index=False))