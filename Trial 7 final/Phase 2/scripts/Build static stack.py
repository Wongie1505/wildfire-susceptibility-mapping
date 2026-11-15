"""Phase 2: build the static predictor stack on a 500 m grid.

# 
var region = ee.FeatureCollection("projects/rainfallpatterns/assets/Chikangawa_forest");

CRS and extent come from the DEM and boundary. The 500 m cell size matches
the burned-area labels used in Phase 3. Origin is snapped to multiples of
500 so every later GEE export can use the exact same grid.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.spatial import cKDTree
from shapely.strtree import STRtree

# Paths, relative to the Phase 2 folder
BASE = Path(__file__).resolve().parents[1]
DEM_PATH = BASE / "data" / "dem" / "Viphya_DEM.tif"
BOUNDARY = BASE / "data" / "boundary" / "Chikangawa_forest.shp"
RIVERS = BASE / "data" / "vectors" / "Rivers" / "rivers.shp"
ROADS = BASE / "data" / "vectors" / "Roads" / "roads.shp"
BUILT = BASE / "data" / "gee_exports" / "GHSL_built_2000.tif"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

CELL = 500
BUILT_MIN_M2 = 500  # a 100 m GHSL cell counts as settlement above this
NODATA = -9999

# ---------- Reference grid ----------
boundary = gpd.read_file(BOUNDARY)
crs = boundary.crs
minx, miny, maxx, maxy = boundary.total_bounds

x0 = np.floor(minx / CELL) * CELL
y1 = np.ceil(maxy / CELL) * CELL
width = int(np.ceil((maxx - x0) / CELL))
height = int(np.ceil((y1 - miny) / CELL))
transform = from_origin(x0, y1, CELL, CELL)

# Cell centres, used for the distance layers
xs = x0 + (np.arange(width) + 0.5) * CELL
ys = y1 - (np.arange(height) + 0.5) * CELL
grid_x, grid_y = np.meshgrid(xs, ys)
centres = shapely.points(grid_x.ravel(), grid_y.ravel())

# Cells whose centre falls inside the forest
inside = features.rasterize(
    [(geom, 1) for geom in boundary.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8",
).astype(bool)


# ---------- Helpers ----------
def to_grid(arr, src_transform, src_crs):
    """Average a fine raster onto the 500 m grid, ignoring NaN."""
    dst = np.full((height, width), np.nan, dtype="float32")
    reproject(
        source=arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return dst


def dist_to_vector(path):
    """Distance in metres from each cell centre to the nearest feature."""
    gdf = gpd.read_file(path).to_crs(crs)
    gdf = gdf[~gdf.geometry.is_empty]

    tree = STRtree(gdf.geometry.values)
    _, dist = tree.query_nearest(centres, return_distance=True, all_matches=False)

    return dist.reshape(height, width).astype("float32")


# ---------- DEM, slope, aspect (computed at native resolution) ----------
with rasterio.open(DEM_PATH) as src:
    dem = src.read(1).astype("float32")
    dem[dem == src.nodata] = np.nan
    res_x, res_y = src.res
    dem_transform, dem_crs = src.transform, src.crs

# Elevation change per metre along rows (south) and columns (east)
dz_row, dz_col = np.gradient(dem, res_y, res_x)
mag = np.hypot(dz_row, dz_col)

slope = np.degrees(np.arctan(mag))

# Downslope direction as unit-vector components; averaging these is safe,
# averaging raw aspect degrees is not. Flat cells have no direction.
with np.errstate(invalid="ignore", divide="ignore"):
    east = np.where(mag > 0, -dz_col / mag, np.nan)
    north = np.where(mag > 0, dz_row / mag, np.nan)

layers = {
    "dem": to_grid(dem, dem_transform, dem_crs),
    "slope": to_grid(slope.astype("float32"), dem_transform, dem_crs),
    "aspect_east": to_grid(east.astype("float32"), dem_transform, dem_crs),
    "aspect_north": to_grid(north.astype("float32"), dem_transform, dem_crs),
}

# ---------- Distance layers ----------
layers["rivers_dist"] = dist_to_vector(RIVERS)
layers["roads_dist"] = dist_to_vector(ROADS)

with rasterio.open(BUILT) as src:
    built = src.read(1)
    rows, cols = np.where(built >= BUILT_MIN_M2)
    bx, by = rasterio.transform.xy(src.transform, rows, cols, offset="center")

d, _ = cKDTree(np.column_stack([bx, by])).query(
    np.column_stack([grid_x.ravel(), grid_y.ravel()])
)
layers["settlement_dist"] = d.reshape(height, width).astype("float32")

# ---------- Mask to the forest and write ----------
names = list(layers)
stack = np.stack([layers[n] for n in names])
stack[:, ~inside] = np.nan

# Cell counts must match across layers, or the stack is misaligned
valid_counts = {n: int(np.isfinite(stack[i]).sum()) for i, n in enumerate(names)}
print("Cells inside forest:", int(inside.sum()))
print("Valid cells per layer:", valid_counts)

with rasterio.open(
    OUT / "static_stack.tif",
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=len(names),
    dtype="float32",
    crs=crs,
    transform=transform,
    nodata=NODATA,
) as dst:
    dst.write(np.nan_to_num(stack, nan=NODATA))
    for i, n in enumerate(names, start=1):
        dst.set_band_description(i, n)

# ---------- Data dictionary ----------
dictionary = pd.DataFrame(
    [
        ("dem", "Viphya_DEM.tif", "m", "Mean of 30 m cells"),
        ("slope", "Derived from DEM", "degrees", "Computed at 30 m, then mean"),
        ("aspect_east", "Derived from DEM", "-1 to 1", "Sine of aspect at 30 m, then mean"),
        ("aspect_north", "Derived from DEM", "-1 to 1", "Cosine of aspect at 30 m, then mean"),
        ("rivers_dist", "rivers.shp", "m", "Cell centre to nearest feature"),
        ("roads_dist", "roads.shp", "m", "Cell centre to nearest feature"),
        ("settlement_dist", "GHSL built-up 2000", "m", f"Cell centre to nearest 100 m cell >= {BUILT_MIN_M2} m2 built"),
    ],
    columns=["layer", "source", "unit", "method"],
)
dictionary["nodata"] = NODATA
dictionary["crs"] = str(crs)
dictionary["cell_size_m"] = CELL
dictionary.to_csv(OUT / "data_dictionary.csv", index=False)