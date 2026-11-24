"""Shared helpers for Phase 11: the 30 m reference grid and map styling."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import Resampling, reproject

# Paths, relative to the Phase 11 folder
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# Every map is built on the DEM grid (30.6 m, EPSG:32736)
REF = DATA / "Viphya_DEM.tif"
BOUNDARY = DATA / "Chikangawa_forest" / "Chikangawa_forest.shp"

# Colours used in the Trial 7 maps
CLASS_COLOURS = ["#00b050", "#e2e24a", "#f5333a"]
FOREST_FILL = "#dfe9d8"


def ref_grid():
    """Return the transform, CRS and shape of the 30 m reference grid."""
    with rasterio.open(REF) as src:
        return src.transform, src.crs, (src.height, src.width)


def extent(transform, shape):
    """Return matplotlib extent (left, right, bottom, top) in metres."""
    height, width = shape
    return (
        transform.c,
        transform.c + width * transform.a,
        transform.f + height * transform.e,
        transform.f,
    )


def resample_to_ref(path, method=Resampling.bilinear):
    """Read a raster and resample it onto the 30 m reference grid."""
    transform, crs, shape = ref_grid()
    dst = np.full(shape, np.nan, dtype="float32")

    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=method,
        )

    return dst


def forest_mask():
    """Boolean mask of the forest boundary on the reference grid."""
    transform, crs, shape = ref_grid()
    gdf = gpd.read_file(BOUNDARY).to_crs(crs)

    return features.rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def save_float_tif(array, path):
    """Write a float array on the reference grid as a GeoTIFF."""
    transform, crs, shape = ref_grid()

    with rasterio.open(
        path, "w", driver="GTiff", height=shape[0], width=shape[1],
        count=1, dtype="float32", crs=crs, transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(array.astype("float32"), 1)


def style_map(ax, title):
    """Apply the Trial 7 map style: no ticks, heavy frame, bold title."""
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
