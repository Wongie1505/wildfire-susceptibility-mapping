"""Script 3: mean pre-season rainfall and dry-season NDVI median (30 m).

Inputs, exported by scripts/gee/phase11_rain_ndvi_export.js:
  data/rain_mean_nov_apr_2002_2024.tif   mean Nov to Apr CHIRPS total, mm
  data/ndvi_dry_median_2002_2024.tif     median dry-season NDVI, 1 May to 31 Oct

Rainfall (about 5.5 km) is resampled bilinearly so it reads as a smooth
surface. NDVI (500 m) is resampled with nearest neighbour so each 500 m
cell stays a sharp block rather than a blurred edge.

Writes to outputs/:
  rain_mean_30m.tif, rain_mean_map.png
  ndvi_median_30m.tif, ndvi_median_map.png
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from rasterio.warp import Resampling

from common import (
    DATA, OUT, extent, forest_mask, ref_grid, resample_to_ref,
    save_float_tif, style_map,
)


def draw_map(data, cmap, label, title, out_name):
    """Draw one continuous map in the Trial 7 layout."""
    transform, _, shape = ref_grid()
    image_extent = extent(transform, shape)
    forest = forest_mask()
    shown = np.where(forest & np.isfinite(data), data, np.nan)

    fig, ax = plt.subplots(figsize=(5.5, 9))
    image = ax.imshow(shown, cmap=cmap, extent=image_extent, interpolation="nearest")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Easting (m), UTM 36S")
    ax.set_ylabel("Northing (m)")
    style_map(ax, title)

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label(label)
    bar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- Colour ramps ----------
# Rainfall: sequential, so wetter reads as darker blue
RAIN_CMAP = LinearSegmentedColormap.from_list(
    "rain", ["#fff7bc", "#7fcdbb", "#225ea8"])

# NDVI: red (bare or stressed) to green (dense vegetation)
NDVI_CMAP = plt.get_cmap("RdYlGn")

# ---------- Rainfall mean ----------
rain = resample_to_ref(
    DATA / "rain_mean_nov_apr_2002_2024.tif",
    method=Resampling.bilinear,
)
save_float_tif(rain, OUT / "rain_mean_30m.tif")
draw_map(
    rain, RAIN_CMAP, "Mean rainfall, Nov to Apr (mm)",
    "Viphya Plantation - Mean Pre-season Rainfall\nCHIRPS, Nov to Apr, 2002 to 2024",
    "rain_mean_map.png",
)

# ---------- NDVI median ----------
ndvi = resample_to_ref(
    DATA / "ndvi_dry_median_2002_2024.tif",
    method=Resampling.nearest,
)
save_float_tif(ndvi, OUT / "ndvi_median_30m.tif")
draw_map(
    ndvi, NDVI_CMAP, "Median NDVI",
    "Viphya Plantation - Dry-season NDVI Median\nMODIS, 1 May to 31 Oct, 2002 to 2024",
    "ndvi_median_map.png",
)

print("Saved rainfall mean and NDVI median maps")