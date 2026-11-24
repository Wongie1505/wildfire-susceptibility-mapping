"""Script 1: base map of roads, rivers and settlements (30 m).

Roads and rivers come from your local shapefiles. Settlements come from the
GHSL built-up surface (2000), the same layer used for the settlement distance
predictor. Every layer is put on the 30 m reference grid before plotting.

Writes to outputs/:
  settlements_30m.tif                  settlement layer on the 30 m grid
  roads_rivers_settlements_map.png     the base map
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (
    DATA, FOREST_FILL, OUT, extent, forest_mask, ref_grid,
    resample_to_ref, save_float_tif, style_map,
)

# ---------- Your local files (change these paths) ----------
ROADS = DATA / "roads.shp"
RIVERS = DATA / "rivers.shp"
GHSL = DATA / "GHSL_built_2000.tif"

# ---------- Settlements on the 30 m grid ----------
transform, crs, shape = ref_grid()
built = resample_to_ref(GHSL)

# Any built surface in the cell counts as a settlement
settle = np.where(np.nan_to_num(built) > 0, 1.0, np.nan).astype("float32")
save_float_tif(settle, OUT / "settlements_30m.tif")

# ---------- Vector layers in the reference CRS ----------
roads = gpd.read_file(ROADS).to_crs(crs)
rivers = gpd.read_file(RIVERS).to_crs(crs)

# ---------- Map ----------
forest = forest_mask()
image_extent = extent(transform, shape)

fig, ax = plt.subplots(figsize=(5.5, 9))

ax.imshow(
    np.where(forest, 1.0, np.nan),
    cmap=ListedColormap([FOREST_FILL]), extent=image_extent,
    interpolation="nearest",
)
ax.imshow(
    settle, cmap=ListedColormap(["#7f3f98"]), extent=image_extent,
    interpolation="nearest",
)

rivers.plot(ax=ax, color="#1f78b4", linewidth=0.8)
roads.plot(ax=ax, color="#333333", linewidth=0.6)

ax.set_xlim(image_extent[0], image_extent[1])
ax.set_ylim(image_extent[2], image_extent[3])
ax.ticklabel_format(style="plain", useOffset=False)
ax.tick_params(labelsize=7)
ax.set_xlabel("Easting (m), UTM 36S")
ax.set_ylabel("Northing (m)")
style_map(ax, "Viphya Plantation\nRoads, Rivers and Settlements")

handles = [
    Patch(color=FOREST_FILL, label="Forest"),
    Patch(color="#7f3f98", label="Settlements (GHSL 2000)"),
    Line2D([0], [0], color="#333333", lw=1.2, label="Roads"),
    Line2D([0], [0], color="#1f78b4", lw=1.2, label="Rivers"),
]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
          fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig(OUT / "roads_rivers_settlements_map.png", dpi=300, bbox_inches="tight")
print("Saved map and settlements_30m.tif")
