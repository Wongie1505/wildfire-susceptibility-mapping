import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely import contains_xy

FIRE_FILE = r'C:\Users\wongi\OneDrive\Desktop\Year 4\Project Thesis\My Research\Project and thesis\Trial-6 after review\phase 9 overlay\data\raw\fire_archive_M-C61_847303.shp'
ROI_FILE = r'C:\Users\wongi\OneDrive\Desktop\Year 4\Project Thesis\My Research\Project and thesis\Trial-6 after review\phase 9 overlay\data\boundary\Chikangawa_forest.shp'
OUT_FILE = r'C:\Users\wongi\OneDrive\Desktop\Year 4\Project Thesis\My Research\Project and thesis\Trial-6 after review\phase 9 overlay\data\non-fire\nonfire_points_2025.shp'

BUFFER_M = 500
SEED = 42
EPSG = 32736

fires = gpd.read_file(FIRE_FILE).to_crs(EPSG)
roi = gpd.read_file(ROI_FILE).to_crs(EPSG)
roi_geom = roi.geometry.union_all()
minx, miny, maxx, maxy = roi.total_bounds

n_target = len(fires)
tree = cKDTree(np.column_stack([fires.geometry.x, fires.geometry.y]))
rng = np.random.default_rng(SEED)

kept = []
total = 0
while total < n_target:
    cand = rng.uniform([minx, miny], [maxx, maxy], size=(100000, 2))
    cand = cand[contains_xy(roi_geom, cand[:, 0], cand[:, 1])]
    dist, _ = tree.query(cand, k=1)
    cand = cand[dist > BUFFER_M]
    kept.append(cand)
    total += len(cand)

points = np.vstack(kept)[:n_target]

out = gpd.GeoDataFrame(
    {'id': range(1, n_target + 1), 'label': 0},
    geometry=gpd.points_from_xy(points[:, 0], points[:, 1]),
    crs=EPSG,
)
out.to_file(OUT_FILE)

print(f'Fire detections: {n_target}')
print(f'Non-fire points written: {len(out)}')
print(f'Saved to: {OUT_FILE}')