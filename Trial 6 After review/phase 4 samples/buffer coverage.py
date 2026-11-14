import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely import contains_xy

# Measures how much of the boundary remains available as background for each buffer size.
# It does not use the model or any accuracy score.
# Run from the Phase_4_samples folder. Output: outputs/buffer_coverage.csv

HERE = os.path.dirname(os.path.abspath(__file__))
FIRE_DIR = os.path.join(HERE, 'data', 'fire')
BOUNDARY_FILE = os.path.join(HERE, 'data', 'boundary', 'Chikangawa_forest.shp')
OUT_DIR = os.path.join(HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_EPSG = 32736
SEED = 42
N_POINTS = 50000          # standard error about 0.2 percentage points
RADII = [0, 250, 500, 750, 1000]
TRAIN_END_YEAR = 2015

boundary = gpd.read_file(BOUNDARY_FILE).to_crs(TARGET_EPSG)
geom = boundary.geometry.union_all()
minx, miny, maxx, maxy = boundary.total_bounds
area_km2 = geom.area / 1e6

detections = pd.read_csv(os.path.join(FIRE_DIR, 'fire_detections_filtered.csv'),
                         parse_dates=['acq_datetime'])
pts = gpd.GeoSeries(gpd.points_from_xy(detections['LONGITUDE'], detections['LATITUDE']),
                    crs=4326).to_crs(TARGET_EPSG)
detections['x'] = pts.x.values
detections['y'] = pts.y.values

# One batch of random points, kept only inside the boundary.
rng = np.random.default_rng(SEED)
draw = rng.uniform([minx, miny], [maxx, maxy], size=(N_POINTS * 4, 2))
draw = draw[contains_xy(geom, draw[:, 0], draw[:, 1])]
while len(draw) < N_POINTS:
    extra = rng.uniform([minx, miny], [maxx, maxy], size=(N_POINTS, 2))
    draw = np.vstack([draw, extra[contains_xy(geom, extra[:, 0], extra[:, 1])]])
candidates = draw[:N_POINTS]

rows = []
periods = {
    'train': detections['acq_datetime'].dt.year <= TRAIN_END_YEAR,
    'test': detections['acq_datetime'].dt.year > TRAIN_END_YEAR,
}
for period, mask in periods.items():
    det = detections.loc[mask, ['x', 'y']].to_numpy()
    dist, _ = cKDTree(det).query(candidates, k=1)
    for r in RADII:
        covered = float(np.mean(dist <= r))
        rows.append({
            'period': period,
            'buffer_m': r,
            'boundary_covered_pct': round(100 * covered, 2),
            'background_available_pct': round(100 * (1 - covered), 2),
            'background_area_km2': round(area_km2 * (1 - covered), 2),
        })

report = pd.DataFrame(rows)
report.to_csv(os.path.join(OUT_DIR, 'buffer_coverage.csv'), index=False)

print('\nBuffer coverage (share of boundary within reach of a detection)')
print('=' * 70)
print(report.to_string(index=False))
print('-' * 70)
print(f'Boundary area: {area_km2:,.1f} km2')
print('RESULT: written to outputs/buffer_coverage.csv')