import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from scipy.spatial import cKDTree
from shapely import contains_xy

# Folder layout (relative to this script):
#   Phase_4_samples/
#       phase4_sampling.py
#       buffer_coverage.py
#       data/fire/       fire_events.csv, fire_detections_filtered.csv   (Phase 2 outputs)
#       data/boundary/   Chikangawa_forest.* (Phase 2 data/boundary)
#       data/period/     8 period layers (Phase 3b data/aligned)
#       data/fixed/      8 fixed layers (Phase 1b data/aligned)
#       outputs/         training_table.csv, sample_counts.csv, sampling_settings.csv

HERE = os.path.dirname(os.path.abspath(__file__))
FIRE_DIR = os.path.join(HERE, 'data', 'fire')
BOUNDARY_FILE = os.path.join(HERE, 'data', 'boundary', 'Chikangawa_forest.shp')
PERIOD_DIR = os.path.join(HERE, 'data', 'period')
FIXED_DIR = os.path.join(HERE, 'data', 'fixed')
OUT_DIR = os.path.join(HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_EPSG = 32736
SEED = 42
BUFFER_M = 500              # chosen from buffer_coverage.csv, see the decisions log
TRAIN_END_YEAR = 2015
BLOCK_M = 5000              # spatial block size for cross-validation
BATCH = 200000
MAX_ROUNDS = 500

PERIOD_BASES = {
    'ndvi': 'viphya_ndvi',
    'ndmi': 'viphya_ndmi',
    'lst': 'viphya_lst',
    'rainfall': 'viphya_rainfall',
}
PERIOD_SUFFIX = {'train': '2002_2015', 'test': '2016_2024'}
FIXED_FILES = {
    'rivers_dist': 'Rivers_Distances.tif',
    'roads_dist': 'Roads_distances.tif',
    'settlement_dist': 'Settlement_Distance.tif',
    'aspect_east': 'Viphya_Aspect_eastness.tif',
    'aspect_north': 'Viphya_Aspect_northness.tif',
    'dem': 'Viphya_DEM.tif',
    'lulc': 'Viphya_LULC.tif',
    'slope': 'Viphya_Slope.tif',
}
PREDICTORS = list(PERIOD_BASES) + list(FIXED_FILES)

rng = np.random.default_rng(SEED)


def load_layer(path):
    """Read one band into memory, with missing values set to NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
    missing = ~np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        missing |= arr == np.float32(nodata)
    arr[missing] = np.nan
    return arr, transform


# Load the fixed layers once; they are shared by both periods.
fixed = {}
ref_transform = None
ref_shape = None
for name, fname in FIXED_FILES.items():
    arr, transform = load_layer(os.path.join(FIXED_DIR, fname))
    if ref_transform is None:
        ref_transform, ref_shape = transform, arr.shape
    assert transform == ref_transform and arr.shape == ref_shape, f'{fname} is not on the DEM grid'
    fixed[name] = arr


def load_period_layers(period):
    layers = dict(fixed)
    for name, base in PERIOD_BASES.items():
        arr, transform = load_layer(os.path.join(PERIOD_DIR, f'{base}_{PERIOD_SUFFIX[period]}.tif'))
        assert transform == ref_transform and arr.shape == ref_shape, f'{base} is not on the DEM grid'
        layers[name] = arr
    return layers


def sample(xs, ys, layers):
    """Return the 12 predictor values at each point, and whether all are valid."""
    rows, cols = rowcol(ref_transform, xs, ys)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    inside = (rows >= 0) & (rows < ref_shape[0]) & (cols >= 0) & (cols < ref_shape[1])
    r = np.clip(rows, 0, ref_shape[0] - 1)
    c = np.clip(cols, 0, ref_shape[1] - 1)
    values = np.column_stack([layers[name][r, c] for name in PREDICTORS]).astype(float)
    values[~inside] = np.nan
    valid = np.isfinite(values).all(axis=1)
    return values, valid


# ---------- points and periods ----------
events = pd.read_csv(os.path.join(FIRE_DIR, 'fire_events.csv'), parse_dates=['date_first'])
detections = pd.read_csv(os.path.join(FIRE_DIR, 'fire_detections_filtered.csv'),
                         parse_dates=['acq_datetime'])
boundary = gpd.read_file(BOUNDARY_FILE).to_crs(TARGET_EPSG)
boundary_geom = boundary.geometry.union_all()
minx, miny, maxx, maxy = boundary.total_bounds

events['period'] = np.where(events['date_first'].dt.year <= TRAIN_END_YEAR, 'train', 'test')
detections['period'] = np.where(detections['acq_datetime'].dt.year <= TRAIN_END_YEAR, 'train', 'test')

ev_xy = gpd.GeoSeries(gpd.points_from_xy(events['rep_longitude'], events['rep_latitude']),
                      crs=4326).to_crs(TARGET_EPSG)
events['x'] = ev_xy.x.values
events['y'] = ev_xy.y.values

det_xy = gpd.GeoSeries(gpd.points_from_xy(detections['LONGITUDE'], detections['LATITUDE']),
                       crs=4326).to_crs(TARGET_EPSG)
detections['x'] = det_xy.x.values
detections['y'] = det_xy.y.values

rows = []
counts = []


def record(period, step, n):
    counts.append({'period': period, 'step': step, 'count': int(n)})


for period in ('train', 'test'):
    layers = load_period_layers(period)
    ev = events[events['period'] == period].copy()
    det = detections[detections['period'] == period]
    record(period, 'events in period', len(ev))

    # Fire points: keep events whose location has all predictors valid.
    fire_values, fire_valid = sample(ev['x'].values, ev['y'].values, layers)
    record(period, 'events with all predictors valid', int(fire_valid.sum()))
    ev = ev[fire_valid]
    fire_values = fire_values[fire_valid]
    n_target = len(ev)

    # Buffer uses only detections from this period.
    tree = cKDTree(det[['x', 'y']].to_numpy())

    accepted_xy, accepted_values = [], []
    drawn = outside = in_buffer = missing = 0

    for _ in range(MAX_ROUNDS):
        if len(accepted_xy) >= n_target:
            break
        cand = rng.uniform([minx, miny], [maxx, maxy], size=(BATCH, 2))
        drawn += BATCH

        inside = contains_xy(boundary_geom, cand[:, 0], cand[:, 1])
        outside += int((~inside).sum())
        cand = cand[inside]

        dist, _ = tree.query(cand, k=1)
        far = dist > BUFFER_M
        in_buffer += int((~far).sum())
        cand = cand[far]

        values, valid = sample(cand[:, 0], cand[:, 1], layers)
        missing += int((~valid).sum())
        cand = cand[valid]
        values = values[valid]

        need = n_target - len(accepted_xy)
        accepted_xy.extend(cand[:need])
        accepted_values.extend(values[:need])

    if len(accepted_xy) < n_target:
        raise SystemExit(f'Could not draw {n_target} non-fire points for {period} '
                         f'after {MAX_ROUNDS} rounds.')

    record(period, 'non-fire candidates drawn', drawn)
    record(period, 'non-fire rejected: outside boundary', outside)
    record(period, f'non-fire rejected: within {BUFFER_M} m of a detection', in_buffer)
    record(period, 'non-fire rejected: predictor missing', missing)
    record(period, 'non-fire points kept', len(accepted_xy))

    for v, (_, e) in zip(fire_values, ev.iterrows()):
        rows.append({'period': period, 'label': 1, 'event_id': int(e['event_id']),
                     'date': e['date_first'].date(), 'x': e['x'], 'y': e['y'],
                     **dict(zip(PREDICTORS, v))})
    for (x, y), v in zip(accepted_xy, accepted_values):
        rows.append({'period': period, 'label': 0, 'event_id': None,
                     'date': None, 'x': x, 'y': y,
                     **dict(zip(PREDICTORS, v))})

table = pd.DataFrame(rows)
table['block_id'] = (np.floor(table['x'] / BLOCK_M).astype(int).astype(str) + '_' +
                     np.floor(table['y'] / BLOCK_M).astype(int).astype(str))
table.insert(0, 'point_id', range(1, len(table) + 1))

for period in ('train', 'test'):
    sub = table[table['period'] == period]
    assert (sub['label'] == 1).sum() == (sub['label'] == 0).sum(), f'{period} is not balanced'

settings = pd.DataFrame([
    {'setting': 'random seed', 'value': SEED},
    {'setting': 'non-fire buffer (m)', 'value': BUFFER_M},
    {'setting': 'training period', 'value': f'2002 to {TRAIN_END_YEAR}'},
    {'setting': 'test period', 'value': f'{TRAIN_END_YEAR + 1} to 2024'},
    {'setting': 'non-fire to fire ratio', 'value': '1:1 per period'},
    {'setting': 'spatial block size (m)', 'value': BLOCK_M},
    {'setting': 'buffer detections', 'value': 'same period only'},
    {'setting': 'fire points', 'value': 'representative event location'},
])

table.to_csv(os.path.join(OUT_DIR, 'training_table_500m.csv'), index=False)
pd.DataFrame(counts).to_csv(os.path.join(OUT_DIR, 'sample_counts.csv'), index=False)
settings.to_csv(os.path.join(OUT_DIR, 'sampling_settings.csv'), index=False)

print('\nPhase 4: sampling (buffer %d m)' % BUFFER_M)
print('=' * 60)
print(pd.DataFrame(counts).to_string(index=False))
print('-' * 60)
print(table.groupby(['period', 'label']).size().to_string())
print('RESULT: complete. Outputs written to outputs/.')