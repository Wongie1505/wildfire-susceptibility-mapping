import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Folder layout (relative to this script):
#   Phase_2_fire_events/
#       phase2_fire_events.py
#       data/raw/       fire archive (6 files)
#       data/boundary/  Chikangawa_forest (6 files)
#       outputs/        written by this script

HERE = os.path.dirname(os.path.abspath(__file__))
FIRE_FILE = os.path.join(HERE, 'data', 'raw', 'fire_archive_M-C61_718732.shp')
BOUNDARY_FILE = os.path.join(HERE, 'data', 'boundary', 'Chikangawa_forest.shp')
OUT_DIR = os.path.join(HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_EPSG = 32736
MONTHS = [7, 8, 9, 10, 11, 12]
MIN_CONFIDENCE = 75
EVENT_DISTANCE_M = 1000
EVENT_DAYS = 4
START_YEAR = 2002
END_YEAR = 2024
TRAIN_END_YEAR = 2015

counts = []


def record(step, n):
    counts.append({'step': step, 'count': int(n)})


# ---------- read and prepare ----------
fires = gpd.read_file(FIRE_FILE)
record('raw records', len(fires))

fires['acq_date'] = pd.to_datetime(fires['ACQ_DATE'])
acq_time = fires['ACQ_TIME'].astype(str).str.zfill(4)
minutes = acq_time.str[:2].astype(int) * 60 + acq_time.str[2:].astype(int)
fires['acq_datetime'] = fires['acq_date'] + pd.to_timedelta(minutes, unit='m')
fires['conf'] = pd.to_numeric(fires['CONFIDENCE'], errors='coerce')

# ---------- filters ----------
fires = fires[fires['acq_date'].dt.month.isin(MONTHS)].copy()
record('July to December', len(fires))
fires = fires[fires['acq_date'].dt.year.between(START_YEAR, END_YEAR)].copy()
record(f'years {START_YEAR} to {END_YEAR}', len(fires))

fires = fires[fires['conf'] >= MIN_CONFIDENCE].copy()
record(f'confidence {MIN_CONFIDENCE} or above', len(fires))

boundary = gpd.read_file(BOUNDARY_FILE).to_crs(TARGET_EPSG)
boundary_geom = boundary.geometry.union_all()
fires_utm = fires.to_crs(TARGET_EPSG)
inside = fires_utm.geometry.within(boundary_geom)
fires = fires[inside.values].copy()
fires_utm = fires_utm[inside.values].copy()
record('inside boundary', len(fires))

# ---------- group into events ----------
xy = np.column_stack([fires_utm.geometry.x.values, fires_utm.geometry.y.values])
times = fires['acq_datetime'].values
n = len(fires)

tree = cKDTree(xy)
pairs = tree.query_pairs(r=EVENT_DISTANCE_M, output_type='ndarray')
if pairs.size:
    i, j = pairs[:, 0], pairs[:, 1]
    gap_days = np.abs((times[i] - times[j]).astype('timedelta64[s]').astype(float)) / 86400
    keep = gap_days <= EVENT_DAYS
    i, j = i[keep], j[keep]
else:
    i = j = np.array([], dtype=int)

graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
n_events, labels = connected_components(graph, directed=False)
fires['event_id'] = labels + 1
record(f'events ({EVENT_DISTANCE_M} m and {EVENT_DAYS} days, chained)', n_events)

# ---------- keep one detection per event ----------
representative = (
    fires.sort_values(['event_id', 'conf', 'acq_datetime'], ascending=[True, False, True])
    .groupby('event_id', as_index=False)
    .head(1)
)
summary = fires.groupby('event_id').agg(
    date_first=('acq_datetime', 'min'),
    date_last=('acq_datetime', 'max'),
    n_detections=('conf', 'size'),
    max_confidence=('conf', 'max'),
).reset_index()

events = representative[['event_id', 'LATITUDE', 'LONGITUDE', 'conf', 'acq_datetime']].merge(
    summary, on='event_id'
)
events = events.rename(columns={
    'LATITUDE': 'rep_latitude', 'LONGITUDE': 'rep_longitude',
    'conf': 'rep_confidence', 'acq_datetime': 'rep_datetime',
})
events = events.sort_values('date_first').reset_index(drop=True)
record('one detection per event (events kept)', len(events))

# ---------- per-year table ----------
years = fires['acq_datetime'].dt.year
year_range = range(START_YEAR, END_YEAR + 1)
detections_by_year = years.value_counts()
events_by_year = events['date_first'].dt.year.value_counts()

per_year = pd.DataFrame({'year': list(year_range)})
per_year['detections'] = per_year['year'].map(detections_by_year).fillna(0).astype(int)
per_year['events'] = per_year['year'].map(events_by_year).fillna(0).astype(int)
per_year['period'] = np.where(per_year['year'] <= TRAIN_END_YEAR, 'train', 'test')

# ---------- checks ----------
assert len(events) == n_events, 'event count does not match the grouping'
assert events['event_id'].is_unique, 'duplicate event IDs'
assert events['n_detections'].sum() == len(fires), 'detections are not all accounted for'

# ---------- write outputs ----------
pd.DataFrame(counts).to_csv(os.path.join(OUT_DIR, 'fire_count_table.csv'), index=False)

detections_out = fires[['event_id', 'LATITUDE', 'LONGITUDE', 'acq_datetime', 'conf']].copy()
detections_out.to_csv(os.path.join(OUT_DIR, 'fire_detections_filtered.csv'), index=False)

events.to_csv(os.path.join(OUT_DIR, 'fire_events.csv'), index=False)
per_year.to_csv(os.path.join(OUT_DIR, 'fire_events_by_year.csv'), index=False)

print('\nPhase 2: fire events')
print('=' * 60)
for c in counts:
    print(f"{c['count']:>8,}  {c['step']}")
print('-' * 60)
print(f"Events: {len(events)}")
print('\nPer year (train up to %d, test after):' % TRAIN_END_YEAR)
print(per_year.to_string(index=False))
print('RESULT: complete. Files written to outputs/. Ready for Phase 3.')