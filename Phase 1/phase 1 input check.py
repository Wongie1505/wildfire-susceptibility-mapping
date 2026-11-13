import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box

# Folder layout (relative to this script, so the folder can be copied or shared):
#   Phase_1_inputs/
#       phase1_input_check.py   (this file)
#       data/raw/               fire archive and boundary
#       data/predictors/        the 12 predictor rasters
#       outputs/                written by this script

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, 'data', 'raw')
PRED_DIR = os.path.join(HERE, 'data', 'predictors')
OUT_DIR = os.path.join(HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


FIRE_FILE = os.path.join(RAW_DIR, 'Raw_Fire/fire_archive_M-C61_718732.shp')
BOUNDARY_FILE = os.path.join(RAW_DIR, 'Chikangawa_forest/Chikangawa_forest.shp')

PREDICTORS = [
    'Rivers_Distances.tif',
    'Roads_distances.tif',
    'Settlement_Distance.tif',
    'Viphya_Aspect_eastness.tif',
    'Viphya_Aspect_northness.tif',
    'Viphya_DEM.tif',
    'viphya_lst.tif',
    'Viphya_LULC.tif',
    'viphya_ndmi.tif',
    'viphya_ndvi.tif',
    'viphya_rainfall.tif',
    'Viphya_Slope.tif',
]

REFERENCE_GRID = 'Viphya_DEM.tif'   # every other predictor must match this grid
TARGET_EPSG = 32736                 # WGS 84 / UTM 36S, metres
EXPECTED_PIXEL = 30.634             # pixel size of the existing rasters, metres
PIXEL_TOLERANCE = 0.01
FIRE_COLUMNS = ['ACQ_DATE', 'CONFIDENCE', 'INSTRUMENT', 'LATITUDE', 'LONGITUDE']

rows = []


def add_row(**kwargs):
    rows.append(kwargs)


def union_geometry(gdf):
    return gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union


# ---------- boundary ----------
boundary_issues = []
if not os.path.exists(BOUNDARY_FILE):
    boundary_issues.append('boundary file not found')
    boundary = None
else:
    boundary_gdf = gpd.read_file(BOUNDARY_FILE)
    boundary_crs = boundary_gdf.crs
    boundary_metric = boundary_gdf.to_crs(TARGET_EPSG)
    boundary_geom = union_geometry(boundary_metric)
    area_m2 = boundary_geom.area
    if boundary_crs is None:
        boundary_issues.append('boundary has no CRS')
    elif boundary_crs.to_epsg() != TARGET_EPSG:
        boundary_issues.append(f'boundary CRS is {boundary_crs.to_string()}, not EPSG:{TARGET_EPSG} (reprojected for area)')
    add_row(
        file='Chikangawa_forest.shp', type='boundary', status='PASS' if not boundary_issues else 'WARN',
        crs=str(boundary_crs), features=len(boundary_gdf),
        area_ha=round(area_m2 / 10000, 3), area_km2=round(area_m2 / 1e6, 3),
        issues='; '.join(boundary_issues),
    )
    boundary = boundary_metric

# ---------- fire archive ----------
fire_issues = []
if not os.path.exists(FIRE_FILE):
    fire_issues.append('fire archive not found')
    fires = None
else:
    fires = gpd.read_file(FIRE_FILE)
    missing = [c for c in FIRE_COLUMNS if c not in fires.columns]
    if missing:
        fire_issues.append(f'missing columns: {missing}')
    fires['conf_num'] = pd.to_numeric(fires['CONFIDENCE'], errors='coerce') if 'CONFIDENCE' in fires.columns else np.nan
    non_numeric = int(fires['conf_num'].isna().sum())
    fires_m = fires.to_crs(TARGET_EPSG)
    if boundary is not None:
        inside = int(fires_m.geometry.within(boundary.geometry.union_all() if hasattr(boundary.geometry, 'union_all') else boundary.geometry.unary_union).sum())
    else:
        inside = -1
    dates = pd.to_datetime(fires['ACQ_DATE'], errors='coerce') if 'ACQ_DATE' in fires.columns else None
    date_range = f'{dates.min().date()} to {dates.max().date()}' if dates is not None and dates.notna().any() else ''
    instruments = ', '.join(sorted(fires['INSTRUMENT'].astype(str).unique())) if 'INSTRUMENT' in fires.columns else ''
    add_row(
        file='fire_archive_M-C61_718732.shp', type='fire',
        status='FAIL' if fire_issues else 'PASS',
        crs=str(fires.crs), features=len(fires),
        area_ha=None, area_km2=None,
        issues='; '.join(fire_issues) if fire_issues else '',
        note=f'date range {date_range}; instruments {instruments}; non-numeric confidence {non_numeric}; inside boundary {inside}',
    )

# ---------- predictor rasters ----------
ref_transform = None
ref_shape = None
ref_path = os.path.join(PRED_DIR, REFERENCE_GRID)
if os.path.exists(ref_path):
    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)

for name in PREDICTORS:
    path = os.path.join(PRED_DIR, name)
    issues = []
    if not os.path.exists(path):
        add_row(file=name, type='predictor', status='FAIL', issues='file not found')
        continue

    with rasterio.open(path) as src:
        crs = src.crs
        epsg = crs.to_epsg() if crs else None
        res_x, res_y = src.res
        nodata = src.nodata
        data = src.read(1, masked=False)
        nodata_count = int(np.sum(data == nodata)) if nodata is not None else 0
        nan_count = int(np.isnan(data).sum()) if np.issubdtype(data.dtype, np.floating) else 0
        bounds = src.bounds
        width, height = src.width, src.height
        same_grid = (ref_transform is not None and src.transform == ref_transform and (src.height, src.width) == ref_shape)
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        overlaps = bool(boundary is not None and bbox.intersects(union_geometry(boundary)))

    if epsg != TARGET_EPSG:
        issues.append(f'CRS is EPSG:{epsg}, expected EPSG:{TARGET_EPSG}')
    if abs(res_x - EXPECTED_PIXEL) > PIXEL_TOLERANCE or abs(res_y - EXPECTED_PIXEL) > PIXEL_TOLERANCE:
        issues.append(f'pixel size {res_x:.4f} x {res_y:.4f}, expected about {EXPECTED_PIXEL}')
    if not same_grid:
        issues.append('grid (extent, origin or size) differs from reference DEM')
    if nodata is None:
        issues.append('no nodata value set')
    if not overlaps:
        issues.append('raster does not overlap the boundary')

    hard_fail = any(s for s in issues if 'CRS' in s or 'grid' in s or 'overlap' in s)
    status = 'FAIL' if hard_fail else ('WARN' if issues else 'PASS')

    add_row(
        file=name, type='predictor', status=status, crs=f'EPSG:{epsg}',
        width=width, height=height,
        res_x=round(res_x, 4), res_y=round(res_y, 4),
        left=round(bounds.left, 2), bottom=round(bounds.bottom, 2),
        right=round(bounds.right, 2), top=round(bounds.top, 2),
        nodata=nodata, dtype=str(data.dtype),
        nodata_pixels=nodata_count + nan_count,
        nodata_share_pct=round(100 * (nodata_count + nan_count) / data.size, 2),
        same_grid_as_reference=same_grid, overlaps_boundary=overlaps,
        issues='; '.join(issues),
    )

# ---------- write and summarise ----------
report = pd.DataFrame(rows)
report.to_csv(os.path.join(OUT_DIR, 'inputs_check.csv'), index=False)

print('\nPhase 1 input check')
print('=' * 60)
for _, r in report.iterrows():
    line = f"[{r['status']}] {r['file']}"
    if isinstance(r.get('issues'), str) and r['issues']:
        line += f"  -> {r['issues']}"
    print(line)

predictors = report[report['type'] == 'predictor']
n_ok = int((predictors['status'] == 'PASS').sum())
n_fail = int((report['status'] == 'FAIL').sum())
print('-' * 60)
print(f'Predictors passing: {n_ok} of {len(PREDICTORS)}')
print(f'Files failing: {n_fail}')
if n_fail == 0 and n_ok == len(PREDICTORS):
    print('RESULT: PASS. Phase 1 complete. Continue to Phase 2.')
else:
    print('RESULT: FIX THE FAILURES ABOVE BEFORE CONTINUING.')
print(f'Report saved to {os.path.join(OUT_DIR, "phase_1_inputs_check.csv")}')