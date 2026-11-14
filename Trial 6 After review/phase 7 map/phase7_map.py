import os
import shutil
import numpy as np
import pandas as pd
import joblib
import rasterio
import geopandas as gpd
from rasterio.features import rasterize

# Folder layout (relative to this script):
#   Phase_7_map/
#       phase7_map.py
#       data/models/RandomForest.joblib            (Phase 5 outputs/models)
#       data/period/   4 period predictors, 2016 to 2024 (Phase 3b data/aligned)
#       data/fixed/    8 static predictors (Phase 1b data/aligned)
#       data/boundary/ Chikangawa_forest.* (study area, D16)
#       outputs/       rf_probability_2016_2024.tif, rf_class_2016_2024.tif,
#                      area_table.csv, prediction_summary.csv
#       outputs/predictors/  copies of the 4 period predictors

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, 'data', 'models', 'RandomForest.joblib')
PERIOD_DIR = os.path.join(HERE, 'data', 'period')
FIXED_DIR = os.path.join(HERE, 'data', 'fixed')
BOUNDARY_FILE = os.path.join(HERE, 'data', 'boundary', 'Chikangawa_forest.shp')
OUT_DIR = os.path.join(HERE, 'outputs')
PRED_OUT = os.path.join(OUT_DIR, 'predictors')
os.makedirs(PRED_OUT, exist_ok=True)

TARGET_EPSG = 32736
CHUNK = 500000
# Class limits from the dissertation: low below 0.33, medium 0.33 to below 0.66,
# high 0.66 and above. The 0.66 to 0.67 gap in the old legend is closed here.
LOW_MAX = 0.33
MEDIUM_MAX = 0.66

# The order must match the order used to train the model in Phase 5.
PREDICTORS = [
    ('ndvi', 'period', 'viphya_ndvi_2016_2024.tif'),
    ('ndmi', 'period', 'viphya_ndmi_2016_2024.tif'),
    ('lst', 'period', 'viphya_lst_2016_2024.tif'),
    ('rainfall', 'period', 'viphya_rainfall_2016_2024.tif'),
    ('rivers_dist', 'fixed', 'Rivers_Distances.tif'),
    ('roads_dist', 'fixed', 'Roads_distances.tif'),
    ('settlement_dist', 'fixed', 'Settlement_Distance.tif'),
    ('aspect_east', 'fixed', 'Viphya_Aspect_eastness.tif'),
    ('aspect_north', 'fixed', 'Viphya_Aspect_northness.tif'),
    ('dem', 'fixed', 'Viphya_DEM.tif'),
    ('lulc', 'fixed', 'Viphya_LULC.tif'),
    ('slope', 'fixed', 'Viphya_Slope.tif'),
]
FOLDERS = {'period': PERIOD_DIR, 'fixed': FIXED_DIR}


def load_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        profile_info = (src.transform, (src.height, src.width), src.crs)
    bad = ~np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        bad |= arr == nodata
    arr[bad] = np.nan
    return arr, profile_info


# The DEM defines the grid. Every predictor must match it.
dem_path = os.path.join(FIXED_DIR, 'Viphya_DEM.tif')
with rasterio.open(dem_path) as ref:
    ref_transform = ref.transform
    ref_shape = (ref.height, ref.width)
    ref_crs = ref.crs
    ref_profile = ref.profile.copy()

arrays = []
for name, folder, fname in PREDICTORS:
    arr, (tr, shape, crs) = load_band(os.path.join(FOLDERS[folder], fname))
    assert tr == ref_transform and shape == ref_shape and crs == ref_crs, f'{fname} is not on the DEM grid'
    arrays.append(arr)

# Boundary mask on the DEM grid. Only cells inside the ROI are predicted.
boundary = gpd.read_file(BOUNDARY_FILE).to_crs(TARGET_EPSG)
inside = rasterize(
    [(geom, 1) for geom in boundary.geometry],
    out_shape=ref_shape, transform=ref_transform, fill=0, dtype='uint8',
).astype(bool)

valid = inside.copy()
for arr in arrays:
    valid &= np.isfinite(arr)

X = np.column_stack([arr[valid] for arr in arrays])
model = joblib.load(MODEL_FILE)
assert model.n_features_in_ == len(PREDICTORS), 'model and predictor list do not match'

prob_values = np.empty(X.shape[0], dtype=np.float32)
for start in range(0, X.shape[0], CHUNK):
    stop = min(start + CHUNK, X.shape[0])
    prob_values[start:stop] = model.predict_proba(X[start:stop])[:, 1]

prob = np.full(ref_shape, -9999.0, dtype=np.float32)
prob[valid] = prob_values

cls = np.zeros(ref_shape, dtype=np.uint8)
cls[valid] = np.where(prob_values < LOW_MAX, 1,
                      np.where(prob_values < MEDIUM_MAX, 2, 3)).astype(np.uint8)

profile = dict(ref_profile)
profile.update(driver='GTiff', count=1, compress='lzw')

prob_profile = dict(profile, dtype='float32', nodata=-9999.0)
with rasterio.open(os.path.join(OUT_DIR, 'rf_probability_2016_2024.tif'), 'w', **prob_profile) as dst:
    dst.write(prob, 1)

class_profile = dict(profile, dtype='uint8', nodata=0)
with rasterio.open(os.path.join(OUT_DIR, 'rf_class_2016_2024.tif'), 'w', **class_profile) as dst:
    dst.write(cls, 1)

# Area per class. Each cell is the DEM pixel size squared.
cell_m2 = abs(ref_transform.a * ref_transform.e)
counts = np.bincount(cls[valid], minlength=4)
total_valid = int(valid.sum())
roi_cells = int(inside.sum())
labels = {1: ('Low', '0.00 to 0.33'), 2: ('Medium', '0.33 to 0.66'), 3: ('High', '0.66 to 1.00')}
rows = []
for code in (1, 2, 3):
    n = int(counts[code])
    rows.append({
        'class': labels[code][0],
        'code': code,
        'probability_range': labels[code][1],
        'cells': n,
        'area_ha': round(n * cell_m2 / 10000, 2),
        'area_km2': round(n * cell_m2 / 1e6, 2),
        'percent_of_predicted_area': round(100 * n / total_valid, 2),
    })
rows.append({
    'class': 'Total predicted', 'code': '', 'probability_range': '',
    'cells': total_valid,
    'area_ha': round(total_valid * cell_m2 / 10000, 2),
    'area_km2': round(total_valid * cell_m2 / 1e6, 2),
    'percent_of_predicted_area': 100.0,
})
rows.append({
    'class': 'ROI (boundary)', 'code': '', 'probability_range': '',
    'cells': roi_cells,
    'area_ha': round(roi_cells * cell_m2 / 10000, 2),
    'area_km2': round(roi_cells * cell_m2 / 1e6, 2),
    'percent_of_predicted_area': round(100 * total_valid / roi_cells, 2),
})
area = pd.DataFrame(rows)
area.to_csv(os.path.join(OUT_DIR, 'area_table.csv'), index=False)

summary = pd.DataFrame([{
    'cell_size_m': round(float(abs(ref_transform.a)), 4),
    'cell_area_m2': round(float(cell_m2), 2),
    'roi_cells': roi_cells,
    'predicted_cells': total_valid,
    'not_predicted_cells_in_roi': roi_cells - total_valid,
    'prob_min': round(float(prob_values.min()), 4),
    'prob_max': round(float(prob_values.max()), 4),
    'prob_mean': round(float(prob_values.mean()), 4),
}])
summary.to_csv(os.path.join(OUT_DIR, 'prediction_summary.csv'), index=False)

# Copy the four period predictors used for the map, as outputs.
for name, folder, fname in PREDICTORS[:4]:
    shutil.copy2(os.path.join(FOLDERS[folder], fname), os.path.join(PRED_OUT, fname))

print('\nPhase 7: map and areas (2016 to 2024 predictors)')
print('=' * 60)
print(area.to_string(index=False))
print('-' * 60)
print(summary.T.to_string(header=False))
print('RESULT: complete. Rasters, area_table.csv and predictor copies written to outputs/.')