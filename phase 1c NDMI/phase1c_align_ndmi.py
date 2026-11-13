import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

# Folder layout (relative to this script):
#   Phase_1c_ndmi/
#       phase1c_align_ndmi.py
#       data/input/viphya_ndmi_modis.tif   GEE export, MOD09A1, 500 m
#       data/reference/Viphya_DEM.tif      reference grid (copy from Phase 1b)
#       data/aligned/viphya_ndmi.tif       output
#       outputs/ndmi_alignment_check.csv

HERE = os.path.dirname(os.path.abspath(__file__))
IN_FILE = os.path.join(HERE, 'data', 'input', 'viphya_ndmi_modis.tif')
REF_FILE = os.path.join(HERE, 'data', 'reference', 'Viphya_DEM.tif')
OUT_DIR = os.path.join(HERE, 'data', 'aligned')
OUT_FILE = os.path.join(OUT_DIR, 'viphya_ndmi.tif')
REPORT_DIR = os.path.join(HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

FLOAT_NODATA = -9999.0

with rasterio.open(REF_FILE) as ref:
    ref_transform = ref.transform
    ref_shape = (ref.height, ref.width)
    ref_crs = ref.crs

with rasterio.open(IN_FILE) as src:
    src_res = src.res[0]
    src_crs = src.crs
    src_transform = src.transform
    declared_nodata = src.nodata
    data = src.read(1).astype(np.float32)

# NaN and the declared nodata value both count as missing.
missing = ~np.isfinite(data)
if declared_nodata is not None and np.isfinite(declared_nodata):
    missing |= data == np.float32(declared_nodata)

input_valid_pct = 100 * (1 - missing.mean())
print(f'Input valid pixels: {input_valid_pct:.2f} percent')

if input_valid_pct == 0:
    raise SystemExit('The export has no valid pixels. Re-export viphya_ndmi_modis from GEE.')

work = data.copy()
work[missing] = FLOAT_NODATA

destination = np.full(ref_shape, FLOAT_NODATA, dtype=np.float32)
reproject(
    source=work,
    destination=destination,
    src_transform=src_transform,
    src_crs=src_crs,
    src_nodata=FLOAT_NODATA,
    dst_transform=ref_transform,
    dst_crs=ref_crs,
    dst_nodata=FLOAT_NODATA,
    resampling=Resampling.bilinear,
)

profile = {
    'driver': 'GTiff',
    'height': ref_shape[0],
    'width': ref_shape[1],
    'count': 1,
    'dtype': np.float32,
    'crs': ref_crs,
    'transform': ref_transform,
    'nodata': FLOAT_NODATA,
    'compress': 'lzw',
}
with rasterio.open(OUT_FILE, 'w', **profile) as dst:
    dst.write(destination, 1)

with rasterio.open(OUT_FILE) as out:
    out_data = out.read(1)
    same_grid = (out.transform == ref_transform
                 and (out.height, out.width) == ref_shape
                 and out.crs == ref_crs)
    nodata_set = out.nodata is not None

is_nodata = (out_data == FLOAT_NODATA) | ~np.isfinite(out_data)
valid = out_data[~is_nodata]
values_ok = bool(valid.size > 0 and valid.min() >= -1.0 and valid.max() <= 1.0)
passed = bool(same_grid and nodata_set and values_ok)

check = pd.DataFrame([{
    'file': 'viphya_ndmi.tif',
    'source_file': 'viphya_ndmi_modis.tif',
    'source_resolution_m': round(float(src_res), 4),
    'output_resolution_m': round(float(ref_transform.a), 4),
    'input_valid_pct': round(float(input_valid_pct), 2),
    'resampling': 'bilinear',
    'same_grid_as_reference': bool(same_grid),
    'nodata_set': nodata_set,
    'output_nodata_share_pct': round(100 * float(is_nodata.mean()), 2),
    'valid_min': round(float(valid.min()), 4) if valid.size else None,
    'valid_max': round(float(valid.max()), 4) if valid.size else None,
    'values_within_minus1_to_1': values_ok,
    'status': 'PASS' if passed else 'FAIL',
}])
check.to_csv(os.path.join(REPORT_DIR, 'ndmi_alignment_check.csv'), index=False)

print('\nPhase 1c: NDMI alignment')
print('=' * 60)
print(check.T.to_string(header=False))
print('-' * 60)
if passed:
    print('RESULT: PASS. viphya_ndmi.tif is on the DEM grid and ready for Phase 2.')
else:
    print('RESULT: FAIL. See ndmi_alignment_check.csv before continuing.')