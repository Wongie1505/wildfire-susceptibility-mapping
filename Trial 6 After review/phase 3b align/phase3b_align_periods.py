import os
import glob
import shutil
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

# Folder layout (relative to this script):
#   Phase_3b_align/
#       phase3b_align_periods.py
#       data/input/       copies of the 8 Phase 3 exports (made by this script)
#       data/reference/   Viphya_DEM.tif, copied from Phase 1b data/aligned
#       data/aligned/     the 8 clean outputs
#       outputs/          periods_alignment_check.csv

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(HERE, '..', 'Phase 3 period medians', 'outputs')
IN_DIR = os.path.join(HERE, 'data', 'input')
REF_FILE = os.path.join(HERE, 'data', 'reference', 'Viphya_DEM.tif')
OUT_DIR = os.path.join(HERE, 'data', 'aligned')
REPORT_DIR = os.path.join(HERE, 'outputs')

FLOAT_NODATA = -9999.0
MAX_MEAN_SHIFT_PCT = 5.0

# file name -> valid range for the layer
LAYERS = {
    'viphya_ndvi_2002_2015.tif': (-1.0, 1.0),
    'viphya_ndvi_2016_2024.tif': (-1.0, 1.0),
    'viphya_ndmi_2002_2015.tif': (-1.0, 1.0),
    'viphya_ndmi_2016_2024.tif': (-1.0, 1.0),
    'viphya_lst_2002_2015.tif': (-20.0, 60.0),
    'viphya_lst_2016_2024.tif': (-20.0, 60.0),
    'viphya_rainfall_2002_2015.tif': (0.0, 2000.0),
    'viphya_rainfall_2016_2024.tif': (0.0, 2000.0),
}

for folder in (IN_DIR, OUT_DIR, REPORT_DIR):
    os.makedirs(folder, exist_ok=True)

# Copy the Phase 3 exports into this phase's input folder.
for name in LAYERS:
    src = os.path.join(SOURCE_DIR, name)
    if not os.path.exists(src):
        raise SystemExit(f'Missing Phase 3 export: {src}')
    shutil.copy2(src, os.path.join(IN_DIR, name))

# Clear old outputs so the aligned folder holds only this run.
for old in glob.glob(os.path.join(OUT_DIR, '*.tif')):
    os.remove(old)

with rasterio.open(REF_FILE) as ref:
    ref_transform = ref.transform
    ref_shape = (ref.height, ref.width)
    ref_crs = ref.crs


def align(name, valid_min, valid_max):
    in_path = os.path.join(IN_DIR, name)
    out_path = os.path.join(OUT_DIR, name)

    with rasterio.open(in_path) as src:
        data = src.read(1).astype(np.float32)
        src_res = float(src.res[0])
        src_crs = src.crs
        src_transform = src.transform
        declared = src.nodata

    # NaN and the declared nodata value are treated as missing.
    missing = ~np.isfinite(data)
    if declared is not None and np.isfinite(declared):
        missing |= data == np.float32(declared)

    input_valid_pct = 100 * (1 - missing.mean())
    input_mean = float(data[~missing].mean()) if (~missing).any() else float('nan')

    work = np.where(missing, FLOAT_NODATA, data).astype(np.float32)
    dest = np.full(ref_shape, FLOAT_NODATA, dtype=np.float32)
    reproject(
        source=work,
        destination=dest,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=FLOAT_NODATA,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        dst_nodata=FLOAT_NODATA,
        resampling=Resampling.bilinear,
    )

    profile = {
        'driver': 'GTiff', 'height': ref_shape[0], 'width': ref_shape[1], 'count': 1,
        'dtype': np.float32, 'crs': ref_crs, 'transform': ref_transform,
        'nodata': FLOAT_NODATA, 'compress': 'lzw',
    }
    with rasterio.open(out_path, 'w', **profile) as out:
        out.write(dest, 1)

    # Verify the written file.
    with rasterio.open(out_path) as out:
        grid_ok = (out.transform == ref_transform and (out.height, out.width) == ref_shape
                   and out.crs == ref_crs)
        nodata_ok = out.nodata == FLOAT_NODATA
        result = out.read(1)

    is_nodata = (result == FLOAT_NODATA) | ~np.isfinite(result)
    valid = result[~is_nodata]
    output_valid_pct = 100 * valid.size / result.size
    out_min = float(valid.min()) if valid.size else float('nan')
    out_max = float(valid.max()) if valid.size else float('nan')
    out_mean = float(valid.mean()) if valid.size else float('nan')
    range_ok = bool(valid.size and out_min >= valid_min and out_max <= valid_max)
    shift_pct = 100 * (out_mean - input_mean) / abs(input_mean) if input_mean else float('nan')
    shift_ok = abs(shift_pct) <= MAX_MEAN_SHIFT_PCT

    status = 'PASS' if (grid_ok and nodata_ok and range_ok) else 'FAIL'
    if status == 'PASS' and not shift_ok:
        status = 'CHECK'

    return {
        'file': name,
        'source_resolution_m': round(src_res, 4),
        'input_valid_pct': round(float(input_valid_pct), 2),
        'output_valid_pct': round(output_valid_pct, 2),
        'input_mean': round(input_mean, 4),
        'output_mean': round(out_mean, 4),
        'mean_shift_pct': round(float(shift_pct), 2),
        'output_min': round(out_min, 4),
        'output_max': round(out_max, 4),
        'valid_range': f'{valid_min} to {valid_max}',
        'same_grid_as_reference': bool(grid_ok),
        'nodata_set': bool(nodata_ok),
        'status': status,
    }


rows = [align(name, lo, hi) for name, (lo, hi) in LAYERS.items()]
report = pd.DataFrame(rows)
report.to_csv(os.path.join(REPORT_DIR, 'periods_alignment_check.csv'), index=False)

aligned_files = sorted(os.listdir(OUT_DIR))
print('\nPhase 3b: align the period layers')
print('=' * 60)
print(report[['file', 'input_valid_pct', 'output_valid_pct', 'mean_shift_pct', 'status']].to_string(index=False))
print('-' * 60)
print(f'Files in data/aligned: {len(aligned_files)}')
if len(aligned_files) == len(LAYERS) and (report['status'] == 'PASS').all():
    print('RESULT: PASS. All eight layers are on the DEM grid and ready for Phase 4.')
elif (report['status'] == 'FAIL').any():
    print('RESULT: FAIL. See periods_alignment_check.csv.')
else:
    print('RESULT: PASS WITH CHECKS. Review the rows marked CHECK (mean shift above 5 percent).')