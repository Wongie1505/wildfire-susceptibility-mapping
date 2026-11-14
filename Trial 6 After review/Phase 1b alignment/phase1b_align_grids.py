import os
import shutil
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

# Folder layout (relative to this script):
#   Phase_1b_align/
#       phase1b_align_grids.py   (this file)
#       data/input/              copies of the 12 predictors from Phase 1
#       data/aligned/            written by this script
#       outputs/                 verification report

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, 'data', 'input')
OUT_DIR = os.path.join(HERE, 'data', 'aligned')
REPORT_DIR = os.path.join(HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

REFERENCE = 'Viphya_DEM.tif'

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

# Categorical layers keep their class codes, so they use nearest neighbour.
# Continuous layers are interpolated.
CATEGORICAL = {'Viphya_LULC.tif'}

FLOAT_NODATA = -9999.0
CATEGORICAL_NODATA = 255

log = []


def same_grid(a, b_transform, b_shape, b_crs):
    return (a.transform == b_transform and (a.height, a.width) == b_shape and a.crs == b_crs)


with rasterio.open(os.path.join(IN_DIR, REFERENCE)) as ref:
    ref_transform = ref.transform
    ref_shape = (ref.height, ref.width)
    ref_crs = ref.crs

for name in PREDICTORS:
    src_path = os.path.join(IN_DIR, name)
    dst_path = os.path.join(OUT_DIR, name)

    with rasterio.open(src_path) as src:
        if same_grid(src, ref_transform, ref_shape, ref_crs):
            src.close()
            shutil.copy2(src_path, dst_path)
            log.append({'file': name, 'action': 'copied (already on reference grid)',
                        'method': '', 'nodata': ''})
            continue

        categorical = name in CATEGORICAL
        method = Resampling.nearest if categorical else Resampling.bilinear
        dtype = np.uint8 if categorical else np.float32
        dst_nodata = CATEGORICAL_NODATA if categorical else FLOAT_NODATA

        if src.nodata is not None:
            src_nodata = src.nodata
        elif np.issubdtype(src.dtypes[0], np.floating):
            src_nodata = np.nan
        else:
            src_nodata = None

        data = src.read(1)
        destination = np.full(ref_shape, dst_nodata, dtype=dtype)

        reproject(
            source=data,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=dst_nodata,
            resampling=method,
        )

        profile = {
            'driver': 'GTiff',
            'height': ref_shape[0],
            'width': ref_shape[1],
            'count': 1,
            'dtype': dtype,
            'crs': ref_crs,
            'transform': ref_transform,
            'nodata': dst_nodata,
            'compress': 'lzw',
        }
        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(destination, 1)

        log.append({'file': name,
                    'action': f'resampled from {src.res[0]:.4f} x {src.res[1]:.4f} m',
                    'method': 'nearest' if categorical else 'bilinear',
                    'nodata': dst_nodata})

# ---------- verify every aligned file against the reference ----------
checks = []
for name in PREDICTORS:
    path = os.path.join(OUT_DIR, name)
    with rasterio.open(path) as out:
        ok_grid = same_grid(out, ref_transform, ref_shape, ref_crs)
        ok_nodata = out.nodata is not None
        share = float(np.mean(out.read(1) == out.nodata)) * 100
    checks.append({
        'file': name,
        'same_grid_as_reference': ok_grid,
        'nodata_set': ok_nodata,
        'nodata_share_pct': round(share, 2),
        'status': 'PASS' if (ok_grid and ok_nodata) else 'FAIL',
    })

pd.DataFrame(log).to_csv(os.path.join(REPORT_DIR, 'alignment_log.csv'), index=False)
verify = pd.DataFrame(checks)
verify.to_csv(os.path.join(REPORT_DIR, 'aligned_check.csv'), index=False)

print('\nPhase 1b: grid alignment')
print('=' * 60)
for row in log:
    print(f"  {row['file']}: {row['action']}" + (f" ({row['method']})" if row['method'] else ''))
print('-' * 60)
print('Verification against the reference DEM grid:')
for _, r in verify.iterrows():
    print(f"  [{r['status']}] {r['file']}  nodata share {r['nodata_share_pct']}%")

n_fail = int((verify['status'] == 'FAIL').sum())
if n_fail == 0:
    print(f'\nRESULT: PASS. All {len(PREDICTORS)} predictors now share the reference grid.')
    print(f'Aligned rasters are in {OUT_DIR}')
else:
    print(f'\nRESULT: {n_fail} file(s) still fail. See {os.path.join(REPORT_DIR, "aligned_check.csv")}')