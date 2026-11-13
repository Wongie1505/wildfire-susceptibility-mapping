import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch

# Draws the two-panel susceptibility map and the class area chart from the Phase 7b
# rasters. Set PERIOD to 2002_2015 later for the diagnostic map.
#
# Folder layout (relative to this script):
#   Phase_8b_maps/
#       phase8b_maps.py
#       data/rasters/        rf_probability_<PERIOD>.tif, rf_class_<PERIOD>.tif, area_table.csv
#       outputs/             map_<PERIOD>.png, class_area_<PERIOD>.png

HERE = os.path.dirname(os.path.abspath(__file__))
PERIOD = '2016_2024'
RASTERS = os.path.join(HERE, 'data', 'rasters')
OUT_DIR = os.path.join(HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

PERIOD_TEXT = PERIOD.replace('_', ' to ')
TITLE = 'Viphya Plantation - Wildfire Susceptibility Mapping'
SUBTITLE = f'Random Forest model, {PERIOD_TEXT} predictors'

SCORE_CMAP = LinearSegmentedColormap.from_list('score', ['#1a9850', '#ffff33', '#800026'])
CLASS_COLOURS = ['#00B050', '#E3E33D', '#F83A3A']
CLASS_CMAP = ListedColormap(CLASS_COLOURS)
CLASS_LABELS = ['Low (0 to 0.33)', 'Medium (0.33 to 0.66)', 'High (0.66 to 1.00)']


def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        transform = src.transform
        nodata = src.nodata
    bad = ~np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        bad |= data == nodata
    return np.ma.masked_where(bad, data), transform


prob, transform = read_raster(os.path.join(RASTERS, f'rf_probability_{PERIOD}.tif'))
cls, _ = read_raster(os.path.join(RASTERS, f'rf_class_{PERIOD}.tif'))
cls = np.ma.masked_where(cls == 0, cls)

rows, cols = prob.shape
extent = [0, cols * abs(transform.a) / 1000, 0, rows * abs(transform.e) / 1000]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
fig.suptitle(TITLE, fontsize=22, fontweight='bold', y=0.98)
fig.text(0.5, 0.935, SUBTITLE, ha='center', fontsize=15)

im1 = ax1.imshow(prob, cmap=SCORE_CMAP, vmin=0, vmax=1, extent=extent, origin='upper')
ax1.set_title('Susceptibility: Continuous', fontsize=15, fontweight='bold')
cbar = fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.03)
cbar.set_label('Susceptibility score', fontsize=12)
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax2.imshow(cls - 1, cmap=CLASS_CMAP, vmin=0, vmax=2, extent=extent, origin='upper')
ax2.set_title('Susceptibility: Classification (Low / Medium / High)', fontsize=15, fontweight='bold')
legend = [Patch(facecolor=c, edgecolor='black', label=l) for c, l in zip(CLASS_COLOURS, CLASS_LABELS)]
ax2.legend(handles=legend, title='Susceptibility ranges', loc='lower right', fontsize=10,
           title_fontsize=11, frameon=True)

for ax in (ax1, ax2):
    ax.set_xlabel('km', fontsize=11)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([])
    ax.set_aspect('equal')
    # North arrow
    ax.annotate('', xy=(0.06, 0.93), xytext=(0.06, 0.82), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', lw=2, color='black'))
    ax.text(0.06, 0.95, 'N', transform=ax.transAxes, ha='center', fontsize=13, fontweight='bold')

fig.tight_layout(rect=[0, 0, 1, 0.92])
map_path = os.path.join(OUT_DIR, f'map_{PERIOD}.png')
fig.savefig(map_path, dpi=300, bbox_inches='tight')
plt.close(fig)

area = pd.read_csv(os.path.join(RASTERS, 'area_table.csv'), dtype={'code': str})
classes = area[area['code'].isin(['1', '2', '3'])].copy()
classes['code'] = classes['code'].astype(int)
classes = classes.sort_values('code')

fig2, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(classes['class'], classes['area_km2'], color=CLASS_COLOURS, edgecolor='black')
for bar, km2, pct in zip(bars, classes['area_km2'], classes['percent_of_predicted_area']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f'{km2:,.1f} km2\n{pct:.1f} percent', ha='center', va='bottom', fontsize=11)
ax.set_ylabel('Area (km2)', fontsize=12)
ax.set_ylim(0, classes['area_km2'].max() * 1.25)
ax.set_title(f'Area per susceptibility class, {PERIOD_TEXT} predictors', fontsize=14, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
fig2.tight_layout()
area_path = os.path.join(OUT_DIR, f'class_area_{PERIOD}.png')
fig2.savefig(area_path, dpi=300, bbox_inches='tight')
plt.close(fig2)

print('\nPhase 8b: maps')
print('=' * 60)
print(f'Written: {map_path}')
print(f'Written: {area_path}')
print('RESULT: complete.')