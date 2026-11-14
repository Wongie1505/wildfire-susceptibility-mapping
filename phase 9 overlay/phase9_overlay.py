import warnings
warnings.filterwarnings('ignore')

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import PathPatch, Polygon, Rectangle, Patch
from matplotlib.path import Path
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# Paths
CLASS_FILE = r'C:\Users\wongi\OneDrive\Desktop\Year 4\Project Thesis\My Research\Project and thesis\Trial-6 after review\phase 9 overlay\data\rasters\rf_class_2016_2024.tif'
FIRE_FILE = r'C:\Users\wongi\OneDrive\Desktop\Year 4\Project Thesis\My Research\Project and thesis\Trial-6 after review\phase 9 overlay\data\raw\fire_archive_M-C61_847303.shp'        # filtered, confidence 75+, inside the ROI
BOUNDARY_FILE = r'C:\Users\wongi\OneDrive\Desktop\Year 4\Project Thesis\My Research\Project and thesis\Trial-6 after review\phase 9 overlay\data\boundary\Chikangawa_forest.shp'
OUTPUT_FOLDER = r'C:\Users\wongi\OneDrive\Desktop\Year 4\Project Thesis\My Research\Project and thesis\Trial-6 after review\phase 9 overlay\outputs'

YEAR = 2025
SCALE_KM = 30
HOT = 0.3         # normalised density above this is drawn as a hotspot
PAD = 2000         # metres of padding around the ROI
GRID = 500         # metres per density grid cell

# Load
with rasterio.open(CLASS_FILE) as src:
    data = src.read(1)
    rb = src.bounds
data = np.ma.masked_where(~np.isin(data, [1, 2, 3]), data)
extent = [rb.left, rb.right, rb.bottom, rb.top]

boundary = gpd.read_file(BOUNDARY_FILE)
fires = gpd.read_file(FIRE_FILE).to_crs(boundary.crs)
fc = np.array([(p.x, p.y) for p in fires.geometry])
n_fires = len(fires)

minx, miny, maxx, maxy = boundary.total_bounds
xlim = (minx - PAD, maxx + PAD)
ylim = (miny - PAD, maxy + PAD)

# Fire density surface from the 2025 detections only
gx = np.arange(*xlim, GRID)
gy = np.arange(*ylim, GRID)
xx, yy = np.meshgrid(gx, gy)
kde = gaussian_kde(fc.T, bw_method=0.1)
dens = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
dens = (dens - dens.min()) / (dens.max() - dens.min())

# Colours and labels (same as the earlier map)
susc_colors = ['#2ecc71', '#f1c40f', '#e74c3c']
susc_cmap = ListedColormap(susc_colors)
susc_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3)
susc_labels = ['Low (0 - 0.33)', 'Medium (0.33 - 0.66)', 'High (0.66 - 1)']


def forest_path():
    geom = boundary.geometry.union_all()
    polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
    verts, codes = [], []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            verts.extend(xy)
            codes += [Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


FPATH = forest_path()


def clip(artist, ax):
    patch = PathPatch(FPATH, transform=ax.transData, fc='none', ec='none')
    ax.add_patch(patch)
    try:
        artist.set_clip_path(patch)
    except Exception:
        for c in artist.collections:
            c.set_clip_path(patch)


def frame(fig):
    fig.add_artist(Rectangle((0.003, 0.003), 0.994, 0.994, transform=fig.transFigure,
                             fill=False, ec='black', lw=1.5))


def north_arrow(fig, x, y, w, h):
    a = fig.add_axes([x, y, w, h])
    a.set_xlim(0, 0.7); a.set_ylim(-0.25, 1); a.axis('off')
    a.add_patch(Polygon([(0.35, 1), (0, 0), (0.35, 0.25)], fc='black', ec='black', lw=1.2))
    a.add_patch(Polygon([(0.35, 1), (0.7, 0), (0.35, 0.25)], fc='white', ec='black', lw=1.2))
    a.text(0.35, -0.22, 'N', ha='center', va='bottom', fontsize=11, fontweight='bold')


def scale_bar(fig, ax, x0, yb):
    ax.apply_aspect()
    pos = ax.get_position()
    fpm = pos.width / (ax.get_xlim()[1] - ax.get_xlim()[0])
    fig.add_artist(Line2D([x0, x0 + SCALE_KM * 1000 * fpm], [yb, yb], color='black', lw=1.5,
                          transform=fig.transFigure))
    for km in range(0, SCALE_KM + 1, 10):
        xt = x0 + km * 1000 * fpm
        fig.add_artist(Line2D([xt, xt], [yb, yb - 0.012], color='black', lw=1.5,
                              transform=fig.transFigure))
        fig.text(xt, yb + 0.012, f'{km} km' if km == SCALE_KM else str(km),
                 ha='center', va='bottom', fontsize=11)


def data_box(fig, x, y, text):
    fig.text(x, y, text, fontsize=10, ha='left', va='bottom', linespacing=1.4,
             bbox=dict(boxstyle='square,pad=0.6', fc='white', ec='black', lw=1.2))


# ---------- the map ----------
fig = plt.figure(figsize=(11, 9.5), facecolor='white')
frame(fig)

ax = fig.add_axes([0.02, 0.03, 0.42, 0.94])
ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_aspect('equal'); ax.axis('off'); ax.set_anchor('W')

ax.imshow(data, cmap=susc_cmap, norm=susc_norm, extent=extent, origin='upper',
          interpolation='nearest', zorder=1)

plt.rcParams['hatch.linewidth'] = 0.8
hot = ax.contourf(xx, yy, dens, levels=[HOT, 1.01], colors='none', hatches=['////'], zorder=2)
clip(hot, ax)
ln = ax.contour(xx, yy, dens, levels=[HOT], colors='black', linewidths=1.3, zorder=2)
clip(ln, ax)
boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.2, zorder=3)

north_arrow(fig, 0.012, 0.87, 0.06, 0.11)

fig.text(0.76, 0.86, 'Viphya Plantation - Wildfire Susceptibility Mapping\n'
         f'{YEAR} fire detections on the 2016 to 2024 susceptibility map',
         fontsize=18, fontweight='bold', ha='center', va='center', linespacing=1.3)

legend_ax = fig.add_axes([0.5, 0.52, 0.3, 0.24]); legend_ax.axis('off')
handles = [Patch(fc=c, ec='black', label=l) for c, l in zip(susc_colors, susc_labels)]
handles.append(Patch(fc='white', ec='black', hatch='////', label='Fire hotspot'))
legend_ax.legend(handles=handles, loc='upper left', title='Legend', title_fontsize=13,
                 fontsize=12, frameon=False, alignment='left')

scale_bar(fig, ax, 0.52, 0.42)
data_box(fig, 0.52, 0.06,
         f'Data Source:  Random Forest model\n(2016 to 2024 predictors)\n'
         f'MODIS fire detections {YEAR} (n = {n_fires:,})\n'
         'Confidence 75 and above; not used in training\n\nCRS: EPSG 32736')

fig.savefig(rf'{OUTPUT_FOLDER}\Map_{YEAR}_Overlay.png', dpi=300, facecolor='white')
plt.close(fig)
print('Done.')