# ================================================
# VIPHYA - WILDFIRE SUSCEPTIBILITY MAPS
# Produces maps for RF, XGBoost and LightGBM
# Best model (RF) is the primary map
# ================================================

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ================================================
# 1. PATHS
# ================================================

data_dir = r"C:\Users\wongi\OneDrive\Desktop\Year 3\Sem 2\My Research\Project and thesis\Trial_five\DATA"

rasters = {
    'Dist_Rivers'    : 'Rivers_Distances.tif',
    'Dist_Roads'     : 'Roads_distances.tif',
    'Dist_Settlement': 'Settlement_Distance.tif',
    'Eastness'       : 'Viphya_Aspect_eastness.tif',
    'Northness'      : 'Viphya_Aspect_northness.tif',
    'Elevation'      : 'Viphya_DEM.tif',
    'LST'            : 'viphya_lst.tif',
    'LULC'           : 'Viphya_LULC.tif',
    'NDMI'           : 'viphya_ndmi.tif',
    'NDVI'           : 'viphya_ndvi.tif',
    'Rainfall'       : 'viphya_rainfall.tif',
    'Slope'          : 'Viphya_Slope.tif',
}

# Load all three models and feature list
rf        = joblib.load('model_rf.pkl')
xgb_model = joblib.load('model_xgb.pkl')
lgb_model = joblib.load('model_lgb.pkl')
features  = joblib.load('model_features.pkl')

print("Models loaded.")
print("Features:", features)

# ================================================
# 2. LOAD REFERENCE RASTER
# ================================================

ref_path = os.path.join(data_dir, 'Viphya_DEM.tif')

with rasterio.open(ref_path) as ref:
    ref_crs       = ref.crs
    ref_transform = ref.transform
    ref_res       = ref.res
    ref_width     = ref.width
    ref_height    = ref.height
    ref_nodata    = ref.nodata
    ref_meta      = ref.meta.copy()
    ref_data      = ref.read(1).astype(np.float32)

print(f"\nReference grid : {ref_width} x {ref_height} pixels")
print(f"Resolution     : {ref_res}")
print(f"CRS            : {ref_crs}")

# Valid pixel mask
if ref_nodata is not None:
    valid_mask = ref_data != ref_nodata
else:
    valid_mask = np.isfinite(ref_data)

print(f"Valid pixels   : {valid_mask.sum():,}")

# ================================================
# 3. LOAD AND RESAMPLE ALL RASTERS
# ================================================

print("\nLoading rasters...")

band_arrays = {}

for col_name, filename in rasters.items():
    filepath = os.path.join(data_dir, filename)

    with rasterio.open(filepath) as src:
        if src.crs == ref_crs and src.res == ref_res:
            arr    = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
        else:
            arr = np.empty((ref_height, ref_width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan

    band_arrays[col_name] = arr
    print(f"  ✓ {col_name}")

# ================================================
# 4. BUILD PIXEL TABLE
# ================================================

print("\nBuilding pixel table...")

rows_idx, cols_idx = np.where(valid_mask)

pixel_df = pd.DataFrame({
    col: band_arrays[col][rows_idx, cols_idx]
    for col in features
})

# Remove NaN rows
nan_mask       = pixel_df.notnull().all(axis=1)
pixel_df_clean = pixel_df[nan_mask]
valid_rows     = rows_idx[nan_mask]
valid_cols     = cols_idx[nan_mask]

print(f"Pixels for prediction: {len(pixel_df_clean):,}")

# ================================================
# 5. PREDICT WITH ALL THREE MODELS
# ================================================

print("\nPredicting with all three models...")

models = {
    'Random_Forest': rf,
    'XGBoost'      : xgb_model,
    'LightGBM'     : lgb_model
}

probabilities = {}

for name, model in models.items():
    print(f"  Running {name}...")
    proba = model.predict_proba(pixel_df_clean)[:, 1]
    probabilities[name] = proba
    print(f"  ✓ {name} — prob range: {proba.min():.3f} to {proba.max():.3f}")

# ================================================
# 6. RECONSTRUCT RASTERS AND SAVE GEOTIFFS
# ================================================

print("\nSaving GeoTIFFs...")

# Metadata for probability maps
prob_meta = ref_meta.copy()
prob_meta.update({
    'driver': 'GTiff',
    'dtype' : 'float32',
    'nodata': -9999,
    'count' : 1
})

# Metadata for classified maps
class_meta = ref_meta.copy()
class_meta.update({
    'driver': 'GTiff',
    'dtype' : 'uint8',
    'nodata': 0,
    'count' : 1
})

saved_files  = {}
class_labels = {1: 'Low', 2: 'Moderate', 3: 'High'}

for name, proba in probabilities.items():

    # --- Probability raster ---
    prob_raster = np.full((ref_height, ref_width), -9999, dtype=np.float32)
    prob_raster[valid_rows, valid_cols] = proba

    prob_path = os.path.join(data_dir, f'susceptibility_prob_{name}.tif')
    with rasterio.open(prob_path, 'w', **prob_meta) as dst:
        dst.write(prob_raster, 1)

    # --- Classified raster (1=Low, 2=Moderate, 3=High) ---
    class_raster = np.zeros((ref_height, ref_width), dtype=np.uint8)
    class_raster[valid_rows, valid_cols] = np.where(
        proba <= 0.33, 1,
        np.where(proba <= 0.66, 2, 3)
    )

    class_path = os.path.join(data_dir, f'susceptibility_class_{name}.tif')
    with rasterio.open(class_path, 'w', **class_meta) as dst:
        dst.write(class_raster, 1)

    saved_files[name] = {
        'prob' : prob_path,
        'class': class_path,
        'raster_prob' : prob_raster,
        'raster_class': class_raster,
        'proba'       : proba
    }

    # Print class distribution
    classes, counts = np.unique(
        class_raster[class_raster > 0], return_counts=True
    )
    total = counts.sum()
    print(f"\n  {name} class distribution:")
    for c, n in zip(classes, counts):
        print(f"    {class_labels[c]:10s}: {n:>8,} pixels  ({n/total*100:.1f}%)")

print("\nAll GeoTIFFs saved.")

# # ================================================
# # 7. PREVIEW PLOTS — ONE FIGURE PER MODEL
# # ================================================

# print("\nGenerating preview plots...")

# cmap_class = mcolors.ListedColormap(['green', 'yellow', 'red'])
# bounds_c   = [0.5, 1.5, 2.5, 3.5]
# norm_c     = mcolors.BoundaryNorm(bounds_c, cmap_class.N)

# legend_elements = [
#     Patch(facecolor='green',  label='Low  (0.00 – 0.33)'),
#     Patch(facecolor='yellow', label='Moderate (0.34 – 0.66)'),
#     Patch(facecolor='red',    label='High (0.67 – 1.00)')
# ]

# for name, data in saved_files.items():

#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#     # Probability map
#     prob_plot = data['raster_prob'].copy().astype(float)
#     prob_plot[prob_plot == -9999] = np.nan

#     im1 = axes[0].imshow(prob_plot, cmap='RdYlGn_r', vmin=0, vmax=1)
#     plt.colorbar(im1, ax=axes[0], label='Fire Probability')
#     axes[0].set_title('Susceptibility Probability\n(Continuous)', fontsize=12)
#     axes[0].axis('off')

#     # Classified map
#     class_plot = data['raster_class'].astype(float)
#     class_plot[class_plot == 0] = np.nan

#     axes[1].imshow(class_plot, cmap=cmap_class, norm=norm_c)
#     axes[1].legend(handles=legend_elements,
#                    loc='lower left', fontsize=9)
#     axes[1].set_title('Susceptibility Class\n(Low / Moderate / High)', fontsize=12)
#     axes[1].axis('off')

#     model_label = name.replace('_', ' ')
#     plt.suptitle(
#         f'Viphya Plantation — Wildfire Susceptibility\n{model_label}',
#         fontsize=14, fontweight='bold'
#     )
#     plt.tight_layout()

#     plot_path = os.path.join(data_dir, f'map_preview_{name}.png')
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"  ✓ Preview saved: map_preview_{name}.png")

# # ================================================
# # 8. SIDE BY SIDE COMPARISON — CLASSIFIED MAPS
# # ================================================

# fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# titles = {
#     'Random_Forest': f'Random Forest\n(AUC = 0.9315)',
#     'XGBoost'      : f'XGBoost\n(AUC = 0.9080)',
#     'LightGBM'     : f'LightGBM\n(AUC = 0.9018)'
# }

# for ax, (name, data) in zip(axes, saved_files.items()):
#     class_plot = data['raster_class'].astype(float)
#     class_plot[class_plot == 0] = np.nan
#     ax.imshow(class_plot, cmap=cmap_class, norm=norm_c)
#     ax.set_title(titles[name], fontsize=12)
#     ax.axis('off')

# # Single legend for all three
# fig.legend(handles=legend_elements,
#            loc='lower center',
#            ncol=3,
#            fontsize=11,
#            bbox_to_anchor=(0.5, 0.02))

# plt.suptitle(
#     'Viphya Plantation — Wildfire Susceptibility Map Comparison',
#     fontsize=14, fontweight='bold'
# )
# plt.tight_layout(rect=[0, 0.08, 1, 1])
# comparison_path = os.path.join(data_dir, 'map_comparison_all_models.png')
# plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
# plt.show()
# print(f"\nComparison figure saved: map_comparison_all_models.png")


# ================================================
# 7. PREVIEW PLOTS — ONE FIGURE PER MODEL
# With fixed legend, scale bar and north arrow
# ================================================

print("\nGenerating preview plots...")

cmap_class = mcolors.ListedColormap(['green', 'yellow', 'red'])
bounds_c   = [0.5, 1.5, 2.5, 3.5]
norm_c     = mcolors.BoundaryNorm(bounds_c, cmap_class.N)

legend_elements = [
    Patch(facecolor='green',  edgecolor='grey', label='Low  (0.00 – 0.33)'),
    Patch(facecolor='yellow', edgecolor='grey', label='Moderate (0.34 – 0.66)'),
    Patch(facecolor='red',    edgecolor='grey', label='High (0.67 – 1.00)')
]

def add_scale_bar(ax, img_width_px, pixel_size_m, location='lower right'):
    """
    Adds a scale bar to a matplotlib axis.
    img_width_px : width of the image in pixels
    pixel_size_m : size of one pixel in metres
    """
    # Work out a round scale bar length in km
    map_width_m   = img_width_px * pixel_size_m
    target_frac   = 0.25                         # bar = ~25% of map width
    raw_m         = map_width_m * target_frac
    # Round to nearest nice km value
    for nice in [1000,2000,5000,10000,20000,50000]:
        if raw_m <= nice:
            bar_m = nice
            break
    else:
        bar_m = 50000

    bar_km   = bar_m / 1000
    bar_frac = bar_m / map_width_m   # fraction of axes width

    # Position
    x0 = 0.65 if location == 'lower right' else 0.05
    y0 = 0.06

    # Draw bar
    ax.annotate('', xy=(x0 + bar_frac, y0),
                xytext=(x0, y0),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-',
                                color='black', lw=2))
    # Tick marks at each end
    for xpos in [x0, x0 + bar_frac]:
        ax.annotate('', xy=(xpos, y0 + 0.015),
                    xytext=(xpos, y0 - 0.015),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='-',
                                    color='black', lw=1.5))
    # Labels
    ax.text(x0, y0 - 0.03, '0',
            ha='center', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(x0 + bar_frac, y0 - 0.03, f'{bar_km:.0f} km',
            ha='center', va='top',
            transform=ax.transAxes, fontsize=8)
    # 'km' unit label above bar centre
    ax.text(x0 + bar_frac/2, y0 + 0.03, 'Scale',
            ha='center', va='bottom',
            transform=ax.transAxes, fontsize=7, color='black')


def add_north_arrow(ax):
    """Adds a north arrow to upper right of axis."""
    x, y = 0.93, 0.88   # axes fraction position
    ax.annotate('', xy=(x, y + 0.07),
                xytext=(x, y),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black',
                                lw=2, mutation_scale=15))
    ax.text(x, y - 0.03, 'N',
            ha='center', va='top',
            transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='black')


# Get pixel size in metres from reference transform
pixel_size_m = abs(ref_transform[0])   # x pixel size from affine transform

for name, data in saved_files.items():

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # ---- Left: Probability map ----
    prob_plot = data['raster_prob'].copy().astype(float)
    prob_plot[prob_plot == -9999] = np.nan

    im1 = axes[0].imshow(prob_plot, cmap='RdYlGn_r', vmin=0, vmax=1)
    cbar = plt.colorbar(im1, ax=axes[0],
                        label='Fire Probability', shrink=0.75)
    axes[0].set_title('Susceptibility Probability\n(Continuous)', fontsize=12)
    axes[0].axis('off')

    # Scale bar on probability map
    add_scale_bar(axes[0], ref_width, pixel_size_m)

    # North arrow on probability map
    add_north_arrow(axes[0])

    # ---- Right: Classified map ----
    class_plot = data['raster_class'].astype(float)
    class_plot[class_plot == 0] = np.nan

    axes[1].imshow(class_plot, cmap=cmap_class, norm=norm_c)
    axes[1].set_title('Susceptibility Class\n(Low / Moderate / High)', fontsize=12)
    axes[1].axis('off')

    # Legend OUTSIDE the map — below the classified panel
    legend = axes[1].legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.04),   # below the axes
        ncol=3,
        fontsize=9,
        frameon=True,
        edgecolor='grey'
    )

    # Scale bar on classified map
    add_scale_bar(axes[1], ref_width, pixel_size_m)

    # North arrow on classified map
    add_north_arrow(axes[1])

    model_label = name.replace('_', ' ')
    plt.suptitle(
        f'Viphya Plantation — Wildfire Susceptibility\n{model_label}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plot_path = os.path.join(data_dir, f'map_preview_{name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Preview saved: map_preview_{name}.png")


# ================================================
# 8. THREE-MODEL COMPARISON
# Scale bar only — no north arrow
# ================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

titles = {
    'Random_Forest': 'Random Forest\n(AUC = 0.9315)',
    'XGBoost'      : 'XGBoost\n(AUC = 0.9080)',
    'LightGBM'     : 'LightGBM\n(AUC = 0.9018)'
}

for ax, (name, data) in zip(axes, saved_files.items()):
    class_plot = data['raster_class'].astype(float)
    class_plot[class_plot == 0] = np.nan
    ax.imshow(class_plot, cmap=cmap_class, norm=norm_c)
    ax.set_title(titles[name], fontsize=12)
    ax.axis('off')

    # Scale bar on each panel
    add_scale_bar(ax, ref_width, pixel_size_m, location='lower right')

# Single legend below all three panels
fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=3,
    fontsize=11,
    frameon=True,
    edgecolor='grey',
    bbox_to_anchor=(0.5, 0.01)
)

plt.suptitle(
    'Viphya Plantation — Wildfire Susceptibility Map Comparison',
    fontsize=14, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.09, 1, 1])

comparison_path = os.path.join(data_dir, 'map_comparison_all_models.png')
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nComparison figure saved: map_comparison_all_models.png")

# ================================================
# 9. SUMMARY
# ================================================

print(f"\n{'='*55}")
print("FILES SAVED TO YOUR DATA FOLDER:")
print('='*55)
for name in saved_files:
    print(f"  susceptibility_prob_{name}.tif")
    print(f"  susceptibility_class_{name}.tif")
    print(f"  map_preview_{name}.png")
print(f"  map_comparison_all_models.png")
print('='*55)
print("\nLoad the .tif files in QGIS for final cartography.")
print("Best model for your thesis: Random Forest (AUC = 0.9315)")