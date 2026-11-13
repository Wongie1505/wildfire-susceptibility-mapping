# ================================================
# VIPHYA - STACK RASTERS AND SAMPLE AT POINTS
# Output: viphya_training_data.csv
# ================================================

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import os

# ================================================
# 1. PATHS
# ================================================

data_dir = r"C:\Users\wongi\OneDrive\Desktop\Year 3\Sem 2\My Research\Project and thesis\Trial_five\DATA"

# All your tif files and their column names in the final CSV
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

# Your merged sample points
points_path = os.path.join(data_dir, r'Fire\Fire_non-fire_merge.gpkg')

# Output CSV
output_csv = os.path.join(data_dir, 'viphya_sampled_data.csv')

# ================================================
# 2. LOAD SAMPLE POINTS
# ================================================

print("Loading sample points...")
gdf = gpd.read_file(points_path)

print(f"Total points: {len(gdf)}")
print(f"Columns: {gdf.columns.tolist()}")
print(f"CRS: {gdf.crs}")
print(f"Label distribution:\n{gdf['Fire'].value_counts()}")

# ================================================
# 3. REPROJECT POINTS TO MATCH RASTER CRS
# We'll match everything to the DEM as reference
# ================================================

ref_path = os.path.join(data_dir, 'Viphya_DEM.tif')

with rasterio.open(ref_path) as ref:
    ref_crs  = ref.crs
    ref_transform = ref.transform
    ref_res  = ref.res
    ref_bounds = ref.bounds
    print(f"\nReference raster CRS: {ref_crs}")
    print(f"Reference resolution: {ref_res}")

# Reproject points to match raster CRS
if gdf.crs != ref_crs:
    print(f"\nReprojecting points from {gdf.crs} to {ref_crs}...")
    gdf = gdf.to_crs(ref_crs)
else:
    print("\nPoints CRS matches raster. No reprojection needed.")

# Get point coordinates as arrays
coords = [(geom.x, geom.y) for geom in gdf.geometry]

# ================================================
# 4. RESAMPLE ALL RASTERS TO REFERENCE RESOLUTION
# AND SAMPLE AT POINT LOCATIONS
# ================================================

print("\nSampling rasters at point locations...")
print("(This may take a few minutes)\n")

results = {}

for col_name, filename in rasters.items():
    filepath = os.path.join(data_dir, filename)

    try:
        with rasterio.open(filepath) as src:

            # If resolution differs from reference, resample
            if src.res != ref_res or src.crs != ref_crs:

                # Calculate new transform for reprojection
                transform, width, height = calculate_default_transform(
                    src.crs,
                    ref_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                    resolution=ref_res
                )

                # Read and reproject
                data = np.empty((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )

                # Sample resampled data at point locations
                # Convert coords to row/col indices
                sampled = []
                for x, y in coords:
                    row, col = rasterio.transform.rowcol(transform, x, y)
                    if 0 <= row < height and 0 <= col < width:
                        val = data[row, col]
                        sampled.append(float(val) if val != src.nodata else np.nan)
                    else:
                        sampled.append(np.nan)

            else:
                # Same CRS and resolution — sample directly
                sampled = [val[0] for val in src.sample(coords)]
                # Replace nodata with nan
                nodata = src.nodata
                if nodata is not None:
                    sampled = [np.nan if v == nodata else float(v)
                               for v in sampled]
                else:
                    sampled = [float(v) for v in sampled]

        results[col_name] = sampled
        print(f"  ✓ {col_name} ({filename}) — sampled {len(sampled)} values")

    except Exception as e:
        print(f"  ✗ {col_name} ({filename}) — ERROR: {e}")

# ================================================
# 5. BUILD DATAFRAME
# ================================================

print("\nBuilding dataframe...")

df = pd.DataFrame(results)
df['fire_label'] = gdf['Fire'].values

print(f"\nDataframe shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nLabel distribution:\n{df['fire_label'].value_counts()}")

# ================================================
# 6. CLEAN - DROP ROWS WITH NaN
# ================================================

before = len(df)
df = df.dropna()
after = len(df)
print(f"\nDropped {before - after} rows with NaN values")
print(f"Final dataset: {after} rows")

# ================================================
# 7. BASIC STATS CHECK
# ================================================

print("\nDescriptive statistics:")
print(df.describe().round(3))

# ================================================
# 8. SAVE CSV
# ================================================

df.to_csv(output_csv, index=False)
print(f"\nCSV saved to: {output_csv}")
print("Done. Ready for modeling.")