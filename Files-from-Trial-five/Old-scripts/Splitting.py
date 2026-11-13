import geopandas as gpd
import os

# =========================
# 1. INPUT PATH
# =========================
input_gpkg = r"C:\Users\wongi\OneDrive\Desktop\Year 3\Sem 2\My Research\Project and thesis\Trial four\Data\Fire\Yearly_July_dec.gpkg"
output_folder = os.path.dirname(input_gpkg)

# =========================
# 2. READ DATA
# =========================
gdf = gpd.read_file(input_gpkg)

print("Total features:", len(gdf))
print("Columns:", list(gdf.columns))

# =========================
# 3. CHECK YEAR FIELD
# =========================
if 'year' not in gdf.columns:
    raise ValueError("The input file does not contain a 'year' column.")

years = sorted(gdf['year'].dropna().unique())

print("\nYear counts:")
for y in years:
    count = len(gdf[gdf['year'] == y])
    print(f"{int(y)}: {count} points")

# =========================
# 4. SPLIT AND SAVE
# =========================
for y in years:
    gdf_year = gdf[gdf['year'] == y].copy()

    output_gpkg = os.path.join(output_folder, f"Fire_{int(y)}.gpkg")

    # remove existing file if it already exists
    if os.path.exists(output_gpkg):
        os.remove(output_gpkg)

    gdf_year.to_file(
        output_gpkg,
        layer=f"Fire_{int(y)}",
        driver="GPKG"
    )

    print(f"Saved {output_gpkg} with {len(gdf_year)} points")

print("\nDone. Each year has been saved as a separate GeoPackage.")