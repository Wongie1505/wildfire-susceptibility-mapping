Phase 11: base maps, ROC, summary pie, rainfall and NDVI means

Everything is put on the 30 m reference grid (Viphya_DEM.tif, EPSG:32736, 30.63 m).

data/ needs:
  Viphya_DEM.tif                  reference grid (included)
  Chikangawa_forest/              forest boundary (included)
  GHSL_built_2000.tif             settlements, 100 m (included)
  test_predictions.csv            Phase 7a test rows (included)
  class_validation.csv, area_table.csv   Phase 7c resampled outputs (included)
  roads.shp, rivers.shp           your local files (ADD THESE)
  rain_mean_nov_apr_2002_2024.tif   CHIRPS Nov to Apr mean (EXPORT: scripts/gee/)
  ndvi_dry_median_2002_2024.tif     MODIS NDVI dry-season median (EXPORT: scripts/gee/)

Run from the scripts folder:
  python s1_roads_rivers_settlements.py
  python s2_roc_and_pie.py
  python s3_rain_ndvi_means.py

GEE step: run scripts/gee/phase11_rain_ndvi_export.js in the Code Editor,
then move the two Drive exports from Phase11_exports into data/.
