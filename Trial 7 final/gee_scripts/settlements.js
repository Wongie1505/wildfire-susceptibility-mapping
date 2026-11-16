// Phase 2: GHSL built-up surface, 2000 epoch, for the settlement distance layer.
// Region = forest boundary plus about 30 km, so the nearest settlement
// is found even when it sits outside the forest.


var region = ee.FeatureCollection("projects/rainfallpatterns/assets/Chikangawa_forest");


// Built surface in square metres per 100 m cell
var built = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2000')
  .select('built_surface')
  .clip(region);

Map.centerObject(region, 9);
Map.addLayer(built, {min: 0, max: 5000}, 'Built surface 2000');

// Export in the same CRS as the DEM
Export.image.toDrive({
  image: built,
  description: 'GHSL_built_2000',
  folder:'Trial7_viphya',
  fileNamePrefix: 'GHSL_built_2000',
  region: region,
  crs: 'EPSG:32736',
  scale: 100,
  maxPixels: 1e9
});