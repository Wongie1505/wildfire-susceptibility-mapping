// Eight exports for the Viphya model.
// NDVI, NDMI, LST and rainfall, each for the training period 2002 to 2015
// and the test period 2016 to 2024.
// Exports are at native resolution. Resampling to the 30.63 m DEM grid is done in Python.

var roi = ee.FeatureCollection("projects/rainfallpatterns/assets/Chikangawa_forest");

var periods = [
  {name: '2002_2015', start: '2002-01-01', end: '2016-01-01'},
  {name: '2016_2024', start: '2016-01-01', end: '2025-01-01'}
];

// Keep only July to December days inside the period
function julyToDecember(collection, start, end) {
  return collection
    .filter(ee.Filter.calendarRange(7, 12, 'month'))
    .filter(ee.Filter.date(start, end));
}

// NDVI from MOD13A3, median of the July to December composites
function ndvi(start, end) {
  return julyToDecember(ee.ImageCollection('MODIS/061/MOD13A3'), start, end)
    .select('NDVI')
    .median()
    .multiply(0.0001)
    .rename('NDVI')
    .clip(roi);
}

// Keep clear-sky pixels only (StateQA bits 0 and 1 equal 00)
function maskClouds(image) {
  return image.updateMask(image.select('StateQA').bitwiseAnd(3).eq(0));
}

// NDMI from MOD09A1. Band 2 is NIR and band 6 is SWIR.
function ndmi(start, end) {
  return julyToDecember(ee.ImageCollection('MODIS/061/MOD09A1'), start, end)
    .map(maskClouds)
    .map(function (image) {
      var nir = image.select('sur_refl_b02').multiply(0.0001);
      var swir = image.select('sur_refl_b06').multiply(0.0001);
      return image.addBands(nir.subtract(swir).divide(nir.add(swir)).rename('NDMI'));
    })
    .select('NDMI')
    .median()
    .rename('NDMI')
    .clip(roi);
}

// Daytime LST in Celsius from MOD11A2. Zero means no data, so it is masked
// before the median; otherwise the scaled zero pulls the median down.
function lst(start, end) {
  return julyToDecember(ee.ImageCollection('MODIS/061/MOD11A2'), start, end)
    .select('LST_Day_1km')
    .map(function (image) {
      return image.updateMask(image.gt(0));
    })
    .median()
    .multiply(0.02)
    .subtract(273.15)
    .rename('LST')
    .clip(roi);
}

// Seasonal rainfall from CHIRPS daily: mean daily rainfall times 184 days (July to December)
function rainfall(start, end) {
  return julyToDecember(ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY'), start, end)
    .select('precipitation')
    .mean()
    .multiply(184)
    .rename('Rainfall')
    .clip(roi);
}

var layers = [
  {name: 'ndvi', build: ndvi, scale: 500},
  {name: 'ndmi', build: ndmi, scale: 500},
  {name: 'lst', build: lst, scale: 1000},
  {name: 'rainfall', build: rainfall, scale: 5566}
];

layers.forEach(function (layer) {
  periods.forEach(function (period) {
    var fileName = 'viphya_' + layer.name + '_' + period.name;
    Export.image.toDrive({
      image: layer.build(period.start, period.end),
      description: fileName,
      folder: 'Viphya_Data',
      fileNamePrefix: fileName,
      scale: layer.scale,
      region: roi,
      crs: 'EPSG:32736',
      maxPixels: 1e10
    });
  });
});

print('Eight exports queued. Run them from the Tasks tab.');