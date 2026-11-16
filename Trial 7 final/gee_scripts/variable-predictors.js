// Phase 4: seasonal predictors for every fire season 2002 to 2024.
// One file per predictor per year, all on the Phase 2 grid (500 m, EPSG:32736).
// Everything is measured BEFORE 1 July of the season it predicts.
//
//   rain_anomaly_YYYY   CHIRPS rain, 1 Nov (Y-1) to 30 Apr (Y), % of normal
//   ndvi_peak_YYYY      highest MODIS NDVI (no cloud mask), 1 Dec (Y-1) to 30 Apr (Y)
//   ndmi_anomaly_YYYY   mean MODIS NDMI, 1 May to 30 Jun (Y), minus normal
//   lst_anomaly_YYYY    mean MODIS day LST (deg C), 1 May to 30 Jun (Y), minus normal
//   lulc_YYYY           MODIS land cover (IGBP) of the PREVIOUS year (Y-1)
//   burned_2001         burned map of the 2001 season, for fire history in 2002
//
// "Normal" = average over the training years only (2002 to 2015).

// ---------- Settings ----------
var FIRST_YEAR = 2002;
var LAST_YEAR = 2024;
var TRAIN_FIRST = 2002;
var TRAIN_LAST = 2015;
var NODATA = -9999;

// Grid copied from static_stack.tif (same values as the Phase 3 script)
var CRS = 'EPSG:32736';
var CELL = 500;
var X0 = 575000;
var Y1 = 8727500;
var NCOL = 76;
var NROW = 158;

var region = ee.FeatureCollection("projects/rainfallpatterns/assets/Chikangawa_forest");
var crsTransform = [CELL, 0, X0, 0, -CELL, Y1];

// ---------- Source data ----------
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').select('precipitation');

// MOD09A1: keep clear pixels only (cloud state, cloud shadow, internal cloud flag)
function maskClear(img) {
  var qa = img.select('StateQA');
  var cloud = qa.bitwiseAnd(3);

  var ok = cloud.eq(0).or(cloud.eq(3))
    .and(qa.bitwiseAnd(4).eq(0))
    .and(qa.rightShift(10).bitwiseAnd(1).eq(0));

  return img.updateMask(ok);
}

// NDVI = (NIR - Red) / (NIR + Red), NDMI = (NIR - SWIR1) / (NIR + SWIR1)
function addIndices(img) {
  var red = img.select('sur_refl_b01').multiply(0.0001);
  var nir = img.select('sur_refl_b02').multiply(0.0001);
  var swir = img.select('sur_refl_b06').multiply(0.0001);

  // Clamp to the valid range, since a slightly negative reflectance can push an index past 1
  return ee.Image.cat([
    nir.subtract(red).divide(nir.add(red)).clamp(-1, 1).rename('ndvi'),
    nir.subtract(swir).divide(nir.add(swir)).clamp(-1, 1).rename('ndmi')
  ]).copyProperties(img, ['system:time_start']);
}

// Clear pixels only, used for NDMI
var mod09 = ee.ImageCollection('MODIS/061/MOD09A1')
  .map(maskClear)
  .map(addIndices);

// No cloud mask, used for the NDVI peak. Clouds lower NDVI, so the maximum
// over many 8-day composites already skips them. Masking left gaps in the rainy season.
var mod09All = ee.ImageCollection('MODIS/061/MOD09A1')
  .map(addIndices);

// MOD11A2 day LST: keep good and other-quality pixels, convert to deg C
var mod11 = ee.ImageCollection('MODIS/061/MOD11A2').map(function (img) {
  var good = img.select('QC_Day').bitwiseAnd(3).lte(1);

  return img.select('LST_Day_1km')
    .updateMask(good)
    .multiply(0.02)
    .subtract(273.15)
    .rename('lst')
    .copyProperties(img, ['system:time_start']);
});

var landCover = ee.ImageCollection('MODIS/061/MCD12Q1').select('LC_Type1');
var mcd64 = ee.ImageCollection('MODIS/061/MCD64A1');

// ---------- One measurement per season (year y) ----------
function rainTotal(y) {
  return chirps
    .filterDate(ee.Date.fromYMD(y - 1, 11, 1), ee.Date.fromYMD(y, 5, 1))
    .sum();
}

function ndviPeak(y) {
  return mod09All
    .filterDate(ee.Date.fromYMD(y - 1, 12, 1), ee.Date.fromYMD(y, 5, 1))
    .select('ndvi')
    .max();
}

function ndmiMean(y) {
  // 1. Clear observations in May and June
  var preferred = mod09
    .filterDate(ee.Date.fromYMD(y, 5, 1), ee.Date.fromYMD(y, 7, 1))
    .select('ndmi')
    .mean();

  // 2. Where none exist, clear observations from April to June
  var wider = mod09
    .filterDate(ee.Date.fromYMD(y, 4, 1), ee.Date.fromYMD(y, 7, 1))
    .select('ndmi')
    .mean();

  // 3. Last resort, May and June without the cloud mask
  var unmasked = mod09All
    .filterDate(ee.Date.fromYMD(y, 5, 1), ee.Date.fromYMD(y, 7, 1))
    .select('ndmi')
    .mean();

  return preferred.unmask(wider).unmask(unmasked);
}

function lstMean(y) {
  return mod11
    .filterDate(ee.Date.fromYMD(y, 5, 1), ee.Date.fromYMD(y, 7, 1))
    .mean();
}

// Land cover of the previous year, so the map is never shaped by this season's fires
function lulcPrevYear(y) {
  return ee.Image(landCover
    .filterDate(ee.Date.fromYMD(y - 1, 1, 1), ee.Date.fromYMD(y, 1, 1))
    .first());
}

// ---------- Normals from the training years only ----------
function normal(measure) {
  var images = [];

  for (var y = TRAIN_FIRST; y <= TRAIN_LAST; y++) {
    images.push(measure(y));
  }

  return ee.ImageCollection.fromImages(images).mean();
}

var rainNormal = normal(rainTotal);
var ndmiNormal = normal(ndmiMean);
var lstNormal = normal(lstMean);

// ---------- Anomalies ----------
function rainAnomaly(y) {
  return rainTotal(y).divide(rainNormal).multiply(100).rename('rain_anomaly');
}

function ndmiAnomaly(y) {
  return ndmiMean(y).subtract(ndmiNormal).rename('ndmi_anomaly');
}

function lstAnomaly(y) {
  return lstMean(y).subtract(lstNormal).rename('lst_anomaly');
}

// ---------- Export one file ----------
// Default nearest-neighbour resampling puts every source onto the 500 m grid.
// CHIRPS (about 5 km) and LST (1 km) keep their coarse values, repeated.
function exportFile(image, name, folder) {
  Export.image.toDrive({
    image: image.unmask(NODATA),
    description: name,
    fileNamePrefix: name,
    folder: folder,
    region: region,
    crs: CRS,
    crsTransform: crsTransform,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
  });
}

// ---------- Every season ----------
for (var y = FIRST_YEAR; y <= LAST_YEAR; y++) {
  exportFile(rainAnomaly(y).toFloat(), 'rain_anomaly_' + y, 'P4_rain_anomaly');
  exportFile(ndviPeak(y).rename('ndvi_peak').toFloat(), 'ndvi_peak_' + y, 'P4_ndvi_peak');
  exportFile(ndmiAnomaly(y).toFloat(), 'ndmi_anomaly_' + y, 'P4_ndmi_anomaly');
  exportFile(lstAnomaly(y).toFloat(), 'lst_anomaly_' + y, 'P4_lst_anomaly');
  exportFile(lulcPrevYear(y).rename('lulc').toInt16(), 'lulc_' + y, 'P4_lulc');
}

// ---------- Burned map of the 2001 season (same rule as Phase 3) ----------
var days2001 = mcd64
  .filterDate('2001-07-01', '2002-01-01')
  .map(function (img) {
    return img.select('BurnDate').unmask(0);
  });

var burned2001 = days2001
  .map(function (d) {
    return d.gt(0);
  })
  .max()
  .rename('burned')
  .toInt16();

exportFile(burned2001, 'burned_2001', 'P4_fire_history');

// ---------- Checks in the console ----------
// Averages over the forest area. Expect rain_anomaly near 100 in training
// years, ndvi_peak between 0.4 and 0.9, ndmi_anomaly and lst_anomaly near 0.
function checkYear(y) {
  var all = ee.Image.cat([
    rainAnomaly(y), ndviPeak(y).rename('ndvi_peak'),
    ndmiAnomaly(y), lstAnomaly(y)
  ]);

  print('Mean over region, ' + y + ':', all.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: CELL,
    maxPixels: 1e9
  }));
}

checkYear(2010);
checkYear(2020);