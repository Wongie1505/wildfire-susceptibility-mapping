// Phase 11: GEE exports for the rainfall mean and dry-season NDVI median.
//
// Rainfall: mean of the Nov to Apr CHIRPS totals, seasons 2002 to 2024.
//           Season Y runs 1 Nov (Y-1) to 30 Apr (Y).
// NDVI:     median of clear MOD09A1 NDVI over the dry season, 1 May to
//           31 Oct, for 2002 to 2024. One image for the whole period.
//
// Both are exported in EPSG:32736 over the DEM extent. Resampling to 30 m
// is done locally in Python, so the exports keep their native resolution.

// ---------- Settings ----------
var CRS = 'EPSG:32736';
var X0 = 575095.8277;
var Y1 = 8727438.3706;
var X1 = 612867.7957;
var Y0 = 8648922.9173;
var REGION = ee.Geometry.Rectangle([X0, Y0, X1, Y1], CRS, false);

var FIRST_YEAR = 2002;
var LAST_YEAR = 2024;
var EXPORT_FOLDER = 'Phase11_exports';

// ---------- Rainfall: mean Nov to Apr total ----------
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .select('precipitation');

function seasonTotal(year) {
  var start = ee.Date.fromYMD(year - 1, 11, 1);
  var end = ee.Date.fromYMD(year, 5, 1);

  return chirps
    .filterDate(start, end)
    .sum()
    .set('year', year);
}

var seasons = ee.ImageCollection(
  ee.List.sequence(FIRST_YEAR, LAST_YEAR).map(seasonTotal)
);

var rainMean = seasons.mean()
  .rename('rain_mm')
  .toFloat()
  .clip(REGION);

Export.image.toDrive({
  image: rainMean,
  description: 'rain_mean_nov_apr_2002_2024',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'rain_mean_nov_apr_2002_2024',
  region: REGION,
  crs: CRS,
  scale: 5566,
  maxPixels: 1e9
});

// ---------- NDVI: clear dry-season median ----------
// MOD09A1 StateQA: keep clear pixels only (cloud state, cloud shadow, internal cloud flag)
function maskClear(img) {
  var qa = img.select('StateQA');
  var cloud = qa.bitwiseAnd(3);

  var ok = cloud.eq(0).or(cloud.eq(3))
    .and(qa.bitwiseAnd(4).eq(0))
    .and(qa.rightShift(10).bitwiseAnd(1).eq(0));

  return img.updateMask(ok);
}

// NDVI = (NIR - Red) / (NIR + Red), with bands 2 and 1 scaled by 0.0001
function addNdvi(img) {
  var red = img.select('sur_refl_b01').multiply(0.0001);
  var nir = img.select('sur_refl_b02').multiply(0.0001);

  return nir.subtract(red)
    .divide(nir.add(red))
    .clamp(-1, 1)
    .rename('ndvi')
    .copyProperties(img, ['system:time_start']);
}

var dryNdvi = ee.ImageCollection('MODIS/061/MOD09A1')
  .filter(ee.Filter.calendarRange(5, 10, 'month'))
  .filterDate(
    ee.Date.fromYMD(FIRST_YEAR, 1, 1),
    ee.Date.fromYMD(LAST_YEAR + 1, 1, 1)
  )
  .map(maskClear)
  .map(addNdvi);

var ndviMedian = dryNdvi.median()
  .toFloat()
  .clip(REGION);

Export.image.toDrive({
  image: ndviMedian,
  description: 'ndvi_dry_median_2002_2024',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'ndvi_dry_median_2002_2024',
  region: REGION,
  crs: CRS,
  scale: 500,
  maxPixels: 1e9
});
