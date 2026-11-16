// Phase 3: burned-area maps per fire season, MODIS MCD64A1 (Collection 6.1).
// For each year 2002 to 2024 this exports two separate files:
//   burned_YYYY.tif     1 = burned between 1 July and 31 December,
//                       0 = not burned, -9999 = no usable label
//   burn_day_YYYY.tif   day of year the cell first burned that season,
//                       0 = not burned, -9999 = no usable label

// ---------- Settings ----------
var FIRST_YEAR = 2002;
var LAST_YEAR = 2024;
var SEASON_START_MONTH = 7; // July. The season runs to 31 December.
var NODATA = -9999;
var DRIVE_FOLDER = 'Trial7_Viphya';

// A "not burned" label needs good QA in at least this many of the 6 months.
// 0 = no QA requirement. Raise it (for example to 4) after checking the
// "Mean valid months" printout below.
var MIN_VALID_MONTHS = 0;

// Grid copied from static_stack.tif (layer properties in QGIS).
// If any value differs, change it here before running.
var CRS = 'EPSG:32736';
var CELL = 500;
var X0 = 575000;  // left edge
var Y1 = 8727500; // top edge
var NCOL = 76;
var NROW = 158;


var region = ee.FeatureCollection("projects/rainfallpatterns/assets/Chikangawa_forest");
var crsTransform = [CELL, 0, X0, 0, -CELL, Y1];

var mcd = ee.ImageCollection('MODIS/061/MCD64A1');

// ---------- Check: each season should have 6 monthly images ----------
var counts = ee.List.sequence(FIRST_YEAR, LAST_YEAR).map(function (y) {
  var start = ee.Date.fromYMD(y, SEASON_START_MONTH, 1);
  var end = ee.Date.fromYMD(ee.Number(y).add(1), 1, 1);
  return mcd.filterDate(start, end).size();
});
print('Images per season (expect 6 each):', counts);

// ---------- Labels for one fire season ----------
function seasonLayers(year) {
  var start = ee.Date.fromYMD(year, SEASON_START_MONTH, 1);
  var end = ee.Date.fromYMD(year + 1, 1, 1);
  var col = mcd.filterDate(start, end);

  // GEE can mask BurnDate = 0, so a masked value means "no burn recorded".
  // Fill it with 0 before testing for burns.
  var days = col.map(function (img) {
    return img.select('BurnDate').unmask(0);
  });

  // Burned if any month in the season has a burn day (1 to 366)
  var burnedAny = days.map(function (d) {
    return d.gt(0);
  }).max();

  // Earliest burn day in the season
  var burnDay = days.map(function (d) {
    return d.updateMask(d.gt(0));
  }).min();

  // Months with land and enough valid observations
  // (QA bit 0 = land, bit 1 = sufficient data)
  var validMonths = col.map(function (img) {
    return img.select('QA').unmask(0).bitwiseAnd(3).eq(3);
  }).sum();

  var usable = burnedAny.or(validMonths.gte(MIN_VALID_MONTHS));

  // Cells with no usable label get NODATA. Output grid uses the default
  // nearest-neighbour resampling from the 463 m sinusoidal source.
  var burned = burnedAny.updateMask(usable)
    .unmask(NODATA).toInt16().rename('burned');

  var burnDayOut = burnDay.unmask(0).updateMask(usable)
    .unmask(NODATA).toInt16().rename('burn_day');

  return {burned: burned, burnDay: burnDayOut, valid: validMonths};
}

// ---------- Export one file ----------
function exportFile(image, name) {
  Export.image.toDrive({
    image: image,
    description: name,
    fileNamePrefix: name,
    folder: DRIVE_FOLDER,
    region: region,
    crs: CRS,
    crsTransform: crsTransform,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
  });
}

// ---------- Build and export every season ----------
var check;

for (var y = FIRST_YEAR; y <= LAST_YEAR; y++) {
  var s = seasonLayers(y);

  exportFile(s.burned, 'burned_' + y);
  exportFile(s.burnDay, 'burn_day_' + y);

  // Keep 2015 for the console checks
  if (y === 2015) {
    check = s;
  }
}

// ---------- Checks in the console (2015) ----------
// Share of cells burned, expect roughly 0.05 to 0.30
print('Burned share 2015:', check.burned.eq(1).reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: region,
  scale: CELL,
  maxPixels: 1e9
}));

// Average number of months with good QA (6 = all good)
print('Mean valid months 2015:', check.valid.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: region,
  scale: CELL,
  maxPixels: 1e9
}));

Map.centerObject(region, 9);
Map.addLayer(check.burned.updateMask(check.burned.eq(1)), {palette: ['red']}, 'Burned 2015');