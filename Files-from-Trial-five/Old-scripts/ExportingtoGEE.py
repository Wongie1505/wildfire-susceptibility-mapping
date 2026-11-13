# """
# =============================================================
#  GEE MONTHLY BATCH EXPORT - Wildfire Susceptibility Project
#  Variables : NDVI, NDMI, RAIN, TEMP_MEAN
#  Period    : July-December, 2002-2025
#  Resolution: 30m
#  CRS       : EPSG:32736 (Malawi UTM Zone 36S)
#  Folder    : project_trial_four

#  Logic is 1:1 translation of confirmed working GEE JS scripts.
#  Nothing changed except Python syntax + batch submission.

#  SETUP (run once in terminal):
#    pip install earthengine-api
#    earthengine authenticate

#  Then update the two placeholders below and run.
# =============================================================
# """

# import ee
# import time
# import logging
# from datetime import datetime

# # -------------------------------------------------------
# # LOGGING
# # -------------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s  %(levelname)s  %(message)s',
#     datefmt='%H:%M:%S'
# )
# log = logging.getLogger(__name__)

# # -------------------------------------------------------
# # 0. AUTHENTICATE & INITIALISE
# #    Uncomment ee.Authenticate() on first run only.
# # -------------------------------------------------------
# # ee.Authenticate()
# ee.Initialize(project='rainfallpatterns')    # <-- REPLACE

# # -------------------------------------------------------
# # 1. ROI
# #    Paste your GEE asset path here after uploading SHP.
# #    e.g. 'projects/ee-yourname/assets/Chikangawa_forest'
# # -------------------------------------------------------
# ROI_ASSET = 'projects/rainfallpatterns/assets/Chikangawa_forest' 

# roi = ee.FeatureCollection(ROI_ASSET)

# # -------------------------------------------------------
# # 2. SETTINGS  (identical to all four JS scripts)
# # -------------------------------------------------------
# START_YEAR   = 2002
# END_YEAR     = 2002
# MONTHS       = [7]
# DRIVE_FOLDER = 'project_trial_four'
# EXPORT_CRS   = 'EPSG:32736'
# EXPORT_SCALE = 30

# # Retry / delay settings
# MAX_RETRIES       = 5
# RETRY_DELAY_S     = 15
# TASK_START_WAIT_S = 3

# # -------------------------------------------------------
# # 3. DATASETS  (same as JS scripts)
# # -------------------------------------------------------
# lt5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
# le7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
# lc8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
# lc9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')

# chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
#            .filterBounds(roi) \
#            .select('precipitation')

# modis_lst = ee.ImageCollection('MODIS/061/MOD11A2') \
#               .filterBounds(roi) \
#               .select(['LST_Day_1km', 'LST_Night_1km'])

# # -------------------------------------------------------
# # 4. SHARED HELPERS
# # -------------------------------------------------------

# # -- resampleTo30m (identical to all JS scripts) --
# def resample_to_30m(image):
#     return (image
#             .resample('bilinear')
#             .reproject(crs=EXPORT_CRS, scale=EXPORT_SCALE))

# # -- maskAndScaleLandsat (identical to NDVI/NDMI JS scripts) --
# def mask_and_scale_landsat(image):
#     qa     = image.select('QA_PIXEL')
#     radsat = image.select('QA_RADSAT')
#     mask = (qa.bitwiseAnd(1 << 1).eq(0)
#               .And(qa.bitwiseAnd(1 << 2).eq(0))
#               .And(qa.bitwiseAnd(1 << 3).eq(0))
#               .And(qa.bitwiseAnd(1 << 4).eq(0))
#               .And(qa.bitwiseAnd(1 << 5).eq(0))
#               .And(radsat.eq(0)))
#     scaled = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
#     return image.addBands(scaled, None, True).updateMask(mask)

# # -- NDVI band functions --
# def add_ndvi_l57(image):
#     return image.addBands(
#         image.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI'))

# def add_ndvi_l89(image):
#     return image.addBands(
#         image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI'))

# # -- NDMI band functions --
# def add_ndmi_l57(image):
#     return image.addBands(
#         image.normalizedDifference(['SR_B4', 'SR_B5']).rename('NDMI'))

# def add_ndmi_l89(image):
#     return image.addBands(
#         image.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDMI'))

# # -- Merge all Landsat sensors (done once) --
# all_ndvi = (lt5.filterBounds(roi).map(mask_and_scale_landsat).map(add_ndvi_l57).select('NDVI')
#             .merge(le7.filterBounds(roi).map(mask_and_scale_landsat).map(add_ndvi_l57).select('NDVI'))
#             .merge(lc8.filterBounds(roi).map(mask_and_scale_landsat).map(add_ndvi_l89).select('NDVI'))
#             .merge(lc9.filterBounds(roi).map(mask_and_scale_landsat).map(add_ndvi_l89).select('NDVI')))

# all_ndmi = (lt5.filterBounds(roi).map(mask_and_scale_landsat).map(add_ndmi_l57).select('NDMI')
#             .merge(le7.filterBounds(roi).map(mask_and_scale_landsat).map(add_ndmi_l57).select('NDMI'))
#             .merge(lc8.filterBounds(roi).map(mask_and_scale_landsat).map(add_ndmi_l89).select('NDMI'))
#             .merge(lc9.filterBounds(roi).map(mask_and_scale_landsat).map(add_ndmi_l89).select('NDMI')))

# # -------------------------------------------------------
# # 5. MONTHLY PRODUCT FUNCTIONS
# #    Direct 1:1 translation of confirmed working JS logic
# # -------------------------------------------------------

# def make_month_ndvi(year, month):
#     """Exact translation of makeMonthNdvi() from working JS."""
#     monthly = (all_ndvi
#                .filter(ee.Filter.calendarRange(year, year, 'year'))
#                .filter(ee.Filter.calendarRange(month, month, 'month')))

#     ndvi = ee.Image(ee.Algorithms.If(
#         monthly.size().gt(0),
#         monthly.median().rename('NDVI').clip(roi),
#         ee.Image.constant(0).rename('NDVI').clip(roi).updateMask(ee.Image.constant(0))
#     ))
#     return ndvi


# def make_month_ndmi(year, month):
#     """Exact translation of makeMonthNdmi() from working JS."""
#     monthly = (all_ndmi
#                .filter(ee.Filter.calendarRange(year, year, 'year'))
#                .filter(ee.Filter.calendarRange(month, month, 'month')))

#     ndmi = ee.Image(ee.Algorithms.If(
#         monthly.size().gt(0),
#         monthly.median().rename('NDMI').clip(roi),
#         ee.Image.constant(0).rename('NDMI').clip(roi).updateMask(ee.Image.constant(0))
#     ))
#     return ndmi


# def make_month_rain(year, month):
#     """Exact translation of makeMonthRain() from working JS."""
#     monthly = (chirps
#                .filter(ee.Filter.calendarRange(year, year, 'year'))
#                .filter(ee.Filter.calendarRange(month, month, 'month')))

#     rain = ee.Image(ee.Algorithms.If(
#         monthly.size().gt(0),
#         monthly.sum().rename('RAIN'),
#         ee.Image.constant(-9999).rename('RAIN').toFloat()
#     ))
#     return resample_to_30m(rain.toFloat()).clip(roi)


# def make_month_temp(year, month):
#     """Exact translation of makeMonthTemp() from working JS."""
#     def convert_lst(img):
#         day_c   = img.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('LST_Day_C')
#         night_c = img.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('LST_Night_C')
#         return (day_c.add(night_c).divide(2)
#                 .rename('LST_Mean_C')
#                 .copyProperties(img, img.propertyNames()))

#     monthly = (modis_lst
#                .filter(ee.Filter.calendarRange(year, year, 'year'))
#                .filter(ee.Filter.calendarRange(month, month, 'month'))
#                .map(convert_lst))

#     temp = ee.Image(ee.Algorithms.If(
#         monthly.size().gt(0),
#         monthly.select('LST_Mean_C').mean().rename('TEMP_MEAN'),
#         ee.Image.constant(-9999).rename('TEMP_MEAN').toFloat()
#     ))
#     return resample_to_30m(temp.toFloat()).clip(roi)

# # -------------------------------------------------------
# # 6. ROBUST TASK SUBMISSION WITH RETRY
# # -------------------------------------------------------
# def submit_task(image, label):
#     task = ee.batch.Export.image.toDrive(
#         image          = image,
#         description    = label,
#         folder         = DRIVE_FOLDER,
#         fileNamePrefix = label,
#         region         = roi.geometry(),
#         crs            = EXPORT_CRS,
#         scale          = EXPORT_SCALE,
#         maxPixels      = int(1e13),
#         fileFormat     = 'GeoTIFF'
#     )

#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             task.start()
#             log.info(f'  ✓ Submitted  {label}  (attempt {attempt})')
#             return True
#         except ee.EEException as e:
#             log.warning(f'  ✗ EE error [{label}] attempt {attempt}: {e}')
#         except Exception as e:
#             log.warning(f'  ✗ Error [{label}] attempt {attempt}: {e}')

#         if attempt < MAX_RETRIES:
#             wait = RETRY_DELAY_S * attempt
#             log.info(f'    Retrying in {wait}s ...')
#             time.sleep(wait)

#     log.error(f'  ✗✗ FAILED after {MAX_RETRIES} attempts: {label}')
#     return False

# # -------------------------------------------------------
# # 7. MAIN EXPORT LOOP
# #    Mirrors the for loops in all four JS scripts
# # -------------------------------------------------------
# def run_exports():
#     total     = (END_YEAR - START_YEAR + 1) * len(MONTHS) * 4
#     submitted = 0
#     failed    = []

#     log.info('=' * 65)
#     log.info(f'Monthly GEE Export  |  {total} tasks  |  {START_YEAR}-{END_YEAR}')
#     log.info(f'Drive folder : {DRIVE_FOLDER}')
#     log.info(f'Resolution   : {EXPORT_SCALE}m  |  CRS: {EXPORT_CRS}')
#     log.info('=' * 65)

#     for y in range(START_YEAR, END_YEAR + 1):
#         for m in MONTHS:
#             mm = f'{m:02d}'
#             log.info(f'\n-- {y}-{mm} ------------------------------------------')

#             # Define all four variables for this month
#             # (label, build_function)
#             var_defs = [
#                 (f'NDVI_{y}_{mm}',      make_month_ndvi),
#                 (f'NDMI_{y}_{mm}',      make_month_ndmi),
#                 (f'RAIN_{y}_{mm}',      make_month_rain),
#                 (f'TEMP_MEAN_{y}_{mm}', make_month_temp),
#             ]

#             for label, func in var_defs:
#                 try:
#                     image = func(y, m)
#                 except Exception as e:
#                     log.error(f'  Build failed [{label}]: {e}')
#                     failed.append(label)
#                     continue

#                 ok = submit_task(image, label)
#                 if ok:
#                     submitted += 1
#                 else:
#                     failed.append(label)

#                 time.sleep(TASK_START_WAIT_S)

#     # Summary
#     log.info('\n' + '=' * 65)
#     log.info(f'DONE.  {submitted}/{total} tasks submitted.')
#     if failed:
#         log.warning(f'Failed ({len(failed)} tasks):')
#         for f in failed:
#             log.warning(f'  * {f}')
#     else:
#         log.info('All tasks submitted successfully ✓')
#     log.info('Monitor: https://code.earthengine.google.com/tasks')
#     log.info('=' * 65)
#     return submitted, failed

# # -------------------------------------------------------
# # ENTRY POINT
# # -------------------------------------------------------
# if __name__ == '__main__':
#     start_time = datetime.now()
#     submitted, failed = run_exports()
#     elapsed = datetime.now() - start_time
#     log.info(f'Total submission time: {elapsed}')




"""
=============================================================
 GEE MONTHLY BATCH EXPORT - Wildfire Susceptibility Project
 Variables : NDVI, NDMI, RAIN, TEMP_MEAN
 Period    : July-December, 2002-2025
 Resolution: 30m
 CRS       : EPSG:32736 (Malawi UTM Zone 36S)
 Folder    : project_trial_four

 Logic is 1:1 translation of confirmed working GEE JS scripts.
 Added preprocessing for data integrity (ML-safe outputs):
   1. Scene-level cloud cover filter (<= 70%) — drops heavily
      cloudy scenes entirely before pixel-level masking.
      This is what prevents white blobs in monthly composites.
   2. Pixel-level QA mask unchanged (your working JS logic).
   3. NDVI/NDMI valid range clamp [-1, 1] — removes any
      residual edge artifacts from scaling.
 RAIN and TEMP unchanged — no clouds affect those sources.

 SETUP (run once in terminal):
   pip install earthengine-api
   earthengine authenticate

 Then update the two placeholders below and run.
=============================================================
"""

import ee
import time
import logging
from datetime import datetime

# -------------------------------------------------------
# LOGGING
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# -------------------------------------------------------
# 0. AUTHENTICATE & INITIALISE
#    Uncomment ee.Authenticate() on first run only.
# -------------------------------------------------------
# ee.Authenticate()
ee.Initialize(project='rainfallpatterns')    # <-- REPLACE

# -------------------------------------------------------
# 1. ROI
#    Paste your GEE asset path here after uploading SHP.
#    e.g. 'projects/ee-yourname/assets/Chikangawa_forest'
# -------------------------------------------------------
ROI_ASSET = 'projects/rainfallpatterns/assets/Chikangawa_forest' # <-- REPLACE

roi = ee.FeatureCollection(ROI_ASSET)

# -------------------------------------------------------
# 2. SETTINGS  (identical to all four JS scripts)
# -------------------------------------------------------
START_YEAR   = 2002
END_YEAR     = 2025
MONTHS       = [7,8,9,1,11,12]
DRIVE_FOLDER = 'project_trial_four'
EXPORT_CRS   = 'EPSG:32736'
EXPORT_SCALE = 30

# Retry / delay settings
MAX_RETRIES       = 5
RETRY_DELAY_S     = 15
TASK_START_WAIT_S = 3

# -------------------------------------------------------
# 3. DATASETS  (same as JS scripts)
# -------------------------------------------------------
lt5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
le7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
lc8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
lc9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')

chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
           .filterBounds(roi) \
           .select('precipitation')

modis_lst = ee.ImageCollection('MODIS/061/MOD11A2') \
              .filterBounds(roi) \
              .select(['LST_Day_1km', 'LST_Night_1km'])

# -------------------------------------------------------
# 4. SHARED HELPERS
# -------------------------------------------------------

# -- resampleTo30m (identical to all JS scripts) --
def resample_to_30m(image):
    return (image
            .resample('bilinear')
            .reproject(crs=EXPORT_CRS, scale=EXPORT_SCALE))

# -------------------------------------------------------
# PREPROCESSING ADDITION 1: Scene-level cloud cover filter
# Drops any Landsat scene where >70% of the whole image
# is cloud-covered BEFORE pixel-level masking happens.
# This is the primary fix for white blobs — a scene that
# is 80-90% cloud will still produce white areas even after
# pixel masking, because not enough clear pixels exist for
# a clean median. 70% is a safe threshold for Malawi's
# Jul-Dec dry/transition season — tighten to 50% if blobs
# persist on early wet season months (Nov-Dec).
# -------------------------------------------------------
CLOUD_COVER_MAX = 70   # percent — scenes above this are dropped entirely

def filter_cloud_cover(collection):
    """Filter collection to scenes with cloud cover <= CLOUD_COVER_MAX."""
    return collection.filter(
        ee.Filter.lte('CLOUD_COVER', CLOUD_COVER_MAX)
    )

# -- maskAndScaleLandsat (your working JS logic, unchanged) --
def mask_and_scale_landsat(image):
    qa     = image.select('QA_PIXEL')
    radsat = image.select('QA_RADSAT')
    mask = (qa.bitwiseAnd(1 << 1).eq(0)
              .And(qa.bitwiseAnd(1 << 2).eq(0))
              .And(qa.bitwiseAnd(1 << 3).eq(0))
              .And(qa.bitwiseAnd(1 << 4).eq(0))
              .And(qa.bitwiseAnd(1 << 5).eq(0))
              .And(radsat.eq(0)))
    scaled = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    return image.addBands(scaled, None, True).updateMask(mask)

# -- NDVI band functions --
def add_ndvi_l57(image):
    return image.addBands(
        image.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI'))

def add_ndvi_l89(image):
    return image.addBands(
        image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI'))

# -- NDMI band functions --
def add_ndmi_l57(image):
    return image.addBands(
        image.normalizedDifference(['SR_B4', 'SR_B5']).rename('NDMI'))

def add_ndmi_l89(image):
    return image.addBands(
        image.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDMI'))

# -- Merge all Landsat sensors with cloud cover filter applied (done once) --
# filter_cloud_cover() is the only addition vs your working JS
all_ndvi = (filter_cloud_cover(lt5).filterBounds(roi).map(mask_and_scale_landsat).map(add_ndvi_l57).select('NDVI')
            .merge(filter_cloud_cover(le7).filterBounds(roi).map(mask_and_scale_landsat).map(add_ndvi_l57).select('NDVI'))
            .merge(filter_cloud_cover(lc8).filterBounds(roi).map(mask_and_scale_landsat).map(add_ndvi_l89).select('NDVI'))
            .merge(filter_cloud_cover(lc9).filterBounds(roi).map(mask_and_scale_landsat).map(add_ndvi_l89).select('NDVI')))

all_ndmi = (filter_cloud_cover(lt5).filterBounds(roi).map(mask_and_scale_landsat).map(add_ndmi_l57).select('NDMI')
            .merge(filter_cloud_cover(le7).filterBounds(roi).map(mask_and_scale_landsat).map(add_ndmi_l57).select('NDMI'))
            .merge(filter_cloud_cover(lc8).filterBounds(roi).map(mask_and_scale_landsat).map(add_ndmi_l89).select('NDMI'))
            .merge(filter_cloud_cover(lc9).filterBounds(roi).map(mask_and_scale_landsat).map(add_ndmi_l89).select('NDMI')))

# -------------------------------------------------------
# 5. MONTHLY PRODUCT FUNCTIONS
#    Direct 1:1 translation of confirmed working JS logic
# -------------------------------------------------------

def make_month_ndvi(year, month):
    """
    makeMonthNdvi() + scene cloud filter + valid range clamp.
    Clamp to [-1, 1] removes residual scaling artifacts for ML.
    """
    monthly = (all_ndvi
               .filter(ee.Filter.calendarRange(year, year, 'year'))
               .filter(ee.Filter.calendarRange(month, month, 'month')))

    ndvi = ee.Image(ee.Algorithms.If(
        monthly.size().gt(0),
        monthly.median().rename('NDVI').clip(roi),
        ee.Image.constant(0).rename('NDVI').clip(roi).updateMask(ee.Image.constant(0))
    ))

    # PREPROCESSING ADDITION 2: clamp to valid NDVI range
    return ndvi.clamp(-1, 1)


def make_month_ndmi(year, month):
    """
    makeMonthNdmi() + scene cloud filter + valid range clamp.
    Clamp to [-1, 1] removes residual scaling artifacts for ML.
    """
    monthly = (all_ndmi
               .filter(ee.Filter.calendarRange(year, year, 'year'))
               .filter(ee.Filter.calendarRange(month, month, 'month')))

    ndmi = ee.Image(ee.Algorithms.If(
        monthly.size().gt(0),
        monthly.median().rename('NDMI').clip(roi),
        ee.Image.constant(0).rename('NDMI').clip(roi).updateMask(ee.Image.constant(0))
    ))

    # PREPROCESSING ADDITION 2: clamp to valid NDMI range
    return ndmi.clamp(-1, 1)


def make_month_rain(year, month):
    """Exact translation of makeMonthRain() from working JS."""
    monthly = (chirps
               .filter(ee.Filter.calendarRange(year, year, 'year'))
               .filter(ee.Filter.calendarRange(month, month, 'month')))

    rain = ee.Image(ee.Algorithms.If(
        monthly.size().gt(0),
        monthly.sum().rename('RAIN'),
        ee.Image.constant(-9999).rename('RAIN').toFloat()
    ))
    return resample_to_30m(rain.toFloat()).clip(roi)


def make_month_temp(year, month):
    """Exact translation of makeMonthTemp() from working JS."""
    def convert_lst(img):
        day_c   = img.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('LST_Day_C')
        night_c = img.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('LST_Night_C')
        return (day_c.add(night_c).divide(2)
                .rename('LST_Mean_C')
                .copyProperties(img, img.propertyNames()))

    monthly = (modis_lst
               .filter(ee.Filter.calendarRange(year, year, 'year'))
               .filter(ee.Filter.calendarRange(month, month, 'month'))
               .map(convert_lst))

    temp = ee.Image(ee.Algorithms.If(
        monthly.size().gt(0),
        monthly.select('LST_Mean_C').mean().rename('TEMP_MEAN'),
        ee.Image.constant(-9999).rename('TEMP_MEAN').toFloat()
    ))
    return resample_to_30m(temp.toFloat()).clip(roi)

# -------------------------------------------------------
# 6. ROBUST TASK SUBMISSION WITH RETRY
# -------------------------------------------------------
def submit_task(image, label):
    task = ee.batch.Export.image.toDrive(
        image          = image,
        description    = label,
        folder         = DRIVE_FOLDER,
        fileNamePrefix = label,
        region         = roi.geometry(),
        crs            = EXPORT_CRS,
        scale          = EXPORT_SCALE,
        maxPixels      = int(1e13),
        fileFormat     = 'GeoTIFF'
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            task.start()
            log.info(f'  ✓ Submitted  {label}  (attempt {attempt})')
            return True
        except ee.EEException as e:
            log.warning(f'  ✗ EE error [{label}] attempt {attempt}: {e}')
        except Exception as e:
            log.warning(f'  ✗ Error [{label}] attempt {attempt}: {e}')

        if attempt < MAX_RETRIES:
            wait = RETRY_DELAY_S * attempt
            log.info(f'    Retrying in {wait}s ...')
            time.sleep(wait)

    log.error(f'  ✗✗ FAILED after {MAX_RETRIES} attempts: {label}')
    return False

# -------------------------------------------------------
# 7. MAIN EXPORT LOOP
#    Mirrors the for loops in all four JS scripts
# -------------------------------------------------------
def run_exports():
    total     = (END_YEAR - START_YEAR + 1) * len(MONTHS) * 4
    submitted = 0
    failed    = []

    log.info('=' * 65)
    log.info(f'Monthly GEE Export  |  {total} tasks  |  {START_YEAR}-{END_YEAR}')
    log.info(f'Drive folder : {DRIVE_FOLDER}')
    log.info(f'Resolution   : {EXPORT_SCALE}m  |  CRS: {EXPORT_CRS}')
    log.info('=' * 65)

    for y in range(START_YEAR, END_YEAR + 1):
        for m in MONTHS:
            mm = f'{m:02d}'
            log.info(f'\n-- {y}-{mm} ------------------------------------------')

            # Define all four variables for this month
            # (label, build_function)
            var_defs = [
                (f'NDVI_{y}_{mm}',      make_month_ndvi),
                (f'NDMI_{y}_{mm}',      make_month_ndmi),
                (f'RAIN_{y}_{mm}',      make_month_rain),
                (f'TEMP_MEAN_{y}_{mm}', make_month_temp),
            ]

            for label, func in var_defs:
                try:
                    image = func(y, m)
                except Exception as e:
                    log.error(f'  Build failed [{label}]: {e}')
                    failed.append(label)
                    continue

                ok = submit_task(image, label)
                if ok:
                    submitted += 1
                else:
                    failed.append(label)

                time.sleep(TASK_START_WAIT_S)

    # Summary
    log.info('\n' + '=' * 65)
    log.info(f'DONE.  {submitted}/{total} tasks submitted.')
    if failed:
        log.warning(f'Failed ({len(failed)} tasks):')
        for f in failed:
            log.warning(f'  * {f}')
    else:
        log.info('All tasks submitted successfully ✓')
    log.info('Monitor: https://code.earthengine.google.com/tasks')
    log.info('=' * 65)
    return submitted, failed

# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
if __name__ == '__main__':
    start_time = datetime.now()
    submitted, failed = run_exports()
    elapsed = datetime.now() - start_time
    log.info(f'Total submission time: {elapsed}')