"""
generar_dataset_temporal.py — Dataset Sentinel-2 con mínima ventana temporal.

Timing por tile (en orden de preferencia):
  1. fire_arrival.csv  (VIIRS SNPP + NOAA-20, ~1.5h precisión por pasada)
  2. fire_timing.csv   (MODIS diario + S2, ~3h-días)

Outputs (en dataset_temporal/):
  imagenes_pre/tile_XXX.tif   — S2 individual más reciente antes del fuego
  imagenes/tile_XXX.tif       — S2 individual más temprana después del fuego
  mascaras/tile_XXX.tif       — máscara binaria (dNBR composite)
  metadata_temporal.csv       — fire_arrival_utc, lag_pre_h, lag_post_h, precision
"""

import ee
import geemap
import os
import csv
import datetime
import numpy as np
from pathlib import Path

ee.Initialize()

# ── Configuración ─────────────────────────────────────────────────────────────
ROI = ee.Geometry.Rectangle([-71.917, -42.821, -71.359, -42.500])

# Parámetros de búsqueda de imagen S2 por tile
GAP_MAX_PRE  = 20   # días máximos de antelación para imagen pre-fuego
GAP_MAX_POST = 20   # días máximos de demora para imagen post-fuego
CLOUD_MAX    = 20   # % nubosidad máxima para aceptar una imagen individual

# Parámetros de máscara (mismos que generar_dataset.py — no se tocan)
BANDAS_IMG      = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
DNBR_UMBRAL     = 0.22
PRE_NBR_MIN     = 0.10
POST_NBR_MAX    = 0.40
NDSI_NIEVE_MAX  = 0.35
BRILLO_MAX      = 3500
CROMA_MIN       = 500
MIN_PIXELES     = 60
DILATACION_PX   = 1
ESCALA          = 20
N_TILES         = 30

# Ventanas del composite para la máscara (sin cambio respecto a original)
FECHA_MASK_PRE_INI  = "2025-11-01"
FECHA_MASK_PRE_FIN  = "2025-12-08"
FECHA_MASK_POST_INI = "2026-02-01"
FECHA_MASK_POST_FIN = "2026-03-08"

VIIRS_CSV  = Path('fire_arrival.csv')   # fuente primaria (VIIRS, ~1.5h precisión)
MODIS_CSV  = Path('fire_timing.csv')   # fallback (MODIS/S2)
OUTPUT_DIR = "dataset_temporal"
PRE_DIR    = os.path.join(OUTPUT_DIR, "imagenes_pre")
POST_DIR   = os.path.join(OUTPUT_DIR, "imagenes")
MASK_DIR   = os.path.join(OUTPUT_DIR, "mascaras")
META_CSV   = os.path.join(OUTPUT_DIR, "metadata_temporal.csv")

os.makedirs(PRE_DIR,  exist_ok=True)
os.makedirs(POST_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)


# ── Cargar fuentes de timing (VIIRS primero, fallback MODIS/S2) ───────────────
timing = {}   # tile_name → {'fire_arrival_utc': str, 'precision': str}

# Fallback: MODIS/S2 (fire_timing.csv)
if MODIS_CSV.exists():
    with open(MODIS_CSV) as f:
        for row in csv.DictReader(f):
            timing[row['tile']] = {
                'fire_arrival_utc': row.get('fire_arrival_utc', ''),
                'precision':        row.get('precision', 'unknown'),
            }
    print(f"fire_timing.csv   cargado: {len(timing)} tiles")

# Primario: VIIRS (fire_arrival.csv) — sobreescribe donde hay datos más precisos
viirs_count = 0
if VIIRS_CSV.exists():
    with open(VIIRS_CSV) as f:
        for row in csv.DictReader(f):
            if row.get('first_fire_utc'):
                timing[row['tile_id']] = {
                    'fire_arrival_utc': row['first_fire_utc'],
                    'precision':        row.get('precision', 'VIIRS_375m'),
                }
                viirs_count += 1
    print(f"fire_arrival.csv  cargado: {viirs_count} tiles con VIIRS (precisión ~1.5h)")


# ── Cloud masking ─────────────────────────────────────────────────────────────
def mask_clouds(img):
    scl = img.select('SCL')
    mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
               .And(scl.neq(10)).And(scl.neq(11)))
    return img.updateMask(mask)


# ── Máscara dNBR (composite, mejor calidad) ───────────────────────────────────
s2_base_mask = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(ROI)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

pre_masked  = s2_base_mask.filterDate(FECHA_MASK_PRE_INI,  FECHA_MASK_PRE_FIN).map(mask_clouds).median()
post_masked = s2_base_mask.filterDate(FECHA_MASK_POST_INI, FECHA_MASK_POST_FIN).map(mask_clouds).median()

pre_nbr  = pre_masked.normalizedDifference(['B8', 'B12']).rename('NBR_pre')
post_nbr = post_masked.normalizedDifference(['B8', 'B12']).rename('NBR_post')
dnbr     = pre_nbr.subtract(post_nbr).rename('dNBR')

post_vis = s2_base_mask.filterDate(FECHA_MASK_POST_INI, FECHA_MASK_POST_FIN).median().select(['B2','B3','B4'])
post_brightness = post_vis.reduce(ee.Reducer.mean())
post_chroma     = post_vis.reduce(ee.Reducer.max()).subtract(post_vis.reduce(ee.Reducer.min()))
post_ndsi       = (s2_base_mask.filterDate(FECHA_MASK_POST_INI, FECHA_MASK_POST_FIN)
                   .median().normalizedDifference(['B3','B11']).rename('NDSI_post'))

nieve_o_gris = (post_ndsi.gt(NDSI_NIEVE_MAX)
                .Or(post_brightness.gt(BRILLO_MAX).And(post_chroma.lt(CROMA_MIN))))

burned_raw  = (dnbr.gt(DNBR_UMBRAL).And(pre_nbr.gt(PRE_NBR_MIN))
               .And(post_nbr.lt(POST_NBR_MAX)).And(nieve_o_gris.Not()))
burned_cand = burned_raw.updateMask(burned_raw)
burned_clean = (burned_cand.connectedPixelCount(MIN_PIXELES + 1, True)
                .gte(MIN_PIXELES).unmask(0))
burned = burned_clean.focal_max(radius=DILATACION_PX, units='pixels').rename('quemado')

print("Máscara dNBR composite calculada.")


# ── Grilla de tiles ───────────────────────────────────────────────────────────
TILE_DEG = (ESCALA * 256) / 111320
coords   = ROI.bounds().getInfo()['coordinates'][0]
lon_min, lat_min = coords[0]
lon_max, lat_max = coords[2]
lons = np.arange(lon_min, lon_max, TILE_DEG)
lats = np.arange(lat_min, lat_max, TILE_DEG)
tiles = [
    ee.Geometry.Rectangle([lon, lat,
                            min(lon + TILE_DEG, lon_max),
                            min(lat + TILE_DEG, lat_max)])
    for lon in lons for lat in lats
][:N_TILES]
tile_names = [f'tile_{i:03d}' for i in range(len(tiles))]
print(f"Tiles: {len(tiles)}")


# ── Colección S2 para imágenes individuales ───────────────────────────────────
s2_individual = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(ROI)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_MAX))
                 .select(BANDAS_IMG))


def get_closest_image(tile_geom, fire_dt, direction='pre'):
    """
    Find the closest S2 image (pre-fire or post-fire) to fire_dt.
    direction='pre': latest image BEFORE fire_dt (within GAP_MAX_PRE days)
    direction='post': earliest image AFTER fire_dt (within GAP_MAX_POST days)
    Returns (ee.Image or None, date_str or None)
    """
    if direction == 'pre':
        start = (fire_dt - datetime.timedelta(days=GAP_MAX_PRE)).isoformat()
        end   = fire_dt.isoformat()
        col   = (s2_individual.filterBounds(tile_geom)
                              .filterDate(start, end)
                              .sort('system:time_start', False))  # most recent first
    else:
        start = (fire_dt + datetime.timedelta(days=1)).isoformat()
        end   = (fire_dt + datetime.timedelta(days=GAP_MAX_POST)).isoformat()
        col   = (s2_individual.filterBounds(tile_geom)
                              .filterDate(start, end)
                              .sort('system:time_start', True))   # earliest first

    size = col.size().getInfo()
    if size == 0:
        return None, None

    img      = col.first()
    date_str = img.date().format('YYYY-MM-dd').getInfo()
    return img, date_str


# ── Descarga ──────────────────────────────────────────────────────────────────
with open(META_CSV, 'w', newline='') as csvfile:
    fieldnames = [
        'tile', 'lon_min', 'lat_min', 'lon_max', 'lat_max',
        'fire_arrival_utc', 'timing_precision',
        'fecha_pre_img',  'lag_pre_h',
        'fecha_post_img', 'lag_post_h',
        'pct_quemado', 'escala_m', 'status'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for idx, (tile_geom, nombre) in enumerate(zip(tiles, tile_names)):
        print(f"\n[{idx+1:02d}/{N_TILES}] {nombre}")

        bounds  = tile_geom.bounds().getInfo()['coordinates'][0]
        lo, la  = bounds[0]
        hi, ha  = bounds[2]

        t_row   = timing.get(nombre, {})
        arrival = t_row.get('fire_arrival_utc', '')
        prec    = t_row.get('precision', 'unknown')

        fire_datetime = None   # datetime con timezone (precisión de minutos)
        fire_dt       = None   # date (para búsqueda de imágenes S2)
        if arrival:
            fire_datetime = datetime.datetime.fromisoformat(arrival)
            if fire_datetime.tzinfo is None:
                fire_datetime = fire_datetime.replace(tzinfo=datetime.timezone.utc)
            fire_dt = fire_datetime.date()

        # ── Imágenes individuales pre/post ─────────────────────────────────
        img_pre,  date_pre  = get_closest_image(tile_geom, fire_dt, 'pre')  if fire_dt else (None, None)
        img_post, date_post = get_closest_image(tile_geom, fire_dt, 'post') if fire_dt else (None, None)

        if img_pre is None or img_post is None:
            status = 'SKIP_no_S2'
            print(f"  ✗ Sin imágenes S2 dentro de la ventana — SKIP")
            writer.writerow({
                'tile': nombre, 'lon_min': round(lo,6), 'lat_min': round(la,6),
                'lon_max': round(hi,6), 'lat_max': round(ha,6),
                'fire_arrival_utc': arrival, 'timing_precision': prec,
                'fecha_pre_img': '', 'lag_pre_h': '',
                'fecha_post_img': '', 'lag_post_h': '',
                'pct_quemado': '', 'escala_m': ESCALA, 'status': status
            })
            continue

        # Calcular lag en horas usando timestamp exacto cuando está disponible.
        # S2 pasa sobre esta región a ~14:44 UTC; usamos esa hora para el cálculo.
        S2_PASS_UTC = datetime.time(14, 44, tzinfo=datetime.timezone.utc)

        def lag_h(img_date_str, fire_exact_dt):
            img_dt = datetime.datetime.combine(
                datetime.date.fromisoformat(img_date_str),
                S2_PASS_UTC.replace(tzinfo=None),
                tzinfo=datetime.timezone.utc
            )
            return round(abs((img_dt - fire_exact_dt).total_seconds()) / 3600, 1)

        lag_pre  = lag_h(date_pre,  fire_datetime) if fire_datetime else abs((datetime.date.fromisoformat(date_pre)  - fire_dt).days) * 24
        lag_post = lag_h(date_post, fire_datetime) if fire_datetime else abs((datetime.date.fromisoformat(date_post) - fire_dt).days) * 24

        print(f"  Fire: {arrival}  [{prec}]")
        print(f"  PRE:  {date_pre}  ({lag_pre}h antes)")
        print(f"  POST: {date_post}  ({lag_post}h después)")

        ruta_pre  = os.path.join(PRE_DIR,  f"{nombre}.tif")
        ruta_post = os.path.join(POST_DIR, f"{nombre}.tif")
        ruta_mask = os.path.join(MASK_DIR, f"{nombre}.tif")

        try:
            geemap.download_ee_image(
                image=img_pre.clip(tile_geom), filename=ruta_pre,
                region=tile_geom, scale=ESCALA, crs="EPSG:4326")

            geemap.download_ee_image(
                image=img_post.clip(tile_geom), filename=ruta_post,
                region=tile_geom, scale=ESCALA, crs="EPSG:4326")

            geemap.download_ee_image(
                image=burned.clip(tile_geom), filename=ruta_mask,
                region=tile_geom, scale=ESCALA, crs="EPSG:4326")

            pct_stats = burned.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=tile_geom, scale=ESCALA, maxPixels=1e8)
            pct = round((pct_stats.get('quemado').getInfo() or 0) * 100, 2)

            status = 'OK'
            print(f"  ✓ {pct}% quemado  lag total={lag_pre + lag_post}h")

        except Exception as e:
            status = f'ERROR: {e}'
            pct    = ''
            print(f"  ✗ Error: {e}")

        writer.writerow({
            'tile': nombre, 'lon_min': round(lo,6), 'lat_min': round(la,6),
            'lon_max': round(hi,6), 'lat_max': round(ha,6),
            'fire_arrival_utc': arrival, 'timing_precision': prec,
            'fecha_pre_img':  date_pre  or '',
            'lag_pre_h':      lag_pre   if date_pre  else '',
            'fecha_post_img': date_post or '',
            'lag_post_h':     lag_post  if date_post else '',
            'pct_quemado': pct, 'escala_m': ESCALA, 'status': status
        })

print(f"\n✓ Dataset temporal en: {OUTPUT_DIR}/")
print(f"  Metadata: {META_CSV}")
