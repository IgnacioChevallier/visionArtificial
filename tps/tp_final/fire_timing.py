"""
fire_timing.py — Per-tile fire arrival timing for the "Puerto Café" fire.

Estrategia:
  A) Tiles quemados ANTES de Jan 4:
     Comparar Nov-25 (ref) vs Jan-4. Si dNBR > 0.20 → quemado en [Dec 8, Jan 4].
  B) Tiles quemados DESPUÉS de Jan 4:
     Comparar Jan-4 (ref) vs Jan-9, Jan-11, Jan-19, Jan-24, Jan-29 consecutivamente.
     Primer par donde dNBR > 0.10 → ventana de quemado.
  C) En ambos casos, afinar con MODIS Terra+Aqua dentro de la ventana S2.

Salida:
  fire_timing.csv   — resumen por tile (fecha y precisión de llegada del fuego)
"""

import ee
import csv
import datetime
import numpy as np
from pathlib import Path

ee.Initialize()

# ── Configuración ─────────────────────────────────────────────────────────────
ROI             = ee.Geometry.Rectangle([-71.917, -42.821, -71.359, -42.500])
FECHA_INI       = '2025-11-01'
FECHA_FIN       = '2026-02-01'
FECHA_FUEGO_INI = '2025-12-08'

# Umbral dNBR para considerar un tile como "quemado" en un período
# (sobre el valor MEDIO por tile — mezcla de píxeles quemados y no quemados)
DNBR_PRE_JAN4  = 0.20   # Nov25→Jan4: umbral alto (hay leve sesgo estacional +0.02)
DNBR_POST_JAN4 = 0.10   # Jan4→siguiente: umbral bajo (sin sesgo estacional)
S2_VALID_FRAC  = 0.30   # fracción mínima de px válidos para usar imagen
MODIS_FIRE_MIN = 7      # FireMask >= 7
N_TILES        = 30
ESCALA_S2      = 20
ESCALA_MODIS   = 1000

# Tiempos de sobrepaso MODIS para -42°S, -71.5°W (UTC)
# Terra descending ~10:30 LST → ~15:15 UTC
# Aqua ascending   ~13:30 LST → ~18:15 UTC
TERRA_UTC = datetime.time(15, 15)
AQUA_UTC  = datetime.time(18, 15)

OUTPUT_CSV = Path('fire_timing.csv')


# ── Cloud masking S2 ──────────────────────────────────────────────────────────
def mask_clouds(img):
    scl = img.select('SCL')
    return img.updateMask(
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    )


# ── Grilla de tiles (igual que generar_dataset.py) ────────────────────────────
TILE_DEG = (20 * 256) / 111320
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
tile_fc    = ee.FeatureCollection([
    ee.Feature(geom, {'tile_id': name})
    for geom, name in zip(tiles, tile_names)
])
print(f"Tiles: {len(tiles)}")


# ── S2 colección con NBR ──────────────────────────────────────────────────────
def add_nbr(img):
    return (img.normalizedDifference(['B8', 'B12'])
               .rename('NBR')
               .copyProperties(img, ['system:time_start']))

s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(ROI)
          .filterDate(FECHA_INI, FECHA_FIN)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
          .map(mask_clouds)
          .map(add_nbr))


def daily_median(date_str):
    """Median NBR composite for a single calendar date."""
    d1 = date_str
    d2 = (datetime.date.fromisoformat(date_str) + datetime.timedelta(1)).isoformat()
    return s2_col.filterDate(d1, d2).median().rename('NBR')


def batch_dnbr_stats(img_ref, img_new):
    """
    Compute dNBR (ref - new) statistics and valid-pixel fraction for all tiles.
    Returns dict: tile_id → {'mean': float|None, 'p25': float|None, 'valid': float}
    dNBR > 0 → burned (NBR decreased from ref to new)
    """
    dnbr      = img_ref.subtract(img_new).rename('dNBR')
    valid_new = img_new.mask().rename('valid')   # 1=valid, 0=masked (from .mask())
    stack     = dnbr.addBands(valid_new)

    stats = stack.reduceRegions(
        collection=tile_fc,
        reducer=(ee.Reducer.mean()
                 .combine(ee.Reducer.percentile([25, 75]), sharedInputs=True)
                 .combine(ee.Reducer.count(), sharedInputs=True)),
        scale=ESCALA_S2
    ).getInfo()['features']

    out = {}
    for feat in stats:
        p    = feat['properties']
        name = p['tile_id']
        out[name] = {
            'mean':  p.get('dNBR_mean'),
            'p25':   p.get('dNBR_p25'),
            'p75':   p.get('dNBR_p75'),
            'valid': p.get('valid_mean', 0) or 0,
        }
    return out


def is_burned(stats, threshold_mean, threshold_p75=None):
    """Tile is considered burned if mean OR p75 exceeds threshold."""
    m   = stats.get('mean')
    p75 = stats.get('p75')
    v   = stats.get('valid', 0)
    if v < S2_VALID_FRAC or m is None:
        return False
    if m >= threshold_mean:
        return True
    if threshold_p75 is not None and p75 is not None and p75 >= threshold_p75:
        return True
    return False


# ── Imágenes de referencia clave ──────────────────────────────────────────────
print("Cargando imágenes de referencia...")
# Nov 25: última imagen claramente pre-fuego (0-1% nubes)
img_nov25 = daily_median('2025-11-25')
# Jan 4: primera imagen post-fuego sin nubes (0% nubes)
img_jan4  = daily_median('2026-01-04')

# Fechas post-Jan4 para análisis consecutivo
POST_JAN4_DATES = ['2026-01-09', '2026-01-11', '2026-01-19', '2026-01-24', '2026-01-29']


# ── Fase 1A: tiles quemados ANTES de Jan 4 (Dec burns) ───────────────────────
print("\n── Fase 1A: dNBR Nov25 → Jan4 (detecta quemas de diciembre) ──")
stats_dec = batch_dnbr_stats(img_nov25, img_jan4)

tile_s2_window = {}
for name, s in stats_dec.items():
    # Criterio: mean > 0.20  O  p75 > 0.30 (muchos píxeles quemados)
    if is_burned(s, threshold_mean=DNBR_PRE_JAN4, threshold_p75=0.30):
        tile_s2_window[name] = (datetime.date.fromisoformat(FECHA_FUEGO_INI),
                                datetime.date(2026, 1, 4))
        print(f"  {name}  mean={s['mean']:.3f}  p75={s['p75']:.3f}  valid={s['valid']:.2f}  → ANTES Jan4")

print(f"Tiles con quema pre-Jan4: {len(tile_s2_window)}")


# ── Fase 1B: tiles quemados DESPUÉS de Jan 4 ─────────────────────────────────
print("\n── Fase 1B: dNBR Jan4 → fechas posteriores ──")
prev_date = datetime.date(2026, 1, 4)

for date_str in POST_JAN4_DATES:
    img_new = daily_median(date_str)
    stats   = batch_dnbr_stats(img_jan4, img_new)

    n_new = 0
    for name, s in stats.items():
        if name in tile_s2_window:
            continue
        # Criterio menos estricto: mean > 0.10  O  p75 > 0.15
        if is_burned(s, threshold_mean=DNBR_POST_JAN4, threshold_p75=0.15):
            tile_s2_window[name] = (prev_date,
                                    datetime.date.fromisoformat(date_str))
            n_new += 1

    total = len(tile_s2_window)
    print(f"  Jan4 → {date_str}  tiles nuevos: {n_new}  total: {total}/{len(tiles)}")
    prev_date = datetime.date.fromisoformat(date_str)

# Tiles sin ventana: asignar ventana amplia (toda la duración del incendio)
for name in tile_names:
    if name not in tile_s2_window:
        tile_s2_window[name] = (datetime.date.fromisoformat(FECHA_FUEGO_INI),
                                datetime.date(2026, 1, 29))
print(f"\nTotal tiles con ventana S2: {len(tile_s2_window)}")


# ── Fase 2: MODIS diario dentro de cada ventana ───────────────────────────────
print("\n── Fase 2: MODIS Terra + Aqua por día ──")

terra_col = (ee.ImageCollection('MODIS/061/MOD14A1')
             .filterBounds(ROI)
             .filterDate(FECHA_FUEGO_INI, FECHA_FIN)
             .select('FireMask')
             .sort('system:time_start'))
aqua_col  = (ee.ImageCollection('MODIS/061/MYD14A1')
             .filterBounds(ROI)
             .filterDate(FECHA_FUEGO_INI, FECHA_FIN)
             .select('FireMask')
             .sort('system:time_start'))

n_terra = terra_col.size().getInfo()
n_aqua  = aqua_col.size().getInfo()
terra_ee = terra_col.toList(n_terra)
aqua_ee  = aqua_col.toList(n_aqua)

def ts2date(ts):
    return datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc).date()

terra_dates = {ts2date(ts): i for i, ts in enumerate(
    terra_col.aggregate_array('system:time_start').getInfo())}
aqua_dates  = {ts2date(ts): i for i, ts in enumerate(
    aqua_col.aggregate_array('system:time_start').getInfo())}
all_modis_dates = sorted(set(terra_dates) | set(aqua_dates))

tile_modis = {}   # tile_id → {'date': date, 'source': 'Terra'|'Aqua'}

for day in all_modis_dates:
    pending = [n for n in tile_names
               if n not in tile_modis and n in tile_s2_window]
    if not pending:
        break

    pending_fc = ee.FeatureCollection([
        ee.Feature(tiles[tile_names.index(n)], {'tile_id': n})
        for n in pending
    ])

    results_day = {n: {} for n in pending}

    if day in terra_dates:
        img_t = ee.Image(terra_ee.get(terra_dates[day])).gte(MODIS_FIRE_MIN)
        for feat in img_t.reduceRegions(
                collection=pending_fc,
                reducer=ee.Reducer.sum(),
                scale=ESCALA_MODIS).getInfo()['features']:
            n = feat['properties']['tile_id']
            results_day[n]['terra'] = (feat['properties'].get('sum', 0) or 0) > 0

    if day in aqua_dates:
        img_a = ee.Image(aqua_ee.get(aqua_dates[day])).gte(MODIS_FIRE_MIN)
        for feat in img_a.reduceRegions(
                collection=pending_fc,
                reducer=ee.Reducer.sum(),
                scale=ESCALA_MODIS).getInfo()['features']:
            n = feat['properties']['tile_id']
            results_day[n]['aqua'] = (feat['properties'].get('sum', 0) or 0) > 0

    for name, det in results_day.items():
        t = det.get('terra', False)
        a = det.get('aqua',  False)
        win = tile_s2_window.get(name)
        if (t or a) and win and win[0] <= day <= win[1]:
            tile_modis[name] = {'date': day, 'source': 'Terra' if t else 'Aqua'}
            print(f"  {name}  {day}  {'Terra' if t else 'Aqua'}  🔥")


# ── Fase 3: Ensamblar CSV ─────────────────────────────────────────────────────
print("\n── Generando fire_timing.csv ──")
rows = []
for idx, (tile_geom, name) in enumerate(zip(tiles, tile_names)):
    bounds  = tile_geom.bounds().getInfo()['coordinates'][0]
    lo, la  = bounds[0]
    hi, ha  = bounds[2]
    win     = tile_s2_window.get(name)
    modis_r = tile_modis.get(name)

    if modis_r:
        pass_time    = TERRA_UTC if modis_r['source'] == 'Terra' else AQUA_UTC
        fire_arrival = datetime.datetime.combine(
            modis_r['date'], pass_time, tzinfo=datetime.timezone.utc)
        precision = 'MODIS_~3h'
        source    = modis_r['source']
    elif win:
        mid = win[0] + (win[1] - win[0]) / 2
        fire_arrival = datetime.datetime.combine(
            mid, datetime.time(12, 0), tzinfo=datetime.timezone.utc)
        precision = f'S2_window_{(win[1] - win[0]).days}d'
        source    = 'S2_dNBR'
    else:
        fire_arrival = None
        precision    = 'unknown'
        source       = 'none'

    rows.append({
        'tile':               name,
        'lon_min':            round(lo, 6),
        'lat_min':            round(la, 6),
        'lon_max':            round(hi, 6),
        'lat_max':            round(ha, 6),
        's2_window_start':    win[0].isoformat() if win else '',
        's2_window_end':      win[1].isoformat() if win else '',
        's2_window_days':     (win[1] - win[0]).days if win else '',
        'modis_first_date':   modis_r['date'].isoformat() if modis_r else '',
        'modis_first_source': source if modis_r else '',
        'fire_arrival_utc':   fire_arrival.isoformat() if fire_arrival else '',
        'precision':          precision,
    })

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

modis_c = sum(1 for r in rows if r['modis_first_date'])
s2_c    = sum(1 for r in rows if r['precision'].startswith('S2'))
unk_c   = sum(1 for r in rows if r['precision'] == 'unknown')
print(f"\n✓ {OUTPUT_CSV}  ({len(rows)} tiles)")
print(f"  MODIS (~3h precisión): {modis_c} tiles")
print(f"  S2 window (±días):     {s2_c} tiles")
print(f"  Sin datos:             {unk_c} tiles")
print("\nPrimeras filas:")
for r in rows[:8]:
    print(f"  {r['tile']}  arrival={r['fire_arrival_utc'] or 'N/A':28}  [{r['precision']}]")
