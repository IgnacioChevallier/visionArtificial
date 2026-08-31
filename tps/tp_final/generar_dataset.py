import argparse
import glob
import multiprocessing as mp
import ee
import geemap
import os
import csv
import numpy as np
from functools import reduce

# ── Directorios y constantes puras (necesarias en workers y en __main__) ──────
OUTPUT_DIR   = "dataset"
PRE_IMG_DIR  = os.path.join(OUTPUT_DIR, "imagenes_pre")
IMG_DIR      = os.path.join(OUTPUT_DIR, "imagenes")
MASK_DIR     = os.path.join(OUTPUT_DIR, "mascaras")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

ROIS_COORDS = [
    [-72.014, -42.348, -71.774, -42.263],
    [-71.578, -42.416, -71.201, -42.079],
    [-71.927, -42.818, -71.296, -42.457],
]

FECHA_PRE_INI  = "2023-11-01"
FECHA_PRE_FIN  = "2023-12-31"
FECHA_POST_INI = "2026-02-01"
FECHA_POST_FIN = "2026-03-08"

BANDAS_IMG     = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
DNBR_UMBRAL    = 0.30
PRE_NBR_MIN    = 0.15
POST_NBR_MAX   = 0.40
NDSI_NIEVE_MAX = 0.25
BRILLO_MAX     = 3500
CROMA_MIN      = 500
NDWI_AGUA_MAX  = 0.05
MIN_PIXELES    = 150
DILATACION_PX  = 1
ESCALA         = 20
MAX_WORKERS    = 4     # procesos concurrentes; >6 puede saturar la cuota de GEE


# ── Worker: se construye una vez por proceso ───────────────────────────────────
_GEE = {}   # se llena en _init_worker, se usa en _download_one

def _mask_clouds(img):
    scl = img.select('SCL')
    return img.updateMask(
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    )

def _init_worker(cfg):
    """Inicializa GEE y construye las expresiones lazy. Se llama una vez por proceso."""
    import ee, geemap
    ee.Initialize()

    rois = [ee.Geometry.Rectangle(r) for r in cfg['rois_coords']]
    roi  = reduce(lambda a, b: a.union(b), rois)

    s2_base  = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    pre_col  = s2_base.filterDate(cfg['fecha_pre_ini'],  cfg['fecha_pre_fin'])
    post_col = s2_base.filterDate(cfg['fecha_post_ini'], cfg['fecha_post_fin'])

    pre_masked  = pre_col.map(_mask_clouds).median()
    post_masked = post_col.map(_mask_clouds).median()
    pre_full    = pre_col.median()
    post_full   = post_col.median()

    pre_nbr  = pre_masked.normalizedDifference(['B8', 'B12']).rename('NBR_pre')
    post_nbr = post_masked.normalizedDifference(['B8', 'B12']).rename('NBR_post')
    dnbr     = pre_nbr.subtract(post_nbr).rename('dNBR')

    post_vis        = post_full.select(['B2', 'B3', 'B4'])
    post_brightness = post_vis.reduce(ee.Reducer.mean())
    post_chroma     = (post_vis.reduce(ee.Reducer.max())
                               .subtract(post_vis.reduce(ee.Reducer.min())))
    post_ndsi = post_full.normalizedDifference(['B3', 'B11']).rename('NDSI_post')
    pre_ndwi  = pre_full.normalizedDifference(['B3', 'B8']).rename('NDWI_pre')

    nieve_o_gris = (
        post_ndsi.gt(cfg['ndsi_nieve_max'])
            .Or(post_brightness.gt(cfg['brillo_max']).And(post_chroma.lt(cfg['croma_min'])))
    )
    agua_pre = pre_ndwi.gt(cfg['ndwi_agua_max'])

    burned_raw = (
        dnbr.gt(cfg['dnbr_umbral'])
            .And(pre_nbr.gt(cfg['pre_nbr_min']))
            .And(post_nbr.lt(cfg['post_nbr_max']))
            .And(nieve_o_gris.Not())
            .And(agua_pre.Not())
    )
    burned_candidates = burned_raw.updateMask(burned_raw)
    burned_clean = (burned_candidates
                      .connectedPixelCount(cfg['min_pixeles'] + 1, True)
                      .gte(cfg['min_pixeles'])
                      .unmask(0))
    burned = (burned_clean
                .focal_max(radius=cfg['dilatacion_px'], units='pixels')
                .rename('quemado'))

    _GEE['burned']     = burned
    _GEE['imagen_pre'] = pre_full.select(cfg['bandas_img'])
    _GEE['imagen_post'] = post_full.select(cfg['bandas_img'])
    _GEE['cfg']        = cfg


def _download_one(args):
    """Descarga pre/post/mascara de un tile. Devuelve ('ok', row) o ('error', row)."""
    import ee, geemap
    idx, tile_coords = args
    cfg     = _GEE['cfg']
    nombre  = f"tile_{cfg['existing_tiles'] + idx:03d}"
    total   = cfg['total']
    escala  = cfg['escala']
    tile_geom = ee.Geometry.Rectangle(tile_coords)

    ruta_pre  = os.path.join(PRE_IMG_DIR, f"{nombre}.tif")
    ruta_img  = os.path.join(IMG_DIR,     f"{nombre}.tif")
    ruta_mask = os.path.join(MASK_DIR,    f"{nombre}.tif")

    print(f"[{idx+1:03d}/{total}] Iniciando {nombre}...")
    try:
        geemap.download_ee_image(
            image=_GEE['imagen_pre'].clip(tile_geom), filename=ruta_pre,
            region=tile_geom, scale=escala, crs="EPSG:4326"
        )
        geemap.download_ee_image(
            image=_GEE['imagen_post'].clip(tile_geom), filename=ruta_img,
            region=tile_geom, scale=escala, crs="EPSG:4326"
        )
        geemap.download_ee_image(
            image=_GEE['burned'].clip(tile_geom), filename=ruta_mask,
            region=tile_geom, scale=escala, crs="EPSG:4326"
        )

        stats  = _GEE['burned'].reduceRegion(
            reducer=ee.Reducer.mean(), geometry=tile_geom,
            scale=escala, maxPixels=1e8
        )
        pct = round((stats.get('quemado').getInfo() or 0) * 100, 2)
        lon_min, lat_min, lon_max, lat_max = tile_coords
        row = [nombre, lon_min, lat_min, lon_max, lat_max, pct,
               f"{cfg['fecha_pre_ini']}/{cfg['fecha_pre_fin']}",
               f"{cfg['fecha_post_ini']}/{cfg['fecha_post_fin']}",
               escala]
        print(f"    ✓ {nombre} — {pct}% quemado")
        return ('ok', row)

    except Exception as e:
        print(f"    ✗ {nombre} — Error: {e}")
        return ('error', [nombre, "", "", "", "", "", "", "", "ERROR"])


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true",
                        help="Eliminar todo el contenido del dataset y salir")
    args = parser.parse_args()

    if args.clear:
        for directory in [PRE_IMG_DIR, IMG_DIR, MASK_DIR]:
            for f in glob.glob(os.path.join(directory, "*.tif")):
                os.remove(f)
        if os.path.isfile(METADATA_CSV):
            os.remove(METADATA_CSV)
        print("Dataset limpiado.")
        raise SystemExit(0)

    os.makedirs(PRE_IMG_DIR, exist_ok=True)
    os.makedirs(IMG_DIR,     exist_ok=True)
    os.makedirs(MASK_DIR,    exist_ok=True)

    # ── Verificar disponibilidad de imágenes (proceso principal) ──────────────
    ee.Initialize()
    rois = [ee.Geometry.Rectangle(r) for r in ROIS_COORDS]
    roi  = reduce(lambda a, b: a.union(b), rois)
    s2_base  = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    pre_count  = s2_base.filterDate(FECHA_PRE_INI,  FECHA_PRE_FIN).size().getInfo()
    post_count = s2_base.filterDate(FECHA_POST_INI, FECHA_POST_FIN).size().getInfo()
    print(f"Imágenes Sentinel-2 PRE encontradas:  {pre_count}")
    print(f"Imágenes Sentinel-2 POST encontradas: {post_count}")
    if pre_count == 0 or post_count == 0:
        raise RuntimeError(
            "No hay imágenes Sentinel-2 para alguna de las ventanas. "
            "Ampliá las fechas o relajá CLOUDY_PIXEL_PERCENTAGE."
        )

    # ── Grilla de tiles (coordenadas como listas planas, sin objetos EE) ──────
    TILE_DEG = (ESCALA * 256) / 111320
    tile_coords_list = []
    for i, (w, s, e, n) in enumerate(ROIS_COORDS):
        lons = np.arange(w, e, TILE_DEG)
        lats = np.arange(s, n, TILE_DEG)
        roi_tiles = [
            [lon, lat, min(lon + TILE_DEG, e), min(lat + TILE_DEG, n)]
            for lon in lons
            for lat in lats
        ]
        print(f"ROI {i+1}: {len(roi_tiles)} tiles")
        tile_coords_list.extend(roi_tiles)

    total          = len(tile_coords_list)
    existing_tiles = len([f for f in os.listdir(MASK_DIR) if f.endswith(".tif")])
    print(f"\nTotal tiles a procesar: {total}")
    print(f"Tiles existentes: {existing_tiles} — próximo índice: {existing_tiles}")
    print(f"Descargando con {MAX_WORKERS} procesos en paralelo...\n")

    cfg = dict(
        rois_coords=ROIS_COORDS,
        fecha_pre_ini=FECHA_PRE_INI,   fecha_pre_fin=FECHA_PRE_FIN,
        fecha_post_ini=FECHA_POST_INI, fecha_post_fin=FECHA_POST_FIN,
        bandas_img=BANDAS_IMG,
        dnbr_umbral=DNBR_UMBRAL,       pre_nbr_min=PRE_NBR_MIN,
        post_nbr_max=POST_NBR_MAX,     ndsi_nieve_max=NDSI_NIEVE_MAX,
        brillo_max=BRILLO_MAX,         croma_min=CROMA_MIN,
        ndwi_agua_max=NDWI_AGUA_MAX,   min_pixeles=MIN_PIXELES,
        dilatacion_px=DILATACION_PX,   escala=ESCALA,
        existing_tiles=existing_tiles, total=total,
    )

    pool_args = list(enumerate(tile_coords_list))

    # ── Descarga paralela ──────────────────────────────────────────────────────
    csv_exists = os.path.isfile(METADATA_CSV)
    with open(METADATA_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(["tile", "lon_min", "lat_min", "lon_max", "lat_max",
                              "pct_quemado", "fecha_pre", "fecha_post", "escala_m"])

        with mp.Pool(processes=MAX_WORKERS,
                     initializer=_init_worker, initargs=(cfg,)) as pool:
            for status, row in pool.imap_unordered(_download_one, pool_args):
                writer.writerow(row)
                csvfile.flush()

    print(f"\nDataset generado en: {OUTPUT_DIR}/")
