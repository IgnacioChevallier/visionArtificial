import ee
import geemap
import os
import csv
import numpy as np

# ── Inicialización ────────────────────────────────────────────────────────────
ee.Initialize()

# ── Configuración ─────────────────────────────────────────────────────────────
ROI = ee.Geometry.Rectangle([
    -71.927,  # west
    -42.818,  # south
    -71.296,  # east
    -42.457   # north
])

FECHA_PRE_INI  = "2025-11-01"
FECHA_PRE_FIN  = "2025-12-08"   # un día antes del inicio del incendio "Puerto Café"
FECHA_POST_INI = "2026-02-01"
FECHA_POST_FIN = "2026-03-08"   # cicatriz formada, fuego contenido el 18 feb

BANDAS_IMG   = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # bandas a guardar en el tile
DNBR_UMBRAL       = 0.27  # menor = mascara mas generosa
PRE_NBR_MIN      = 0.15  # vegetacion minima antes del incendio
POST_NBR_MAX     = 0.40  # NBR maximo despues; mayor = mascara mas generosa
NDSI_NIEVE_MAX   = 0.25  # valores mayores suelen ser nieve/hielo
BRILLO_MAX       = 3500  # excluye pixeles muy blancos en bandas visibles
CROMA_MIN        = 500   # excluye blancos/grises con poca diferencia RGB
MIN_PIXELES      = 120   # componentes conexas menores a esto se eliminan
DILATACION_PX    = 1     # expande levemente la cicatriz detectada
ESCALA           = 20    # metros/píxel (resolución nativa S2 para B8/B11/B12)
N_TILES      = 30     # cantidad de tiles a generar

OUTPUT_DIR   = "dataset"
PRE_IMG_DIR  = os.path.join(OUTPUT_DIR, "imagenes_pre")
IMG_DIR      = os.path.join(OUTPUT_DIR, "imagenes")
MASK_DIR     = os.path.join(OUTPUT_DIR, "mascaras")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

os.makedirs(PRE_IMG_DIR, exist_ok=True)
os.makedirs(IMG_DIR,     exist_ok=True)
os.makedirs(MASK_DIR,    exist_ok=True)


# ── Cloud masking con SCL ─────────────────────────────────────────────────────
def mask_clouds(img):
    scl = img.select('SCL')
    mask = (scl.neq(3)          # cloud shadow
              .And(scl.neq(8))  # cloud medium prob
              .And(scl.neq(9))  # cloud high prob
              .And(scl.neq(10)) # cirrus
              .And(scl.neq(11)))# snow
    return img.updateMask(mask)


# ── Colecciones base ──────────────────────────────────────────────────────────
s2_base = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(ROI)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

pre_collection  = s2_base.filterDate(FECHA_PRE_INI,  FECHA_PRE_FIN)
post_collection = s2_base.filterDate(FECHA_POST_INI, FECHA_POST_FIN)

pre_count = pre_collection.size().getInfo()
post_count = post_collection.size().getInfo()
print(f"Imagenes Sentinel-2 PRE encontradas: {pre_count}")
print(f"Imagenes Sentinel-2 POST encontradas: {post_count}")

if pre_count == 0 or post_count == 0:
    raise RuntimeError(
        "No hay imagenes Sentinel-2 para una de las ventanas con los filtros actuales. "
        "Ampliar FECHA_PRE/POST o relajar CLOUDY_PIXEL_PERCENTAGE."
    )

# Se usa la colección enmascarada para calcular dNBR, pero la colección completa
# para exportar RGB sin huecos negros en la visualización.
pre_masked_collection = pre_collection.map(mask_clouds)
post_masked_collection = post_collection.map(mask_clouds)


# ── Composiciones PRE y POST ──────────────────────────────────────────────────
pre_masked  = pre_masked_collection.median()
post_masked = post_masked_collection.median()
pre_full    = pre_collection.median()
post_full   = post_collection.median()


# ── dNBR y máscara binaria ────────────────────────────────────────────────────
pre_nbr  = pre_masked.normalizedDifference(['B8', 'B12']).rename('NBR_pre')
post_nbr = post_masked.normalizedDifference(['B8', 'B12']).rename('NBR_post')
dnbr     = pre_nbr.subtract(post_nbr).rename('dNBR')

post_vis = post_full.select(['B2', 'B3', 'B4'])
post_brightness = post_vis.reduce(ee.Reducer.mean())
post_chroma = (post_vis.reduce(ee.Reducer.max())
                     .subtract(post_vis.reduce(ee.Reducer.min())))
post_ndsi = post_full.normalizedDifference(['B3', 'B11']).rename('NDSI_post')

nieve_o_gris_claro = (
    post_ndsi.gt(NDSI_NIEVE_MAX)
        .Or(post_brightness.gt(BRILLO_MAX).And(post_chroma.lt(CROMA_MIN)))
)

burned_raw = (
    dnbr.gt(DNBR_UMBRAL)
        .And(pre_nbr.gt(PRE_NBR_MIN))
        .And(post_nbr.lt(POST_NBR_MAX))
        .And(nieve_o_gris_claro.Not())
)

# Eliminar falsos positivos pequeños y unir un poco la cicatriz detectada.
# Importante: connectedPixelCount debe correr solo sobre candidatos quemados.
# Si se aplica sobre 0/1 sin enmascarar, tambien conserva regiones enormes de no quemado.
burned_candidates = burned_raw.updateMask(burned_raw)
burned_clean = (burned_candidates
                  .connectedPixelCount(MIN_PIXELES + 1, True)
                  .gte(MIN_PIXELES)
                  .unmask(0))
burned = (burned_clean
            .focal_max(radius=DILATACION_PX, units='pixels')
            .rename('quemado'))


# ── Imágenes a exportar (composiciones PRE y POST con bandas seleccionadas) ───
imagen_pre  = pre_full.select(BANDAS_IMG)
imagen_post = post_full.select(BANDAS_IMG)


# ── Grilla de tiles sobre el ROI ──────────────────────────────────────────────
# Cada tile cubre 256px × 20m/px ≈ 5.12 km por lado
coords = ROI.bounds().getInfo()['coordinates'][0]
lon_min, lat_min = coords[0]
lon_max, lat_max = coords[2]

TILE_DEG = (ESCALA * 256) / 111320  # conversión metros → grados (aprox.)
lons = np.arange(lon_min, lon_max, TILE_DEG)
lats = np.arange(lat_min, lat_max, TILE_DEG)

tiles = [
    ee.Geometry.Rectangle([
        lon, lat,
        min(lon + TILE_DEG, lon_max),
        min(lat + TILE_DEG, lat_max)
    ])
    for lon in lons
    for lat in lats
]

print(f"Tiles disponibles en el ROI: {len(tiles)}")
tiles_a_procesar = tiles


# ── Descarga ──────────────────────────────────────────────────────────────────
with open(METADATA_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "tile", "lon_min", "lat_min", "lon_max", "lat_max",
        "pct_quemado", "fecha_pre", "fecha_post", "escala_m"
    ])

    for idx, tile_geom in enumerate(tiles_a_procesar):
        nombre    = f"tile_{idx:03d}"
        ruta_pre  = os.path.join(PRE_IMG_DIR, f"{nombre}.tif")
        ruta_img  = os.path.join(IMG_DIR,     f"{nombre}.tif")
        ruta_mask = os.path.join(MASK_DIR,    f"{nombre}.tif")

        print(f"[{idx+1:02d}/{N_TILES}] Descargando {nombre}...")

        try:
            # Imagen PRE multibanda
            geemap.download_ee_image(
                image=imagen_pre.clip(tile_geom),
                filename=ruta_pre,
                region=tile_geom,
                scale=ESCALA,
                crs="EPSG:4326"
            )

            # Imagen POST multibanda
            geemap.download_ee_image(
                image=imagen_post.clip(tile_geom),
                filename=ruta_img,
                region=tile_geom,
                scale=ESCALA,
                crs="EPSG:4326"
            )

            # Máscara binaria
            geemap.download_ee_image(
                image=burned.clip(tile_geom),
                filename=ruta_mask,
                region=tile_geom,
                scale=ESCALA,
                crs="EPSG:4326"
            )

            # Porcentaje quemado para metadata y balanceo
            stats = burned.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=tile_geom,
                scale=ESCALA,
                maxPixels=1e8
            )
            pct = round((stats.get('quemado').getInfo() or 0) * 100, 2)

            bounds = tile_geom.bounds().getInfo()['coordinates'][0]
            writer.writerow([
                nombre,
                bounds[0][0], bounds[0][1],  # lon_min, lat_min
                bounds[2][0], bounds[2][1],  # lon_max, lat_max
                pct,
                f"{FECHA_PRE_INI}/{FECHA_PRE_FIN}",
                f"{FECHA_POST_INI}/{FECHA_POST_FIN}",
                ESCALA
            ])

            print(f"    ✓ {nombre} — {pct}% quemado")

        except Exception as e:
            print(f"    ✗ {nombre} — Error: {e}")
            writer.writerow([nombre, "", "", "", "", "", "", "", "ERROR"])

print(f"\nDataset generado en: {OUTPUT_DIR}/")
