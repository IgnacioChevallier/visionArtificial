# Plan: Dataset Satelital de Áreas Quemadas — Los Alerces 2026
## Pipeline: Sentinel-2 SR + dNBR

---

## Idea central

Se usan dos composiciones Sentinel-2 (PRE y POST incendio) para calcular el índice
**dNBR** (Differenced Normalized Burn Ratio). Los píxeles con dNBR > 0.25 se clasifican
como quemados. Eso genera la máscara binaria. Sin ML, sin productos externos.

```
Sentinel-2 PRE  ──┐
                  ├──► dNBR = NBR_pre - NBR_post ──► threshold 0.25 ──► máscara binaria
Sentinel-2 POST ──┘
```

---

## Fechas

| Período | Rango                   | Justificación                                                              |
|---------|-------------------------|----------------------------------------------------------------------------|
| PRE     | 2025-11-01 → 2025-12-08 | Vegetación intacta. El incendio "Puerto Café" inició el 9 de diciembre de 2025 |
| POST    | 2026-02-05 → 2026-02-20 | Cicatriz completamente formada. El fuego fue declarado contenido el 18 de febrero de 2026 tras arrasar +16.000 ha |

> **Nota:** el PRE usa noviembre 2025 (no enero 2026) porque el fuego ya había
> comenzado el 9 de diciembre. Usar enero como PRE hubiera capturado vegetación
> parcialmente quemada, contaminando el ground truth del dNBR.

---

## Dependencias

```bash
pip install earthengine-api geemap numpy
```

Autenticación (una sola vez):

```bash
earthengine authenticate
```

---

## Estructura de salida

```
dataset/
├── imagenes/        # tiles Sentinel-2 multibanda .tif (B2 B3 B4 B8 B11 B12)
│   ├── tile_000.tif
│   ├── tile_001.tif
│   └── ...
├── mascaras/        # máscaras binarias .tif  (1=quemado, 0=no quemado)
│   ├── tile_000.tif
│   └── ...
└── metadata.csv     # por tile: coordenadas, pct_quemado, fechas usadas
```

---

## Script: `generar_dataset.py`

```python
import ee
import geemap
import os
import csv
import numpy as np

# ── Inicialización ────────────────────────────────────────────────────────────
ee.Initialize()

# ── Configuración ─────────────────────────────────────────────────────────────
ROI = ee.Geometry.Rectangle([-72.0, -42.8, -71.2, -42.0])  # Parque Nacional Los Alerces

FECHA_PRE_INI  = "2025-11-01"
FECHA_PRE_FIN  = "2025-12-08"   # un día antes del inicio del incendio "Puerto Café"
FECHA_POST_INI = "2026-02-05"
FECHA_POST_FIN = "2026-02-20"   # cicatriz formada, fuego contenido el 18 feb

BANDAS_IMG   = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # bandas a guardar en el tile
DNBR_UMBRAL  = 0.25   # threshold de quemado
MIN_PIXELES  = 100    # componentes conexas menores a esto se eliminan
ESCALA       = 20     # metros/píxel (resolución nativa S2 para B8/B11/B12)
N_TILES      = 30     # cantidad de tiles a generar

OUTPUT_DIR   = "dataset"
IMG_DIR      = os.path.join(OUTPUT_DIR, "imagenes")
MASK_DIR     = os.path.join(OUTPUT_DIR, "mascaras")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

os.makedirs(IMG_DIR,  exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)


# ── Cloud masking con SCL ─────────────────────────────────────────────────────
def mask_clouds(img):
    scl = img.select('SCL')
    mask = (scl.neq(3)          # cloud shadow
              .And(scl.neq(8))  # cloud medium prob
              .And(scl.neq(9))  # cloud high prob
              .And(scl.neq(10)) # cirrus
              .And(scl.neq(11)))# snow
    return img.updateMask(mask)


# ── Colección base ────────────────────────────────────────────────────────────
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(ROI)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_clouds))


# ── Composiciones PRE y POST ──────────────────────────────────────────────────
pre  = s2.filterDate(FECHA_PRE_INI,  FECHA_PRE_FIN).median()
post = s2.filterDate(FECHA_POST_INI, FECHA_POST_FIN).median()


# ── dNBR y máscara binaria ────────────────────────────────────────────────────
pre_nbr  = pre.normalizedDifference(['B8', 'B12']).rename('NBR_pre')
post_nbr = post.normalizedDifference(['B8', 'B12']).rename('NBR_post')
dnbr     = pre_nbr.subtract(post_nbr).rename('dNBR')

burned_raw = dnbr.gt(DNBR_UMBRAL)

# Eliminar falsos positivos pequeños (manchas < MIN_PIXELES píxeles contiguos)
burned = (burned_raw
            .connectedPixelCount(MIN_PIXELES + 1, True)
            .gte(MIN_PIXELES)
            .rename('quemado'))


# ── Imagen a exportar (composición POST con bandas seleccionadas) ──────────────
imagen_post = post.select(BANDAS_IMG)


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
tiles_a_procesar = tiles[:N_TILES]


# ── Descarga ──────────────────────────────────────────────────────────────────
with open(METADATA_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "tile", "lon_min", "lat_min", "lon_max", "lat_max",
        "pct_quemado", "fecha_pre", "fecha_post", "escala_m"
    ])

    for idx, tile_geom in enumerate(tiles_a_procesar):
        nombre    = f"tile_{idx:03d}"
        ruta_img  = os.path.join(IMG_DIR,  f"{nombre}.tif")
        ruta_mask = os.path.join(MASK_DIR, f"{nombre}.tif")

        print(f"[{idx+1:02d}/{N_TILES}] Descargando {nombre}...")

        try:
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
```

---

## Parámetros ajustables

| Variable       | Default         | Cuándo cambiarlo                               |
|----------------|-----------------|------------------------------------------------|
| `FECHA_PRE_*`  | nov 01 – dic 08 2025 | Ajustar si se quiere ampliar la ventana PRE      |
| `FECHA_POST_*` | feb 05–20 2026       | Ajustar si hay mucha nubosidad en ese período    |
| `DNBR_UMBRAL`  | 0.25            | Subir si hay muchos falsos positivos           |
| `MIN_PIXELES`  | 100             | Subir para eliminar manchas más grandes        |
| `ESCALA`       | 20 m/px         | Bajar a 10 para mayor detalle (más lento)      |
| `N_TILES`      | 30              | Aumentar para llegar a 500 muestras finales    |

---

## Notas

**¿Por qué guardar la imagen POST y no la PRE?**
La imagen guardada es la composición POST-incendio: es lo que un modelo vería en producción
para detectar áreas quemadas. La máscara dNBR es el ground truth.

**Cobertura de nubes en Patagonia**
En verano (POST: febrero) la nubosidad suele ser baja. En primavera (PRE: noviembre)
puede haber más variabilidad. Si `.median()` devuelve píxeles enmascarados,
ampliar `FECHA_PRE_INI` hacia octubre 2025.

**Filtrar por balanceo después de la descarga**
El `metadata.csv` incluye `pct_quemado` por tile. Para quedarse solo con tiles
que cumplan el criterio de mínimo 10% quemado:

```python
import pandas as pd
df = pd.read_csv("dataset/metadata.csv")
balanceado = df[df["pct_quemado"] >= 10]
print(f"Tiles útiles: {len(balanceado)}/{len(df)}")
```
