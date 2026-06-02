# TP Final - Dataset satelital de áreas quemadas

Este proyecto genera un dataset de áreas quemadas para la zona del Parque Nacional Los Alerces a partir de imágenes Sentinel-2. Existen dos flujos de trabajo complementarios:

- **Dataset estático** (`generar_dataset.py`): composiciones PRE/POST con ventanas temporales fijas, optimizadas para calidad de máscara.
- **Dataset temporal** (`fire_timing.py` + `generar_dataset_temporal.py`): imágenes individuales seleccionadas según el momento exacto en que el fuego pasó por cada tile, minimizando el desfase temporal para validar modelos de predicción de progreso de incendio.

## Objetivo

Construir un conjunto de datos con:

- imágenes RGB previas al incendio;
- imágenes RGB posteriores al incendio;
- máscaras binarias de áreas quemadas;
- metadatos por tile: coordenadas, fechas, porcentaje quemado y (en el dataset temporal) hora de llegada del fuego.

La máscara se obtiene sin aprendizaje automático ni etiquetas externas: se deriva directamente del índice `dNBR` calculado entre composiciones `PRE` y `POST`.

---

## Estructura del proyecto

```text
tp_final/
├── generar_dataset.py           # flujo estático: composites + máscaras
├── generar_dataset_temporal.py  # flujo temporal: imágenes por fecha de quemado
├── fire_timing.py               # detecta cuándo quemó cada tile (MODIS + S2)
├── remask_local.py              # re-aplica máscaras localmente sin GEE
├── visualizar.py                # explorador interactivo de tiles
├── requirements.txt
├── fire_timing.csv              # output de fire_timing.py
├── docs/
│   └── README.md
├── dataset/                     # output de generar_dataset.py
│   ├── imagenes_pre/            # composites PRE (multibanda .tif)
│   ├── imagenes/                # composites POST (multibanda .tif)
│   ├── mascaras/                # máscaras binarias .tif
│   └── metadata.csv
└── dataset_temporal/            # output de generar_dataset_temporal.py
    ├── imagenes_pre/            # imagen individual más cercana pre-fuego
    ├── imagenes/                # imagen individual más cercana post-fuego
    ├── mascaras/                # mismas máscaras binarias (composite dNBR)
    └── metadata_temporal.csv   # incluye lag_pre_h, lag_post_h, precision
```

Cada tile exportado contiene las bandas Sentinel-2 `B2`, `B3`, `B4`, `B8`, `B11` y `B12`. Para visualizar color verdadero se usan `B4`, `B3` y `B2` como canales rojo, verde y azul.

---

## Flujo estático: `generar_dataset.py`

### 1. Región de interés

```python
ROI = ee.Geometry.Rectangle([-71.917, -42.821, -71.359, -42.500])
```

Cubre el área del incendio "Puerto Café" en el Parque Nacional Los Alerces, Argentina.

### 2. Selección temporal

| Composición | Fechas | Motivo |
|-------------|--------|--------|
| `PRE` | `2025-11-01` a `2025-12-08` | antes del inicio del incendio |
| `POST` | `2026-02-01` a `2026-03-08` | cicatriz ya formada, fuego contenido |

### 3. Filtrado de escenas y nubes

Se filtran escenas con menos de `20%` de nubosidad global. Luego se enmascaran píxeles con la banda `SCL` para excluir sombras de nube (`3`), nubes de prob. media (`8`), alta (`9`), cirrus (`10`) y nieve (`11`).

Se mantienen dos versiones de cada composición:
- `pre_masked` / `post_masked`: para calcular dNBR (menos contaminación atmosférica).
- `pre_full` / `post_full`: para exportar RGB sin huecos negros en la visualización.

### 4. Cálculo de NBR y dNBR

```text
NBR  = (B8 - B12) / (B8 + B12)
dNBR = NBR_pre - NBR_post
```

Un valor alto de `dNBR` indica caída marcada de vegetación sana, patrón compatible con área quemada.

### 5. Binarización y limpieza

La máscara se obtiene combinando cuatro condiciones (todas deben cumplirse):

```text
quemado = (dNBR > DNBR_UMBRAL)
          AND (NBR_pre > PRE_NBR_MIN)      ← había vegetación antes
          AND (NBR_post < POST_NBR_MAX)    ← la vegetación desapareció
          AND NOT(nieve_o_gris_claro)      ← no es nieve ni nubes brillantes
```

El filtro de nieve/brillo combina NDSI y brillo visible:

```text
nieve_o_gris = (NDSI_post > 0.35) OR (brightness > 3500 AND chroma < 500)
```

Después se eliminan componentes conexas menores a `MIN_PIXELES` píxeles y se dilata el resultado en `DILATACION_PX` píxel para unir bordes próximos.

### 6. Exportación por tiles

La región se divide en tiles de `256 × 256` píxeles a `20 m/px` (~5.12 km por lado). Para cada tile se exportan imagen PRE, POST, máscara binaria y una fila en `metadata.csv`.

### Parámetros principales

| Variable | Valor | Función |
|----------|-------|---------|
| `DNBR_UMBRAL` | `0.22` | umbral mínimo de cambio espectral |
| `PRE_NBR_MIN` | `0.10` | vegetación mínima pre-fuego |
| `POST_NBR_MAX` | `0.40` | NBR máximo post-fuego (captura quemas de baja severidad) |
| `MIN_PIXELES` | `60` | tamaño mínimo de región conectada (~2.4 ha a 20 m/px) |
| `DILATACION_PX` | `1` | expansión de bordes |
| `ESCALA` | `20` | resolución espacial en m/px |
| `N_TILES` | `30` | cantidad máxima de tiles |

> **Nota sobre `MIN_PIXELES = 60`:** Reducir de 120 a 60 captura manchas pequeñas reales (área mínima ~2.4 ha) sin introducir ruido. Clusters de 1–59 px a 20 m/px corresponden a áreas menores a 2.4 ha que se consideran ruido o artefactos.

---

## Re-enmascaramiento local: `remask_local.py`

Permite iterar parámetros de máscara **sin volver a descargar datos de GEE**, usando los TIF ya descargados en `dataset/`.

```bash
python remask_local.py
```

Genera imágenes comparativas en `remask_comparisons/` para cada combinación de parámetros definida en `PARAM_GRID`. Útil para ajustar umbrales rápidamente.

Los índices se calculan localmente a partir de las bandas guardadas:

| Banda en el TIF | Sentinel-2 | Uso |
|-----------------|-----------|-----|
| 1 (idx 0) | B2 | brillo, chroma, NDSI |
| 2 (idx 1) | B3 | brillo, chroma, NDSI |
| 3 (idx 2) | B4 | brillo, chroma |
| 4 (idx 3) | B8 | NBR (NIR) |
| 5 (idx 4) | B11 | NDSI |
| 6 (idx 5) | B12 | NBR (SWIR) |

---

## Detección de timing de fuego: `fire_timing.py`

Determina **cuándo pasó el frente de fuego por cada tile** durante el incendio "Puerto Café" (diciembre 2025 – enero 2026). La precisión obtenida varía según los datos disponibles.

### Fuentes de datos

| Fuente | Colección GEE | Resolución temporal | Resolución espacial | Cobertura |
|--------|--------------|---------------------|---------------------|-----------|
| MODIS Terra | `MODIS/061/MOD14A1` | Diario | 1 km | ✅ dic 2025 – ene 2026 |
| MODIS Aqua | `MODIS/061/MYD14A1` | Diario | 1 km | ✅ dic 2025 – ene 2026 |
| Sentinel-2 | `COPERNICUS/S2_SR_HARMONIZED` | ~5 días | 20 m | ✅ ventana de cambio |

> **GOES-16** (`NOAA/GOES/16/FDCF`, 10 min) solo está disponible en GEE hasta abril 2025 y no cubre este período.

### Tiempos de sobrepaso MODIS sobre Los Alerces (-42.6°S, -71.5°W)

| Satélite | Sobrepaso local | UTC aprox. |
|----------|----------------|------------|
| Terra (MOD14A1) | ~10:30 AM | ~15:15 UTC |
| Aqua (MYD14A1) | ~13:30 PM | ~18:15 UTC |

Combinando ambos se obtienen hasta dos detecciones por día con diferencia de ~3 horas.

### Algoritmo

**Fase 1A — Quemas anteriores al 4 de enero:**
Compara NBR(Nov 25) con NBR(Jan 4). Si `dNBR_mean > 0.20` o `dNBR_p75 > 0.30` en el tile, se considera quemado antes del Jan 4 y se asigna ventana `[Dec 8, Jan 4]`.

**Fase 1B — Quemas posteriores al 4 de enero:**
Jan 4 sirve de referencia (imagen 0% nubosidad). Se compara contra Jan 9, 11, 19, 24 y 29 consecutivamente. Primer fecha donde `dNBR_mean > 0.10` o `dNBR_p75 > 0.15` = ventana de quemado para ese tile.

**Fase 2 — Refinamiento con MODIS:**
Dentro de cada ventana S2, se busca el primer día en que MODIS Terra o Aqua detecta píxeles de fuego activo (`FireMask ≥ 7`) en el tile. El tiempo de llegada se estima según el sobrepaso del satélite que lo detectó (~3h de precisión).

### Output: `fire_timing.csv`

| Columna | Descripción |
|---------|-------------|
| `fire_arrival_utc` | Mejor estimado de llegada del fuego (ISO 8601, UTC) |
| `precision` | `MODIS_~3h`, `S2_window_Nd` o `unknown` |
| `s2_window_start/end` | Ventana S2 que acota el quemado |
| `modis_first_date` | Día de primera detección MODIS |
| `modis_first_source` | `Terra` o `Aqua` |

### Progresión del fuego detectada

El análisis mostró que el fuego activo se propagó principalmente en **enero 2026**:

| Fecha (UTC) | Tiles | Satélite |
|-------------|-------|---------|
| 2026-01-06 | 009, 015, 016, 023 | Terra + Aqua |
| 2026-01-07 | 022, 029 | Terra |
| 2026-01-08 | 024 | Aqua |
| 2026-01-09 | 002 | Aqua |
| 2026-01-10 | 008, 025 | Aqua |
| 2026-01-19 | 028 | Aqua |
| 2026-01-21 | 014, 021 | Terra |
| 2026-01-27 | 017, 018 | Terra |

```bash
python fire_timing.py
```

---

## Dataset temporal: `generar_dataset_temporal.py`

Usa `fire_timing.csv` para descargar **imágenes Sentinel-2 individuales** (no composites) lo más cercanas posible al momento de quemado de cada tile.

### Motivación

La ventana temporal entre la imagen y el evento de fuego importa para validar modelos de predicción de progreso de incendio: menor desfase → menor incertidumbre. El dataset estático tiene un gap de 60–90 días; el temporal lo reduce a 5–15 días para los tiles con detección MODIS.

| Dataset | Imagen PRE | Imagen POST | Gap típico |
|---------|-----------|------------|------------|
| Estático | composite nov-dic 2025 | composite feb-mar 2026 | 60–90 días |
| **Temporal** | **imagen individual** | **imagen individual** | **5–15 días** |

### Lógica de selección por tile

Para cada tile con fecha de quemado `fire_date`:

```text
PRE  = imagen S2 más reciente con < 20% nubes en [fire_date - 20d, fire_date]
POST = imagen S2 más temprana con < 20% nubes en [fire_date + 1d, fire_date + 20d]
```

Si no existe imagen dentro de la ventana, el tile se marca `SKIP` en el CSV.

La **máscara binaria** se mantiene idéntica a la del dataset estático (composite dNBR), ya que la calidad de segmentación es mejor con múltiples imágenes.

### Ejemplos de gaps obtenidos

| Tile | Fuego llega | PRE (lag) | POST (lag) | Gap total |
|------|-------------|-----------|-----------|-----------|
| tile_009 | Jan 6 | Jan 4 (+48 h) | Jan 9 (+72 h) | **5 días** |
| tile_008 | Jan 10 | Jan 9 (+24 h) | Jan 19 (+216 h) | ~10 días |
| tile_014 | Jan 21 | Jan 19 (+48 h) | Jan 24 (+72 h) | **5 días** |

### Output: `dataset_temporal/metadata_temporal.csv`

| Columna | Descripción |
|---------|-------------|
| `fire_arrival_utc` | Hora de llegada del fuego (de `fire_timing.csv`) |
| `timing_precision` | Precisión del timing (`MODIS_~3h`, `S2_window_Nd`) |
| `fecha_pre_img` | Fecha de la imagen PRE descargada |
| `lag_pre_h` | Horas entre la imagen PRE y el fuego |
| `fecha_post_img` | Fecha de la imagen POST descargada |
| `lag_post_h` | Horas entre el fuego y la imagen POST |
| `pct_quemado` | Porcentaje de área quemada (de la máscara dNBR) |
| `status` | `OK` o `SKIP_no_S2` |

```bash
python generar_dataset_temporal.py
```

---

## Visualización

```bash
# Explorar el dataset estático
python visualizar.py

# Abrir desde un tile específico
python visualizar.py tile_008

# El visualizador también funciona apuntando a dataset_temporal/
# (editar las rutas PRE_IMG_DIR, POST_IMG_DIR, MASK_DIR al inicio del script)
```

Controles:
- ← →: navegar entre tiles
- `s`: guardar la figura como PNG
- `q`: cerrar

La visualización normaliza por percentiles 2–98 y aplica corrección gamma `0.8`. Esto no modifica los valores almacenados en los `.tif`.

---

## Uso completo

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Autenticar Google Earth Engine (una vez)
earthengine authenticate

# 3. Generar dataset estático (composites, mejor máscara)
python generar_dataset.py

# 4. Detectar timing de fuego por tile
python fire_timing.py
# → genera fire_timing.csv

# 5. Generar dataset temporal (imágenes individuales, menor gap)
python generar_dataset_temporal.py
# → genera dataset_temporal/

# 6. Visualizar resultados
python visualizar.py
```

### Re-ajustar parámetros de máscara sin bajar datos nuevos

```bash
python remask_local.py
# → genera comparativas en remask_comparisons/
```

---

## Dependencias

Ver `requirements.txt`. Las principales son:

| Paquete | Uso |
|---------|-----|
| `earthengine-api` | acceso a Google Earth Engine |
| `geemap` | descarga de imágenes GEE como GeoTIFF |
| `numpy` | operaciones matriciales |
| `rasterio` | lectura/escritura de GeoTIFF |
| `matplotlib` | visualización |
| `scipy` | operaciones morfológicas (remask_local.py) |
