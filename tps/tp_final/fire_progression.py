"""
fire_progression.py — Progresión horaria del incendio "Puerto Café" vía NASA FIRMS.

Descarga detecciones de fuego activo VIIRS (SNPP + NOAA-20) de la NASA FIRMS API.
Cada detección incluye lat, lon, acq_date, acq_time (HHMM UTC), FRP, confidence.

Resolución temporal: ~4 pasadas/día combinando ambos satélites → gaps de ~3-6h.
Resistencia a nubes: IR térmico (3.74 µm) penetra humo moderado y nubes finas.

Requiere MAP_KEY en .env  →  https://firms.modaps.eosdis.nasa.gov/api/

Outputs:
  fire_progression.csv      — una fila por pasada × tile
  fire_arrival.csv          — primera detección por tile (reemplaza fire_timing.csv)
  fire_progression_map.tif  — raster 375m con timestamp de primera detección
"""

import os, csv, datetime
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box, Point
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ─────────────────────────────────────────────────────────────
MAP_KEY  = os.environ["MAP_KEY"]
BBOX     = "-71.917,-42.821,-71.359,-42.500"   # mismo ROI que generar_dataset.py
START    = "2025-12-08"
END      = "2026-01-31"

CONF_MIN  = "nominal"       # excluir detecciones de baja confianza
GAP_PASS  = 30             # minutos — detecciones dentro de este margen = misma pasada
CHUNK_DAYS = 5             # FIRMS API limita a máximo 5 días por request
N_TILES  = 30
ESCALA   = 20              # m/px de los tiles S2

OUTPUT_PROG  = Path("fire_progression.csv")
OUTPUT_ARR   = Path("fire_arrival.csv")
OUTPUT_MAP   = Path("fire_progression_map.tif")


# ── Grilla de tiles (igual que generar_dataset.py) ────────────────────────────
TILE_DEG = (ESCALA * 256) / 111320
lon_min, lat_min = -71.917, -42.821
lon_max, lat_max = -71.359, -42.500
lons = np.arange(lon_min, lon_max, TILE_DEG)
lats = np.arange(lat_min, lat_max, TILE_DEG)

tile_records = []
for lon in lons:
    for lat in lats:
        tile_records.append({
            "tile_id": f"tile_{len(tile_records):03d}",
            "geometry": box(lon, lat,
                            min(lon + TILE_DEG, lon_max),
                            min(lat + TILE_DEG, lat_max))
        })
tiles_gdf = gpd.GeoDataFrame(tile_records[:N_TILES], crs="EPSG:4326")
print(f"Tiles: {len(tiles_gdf)}")


# ── Descarga FIRMS ─────────────────────────────────────────────────────────────
def download_firms(satellite):
    """Descarga en chunks de CHUNK_DAYS días (límite de la API FIRMS)."""
    from io import StringIO
    start_dt = pd.Timestamp(START)
    end_dt   = pd.Timestamp(END)
    chunks   = []
    current  = start_dt
    while current <= end_dt:
        chunk_end = min(current + pd.Timedelta(days=CHUNK_DAYS - 1), end_dt)
        days_in   = (chunk_end - current).days + 1
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
               f"{MAP_KEY}/{satellite}/{BBOX}/{days_in}/{current.strftime('%Y-%m-%d')}")
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and resp.text.strip():
            try:
                chunk_df = pd.read_csv(StringIO(resp.text))
                if not chunk_df.empty:
                    chunks.append(chunk_df)
            except Exception:
                pass
        current += pd.Timedelta(days=CHUNK_DAYS)

    if not chunks:
        print(f"    → sin detecciones")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["satellite_src"] = satellite
    df["datetime_utc"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " +
        df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M", utc=True
    )
    if "confidence" in df.columns:
        df = df[df["confidence"] != "low"]
    df = df.drop_duplicates(subset=["latitude","longitude","datetime_utc"])
    print(f"    → {len(df)} detecciones (conf≥nominal)")
    return df

print("Descargando VIIRS SNPP…")
snpp = download_firms("VIIRS_SNPP_SP")
print("Descargando VIIRS NOAA-20…")
noaa = download_firms("VIIRS_NOAA20_SP")

all_det = pd.concat([snpp, noaa], ignore_index=True)
if all_det.empty:
    print("ERROR: sin detecciones. Verificar MAP_KEY y fechas.")
    exit(1)

all_det = all_det.sort_values("datetime_utc").reset_index(drop=True)
print(f"\nTotal detecciones combinadas: {len(all_det)}")
print(f"Rango: {all_det['datetime_utc'].min()} → {all_det['datetime_utc'].max()}")


# ── Spatial join: asignar detecciones a tiles ─────────────────────────────────
det_gdf = gpd.GeoDataFrame(
    all_det,
    geometry=gpd.points_from_xy(all_det["longitude"], all_det["latitude"]),
    crs="EPSG:4326"
)
joined = gpd.sjoin(det_gdf, tiles_gdf[["tile_id","geometry"]],
                   how="left", predicate="within")
in_roi = joined.dropna(subset=["tile_id"])
print(f"Detecciones dentro del ROI: {len(in_roi)}")


# ── Agrupar por pasada (detecciones < GAP_PASS min = misma pasada) ─────────────
def group_passes(df):
    """
    Dentro de cada tile, agrupa detecciones que ocurrieron en el mismo paso
    del satélite (diferencia < GAP_PASS minutos entre filas consecutivas).
    """
    df = df.sort_values("datetime_utc").copy()
    diffs = df["datetime_utc"].diff().dt.total_seconds().fillna(0)
    df["pass_group"] = (diffs > GAP_PASS * 60).cumsum()
    return df


prog_rows = []   # progresión por tile × pasada
arr_rows  = []   # llegada por tile

for tile_id, group in in_roi.groupby("tile_id"):
    group = group_passes(group)
    passes = (group.groupby("pass_group")
              .agg(
                  first_detection_utc=("datetime_utc", "min"),
                  n_pixels=("latitude", "count"),
                  mean_frp=("frp", "mean") if "frp" in group.columns else ("latitude", "count"),
                  satellite=("satellite_src", lambda x: x.mode()[0])
              )
              .reset_index(drop=True)
              .sort_values("first_detection_utc"))

    # Calcular lag entre pasadas consecutivas
    passes["lag_since_prev_h"] = (
        passes["first_detection_utc"].diff().dt.total_seconds() / 3600
    ).round(2)

    for _, row in passes.iterrows():
        prog_rows.append({
            "tile_id":              tile_id,
            "detection_utc":        row["first_detection_utc"].isoformat(),
            "satellite":            row["satellite"],
            "n_pixels_fire":        int(row["n_pixels"]),
            "mean_frp_mw":          round(row["mean_frp"], 1) if pd.notna(row.get("mean_frp")) else "",
            "lag_since_prev_h":     row["lag_since_prev_h"] if pd.notna(row["lag_since_prev_h"]) else "",
        })

    first = passes.iloc[0]
    last  = passes.iloc[-1]
    arr_rows.append({
        "tile_id":              tile_id,
        "first_fire_utc":       first["first_detection_utc"].isoformat(),
        "last_fire_utc":        last["first_detection_utc"].isoformat(),
        "n_passes_with_fire":   len(passes),
        "total_active_h":       round(
            (last["first_detection_utc"] - first["first_detection_utc"])
            .total_seconds() / 3600, 1),
        "precision":            "VIIRS_375m",
    })
    print(f"  {tile_id}: {len(passes)} pasadas  "
          f"[{first['first_detection_utc'].strftime('%m-%d %H:%M')} → "
          f"{last['first_detection_utc'].strftime('%m-%d %H:%M')} UTC]")


# ── Escribir CSVs ──────────────────────────────────────────────────────────────
prog_df = pd.DataFrame(prog_rows)
prog_df.to_csv(OUTPUT_PROG, index=False)
print(f"\n✓ {OUTPUT_PROG}  ({len(prog_df)} filas)")

arr_df = pd.DataFrame(arr_rows).sort_values("first_fire_utc")
arr_df.to_csv(OUTPUT_ARR, index=False)
print(f"✓ {OUTPUT_ARR}  ({len(arr_df)} tiles)")

if arr_rows:
    print("\nPrimeras detecciones por tile:")
    print(f"  {'Tile':10} {'Primera detección UTC':28} {'Pasadas':>8}")
    for r in arr_rows:
        print(f"  {r['tile_id']:10} {r['first_fire_utc']:28} {r['n_passes_with_fire']:>8}")


# ── Raster de primera detección (375m) ────────────────────────────────────────
# Cada pixel VIIRS = primera vez que se detectó fuego allí (epoch Unix en segundos)
if not in_roi.empty:
    PIXEL_DEG = 375 / 111320  # ~0.00337°
    cols = int((lon_max - lon_min) / PIXEL_DEG) + 1
    rows = int((lat_max - lat_min) / PIXEL_DEG) + 1
    raster = np.zeros((rows, cols), dtype=np.float64)

    for _, row in in_roi.sort_values("datetime_utc").iterrows():
        c = int((row["longitude"] - lon_min) / PIXEL_DEG)
        r = int((row["latitude"]  - lat_min) / PIXEL_DEG)
        if 0 <= r < rows and 0 <= c < cols:
            epoch = row["datetime_utc"].timestamp()
            if raster[r, c] == 0 or epoch < raster[r, c]:
                raster[r, c] = epoch

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, cols, rows)
    with rasterio.open(
        OUTPUT_MAP, "w", driver="GTiff",
        height=rows, width=cols, count=1,
        dtype=rasterio.float64, crs="EPSG:4326",
        transform=transform, nodata=0
    ) as dst:
        dst.write(raster, 1)
    print(f"✓ {OUTPUT_MAP}  ({cols}×{rows} px a ~375m)")
