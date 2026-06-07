import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.merge import merge


# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
POST_IMG_DIR = DATASET_DIR / "imagenes"
PRE_IMG_DIR  = DATASET_DIR / "imagenes_pre"
MASK_DIR = DATASET_DIR / "mascaras"
SAVED_DIR = BASE_DIR / "saved"

# Orden guardado: B2, B3, B4, B8, B11, B12
BANDA_R = 2  # B4 - Red
BANDA_G = 1  # B3 - Green
BANDA_B = 0  # B2 - Blue
GAMMA = 0.8


def buscar_pares_tiles(img_dir):
    etiqueta = "PRE" if img_dir == PRE_IMG_DIR else "POST"
    tiles_img  = {path.stem: path for path in img_dir.glob("*.tif")}
    tiles_mask = {path.stem: path for path in MASK_DIR.glob("*.tif")}

    if not tiles_img:
        print(f"Error: no se encontraron tiles {etiqueta} en {img_dir}")
        sys.exit(1)

    if not tiles_mask:
        print(f"Error: no se encontraron máscaras en {MASK_DIR}")
        sys.exit(1)

    nombres_comunes = sorted(tiles_img.keys() & tiles_mask.keys())
    if not nombres_comunes:
        print(f"Error: no hay tiles {etiqueta} con máscara correspondiente.")
        sys.exit(1)

    faltan_mascaras = sorted(tiles_img.keys() - tiles_mask.keys())
    faltan_img      = sorted(tiles_mask.keys() - tiles_img.keys())

    if faltan_mascaras:
        print(f"Advertencia: se omiten tiles {etiqueta} sin máscara:")
        print("  " + ", ".join(faltan_mascaras))

    if faltan_img:
        print(f"Advertencia: se omiten máscaras sin tile {etiqueta}:")
        print("  " + ", ".join(faltan_img))

    return [(tiles_img[nombre], tiles_mask[nombre]) for nombre in nombres_comunes]


def mergear_rasters(paths):
    datasets = [rasterio.open(path) for path in paths]
    try:
        mosaico, transform = merge(datasets)
    finally:
        for dataset in datasets:
            dataset.close()

    return mosaico, transform


def normalizar_rgb(mosaico_post):
    rgb = np.stack(
        [
            mosaico_post[BANDA_R],
            mosaico_post[BANDA_G],
            mosaico_post[BANDA_B],
        ],
        axis=-1,
    ).astype(float)

    nodata_mask = np.all(rgb == 0, axis=-1)

    valores_validos = rgb[np.isfinite(rgb) & (rgb > 0)]
    if valores_validos.size == 0:
        raise ValueError("El mosaico POST no contiene valores positivos válidos para RGB.")

    p2, p98 = np.percentile(valores_validos, (2, 98))
    if p98 <= p2:
        raise ValueError("No se pudo normalizar RGB: percentiles inválidos.")

    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    rgb = np.power(rgb, GAMMA)
    rgb[nodata_mask] = 1.0  # nodata → blanco
    return rgb


def superponer_mascara(rgb, mosaico_mascara):
    mascara = np.squeeze(mosaico_mascara)
    if mascara.shape != rgb.shape[:2]:
        raise ValueError(
            "La máscara mergeada no coincide en tamaño con el mosaico POST "
            f"({mascara.shape} != {rgb.shape[:2]})."
        )

    overlay = rgb.copy()
    quemado = mascara == 1
    overlay[quemado, 0] = 1.0
    overlay[quemado, 1] *= 0.3
    overlay[quemado, 2] *= 0.3
    return overlay, quemado


def crear_ruta_guardado(epoca, con_mascara):
    SAVED_DIR.mkdir(exist_ok=True)
    contador = 1
    sufijo = "_mascara" if con_mascara else ""
    prefijo = f"roi_{epoca}{sufijo}"

    while True:
        salida = SAVED_DIR / f"{prefijo}_{contador:02d}.png"
        if not salida.exists():
            return salida
        contador += 1


def guardar_visualizacion(imagen, salida, epoca, quemado=None):
    label = "PRE" if epoca == "pre" else "POST"
    if quemado is not None:
        pct_quemado = round(quemado.sum() / quemado.size * 100, 2)
        titulo = f"ROI completo - {label} con máscara quemada ({pct_quemado}% quemado)"
    else:
        titulo = f"ROI completo - {label}"

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(imagen)
    ax.set_title(titulo)
    ax.axis("off")

    fig.savefig(salida, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualiza el ROI PRE o POST con o sin máscara.")
    parser.add_argument("--prev",   action="store_true", help="Usar imágenes PRE-incendio en lugar de POST.")
    parser.add_argument("--nomask", action="store_true", help="No superponer la máscara quemada.")
    args = parser.parse_args()

    img_dir = PRE_IMG_DIR if args.prev else POST_IMG_DIR
    epoca   = "pre" if args.prev else "post"

    pares = buscar_pares_tiles(img_dir)
    paths_img = [img for img, _ in pares]

    print(f"Mergeando {len(pares)} tiles ({epoca.upper()})...")
    mosaico, _ = mergear_rasters(paths_img)
    rgb = normalizar_rgb(mosaico)

    if args.nomask:
        salida = crear_ruta_guardado(epoca, con_mascara=False)
        guardar_visualizacion(rgb, salida, epoca)
    else:
        paths_mask = [mask for _, mask in pares]
        mosaico_mascara, _ = mergear_rasters(paths_mask)
        overlay, quemado = superponer_mascara(rgb, mosaico_mascara)
        salida = crear_ruta_guardado(epoca, con_mascara=True)
        guardar_visualizacion(overlay, salida, epoca, quemado=quemado)

    print(f"Guardado: {salida}")


if __name__ == "__main__":
    main()
