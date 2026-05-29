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
MASK_DIR = DATASET_DIR / "mascaras"
SAVED_DIR = BASE_DIR / "saved"

# Orden guardado: B2, B3, B4, B8, B11, B12
BANDA_R = 2  # B4 - Red
BANDA_G = 1  # B3 - Green
BANDA_B = 0  # B2 - Blue
GAMMA = 0.8


def buscar_pares_tiles():
    tiles_post = {path.stem: path for path in POST_IMG_DIR.glob("*.tif")}
    tiles_mask = {path.stem: path for path in MASK_DIR.glob("*.tif")}

    if not tiles_post:
        print(f"Error: no se encontraron tiles POST en {POST_IMG_DIR}")
        sys.exit(1)

    if not tiles_mask:
        print(f"Error: no se encontraron máscaras en {MASK_DIR}")
        sys.exit(1)

    nombres_comunes = sorted(tiles_post.keys() & tiles_mask.keys())
    if not nombres_comunes:
        print("Error: no hay tiles POST con máscara correspondiente.")
        sys.exit(1)

    faltan_mascaras = sorted(tiles_post.keys() - tiles_mask.keys())
    faltan_post = sorted(tiles_mask.keys() - tiles_post.keys())

    if faltan_mascaras:
        print("Advertencia: se omiten tiles POST sin máscara:")
        print("  " + ", ".join(faltan_mascaras))

    if faltan_post:
        print("Advertencia: se omiten máscaras sin tile POST:")
        print("  " + ", ".join(faltan_post))

    return [(tiles_post[nombre], tiles_mask[nombre]) for nombre in nombres_comunes]


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

    valores_validos = rgb[np.isfinite(rgb) & (rgb > 0)]
    if valores_validos.size == 0:
        raise ValueError("El mosaico POST no contiene valores positivos válidos para RGB.")

    p2, p98 = np.percentile(valores_validos, (2, 98))
    if p98 <= p2:
        raise ValueError("No se pudo normalizar RGB: percentiles inválidos.")

    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    return np.power(rgb, GAMMA)


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


def crear_ruta_guardado():
    SAVED_DIR.mkdir(exist_ok=True)
    contador = 1

    while True:
        salida = SAVED_DIR / f"roi_post_mascara_{contador:02d}.png"
        if not salida.exists():
            return salida
        contador += 1


def guardar_visualizacion(overlay, quemado, salida):
    pct_quemado = round(quemado.sum() / quemado.size * 100, 2)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(overlay)
    ax.set_title(f"ROI completo - POST con máscara quemada ({pct_quemado}% quemado)")
    ax.axis("off")

    fig.savefig(salida, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    pares = buscar_pares_tiles()
    paths_post = [post for post, _ in pares]
    paths_mask = [mask for _, mask in pares]

    print(f"Mergeando {len(pares)} pares de tiles...")
    mosaico_post, _ = mergear_rasters(paths_post)
    mosaico_mascara, _ = mergear_rasters(paths_mask)

    rgb = normalizar_rgb(mosaico_post)
    overlay, quemado = superponer_mascara(rgb, mosaico_mascara)

    salida = crear_ruta_guardado()
    guardar_visualizacion(overlay, quemado, salida)
    print(f"Guardado: {salida}")


if __name__ == "__main__":
    main()
