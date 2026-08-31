import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATASET_DIR  = BASE_DIR / "dataset"
PRE_IMG_DIR  = DATASET_DIR / "imagenes_pre"
POST_IMG_DIR = DATASET_DIR / "imagenes"
MASK_DIR     = DATASET_DIR / "mascaras"
SAVED_DIR    = BASE_DIR / "saved"

if "s" in plt.rcParams["keymap.save"]:
    plt.rcParams["keymap.save"].remove("s")

# Índices de bandas dentro del .tif (0-based)
# Orden guardado: B2, B3, B4, B8, B11, B12
BANDA_R = 2  # B4 - Red
BANDA_G = 1  # B3 - Green
BANDA_B = 0  # B2 - Blue
GAMMA = 0.8  # Menor que 1 aclara la imagen para visualización


def cargar_rgb(path_tif):
    with rasterio.open(path_tif) as src:
        r = src.read(BANDA_R + 1).astype(float)
        g = src.read(BANDA_G + 1).astype(float)
        b = src.read(BANDA_B + 1).astype(float)

    rgb = np.stack([r, g, b], axis=-1)
    p2, p98 = np.percentile(rgb[rgb > 0], (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    return np.power(rgb, GAMMA)


def cargar_mascara(path_tif):
    with rasterio.open(path_tif) as src:
        return src.read(1).astype(float)


def dibujar(fig, axes, nombre_tile):
    path_pre  = PRE_IMG_DIR  / f"{nombre_tile}.tif"
    path_post = POST_IMG_DIR / f"{nombre_tile}.tif"
    path_mask = MASK_DIR     / f"{nombre_tile}.tif"

    if not path_pre.exists() or not path_post.exists() or not path_mask.exists():
        print(f"Advertencia: archivos no encontrados para {nombre_tile}")
        return

    rgb_pre  = cargar_rgb(path_pre)
    rgb_post = cargar_rgb(path_post)
    mascara  = cargar_mascara(path_mask)

    overlay = rgb_post.copy()
    quemado = mascara == 1
    overlay[quemado, 0] = 1.0
    overlay[quemado, 1] *= 0.3
    overlay[quemado, 2] *= 0.3

    pct = round(quemado.sum() / mascara.size * 100, 1)

    for ax in axes:
        ax.cla()

    fig.suptitle(f"Tile: {nombre_tile}  |  Área quemada: {pct}%", fontsize=13)

    axes[0].imshow(rgb_pre)
    axes[0].set_title("Antes del incendio")
    axes[0].axis("off")

    axes[1].imshow(rgb_post)
    axes[1].set_title("Después del incendio")
    axes[1].axis("off")

    axes[2].imshow(mascara, cmap="gray_r", vmin=0, vmax=1)
    axes[2].set_title("Máscara binaria")
    axes[2].axis("off")
    parche_q  = mpatches.Patch(color="black", label="Quemado (1)")
    parche_nq = mpatches.Patch(color="white", label="No quemado (0)")
    axes[2].legend(handles=[parche_q, parche_nq], loc="lower right", fontsize=8)

    axes[3].imshow(overlay)
    axes[3].set_title("Superposición")
    axes[3].axis("off")

    fig.canvas.draw_idle()


def crear_ruta_guardado(nombre_tile, sufijo="visualizacion"):
    SAVED_DIR.mkdir(exist_ok=True)
    contador = 1
    while True:
        salida = SAVED_DIR / f"{nombre_tile}_{sufijo}_{contador:02d}.png"
        if not salida.exists():
            return salida
        contador += 1


def explorar(tiles, indice_inicial):
    state = {"idx": indice_inicial}

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.text(0.5, 0.01, "← → para navegar  |  's' para guardar  |  't' para guardar con fondo transparente  |  'q' para salir",
             ha="center", fontsize=9, color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    dibujar(fig, axes, tiles[state["idx"]].stem)

    def on_key(event):
        if event.key == "right":
            state["idx"] = (state["idx"] + 1) % len(tiles)
        elif event.key == "left":
            state["idx"] = (state["idx"] - 1) % len(tiles)
        elif event.key == "s":
            nombre = tiles[state["idx"]].stem
            salida = crear_ruta_guardado(nombre)
            fig.savefig(salida, dpi=150, bbox_inches="tight")
            print(f"Guardado: {salida}")
            return
        elif event.key == "t":
            nombre = tiles[state["idx"]].stem
            salida = crear_ruta_guardado(nombre, sufijo="transparente")
            fig.savefig(salida, dpi=150, bbox_inches="tight", transparent=True)
            print(f"Guardado (transparente): {salida}")
            return
        elif event.key == "q":
            plt.close(fig)
            return
        else:
            return

        dibujar(fig, axes, tiles[state["idx"]].stem)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tiles = sorted(POST_IMG_DIR.glob("*.tif"))
    if not tiles:
        print("No se encontraron tiles en dataset/imagenes/")
        sys.exit(1)

    if len(sys.argv) >= 2:
        nombre = sys.argv[1].replace(".tif", "")
        stems  = [t.stem for t in tiles]
        indice = stems.index(nombre) if nombre in stems else 0
    else:
        indice = 0
        print(f"Navegando desde: {tiles[0].stem}  ({len(tiles)} tiles disponibles)")

    explorar(tiles, indice)
