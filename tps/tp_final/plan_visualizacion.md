# Plan: Visualización del Dataset — Los Alerces 2026

Script Python standalone que toma un tile del dataset generado y produce
una figura con **3 paneles**: imagen RGB, máscara binaria, y superposición.

---

## Dependencias

```bash
pip install rasterio matplotlib numpy
```

---

## Qué muestra cada panel

```
┌─────────────────┬─────────────────┬─────────────────┐
│   RGB (B4-B3-B2)│  Máscara binaria│   Superposición │
│   imagen POST   │  1=rojo 0=negro │  RGB + máscara  │
└─────────────────┴─────────────────┴─────────────────┘
```

- **Panel 1 — RGB**: composición color verdadero de la imagen Sentinel-2 POST.
- **Panel 2 — Máscara**: área quemada en rojo, no quemada en negro.
- **Panel 3 — Superposición**: RGB con la máscara quemada encima en rojo semitransparente.

---

## Script: `visualizar.py`

```python
import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────────────────
DATASET_DIR = Path("dataset")
IMG_DIR     = DATASET_DIR / "imagenes"
MASK_DIR    = DATASET_DIR / "mascaras"

# Índices de bandas dentro del .tif (0-based)
# Orden guardado: B2, B3, B4, B8, B11, B12
BANDA_R = 2  # B4 - Red
BANDA_G = 1  # B3 - Green
BANDA_B = 0  # B2 - Blue


def cargar_rgb(path_tif):
    """Lee las bandas RGB y devuelve array (H, W, 3) normalizado a [0, 1]."""
    with rasterio.open(path_tif) as src:
        r = src.read(BANDA_R + 1).astype(float)  # rasterio usa índices 1-based
        g = src.read(BANDA_G + 1).astype(float)
        b = src.read(BANDA_B + 1).astype(float)

    rgb = np.stack([r, g, b], axis=-1)

    # Normalización por percentil para mejor contraste visual
    p2, p98 = np.percentile(rgb[rgb > 0], (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    return rgb


def cargar_mascara(path_tif):
    """Lee la máscara binaria y devuelve array (H, W)."""
    with rasterio.open(path_tif) as src:
        return src.read(1).astype(float)


def visualizar(nombre_tile):
    path_img  = IMG_DIR  / f"{nombre_tile}.tif"
    path_mask = MASK_DIR / f"{nombre_tile}.tif"

    if not path_img.exists():
        print(f"Error: no se encontró {path_img}")
        sys.exit(1)
    if not path_mask.exists():
        print(f"Error: no se encontró {path_mask}")
        sys.exit(1)

    rgb     = cargar_rgb(path_img)
    mascara = cargar_mascara(path_mask)

    # Superposición: RGB con área quemada en rojo semitransparente
    overlay = rgb.copy()
    quemado = mascara == 1
    overlay[quemado, 0] = 1.0   # canal R al máximo
    overlay[quemado, 1] *= 0.3  # canal G atenuado
    overlay[quemado, 2] *= 0.3  # canal B atenuado

    pct = round(quemado.sum() / mascara.size * 100, 1)

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Tile: {nombre_tile}  |  Área quemada: {pct}%", fontsize=13)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB (color verdadero)")
    axes[0].axis("off")

    axes[1].imshow(mascara, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_title("Máscara binaria")
    axes[1].axis("off")
    parche_q  = mpatches.Patch(color="red",   label="Quemado (1)")
    parche_nq = mpatches.Patch(color="green", label="No quemado (0)")
    axes[1].legend(handles=[parche_q, parche_nq], loc="lower right", fontsize=8)

    axes[2].imshow(overlay)
    axes[2].set_title("Superposición")
    axes[2].axis("off")

    plt.tight_layout()

    salida = f"{nombre_tile}_visualizacion.png"
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"Guardado: {salida}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Sin argumento: visualiza el primer tile disponible
        tiles = sorted(IMG_DIR.glob("*.tif"))
        if not tiles:
            print("No se encontraron tiles en dataset/imagenes/")
            sys.exit(1)
        nombre = tiles[0].stem
        print(f"Tile no especificado. Usando: {nombre}")
    else:
        nombre = sys.argv[1].replace(".tif", "")

    visualizar(nombre)
```

---

## Uso

```bash
# Visualizar el primer tile disponible
python visualizar.py

# Visualizar un tile específico
python visualizar.py tile_000
python visualizar.py tile_007
```

Genera un `.png` en el directorio actual y abre la figura en pantalla.

---

## Visualizar todos los tiles de una vez

Para hacer un repaso rápido de todo el dataset, agregar al final del script:

```bash
# En bash: recorre todos los tiles y genera un PNG por cada uno
for f in dataset/imagenes/*.tif; do
    python visualizar.py "$(basename "$f" .tif)"
done
```

---

## Notas

**Normalización por percentil**
Sentinel-2 en formato reflectancia tiene valores típicos entre 0 y 10000.
La normalización por percentil 2–98 descarta outliers y mejora el contraste
visual sin afectar los valores de los `.tif` originales.

**Resolución de la máscara**
Si la máscara tiene distinto tamaño que la imagen (por diferencia de resolución),
`matplotlib` la escala automáticamente para visualización. Los archivos en disco
no se modifican.
