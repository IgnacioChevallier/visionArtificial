"""
Local re-masking script — iterates parameter combinations on already-downloaded TIFs.
Generates comparison PNGs without calling Google Earth Engine.

Band order in saved TIFs: [B2, B3, B4, B8, B11, B12]  (1-indexed for rasterio)
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from itertools import product
from scipy.ndimage import label, binary_dilation

DATASET_DIR  = Path("dataset")
PRE_DIR      = DATASET_DIR / "imagenes_pre"
POST_DIR     = DATASET_DIR / "imagenes"
MASK_DIR     = DATASET_DIR / "mascaras"
OUT_DIR      = Path("remask_comparisons")
OUT_DIR.mkdir(exist_ok=True)

TILES = ["tile_001", "tile_004", "tile_008", "tile_009"]

# Band indices (0-based) within the saved TIF
IDX = {"B2": 0, "B3": 1, "B4": 2, "B8": 3, "B11": 4, "B12": 5}

# Fixed parameters
MIN_PIXELES   = 120
DILATACION_PX = 1

# Parameter grid to sweep
PARAM_GRID = list(product(
    [0.30, 0.25, 0.22, 0.20],  # DNBR_UMBRAL
    [0.15, 0.10],               # PRE_NBR_MIN
    [0.30, 0.35],               # POST_NBR_MAX
))


def read_bands(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def norm_diff(a, b):
    denom = a + b
    nd = np.where(denom == 0, 0.0, (a - b) / denom)
    return nd


def compute_mask(pre, post, dnbr_umbral, pre_nbr_min, post_nbr_max):
    pre_B8,  pre_B12 = pre[IDX["B8"]],  pre[IDX["B12"]]
    post_B8, post_B12 = post[IDX["B8"]], post[IDX["B12"]]
    post_B3, post_B11 = post[IDX["B3"]], post[IDX["B11"]]
    post_B2, post_B3v, post_B4 = post[IDX["B2"]], post[IDX["B3"]], post[IDX["B4"]]

    nbr_pre  = norm_diff(pre_B8,  pre_B12)
    nbr_post = norm_diff(post_B8, post_B12)
    dnbr     = nbr_pre - nbr_post

    ndsi      = norm_diff(post_B3, post_B11)
    vis       = np.stack([post_B2, post_B3v, post_B4], axis=0)
    brightness = vis.mean(axis=0)
    chroma    = vis.max(axis=0) - vis.min(axis=0)

    snow_or_bright = (ndsi > 0.35) | ((brightness > 3500) & (chroma < 500))

    burned_raw = (
        (dnbr > dnbr_umbral) &
        (nbr_pre > pre_nbr_min) &
        (nbr_post < post_nbr_max) &
        (~snow_or_bright)
    )

    # Connected-component filter (keep regions >= MIN_PIXELES)
    structure = np.ones((3, 3), dtype=int)
    labeled, _ = label(burned_raw, structure=structure)
    region_sizes = np.bincount(labeled.ravel())
    big_enough = region_sizes >= MIN_PIXELES
    big_enough[0] = False
    burned_clean = big_enough[labeled]

    # Dilation
    burned = binary_dilation(burned_clean, iterations=DILATACION_PX)
    return burned.astype(np.uint8), dnbr, nbr_pre, nbr_post


def rgb_from_bands(bands, gamma=0.8):
    r, g, b = bands[IDX["B4"]], bands[IDX["B3"]], bands[IDX["B2"]]
    rgb = np.stack([r, g, b], axis=-1)
    valid = rgb[rgb > 0]
    if valid.size == 0:
        return rgb
    p2, p98 = np.percentile(valid, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-9), 0, 1)
    return np.power(rgb, gamma)


def save_comparison(tile, pre, post, orig_mask, new_mask, params, label_str):
    dnbr_u, pre_min, post_max = params
    rgb_pre  = rgb_from_bands(pre)
    rgb_post = rgb_from_bands(post)

    overlay_orig = rgb_post.copy()
    overlay_orig[orig_mask == 1, 0] = 1.0
    overlay_orig[orig_mask == 1, 1] *= 0.3
    overlay_orig[orig_mask == 1, 2] *= 0.3

    overlay_new = rgb_post.copy()
    overlay_new[new_mask == 1, 0] = 1.0
    overlay_new[new_mask == 1, 1] *= 0.3
    overlay_new[new_mask == 1, 2] *= 0.3

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(
        f"{tile}  |  {label_str}\n"
        f"orig={orig_mask.mean()*100:.1f}%  new={new_mask.mean()*100:.1f}%  "
        f"gain={( new_mask.mean()-orig_mask.mean())*100:+.1f}%",
        fontsize=11
    )

    axes[0].imshow(rgb_pre);         axes[0].set_title("Pre-fire");        axes[0].axis("off")
    axes[1].imshow(rgb_post);        axes[1].set_title("Post-fire");       axes[1].axis("off")
    axes[2].imshow(orig_mask, cmap="gray_r", vmin=0, vmax=1)
    axes[2].set_title("Original mask"); axes[2].axis("off")
    axes[3].imshow(new_mask,  cmap="gray_r", vmin=0, vmax=1)
    axes[3].set_title("New mask");      axes[3].axis("off")
    axes[4].imshow(overlay_new);     axes[4].set_title("New overlay");     axes[4].axis("off")

    plt.tight_layout()
    fname = OUT_DIR / f"{tile}_{label_str.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def run_sweep():
    print(f"Sweeping {len(PARAM_GRID)} parameter combinations on {len(TILES)} tiles...\n")

    results = []

    for dnbr_u, pre_min, post_max in PARAM_GRID:
        label_str = f"dNBR{dnbr_u}_preNBR{pre_min}_postNBR{post_max}"
        total_gain = 0.0
        combo_results = []

        for tile in TILES:
            pre  = read_bands(PRE_DIR  / f"{tile}.tif")
            post = read_bands(POST_DIR / f"{tile}.tif")
            orig = read_bands(MASK_DIR / f"{tile}.tif")[0]

            new_mask, dnbr, nbr_pre, nbr_post = compute_mask(
                pre, post, dnbr_u, pre_min, post_max
            )

            gain = new_mask.mean() - (orig > 0.5).mean()
            total_gain += gain
            combo_results.append((tile, orig, new_mask, gain))

        results.append((dnbr_u, pre_min, post_max, total_gain, combo_results))

    # Sort by total area gain (most improvement first)
    results.sort(key=lambda x: x[3], reverse=True)

    print("Top 5 parameter combinations by total gain:")
    print(f"{'dNBR_U':>8} {'PRE_MIN':>8} {'POST_MAX':>9} {'gain%':>8}")
    for dnbr_u, pre_min, post_max, gain, _ in results[:5]:
        print(f"{dnbr_u:>8.2f} {pre_min:>8.2f} {post_max:>9.2f} {gain*100:>+7.2f}%")

    # Save comparisons for best 3 combos + the baseline (0.30, 0.15, 0.30)
    baseline = next((r for r in results if r[0] == 0.30 and r[1] == 0.15 and r[2] == 0.30), None)
    to_save = results[:3]
    if baseline and baseline not in to_save:
        to_save.append(baseline)

    print(f"\nSaving comparison images to {OUT_DIR}/")
    for dnbr_u, pre_min, post_max, gain, combo_results in to_save:
        label_str = f"dNBR{dnbr_u}_preNBR{pre_min}_postNBR{post_max}"
        for tile, orig, new_mask, tile_gain in combo_results:
            fname = save_comparison(
                tile, read_bands(PRE_DIR / f"{tile}.tif"),
                read_bands(POST_DIR / f"{tile}.tif"),
                (orig > 0.5).astype(np.uint8), new_mask,
                (dnbr_u, pre_min, post_max), label_str
            )
        print(f"  {label_str}  (total gain {gain*100:+.2f}%)")

    print("\nDone.")
    return results


if __name__ == "__main__":
    run_sweep()
