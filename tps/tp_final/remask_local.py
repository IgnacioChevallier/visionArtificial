"""
Local re-masking script — evaluates parameter candidates on already-downloaded TIFs.
Generates comparison PNGs without calling Google Earth Engine.

Band order in saved TIFs: [B2, B3, B4, B8, B11, B12]  (0-indexed)

Two tile groups:
  FP_TILES  — tiles with known false positives (want LESS detected area)
  REF_TILES — tiles with real burns (want to PRESERVE detected area)

Scoring: best candidate minimises FP area while keeping REF area high.
Note: tiles 006 and 030 have heavy cloud/snow in the POST TIF (full composite),
so their local NBR differs from GEE (which uses cloud-masked medians). Their
scores are marked with (*) and treated as indicative only.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from itertools import product
from scipy.ndimage import label, binary_dilation

DATASET_DIR = Path("dataset")
PRE_DIR     = DATASET_DIR / "imagenes_pre"
POST_DIR    = DATASET_DIR / "imagenes"
MASK_DIR    = DATASET_DIR / "mascaras"
OUT_DIR     = Path("remask_comparisons")
OUT_DIR.mkdir(exist_ok=True)

# Known false-positive tiles
FP_TILES  = ["tile_003", "tile_006", "tile_007", "tile_014",  # mountain / snow
             "tile_052", "tile_053", "tile_061", "tile_062"]  # dried lakes / rivers / grass
# Cloud/snow affected — local NBR unreliable, shown for reference only
CLOUD_TILES = []
# Tiles with real burns — must not lose too much detection
REF_TILES = ["tile_001", "tile_004", "tile_008", "tile_009",  # confirmed burns
             "tile_016", "tile_017"]                           # steppe burns currently under-detected

IDX = {"B2": 0, "B3": 1, "B4": 2, "B8": 3, "B11": 4, "B12": 5}

# Fixed parameters (same as final GEE params)
MIN_PIXELES   = 150
DILATACION_PX = 1
NDSI_MAX      = 0.25   # more aggressive snow filter
BRILLO_MAX    = 3500
CROMA_MIN     = 500
NDWI_AGUA_MAX = 0.05   # pre-fire pixels with NDWI above this are water, excluded

# Candidate parameter sets to evaluate
# Format: (DNBR_UMBRAL, PRE_NBR_MIN, POST_NBR_MAX, NDSI_MAX, NDWI_MAX, MIN_PIXELES)
#
# Tensions to resolve:
#   FN 016/017 (steppe under-detected): lower DNBR_UMBRAL, lower PRE_NBR_MIN, higher POST_NBR_MAX
#   FP 052/053/061 (dried lakes/rivers): higher NDWI_MAX (excludes moist lake margins in PRE)
#   FP 003/006/007/014 (mountain/snow):  higher PRE_NBR_MIN, lower NDSI_MAX, higher MIN_PIXELES
#   → PRE_NBR_MIN is the shared knob; compensate with MIN_PIXELES for mountain blobs
CANDIDATES = [
    (0.30, 0.15, 0.40, 0.25, 0.05, 150),  # current baseline
    # --- fix steppe FN: relax POST_NBR_MAX + lower dNBR ---
    (0.27, 0.15, 0.45, 0.25, 0.05, 150),
    (0.27, 0.10, 0.45, 0.25, 0.05, 200),
    # --- fix water FP: raise NDWI_MAX to exclude lake/river margins ---
    (0.30, 0.15, 0.40, 0.25, 0.12, 150),
    (0.27, 0.15, 0.45, 0.25, 0.12, 150),
    # --- fix mountain FP: higher PRE_NBR_MIN + tighter snow ---
    (0.27, 0.20, 0.45, 0.18, 0.12, 200),
    # --- combined best guess ---
    (0.27, 0.15, 0.45, 0.18, 0.12, 200),
    (0.25, 0.10, 0.50, 0.18, 0.12, 250),
]


def read_bands(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def norm_diff(a, b):
    denom = a + b
    return np.where(denom == 0, 0.0, (a - b) / denom)


def compute_mask(pre, post, dnbr_umbral, pre_nbr_min, post_nbr_max,
                 ndsi_max=NDSI_MAX, ndwi_agua_max=NDWI_AGUA_MAX, min_px=MIN_PIXELES):
    pre_B3,  pre_B8,  pre_B12  = pre[IDX["B3"]],  pre[IDX["B8"]],  pre[IDX["B12"]]
    post_B8, post_B12 = post[IDX["B8"]], post[IDX["B12"]]
    post_B2, post_B3, post_B4 = post[IDX["B2"]], post[IDX["B3"]], post[IDX["B4"]]
    post_B11 = post[IDX["B11"]]

    nbr_pre  = norm_diff(pre_B8,  pre_B12)
    nbr_post = norm_diff(post_B8, post_B12)
    dnbr     = nbr_pre - nbr_post

    ndsi       = norm_diff(post_B3, post_B11)
    vis        = np.stack([post_B2, post_B3, post_B4], axis=0)
    brightness = vis.mean(axis=0)
    chroma     = vis.max(axis=0) - vis.min(axis=0)
    pre_ndwi   = norm_diff(pre_B3, pre_B8)

    snow_or_bright = (ndsi > ndsi_max) | ((brightness > BRILLO_MAX) & (chroma < CROMA_MIN))
    agua_pre       = pre_ndwi > ndwi_agua_max

    burned_raw = (
        (dnbr > dnbr_umbral) &
        (nbr_pre > pre_nbr_min) &
        (nbr_post < post_nbr_max) &
        (~snow_or_bright) &
        (~agua_pre)
    )

    structure    = np.ones((3, 3), dtype=int)
    labeled, _   = label(burned_raw, structure=structure)
    region_sizes = np.bincount(labeled.ravel())
    big_enough   = region_sizes >= min_px
    big_enough[0] = False
    burned_clean = big_enough[labeled]

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


def save_comparison(tile, pre, post, orig_mask, new_mask, label_str, cloud_note=False):
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

    cloud_warn = " [*cloud/snow: local NBR unreliable]" if cloud_note else ""
    title = (
        f"{tile}{cloud_warn}  |  {label_str}\n"
        f"orig={orig_mask.mean()*100:.1f}%  "
        f"new={new_mask.mean()*100:.1f}%  "
        f"delta={( new_mask.mean()-orig_mask.mean())*100:+.1f}%"
    )

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(title, fontsize=10)
    axes[0].imshow(rgb_pre);          axes[0].set_title("Pre-fire");       axes[0].axis("off")
    axes[1].imshow(rgb_post);         axes[1].set_title("Post-fire");      axes[1].axis("off")
    axes[2].imshow(orig_mask, cmap="gray_r", vmin=0, vmax=1)
    axes[2].set_title("Original mask"); axes[2].axis("off")
    axes[3].imshow(new_mask,  cmap="gray_r", vmin=0, vmax=1)
    axes[3].set_title("New mask");     axes[3].axis("off")
    axes[4].imshow(overlay_new);      axes[4].set_title("New overlay");    axes[4].axis("off")

    plt.tight_layout()
    fname = OUT_DIR / f"{tile}_{label_str.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_dnbr_diagnostic(tile, pre, post):
    """Show dNBR, NDSI, and NBR maps to diagnose why a tile has false positives."""
    pre_B8,  pre_B12  = pre[IDX["B8"]],  pre[IDX["B12"]]
    post_B8, post_B12 = post[IDX["B8"]], post[IDX["B12"]]
    post_B3, post_B11 = post[IDX["B3"]], post[IDX["B11"]]

    nbr_pre  = norm_diff(pre_B8,  pre_B12)
    nbr_post = norm_diff(post_B8, post_B12)
    dnbr     = nbr_pre - nbr_post
    ndsi     = norm_diff(post_B3, post_B11)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{tile} — spectral diagnostics (local full composite)", fontsize=11)

    im0 = axes[0].imshow(dnbr, cmap="RdYlGn_r", vmin=-0.5, vmax=0.8)
    axes[0].set_title("dNBR"); axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(nbr_pre, cmap="RdYlGn", vmin=-0.5, vmax=0.8)
    axes[1].set_title("NBR pre"); axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(nbr_post, cmap="RdYlGn", vmin=-0.5, vmax=0.8)
    axes[2].set_title("NBR post"); axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(ndsi, cmap="Blues", vmin=-0.3, vmax=0.6)
    axes[3].set_title("NDSI post (snow>0.25)")
    axes[3].contour(ndsi, levels=[0.25], colors="red", linewidths=0.8)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    fname = OUT_DIR / f"{tile}_diagnostico_espectral.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def evaluate_candidates():
    all_tiles  = FP_TILES + CLOUD_TILES + REF_TILES
    tile_data  = {}
    for t in all_tiles:
        tile_data[t] = {
            "pre":  read_bands(PRE_DIR  / f"{t}.tif"),
            "post": read_bands(POST_DIR / f"{t}.tif"),
            "orig": (read_bands(MASK_DIR / f"{t}.tif")[0] > 0.5).astype(np.uint8),
        }

    # Generate spectral diagnostics for cloud/snow tiles
    print("Generating spectral diagnostics for cloud/snow tiles...")
    for t in CLOUD_TILES + FP_TILES:
        fname = save_dnbr_diagnostic(t, tile_data[t]["pre"], tile_data[t]["post"])
        print(f"  {fname}")

    # Evaluate each candidate
    print(f"\nEvaluating {len(CANDIDATES)} candidates...\n")
    header = (f"{'dNBR':>6} {'preN':>5} {'postN':>5} {'NDSI':>5} {'NDWI':>5} {'px':>4} | "
              f"{'FP tiles (want -)':^54} | {'REF tiles (want +)':^42} | {'score':>7}")
    print(header)
    print("-" * len(header))

    scored = []
    for dnbr_u, pre_min, post_max, ndsi_max, ndwi_max, min_px in CANDIDATES:
        is_baseline = (dnbr_u == 0.30 and pre_min == 0.15 and post_max == 0.40
                       and ndsi_max == 0.25 and ndwi_max == 0.05 and min_px == 150)
        label_str = f"dNBR{dnbr_u}_preN{pre_min}_postN{post_max}_NDSI{ndsi_max}_NDWI{ndwi_max}_px{min_px}"

        fp_deltas, ref_deltas = [], []
        tile_results = {}

        for t in all_tiles:
            new_mask, *_ = compute_mask(
                tile_data[t]["pre"], tile_data[t]["post"],
                dnbr_u, pre_min, post_max,
                ndsi_max=ndsi_max, ndwi_agua_max=ndwi_max, min_px=min_px
            )
            orig = tile_data[t]["orig"]
            delta = float(new_mask.mean()) - float(orig.mean())
            tile_results[t] = (orig, new_mask, delta)

            if t in FP_TILES:
                fp_deltas.append(delta)
            elif t in CLOUD_TILES:
                pass
            elif t in REF_TILES:
                ref_deltas.append(delta)

        fp_summary  = "  ".join(f"{d*100:+.1f}%" for d in fp_deltas)
        ref_summary = "  ".join(f"{d*100:+.1f}%" for d in ref_deltas)

        # Score: penalise FP gain, reward REF gain
        fp_gain  = np.mean(fp_deltas)
        ref_gain = np.mean(ref_deltas)
        score = ref_gain - fp_gain  # higher is better

        tag = " ← baseline" if is_baseline else ""
        print(f"{dnbr_u:>6.2f} {pre_min:>5.2f} {post_max:>5.2f} {ndsi_max:>5.2f} {ndwi_max:>5.2f} {min_px:>4d} | "
              f"{fp_summary:<54} | {ref_summary:<42} | {score*100:>+6.2f}%{tag}")

        scored.append((score, dnbr_u, pre_min, post_max, ndsi_max, ndwi_max, min_px, label_str, tile_results))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, *_, best_label, best_results = scored[0]

    # Save images for the best candidate and the baseline
    baseline = next((s for s in scored if s[1] == 0.30 and s[2] == 0.15 and s[5] == 0.05 and s[6] == 150), None)
    save_set = []
    seen = set()
    for s in [scored[0], baseline]:
        if s is not None and s[7] not in seen:
            save_set.append(s)
            seen.add(s[7])

    print(f"\nBest candidate: {best_label}  (score {best_score*100:+.2f}%)")
    print(f"\nSaving comparison images...")
    for s in save_set:
        _, dnbr_u, pre_min, post_max, ndsi_max, ndwi_max, min_px, label_str, tile_results = s
        for t in all_tiles:
            orig, new_mask, delta = tile_results[t]
            cloud_note = t in CLOUD_TILES
            save_comparison(t, tile_data[t]["pre"], tile_data[t]["post"],
                            orig, new_mask, label_str, cloud_note=cloud_note)
        print(f"  {label_str}")

    print(f"\nDone. Images in {OUT_DIR}/")
    return scored


if __name__ == "__main__":
    evaluate_candidates()
