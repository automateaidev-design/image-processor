import io
import os
import gc
import asyncio
from typing import Optional, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from PIL import Image

from transparent_background import Remover

# -------------------------------------------------
# Stability / performance
# -------------------------------------------------

try:
    cv2.setNumThreads(1)
except Exception:
    pass

PIL_MAX_IMAGE_PIXELS = int(os.getenv("PIL_MAX_IMAGE_PIXELS", "60000000"))
Image.MAX_IMAGE_PIXELS = PIL_MAX_IMAGE_PIXELS

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
QUEUE_TIMEOUT_S = float(os.getenv("QUEUE_TIMEOUT_S", "15"))

ENABLE_SELF_RESTART = os.getenv("ENABLE_SELF_RESTART", "0") == "1"
SELF_RESTART_SECONDS = int(os.getenv("SELF_RESTART_SECONDS", "600"))

# -------------------------------------------------
# App + model
# -------------------------------------------------

app = FastAPI(title="image-processor", version="5.3.0")

# InSPyReNet options
# INSPYRE_JIT=on  → TorchScript JIT (faster after warmup, larger startup cost)
# INSPYRE_JIT=off → default (safer on CPU-only / limited RAM environments)
INSPYRE_JIT = os.getenv("INSPYRE_JIT", "off").strip().lower() == "on"

# Alpha threshold for InSPyReNet's hard-threshold mode.
# 0.0 = use the model's native soft alpha (recommended — keeps smooth edges).
# 0.1–0.9 = binarise at that level (useful if you want hard masks).
INSPYRE_THRESHOLD = float(os.getenv("INSPYRE_THRESHOLD", "0.0"))

TARGET_W = int(os.getenv("TARGET_W", "1400"))
TARGET_H = int(os.getenv("TARGET_H", "1700"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "8000"))

_remover: Optional[Remover] = None
SEM = asyncio.Semaphore(max(1, MAX_CONCURRENCY))


@app.on_event("startup")
async def _startup():
    global _remover
    _remover = Remover(jit=INSPYRE_JIT)

    if ENABLE_SELF_RESTART:
        async def _restarter():
            await asyncio.sleep(max(60, SELF_RESTART_SECONDS))
            os._exit(0)
        asyncio.create_task(_restarter())


# -------------------------------------------------
# Utility helpers
# -------------------------------------------------

def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def pil_open_rgb(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def build_filename(prefix: str, mpn: str, sku: str, ext: str) -> str:
    def clean(s: str) -> str:
        s = (s or "").strip().replace(" ", "_")
        s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))
        return s[:140] if s else "na"
    return f"{clean(prefix)}_{clean(mpn)}_{clean(sku)}.{ext}"


def save_lossless_webp(rgba_arr: np.ndarray) -> bytes:
    img = Image.fromarray(rgba_arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="WEBP", lossless=True, method=6, exact=True)
    return buf.getvalue()


def save_png(rgba_arr: np.ndarray) -> bytes:
    img = Image.fromarray(rgba_arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def alpha_bbox(alpha: np.ndarray, thresh: int = 8) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(alpha > thresh)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def object_aware_fit(
    rgba: np.ndarray,
    target_w: int,
    target_h: int,
    padding_ratio: float = 0.02,
    alpha_thresh: int = 8,
) -> np.ndarray:
    pad = clamp_float(padding_ratio, 0.0, 0.49)
    alpha = rgba[:, :, 3]
    bbox = alpha_bbox(alpha, thresh=alpha_thresh)
    if bbox is None:
        return np.zeros((target_h, target_w, 4), dtype=np.uint8)

    x1, y1, x2, y2 = bbox
    obj = rgba[y1:y2, x1:x2, :]
    obj_h, obj_w = obj.shape[:2]

    inner_w = max(1, int(round(target_w * (1.0 - 2.0 * pad))))
    inner_h = max(1, int(round(target_h * (1.0 - 2.0 * pad))))
    scale = min(inner_w / obj_w, inner_h / obj_h)
    new_w = max(1, int(round(obj_w * scale)))
    new_h = max(1, int(round(obj_h * scale)))

    obj_resized = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    out = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2

    patch = out[y0:y0 + new_h, x0:x0 + new_w, :]
    fg = obj_resized.astype(np.float32) / 255.0
    bg = patch.astype(np.float32) / 255.0
    fg_a = fg[:, :, 3:4]
    out_rgb = fg[:, :, :3] * fg_a + bg[:, :, :3] * (1.0 - fg_a)
    out_a = fg_a + bg[:, :, 3:4] * (1.0 - fg_a)
    patch[:, :, :3] = (out_rgb * 255.0).clip(0, 255).astype(np.uint8)
    patch[:, :, 3] = (out_a[:, :, 0] * 255.0).clip(0, 255).astype(np.uint8)
    out[y0:y0 + new_h, x0:x0 + new_w, :] = patch
    return out


def dynamic_corner_patch_size(h: int, w: int) -> int:
    return max(6, min(36, int(round(min(h, w) * 0.025))))


def corner_patch_stats(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    p = dynamic_corner_patch_size(h, w)
    tl, tr = rgb[0:p, 0:p], rgb[0:p, w - p:w]
    bl, br = rgb[h - p:h, 0:p], rgb[h - p:h, w - p:w]

    corners = np.concatenate([x.reshape(-1, 3) for x in (tl, tr, bl, br)], axis=0).astype(np.uint8)
    corners_lab = cv2.cvtColor(corners.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    per_corner_means = []
    for patch in (tl, tr, bl, br):
        lab = cv2.cvtColor(patch.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        per_corner_means.append(lab.mean(axis=0))

    per_corner_means = np.stack(per_corner_means, axis=0)
    dists = [float(np.linalg.norm(per_corner_means[i] - per_corner_means[j]))
             for i in range(4) for j in range(i + 1, 4)]

    return {
        "mean_lab": corners_lab.mean(axis=0),
        "internal_spread": float(np.mean(np.std(corners_lab, axis=0))),
        "inter_corner_variation": float(max(dists)) if dists else 0.0,
        "patch_size": p,
    }


def is_flat_background(stats: dict, max_dim: int) -> bool:
    s, v = stats["internal_spread"], stats["inter_corner_variation"]
    if max_dim < 700:
        return s <= 7.0 and v <= 10.0
    if max_dim < 1600:
        return s <= 6.0 and v <= 8.0
    return s <= 5.0 and v <= 7.0


def mask_connected_to_border(candidate: np.ndarray) -> np.ndarray:
    candidate_u8 = candidate.astype(np.uint8)
    _, labels = cv2.connectedComponents(candidate_u8, connectivity=4)
    border_labels = set()
    for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        border_labels.update(np.unique(edge).tolist())
    border_labels.discard(0)
    if not border_labels:
        return np.zeros_like(candidate_u8, dtype=bool)
    return np.isin(labels, list(border_labels))


def cleanup_binary_mask(mask: np.ndarray, max_dim: int) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    k_open = 0 if max_dim < 700 else 1
    k_close = 1 if max_dim < 700 else 2
    if k_open > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_open + 1, 2 * k_open + 1))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, ko)
    if k_close > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_close + 1, 2 * k_close + 1))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kc)
    return mask_u8 > 0


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def remove_tiny_islands(mask: np.ndarray, max_dim: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask
    min_area = 12 if max_dim < 700 else 40 if max_dim < 1600 else 100
    out = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 1
    return out > 0


def touches_border(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(np.mean(border.astype(np.float32)))


def largest_component_ratio(mask: np.ndarray) -> float:
    total = int(mask.astype(np.uint8).sum())
    if total == 0:
        return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / float(total)


def mask_score(mask: np.ndarray) -> float:
    area_ratio = float(mask.mean())
    border_touch = touches_border(mask)
    component_ratio = largest_component_ratio(mask)
    score = 0.0
    if 0.02 <= area_ratio <= 0.92:
        score += 2.5
    elif 0.005 <= area_ratio <= 0.98:
        score += 1.0
    else:
        score -= 3.0
    if border_touch <= 0.01:
        score += 2.0
    elif border_touch <= 0.04:
        score += 1.0
    elif border_touch > 0.12:
        score -= 2.0
    if component_ratio >= 0.93:
        score += 2.0
    elif component_ratio >= 0.80:
        score += 1.0
    else:
        score -= 1.0
    return score


def sample_background_color(rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimate background colour.
    Primary: pixels where model alpha == 0 (true background, most accurate).
    Fallback: image corner patches.
    """
    if alpha is not None:
        bg_pixels = rgb[alpha == 0].astype(np.float32)
        if len(bg_pixels) >= 200:
            return bg_pixels.mean(axis=0).reshape(1, 1, 3)
    h, w = rgb.shape[:2]
    p = max(4, min(20, int(min(h, w) * 0.02)))
    corners = np.concatenate([
        rgb[0:p, 0:p].reshape(-1, 3),
        rgb[0:p, w - p:w].reshape(-1, 3),
        rgb[h - p:h, 0:p].reshape(-1, 3),
        rgb[h - p:h, w - p:w].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    return corners.mean(axis=0).reshape(1, 1, 3)


def self_clip_alpha(alpha: np.ndarray, fg_thresh: int = 40, dilation_px: int = 8) -> np.ndarray:
    """
    Use the model's own high-confidence foreground pixels as an anchor and
    zero out any alpha that lies further than dilation_px from that anchor.

    This kills halo pixels in a colour-independent way:
      - Interior object pixels have alpha > fg_thresh → they are the anchor
        → they can never be removed by this operation
      - Real edge pixels are within dilation_px of the anchor → preserved
      - Halo pixels are outside the allowed zone → zeroed
    """
    confident = (alpha > fg_thresh).astype(np.uint8)
    if confident.sum() == 0:
        return alpha

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
    )
    allowed_zone = cv2.dilate(confident, k, iterations=1)
    sigma = max(1.0, dilation_px / 3.0)
    allowed_soft = np.clip(
        cv2.GaussianBlur(allowed_zone.astype(np.float32), (0, 0), sigmaX=sigma),
        0.0, 1.0
    )
    return (alpha.astype(np.float32) * allowed_soft).clip(0, 255).astype(np.uint8)


def smooth_alpha_edges(alpha: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Apply a gentle Gaussian blur to the alpha channel, confined to the
    narrow transition band at the object edge.

    Purpose: smooth staircase/pixelation artifacts on low-resolution source
    images where the model's alpha boundary inherits the pixel grid of the
    input. Has no visible effect on large clean images because:
      - sigma scales with image size (tiny on large images)
      - effect is masked to the edge band only (interior never touched)
      - the blur is soft — it rounds jagged steps, not reshape edges

    Sigma scaling:
      small image  (<700px)  → sigma 0.8  (barely any smoothing)
      medium image (<1600px) → sigma 1.2
      large image  (>=1600px)→ sigma 0.6  (large images rarely need it)
    """
    if max_dim < 700:
        sigma = 0.8
    elif max_dim < 1600:
        sigma = 1.2
    else:
        sigma = 0.6

    # Build edge band: dilate/erode the solid-fg region
    solid = (alpha > 200).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    edge_band = (cv2.dilate(solid, k) - cv2.erode(solid, k)).astype(bool)

    # Blur the full alpha, then blend only within the edge band
    alpha_f = alpha.astype(np.float32)
    blurred = cv2.GaussianBlur(alpha_f, (0, 0), sigmaX=sigma)

    result = alpha_f.copy()
    result[edge_band] = blurred[edge_band]
    return np.clip(result, 0, 255).astype(np.uint8)


def colour_unmix(rgb: np.ndarray, alpha: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
    """
    Reverse background colour contamination on semi-transparent edge pixels.
    Only applied within the narrow band around the alpha boundary.
    """
    alpha_f = alpha.astype(np.float32) / 255.0
    rgb_f = rgb.astype(np.float32)

    solid = (alpha > 200).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    edge_band = (cv2.dilate(solid, k) - cv2.erode(solid, k)).astype(bool)

    safe_alpha = np.maximum(alpha_f, 0.001)[..., np.newaxis]
    decontaminated = (rgb_f - bg_color * (1.0 - alpha_f[..., np.newaxis])) / safe_alpha
    decontaminated = np.clip(decontaminated, 0, 255)

    transition = (alpha_f > 0.04) & (alpha_f < 0.96) & edge_band
    result_rgb = rgb_f.copy()
    result_rgb[transition] = decontaminated[transition]
    return np.clip(result_rgb, 0, 255).astype(np.uint8)


# -------------------------------------------------
# InSPyReNet model inference
# -------------------------------------------------

def inspyrenet_rgba(rgb: np.ndarray) -> np.ndarray:
    """
    Run InSPyReNet (transparent_background) and return H×W×4 uint8 RGBA.

    Uses INSPYRE_THRESHOLD env var:
      0.0  → native soft alpha from the model (default, best for smooth edges)
      >0.0 → threshold the alpha at that level (harder mask, e.g. 0.5)
    """
    pil_img = Image.fromarray(rgb, mode="RGB")

    if INSPYRE_THRESHOLD > 0.0:
        result = _remover.process(pil_img, type="rgba", threshold=INSPYRE_THRESHOLD)
    else:
        result = _remover.process(pil_img, type="rgba")

    if result.mode != "RGBA":
        result = result.convert("RGBA")
    return np.array(result)   # H × W × 4, uint8


def floodfill_mask(rgb: np.ndarray) -> Tuple[np.ndarray, dict]:
    h, w = rgb.shape[:2]
    max_dim = max(h, w)
    stats = corner_patch_stats(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_lab = stats["mean_lab"]
    base_tol = 12.0 if max_dim < 700 else 11.0 if max_dim < 1600 else 10.0
    tol = clamp_float(
        base_tol + stats["internal_spread"] * 1.0 + stats["inter_corner_variation"] * 0.35,
        8.0, 18.0
    )
    dist = np.sqrt(np.sum((lab - mean_lab.reshape(1, 1, 3)) ** 2, axis=2))
    fg_mask = ~mask_connected_to_border(dist <= tol)
    fg_mask = cleanup_binary_mask(fg_mask, max_dim)
    fg_mask = remove_tiny_islands(fg_mask, max_dim)
    fg_mask = largest_component(fg_mask)
    if max_dim >= 700:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.dilate(fg_mask.astype(np.uint8), k, iterations=1) > 0
    return fg_mask, {"tolerance": tol}


def build_rgba_from_floodfill(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Feathered RGBA from binary floodfill. Only used when model clearly fails."""
    mask_u8 = (mask.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = np.clip(
        cv2.GaussianBlur(cv2.erode(mask_u8, k, iterations=1), (7, 7), 0),
        0, 255
    ).astype(np.uint8)
    bg_color = sample_background_color(rgb)
    rgb_clean = colour_unmix(rgb, alpha, bg_color)
    return np.dstack([rgb_clean, alpha])


def choose_best_rgba(rgb: np.ndarray) -> np.ndarray:
    """
    Full pipeline:
      1. InSPyReNet → smooth RGBA (replaces rembg)
      2. self_clip_alpha() → kill halo using model's own confident-fg anchor
      3. colour_unmix() → remove bg colour bleed from edge pixels
      4. Floodfill fallback only when model score is clearly bad (delta >= 2)
    """
    h, w = rgb.shape[:2]
    max_dim = max(h, w)

    # --- InSPyReNet inference ---
    rb_rgba = inspyrenet_rgba(rgb)
    rb_alpha = rb_rgba[:, :, 3]
    rb_rgb = rb_rgba[:, :, :3]

    bg_color = sample_background_color(rgb, alpha=rb_alpha)

    # Score (using binary version of alpha for metric only)
    rb_bin = largest_component(
        remove_tiny_islands(cleanup_binary_mask(rb_alpha > 8, max_dim), max_dim)
    )
    rb_score = mask_score(rb_bin)

    # Floodfill fallback check (scoring only — NOT used as clip)
    stats = corner_patch_stats(rgb)
    ff_mask = None
    ff_score = -999.0
    if is_flat_background(stats, max_dim):
        try:
            ff_mask, _ = floodfill_mask(rgb)
            ff_score = mask_score(ff_mask)
        except Exception:
            pass

    if ff_mask is not None and ff_score >= rb_score + 2.0:
        return build_rgba_from_floodfill(rgb, ff_mask)

    # Main path: self-clip → edge smoothing → colour unmix → alpha curve
    clipped_alpha = self_clip_alpha(rb_alpha, fg_thresh=40, dilation_px=8)
    smooth_alpha = smooth_alpha_edges(clipped_alpha, max_dim)
    rgb_clean = colour_unmix(rb_rgb, smooth_alpha, bg_color)

    # Power curve: suppresses low-alpha fringe pixels toward 0 while
    # leaving solid pixels (alpha ~255) virtually unchanged.
    # a_out = (a/255)^2.2 * 255
    # At alpha=20  → output ~1   (fringe invisible)
    # At alpha=40  → output ~4   (halo effectively gone)
    # At alpha=128 → output ~60  (soft edge tightened)
    # At alpha=220 → output ~170 (solid pixels barely affected)
    # At alpha=255 → output 255  (unchanged)
    # Increase exponent to cut harder; decrease if thin detail clips.
    a_f = smooth_alpha.astype(np.float32) / 255.0
    a_curved = np.power(np.clip(a_f, 0.0, 1.0), 2.2) * 255.0

    # Hard zero floor: any pixel below alpha 20 after the curve is
    # invisible noise — zero it completely to eliminate grey fog.
    a_curved[a_curved < 20] = 0.0

    final_alpha = np.clip(a_curved, 0, 255).astype(np.uint8)

    return np.dstack([rgb_clean, final_alpha])


def resize_if_huge(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_DIM:
        return img
    scale = MAX_IMAGE_DIM / float(max(w, h))
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)


# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": "inspyrenet",
        "jit": INSPYRE_JIT,
        "threshold": INSPYRE_THRESHOLD,
        "max_concurrency": MAX_CONCURRENCY,
        "version": "5.3.0",
    }


@app.get("/ready")
def ready():
    if SEM.locked():
        return JSONResponse({"ready": False, "reason": "busy"}, status_code=503)
    return {"ready": True}


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    mpn: str = Form(""),
    sku: str = Form(""),
    prefix: str = Form("partlogic"),
    output: str = Form("webp"),
):
    try:
        await asyncio.wait_for(SEM.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": "Service busy", "hint": "Reduce n8n parallelism or increase MAX_CONCURRENCY"},
            status_code=503,
        )

    pil_img = rgb = rgba = fitted = None

    try:
        if _remover is None:
            return JSONResponse({"error": "Model not ready"}, status_code=503)

        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": f"File too large. Max {MAX_UPLOAD_MB}MB"}, status_code=413)

        pil_img = pil_open_rgb(raw)
        pil_img = resize_if_huge(pil_img)
        rgb = np.array(pil_img)
        rgba = choose_best_rgba(rgb)

        if rgba is None or rgba[:, :, 3].max() == 0:
            return JSONResponse({"error": "Failed to extract foreground"}, status_code=422)

        fitted = object_aware_fit(rgba, target_w=TARGET_W, target_h=TARGET_H, padding_ratio=0.02)

        out_fmt = (output or "webp").strip().lower()
        if out_fmt == "png":
            out_bytes, media_type, ext = save_png(fitted), "image/png", "png"
        else:
            out_bytes, media_type, ext = save_lossless_webp(fitted), "image/webp", "webp"

        filename = build_filename(prefix=prefix, mpn=mpn, sku=sku, ext=ext)
        return Response(
            content=out_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        try:
            SEM.release()
        except Exception:
            pass
        try:
            del pil_img, rgb, rgba, fitted
        except Exception:
            pass
        gc.collect()
