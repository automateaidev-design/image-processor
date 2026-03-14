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

from rembg import remove, new_session

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
# App + model session
# -------------------------------------------------

app = FastAPI(title="image-processor", version="4.7.0")

REMBG_MODEL = os.getenv("REMBG_MODEL", "isnet-general-use")
TARGET_W = int(os.getenv("TARGET_W", "1400"))
TARGET_H = int(os.getenv("TARGET_H", "1700"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "8000"))

_session = None
SEM = asyncio.Semaphore(max(1, MAX_CONCURRENCY))


@app.on_event("startup")
async def _startup():
    global _session
    _session = new_session(REMBG_MODEL)

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


def ensure_rgba(png_bytes: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(png_bytes))
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im


def build_filename(prefix: str, mpn: str, sku: str, ext: str) -> str:
    def clean(s: str) -> str:
        s = (s or "").strip()
        s = s.replace(" ", "_")
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
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


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

    tl = rgb[0:p, 0:p]
    tr = rgb[0:p, w - p:w]
    bl = rgb[h - p:h, 0:p]
    br = rgb[h - p:h, w - p:w]

    corners = np.concatenate([x.reshape(-1, 3) for x in (tl, tr, bl, br)], axis=0).astype(np.uint8)
    corners_lab = cv2.cvtColor(corners.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    per_corner_means = []
    for patch in (tl, tr, bl, br):
        arr = patch.reshape(-1, 1, 3).astype(np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        per_corner_means.append(lab.mean(axis=0))

    per_corner_means = np.stack(per_corner_means, axis=0)
    mean_lab = corners_lab.mean(axis=0)
    internal_spread = float(np.mean(np.std(corners_lab, axis=0)))

    corner_mean_dists = []
    for i in range(4):
        for j in range(i + 1, 4):
            corner_mean_dists.append(float(np.linalg.norm(per_corner_means[i] - per_corner_means[j])))

    inter_corner_variation = float(max(corner_mean_dists)) if corner_mean_dists else 0.0

    return {
        "mean_lab": mean_lab,
        "internal_spread": internal_spread,
        "inter_corner_variation": inter_corner_variation,
        "patch_size": p,
    }


def is_flat_background(stats: dict, max_dim: int) -> bool:
    internal_spread = stats["internal_spread"]
    inter_corner_variation = stats["inter_corner_variation"]

    if max_dim < 700:
        return internal_spread <= 7.0 and inter_corner_variation <= 10.0
    if max_dim < 1600:
        return internal_spread <= 6.0 and inter_corner_variation <= 8.0
    return internal_spread <= 5.0 and inter_corner_variation <= 7.0


def mask_connected_to_border(candidate: np.ndarray) -> np.ndarray:
    candidate_u8 = candidate.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(candidate_u8, connectivity=4)

    if num_labels <= 1:
        return np.zeros_like(candidate_u8, dtype=bool)

    border_labels = set()
    border_labels.update(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())
    border_labels.discard(0)

    if not border_labels:
        return np.zeros_like(candidate_u8, dtype=bool)

    return np.isin(labels, list(border_labels))


def cleanup_binary_mask(mask: np.ndarray, max_dim: int) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)

    if max_dim < 700:
        k_open = 0
        k_close = 1
    elif max_dim < 1600:
        k_open = 1
        k_close = 2
    else:
        k_open = 1
        k_close = 2

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
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def remove_tiny_islands(mask: np.ndarray, max_dim: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask

    min_area = 12 if max_dim < 700 else 40 if max_dim < 1600 else 100
    out = np.zeros_like(mask_u8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
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

    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return float(largest) / float(total)


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
    elif border_touch <= 0.12:
        score += 0.0
    else:
        score -= 2.0

    if component_ratio >= 0.93:
        score += 2.0
    elif component_ratio >= 0.80:
        score += 1.0
    else:
        score -= 1.0

    return score


def sample_background_color_corners(rgb: np.ndarray) -> np.ndarray:
    """Estimate background colour from image corner patches. Returns (1,1,3) float32."""
    h, w = rgb.shape[:2]
    p = max(4, min(20, int(min(h, w) * 0.02)))
    corners = np.concatenate([
        rgb[0:p, 0:p].reshape(-1, 3),
        rgb[0:p, w - p:w].reshape(-1, 3),
        rgb[h - p:h, 0:p].reshape(-1, 3),
        rgb[h - p:h, w - p:w].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    return corners.mean(axis=0).reshape(1, 1, 3)


def sample_background_color(rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimate background colour.
    Primary: pixels where rembg alpha == 0 (definite background, most accurate).
    Fallback: corner patch sampling.
    """
    if alpha is not None:
        bg_pixels = rgb[alpha == 0].astype(np.float32)
        if len(bg_pixels) >= 200:
            return bg_pixels.mean(axis=0).reshape(1, 1, 3)
    return sample_background_color_corners(rgb)


def apply_floodfill_clip(rb_alpha: np.ndarray, ff_mask: np.ndarray) -> np.ndarray:
    """
    Use the floodfill boundary as a soft spatial clip over rembg's alpha.

    This is the primary halo-removal mechanism for flat-background images.
    It works regardless of object colour, including the hard case of grey
    objects on grey/white backgrounds where colour-based halo suppression
    cannot distinguish halo pixels from real object edge pixels.

    The floodfill mask accurately identifies the hard outer boundary of
    the object (background-connected region vs foreground).  Multiplying
    rembg's soft alpha by a blurred version of this mask zeros out halo
    pixels that lie outside the floodfill boundary while preserving
    rembg's fine-grained smooth alpha for the interior and true edge zone.

    Gaussian sigma = 3.0 (~6 px transition) gives a gentle enough blend
    that the clip doesn't introduce a hard visible ring.
    """
    ff_u8 = (ff_mask.astype(np.uint8) * 255)
    ff_soft = cv2.GaussianBlur(ff_u8.astype(np.float32), (0, 0), sigmaX=3.0)
    ff_soft = np.clip(ff_soft / 255.0, 0.0, 1.0)
    clipped = (rb_alpha.astype(np.float32) * ff_soft).clip(0, 255).astype(np.uint8)
    return clipped


def build_edge_band_mask(alpha: np.ndarray, band_px: int = 12) -> np.ndarray:
    """
    Boolean mask covering only the narrow transition band at the alpha edge.
    Used to spatially constrain halo suppression to the fringe zone only.
    """
    solid = (alpha > 200).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    inner = cv2.erode(solid, k, iterations=1)
    outer = cv2.dilate(solid, k, iterations=1)
    return (outer - inner).astype(bool)


def decontaminate_rgba(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg_color: np.ndarray,
    suppress_halo: bool = True,
) -> np.ndarray:
    """
    Remove background colour contamination from semi-transparent edge pixels.

    Pass 1 — colour unmix (always applied):
        Reverses the background colour bleed in the feathered transition zone.
        true_fg = (blended_rgb - bg * (1 - a)) / a
        Applied where 4% < alpha < 96% AND within the alpha edge band.

    Pass 2 — halo suppression (applied only when suppress_halo=True):
        For non-flat-background images where no spatial floodfill clip is
        available, further suppress pixels whose colour remains close to bg.
        Spatially constrained to the edge band to avoid eating similar-
        coloured object interiors.
        suppress_halo=False when the floodfill clip has already handled
        halo removal spatially (flat-bg path).
    """
    alpha_f = alpha.astype(np.float32) / 255.0
    rgb_f = rgb.astype(np.float32)

    edge_band = build_edge_band_mask(alpha, band_px=12)

    # ------------------------------------------------------------------
    # Pass 1: colour unmix in feathered transition zone within edge band
    # ------------------------------------------------------------------
    safe_alpha = np.maximum(alpha_f, 0.001)[..., np.newaxis]
    decontaminated = (rgb_f - bg_color * (1.0 - alpha_f[..., np.newaxis])) / safe_alpha
    decontaminated = np.clip(decontaminated, 0, 255)

    transition = (alpha_f > 0.04) & (alpha_f < 0.96) & edge_band
    result_rgb = rgb_f.copy()
    result_rgb[transition] = decontaminated[transition]
    result_rgb = np.clip(result_rgb, 0, 255)

    # ------------------------------------------------------------------
    # Pass 2: colour-distance halo suppression (non-flat-bg path only)
    # ------------------------------------------------------------------
    if suppress_halo:
        dist_to_bg = np.sqrt(np.sum((result_rgb - bg_color) ** 2, axis=2))
        halo_thresh = 30.0
        halo_scale = np.clip(dist_to_bg / halo_thresh, 0.0, 1.0)
        suppress_zone = (alpha_f < 0.85) & edge_band
        alpha_f_out = alpha_f.copy()
        alpha_f_out[suppress_zone] *= halo_scale[suppress_zone]
        alpha_f_out[alpha_f_out < 0.04] = 0.0
        result_alpha = np.clip(alpha_f_out * 255.0, 0, 255).astype(np.uint8)
    else:
        result_alpha = alpha

    result_rgb_u8 = result_rgb.clip(0, 255).astype(np.uint8)
    return np.dstack([result_rgb_u8, result_alpha])


def rembg_rgba(rgb: np.ndarray) -> np.ndarray:
    """
    Run rembg and return full RGBA, preserving the neural-net's smooth alpha.
    """
    pil_img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_in = buf.getvalue()

    cutout_png = remove(
        png_in,
        session=_session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=230,
        alpha_matting_background_threshold=15,
        alpha_matting_erode_size=2,
    )

    rgba_pil = ensure_rgba(cutout_png)
    return np.array(rgba_pil)   # H x W x 4, uint8


def build_rgba_from_floodfill(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Build RGBA from binary floodfill mask + decontamination.
    Used only when floodfill clearly beats rembg on scoring.
    """
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_eroded = cv2.erode(mask_u8, kernel, iterations=1)
    alpha = cv2.GaussianBlur(mask_eroded, (7, 7), 0)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    bg_color = sample_background_color_corners(rgb)
    return decontaminate_rgba(rgb, alpha, bg_color, suppress_halo=False)


def floodfill_mask(rgb: np.ndarray) -> Tuple[np.ndarray, dict]:
    h, w = rgb.shape[:2]
    max_dim = max(h, w)

    stats = corner_patch_stats(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    mean_lab = stats["mean_lab"]
    internal_spread = stats["internal_spread"]
    inter_corner_variation = stats["inter_corner_variation"]

    if max_dim < 700:
        base_tol = 12.0
    elif max_dim < 1600:
        base_tol = 11.0
    else:
        base_tol = 10.0

    tol = base_tol + (internal_spread * 1.0) + (inter_corner_variation * 0.35)
    tol = clamp_float(tol, 8.0, 18.0)

    dist = np.sqrt(np.sum((lab - mean_lab.reshape(1, 1, 3)) ** 2, axis=2))
    candidate_bg = dist <= tol

    bg_mask = mask_connected_to_border(candidate_bg)
    fg_mask = ~bg_mask

    fg_mask = cleanup_binary_mask(fg_mask, max_dim)
    fg_mask = remove_tiny_islands(fg_mask, max_dim)
    fg_mask = largest_component(fg_mask)

    if max_dim >= 700:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.dilate(fg_mask.astype(np.uint8), kernel, iterations=1) > 0

    meta = {
        "tolerance": tol,
        "internal_spread": internal_spread,
        "inter_corner_variation": inter_corner_variation,
    }
    return fg_mask, meta


def choose_best_rgba(rgb: np.ndarray) -> np.ndarray:
    """
    Produce the cleanest possible RGBA cutout.

    Strategy for flat-background images (white/grey studio shots — the
    majority of industrial part photos):

      1. Run rembg → get smooth neural-net alpha (best for fine detail)
      2. Run floodfill → get accurate hard outer boundary (best for halo)
      3. Multiply rembg alpha by soft floodfill boundary clip
         → zeros out outer halo pixels (spatial, colour-independent)
         → preserves rembg's smooth feathering inside the object
      4. Apply colour unmix (Pass 1 decontamination) — no colour-based
         halo suppression needed since step 3 already handled halo spatially

    This combined approach solves the grey-object-on-grey-background
    problem where colour-based suppression cannot distinguish halo from
    real edge: floodfill is colour-agnostic and always correctly identifies
    the hard bg/fg boundary for uniform backgrounds.

    Edge cases:
      - floodfill fails (score too low): fall back to rembg + suppress_halo
      - rembg completely fails (ff much better): use floodfill alpha only
      - non-flat background: rembg only with colour-based suppression
    """
    h, w = rgb.shape[:2]
    max_dim = max(h, w)

    stats = corner_patch_stats(rgb)
    flat_bg = is_flat_background(stats, max_dim)

    # Always run rembg — we need its smooth alpha regardless of path
    rb_rgba = rembg_rgba(rgb)
    rb_alpha = rb_rgba[:, :, 3]
    rb_rgb = rb_rgba[:, :, :3]

    bg_color = sample_background_color(rgb, alpha=rb_alpha)

    # Binary rembg mask for scoring
    rb_mask_bin = rb_alpha > 8
    rb_mask_bin = cleanup_binary_mask(rb_mask_bin, max_dim)
    rb_mask_bin = remove_tiny_islands(rb_mask_bin, max_dim)
    rb_mask_bin = largest_component(rb_mask_bin)
    rb_score = mask_score(rb_mask_bin)

    if flat_bg:
        try:
            ff_mask, _ = floodfill_mask(rgb)
            ff_score = mask_score(ff_mask)
        except Exception:
            ff_mask = None
            ff_score = -999.0

        if ff_mask is not None:
            if ff_score >= rb_score + 2.0:
                # rembg really failed — use floodfill alpha only
                return build_rgba_from_floodfill(rgb, ff_mask)

            # Normal flat-bg path: clip rembg alpha with floodfill boundary.
            # This removes halo regardless of object/bg colour contrast.
            # suppress_halo=False because spatial clip has handled it.
            clipped_alpha = apply_floodfill_clip(rb_alpha, ff_mask)
            return decontaminate_rgba(rb_rgb, clipped_alpha, bg_color, suppress_halo=False)

    # Non-flat background or floodfill failed:
    # rembg alpha only, with colour-based halo suppression
    return decontaminate_rgba(rb_rgb, rb_alpha, bg_color, suppress_halo=True)


def resize_if_huge(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_DIM:
        return img
    scale = MAX_IMAGE_DIM / float(max(w, h))
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)


# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": REMBG_MODEL,
        "max_concurrency": MAX_CONCURRENCY,
        "version": "4.7.0",
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

    pil_img = None
    rgb = None
    rgba = None
    fitted = None

    try:
        if _session is None:
            return JSONResponse({"error": "Model session not ready"}, status_code=503)

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

        fitted = object_aware_fit(
            rgba,
            target_w=TARGET_W,
            target_h=TARGET_H,
            padding_ratio=0.02,
            alpha_thresh=8,
        )

        out_fmt = (output or "webp").strip().lower()
        if out_fmt == "png":
            out_bytes = save_png(fitted)
            media_type = "image/png"
            ext = "png"
        else:
            out_bytes = save_lossless_webp(fitted)
            media_type = "image/webp"
            ext = "webp"

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
            del pil_img
            del rgb
            del rgba
            del fitted
        except Exception:
            pass

        gc.collect()
