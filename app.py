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

app = FastAPI(title="image-processor", version="4.2.0")

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
    img.save(
        buf,
        format="WEBP",
        lossless=True,
        method=6,
        exact=True,
    )
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


def rembg_mask(rgb: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_in = buf.getvalue()

    cutout_png = remove(
        png_in,
        session=_session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=245,
        alpha_matting_background_threshold=12,
        alpha_matting_erode_size=1,
    )

    rgba_pil = ensure_rgba(cutout_png)
    rgba = np.array(rgba_pil)
    return rgba[:, :, 3] > 8


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


def choose_best_mask(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    max_dim = max(h, w)

    stats = corner_patch_stats(rgb)
    flat_bg = is_flat_background(stats, max_dim)

    ff_mask = None
    ff_score = -999.0

    if flat_bg:
        try:
            ff_mask, _ = floodfill_mask(rgb)
            ff_score = mask_score(ff_mask)
        except Exception:
            ff_mask = None
            ff_score = -999.0

    rb_mask = rembg_mask(rgb)
    rb_mask = cleanup_binary_mask(rb_mask, max_dim)
    rb_mask = remove_tiny_islands(rb_mask, max_dim)
    rb_mask = largest_component(rb_mask)
    rb_score = mask_score(rb_mask)

    if ff_mask is not None and ff_score >= rb_score + 0.5:
        return ff_mask

    return rb_mask


def build_rgba(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:

    mask_u8 = (mask.astype(np.uint8) * 255)

    # --- slight erosion (remove halo pixels) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask_eroded = cv2.erode(mask_u8,kernel,iterations=1)

    # --- feather edge ---
    alpha = cv2.GaussianBlur(mask_eroded,(5,5),0)

    alpha = np.clip(alpha,0,255).astype(np.uint8)

    rgba = np.dstack([rgb,alpha])

    return rgba

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
        "version": "4.2.0",
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
    mask = None
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
        mask = choose_best_mask(rgb)

        if mask is None or not mask.any():
            return JSONResponse({"error": "Failed to extract foreground"}, status_code=422)

        rgba = build_rgba(rgb, mask)
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
            del mask
            del rgba
            del fitted
        except Exception:
            pass

        gc.collect()
