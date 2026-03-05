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

# ----------------------------
# Stability / performance knobs
# ----------------------------

# Prevent OpenCV from spawning huge thread pools (common cause of instability on small containers)
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# Prevent Pillow decompression bombs / insane uploads from killing RAM
# (set via env if you want; default is conservative for e-comm photos)
PIL_MAX_IMAGE_PIXELS = int(os.getenv("PIL_MAX_IMAGE_PIXELS", "60000000"))  # 60 MP
Image.MAX_IMAGE_PIXELS = PIL_MAX_IMAGE_PIXELS

# Concurrency cap (backpressure) - set to 1..3 for heavy rembg + matting
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
QUEUE_TIMEOUT_S = float(os.getenv("QUEUE_TIMEOUT_S", "15"))  # if busy, fail fast

# Force periodic self-restart (optional; better to use gunicorn --max-requests)
# Set ENABLE_SELF_RESTART=1 and SELF_RESTART_SECONDS=600 if you really want it.
ENABLE_SELF_RESTART = os.getenv("ENABLE_SELF_RESTART", "0") == "1"
SELF_RESTART_SECONDS = int(os.getenv("SELF_RESTART_SECONDS", "600"))

# ----------------------------
# App + model session
# ----------------------------

app = FastAPI(title="image-processor", version="2.2.0")

REMBG_MODEL = os.getenv("REMBG_MODEL", "u2net")
_session = None  # initialized per worker on startup

TARGET_W = int(os.getenv("TARGET_W", "1400"))
TARGET_H = int(os.getenv("TARGET_H", "1700"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Global semaphore for backpressure (per worker)
SEM = asyncio.Semaphore(max(1, MAX_CONCURRENCY))


@app.on_event("startup")
async def _startup():
    """
    Create the rembg session *inside* the worker process.
    This is more stable with gunicorn preloading/forking and keeps memory predictable.
    """
    global _session

    # Load model/session once per worker
    _session = new_session(REMBG_MODEL)

    # Optional self-restart loop (only if you turn it on)
    if ENABLE_SELF_RESTART:
        async def _restarter():
            await asyncio.sleep(max(60, SELF_RESTART_SECONDS))
            # Exiting the process lets Railway/gunicorn restart cleanly (clears RAM)
            os._exit(0)  # noqa: S606

        asyncio.create_task(_restarter())


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def pil_open_rgb(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def pre_upscale_if_small(img: Image.Image, min_max_dim: int = 900) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m >= min_max_dim:
        return img
    scale = min_max_dim / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)


def ensure_rgba(png_bytes: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(png_bytes))
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im


def dilate(alpha: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return alpha
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(alpha, kernel, iterations=1)


def erode(alpha: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return alpha
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(alpha, kernel, iterations=1)


def feather(alpha: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return alpha
    k = 2 * px + 1
    return cv2.GaussianBlur(alpha, (k, k), 0)


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
    padding_ratio: float = 0.0,
    alpha_thresh: int = 8,
) -> np.ndarray:
    pad = float(padding_ratio)
    pad = max(0.0, min(0.49, pad))

    alpha = rgba[:, :, 3]
    bbox = alpha_bbox(alpha, thresh=alpha_thresh)
    if bbox is None:
        return np.zeros((target_h, target_w, 4), dtype=np.uint8)

    x1, y1, x2, y2 = bbox
    obj = rgba[y1:y2, x1:x2, :]

    obj_h, obj_w = obj.shape[:2]
    inner_w = int(round(target_w * (1.0 - 2.0 * pad)))
    inner_h = int(round(target_h * (1.0 - 2.0 * pad)))
    inner_w = max(1, inner_w)
    inner_h = max(1, inner_h)

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
    img.save(buf, format="WEBP", lossless=True, quality=100, method=6)
    return buf.getvalue()


def save_png(rgba_arr: np.ndarray) -> bytes:
    img = Image.fromarray(rgba_arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def decontaminate_edge_rgb(rgba: np.ndarray, alpha_max: int = 200, inpaint_radius: int = 3) -> np.ndarray:
    """
    Removes “halo” by fixing RGB values in semi-transparent edge pixels.
    We inpaint ONLY where alpha is between 1..alpha_max (edge band).
    """
    alpha = rgba[:, :, 3]
    mask = ((alpha > 0) & (alpha < alpha_max)).astype(np.uint8) * 255
    if mask.max() == 0:
        return rgba

    rgb = rgba[:, :, :3]
    rgb_fixed = cv2.inpaint(rgb, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    out = rgba.copy()
    out[:, :, :3] = rgb_fixed
    return out


@app.get("/health")
def health():
    return {"ok": True, "model": REMBG_MODEL, "max_concurrency": MAX_CONCURRENCY}


@app.get("/ready")
def ready():
    # Simple readiness: if we can immediately acquire 1 slot, we're "ready".
    # This prevents callers from piling onto an overloaded instance.
    if SEM.locked():
        return JSONResponse({"ready": False, "reason": "busy"}, status_code=503)
    return {"ready": True}


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),

    mpn: str = Form(""),
    sku: str = Form(""),
    prefix: str = Form("partlogic"),

    output: str = Form("webp"),  # webp | png

    # Rembg controls (tune in n8n without redeploy)
    alpha_matting: bool = Form(True),
    # Better defaults for white/industrial products:
    alpha_matting_foreground_threshold: int = Form(250),
    alpha_matting_background_threshold: int = Form(15),
    alpha_matting_erode_size: int = Form(2),

    # Edge controls (better defaults for crisp ecommerce cutouts)
    edge_erode_px: int = Form(1),      # 1 is usually enough; 2 can shrink too much
    edge_dilate_px: int = Form(0),     # keep 0 unless you need thicker edges
    edge_feather_px: int = Form(0),    # KEEP 0 to avoid grey fringe

    # Decontaminate edge RGB
    decontaminate: bool = Form(True),
    decontaminate_alpha_max: int = Form(200),
    decontaminate_inpaint_radius: int = Form(8),  # 200 is way too high; keep 6-12

    # Canvas controls
    padding_ratio: float = Form(0.0),
    target_w: int = Form(TARGET_W),
    target_h: int = Form(TARGET_H),

    # Quality/perf
    pre_upscale: bool = Form(True),
    pre_upscale_min_dim: int = Form(900),
):
    # Backpressure: if overloaded, fail fast rather than timing out and wedging n8n
    try:
        await asyncio.wait_for(SEM.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": "Service busy", "hint": "Reduce n8n parallelism or increase MAX_CONCURRENCY"},
            status_code=503,
        )

    rgba = None
    fitted = None
    pil_img = None
    rgba_pil = None
    tmp = None

    try:
        if _session is None:
            # Should not happen if startup ran; still guard
            return JSONResponse({"error": "Model session not ready"}, status_code=503)

        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": f"File too large. Max {MAX_UPLOAD_MB}MB"}, status_code=413)

        pil_img = pil_open_rgb(raw)

        # Hard cap dimensions to protect RAM if someone uploads a massive image
        w, h = pil_img.size
        max_dim = int(os.getenv("MAX_IMAGE_DIM", "8000"))
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

        if pre_upscale:
            pil_img = pre_upscale_if_small(pil_img, min_max_dim=clamp_int(pre_upscale_min_dim, 256, 2400))

        tmp = io.BytesIO()
        pil_img.save(tmp, format="PNG")
        png_in = tmp.getvalue()

        cutout_png = remove(
            png_in,
            session=_session,
            alpha_matting=bool(alpha_matting),
            alpha_matting_foreground_threshold=clamp_int(alpha_matting_foreground_threshold, 0, 255),
            alpha_matting_background_threshold=clamp_int(alpha_matting_background_threshold, 0, 255),
            alpha_matting_erode_size=clamp_int(alpha_matting_erode_size, 0, 50),
        )

        rgba_pil = ensure_rgba(cutout_png)
        rgba = np.array(rgba_pil)  # RGBA uint8

        # --- Alpha shaping (keep crisp to avoid halo) ---
        a = rgba[:, :, 3]
        a = erode(a, clamp_int(edge_erode_px, 0, 30))
        a = dilate(a, clamp_int(edge_dilate_px, 0, 30))
        a = feather(a, clamp_int(edge_feather_px, 0, 30))
        rgba[:, :, 3] = a

        # --- RGB decontamination on semi-transparent edge band ---
        if decontaminate:
            rgba = decontaminate_edge_rgb(
                rgba,
                alpha_max=clamp_int(decontaminate_alpha_max, 1, 254),
                inpaint_radius=clamp_int(decontaminate_inpaint_radius, 1, 12),
            )

        tw = clamp_int(target_w, 200, 6000)
        th = clamp_int(target_h, 200, 8000)
        fitted = object_aware_fit(
            rgba,
            target_w=tw,
            target_h=th,
            padding_ratio=float(padding_ratio),
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
                # Helps avoid browser caching while you're iterating
                "Cache-Control": "no-store",
            },
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        # Always release semaphore + aggressively free memory between requests
        try:
            SEM.release()
        except Exception:
            pass

        # Drop big objects
        try:
            del rgba
            del fitted
            del pil_img
            del rgba_pil
            del tmp
        except Exception:
            pass

        gc.collect()
