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

# ----------------------------
# App + model session
# ----------------------------

app = FastAPI(title="image-processor", version="2.4.0")

REMBG_MODEL = os.getenv("REMBG_MODEL", "isnet-general-use")
_session = None

TARGET_W = int(os.getenv("TARGET_W", "1400"))
TARGET_H = int(os.getenv("TARGET_H", "1700"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

SEM = asyncio.Semaphore(max(1, MAX_CONCURRENCY))


@app.on_event("startup")
async def _startup():
    global _session
    _session = new_session(REMBG_MODEL)

    if ENABLE_SELF_RESTART:
        async def _restarter():
            await asyncio.sleep(max(60, SELF_RESTART_SECONDS))
            os._exit(0)  # noqa: S606

        asyncio.create_task(_restarter())


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def pil_open_rgb(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        img = img.convert("RGB")
    return img


def pre_upscale_if_small(img: Image.Image, min_max_dim: int = 1100) -> Image.Image:
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


def boost_edge_contrast(
    img: Image.Image,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    saturation_boost: float = 1.08,
) -> Image.Image:
    arr = np.array(img)

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=max(1.0, float(clahe_clip_limit)),
        tileGridSize=(max(2, int(clahe_tile_size)), max(2, int(clahe_tile_size))),
    )
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if abs(float(saturation_boost) - 1.0) > 1e-6:
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation_boost), 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return Image.fromarray(enhanced)


def decontaminate_edge_rgb(
    rgba: np.ndarray,
    alpha_min: int = 8,
    alpha_max: int = 160,
    inpaint_radius: int = 4,
) -> np.ndarray:
    alpha = rgba[:, :, 3]
    mask = ((alpha >= alpha_min) & (alpha <= alpha_max)).astype(np.uint8) * 255
    if mask.max() == 0:
        return rgba

    rgb = rgba[:, :, :3]
    rgb_fixed = cv2.inpaint(
        rgb,
        mask,
        inpaintRadius=max(1, int(inpaint_radius)),
        flags=cv2.INPAINT_TELEA,
    )
    out = rgba.copy()
    out[:, :, :3] = rgb_fixed
    return out


@app.get("/health")
def health():
    return {"ok": True, "model": REMBG_MODEL, "max_concurrency": MAX_CONCURRENCY}


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

    alpha_matting: bool = Form(True),
    alpha_matting_foreground_threshold: int = Form(235),
    alpha_matting_background_threshold: int = Form(20),
    alpha_matting_erode_size: int = Form(1),

    edge_erode_px: int = Form(0),
    edge_dilate_px: int = Form(0),
    edge_feather_px: int = Form(0),

    decontaminate: bool = Form(True),
    decontaminate_alpha_min: int = Form(8),
    decontaminate_alpha_max: int = Form(160),
    decontaminate_inpaint_radius: int = Form(4),

    padding_ratio: float = Form(0.02),
    target_w: int = Form(TARGET_W),
    target_h: int = Form(TARGET_H),

    pre_upscale: bool = Form(True),
    pre_upscale_min_dim: int = Form(1100),

    enhance_contrast: bool = Form(True),
    clahe_clip_limit: float = Form(2.0),
    clahe_tile_size: int = Form(8),
    saturation_boost: float = Form(1.08),
):
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
            return JSONResponse({"error": "Model session not ready"}, status_code=503)

        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": f"File too large. Max {MAX_UPLOAD_MB}MB"}, status_code=413)

        pil_img = pil_open_rgb(raw)

        w, h = pil_img.size
        max_dim = int(os.getenv("MAX_IMAGE_DIM", "8000"))
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

        if pre_upscale:
            pil_img = pre_upscale_if_small(
                pil_img,
                min_max_dim=clamp_int(pre_upscale_min_dim, 256, 2400),
            )

        if enhance_contrast:
            pil_img = boost_edge_contrast(
                pil_img,
                clahe_clip_limit=clamp_float(clahe_clip_limit, 1.0, 6.0),
                clahe_tile_size=clamp_int(clahe_tile_size, 2, 32),
                saturation_boost=clamp_float(saturation_boost, 1.0, 1.5),
            )

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
        rgba = np.array(rgba_pil)

        a = rgba[:, :, 3]
        a = erode(a, clamp_int(edge_erode_px, 0, 30))
        a = dilate(a, clamp_int(edge_dilate_px, 0, 30))
        a = feather(a, clamp_int(edge_feather_px, 0, 30))
        rgba[:, :, 3] = a

        if decontaminate:
            rgba = decontaminate_edge_rgb(
                rgba,
                alpha_min=clamp_int(decontaminate_alpha_min, 0, 254),
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
            del rgba
            del fitted
            del pil_img
            del rgba_pil
            del tmp
        except Exception:
            pass

        gc.collect()
