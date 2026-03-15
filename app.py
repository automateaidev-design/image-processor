import io
import os
import gc
import asyncio
import tempfile
import subprocess
from typing import Optional, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from PIL import Image

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
QUEUE_TIMEOUT_S = float(os.getenv("QUEUE_TIMEOUT_S", "30"))

ENABLE_SELF_RESTART = os.getenv("ENABLE_SELF_RESTART", "0") == "1"
SELF_RESTART_SECONDS = int(os.getenv("SELF_RESTART_SECONDS", "600"))

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(title="image-processor", version="7.1.0")

TARGET_W = int(os.getenv("TARGET_W", "1400"))
TARGET_H = int(os.getenv("TARGET_H", "1700"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "8000"))

# Timeout for rembg-cli subprocess (seconds)
REMBG_TIMEOUT = int(os.getenv("REMBG_TIMEOUT", "60"))


SEM = asyncio.Semaphore(max(1, MAX_CONCURRENCY))


@app.on_event("startup")
async def _startup():
    if ENABLE_SELF_RESTART:
        async def _restarter():
            await asyncio.sleep(max(60, SELF_RESTART_SECONDS))
            os._exit(0)
        asyncio.create_task(_restarter())


# -------------------------------------------------
# Utility helpers
# -------------------------------------------------

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
    """
    Crop to the object bounding box, scale to fit within the target canvas
    with padding, and centre on a transparent background.
    Identical to the original resizing logic.
    """
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


def resize_if_huge(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_DIM:
        return img
    scale = MAX_IMAGE_DIM / float(max(w, h))
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)


# -------------------------------------------------
# rembg-cli inference
# -------------------------------------------------

def rembg_cli_rgba(pil_img: Image.Image) -> np.ndarray:
    """
    Pass the image through rembg-cli exactly as the upstream repo does,
    then return the raw H×W×4 RGBA numpy array.
    No additional post-processing — the cli output is used as-is.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.png")
        out_path = os.path.join(tmpdir, "output.png")

        pil_img.save(in_path)

        result = subprocess.run(
            ["rembg", "-i", in_path, "-o", out_path],
            capture_output=True,
            timeout=REMBG_TIMEOUT,
        )

        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            raise RuntimeError(f"rembg-cli failed (exit {result.returncode}): {err}")

        out_img = Image.open(out_path)
        if out_img.mode != "RGBA":
            out_img = out_img.convert("RGBA")
        return np.array(out_img)


# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": "rembg-cli",
        "max_concurrency": MAX_CONCURRENCY,
        "version": "7.1.0",
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

    pil_img = rgba = fitted = None

    try:
        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": f"File too large. Max {MAX_UPLOAD_MB}MB"}, status_code=413)

        pil_img = pil_open_rgb(raw)
        pil_img = resize_if_huge(pil_img)

        # rembg-cli: exact upstream processing
        rgba = rembg_cli_rgba(pil_img)

        if rgba is None or rgba[:, :, 3].max() == 0:
            return JSONResponse({"error": "Failed to extract foreground"}, status_code=422)

        # Resize/fit to target canvas — same as original
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

    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "rembg-cli timed out"}, status_code=504)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        try:
            SEM.release()
        except Exception:
            pass
        try:
            del pil_img, rgba, fitted
        except Exception:
            pass
        gc.collect()
