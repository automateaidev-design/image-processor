import io
import os
import gc
import json
import shutil
import asyncio
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from PIL import Image

try:
    cv2.setNumThreads(1)
except Exception:
    pass

PIL_MAX_IMAGE_PIXELS = int(os.getenv("PIL_MAX_IMAGE_PIXELS", "60000000"))
Image.MAX_IMAGE_PIXELS = PIL_MAX_IMAGE_PIXELS

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
QUEUE_TIMEOUT_S = float(os.getenv("QUEUE_TIMEOUT_S", "30"))

TARGET_W = int(os.getenv("TARGET_W", "1400"))
TARGET_H = int(os.getenv("TARGET_H", "1700"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "8000"))

RMBG_CLI_DIR = os.getenv("RMBG_CLI_DIR", "/opt/rembg-cli")
NODE_BIN = os.getenv("NODE_BIN", "node")
X_API_KEY = os.getenv("X_API_KEY", "")

app = FastAPI(title="image-processor", version="5.0.0")
SEM = asyncio.Semaphore(max(1, MAX_CONCURRENCY))


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def pil_open_rgb(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_if_huge(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_DIM:
        return img
    scale = MAX_IMAGE_DIM / float(max(w, h))
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)


def ensure_rgba_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im


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


def build_filename(prefix: str, mpn: str, sku: str, ext: str) -> str:
    def clean(s: str) -> str:
        s = (s or "").strip()
        s = s.replace(" ", "_")
        s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))
        return s[:140] if s else "na"

    return f"{clean(prefix)}_{clean(mpn)}_{clean(sku)}.{ext}"


def run_rembg_cli(input_bytes: bytes) -> bytes:
    if not X_API_KEY:
        raise RuntimeError("Missing X_API_KEY environment variable")

    cli_entry = os.path.join(RMBG_CLI_DIR, "index.js")
    if not os.path.exists(cli_entry):
        raise RuntimeError(f"rembg-cli not found at {cli_entry}")

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input.png")
        out_path = os.path.join(td, "output.png")

        with open(in_path, "wb") as f:
            f.write(input_bytes)

        env = os.environ.copy()
        env["X_API_KEY"] = X_API_KEY

        # This assumes the cloned repo exposes a CLI that accepts:
        # node index.js <input> <output>
        # If your specific fork differs, only this command line needs adjusting.
        cmd = [NODE_BIN, cli_entry, in_path, out_path]

        proc = subprocess.run(
            cmd,
            cwd=RMBG_CLI_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"rembg-cli failed: {proc.stderr.strip() or proc.stdout.strip() or 'unknown error'}"
            )

        if not os.path.exists(out_path):
            raise RuntimeError("rembg-cli did not produce an output file")

        with open(out_path, "rb") as f:
            return f.read()


@app.get("/health")
def health():
    cli_entry = os.path.join(RMBG_CLI_DIR, "index.js")
    return {
        "ok": True,
        "version": "5.0.0",
        "rembg_cli_present": os.path.exists(cli_entry),
        "x_api_key_present": bool(X_API_KEY),
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
            {"error": "Service busy", "hint": "Reduce parallelism or increase MAX_CONCURRENCY"},
            status_code=503,
        )

    pil_img = None
    cutout_img = None
    rgba = None
    fitted = None

    try:
        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": f"File too large. Max {MAX_UPLOAD_MB}MB"}, status_code=413)

        pil_img = pil_open_rgb(raw)
        pil_img = resize_if_huge(pil_img)

        tmp = io.BytesIO()
        pil_img.save(tmp, format="PNG")
        png_in = tmp.getvalue()

        cutout_bytes = run_rembg_cli(png_in)
        cutout_img = ensure_rgba_from_bytes(cutout_bytes)
        rgba = np.array(cutout_img)

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
            del pil_img
            del cutout_img
            del rgba
            del fitted
        except Exception:
            pass

        gc.collect()
