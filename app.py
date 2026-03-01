import os
import io
import asyncio
from typing import Optional

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from rembg import remove

app = FastAPI(title="image-processor")

# Limit concurrent heavy requests to avoid 502s on Railway
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
_sema = asyncio.Semaphore(MAX_CONCURRENCY)

# Default canvas ratio: 14:17 (width:height)
DEFAULT_W = int(os.getenv("OUT_W", "1400"))
DEFAULT_H = int(os.getenv("OUT_H", "1700"))

def _to_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))

def _clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))

def _safe_filename(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))
    return s[:180] if s else "partlogic_image"

def _expand_and_smooth_alpha(alpha: np.ndarray, dilate_px: int, feather_px: int) -> np.ndarray:
    """
    alpha: uint8 0..255
    - dilate_px expands mask outward to keep edges (prevents eating).
    - feather_px smooths edge to remove pixelation.
    """
    out = alpha.copy()

    if dilate_px > 0:
        # Binary mask then dilate, then merge back by taking max
        binmask = (out > 0).astype(np.uint8) * 255
        k = _clamp_int(dilate_px, 0, 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        dil = cv2.dilate(binmask, kernel, iterations=1)
        out = np.maximum(out, dil)

    if feather_px > 0:
        r = _clamp_int(feather_px, 0, 30)
        # kernel size must be odd
        ks = 2 * r + 1
        out = cv2.GaussianBlur(out, (ks, ks), 0)

    return out

def _object_aware_fit_rgba(
    rgba: Image.Image,
    out_w: int,
    out_h: int,
    padding_ratio: float = 0.00,
    alpha_bbox_threshold: int = 8,
) -> Image.Image:
    """
    Takes RGBA image, finds alpha bbox, adds padding, then scales object to fit
    inside 14:17 canvas as large as possible without cropping.
    """
    rgba = rgba.convert("RGBA")
    np_rgba = np.array(rgba)
    alpha = np_rgba[:, :, 3]

    ys, xs = np.where(alpha > alpha_bbox_threshold)
    if len(xs) == 0 or len(ys) == 0:
        # No alpha content; just center the original
        canvas = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        thumb = rgba.copy()
        thumb.thumbnail((out_w, out_h), Image.LANCZOS)
        x0 = (out_w - thumb.width) // 2
        y0 = (out_h - thumb.height) // 2
        canvas.paste(thumb, (x0, y0), thumb)
        return canvas

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # padding around bbox
    bw = (x2 - x1 + 1)
    bh = (y2 - y1 + 1)
    pad = int(round(max(bw, bh) * padding_ratio))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(rgba.width - 1, x2 + pad)
    y2 = min(rgba.height - 1, y2 + pad)

    crop = rgba.crop((x1, y1, x2 + 1, y2 + 1))

    # scale to fit inside canvas without cropping
    scale = min(out_w / crop.width, out_h / crop.height)
    new_w = max(1, int(round(crop.width * scale)))
    new_h = max(1, int(round(crop.height * scale)))

    resized = crop.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas.paste(resized, (x0, y0), resized)
    return canvas

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/process")
async def process(
    file: UploadFile = File(...),

    # output controls
    output: str = Form("webp"),  # webp or png
    out_w: int = Form(DEFAULT_W),
    out_h: int = Form(DEFAULT_H),

    # object fit controls
    padding_ratio: float = Form(0.00),

    # rembg controls
    alpha_matting: str = Form("true"),
    alpha_matting_foreground_threshold: int = Form(200),
    alpha_matting_background_threshold: int = Form(5),
    alpha_matting_erode_size: int = Form(0),

    # edge keep controls (prevents eating)
    edge_dilate_px: int = Form(6),
    edge_feather_px: int = Form(2),

    # naming (optional)
    filename: Optional[str] = Form(None),
):
    async with _sema:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload")

        out_w = _clamp_int(out_w, 200, 6000)
        out_h = _clamp_int(out_h, 200, 6000)
        padding_ratio = _clamp_float(padding_ratio, 0.0, 0.5)

        # rembg remove() expects bytes and returns bytes (PNG)
        am = _to_bool(alpha_matting, True)
        fg = _clamp_int(alpha_matting_foreground_threshold, 0, 255)
        bg = _clamp_int(alpha_matting_background_threshold, 0, 255)
        er = _clamp_int(alpha_matting_erode_size, 0, 60)

        try:
            removed_png_bytes = remove(
                data,
                alpha_matting=am,
                alpha_matting_foreground_threshold=fg,
                alpha_matting_background_threshold=bg,
                alpha_matting_erode_size=er,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"rembg failed: {e}")

        rgba = Image.open(io.BytesIO(removed_png_bytes)).convert("RGBA")

        # edge keep adjustments
        np_rgba = np.array(rgba)
        alpha = np_rgba[:, :, 3].astype(np.uint8)
        dil = _clamp_int(edge_dilate_px, 0, 30)
        fea = _clamp_int(edge_feather_px, 0, 30)
        alpha2 = _expand_and_smooth_alpha(alpha, dilate_px=dil, feather_px=fea)
        np_rgba[:, :, 3] = alpha2
        rgba = Image.fromarray(np_rgba, mode="RGBA")

        # fit into 14:17 canvas as large as possible without cropping
        final_img = _object_aware_fit_rgba(
            rgba,
            out_w=out_w,
            out_h=out_h,
            padding_ratio=padding_ratio,
            alpha_bbox_threshold=8,
        )

        # encode
        fmt = (output or "webp").strip().lower()
        buf = io.BytesIO()

        if fmt == "png":
            final_img.save(buf, format="PNG", optimize=False)
            media = "image/png"
            ext = "png"
        else:
            # lossless webp, keep alpha
            final_img.save(
                buf,
                format="WEBP",
                lossless=True,
                quality=100,
                method=6,
            )
            media = "image/webp"
            ext = "webp"

        out_bytes = buf.getvalue()

        # filename returned in header (n8n can read it)
        if filename:
            fn = _safe_filename(filename)
        else:
            base = os.path.splitext(file.filename or "partlogic_image")[0]
            fn = _safe_filename(base)
        final_name = f"{fn}.{ext}"

        return Response(
            content=out_bytes,
            media_type=media,
            headers={
                "Content-Disposition": f'inline; filename="{final_name}"',
                "X-Filename": final_name,
            },
        )
