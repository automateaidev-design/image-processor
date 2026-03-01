import io
import os
from typing import Optional, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from PIL import Image

from rembg import remove, new_session

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="image-processor", version="2.0.0")

# Reuse model session across requests (big speed boost)
REMBG_MODEL = os.getenv("REMBG_MODEL", "u2net")  # u2net is good general default
session = new_session(REMBG_MODEL)

# Target aspect ratio: 14:17
TARGET_W = int(os.getenv("TARGET_W", "1400"))
TARGET_H = int(os.getenv("TARGET_H", "1700"))

# Safety / limits
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


# -----------------------------
# Helpers
# -----------------------------
def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def read_upload_bytes(file: UploadFile) -> bytes:
    data = file.file.read()
    if not data:
        raise ValueError("Empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large ({len(data)} bytes). Max is {MAX_UPLOAD_BYTES} bytes.")
    return data


def pil_open_rgb(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    # Convert paletted etc. to RGB for consistent handling
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def pre_upscale_if_small(img: Image.Image, min_max_dim: int = 900) -> Image.Image:
    """
    Upscale small images before segmentation.
    This improves edge quality a lot for tiny supplier thumbnails.
    """
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


def pil_to_cv_rgba(img: Image.Image) -> np.ndarray:
    arr = np.array(img)  # RGBA uint8
    return arr


def cv_rgba_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, mode="RGBA")


def dilate_alpha(alpha: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return alpha
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(alpha, kernel, iterations=1)


def feather_alpha(alpha: np.ndarray, px: int) -> np.ndarray:
    """
    Feather (blur) alpha edges. For industrial parts, default should be 0 (crisp).
    """
    if px <= 0:
        return alpha
    k = 2 * px + 1
    # Use a gentle gaussian blur; keep it modest to avoid halos
    return cv2.GaussianBlur(alpha, (k, k), 0)


def alpha_bbox(alpha: np.ndarray, thresh: int = 8) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns (x1, y1, x2, y2) bbox where alpha > thresh
    """
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
    """
    Crop to object bbox, then scale to fit within target canvas (14:17),
    as large as possible WITHOUT cropping the object.
    """
    pad = float(padding_ratio)
    pad = max(0.0, min(0.49, pad))  # sanity

    alpha = rgba[:, :, 3]
    bbox = alpha_bbox(alpha, thresh=alpha_thresh)
    if bbox is None:
        # No alpha? return transparent canvas
        return np.zeros((target_h, target_w, 4), dtype=np.uint8)

    x1, y1, x2, y2 = bbox

    # Crop to object bbox
    obj = rgba[y1:y2, x1:x2, :]

    obj_h, obj_w = obj.shape[:2]

    # Inner canvas size after padding
    inner_w = int(round(target_w * (1.0 - 2.0 * pad)))
    inner_h = int(round(target_h * (1.0 - 2.0 * pad)))
    inner_w = max(1, inner_w)
    inner_h = max(1, inner_h)

    # Scale factor to fit object inside inner box
    scale = min(inner_w / obj_w, inner_h / obj_h)
    new_w = max(1, int(round(obj_w * scale)))
    new_h = max(1, int(round(obj_h * scale)))

    # Resize object
    obj_resized = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create output canvas
    out = np.zeros((target_h, target_w, 4), dtype=np.uint8)

    # Center placement
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2

    # Alpha composite onto canvas
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
        # keep it filesystem/url safe-ish
        s = s.replace(" ", "_")
        s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))
        return s[:140] if s else "na"

    return f"{clean(prefix)}_{clean(mpn)}_{clean(sku)}.{ext}"


def save_lossless_webp(rgba_arr: np.ndarray) -> bytes:
    img = cv_rgba_to_pil(rgba_arr)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", lossless=True, quality=100, method=6)
    return buf.getvalue()


def save_png(rgba_arr: np.ndarray) -> bytes:
    img = cv_rgba_to_pil(rgba_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "model": REMBG_MODEL}


@app.post("/process")
async def process_image(
    file: UploadFile = File(..., description="Image file upload (multipart/form-data)"),

    # Naming
    mpn: str = Form(""),
    sku: str = Form(""),
    prefix: str = Form("partlogic"),

    # Output control
    output: str = Form("webp"),  # webp | png

    # Background removal knobs (these map to your "magic wand threshold" feeling)
    alpha_matting: bool = Form(True),
    alpha_matting_foreground_threshold: int = Form(220),
    alpha_matting_background_threshold: int = Form(3),
    alpha_matting_erode_size: int = Form(0),

    # Edge control (industrial defaults = crisp)
    edge_dilate_px: int = Form(4),
    edge_feather_px: int = Form(0),

    # Canvas control
    padding_ratio: float = Form(0.0),   # set 0 if you want the object as large as possible
    target_w: int = Form(TARGET_W),
    target_h: int = Form(TARGET_H),

    # Quality/perf
    pre_upscale: bool = Form(True),
    pre_upscale_min_dim: int = Form(900),
):
    try:
        # Read upload
        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": f"File too large. Max {MAX_UPLOAD_MB}MB"}, status_code=413)

        # Load and optionally upscale BEFORE segmentation
        pil_img = pil_open_rgb(raw)
        if pre_upscale:
            pil_img = pre_upscale_if_small(pil_img, min_max_dim=clamp_int(pre_upscale_min_dim, 256, 2400))

        # Convert to PNG bytes for rembg
        tmp = io.BytesIO()
        pil_img.save(tmp, format="PNG")
        png_in = tmp.getvalue()

        # Background removal
        # remove() expects bytes -> returns bytes (PNG with alpha)
        cutout_png = remove(
            png_in,
            session=session,
            alpha_matting=bool(alpha_matting),
            alpha_matting_foreground_threshold=clamp_int(alpha_matting_foreground_threshold, 0, 255),
            alpha_matting_background_threshold=clamp_int(alpha_matting_background_threshold, 0, 255),
            alpha_matting_erode_size=clamp_int(alpha_matting_erode_size, 0, 50),
        )

        # Post-process alpha for crisp industrial edges
        rgba = ensure_rgba(cutout_png)
        rgba_cv = pil_to_cv_rgba(rgba)

        alpha = rgba_cv[:, :, 3]
        alpha = dilate_alpha(alpha, clamp_int(edge_dilate_px, 0, 30))
        alpha = feather_alpha(alpha, clamp_int(edge_feather_px, 0, 30))

        # Re-apply alpha
        rgba_cv[:, :, 3] = alpha

        # Object-aware fit into 14:17 canvas (no cropping)
        tw = clamp_int(target_w, 200, 6000)
        th = clamp_int(target_h, 200, 8000)
        fitted = object_aware_fit(
            rgba_cv,
            target_w=tw,
            target_h=th,
            padding_ratio=float(padding_ratio),
            alpha_thresh=8,
        )

        # Output bytes
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
                "Content-Disposition": f'inline; filename="{filename}"'
            },
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
