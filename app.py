import io
import os
import zipfile
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse

from PIL import Image, ImageFilter, ImageChops
from rembg import remove


app = FastAPI(title="PartLogic Image Processor", version="1.2.0")


# -----------------------------
# Helpers
# -----------------------------

def _safe(s: Optional[str]) -> str:
    return (s or "").strip()


def build_filename(mpn: str, sku: str, prefix: str = "partlogic", ext: str = "webp") -> str:
    mpn = _safe(mpn)
    sku = _safe(sku)
    base = f'{prefix}_{mpn}_{sku}' if (mpn or sku) else prefix
    # very light sanitization
    base = "".join(c for c in base if c.isalnum() or c in ("_", "-", "."))
    return f"{base}.{ext}"


def parse_canvas(canvas: str, default=(1400, 1700)) -> Tuple[int, int]:
    """
    canvas: "1400x1700"
    """
    try:
        w, h = canvas.lower().split("x")
        w, h = int(w), int(h)
        if w <= 0 or h <= 0:
            return default
        return w, h
    except Exception:
        return default


def to_rgba(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGBA")
    return img


def alpha_bbox(img_rgba: Image.Image, alpha_threshold: int = 8):
    """
    Return bounding box of alpha channel above threshold.
    """
    a = img_rgba.split()[-1]
    # convert to binary-ish mask for bbox
    mask = a.point(lambda p: 255 if p > alpha_threshold else 0)
    return mask.getbbox()


def object_aware_fit(img_rgba: Image.Image, canvas_w: int, canvas_h: int, max_fill: float = 1.0) -> Image.Image:
    """
    Scales the detected object (via alpha bbox) to be as large as possible within the canvas
    without cropping, preserving aspect ratio. Object may touch sides or top/bottom.
    max_fill allows optional margin (<1.0). We keep 1.0 (tight) per your request.
    """
    img_rgba = to_rgba(img_rgba)

    bbox = alpha_bbox(img_rgba, alpha_threshold=8)
    if not bbox:
        # no alpha info; just fit whole image
        bbox = (0, 0, img_rgba.width, img_rgba.height)

    x0, y0, x1, y1 = bbox
    obj = img_rgba.crop((x0, y0, x1, y1))

    # Fit object into canvas
    obj_w, obj_h = obj.size
    if obj_w <= 0 or obj_h <= 0:
        return Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    scale = min((canvas_w / obj_w), (canvas_h / obj_h)) * max_fill
    new_w = max(1, int(round(obj_w * scale)))
    new_h = max(1, int(round(obj_h * scale)))

    obj_resized = obj.resize((new_w, new_h), resample=Image.LANCZOS)

    # Center on canvas
    out = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    px = (canvas_w - new_w) // 2
    py = (canvas_h - new_h) // 2
    out.paste(obj_resized, (px, py), obj_resized)

    return out


def defringe_edges(img_rgba: Image.Image, strength: int = 1) -> Image.Image:
    """
    Reduces edge halos by slightly shrinking alpha and smoothing edge.
    strength=1 is conservative.
    """
    img_rgba = to_rgba(img_rgba)
    r, g, b, a = img_rgba.split()

    # Slightly erode alpha by using MinFilter, then soften a touch
    # (strength controls how aggressively we shrink edges)
    if strength > 0:
        a2 = a.filter(ImageFilter.MinFilter(size=3))
        # very small blur to avoid jaggies from erosion
        a2 = a2.filter(ImageFilter.GaussianBlur(radius=0.35))
    else:
        a2 = a

    out = Image.merge("RGBA", (r, g, b, a2))
    return out


def rembg_cutout(pil_img: Image.Image) -> Image.Image:
    """
    Background removal with better edges using alpha matting.
    """
    pil_img = pil_img.convert("RGBA")

    # rembg params tuned to reduce halos but not delete object
    # You can tweak these later if needed.
    out = remove(
        pil_img,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=15,
        alpha_matting_erode_size=12
    )

    if not isinstance(out, Image.Image):
        # rembg can return bytes in some configurations, handle safely
        out = Image.open(io.BytesIO(out)).convert("RGBA")

    # extra conservative defringe
    out = defringe_edges(out, strength=1)
    return out


def encode_lossless_webp(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(
        buf,
        format="WEBP",
        lossless=True,
        quality=100,
        method=6
    )
    buf.seek(0)
    return buf


def composite_background(img_rgba: Image.Image, bg: str) -> Image.Image:
    """
    bg: "transparent" (default) or "white"
    """
    bg = (_safe(bg) or "transparent").lower()
    img_rgba = to_rgba(img_rgba)

    if bg == "white":
        base = Image.new("RGB", (img_rgba.width, img_rgba.height), (255, 255, 255))
        base.paste(img_rgba, (0, 0), img_rgba)
        return base
    return img_rgba


# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process")
async def process(
    # Backwards-compatible: accept either "file" or "files"
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),

    mpn: str = Form(""),
    sku: str = Form(""),
    prefix: str = Form("partlogic"),

    # Output controls
    background: str = Form("transparent"),     # "transparent" or "white"
    canvas: str = Form("1400x1700"),           # must be 14:17 for your workflow
):
    # Collect uploads
    uploads: List[UploadFile] = []
    if files:
        uploads.extend(files)
    if file:
        uploads.append(file)

    if not uploads:
        return JSONResponse({"error": "No file(s) uploaded. Use form-data field 'file' or 'files'."}, status_code=400)

    canvas_w, canvas_h = parse_canvas(canvas, default=(1400, 1700))

    # If multiple uploads, return a ZIP to preserve all outputs
    if len(uploads) > 1:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            for idx, up in enumerate(uploads, start=1):
                raw = await up.read()
                pil = Image.open(io.BytesIO(raw)).convert("RGBA")

                cut = rembg_cutout(pil)
                fitted = object_aware_fit(cut, canvas_w, canvas_h, max_fill=1.0)
                final_img = composite_background(fitted, background)

                out_name = build_filename(mpn, sku, prefix=prefix, ext="webp")
                # avoid collisions inside zip
                if idx > 1:
                    out_name = out_name.replace(".webp", f"_{idx}.webp")

                wb = encode_lossless_webp(final_img)
                z.writestr(out_name, wb.getvalue())

        zbuf.seek(0)
        headers = {"Content-Disposition": 'attachment; filename="partlogic_processed_images.zip"'}
        return StreamingResponse(zbuf, media_type="application/zip", headers=headers)

    # Single upload -> single WebP
    up = uploads[0]
    raw = await up.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGBA")

    cut = rembg_cutout(pil)
    fitted = object_aware_fit(cut, canvas_w, canvas_h, max_fill=1.0)
    final_img = composite_background(fitted, background)

    out_name = build_filename(mpn, sku, prefix=prefix, ext="webp")
    buf = encode_lossless_webp(final_img)

    headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
    return StreamingResponse(buf, media_type="image/webp", headers=headers)
