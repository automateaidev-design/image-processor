# app.py — COMPLETE DROP-IN REPLACEMENT
# ------------------------------------------------------------
# FastAPI endpoint that accepts multipart/form-data:
#   - file: image upload (required)
#   - mpn: text (optional)
#   - sku: text (optional)
# Returns:
#   - a single edited PNG (transparent background)
#   - object-aware scaled into a fixed 14:17 canvas (max size, no cropping)
#   - Content-Disposition filename: partlogic_<mpn>_<sku>.png (sanitised)
#
# Requirements (you already have most):
#   fastapi
#   uvicorn
#   rembg
#   onnxruntime
#   pillow
#   opencv-python-headless
#   numpy
#   python-multipart   <-- REQUIRED for File/Form
# ------------------------------------------------------------

import io
import re
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps
from rembg import remove


APP_TITLE = "PartLogic Image Processor"
APP_VERSION = "1.0.0"

# Output canvas settings
CANVAS_RATIO = (14, 17)      # width : height
CANVAS_LONG_SIDE = 1700      # final height; width auto from ratio


app = FastAPI(title=APP_TITLE, version=APP_VERSION)


# ----------------------------
# Helpers
# ----------------------------

def _safe_filename_part(s: Optional[str]) -> str:
    """
    Shopify-safe-ish filename component: keep letters/numbers/_- only, collapse repeats.
    If empty/None -> 'NA'
    """
    if not s:
        return "NA"
    s = str(s).strip()
    if not s:
        return "NA"
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    s = re.sub(r"[_\-]{2,}", "_", s)
    return s or "NA"


def object_aware_scale_aspect(
    rgba: Image.Image,
    canvas_ratio=(14, 17),   # width : height
    canvas_long_side=1700,   # final height; width computed from ratio
    resample=Image.LANCZOS
) -> Image.Image:
    """
    Object-aware scaling to a fixed aspect-ratio transparent canvas.

    - Keeps intrinsic object proportions (no stretching)
    - Object becomes as large as possible without cropping
    - Transparent background
    """

    rgba = rgba.convert("RGBA")

    # 1) Tight bbox from alpha
    alpha = rgba.split()[-1]
    bbox = alpha.getbbox()

    # If no alpha content detected, return blank canvas
    canvas_h = int(canvas_long_side)
    canvas_w = int(round(canvas_h * canvas_ratio[0] / canvas_ratio[1]))
    if bbox is None:
        return Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    cropped = rgba.crop(bbox)
    cw, ch = cropped.size

    # 2) Scale so object fits fully inside canvas
    scale = min(canvas_w / cw, canvas_h / ch)

    # Optional tiny safety margin (prevents accidental edge clipping)
    scale *= 0.995

    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    resized = cropped.resize((new_w, new_h), resample)

    # 3) Center on transparent canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    canvas.alpha_composite(resized, (x, y))
    return canvas


def read_upload_as_pil(file: UploadFile) -> Image.Image:
    """
    Reads UploadFile into a PIL Image. Handles EXIF orientation.
    """
    try:
        raw = file.file.read()
        if not raw:
            raise ValueError("Empty file")
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)  # correct phone camera rotations
        # Ensure loaded
        img.load()
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")


def rembg_remove_to_rgba(img: Image.Image) -> Image.Image:
    """
    Background removal using rembg.
    Returns RGBA PIL image.
    """
    # rembg.remove works best on bytes / numpy; we'll do numpy -> bytes output
    try:
        img = img.convert("RGBA")
        arr = np.array(img)
        out = remove(arr)  # returns numpy array (usually) with alpha
        out_img = Image.fromarray(out).convert("RGBA")
        return out_img
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {e}")


# ----------------------------
# Routes
# ----------------------------

@app.get("/health")
def health():
    return {"ok": True, "service": APP_TITLE, "version": APP_VERSION}


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    mpn: Optional[str] = Form(None),
    sku: Optional[str] = Form(None),
):
    """
    Multipart/form-data:
      file=<image>
      mpn=<text> (optional)
      sku=<text> (optional)

    Returns a single PNG (transparent background), object-aware scaled into 14:17 canvas.
    """
    # Basic content-type sanity check (not strict, but helps)
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    img = read_upload_as_pil(file)

    # 1) Remove background
    rgba = rembg_remove_to_rgba(img)

    # 2) Object-aware scale into fixed 14:17 canvas
    final_img = object_aware_scale_aspect(
        rgba,
        canvas_ratio=CANVAS_RATIO,
        canvas_long_side=CANVAS_LONG_SIDE
    )

    # 3) Build output filename
    mpn_part = _safe_filename_part(mpn)
    sku_part = _safe_filename_part(sku)
    out_name = f"partlogic_{mpn_part}_{sku_part}.png"

    # 4) Stream PNG bytes
    buf = io.BytesIO()
    final_img.save(buf, format="PNG", optimize=False)  # max quality (lossless); no palette quantization
    buf.seek(0)

    headers = {
        "Content-Disposition": f'attachment; filename="{out_name}"'
    }
    return StreamingResponse(buf, media_type="image/png", headers=headers)


# Optional: helpful root message
@app.get("/")
def root():
    return JSONResponse(
        {
            "service": APP_TITLE,
            "version": APP_VERSION,
            "endpoints": {
                "health": "GET /health",
                "process": "POST /process (multipart/form-data: file, mpn?, sku?)",
                "docs": "GET /docs",
            },
            "output": {
                "format": "PNG",
                "background": "transparent",
                "canvas_ratio": f"{CANVAS_RATIO[0]}:{CANVAS_RATIO[1]}",
                "canvas_pixels": f"{int(round(CANVAS_LONG_SIDE * CANVAS_RATIO[0] / CANVAS_RATIO[1]))}x{CANVAS_LONG_SIDE}",
                "filename": 'partlogic_<mpn>_<sku>.png (NA if missing)',
            },
        }
    )
