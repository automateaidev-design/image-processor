import io
import os
from typing import Optional, Tuple

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse

from rembg import remove, new_session


app = FastAPI(title="image-processor")

# Use a higher-quality general model. If you already use something else, you can change this.
# Good options: "isnet-general-use" (often better edges), "u2net"
REMBG_MODEL = os.getenv("REMBG_MODEL", "isnet-general-use")
session = new_session(REMBG_MODEL)


# ----------------------------
# Helpers
# ----------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _parse_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _safe_token(s: str) -> str:
    """
    filename-safe (no slashes, etc). Keeps letters/numbers/_- only.
    """
    s = (s or "").strip()
    if not s:
        return ""
    s = s.replace(" ", "-")
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_"))
    return s[:160]


def _pil_to_rgba(im: Image.Image) -> Image.Image:
    if im.mode != "RGBA":
        return im.convert("RGBA")
    return im


def _edge_refine_alpha(rgba: Image.Image, dilate_px: int = 2, feather_px: int = 2) -> Image.Image:
    """
    Refines alpha edges:
    - small dilation to prevent clipping (“eaten” object edges)
    - small gaussian blur to anti-alias the edge
    """
    rgba = _pil_to_rgba(rgba)
    arr = np.array(rgba)  # H,W,4
    alpha = arr[:, :, 3].astype(np.uint8)

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        alpha = cv2.dilate(alpha, kernel, iterations=1)

    if feather_px > 0:
        k = 2 * feather_px + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), sigmaX=0)

    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def _bbox_from_alpha(alpha: np.ndarray, thresh: int = 8) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns bounding box (x0,y0,x1,y1) for alpha>thresh, else None.
    """
    ys, xs = np.where(alpha > thresh)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def _fit_object_on_canvas(
    rgba: Image.Image,
    canvas_w: int,
    canvas_h: int,
    padding_ratio: float = 0.00
) -> Image.Image:
    """
    Object-aware scale:
    - crop to object bbox
    - scale up to fit within canvas (minus padding)
    - center on transparent canvas
    - never crop the object
    """
    rgba = _pil_to_rgba(rgba)
    arr = np.array(rgba)
    alpha = arr[:, :, 3].astype(np.uint8)

    bbox = _bbox_from_alpha(alpha, thresh=8)
    if bbox is None:
        # Nothing detected; just return centered original on canvas
        out = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        x = (canvas_w - rgba.width) // 2
        y = (canvas_h - rgba.height) // 2
        out.paste(rgba, (x, y), rgba)
        return out

    x0, y0, x1, y1 = bbox
    cropped = rgba.crop((x0, y0, x1, y1))

    pad = _clamp(padding_ratio, 0.0, 0.40)
    target_w = int(canvas_w * (1.0 - 2.0 * pad))
    target_h = int(canvas_h * (1.0 - 2.0 * pad))

    # scale to fit (no cropping)
    scale = min(target_w / cropped.width, target_h / cropped.height)
    new_w = max(1, int(round(cropped.width * scale)))
    new_h = max(1, int(round(cropped.height * scale)))

    resized = cropped.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    out = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    out.paste(resized, (x, y), resized)
    return out


def _encode_lossless_webp(rgba: Image.Image) -> bytes:
    rgba = _pil_to_rgba(rgba)
    buf = io.BytesIO()
    rgba.save(buf, format="WEBP", lossless=True, method=6, exact=True)
    return buf.getvalue()


# ----------------------------
# Routes
# ----------------------------

@app.get("/health")
def health():
    return {"ok": True, "model": REMBG_MODEL}


@app.post("/process")
async def process(
    file: UploadFile = File(...),

    # naming inputs (from your sheet)
    mpn: Optional[str] = Form(None),
    sku: Optional[str] = Form(None),

    # output controls
    output: str = Form("webp"),  # "webp" only (kept for compatibility)
    canvas_w: int = Form(1400),
    canvas_h: int = Form(1700),  # 14:17 default
    padding_ratio: float = Form(0.00),

    # rembg controls
    alpha_matting: Optional[str] = Form("true"),
    alpha_matting_foreground_threshold: int = Form(245),
    alpha_matting_background_threshold: int = Form(10),
    alpha_matting_erode_size: int = Form(10),

    # edge refinement (these fix your jaggies + object clipping)
    edge_dilate_px: int = Form(2),
    edge_feather_px: int = Form(2),

    # improves quality on small images: upscale before segmentation (1 = off, 2 recommended)
    pre_upscale: int = Form(2),
):
    try:
        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "Empty upload"}, status_code=400)

        # load image
        img = Image.open(io.BytesIO(raw)).convert("RGBA")

        # optional pre-upscale for better masks on small images
        pre_upscale = int(_clamp(float(pre_upscale), 1, 4))
        if pre_upscale > 1:
            img = img.resize(
                (img.width * pre_upscale, img.height * pre_upscale),
                resample=Image.Resampling.LANCZOS
            )

        # Background removal via rembg
        use_matting = _parse_bool(alpha_matting, default=True)

        removed_bytes = remove(
            img,
            session=session,
            alpha_matting=use_matting,
            alpha_matting_foreground_threshold=int(alpha_matting_foreground_threshold),
            alpha_matting_background_threshold=int(alpha_matting_background_threshold),
            alpha_matting_erode_size=int(alpha_matting_erode_size),
            post_process_mask=True,  # important
        )

        cutout = Image.open(io.BytesIO(removed_bytes)).convert("RGBA")

        # refine edges (prevents “eaten” edges + reduces jaggies)
        cutout = _edge_refine_alpha(
            cutout,
            dilate_px=int(_clamp(float(edge_dilate_px), 0, 12)),
            feather_px=int(_clamp(float(edge_feather_px), 0, 12)),
        )

        # Fit onto 14:17 canvas as large as possible, no cropping
        canvas_w = int(_clamp(float(canvas_w), 64, 6000))
        canvas_h = int(_clamp(float(canvas_h), 64, 6000))
        padding_ratio = float(_clamp(float(padding_ratio), 0.0, 0.40))

        final_img = _fit_object_on_canvas(
            cutout,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            padding_ratio=padding_ratio
        )

        # Build filename: partlogic_MPN_SKU.webp
        mpn_s = _safe_token(mpn or "no-mpn")
        sku_s = _safe_token(sku or "no-sku")
        out_name = f"partlogic_{mpn_s}_{sku_s}.webp"

        out_bytes = _encode_lossless_webp(final_img)

        return Response(
            content=out_bytes,
            media_type="image/webp",
            headers={
                "Content-Disposition": f'inline; filename="{out_name}"',
                "X-Output-Filename": out_name,
            },
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
