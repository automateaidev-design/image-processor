from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from rembg import remove, new_session
from PIL import Image
import io

app = FastAPI(title="image-processor")

# Use a strong general model. u2net is a good default.
# If you later want to test: "u2netp" (faster, sometimes worse) or "isnet-general-use" (often better edges).
SESSION = new_session("u2net")


def _load_image_from_upload(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        img = Image.open(io.BytesIO(data))
        # Normalize to RGBA workflow, preserve alpha if any
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    # Lossless, max quality, no resizing
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _to_jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    # JPEG needs RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    # If your n8n node sends form fields, these are handy
    return_mask: bool = Form(False),
    return_preview_jpeg: bool = Form(False),

    # Edge quality controls (defaults are tuned for product photos)
    alpha_matting: bool = Form(True),
    alpha_matting_foreground_threshold: int = Form(245),
    alpha_matting_background_threshold: int = Form(10),
    alpha_matting_erode_size: int = Form(10),

    # Preview background (only used if return_preview_jpeg=True)
    preview_bg: str = Form("white"),  # "white" or "black"
    preview_jpeg_quality: int = Form(95),
):
    """
    Returns:
      - Default: PNG with transparent background (lossless)
      - Optional: mask PNG
      - Optional: JPEG preview composited over white/black background
    """

    # Read upload
    img = _load_image_from_upload(file)
    in_bytes = _to_png_bytes(img)  # normalize input as PNG bytes for rembg

    # Rembg settings
    kwargs = {}
    if alpha_matting:
        kwargs.update(
            dict(
                alpha_matting=True,
                alpha_matting_foreground_threshold=int(alpha_matting_foreground_threshold),
                alpha_matting_background_threshold=int(alpha_matting_background_threshold),
                alpha_matting_erode_size=int(alpha_matting_erode_size),
            )
        )

    # Perform background removal
    try:
        out_bytes = remove(in_bytes, session=SESSION, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {e}")

    # Parse result (PNG with alpha)
    cutout = Image.open(io.BytesIO(out_bytes)).convert("RGBA")

    # If only cutout requested, return it as image/png
    if not return_mask and not return_preview_jpeg:
        return Response(content=_to_png_bytes(cutout), media_type="image/png")

    # Build multipart-ish JSON response with base64 is possible,
    # but simplest for n8n is returning files as separate endpoints.
    # So we return a JSON with binary bytes in separate fields is not ideal in FastAPI.
    # Instead: return a single PNG by default, OR let the client call /process?mode=...
    #
    # If you want both images in one response for n8n, easiest is to return a zip.

    # Create a ZIP containing requested outputs (drop-in friendly for n8n)
    import zipfile

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("cutout.png", _to_png_bytes(cutout))

        if return_mask:
            # mask = alpha channel
            alpha = cutout.split()[-1]
            mask = Image.merge("RGBA", (alpha, alpha, alpha, Image.new("L", alpha.size, 255)))
            z.writestr("mask.png", _to_png_bytes(mask))

        if return_preview_jpeg:
            bg = (255, 255, 255) if preview_bg.lower() == "white" else (0, 0, 0)
            background = Image.new("RGB", cutout.size, bg)
            composed = Image.alpha_composite(background.convert("RGBA"), cutout).convert("RGB")
            z.writestr("preview.jpg", _to_jpeg_bytes(composed, quality=int(preview_jpeg_quality)))

    return Response(content=zbuf.getvalue(), media_type="application/zip")
