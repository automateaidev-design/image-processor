import io
import os

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageFile

# Prevent thread oversubscription in image libraries
os.environ["OMP_NUM_THREADS"] = "1"

# Pillow stability settings
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

app = FastAPI(title="Image Processor")

CANVAS_WIDTH = 1400
CANVAS_HEIGHT = 1700

# High-quality WebP settings
WEBP_QUALITY = 95
WEBP_METHOD = 6

# Reuse HTTP connections across requests
session = requests.Session()


class ProcessRequest(BaseModel):
    image_url: str
    filename: str = "processed.webp"


def sanitize_filename(filename: str) -> str:
    filename = (filename or "processed").split("/")[-1].split("\\")[-1].strip()
    if not filename:
        filename = "processed"
    if "." in filename:
        filename = filename.rsplit(".", 1)[0]
    return f"{filename}.webp"


def trim_transparent_edges(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = img.getchannel("A")
    bbox = alpha.getbbox()

    if not bbox:
        return img

    return img.crop(bbox)


def fit_to_canvas(img: Image.Image) -> Image.Image:
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Invalid image dimensions")

    # Keep proportions intact and enlarge until one edge touches the canvas
    scale = min(CANVAS_WIDTH / src_w, CANVAS_HEIGHT / src_h)

    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))

    x = (CANVAS_WIDTH - new_w) // 2
    y = (CANVAS_HEIGHT - new_h) // 2

    canvas.paste(resized, (x, y), resized)
    return canvas


def normalise_input_url(image_url: str) -> str:
    image_url = (image_url or "").strip()

    # n8n sometimes accidentally sends a leading "="
    if image_url.startswith("="):
        image_url = image_url[1:].strip()

    if not image_url:
        raise ValueError("image_url is empty")

    return image_url


@app.get("/")
def root():
    return {"message": "app is live"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process-image")
def process_image(payload: ProcessRequest):
    try:
        image_url = normalise_input_url(payload.image_url)

        resp = session.get(
            image_url,
            timeout=60,
            allow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": image_url,
            },
        )
        resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")

        # Remove transparent borders first
        img = trim_transparent_edges(img)

        # Resize proportionally and center on transparent 1400x1700 canvas
        final_img = fit_to_canvas(img)

        output = io.BytesIO()
        final_img.save(
            output,
            format="WEBP",
            quality=WEBP_QUALITY,
            method=WEBP_METHOD,
        )
        output.seek(0)

        filename = sanitize_filename(payload.filename)

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }

        return StreamingResponse(
            output,
            media_type="image/webp",
            headers=headers,
        )

    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 400
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image: HTTP {status_code} for url: {image_url}",
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
