import io

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="Image Processor")


class ProcessRequest(BaseModel):
    image_url: str
    filename: str | None = "processed.png"


def trim_transparent_edges(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = img.getchannel("A")
    bbox = alpha.getbbox()

    # If the image is fully transparent or bbox can't be found, return as-is
    if not bbox:
        return img

    return img.crop(bbox)


def fit_to_canvas(img: Image.Image, canvas_w: int = 1400, canvas_h: int = 1700) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Invalid source image dimensions")

    # Scale proportionally until one edge touches the canvas
    scale = min(canvas_w / src_w, canvas_h / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Transparent canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Center the resized image
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    canvas.paste(resized, (x, y), resized)

    return canvas


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process-image")
def process_image(payload: ProcessRequest):
    try:
        resp = requests.get(payload.image_url, timeout=60)
        resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")

        # Remove transparent outer padding first
        img = trim_transparent_edges(img)

        # Fit onto 1400 x 1700 transparent canvas
        final_img = fit_to_canvas(img, 1400, 1700)

        output = io.BytesIO()
        final_img.save(output, format="PNG")
        output.seek(0)

        filename = payload.filename or "processed.png"
        if not filename.lower().endswith(".png"):
            filename += ".png"

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }

        return StreamingResponse(
            output,
            media_type="image/png",
            headers=headers,
        )

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
