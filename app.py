import io
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI(title="Image Processor")

CANVAS_WIDTH = 1400
CANVAS_HEIGHT = 1700

# High quality WebP settings
WEBP_QUALITY = 95
WEBP_METHOD = 6


class ProcessRequest(BaseModel):
    image_url: str
    filename: str = "processed.webp"


def sanitize_filename(filename: str) -> str:
    filename = filename.split("/")[-1]
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

    scale = min(CANVAS_WIDTH / src_w, CANVAS_HEIGHT / src_h)

    new_w = round(src_w * scale)
    new_h = round(src_h * scale)

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))

    x = (CANVAS_WIDTH - new_w) // 2
    y = (CANVAS_HEIGHT - new_h) // 2

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

        # remove transparent borders
        img = trim_transparent_edges(img)

        # scale + center
        final_img = fit_to_canvas(img)

        output = io.BytesIO()

        final_img.save(
            output,
            format="WEBP",
            quality=WEBP_QUALITY,
            method=WEBP_METHOD
        )

        output.seek(0)

        filename = sanitize_filename(payload.filename)

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }

        return StreamingResponse(
            output,
            media_type="image/webp",
            headers=headers
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
