from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from rembg import remove
from PIL import Image
import numpy as np
import cv2
import io

app = FastAPI()

def get_bbox(alpha):
    coords = np.argwhere(alpha > 0)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return (x0, y0, x1+1, y1+1)

def normalize(img_bgr, strength=0.6):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0 + strength*2, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    padding_ratio: float = Form(0.15),
    normalize_strength: float = Form(0.6),
):
    raw = await file.read()

    # remove background
    cut = remove(raw)
    rgba = Image.open(io.BytesIO(cut)).convert("RGBA")

    alpha = np.array(rgba.split()[-1])
    bbox = get_bbox(alpha)
    if not bbox:
        return Response(content=b"", media_type="image/png")

    cropped = rgba.crop(bbox)

    # convert to RGB on white bg for normalization
    rgb = Image.new("RGB", cropped.size, (255,255,255))
    rgb.paste(cropped, mask=cropped.split()[-1])

    bgr = np.array(rgb)[:,:,::-1]
    bgr = normalize(bgr, normalize_strength)
    rgb2 = Image.fromarray(bgr[:,:,::-1])

    # rebuild alpha
    alpha = cropped.split()[-1]
    rgba2 = Image.new("RGBA", cropped.size, (0,0,0,0))
    rgba2.paste(rgb2.convert("RGBA"), (0,0))
    rgba2.putalpha(alpha)

    # padding
    w, h = rgba2.size
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)
    canvas = Image.new("RGBA", (w+2*pad_w, h+2*pad_h), (0,0,0,0))
    canvas.paste(rgba2, (pad_w, pad_h), mask=rgba2)

    # white background final
    final = Image.new("RGB", canvas.size, (255,255,255))
    final.paste(canvas, mask=canvas.split()[-1])

    buf = io.BytesIO()
    final.save(buf, format="PNG", optimize=True)
    return Response(buf.getvalue(), media_type="image/png")
