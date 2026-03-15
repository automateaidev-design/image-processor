import io
import os
import gc
import asyncio
from typing import Optional, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from PIL import Image
from rembg import remove, new_session

print(f"Starting app with PORT={os.getenv('PORT')}", flush=True)

try:
    cv2.setNumThreads(1)
except Exception:
    pass

PIL_MAX_IMAGE_PIXELS = int(os.getenv("PIL_MAX_IMAGE_PIXELS", "60000000"))
Image.MAX_IMAGE_PIXELS = PIL_MAX_IMAGE_PIXELS

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
QUEUE_TIMEOUT_S = float(os.getenv("QUEUE_TIMEOUT_S", "30"))
TARGET_W = int(os.getenv("TARGET_W", "1400"))
TARGET_H = int(os.getenv("TARGET_H", "1700"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "3000"))
RMBG_MODEL = os.getenv("RMBG_MODEL", "isnet-general-use")

app = FastAPI(title="image-processor", version="7.0.0")
SEM = asyncio.Semaphore(max(1, MAX_CONCURRENCY))
SESSION = None

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"incoming request: {request.method} {request.url.path}", flush=True)
    response = await call_next(request)
    print(f"completed request: {request.method} {request.url.path} -> {response.status_code}", flush=True)
    return response

@app.get("/")
def root():
    return {"ok": True, "message": "service alive"}

@app.get("/health")
def health():
    return {"ok": True, "port": os.getenv("PORT"), "model": RMBG_MODEL}

def get_session():
    global SESSION
    if SESSION is None:
        print(f"loading rembg model: {RMBG_MODEL}", flush=True)
        SESSION = new_session(RMBG_MODEL)
        print("rembg model loaded", flush=True)
    return SESSION

def pil_open_rgb(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def resize_if_huge(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_DIM:
        return img
    scale = MAX_IMAGE_DIM / float(max(w, h))
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    output: str = Form("png"),
):
    try:
        await asyncio.wait_for(SEM.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "Service busy"}, status_code=503)

    try:
        print("process start", flush=True)
        raw = await file.read()
        print(f"upload bytes: {len(raw) if raw else 0}", flush=True)

        if not raw:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": f"File too large. Max {MAX_UPLOAD_MB}MB"}, status_code=413)

        pil_img = pil_open_rgb(raw)
        pil_img = resize_if_huge(pil_img)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        png_in = buf.getvalue()

        print("running rembg", flush=True)
        out = remove(png_in, session=get_session())
        print(f"rembg output bytes: {len(out)}", flush=True)

        return Response(content=out, media_type="image/png")

    except Exception as e:
        print(f"process error: {repr(e)}", flush=True)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            SEM.release()
        except Exception:
            pass
        gc.collect()
