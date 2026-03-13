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

# ----------------------------
# Stability / performance knobs
# ----------------------------

try:
    cv2.setNumThreads(1)
except Exception:
    pass

PIL_MAX_IMAGE_PIXELS = int(os.getenv("PIL_MAX_IMAGE_PIXELS", "60000000"))
Image.MAX_IMAGE_PIXELS = PIL_MAX_IMAGE_PIXELS

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
QUEUE_TIMEOUT_S = float(os.getenv("QUEUE_TIMEOUT_S", "15"))

ENABLE_SELF_RESTART = os.getenv("ENABLE_SELF_RESTART", "0") == "1"
SELF_RESTART_SECONDS = int(os.getenv("SELF_RESTART_SECONDS", "600"))

PROCESS_TIMEOUT_S = int(os.getenv("PROCESS_TIMEOUT_S", "240"))

# ----------------------------
# App + model session
# ----------------------------

app = FastAPI(title="image-processor", version="2.3.0")

REMBG_MODEL = os.getenv("REMBG_MODEL", "u2net")
_session = None

TARGET_W = int(os.getenv("TARGET_W", "1400"))
TARGET_H = int(os.getenv("TARGET_H", "1700"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

SEM = asyncio.Semaphore(max(1, MAX_CONCURRENCY))


@app.on_event("startup")
async def _startup():
    global _session
    _session = new_session(REMBG_MODEL)

    if ENABLE_SELF_RESTART:
        async def _restarter():
            await asyncio.sleep(max(60, SELF_RESTART_SECONDS))
            os._exit(0)

        asyncio.create_task(_restarter())


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def pil_open_rgb(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def pre_upscale_if_small(img: Image.Image, min_max_dim: int = 900) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m >= min_max_dim:
        return img
    scale = min_max_dim / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def ensure_rgba(png_bytes: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(png_bytes))
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im


def dilate(alpha: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return alpha
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(alpha, kernel, iterations=1)


def erode(alpha: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return alpha
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(alpha, kernel, iterations=1)


def feather(alpha: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return alpha
    k = 2 * px + 1
    return cv2.GaussianBlur(alpha, (k, k), 0)


def alpha_bbox(alpha: np.ndarray, thresh: int = 8) -> Optional[Tuple[int, int, int, int]]:
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
    pad = clamp_float(padding_ratio, 0.0, 0.49)

    alpha = rgba[:, :, 3]
    bbox = alpha_bbox(alpha, thresh=alpha_thresh)
    if bbox is None:
        return np.zeros((target_h, target_w, 4), dtype=np.uint8)

    x1, y1, x2, y2 = bbox
    obj = rgba[y1:y2, x1:x2, :]

    obj_h, obj_w = obj.shape[:2]
    inner_w = int(round(target_w * (1.0 - 2.0 * pad)))
    inner_h = int(round(target_h * (1.0 - 2.0 * pad)))
    inner_w = max(1, inner_w)
    inner_h = max(1, inner_h)

    scale = min(inner_w / obj_w, inner_h / obj_h)
    new_w = max(1, int(round(obj_w * scale)))
    new_h = max(1, int(round(obj_h * scale)))

    obj_resized = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    out = np.zeros((target_h, target_w, 4), dtype=np.uint8)

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2

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
        s = s.replace(" ", "_")
        s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))
        return s[:140] if s else "na"

    return f"{clean(prefix)}_{clean(mpn)}_{clean(sku)}.{ext}"


def save_lossless_webp(rgba_arr: np.ndarray) -> bytes:
    img = Image.fromarray(rgba_arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="WEBP", lossless=True, quality=100, method=6)
    return buf.getvalue()


def save_png(rgba_arr: np.ndarray) -> bytes:
    img = Image.fromarray(rgba_arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def decontaminate_edge_rgb(rgba: np.ndarray, alpha_max: int = 200, inpaint_radius: int = 3) -> np.ndarray:
    alpha = rgba[:, :, 3]
    mask = ((alpha > 0) & (alpha < alpha_max)).astype(np.uint8) * 255
    if mask.max() == 0:
        return rgba

    rgb = rgba[:, :, :3]
    rgb_fixed = cv2.inpaint(rgb, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    out = rgba.copy()
    out[:, :, :3] = rgb_fixed
    return out


def rgba_from_rgb_alpha(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    out = np.dstack([rgb, alpha]).astype(np.uint8)
    return out


def largest_component(alpha: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    mask = (alpha > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return alpha

    h, w = alpha.shape[:2]
    min_area = max(1, int(h * w * min_area_ratio))

    areas = stats[:, cv2.CC_STAT_AREA]
    best_label = 0
    best_area = 0
    for i in range(1, num_labels):
        area = int(areas[i])
        if area >= min_area and area > best_area:
            best_area = area
            best_label = i

    if best_label == 0:
        return alpha

    keep = (labels == best_label).astype(np.uint8) * 255
    out = alpha.copy()
    out[keep == 0] = 0
    return out


def estimate_corner_bg_bgr(img_bgr: np.ndarray, patch_frac: float = 0.08) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    ph = max(8, int(h * patch_frac))
    pw = max(8, int(w * patch_frac))

    patches = [
        img_bgr[0:ph, 0:pw],
        img_bgr[0:ph, w - pw:w],
        img_bgr[h - ph:h, 0:pw],
        img_bgr[h - ph:h, w - pw:w],
    ]
    pixels = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0).astype(np.float32)
    mean = pixels.mean(axis=0)
    std = float(np.mean(pixels.std(axis=0)))
    return mean, std


def uniform_background_mask(
    rgb: np.ndarray,
    bg_tolerance: int = 26,
    patch_frac: float = 0.08,
    corner_uniformity_max_std: float = 22.0,
    edge_margin_frac: float = 0.02,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Returns alpha mask for images with a mostly-uniform background based on corner sampling.
    Foreground = pixels sufficiently different from estimated corner background.
    """
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    bg_mean_bgr, bg_std = estimate_corner_bg_bgr(img_bgr, patch_frac=patch_frac)
    info = {
        "bg_std": float(bg_std),
        "mode": "uniform-bg",
        "bg_mean_bgr": [float(x) for x in bg_mean_bgr],
    }

    if bg_std > corner_uniformity_max_std:
        return None, info

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(np.uint8([[bg_mean_bgr]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    dist = np.sqrt(np.sum((lab - bg_lab) ** 2, axis=2))
    fg_mask = (dist >= float(bg_tolerance)).astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    fg_mask = largest_component(fg_mask, min_area_ratio=0.002)

    bbox = alpha_bbox(fg_mask, thresh=8)
    if bbox is None:
        return None, info

    x1, y1, x2, y2 = bbox
    box_area = max(1, (x2 - x1) * (y2 - y1))
    fill_ratio = float(np.count_nonzero(fg_mask)) / float(h * w)

    touches_left = x1 <= int(w * edge_margin_frac)
    touches_right = x2 >= w - int(w * edge_margin_frac)
    touches_top = y1 <= int(h * edge_margin_frac)
    touches_bottom = y2 >= h - int(h * edge_margin_frac)
    touches_edges = sum([touches_left, touches_right, touches_top, touches_bottom])

    info.update({
        "fill_ratio": fill_ratio,
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "touches_edges": touches_edges,
        "bbox_area_ratio": box_area / float(h * w),
    })

    if fill_ratio < 0.01:
        return None, info

    return fg_mask, info


def rembg_cutout(
    rgb_img: Image.Image,
    alpha_matting: bool,
    alpha_matting_foreground_threshold: int,
    alpha_matting_background_threshold: int,
    alpha_matting_erode_size: int,
) -> np.ndarray:
    tmp = io.BytesIO()
    rgb_img.save(tmp, format="PNG")
    png_in = tmp.getvalue()

    cutout_png = remove(
        png_in,
        session=_session,
        alpha_matting=bool(alpha_matting),
        alpha_matting_foreground_threshold=clamp_int(alpha_matting_foreground_threshold, 0, 255),
        alpha_matting_background_threshold=clamp_int(alpha_matting_background_threshold, 0, 255),
        alpha_matting_erode_size=clamp_int(alpha_matting_erode_size, 0, 50),
    )
    rgba_pil = ensure_rgba(cutout_png)
    return np.array(rgba_pil)


def apply_edge_postprocess(
    rgba: np.ndarray,
    edge_erode_px: int,
    edge_dilate_px: int,
    edge_feather_px: int,
    decontaminate: bool,
    decontaminate_alpha_max: int,
    decontaminate_inpaint_radius: int,
) -> np.ndarray:
    a = rgba[:, :, 3]
    a = erode(a, clamp_int(edge_erode_px, 0, 30))
    a = dilate(a, clamp_int(edge_dilate_px, 0, 30))
    a = feather(a, clamp_int(edge_feather_px, 0, 30))
    rgba[:, :, 3] = a

    if decontaminate:
        rgba = decontaminate_edge_rgb(
            rgba,
            alpha_max=clamp_int(decontaminate_alpha_max, 1, 254),
            inpaint_radius=clamp_int(decontaminate_inpaint_radius, 1, 12),
        )
    return rgba


def pick_best_cutout(
    rgb: np.ndarray,
    uniform_rgba: Optional[np.ndarray],
    rembg_rgba: Optional[np.ndarray],
    prefer_uniform_bg: bool = True,
) -> Tuple[np.ndarray, str]:
    """
    Hybrid chooser:
    - If a strong uniform-background mask exists, prefer it for catalogue/product images.
    - Otherwise use rembg.
    """
    if uniform_rgba is not None and prefer_uniform_bg:
        ua = uniform_rgba[:, :, 3]
        ub = alpha_bbox(ua, thresh=8)
        if ub is not None:
            return uniform_rgba, "uniform-bg"

    if rembg_rgba is not None:
        return rembg_rgba, "rembg"

    if uniform_rgba is not None:
        return uniform_rgba, "uniform-bg"

    h, w = rgb.shape[:2]
    return np.zeros((h, w, 4), dtype=np.uint8), "empty"


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": REMBG_MODEL,
        "max_concurrency": MAX_CONCURRENCY,
        "process_timeout_s": PROCESS_TIMEOUT_S,
    }


@app.get("/ready")
def ready():
    if SEM.locked():
        return JSONResponse({"ready": False, "reason": "busy"}, status_code=503)
    return {"ready": True}


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),

    mpn: str = Form(""),
    sku: str = Form(""),
    prefix: str = Form("partlogic"),

    output: str = Form("webp"),  # webp | png

    # Strategy
    segmentation_mode: str = Form("auto"),  # auto | rembg | uniform_bg
    prefer_uniform_bg: bool = Form(True),

    # Uniform background controls
    uniform_bg_tolerance: int = Form(26),
    uniform_bg_corner_patch_frac: float = Form(0.08),
    uniform_bg_max_corner_std: float = Form(22.0),

    # Rembg controls
    alpha_matting: bool = Form(False),
    alpha_matting_foreground_threshold: int = Form(250),
    alpha_matting_background_threshold: int = Form(15),
    alpha_matting_erode_size: int = Form(2),

    # Edge controls
    edge_erode_px: int = Form(1),
    edge_dilate_px: int = Form(0),
    edge_feather_px: int = Form(0),

    # Decontaminate edge RGB
    decontaminate: bool = Form(True),
    decontaminate_alpha_max: int = Form(200),
    decontaminate_inpaint_radius: int = Form(8),

    # Canvas controls
    padding_ratio: float = Form(0.0),
    target_w: int = Form(TARGET_W),
    target_h: int = Form(TARGET_H),

    # Quality/perf
    pre_upscale: bool = Form(True),
    pre_upscale_min_dim: int = Form(900),

    # Response extras
    debug_headers: bool = Form(False),
):
    try:
        await asyncio.wait_for(SEM.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": "Service busy", "hint": "Reduce n8n parallelism or increase MAX_CONCURRENCY"},
            status_code=503,
        )

    raw = None
    pil_img = None
    rgb = None
    uniform_alpha = None
    uniform_rgba = None
    rembg_rgba = None
    chosen_rgba = None
    fitted = None

    try:
        if _session is None:
            return JSONResponse({"error": "Model session not ready"}, status_code=503)

        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": f"File too large. Max {MAX_UPLOAD_MB}MB"}, status_code=413)

        pil_img = pil_open_rgb(raw)

        w, h = pil_img.size
        max_dim = int(os.getenv("MAX_IMAGE_DIM", "8000"))
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        if pre_upscale:
            pil_img = pre_upscale_if_small(pil_img, min_max_dim=clamp_int(pre_upscale_min_dim, 256, 2400))

        rgb = np.array(pil_img.convert("RGB"))

        mode = (segmentation_mode or "auto").strip().lower()
        chosen_mode = "unknown"

        async def _do_process():
            nonlocal uniform_alpha, uniform_rgba, rembg_rgba, chosen_rgba, chosen_mode

            if mode in ("auto", "uniform_bg"):
                uniform_alpha_local, _ = uniform_background_mask(
                    rgb,
                    bg_tolerance=clamp_int(uniform_bg_tolerance, 1, 120),
                    patch_frac=clamp_float(uniform_bg_corner_patch_frac, 0.02, 0.2),
                    corner_uniformity_max_std=clamp_float(uniform_bg_max_corner_std, 1.0, 80.0),
                )
                if uniform_alpha_local is not None:
                    uniform_alpha = uniform_alpha_local
                    uniform_rgba = rgba_from_rgb_alpha(rgb, uniform_alpha)

            if mode in ("auto", "rembg"):
                rembg_rgba = rembg_cutout(
                    pil_img,
                    alpha_matting=bool(alpha_matting),
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size,
                )

            chosen_rgba_local, chosen_mode_local = pick_best_cutout(
                rgb=rgb,
                uniform_rgba=uniform_rgba,
                rembg_rgba=rembg_rgba,
                prefer_uniform_bg=bool(prefer_uniform_bg),
            )
            chosen_rgba = chosen_rgba_local
            chosen_mode = chosen_mode_local

        await asyncio.wait_for(_do_process(), timeout=PROCESS_TIMEOUT_S)

        chosen_rgba = apply_edge_postprocess(
            chosen_rgba,
            edge_erode_px=edge_erode_px,
            edge_dilate_px=edge_dilate_px,
            edge_feather_px=edge_feather_px,
            decontaminate=decontaminate,
            decontaminate_alpha_max=decontaminate_alpha_max,
            decontaminate_inpaint_radius=decontaminate_inpaint_radius,
        )

        tw = clamp_int(target_w, 200, 6000)
        th = clamp_int(target_h, 200, 8000)
        fitted = object_aware_fit(
            chosen_rgba,
            target_w=tw,
            target_h=th,
            padding_ratio=float(padding_ratio),
            alpha_thresh=8,
        )

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

        headers = {
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "no-store",
        }

        if debug_headers:
            headers["X-Segmentation-Mode"] = chosen_mode

        return Response(
            content=out_bytes,
            media_type=media_type,
            headers=headers,
        )

    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": "processing timeout", "hint": "Try segmentation_mode=uniform_bg or disable alpha_matting"},
            status_code=504,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        try:
            SEM.release()
        except Exception:
            pass

        try:
            del raw
            del pil_img
            del rgb
            del uniform_alpha
            del uniform_rgba
            del rembg_rgba
            del chosen_rgba
            del fitted
        except Exception:
            pass

        gc.collect()
