"""
pattern_engine.py - موتور پردازش طرح (بدون رابط گرافیکی)
همه‌ی تنظیمات در کلاس Settings است و از رابط گرافیکی قابل تغییر است.
"""
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ai_providers import AIProvider

CM_PER_INCH = 2.54
FORMATS = ("tiff", "png", "jpeg", "pdf", "svg")


def to_cm(value, unit):
    return value * CM_PER_INCH if unit == "inch" else value


@dataclass
class Settings:
    unit: str = "cm"                # cm | inch
    width: float = 160.0
    height: float = 230.0
    dpi: int = 150                  # 72 تا 1200
    bleed: float = 0.0              # در همان واحد unit
    out_format: str = "tiff"        # tiff | png | jpeg | pdf | svg
    color_mode: str = "RGB"         # RGB | CMYK
    icc_path: str = ""              # پروفایل ICC پرینتر (اختیاری، برای CMYK)
    jpeg_quality: int = 95          # 60 تا 100
    tiff_compression: str = "lzw"   # none | lzw | zip
    normalize_light: bool = True    # حذف سایه‌ی کم‌فرکانس
    repeat_mode: str = "auto"       # auto | off
    min_period_px: int = 10         # کوچک‌ترین تناوب مجاز (پیکسل)
    min_strength: float = 0.15      # آستانه‌ی تشخیص تکرار (۰ تا ۱)
    denoise_k: int = 1              # قدرت حذف چین: ۰ تا ۳
    upscale_mode: str = "lanczos"   # lanczos | ai
    ai_scale: int = 2               # 2 | 4

    def width_cm(self):
        return to_cm(self.width, self.unit)

    def height_cm(self):
        return to_cm(self.height, self.unit)

    def bleed_cm(self):
        return to_cm(self.bleed, self.unit)

    def validate(self):
        checks = [
            (self.unit in ("cm", "inch"), "واحد نامعتبر است"),
            (self.width > 0 and self.height > 0, "ابعاد باید بزرگ‌تر از صفر باشد"),
            (self.bleed >= 0, "Bleed نمی‌تواند منفی باشد"),
            (72 <= self.dpi <= 1200, "DPI باید بین ۷۲ و ۱۲۰۰ باشد"),
            (self.out_format in FORMATS, "فرمت خروجی نامعتبر است"),
            (self.color_mode in ("RGB", "CMYK"), "حالت رنگ نامعتبر است"),
            (not (self.color_mode == "CMYK" and self.out_format in ("png", "svg")),
             "PNG و SVG فقط RGB هستند؛ برای CMYK از TIFF، JPEG یا PDF استفاده کنید"),
            (60 <= self.jpeg_quality <= 100, "کیفیت JPEG باید بین ۶۰ و ۱۰۰ باشد"),
            (self.tiff_compression in ("none", "lzw", "zip"), "فشرده‌سازی TIFF نامعتبر است"),
            (self.repeat_mode in ("auto", "off"), "حالت تکرار نامعتبر است"),
            (0 <= self.denoise_k <= 3, "قدرت حذف چین باید بین ۰ و ۳ باشد"),
            (0 <= self.min_strength <= 1, "آستانه‌ی تکرار باید بین ۰ و ۱ باشد"),
            (self.upscale_mode in ("lanczos", "ai"), "حالت بزرگ‌نمایی نامعتبر است"),
            (self.ai_scale in (2, 4), "مقیاس AI باید ۲ یا ۴ باشد"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)


# ---------- ۱. گوشه‌ها ----------

def order_corners(pts):
    """ترتیب: بالا-چپ، بالا-راست، پایین-راست، پایین-چپ."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def detect_quad(img):
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.2 * h * w:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
            best, best_area = approx.reshape(4, 2), area
    return order_corners(best) if best is not None else None


# ---------- ۲. تصحیح پرسپکتیو ----------

def rectify(img, quad, w_cm, h_cm):
    tl, tr, br, bl = quad
    w_px = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h_px = int(round(w_px * h_cm / w_cm))
    dst = np.float32([[0, 0], [w_px - 1, 0], [w_px - 1, h_px - 1], [0, h_px - 1]])
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (w_px, h_px), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


# ---------- ۳. یکنواخت کردن نور ----------

def normalize_light(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    h, w = L.shape
    bg = cv2.GaussianBlur(L, (0, 0), max(h, w) / 4.0) + 1.0
    lab[:, :, 0] = np.clip(L / bg * np.mean(bg), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------- ۴. تشخیص تکرار ----------

def _region(gray):
    H, W = gray.shape
    s = min(300, min(H, W) // 3)
    return H // 2 - s // 2, W // 2 - s // 2, s


def refine_vector(gray, v, r=6):
    """جستجوی ±r پیکسل و سپس دقت زیرپیکسل با درون‌یابی سهموی."""
    y0, x0, s = _region(gray)
    H, W = gray.shape
    a = gray[y0:y0 + s, x0:x0 + s].astype(np.float32)

    def cost(dx, dy):
        yb, xb = y0 + dy, x0 + dx
        if yb < 0 or xb < 0 or yb + s > H or xb + s > W:
            return np.inf
        b = gray[yb:yb + s, xb:xb + s].astype(np.float32)
        return float(np.mean((a - b) ** 2))

    vx, vy = int(round(v[0])), int(round(v[1]))
    c0, bx, by = np.inf, vx, vy
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            c = cost(vx + du, vy + dv)
            if c < c0:
                c0, bx, by = c, vx + du, vy + dv

    def offset(cm, cp):
        den = cm - 2 * c0 + cp
        if not np.isfinite(den) or abs(den) < 1e-9:
            return 0.0
        return float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))

    ox = offset(cost(bx - 1, by), cost(bx + 1, by))
    oy = offset(cost(bx, by - 1), cost(bx, by + 1))
    return np.array([bx + ox, by + oy], dtype=np.float64)


def find_lattice(img, min_period, min_strength):
    """دو بردار تناوب غیرهم‌راستا، یا None اگر طرح تکرار نداشته باشد."""
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_full = gray_u8.astype(np.float32)
    sc = min(1.0, 1200.0 / max(g_full.shape))
    g = cv2.resize(g_full, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else g_full
    g = g - g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)

    F = np.fft.fft2(g * win)
    ac = np.real(np.fft.ifft2(F * np.conj(F)))
    ac = np.fft.fftshift(ac / (ac.max() + 1e-9)).astype(np.float32)

    h, w = ac.shape
    cy, cx = h // 2, w // 2
    mp = max(3.0, min_period * sc)
    local_max = ac == cv2.dilate(ac, np.ones((7, 7), np.uint8))
    ys, xs = np.nonzero(local_max & (ac > min_strength))

    cands = []
    for y, x in zip(ys, xs):
        dx_s, dy_s = x - cx, y - cy
        if math.hypot(dx_s, dy_s) < mp:
            continue
        v = np.array([round(dx_s / sc), round(dy_s / sc)], dtype=np.int64)
        cands.append((float(np.linalg.norm(v)), v))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])

    v1 = cands[0][1]
    v2 = None
    for _, v in cands[1:]:
        cross = abs(int(v1[0]) * int(v[1]) - int(v1[1]) * int(v[0]))
        if cross / (np.linalg.norm(v1) * np.linalg.norm(v) + 1e-9) > 0.4:
            v2 = v
            break
    if v2 is None:
        return None
    return refine_vector(gray_u8, v1), refine_vector(gray_u8, v2)


# ---------- ۵. حذف چین با مدین روی شبکه ----------

def lattice_median(img, v1, v2, K, block=256):
    if K <= 0:
        return img
    H, W = img.shape[:2]
    shifts = [k * v1 + m * v2 for k in range(-K, K + 1) for m in range(-K, K + 1)]
    gx = np.arange(W, dtype=np.float32)[None, :]
    out = np.empty_like(img)
    for r0 in range(0, H, block):
        r1 = min(H, r0 + block)
        gy = np.arange(r0, r1, dtype=np.float32)[:, None]
        stack = []
        for s in shifts:
            mx = np.broadcast_to(gx + np.float32(s[0]), (r1 - r0, W)).astype(np.float32)
            my = np.broadcast_to(gy + np.float32(s[1]), (r1 - r0, W)).astype(np.float32)
            stack.append(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
        out[r0:r1] = np.median(np.stack(stack), axis=0).astype(img.dtype)
    return out


# ---------- ۶. بزرگ‌نمایی با API ----------

def tiled_ai_upscale(img, scale, provider, log=print, tile=512, pad=48):
    """تصویر را به قطعات با حاشیه‌ی اضافه تقسیم می‌کند تا API محدودیت حجم نداشته باشد."""
    H, W = img.shape[:2]
    out = np.empty((H * scale, W * scale, 3), np.uint8)
    ny, nx = -(-H // tile), -(-W // tile)
    n = 0
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            n += 1
            log(f"بزرگ‌نمایی AI: قطعه {n} از {ny * nx}")
            ch, cw = min(tile, H - y), min(tile, W - x)
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(H, y + ch + pad), min(W, x + cw + pad)
            up = provider.upscale(img[y0:y1, x0:x1], scale)
            oy, ox = (y - y0) * scale, (x - x0) * scale
            out[y * scale:(y + ch) * scale, x * scale:(x + cw) * scale] = \
                up[oy:oy + ch * scale, ox:ox + cw * scale]
    return out


# ---------- ۷. رندر دوره‌ای بدون درز ----------

def render_periodic(cell, v1, v2, out_w, out_h, margin=0.0, block=256):
    H, W = cell.shape[:2]
    B = np.column_stack([v1, v2]).astype(np.float64)
    Binv = np.linalg.inv(B)
    A = np.array([W / 2.0, H / 2.0]) - B @ np.array([0.5, 0.5])

    xs = -margin + (np.arange(out_w) + 0.5) * (W + 2 * margin) / out_w
    out = np.empty((out_h, out_w, cell.shape[2]), dtype=np.uint8)
    for r0 in range(0, out_h, block):
        r1 = min(out_h, r0 + block)
        ys = -margin + (np.arange(r0, r1) + 0.5) * (H + 2 * margin) / out_h
        X, Y = np.meshgrid(xs, ys)
        P = np.stack([X.ravel(), Y.ravel()]) - A[:, None]
        C = Binv @ P
        C -= np.floor(C)
        Q = A[:, None] + B @ C
        mx = Q[0].reshape(r1 - r0, out_w).astype(np.float32)
        my = Q[1].reshape(r1 - r0, out_w).astype(np.float32)
        out[r0:r1] = cv2.remap(cell, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return out


# ---------- ۸. پایپ‌لاین کامل ----------

def run_pipeline(img, corners, s, provider=None, log=print):
    """
    img: تصویر BGR
    corners: چهار گوشه (ترتیب بالا-چپ، بالا-راست، پایین-راست، پایین-چپ) یا None برای تشخیص خودکار
    s: Settings
    provider: AIProvider (فقط وقتی upscale_mode == "ai")
    """
    s.validate()
    if s.upscale_mode == "ai" and provider is None:
        raise ValueError("برای بزرگ‌نمایی AI باید تنظیمات سرویس را کامل کنید")

    quad = order_corners(corners) if corners is not None else detect_quad(img)
    if quad is None:
        raise ValueError("گوشه‌های محصول پیدا نشد؛ گوشه‌ها را دستی تنظیم کنید")

    w_cm, h_cm, bleed_cm = s.width_cm(), s.height_cm(), s.bleed_cm()
    trim_w = round(w_cm / CM_PER_INCH * s.dpi)
    trim_h = round(h_cm / CM_PER_INCH * s.dpi)
    bleed_px = round(bleed_cm / CM_PER_INCH * s.dpi)
    out_w, out_h = trim_w + 2 * bleed_px, trim_h + 2 * bleed_px

    flat = rectify(img, quad, w_cm, h_cm)
    if s.normalize_light:
        flat = normalize_light(flat)
    H, W = flat.shape[:2]

    lat = find_lattice(flat, s.min_period_px, s.min_strength) if s.repeat_mode == "auto" else None

    if lat is None:
        log("تکرار طرح استفاده نشد؛ فقط بزرگ‌نمایی انجام می‌شود و جزئیات جدید ساخته نمی‌شود")
        src = tiled_ai_upscale(flat, s.ai_scale, provider, log) if s.upscale_mode == "ai" else flat
        body = cv2.resize(src, (trim_w, trim_h), interpolation=cv2.INTER_LANCZOS4)
        if bleed_px:
            body = cv2.copyMakeBorder(body, bleed_px, bleed_px, bleed_px, bleed_px,
                                      cv2.BORDER_REFLECT)
        return body

    v1, v2 = lat
    log(f"شبکه‌ی تکرار: v1=({v1[0]:.2f}, {v1[1]:.2f})  v2=({v2[0]:.2f}, {v2[1]:.2f})")
    cell = lattice_median(flat, v1, v2, s.denoise_k)
    margin = bleed_cm / w_cm * W

    if s.upscale_mode == "ai":
        cell = tiled_ai_upscale(cell, s.ai_scale, provider, log)
        k = s.ai_scale
        v1, v2, margin = v1 * k, v2 * k, margin * k

    return render_periodic(cell, v1, v2, out_w, out_h, margin=margin)


# ---------- ۹. ذخیره‌سازی ----------

def to_cmyk(im, icc_path, log=print):
    if not icc_path:
        log("هشدار: بدون پروفایل ICC تبدیل تقریبی است؛ پروفایل پرینتر را وارد کنید")
        return im.convert("CMYK"), None
    from PIL import ImageCms
    src = ImageCms.createProfile("sRGB")
    dst = ImageCms.getOpenProfile(icc_path)
    out = ImageCms.profileToProfile(im, src, dst, renderingIntent=1, outputMode="CMYK")
    with open(icc_path, "rb") as f:
        icc = f.read()
    return out, icc


def to_svg(img, svg_path, max_side=2000):
    import vtracer
    scale = min(1.0, max_side / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    tmp = svg_path + ".tmp.png"
    cv2.imwrite(tmp, small)
    try:
        vtracer.convert_image_to_svg_py(tmp, svg_path, colormode="color",
                                        mode="spline", filter_speckle=4, color_precision=6)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_output(img, path, s, log=print):
    s.validate()
    if s.out_format == "svg":
        to_svg(img, path)
        return

    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    extra = {}
    if s.color_mode == "CMYK":
        im, icc = to_cmyk(im, s.icc_path, log)
        if icc:
            extra["icc_profile"] = icc
    dpi = (s.dpi, s.dpi)

    if s.out_format == "png":
        im.save(path, dpi=dpi)
    elif s.out_format == "jpeg":
        im.save(path, dpi=dpi, quality=s.jpeg_quality, subsampling=0, **extra)
    elif s.out_format == "tiff":
        comp = {"none": "raw", "lzw": "tiff_lzw", "zip": "tiff_deflate"}[s.tiff_compression]
        im.save(path, dpi=dpi, compression=comp, **extra)
    elif s.out_format == "pdf":
        im.save(path, "PDF", resolution=float(s.dpi))
