"""Boruto: Two Blue Vortex x I Monster - "Who Is She?" motion edit.

Every frame is composited in numpy/OpenCV and piped to ffmpeg.

    python3 render.py stills 4.2 9.3 ...   # write single frames to OUT_DIR for review
    python3 render.py video                # full render + audio mix -> OUT_DIR/edit.mp4
"""
import math
import os
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "Edit test images and song")
UP = os.environ.get("UPSCALE_DIR", "/tmp/w/up")
FONTS = os.environ.get("FONT_DIR", "/tmp/w/fonts")
OUT = os.environ.get("OUT_DIR", "/tmp/w/out")

W, H, FPS = 1080, 1920, 30
SONG_START = 15.08
DUR = 39.9
C = np.array([W / 2, H / 2])

# Onset-refined hit times in edit time (song time - SONG_START).
BIG = [8.593, 9.173, 10.746, 14.874, 15.454, 17.024, 21.155, 21.735, 23.308,
       27.433, 28.016, 33.714, 34.297, 35.867]
SOFT = [6.816, 11.333, 13.100, 17.614, 19.381, 23.895, 25.662, 29.212, 31.940,
        36.457, 38.221]
FLASH = [(8.593, 1.0), (10.746, 0.55), (14.874, 0.9), (17.024, 0.5), (21.155, 0.9),
         (23.308, 0.6), (27.433, 1.0), (33.714, 0.9), (35.867, 0.45)]
PUNCH = [9.173, 15.454, 21.735, 28.016, 34.297]
GLITCH = [9.173, 21.735, 24.52, 28.016, 34.297]


# ----------------------------------------------------------------------------- utils

def clamp01(x):
    return max(0.0, min(1.0, x))


def ease_out(x, p=3):
    x = clamp01(x)
    return 1 - (1 - x) ** p


def ease_in_out(x):
    x = clamp01(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def ease_out_expo(x):
    x = clamp01(x)
    return 1 if x >= 1 else 1 - 2 ** (-10 * x)


def lerp(a, b, x):
    return a + (b - a) * x


def env(t, hits, decay):
    v = 0.0
    for h in hits:
        if 0 <= t - h < decay * 8:
            v += math.exp(-(t - h) / decay)
    return v


def noise1(t, seed, freqs=(13.0, 19.7, 29.3)):
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * math.pi, len(freqs))
    return sum(math.sin(2 * math.pi * f * t + p) for f, p in zip(freqs, ph)) / len(freqs)


class Cam:
    """Global camera state from the music: punch-zoom and shake."""

    def __init__(self, t):
        big = env(t, BIG, 0.10)
        soft = env(t, SOFT, 0.14)
        self.zoom = 1 + 0.075 * big + 0.022 * soft
        amp = 34 * env(t, BIG, 0.16) + 7 * soft
        self.dx = amp * noise1(t, 1)
        self.dy = amp * noise1(t, 2)
        self.rot = (1.6 * env(t, BIG, 0.16) + 0.3 * soft) * noise1(t, 3, (9.0, 14.1))


NO_CAM = type("NoCam", (), {"zoom": 1.0, "dx": 0.0, "dy": 0.0, "rot": 0.0})()


# ----------------------------------------------------------------------------- assets

# Watermarks baked into source renders, as fractional (x0, y0, x1, y1) boxes.
ERASE = {"sarada_uchiha": (0.615, 0.0, 1.0, 0.056)}


@lru_cache(maxsize=None)
def img(name, max_side=3200):
    """Premultiplied RGBA uint8 (RGB order)."""
    path = name if os.path.isabs(name) else os.path.join(UP, name)
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    for key, (x0, y0, x1, y1) in ERASE.items():
        if key in name:
            h, w = im.shape[:2]
            im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w), 3] = 0
    s = max_side / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if name.endswith("_fg.png"):
        # Choke and feather hand-cut mask edges so they sit into the plate.
        al = cv2.erode(im[..., 3], np.ones((3, 3), np.uint8))
        im[..., 3] = cv2.GaussianBlur(al, (0, 0), 1.6)
    a = im[..., 3:4].astype(np.float32) / 255
    im[..., :3] = (im[..., :3] * a + 0.5).astype(np.uint8)
    return im


@lru_cache(maxsize=None)
def blurred(name, sigma):
    im = img(name)
    small = cv2.resize(im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4)
    return cv2.resize(small, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)


def layer_matrix(im, fit="cover", s=1.0, fx=0.5, fy=0.5, x=0.0, y=0.0, rot=0.0,
                 depth=1.0, cam=NO_CAM):
    ih, iw = im.shape[:2]
    # Cover overscans so shake, punch-out and rotation never expose frame edges.
    base = {"cover": max(W / iw, H / ih) * 1.08, "contain": min(W / iw, H / ih),
            "width": W / iw, "height": H / ih}[fit]
    sc = base * s * cam.zoom ** depth
    r = math.radians(rot + cam.rot * depth)
    a, b = sc * math.cos(r), sc * math.sin(r)
    px, py = fx * iw, fy * ih
    tx = C[0] + x + cam.dx * depth - (a * px - b * py)
    ty = C[1] + y + cam.dy * depth - (b * px + a * py)
    return np.float32([[a, -b, tx], [b, a, ty]])


def draw(canvas, im, M, alpha=1.0, mode="over"):
    w = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    rgb = w[..., :3].astype(np.float32) * (alpha / 255)
    if mode == "screen":
        canvas[:] = 1 - (1 - canvas) * (1 - rgb)
        return canvas
    a = w[..., 3:4].astype(np.float32) * (alpha / 255)
    canvas *= 1 - a
    canvas += rgb
    return canvas


def put(canvas, name, alpha=1.0, blur=0, mode="over", **kw):
    im = blurred(name, blur) if blur else img(name)
    return draw(canvas, im, layer_matrix(im, **kw), alpha, mode)


def put_wind(canvas, name, t, weight, amp=14.0, **kw):
    """Layer with a travelling-wave displacement (cloth/hair flutter), scaled by `weight` (HxW)."""
    im = img(name)
    w = cv2.warpAffine(im, layer_matrix(im, **kw), (W, H), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    yy, xx = wind_grid()
    phase = xx * 0.011 + yy * 0.004 - t * 7.5
    dx = amp * weight * (np.sin(phase) + 0.4 * np.sin(phase * 2.3 + 1.7))
    dy = amp * 0.6 * weight * np.cos(phase * 0.8 + 0.6)
    w = cv2.remap(w, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = w[..., 3:4].astype(np.float32) / 255
    canvas *= 1 - a
    canvas += w[..., :3].astype(np.float32) / 255
    return canvas


@lru_cache(maxsize=None)
def wind_grid():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return yy, xx


def put_defocus(canvas, name, sigma, **kw):
    """Layer with a per-frame variable defocus (for rack focus pulls)."""
    if sigma < 0.5:
        return put(canvas, name, **kw)
    im = img(name)
    w = cv2.warpAffine(im, layer_matrix(im, **kw), (W, H), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    small = cv2.resize(w, (W // 2, H // 2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    small = cv2.GaussianBlur(small, (0, 0), sigma / 2)
    w = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    canvas *= 1 - w[..., 3:4]
    canvas += w[..., :3]
    return canvas


def blank(v=0.0):
    return np.full((H, W, 3), v, np.float32)


# ----------------------------------------------------------------------------- look

def make_lut(stops):
    xs = [s[0] for s in stops]
    lut = np.zeros((256, 3), np.float32)
    for c in range(3):
        lut[:, c] = np.interp(np.linspace(0, 1, 256), xs, [s[1][c] / 255 for s in stops])
    return lut


GRADES = {
    "blue": (make_lut([(0, (4, 6, 14)), (0.3, (16, 24, 46)), (0.62, (100, 120, 162)),
                       (0.88, (184, 198, 224)), (1, (216, 226, 244))]), 0.0),
    "blue_color": (make_lut([(0, (4, 6, 14)), (0.3, (18, 26, 50)), (0.65, (120, 146, 190)),
                             (1, (242, 247, 255))]), 0.45),
    "red": (make_lut([(0, (10, 3, 5)), (0.32, (48, 12, 18)), (0.66, (160, 118, 124)),
                      (0.9, (214, 192, 192)), (1, (234, 220, 218))]), 0.0),
    "red_color": (make_lut([(0, (10, 3, 5)), (0.35, (54, 14, 20)), (0.7, (196, 160, 160)),
                            (1, (252, 244, 240))]), 0.6),
    "ghost": (make_lut([(0, (2, 4, 10)), (0.4, (30, 44, 70)), (0.8, (170, 200, 230)),
                        (1, (235, 250, 255))]), 0.0),
}
FOG_TINT = {"blue": (0.55, 0.68, 0.95), "blue_color": (0.6, 0.72, 0.95),
            "red": (0.95, 0.55, 0.55), "red_color": (0.95, 0.6, 0.62),
            "ghost": (0.6, 0.8, 1.0)}


def grade(rgb, name, contrast=1.15, invert=False):
    lut, keep = GRADES[name]
    lum = rgb @ np.float32([0.299, 0.587, 0.114])
    if invert:
        lum = 1 - lum
    lum = np.clip((lum - 0.5) * contrast + 0.5, 0, 1)
    mapped = lut[(lum * 255).astype(np.uint8)]
    if keep:
        mapped = mapped * (1 - keep) + np.clip(rgb, 0, 1) * keep
    return mapped


@lru_cache(maxsize=None)
def fog_texture():
    rng = np.random.default_rng(7)
    h, w = 960, 540
    tex = np.zeros((h, w), np.float32)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    for sigma, amp in ((60, 1.0), (28, 0.55), (12, 0.3)):
        n = np.fft.fft2(rng.standard_normal((h, w)))
        n = np.real(np.fft.ifft2(n * np.exp(-2 * (math.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))))
        tex += (amp * n / n.std()).astype(np.float32)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return np.clip((tex - 0.35) * 1.8, 0, 1) ** 1.4


def fog(canvas, t, density, tint, speed=(18, -6)):
    if density <= 0:
        return canvas
    tex = fog_texture()
    th, tw = tex.shape
    out = np.zeros((H // 4, W // 4), np.float32)
    for k, (sx, sy, sc) in enumerate(((speed[0], speed[1], 1.0), (-speed[0] * 0.6, speed[1] * 1.7, 0.7))):
        ox = int(t * sx + k * 173) % tw
        oy = int(t * sy + k * 311) % th
        rolled = np.roll(np.roll(tex, -oy, 0), -ox, 1)
        ch, cw = int(H // 4 * sc), int(W // 4 * sc)
        out += cv2.resize(rolled[:ch, :cw], (W // 4, H // 4), interpolation=cv2.INTER_LINEAR) * (0.6 if k else 1.0)
    f = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * density
    col = np.float32(tint)
    return 1 - (1 - canvas) * (1 - np.clip(f * col, 0, 1))


@lru_cache(maxsize=None)
def particles():
    rng = np.random.default_rng(11)
    n = 90
    return dict(x=rng.uniform(0, 1, n), y=rng.uniform(0, 1, n), r=rng.uniform(0.6, 2.6, n),
                vy=rng.uniform(0.012, 0.05, n), vx=rng.uniform(-0.01, 0.01, n),
                ph=rng.uniform(0, 6.28, n), fl=rng.uniform(0.5, 2.0, n))


def dust(canvas, t, amount, tint):
    if amount <= 0:
        return canvas
    p = particles()
    lay = np.zeros((H // 2, W // 2), np.float32)
    xs = ((p["x"] + p["vx"] * t + 0.01 * np.sin(t * 0.8 + p["ph"])) % 1) * (W // 2)
    ys = ((p["y"] - p["vy"] * t) % 1) * (H // 2)
    br = 0.55 + 0.45 * np.sin(t * p["fl"] * 3 + p["ph"])
    for x, y, r, b in zip(xs, ys, p["r"], br):
        cv2.circle(lay, (int(x), int(y)), int(max(1, r)), float(b), -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 1.6)
    lay = cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    return canvas + lay * amount * np.float32(tint)


def glow(canvas, amount, threshold=0.62):
    if amount <= 0:
        return canvas
    small = cv2.resize(canvas, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hi = np.clip(small - threshold, 0, None) / (1 - threshold)
    g = cv2.GaussianBlur(hi, (0, 0), 10) + 0.6 * cv2.GaussianBlur(hi, (0, 0), 30)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    return 1 - (1 - canvas) * (1 - np.clip(g * amount, 0, 1))


def chroma(canvas, px):
    if px < 0.6:
        return canvas
    out = canvas.copy()
    for ch, sgn in ((0, 1), (2, -1)):
        s = 1 + sgn * px / (W / 2)
        M = np.float32([[s, 0, C[0] * (1 - s)], [0, s, C[1] * (1 - s)]])
        out[..., ch] = cv2.warpAffine(canvas[..., ch], M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return out


def zoom_blur(canvas, strength, n=6):
    if strength < 0.004:
        return canvas
    acc = canvas.copy()
    for i in range(1, n):
        s = 1 + strength * i / n
        M = np.float32([[s, 0, C[0] * (1 - s)], [0, s, C[1] * (1 - s)]])
        acc += cv2.warpAffine(canvas, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return acc / n


def motion_blur_x(canvas, px):
    if px < 2:
        return canvas
    k = int(px) | 1
    return cv2.blur(canvas, (k, 1))


def glitch(canvas, t, amount):
    if amount < 0.05:
        return canvas
    rng = np.random.default_rng(int(t * FPS) * 7919)
    out = canvas.copy()
    for _ in range(int(6 + 10 * amount)):
        y0 = rng.integers(0, H - 20)
        hgt = int(rng.integers(6, 90))
        sh = int(rng.normal(0, 60 * amount))
        out[y0:y0 + hgt] = np.roll(canvas[y0:y0 + hgt], sh, axis=1)
        if rng.random() < 0.4:
            out[y0:y0 + hgt, :, 0] = np.roll(canvas[y0:y0 + hgt, :, 0], sh + 14, axis=1)
    return out


@lru_cache(maxsize=None)
def vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - C[0]) / (W * 0.75)) ** 2 + ((yy - C[1]) / (H * 0.62)) ** 2)
    return np.clip(1 - 0.75 * np.clip(d - 0.35, 0, None) ** 1.6, 0, 1)[..., None]


@lru_cache(maxsize=None)
def grain_frames():
    rng = np.random.default_rng(3)
    fr = []
    for _ in range(8):
        n = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
        fr.append(cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)[..., None])
    return fr


@lru_cache(maxsize=None)
def halftone():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = math.radians(30)
    u = (xx * math.cos(a) + yy * math.sin(a)) / 9
    v = (-xx * math.sin(a) + yy * math.cos(a)) / 9
    return ((np.sin(u * math.pi) * np.sin(v * math.pi)) * 0.5 + 0.5)[..., None]


# ----------------------------------------------------------------------------- text

def font(name, size, weight=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), size)
    if weight:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


FONT_SPECS = {
    "italic": ("CormorantGaramond-Italic[wght].ttf", 500),
    "serif": ("CormorantGaramond[wght].ttf", 500),
    "caps": ("Cinzel[wght].ttf", 600),
    "jp": ("NotoSerifJP[wght].ttf", 500),
}


@lru_cache(maxsize=None)
def glyph_sprite(ch, style, size, color, glow_col, tracking):
    fn, wt = FONT_SPECS[style]
    f = font(fn, size, wt)
    pad = size
    bbox = f.getbbox(ch)
    adv = f.getlength(ch) + tracking
    w = int(max(bbox[2], adv) + 2 * pad)
    h = int(size * 1.6 + 2 * pad)
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((pad, pad), ch, font=f, fill=255)
    a = np.asarray(im).astype(np.float32) / 255
    g = np.clip(cv2.GaussianBlur(a, (0, 0), size * 0.12) * 1.1, 0, 1)
    shadow = np.clip(cv2.GaussianBlur(a, (0, 0), size * 0.3) * 2.2, 0, 1) * 0.75
    # Premultiplied layering: dark halo, then thin light glow, then the glyph.
    rgb = np.zeros(a.shape + (3,), np.float32)
    alpha = shadow.copy()
    rgb = rgb * (1 - g[..., None] * 0.7) + g[..., None] * 0.7 * np.float32(glow_col)
    alpha = alpha * (1 - g * 0.7) + g * 0.7
    rgb = rgb * (1 - a[..., None]) + a[..., None] * np.float32(color)
    alpha = alpha * (1 - a) + a
    rgb = rgb / np.maximum(alpha[..., None], 1e-4)
    return rgb.astype(np.float32), alpha.astype(np.float32), adv, pad


class Text:
    """One line of words; each word reveals letter by letter at its own time."""

    def __init__(self, words, y, style="italic", size=96, color=(1, 1, 1),
                 glow_col=(0.5, 0.65, 1.0), t_out=None, tracking=0, x=0, anim="rise",
                 space=None, accent=None):
        self.words = words  # [(text, t_in)]
        self.y, self.style, self.size = y, style, size
        self.color, self.glow_col, self.t_out = color, glow_col, t_out
        self.tracking, self.x, self.anim = tracking, x, anim
        self.space = space if space is not None else size * 0.32
        self.accent = accent or {}  # word index -> (color, glow)

    def render(self, canvas, t):
        if t < self.words[0][1] - 0.01:
            return canvas
        if self.t_out is not None and t > self.t_out + 0.45:
            return canvas
        out_k = 1.0 if self.t_out is None else 1 - clamp01((t - self.t_out) / 0.4)
        glyphs = []
        for wi, (word, t_in) in enumerate(self.words):
            col, gcol = self.accent.get(wi, (self.color, self.glow_col))
            for li, ch in enumerate(word):
                glyphs.append((ch, t_in + li * 0.035, col, gcol))
            glyphs.append((" ", 0, col, gcol))
        glyphs.pop()
        sprites, total = [], 0.0
        for ch, tin, col, gcol in glyphs:
            if ch == " ":
                sprites.append(None)
                total += self.space
                continue
            sp = glyph_sprite(ch, self.style, self.size, tuple(col), tuple(gcol), self.tracking)
            sprites.append(sp)
            total += sp[2]
        x = C[0] + self.x - total / 2
        for (ch, tin, _, _), sp in zip(glyphs, sprites):
            if sp is None:
                x += self.space
                continue
            rgb, alpha, adv, pad = sp
            k = clamp01((t - tin) / 0.28)
            if k > 0:
                e = ease_out(k)
                a = e * out_k
                ox = oy = 0.0
                if self.anim == "rise":
                    dy = (1 - e) * self.size * 0.35
                    blur = (1 - e) * 10 + (1 - out_k) * 12
                else:  # "slam"
                    dy = 0
                    blur = (1 - e) * 4 + (1 - out_k) * 12
                    sc = 1 + (1 - ease_out_expo(k)) * 0.8
                    if sc > 1.01:
                        h0, w0 = alpha.shape
                        rgb = cv2.resize(rgb, None, fx=sc, fy=sc)
                        alpha = cv2.resize(alpha, None, fx=sc, fy=sc)
                        ox = (alpha.shape[1] - w0) / 2
                        oy = (alpha.shape[0] - h0) / 2
                if blur > 0.5:
                    rgb = cv2.GaussianBlur(rgb, (0, 0), blur)
                    alpha = cv2.GaussianBlur(alpha, (0, 0), blur)
                x0 = int(x - pad - ox)
                y0 = int(self.y - self.size * 0.8 - pad + dy - oy)
                blit(canvas, rgb, alpha * a, x0, y0)
            x += adv
        return canvas


def blit(canvas, rgb, alpha, x0, y0):
    h, w = alpha.shape
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + w, W), min(y0 + h, H)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    sa = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0, None]
    sr = rgb[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    region = canvas[cy0:cy1, cx0:cx1]
    region *= 1 - sa
    region += sr * sa


WHITE = (1.0, 1.0, 1.0)
RED = (1.0, 0.28, 0.3)
BLUE_GLOW = (0.45, 0.62, 1.0)
RED_GLOW = (1.0, 0.25, 0.25)

TEXTS = [
    Text([("調子に乗るなよ", 0.10)], y=1500, style="jp", size=58, t_out=2.45,
         glow_col=(0.3, 0.4, 0.7), space=0),
    Text([("don't", 0.18), ("get", 0.36), ("cocky…", 0.52)], y=1590, style="italic", size=50,
         t_out=2.45, glow_col=(0.3, 0.4, 0.7)),
    Text([("カワキ", 2.68)], y=1500, style="jp", size=58, t_out=3.35,
         glow_col=(0.3, 0.4, 0.7), space=0),
    Text([("…kawaki.", 2.74)], y=1590, style="italic", size=50, t_out=3.35,
         glow_col=(0.3, 0.4, 0.7)),
    Text([("oh,", 3.50)], y=1330, style="italic", size=64, t_out=6.6, glow_col=RED_GLOW),
    Text([("WHO", 4.90), ("IS", 6.30), ("SHE?", 6.816)], y=1480, style="caps", size=118,
         tracking=14, t_out=8.4, anim="slam", glow_col=RED_GLOW,
         accent={2: (RED, RED_GLOW)}),
    Text([("a", 10.84), ("misty", 10.96), ("memory", 12.66)], y=1480, style="italic",
         size=112, t_out=14.55, glow_col=BLUE_GLOW),
    Text([("a", 17.10), ("haunting", 17.82), ("face", 19.40)], y=1480, style="italic",
         size=112, t_out=20.95, glow_col=BLUE_GLOW),
    Text([("is", 22.48), ("she", 23.18)], y=1470, style="italic", size=84, t_out=25.4,
         glow_col=BLUE_GLOW),
    Text([("A", 23.96), ("GHOST?", 24.52)], y=1590, style="caps", size=120, tracking=18,
         t_out=25.4, anim="slam", glow_col=(0.6, 0.85, 1.0)),
    Text([("BORUTO", 25.80)], y=1470, style="caps", size=86, tracking=30, t_out=27.25,
         glow_col=BLUE_GLOW),
    Text([("TWO BLUE VORTEX", 26.05)], y=1560, style="caps", size=40, tracking=22,
         t_out=27.25, glow_col=BLUE_GLOW, space=20),
    Text([("or", 35.66), ("is", 36.08), ("she", 36.48)], y=1010, style="italic", size=84,
         glow_col=RED_GLOW, t_out=39.1),
    Text([("just", 37.26), ("a", 38.12), ("dream?", 38.50)], y=1125, style="italic",
         size=118, glow_col=(1.0, 0.8, 0.85), t_out=39.1),
]


# ----------------------------------------------------------------------------- shots
# Each shot: (t0, t1, fn(lt, dur, t, cam) -> rgb, grade, fx dict)

def sh_cold_eyes(lt, d, t, cam):
    c = blank()
    k = lt / d
    put(c, "images (16)(1)_png.png", fit="width", s=lerp(1.35, 1.55, ease_in_out(k)),
        fx=0.62, fy=0.5, x=lerp(40, -40, k), y=-120, cam=cam)
    flick = 1.0 if lt > 0.35 else (0.3 if int(lt * 30) % 3 else 1.0) * ease_out(lt / 0.35)
    return c * flick


def sh_cold_kawaki(lt, d, t, cam):
    c = blank()
    k = ease_out(lt / d, 2)
    put(c, "Kawakikama_webp.png", fit="height", s=lerp(1.9, 2.2, k), fx=0.52,
        fy=lerp(0.30, 0.22, k), y=-160, cam=cam)
    return c


def sh_who(lt, d, t, cam):
    c = blank()
    k = ease_in_out(lt / d)
    put(c, "images (15)_jpeg.png", fit="height", s=1.12, fx=lerp(0.14, 0.86, k), fy=0.5,
        rot=lerp(-2, 1.5, k), cam=cam)
    return c


def sh_she(lt, d, t, cam):
    c = blank(0.02)
    put(c, "images (15)_jpeg.png", fit="height", s=1.3, fx=0.8, blur=40, alpha=0.55,
        cam=cam, depth=0.3)
    k = ease_out(lt / d, 3)
    put(c, "sarada_uchiha_timeskip_render_png_by_me_102_by_uzimaho13_dm87mcw-375w-2x_png.png",
        fit="width", s=lerp(1.25, 1.55, k), fx=0.5, fy=lerp(0.62, 0.17, k), y=-200, cam=cam)
    return c


def parallax(c, n, lt, d, cam, s0, s1, fx=0.5, fy=0.5, bg_blur=14, bg_dark=0.55,
             drift=(0, 0), rot=(0, 0), fg_extra=0.06, bg_invert=False):
    k = ease_in_out(lt / d)
    bg, fg = f"{n}_bg.png", f"{n}_fg.png"
    s = lerp(s0, s1, k)
    put(c, bg, blur=bg_blur, fit="cover", s=s * 1.02, fx=fx, fy=fy,
        x=drift[0] * k * 0.4, y=drift[1] * k * 0.4, rot=lerp(*rot, k) * 0.5, cam=cam, depth=0.5)
    if bg_invert:
        c[:] = 1 - c
        edge = cv2.warpAffine(np.ones(img(bg).shape[:2], np.float32),
                              layer_matrix(img(bg), fit="cover", s=s * 1.02, fx=fx, fy=fy,
                                           x=drift[0] * k * 0.4, y=drift[1] * k * 0.4,
                                           rot=lerp(*rot, k) * 0.5, cam=cam, depth=0.5), (W, H))
        c *= edge[..., None]
    c *= bg_dark
    put(c, fg, fit="cover", s=s * (1 + fg_extra * k), fx=fx, fy=fy, x=drift[0] * k,
        y=drift[1] * k, rot=lerp(*rot, k), cam=cam, depth=1.2)
    return c


def sh_fall(lt, d, t, cam):
    c = blank()
    return parallax(c, 10, lt, d, cam, 1.12, 1.3, fy=0.45, rot=(-4, 5), drift=(0, 60),
                    bg_blur=18, bg_dark=0.5, fg_extra=0.1)


def sh_misty(lt, d, t, cam):
    """Rack focus: starts lost in the haze, pulls sharp on 'memory', keeps orbiting."""
    c = blank()
    k = ease_in_out(lt / d)
    kf = ease_out(lt / d, 2)
    put(c, "17_bg.png", blur=24, fit="cover", s=lerp(1.15, 1.35, k), fx=0.5, fy=0.4,
        rot=lerp(-3, 2, k), x=lerp(40, -40, k), cam=cam, depth=0.5)
    c *= 0.4
    focus = 16 * (1 - ease_in_out((t - 11.1) / 1.5))
    put_defocus(c, "17_fg.png", focus, fit="cover", s=lerp(1.02, 1.42, kf), fx=0.52,
                fy=lerp(0.4, 0.3, kf), rot=lerp(3, -2.5, k), x=lerp(-60, 30, k), cam=cam,
                depth=1.25)
    return c


def sh_crouch(lt, d, t, cam):
    c = blank()
    return parallax(c, 8, lt, d, cam, 1.15, 1.05, fx=0.45, fy=0.4, drift=(80, 0),
                    bg_blur=16, bg_dark=0.45)


def sh_eyes(lt, d, t, cam):
    c = blank()
    k = ease_out(lt / d, 2)
    put(c, "7_bg.png", blur=20, fit="cover", s=1.35, fx=0.55, fy=0.42, cam=cam, depth=0.5)
    c *= 0.45
    put(c, "7_fg.png", fit="cover", s=lerp(1.9, 1.6, k), fx=0.55, fy=0.4, cam=cam, depth=1.3)
    return c


def sh_throne(lt, d, t, cam):
    c = blank()
    return parallax(c, 12, lt, d, cam, 1.0, 1.45, fx=0.5, fy=0.33, bg_blur=8,
                    bg_dark=0.9, fg_extra=0.08, bg_invert=True)


def sh_face(lt, d, t, cam):
    c = blank()
    k = ease_out(lt / d, 2)
    put(c, "9_bg.png", blur=26, fit="cover", s=1.3, fx=0.75, cam=cam, depth=0.4)
    c *= 0.5
    put_wind(c, "9_fg.png", t, scarf_weight(), fit="height", s=lerp(2.25, 2.0, k), fx=0.83,
             fy=0.33, x=lerp(-40, 30, k), rot=lerp(2, -1, k), cam=cam, depth=1.2)
    return c


@lru_cache(maxsize=None)
def scarf_weight():
    """Flutter only the scarf band below the face, strongest at the trailing (left) end."""
    yy, xx = wind_grid()
    vert = np.clip((yy - 820) / 320, 0, 1)
    horiz = np.clip((900 - xx) / 700, 0.25, 1)
    return cv2.GaussianBlur(vert * horiz, (0, 0), 25)


def sh_slash(lt, d, t, cam):
    c = blank()
    return parallax(c, 11, lt, d, cam, 1.22, 1.08, fx=0.5, fy=0.45, rot=(3, -2),
                    bg_blur=18, bg_dark=0.4, drift=(0, -40))


def sh_ghost(lt, d, t, cam):
    """Ink-on-paper afterimages (darken blend) that trail the camera move."""
    name = "images (13)_jpeg.png"
    base = dict(fit="cover", fx=0.35, fy=0.42)
    k = ease_in_out(lt / d)
    c = blank()
    put(c, name, s=lerp(1.08, 1.3, k), x=lerp(0, -40, k), cam=cam, **base)
    spread = 1 + 2.5 * env(t, [24.52], 0.25)
    for i in range(1, 5):
        kk = ease_in_out(max(lt - i * 0.1, 0) / d)
        echo = blank()
        put(echo, name, s=lerp(1.08, 1.3, kk) + i * 0.03 * spread,
            x=lerp(0, -40, kk) + i * 34 * spread, y=-i * 10 * spread, cam=cam, **base)
        a = 0.45 / i
        c = np.minimum(c, echo * a + c * (1 - a))
    return c


def text_scrim(t):
    """(0..1 strength, centre y px) of the dark band behind the active lyric lines."""
    v, ys = 0.0, []
    for tx in TEXTS:
        t_in = tx.words[0][1]
        t_out = tx.t_out if tx.t_out is not None else DUR
        if t_in - 0.3 <= t <= t_out + 0.5:
            v = max(v, clamp01((t - t_in + 0.3) / 0.3) * clamp01((t_out + 0.5 - t) / 0.5))
            ys.append(tx.y - tx.size * 0.3)
    return v, (sum(ys) / len(ys) if ys else H * 0.77)


@lru_cache(maxsize=None)
def scrim_mask(cy):
    y = np.arange(H, dtype=np.float32)
    band = np.exp(-((y - cy) / (H * 0.12)) ** 2)
    return band[:, None, None]


def sh_vortex(lt, d, t, cam):
    c = blank()
    k = ease_in_out(lt / d)
    put(c, "my-favorite-two-blue-vortex-panels-v0-ozwkciafxfqd1_jpg.png", fit="cover",
        s=lerp(1.0, 1.2, k), fx=0.5, fy=lerp(0.35, 0.62, k), cam=cam)
    return c


def sh_lightning(lt, d, t, cam):
    c = blank()
    k = ease_out(lt / d, 4)
    put(c, "purple-lightning-boruto-uzumaki-boruto-tbv-6-1_png.png", fit="height", s=1.05,
        fx=lerp(0.12, 0.62, k), fy=0.5, rot=lerp(-6, 0, k), cam=cam)
    speed = (1 - k) * 90
    return motion_blur_x(c, speed)


def sh_sarada_eye(lt, d, t, cam):
    c = blank()
    k = ease_in_out(lt / d)
    put(c, "images (19)_jpeg.png", fit="cover", s=lerp(1.25, 2.3, k), fx=lerp(0.3, 0.2, k),
        fy=lerp(0.45, 0.49, k), cam=cam)
    return c


def sh_float(lt, d, t, cam):
    c = blank()
    k = ease_in_out(lt / d)
    put(c, "images (14)_jpeg.png", fit="cover", s=lerp(1.12, 1.25, k), fx=0.5,
        fy=lerp(0.55, 0.42, k), rot=lerp(1.5, -1.5, k), cam=cam)
    return c


def sh_sarada_action(lt, d, t, cam):
    c = blank()
    k = ease_out(lt / d, 3)
    put(c, "images (18)_jpeg.png", fit="cover", s=lerp(1.45, 1.2, k), fx=0.42,
        fy=lerp(0.25, 0.4, k), rot=lerp(4, 0, k), cam=cam)
    return c


def sh_dream(lt, d, t, cam):
    c = blank()
    k = ease_in_out(lt / d)
    put(c, "sarada_uchiha_timeskip_render_png_by_me_102_by_uzimaho13_dm87mcw-375w-2x_png.png",
        fit="width", s=lerp(1.75, 2.0, k), fx=0.5, fy=0.12, y=-260, cam=cam, depth=0.8)
    over = np.zeros_like(c)
    put(over, "17_fg.png", fit="cover", s=lerp(1.0, 0.9, k), fx=0.5, fy=0.22,
        x=lerp(70, -20, k), y=560, cam=cam, depth=1.3)
    # Double exposure: Boruto burns in over the lower frame, Sarada's dark areas take him.
    lum = (over @ np.float32([0.3, 0.6, 0.1]))[..., None]
    fade = np.clip((np.linspace(0, 1, H, dtype=np.float32)[:, None, None] - 0.42) / 0.25, 0, 1)
    a = ease_in_out(lt / 1.4) * fade
    c[:] = c * (1 - 0.55 * a) + np.maximum(c, lum * np.float32([0.85, 0.9, 1.0])) * 0.55 * a
    c[:] = 1 - (1 - c) * (1 - lum * 0.45 * a)
    white = ease_in_out((t - 38.4) / 1.3)
    return c * (1 - white) + white * 0.96 * (1 - clamp01((t - 39.5) / 0.4))


SHOTS = [
    (0.00, 2.60, sh_cold_eyes, "blue", dict(fog=0.18, dust=0.25, glow=0.3)),
    (2.60, 3.45, sh_cold_kawaki, "blue_color", dict(fog=0.3, dust=0.2, glow=0.3, enter="zoom")),
    (3.45, 6.816, sh_who, "red_color", dict(fog=0.25, dust=0.3, glow=0.5, enter="white")),
    (6.816, 8.593, sh_she, "red_color", dict(fog=0.3, dust=0.4, glow=0.55, enter="zoom")),
    (8.593, 10.746, sh_fall, "blue", dict(fog=0.25, dust=0.3, glow=0.45)),
    (10.746, 13.100, sh_misty, "blue_color", dict(fog=0.75, dust=0.45, glow=0.55)),
    (13.100, 14.874, sh_crouch, "blue", dict(fog=0.6, dust=0.35, glow=0.4, enter="whip")),
    (14.874, 17.024, sh_eyes, "blue", dict(fog=0.3, dust=0.2, glow=0.55)),
    (17.024, 19.381, sh_throne, "blue", dict(fog=0.35, dust=0.35, glow=0.5)),
    (19.381, 21.155, sh_face, "blue_color", dict(fog=0.35, dust=0.3, glow=0.5, enter="zoom")),
    (21.155, 23.308, sh_slash, "blue", dict(fog=0.2, dust=0.2, glow=0.45)),
    (23.308, 25.662, sh_ghost, "ghost", dict(fog=0.3, dust=0.4, glow=0.6, invert=24.52)),
    (25.662, 27.433, sh_vortex, "blue", dict(fog=0.3, dust=0.5, glow=0.5, invert=True,
                                             enter="white")),
    (27.433, 29.212, sh_lightning, "blue", dict(fog=0.1, dust=0.1, glow=0.6)),
    (29.212, 31.940, sh_sarada_eye, "red", dict(fog=0.35, dust=0.4, glow=0.5, halftone=0.12,
                                                enter="zoom")),
    (31.940, 33.714, sh_float, "red", dict(fog=0.4, dust=0.45, glow=0.45, enter="whip")),
    (33.714, 35.867, sh_sarada_action, "red", dict(fog=0.15, dust=0.2, glow=0.5)),
    (35.867, DUR, sh_dream, "red_color", dict(fog=0.3, dust=0.55, glow=0.6, enter="white")),
]


def shot_at(t):
    for i, s in enumerate(SHOTS):
        if s[0] <= t < s[1]:
            return i, s
    return len(SHOTS) - 1, SHOTS[-1]


def frame(t):
    i, (t0, t1, fn, gname, fx) = shot_at(t)
    lt, d = t - t0, t1 - t0
    cam = Cam(t)
    c = fn(lt, d, t, cam)
    inv = fx.get("invert", False)
    if isinstance(inv, float):
        inv = t >= inv
    c = grade(np.clip(c, 0, 1), gname, invert=inv)

    enter = fx.get("enter")
    if enter == "zoom":
        c = zoom_blur(c, 0.25 * (1 - ease_out(lt / 0.22)))
    elif enter == "whip":
        c = motion_blur_x(c, 160 * (1 - ease_out(lt / 0.2)))
    elif enter == "white":
        c = c + (1 - c) * (1 - ease_out(lt / 0.35))
    # Pre-roll: blur-out the last frames of a shot into a hard cut.
    tail = t1 - t
    if tail < 0.1 and i + 1 < len(SHOTS) and SHOTS[i + 1][4].get("enter") == "zoom":
        c = zoom_blur(c, 0.18 * (1 - tail / 0.1))

    tint = FOG_TINT[gname]
    c = fog(c, t, fx.get("fog", 0), tint)
    c = dust(c, t, fx.get("dust", 0), tint)
    if fx.get("halftone"):
        c = c * (1 - fx["halftone"] + fx["halftone"] * (0.6 + 0.4 * halftone()))
    c = glow(c, fx.get("glow", 0.4))
    c = chroma(c, 1.5 + 16 * env(t, BIG, 0.09) + 5 * env(t, SOFT, 0.1))
    c = glitch(c, t, 1.2 * env(t, GLITCH, 0.05))
    fl = sum(a * math.exp(-(t - h) / 0.09) for h, a in FLASH if 0 <= t - h < 0.6)
    if fl > 0:
        c = c + (1 - c) * min(fl, 1)
    c = c * vignette()
    sc, cy = text_scrim(t)
    if sc > 0:
        c = c * (1 - 0.55 * sc * scrim_mask(int(cy) // 10 * 10))
    c = c + grain_frames()[int(t * FPS) % 8] * 0.035
    for tx in TEXTS:
        tx.render(c, t)
    if t < 0.25:
        c *= ease_out(t / 0.25)
    if t > DUR - 0.35:
        c *= clamp01((DUR - t) / 0.35)
    return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- output

def render_frame(n):
    return frame(n / FPS).tobytes()


def audio(out_path):
    song = os.path.join(SRC, "I Monster - Who Is She？ (Lyrics) [VR-EiEBqfWw].opus")
    voice = os.path.join(SRC, "139549b76451ec26ea2a48a3fad5aabc.mp4")
    fc = (
        f"[0:a]atrim={SONG_START}:{SONG_START + DUR},asetpts=PTS-STARTPTS,"
        "volume='if(lt(t,3.3),0.22,if(lt(t,3.8),0.22+(t-3.3)/0.5*0.78,1))':eval=frame,"
        f"afade=t=in:d=0.2,afade=t=out:st={DUR - 1.6}:d=1.6[s];"
        "[1:a]atrim=0:3.5,asetpts=PTS-STARTPTS,highpass=f=90,"
        "acompressor=threshold=-30dB:ratio=4:attack=5:release=120,volume=20,"
        "afade=t=out:st=3.25:d=0.25,aecho=0.8:0.5:70|140:0.22|0.12,adelay=60|60[v];"
        "[s][v]amix=inputs=2:normalize=0,alimiter=limit=0.95[a]"
    )
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", song, "-i", voice,
                    "-filter_complex", fc, "-map", "[a]", "-ac", "2", "-ar", "48000",
                    "-c:a", "aac", "-b:a", "256k", out_path], check=True)


def video():
    os.makedirs(OUT, exist_ok=True)
    silent = os.path.join(OUT, "video_only.mp4")
    aud = os.path.join(OUT, "audio.m4a")
    final = os.path.join(OUT, "edit.mp4")
    audio(aud)
    n = int(round(DUR * FPS))
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-preset", "slow", "-crf", "16", "-maxrate", "16M", "-bufsize", "32M",
                           "-pix_fmt", "yuv420p", silent],
                          stdin=subprocess.PIPE)
    with Pool(int(os.environ.get("WORKERS", os.cpu_count() or 4))) as pool:
        for i, buf in enumerate(pool.imap(render_frame, range(n), chunksize=2)):
            ff.stdin.write(buf)
            if i % 60 == 0:
                print(f"frame {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", aud, "-c:v", "copy",
                    "-c:a", "copy", "-shortest", "-movflags", "+faststart", final], check=True)
    print("wrote", final)


def stills(times):
    os.makedirs(OUT, exist_ok=True)
    for t in times:
        im = frame(float(t))
        p = os.path.join(OUT, f"still_{float(t):06.2f}.jpg")
        Image.fromarray(im).save(p, quality=90)
        print(p)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "video"
    if cmd == "stills":
        stills(sys.argv[2:])
    else:
        video()
