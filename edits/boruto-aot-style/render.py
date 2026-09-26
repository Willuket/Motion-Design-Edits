"""Boruto x I Monster - "Who Is She?" (the 'lost embrace' verse), rebuilt shot-for-shot on the
structure of the Attack on Titan "My best work yet" edit.

Every frame is composited in numpy/OpenCV and piped to ffmpeg.

    python3 render.py stills 1.9 10.2 ...   # write single frames to OUT_DIR for review
    python3 render.py video                 # full render + audio -> OUT_DIR/edit.mp4
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
SRC = os.path.join(HERE, "src")
SONG = os.path.join(HERE, "..", "..", "Edit test images and song",
                    "I Monster - Who Is She？ (Lyrics) [VR-EiEBqfWw].opus")
UP = os.environ.get("UPSCALE_DIR", "/tmp/aot/up")
LAYERS = os.environ.get("LAYER_DIR", "/tmp/aot/layers")
FONTS = os.path.join(HERE, "fonts")
OUT = os.environ.get("OUT_DIR", "/tmp/aot/out")

W, H, FPS = 1440, 1080, 30
SONG_START = 163.54
DUR = 31.43
C = np.array([W / 2, H / 2])

# Sung word onsets (edit time), from word-level alignment of this verse.
WORDS = dict(IS=0.0, HE=0.28, A=0.94, LOST=1.46, EMBRACE=2.46, SOME=6.40, ACROSS=7.10,
             THE=7.94, SEA=8.34, OF=8.96, TIME=9.56, A2=12.86, LOVE=13.14, IMMORTAL=13.56,
             SUCH=14.70, AS=15.36, MINE=15.86, WILL=19.48, COME=20.22, TO=21.32, BE=21.96,
             ETERNITY=25.80)
PUNCH = [(0.28, 0.3), (0.94, 0.3), (1.46, 0.45), (2.46, 1.0), (3.55, 0.8), (6.40, 0.5),
         (7.10, 0.4), (7.94, 0.3), (8.34, 0.4), (9.03, 1.0), (9.56, 0.8), (12.07, 0.6),
         (13.56, 0.3), (14.70, 0.4), (15.36, 0.3), (15.86, 0.7), (19.48, 0.3), (20.22, 0.3),
         (21.40, 0.9), (21.96, 0.7), (22.97, 0.6), (23.37, 0.5), (23.90, 0.9)]


# ----------------------------------------------------------------------------- utils

def clamp01(x):
    return max(0.0, min(1.0, x))


def ease_out(x, p=3):
    x = clamp01(x)
    return 1 - (1 - x) ** p


def ease_in(x, p=2):
    return clamp01(x) ** p


def ease_in_out(x):
    x = clamp01(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def ease_out_expo(x):
    x = clamp01(x)
    return 1 if x >= 1 else 1 - 2 ** (-10 * x)


def lerp(a, b, x):
    return a + (b - a) * x


def ramp(t, t0, t1):
    return clamp01((t - t0) / (t1 - t0)) if t1 != t0 else float(t >= t0)


def env(t, hits, decay):
    v = 0.0
    for h, a in hits:
        if 0 <= t - h < decay * 8:
            v += a * math.exp(-(t - h) / decay)
    return v


def noise1(t, seed, freqs=(0.31, 0.53, 0.87)):
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * math.pi, len(freqs))
    return sum(math.sin(2 * math.pi * f * t + p) for f, p in zip(freqs, ph)) / len(freqs)


class Cam:
    """Slow handheld drift plus lyric-synced punch-ins."""

    def __init__(self, t, drift=1.0, punch=1.0):
        p = env(t, PUNCH, 0.12) * punch
        self.zoom = 1 + 0.035 * p
        shake = 9 * env(t, PUNCH, 0.08) * punch
        self.dx = drift * 10 * noise1(t, 1) + shake * noise1(t * 40, 4, (1.0, 1.7, 2.9))
        self.dy = drift * 7 * noise1(t, 2) + shake * noise1(t * 40, 5, (1.1, 1.9, 3.1))
        self.rot = drift * 0.35 * noise1(t, 3) + 0.5 * p * noise1(t * 30, 6, (1.3, 2.1))


NO_CAM = type("NoCam", (), {"zoom": 1.0, "dx": 0.0, "dy": 0.0, "rot": 0.0})()


# ----------------------------------------------------------------------------- assets

def resolve(name):
    for d in (LAYERS, UP, SRC):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


@lru_cache(maxsize=None)
def img(name, max_side=3000):
    """Premultiplied RGBA uint8 (RGB order)."""
    im = cv2.imdecode(np.fromfile(resolve(name), np.uint8), cv2.IMREAD_UNCHANGED)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    s = max_side / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if name.endswith("_fg.png"):
        al = cv2.erode(im[..., 3], np.ones((3, 3), np.uint8))
        im[..., 3] = cv2.GaussianBlur(al, (0, 0), 1.2)
    a = im[..., 3:4].astype(np.float32) / 255
    im[..., :3] = (im[..., :3] * a + 0.5).astype(np.uint8)
    return im


@lru_cache(maxsize=None)
def blurred(name, sigma):
    im = img(name)
    small = cv2.resize(im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4)
    return cv2.resize(small, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)


@lru_cache(maxsize=None)
def framed(name, border=0.045, inner=0.012):
    """Portrait with a dark wooden frame and a thin gold fillet (premultiplied RGBA)."""
    im = img(name)[..., :3]
    h, w = im.shape[:2]
    b, g = int(max(h, w) * border), int(max(h, w) * inner)
    out = np.zeros((h + 2 * b, w + 2 * b, 4), np.uint8)
    yy, xx = np.mgrid[0:h + 2 * b, 0:w + 2 * b].astype(np.float32)
    wood = 26 + 10 * np.sin(xx * 0.05 + np.sin(yy * 0.011) * 3) + 6 * np.sin(yy * 0.21)
    out[..., 0], out[..., 1], out[..., 2] = wood * 1.1, wood * 0.8, wood * 0.6
    out[b - g:b + h + g, b - g:b + w + g, :3] = (150, 118, 64)
    out[b:b + h, b:b + w, :3] = im
    out[..., 3] = 255
    return out


def layer_matrix(im, fit="cover", s=1.0, fx=0.5, fy=0.5, x=0.0, y=0.0, rot=0.0,
                 depth=1.0, cam=NO_CAM, sx=1.0):
    ih, iw = im.shape[:2]
    base = {"cover": max(W / iw, H / ih) * 1.06, "contain": min(W / iw, H / ih),
            "width": W / iw, "height": H / ih, "px": 1.0}[fit]
    sc = base * s * cam.zoom ** depth
    r = math.radians(rot + cam.rot * depth)
    a, b = sc * math.cos(r), sc * math.sin(r)
    px, py = fx * iw, fy * ih
    tx = C[0] + x + cam.dx * depth - (a * sx * px - b * py)
    ty = C[1] + y + cam.dy * depth - (b * sx * px + a * py)
    return np.float32([[a * sx, -b, tx], [b * sx, a, ty]])


def warp(im, M):
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def draw(canvas, im, M, alpha=1.0, mode="over"):
    w = warp(im, M)
    rgb = w[..., :3].astype(np.float32) * (alpha / 255)
    if mode == "screen":
        canvas[:] = 1 - (1 - canvas) * (1 - rgb)
        return canvas
    if mode == "add":
        canvas += rgb
        return canvas
    a = w[..., 3:4].astype(np.float32) * (alpha / 255)
    canvas *= 1 - a
    canvas += rgb
    return canvas


def put(canvas, name, alpha=1.0, blur=0, mode="over", **kw):
    im = blurred(name, blur) if blur else img(name)
    return draw(canvas, im, layer_matrix(im, **kw), alpha, mode)


def layer(name, blur=0, **kw):
    """Warped layer as float (premultiplied rgb, alpha)."""
    im = blurred(name, blur) if blur else img(name)
    w = warp(im, layer_matrix(im, **kw)).astype(np.float32) / 255
    return w[..., :3], w[..., 3:4]


def over(canvas, rgb, a, alpha=1.0):
    canvas *= 1 - a * alpha
    canvas += rgb * alpha
    return canvas


def blank(v=0.0):
    return np.full((H, W, 3), v, np.float32)


def lum(c):
    return c @ np.float32([0.299, 0.587, 0.114])


def mono(c, tint=(1, 1, 1), contrast=1.0, pivot=0.5):
    l = np.clip((lum(c) - pivot) * contrast + pivot, 0, 1)
    return l[..., None] * np.float32(tint)


def saturate(c, s):
    l = lum(c)[..., None]
    return l + (c - l) * s


def screen(a, b):
    return 1 - (1 - a) * (1 - np.clip(b, 0, 1))


# ----------------------------------------------------------------------------- look

@lru_cache(maxsize=None)
def grid():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return yy, xx


@lru_cache(maxsize=None)
def vignette(strength=0.8):
    yy, xx = grid()
    d = np.sqrt(((xx - C[0]) / (W * 0.62)) ** 2 + ((yy - C[1]) / (H * 0.62)) ** 2)
    return np.clip(1 - strength * np.clip(d - 0.4, 0, None) ** 1.5, 0, 1)[..., None]


@lru_cache(maxsize=None)
def grain_frames():
    rng = np.random.default_rng(3)
    fr = []
    for _ in range(10):
        n = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
        fr.append(cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)[..., None])
    return fr


def glow(canvas, amount, threshold=0.6, tint=(1, 1, 1)):
    if amount <= 0:
        return canvas
    small = cv2.resize(canvas, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hi = np.clip(small - threshold, 0, None) / (1 - threshold)
    g = cv2.GaussianBlur(hi, (0, 0), 8) + 0.7 * cv2.GaussianBlur(hi, (0, 0), 28)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR) * np.float32(tint)
    return screen(canvas, g * amount)


def chroma(canvas, px):
    if px < 0.6:
        return canvas
    out = canvas.copy()
    for ch, sgn in ((0, 1), (2, -1)):
        s = 1 + sgn * px / (W / 2)
        M = np.float32([[s, 0, C[0] * (1 - s)], [0, s, C[1] * (1 - s)]])
        out[..., ch] = cv2.warpAffine(canvas[..., ch], M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return out


def zoom_blur(canvas, strength, n=7, center=None):
    if strength < 0.004:
        return canvas
    cx, cy = C if center is None else center
    acc = canvas.copy()
    for i in range(1, n):
        s = 1 + strength * i / n
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        acc += cv2.warpAffine(canvas, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return acc / n


def motion_blur_x(canvas, px):
    if px < 2:
        return canvas
    k = int(px) | 1
    return cv2.blur(canvas, (k, 1))


def defocus(canvas, sigma):
    if sigma < 0.4:
        return canvas
    small = cv2.resize(canvas, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 2), (W, H))


@lru_cache(maxsize=None)
def noise_tex(seed=7, h=540, w=720, sigmas=((50, 1.0), (22, 0.55), (9, 0.3))):
    rng = np.random.default_rng(seed)
    tex = np.zeros((h, w), np.float32)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    for sigma, amp in sigmas:
        n = np.fft.fft2(rng.standard_normal((h, w)))
        n = np.real(np.fft.ifft2(n * np.exp(-2 * (math.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))))
        tex += (amp * n / n.std()).astype(np.float32)
    return (tex - tex.min()) / (tex.max() - tex.min())


def fog(canvas, t, density, tint=(1, 1, 1), speed=(14, -4)):
    if density <= 0:
        return canvas
    tex = np.clip((noise_tex() - 0.35) * 1.8, 0, 1) ** 1.4
    th, tw = tex.shape
    out = np.zeros((H // 4, W // 4), np.float32)
    for k, (sx, sy, sc) in enumerate(((speed[0], speed[1], 1.0), (-speed[0] * 0.6, speed[1] * 1.7, 0.7))):
        ox, oy = int(t * sx + k * 173) % tw, int(t * sy + k * 311) % th
        rolled = np.roll(np.roll(tex, -oy, 0), -ox, 1)
        ch, cw = int(H // 4 * sc), int(W // 4 * sc)
        out += cv2.resize(rolled[:ch, :cw], (W // 4, H // 4)) * (0.6 if k else 1.0)
    f = cv2.resize(out, (W, H))[..., None] * density
    return screen(canvas, f * np.float32(tint))


@lru_cache(maxsize=None)
def motes(seed, n):
    rng = np.random.default_rng(seed)
    return dict(x=rng.uniform(0, 1, n), y=rng.uniform(0, 1, n), r=rng.uniform(0.5, 1.0, n),
                vx=rng.uniform(-0.012, 0.012, n), vy=rng.uniform(-0.03, -0.006, n),
                ph=rng.uniform(0, 6.28, n), fl=rng.uniform(0.6, 2.2, n), b=rng.uniform(0.3, 1, n))


def dust(canvas, t, amount, tint=(1, 1, 1), size=2.2, seed=11, n=110, dark=False):
    """Drifting motes; dark=True draws soot specks instead of lit dust."""
    if amount <= 0:
        return canvas
    p = motes(seed, n)
    lay = np.zeros((H // 2, W // 2), np.float32)
    xs = ((p["x"] + p["vx"] * t + 0.01 * np.sin(t * 0.7 + p["ph"])) % 1) * (W // 2)
    ys = ((p["y"] + p["vy"] * t) % 1) * (H // 2)
    br = p["b"] * (0.6 + 0.4 * np.sin(t * p["fl"] * 2 + p["ph"]))
    for x, y, r, b in zip(xs, ys, p["r"], br):
        cv2.circle(lay, (int(x), int(y)), max(1, int(r * size)), float(b), -1, cv2.LINE_AA)
    lay = cv2.resize(cv2.GaussianBlur(lay, (0, 0), 0.9), (W, H))[..., None]
    if dark:
        return canvas * (1 - np.clip(lay * amount, 0, 1))
    return canvas + lay * amount * np.float32(tint)


@lru_cache(maxsize=None)
def bokeh_set(seed=5, n=26):
    rng = np.random.default_rng(seed)
    return dict(x=rng.uniform(-0.1, 1.1, n), y=rng.uniform(-0.1, 1.1, n),
                r=rng.uniform(18, 95, n), b=rng.uniform(0.25, 1.0, n),
                vx=rng.uniform(-0.03, 0.03, n), vy=rng.uniform(-0.02, 0.02, n),
                ph=rng.uniform(0, 6.28, n))


def bokeh(canvas, t, amount, grow=1.0, tint=(1, 1, 1), seed=5, n=26):
    if amount <= 0:
        return canvas
    p = bokeh_set(seed, n)
    lay = np.zeros((H // 2, W // 2), np.float32)
    for x, y, r, b, vx, vy, ph in zip(p["x"], p["y"], p["r"], p["b"], p["vx"], p["vy"], p["ph"]):
        cx = ((x + vx * t) % 1.2 - 0.1) * W / 2
        cy = ((y + vy * t) % 1.2 - 0.1) * H / 2
        rr = int(r * grow / 2)
        v = b * (0.7 + 0.3 * math.sin(t * 1.3 + ph))
        cv2.circle(lay, (int(cx), int(cy)), rr, float(v * 0.55), -1, cv2.LINE_AA)
        cv2.circle(lay, (int(cx), int(cy)), rr, float(v), max(1, rr // 12), cv2.LINE_AA)
    lay = cv2.resize(cv2.GaussianBlur(lay, (0, 0), 2.2), (W, H))[..., None]
    return screen(canvas, lay * amount * np.float32(tint))


@lru_cache(maxsize=None)
def flare_sprites():
    S = 768
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    dx, dy = xx - S / 2, yy - S / 2
    r = np.hypot(dx, dy) / (S / 2)
    ang = np.arctan2(dy, dx)
    core = np.exp(-(r / 0.035) ** 2) + 0.45 * np.exp(-(r / 0.13) ** 2) + 0.18 * np.exp(-(r / 0.45) ** 2)
    rays = np.zeros_like(r)
    for k, (a0, ln, wd) in enumerate(((0.3, 0.95, 1.4), (1.35, 0.7, 1.1), (2.2, 0.85, 1.2),
                                      (0.3 + math.pi, 0.95, 1.4), (1.35 + math.pi, 0.7, 1.1),
                                      (2.2 + math.pi, 0.85, 1.2), (0.85, 0.5, 0.9), (2.9, 0.45, 0.9))):
        d = np.abs(np.sin(ang - a0)) * r * S / 2
        front = np.cos(ang - a0) > 0
        rays += front * np.exp(-(d / wd) ** 2) * np.exp(-r / (ln * 0.45))
    star = np.clip(core + 0.9 * rays, 0, None)
    L, T = 2400, 96
    sy, sx = np.mgrid[0:T, 0:L].astype(np.float32)
    streak = np.exp(-((sy - T / 2) / 3.5) ** 2) * np.exp(-np.abs(sx - L / 2) / 420)
    streak += 0.35 * np.exp(-((sy - T / 2) / 14) ** 2) * np.exp(-np.abs(sx - L / 2) / 260)
    ring = np.exp(-((r - 0.55) / 0.04) ** 2) * 0.25 + np.exp(-(r / 0.5) ** 2) * 0.08
    return star.astype(np.float32), streak.astype(np.float32), ring.astype(np.float32)


def flare(canvas, x, y, k, size=1.0, tint=(1.0, 0.97, 0.92), streak=1.0, ghosts=True):
    """Star flare + anamorphic streak + ghosts along the axis through frame centre."""
    if k <= 0.01:
        return canvas
    star, st, ring = flare_sprites()
    lay = np.zeros((H // 2, W // 2), np.float32)
    x2, y2 = x / 2, y / 2

    def stamp(spr, sc, cx, cy, amt):
        h, w = spr.shape
        M = np.float32([[sc, 0, cx - w / 2 * sc], [0, sc, cy - h / 2 * sc]])
        lay[:] += cv2.warpAffine(spr, M, (W // 2, H // 2)) * amt

    stamp(star, 0.8 * size, x2, y2, 1.0)
    if streak:
        stamp(st, 0.55 * size, x2, y2, 0.7 * streak)
    if ghosts:
        for f, sc, amt in ((-0.4, 0.18, 0.5), (-0.9, 0.32, 0.35), (0.35, 0.1, 0.6), (-1.35, 0.5, 0.25)):
            gx, gy = W / 4 + (x2 - W / 4) * f, H / 4 + (y2 - H / 4) * f
            stamp(ring, sc * size, gx, gy, amt)
    lay = cv2.resize(lay, (W, H))[..., None]
    return screen(canvas, lay * k * np.float32(tint))


@lru_cache(maxsize=None)
def sprocket_strip():
    """Film-strip perforations along the top and bottom edges (alpha, rgb)."""
    band = 30
    a = np.zeros((H, W), np.float32)
    col = np.zeros((H, W), np.float32)
    a[:band] = 1.0
    a[-band:] = 1.0
    for x in range(-10, W + 40, 62):
        for y0 in (7, H - band + 7):
            cv2.rectangle(col, (x, y0), (x + 30, y0 + 16), 0.2, -1)
    a = cv2.GaussianBlur(a, (0, 0), 1.0)
    col = cv2.GaussianBlur(col, (0, 0), 1.2)
    shade = np.zeros((H, W), np.float32)
    shade[band:band + 20] = np.linspace(0.6, 0, 20)[:, None]
    shade[H - band - 20:H - band] = np.linspace(0, 0.6, 20)[:, None]
    return a[..., None], col[..., None], shade[..., None]


def sprockets(canvas, amount, top=True, bottom=True, t=0.0):
    if amount <= 0:
        return canvas
    a, col, shade = sprocket_strip()
    mask = np.ones((H, 1, 1), np.float32)
    if not top:
        mask[:H // 2] = 0
    if not bottom:
        mask[H // 2:] = 0
    jitter = int(2 * math.sin(t * 17.0))
    a = np.roll(a, jitter, 0) * mask
    canvas = canvas * (1 - shade * amount * mask) * (1 - a * amount) + col * a * amount
    return canvas


def sparkle(canvas, x, y, size, k):
    """Four-point star glint."""
    if k <= 0.01:
        return canvas
    r = int(size * 3)
    x0, y0, x1, y1 = int(x) - r, int(y) - r, int(x) + r, int(y) + r
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        return canvas
    yy, xx = np.mgrid[-r:r, -r:r].astype(np.float32)
    s = np.exp(-(np.abs(xx) / (size * 0.9)) - (yy / 1.3) ** 2) + np.exp(-(np.abs(yy) / (size * 0.9)) - (xx / 1.3) ** 2)
    s += np.exp(-(xx ** 2 + yy ** 2) / (size * 0.35) ** 2)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    region = canvas[cy0:cy1, cx0:cx1]
    region[:] = screen(region, s[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0, None] * k)
    return canvas


def light_shafts(t, angle=-58, amount=1.0, seed=3):
    """Soft volumetric beams (HxWx1) raking down from the top-left."""
    yy, xx = grid()
    a = math.radians(angle)
    u = (xx * math.cos(a) + yy * math.sin(a))
    tex = noise_tex(seed, 64, 720, ((30, 1.0), (10, 0.5)))[32]
    idx = np.floor(u / 2.2 + t * 8).astype(np.int32) % 720
    beams = tex[idx] ** 3
    fall = np.clip(1 - (yy / H) * 0.9 - (xx / W) * 0.4, 0, 1)
    return (beams * fall * amount)[..., None]


# ----------------------------------------------------------------------------- text

FONT_FILES = {"bebas": ("BebasNeue-Regular.ttf", None), "cinzel": ("Cinzel[wght].ttf", 600),
              "cinzel_b": ("Cinzel[wght].ttf", 900), "cinzel_l": ("Cinzel[wght].ttf", 420),
              "corm": ("CormorantGaramond[wght].ttf", 500)}


@lru_cache(maxsize=None)
def font(key, size):
    fn, wt = FONT_FILES[key]
    f = ImageFont.truetype(os.path.join(FONTS, fn), size)
    if wt:
        try:
            f.set_variation_by_axes([wt])
        except Exception:
            pass
    return f


@lru_cache(maxsize=512)
def text_alpha(s, fkey, size, tracking=0.0):
    f = font(fkey, size)
    pad = int(size * 0.7)
    adv = [f.getlength(ch) for ch in s]
    width = sum(adv) + tracking * (len(s) - 1)
    asc, desc = f.getmetrics()
    im = Image.new("L", (int(width) + 2 * pad, asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(im)
    x = pad
    for ch, a in zip(s, adv):
        d.text((x, pad), ch, font=f, fill=255)
        x += a + tracking
    a = np.asarray(im).astype(np.float32) / 255
    ys, xs = np.nonzero(a > 0.3)
    ink = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, ys.min(), ys.max()) if len(xs) else (a.shape[1] / 2, a.shape[0] / 2, 0, 1)
    return a, ink


@lru_cache(maxsize=512)
def text_sprite(s, fkey, size, tracking=0.0, fill=(1, 1, 1), glow_col=(1, 1, 1), glow_amt=0.5,
                shadow=0.6, glow_sigma=0.18):
    """Premultiplied (rgb, alpha, ink centre). fill may be 'chrome'."""
    a, (icx, icy, top, bot) = text_alpha(s, fkey, size, tracking)
    h, w = a.shape
    if fill == "chrome":
        tt = np.clip((np.arange(h, dtype=np.float32) - top) / max(bot - top, 1), 0, 1)
        prof = np.interp(tt, [0, 0.42, 0.52, 0.8, 1], [1.0, 0.86, 0.5, 0.82, 0.66]).astype(np.float32)
        col = np.repeat(prof[:, None], w, 1)[..., None] * np.float32([0.9, 0.95, 1.0])
        b = cv2.GaussianBlur(a, (0, 0), max(size * 0.025, 0.8))
        gy = np.gradient(b, axis=0)
        col = np.clip(col + (-gy * 6)[..., None], 0, 1)
    else:
        col = np.broadcast_to(np.float32(fill), (h, w, 3))
    g = np.clip(cv2.GaussianBlur(a, (0, 0), size * glow_sigma) * 1.2, 0, 1) * glow_amt
    sh = np.clip(cv2.GaussianBlur(a, (0, 0), size * 0.35) * 2.0, 0, 1) * shadow
    rgb = np.zeros((h, w, 3), np.float32)
    al = sh.copy()
    rgb = rgb * (1 - g[..., None]) + g[..., None] * np.float32(glow_col)
    al = al * (1 - g) + g
    rgb = rgb * (1 - a[..., None]) + a[..., None] * col
    al = al * (1 - a) + a
    return rgb.astype(np.float32), al.astype(np.float32), (icx, icy)


def blit_affine(canvas, rgb, al, M, opacity=1.0, blur=0.0, blur_x=0.0, mode="over"):
    """Warp a premultiplied sprite with 2x3 M, only over its destination bbox."""
    if opacity <= 0.003:
        return canvas
    h, w = al.shape[:2]
    pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    d = pts @ M[:, :2].T + M[:, 2]
    pad = int(3 * max(blur, blur_x / 2) + 4)
    x0, y0 = int(max(d[:, 0].min() - pad, 0)), int(max(d[:, 1].min() - pad, 0))
    x1, y1 = int(min(d[:, 0].max() + pad, W)), int(min(d[:, 1].max() + pad, H))
    if x1 <= x0 or y1 <= y0:
        return canvas
    M2 = M.copy()
    M2[:, 2] -= (x0, y0)
    size = (x1 - x0, y1 - y0)
    r = cv2.warpAffine(rgb, M2, size, flags=cv2.INTER_LINEAR)
    a = cv2.warpAffine(al, M2, size, flags=cv2.INTER_LINEAR)
    if blur > 0.4:
        r, a = cv2.GaussianBlur(r, (0, 0), blur), cv2.GaussianBlur(a, (0, 0), blur)
    if blur_x > 2:
        k = int(blur_x) | 1
        r, a = cv2.blur(r, (k, 1)), cv2.blur(a, (k, 1))
    if a.ndim == 2:
        a = a[..., None]
    region = canvas[y0:y1, x0:x1]
    if mode == "screen":
        region[:] = screen(region, r * opacity)
    else:
        region *= 1 - a * opacity
        region += r * opacity
    return canvas


def sprite_matrix(anchor, cx, cy, scale=1.0, rot=0.0, sx=1.0):
    r = math.radians(rot)
    a, b = scale * math.cos(r), scale * math.sin(r)
    ax, ay = anchor
    return np.float32([[a * sx, -b, cx - (a * sx * ax - b * ay)],
                       [b * sx, a, cy - (b * sx * ax + a * ay)]])


def text(canvas, s, fkey, size, cx, cy, opacity=1.0, scale=1.0, rot=0.0, blur=0.0, blur_x=0.0,
         tracking=0.0, anchor="c", mode="over", **style):
    if opacity <= 0.003 or not s:
        return canvas
    rgb, al, (icx, icy) = text_sprite(s, fkey, size, tracking, **style)
    if anchor == "l":
        a, (_, _, _, _) = text_alpha(s, fkey, size, tracking)
        xs = np.nonzero(a.max(0) > 0.3)[0]
        icx = xs.min() if len(xs) else icx
    elif anchor == "r":
        a, _ = text_alpha(s, fkey, size, tracking)
        xs = np.nonzero(a.max(0) > 0.3)[0]
        icx = xs.max() if len(xs) else icx
    M = sprite_matrix((icx, icy), cx, cy, scale, rot)
    return blit_affine(canvas, rgb, al, M, opacity, blur, blur_x, mode)


def reveal(t, t_in, dur=0.3):
    """(opacity, blur, rise) for a soft focus-in."""
    k = ease_out(ramp(t, t_in, t_in + dur))
    return k, (1 - k) * 12, (1 - k)


def word_seq(t, words, hold_out=None):
    """Pick the current word from [(text, t_in)] (each replaces the previous)."""
    cur = None
    for w, ti in words:
        if t >= ti - 0.02:
            cur = (w, ti)
    return cur


def letters(canvas, t, s, fkey, size, cy, t0, step, tracking, opacity=1.0, **style):
    """Letter-by-letter reveal of a centred, tracked word."""
    a_full, _ = text_alpha(s, fkey, size, tracking)
    f = font(fkey, size)
    adv = [f.getlength(ch) for ch in s]
    total = sum(adv) + tracking * (len(s) - 1)
    x = C[0] - total / 2
    for i, ch in enumerate(s):
        k = ease_out(ramp(t, t0 + i * step, t0 + i * step + 0.35))
        if ch != " " and k > 0:
            text(canvas, ch, fkey, size, x + adv[i] / 2, cy - (1 - k) * 10, opacity * k,
                 blur=(1 - k) * 8, **style)
        x += adv[i] + tracking
    return canvas


# ----------------------------------------------------------------------------- shots
# Each shot fn(lt, d, t) -> rgb in [0,1] (text included). Post look is applied in frame().

def sh_hands(lt, d, t):
    """IS HE A LOST / EMBRACE - reaching hands, text in a slot of light."""
    cam = Cam(t)
    k = ease_in_out(lt / d)
    push = lerp(1.0, 1.1, k) + 0.12 * ease_in_out(ramp(t, 2.3, 4.6))
    kw = dict(fit="height", s=push, fx=0.5, fy=0.47, rot=lerp(-1.0, 1.5, k))
    bg_rgb, _ = layer("hands_jpeg.png", blur=16, cam=cam, depth=0.6, **kw)
    fg_rgb, fg_a = layer("hands_fg.png", cam=cam, **kw)
    # Crush the rocky backdrop to near black; the hands carry the light.
    c = mono(bg_rgb, (0.55, 0.57, 0.62)) * 0.12
    hands = mono(fg_rgb / np.maximum(fg_a, 1e-3), contrast=1.35, pivot=0.55)
    yy, xx = grid()
    warm = np.clip((xx - W * 0.45) / (W * 0.3), 0, 1)[..., None]
    gold = ease_out(ramp(t, 0.25, 0.7)) * (1 - ease_in_out(ramp(t, 1.4, 2.4)))
    tint = (1 - warm) * np.float32([0.94, 0.96, 1.0]) + warm * lerp(np.float32([1.0, 0.97, 0.94]), np.float32([1.25, 0.9, 0.48]), gold)
    top = np.clip(1.25 - yy / H * 1.1, 0.25, 1.2)[..., None]
    c = over(c, np.clip(hands * tint * top, 0, 1) * fg_a, fg_a)
    c *= ease_out(ramp(t, 0.12, 0.9)) * (1 - ease_in(ramp(t, 4.45, 5.15)) * 0.95)

    # Orange practical light bleeding in from the right while EMBRACE burns.
    leak = ease_in_out(ramp(t, 2.9, 3.5)) * (1 - ease_in_out(ramp(t, 4.3, 5.0)))
    if leak > 0:
        lx, ly = W * 0.93 + 30 * math.sin(t * 2), H * 0.2
        g = np.exp(-(((xx - lx) / 190) ** 2 + ((yy - ly) / 70) ** 2))[..., None]
        c = screen(c, g * leak * np.float32([1.0, 0.55, 0.22]) * 1.1)

    # --- the slot of light and its text
    cx, cy = C[0] + cam.dx * 0.8, H * 0.43 + cam.dy * 0.8
    phrase = [w for w, ti in (("IS", 0.0), ("HE", 0.28), ("A", 0.94), ("LOST", 1.46)) if t >= ti - 0.02]
    emb = ramp(t, 2.46, 2.62)
    rot = lerp(0, -7, ease_out(ramp(t, 2.46, 3.2))) + lerp(0, -4, ease_in_out(ramp(t, 3.6, 5.0)))
    sc = lerp(1.0, 1.35, ease_out(ramp(t, 2.46, 3.1))) + 0.15 * ease_in_out(ramp(t, 3.6, 5.2))
    fade = 1 - ease_in(ramp(t, 4.6, 5.2))
    bar_w = (70 + 38 * len(" ".join(phrase))) if emb < 1 else 470
    bar_w = lerp(bar_w, 470, emb)
    bar = slot_bar(int(bar_w), 74)
    M = sprite_matrix((bar[1].shape[1] / 2, bar[1].shape[0] / 2), cx, cy, sc, rot)
    blit_affine(c, bar[0], bar[1], M, 0.95 * fade * ease_out(ramp(t, 0.0, 0.2)))
    if emb < 1:
        text(c, " ".join(phrase), "bebas", 58, cx, cy + 2, (1 - emb) * fade, sc, rot,
             tracking=2, glow_amt=0.35, shadow=0.0)
    if emb > 0:
        red = ease_out(ramp(t, 3.52, 3.62))
        jit = env(t, [(3.55, 1.0)], 0.1)
        blur_x = 40 * jit + 18 * env(t, [(4.1, 1.0)], 0.12)
        ox = 14 * jit * math.sin(t * 90)
        if red > 0:
            text(c, "?", "cinzel_b", 118, cx + 8 * sc, cy - 4, red * fade * 0.95, sc, rot,
                 glow_amt=0.6, shadow=0.0, blur_x=blur_x * 0.5)
        col = tuple(np.float32(lerp(np.float32([1, 1, 1]), np.float32([0.95, 0.12, 0.12]), red)).tolist())
        gcol = tuple(np.float32(lerp(np.float32([1, 1, 1]), np.float32([1.0, 0.1, 0.1]), red)).tolist())
        text(c, "EMBRACE", "cinzel_b", 60, cx + ox, cy + 2, emb * fade, sc * lerp(1.25, 1.0, ease_out(emb)),
             rot, blur=(1 - emb) * 6, blur_x=blur_x, tracking=1, fill=col, glow_col=gcol,
             glow_amt=0.55 + 0.4 * red, shadow=0.0)
    c = c * ease_out(ramp(t, 0.0, 0.08))
    return c


@lru_cache(maxsize=None)
def slot_bar(w, h):
    """A dark glass slot with light spilling along its top and bottom lips."""
    pw, ph = w + 360, h + 160
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    cx, cy = pw / 2, ph / 2
    inside = ((np.abs(xx - cx) < w / 2) & (np.abs(yy - cy) < h / 2)).astype(np.float32)
    inside = cv2.GaussianBlur(inside, (0, 0), 2.0)
    along = np.exp(-((xx - cx) / (w * 0.42)) ** 4)
    lip = np.zeros_like(xx)
    for ly in (cy - h / 2, cy + h / 2):
        lip += np.exp(-((yy - ly) / 1.6) ** 2) * along
        lip += 0.35 * np.exp(-((yy - ly) / 9) ** 2) * along
    lip += 0.25 * np.exp(-((yy - cy) / (h * 0.9)) ** 2) * np.exp(-((xx - cx) / (w * 0.7)) ** 2) * (1 - inside)
    rgb = np.zeros((ph, pw, 3), np.float32)
    al = inside * 0.82
    rgb = rgb * (1 - inside[..., None]) + inside[..., None] * np.float32([0.035, 0.035, 0.04])
    lip = np.clip(lip, 0, 1)
    rgb = rgb * (1 - lip[..., None]) + lip[..., None] * np.float32([1.0, 0.98, 0.95])
    al = al * (1 - lip) + lip
    return rgb, al.astype(np.float32)


def sh_silhouette(lt, d, t):
    """SOMEWHERE ACROSS THE SEA OF - backlit Boruto, one word at a time."""
    cam = Cam(t)
    k = ease_in_out(lt / d)
    kw = dict(fit="width", s=lerp(1.05, 1.17, k), fx=0.52, fy=lerp(0.3, 0.26, k), rot=lerp(1.5, -1, k))
    bg, _ = layer("silhouette_bg.png", blur=10, cam=cam, depth=0.5, **kw)
    fg, fa = layer("silhouette_fg.png", cam=cam, **kw)
    c = mono(bg, (0.72, 0.78, 0.9), contrast=1.4) * 0.5
    # Light source just over his shoulder; the figure goes to silhouette with a hot rim.
    M = layer_matrix(img("silhouette_fg.png"), cam=cam, **kw)
    lx, ly = (M @ np.float32([0.47 * img("silhouette_fg.png").shape[1], 0.075 * img("silhouette_fg.png").shape[0], 1]))
    lx, ly = float(lx) - 60, float(ly) + 10
    yy, xx = grid()
    rad = np.exp(-(((xx - lx) ** 2 + (yy - ly) ** 2) / (2 * 260 ** 2)))[..., None]
    c = screen(c, rad * 0.55 * np.float32([0.9, 0.95, 1.0]))
    body = mono(fg / np.maximum(fa, 1e-3), (0.85, 0.9, 1.0), contrast=1.2) * 0.22
    sh = cv2.warpAffine(fa[..., 0], np.float32([[1, 0, 9], [0, 1, 7]]), (W, H))[..., None]
    rim = np.clip(fa - sh, 0, 1)
    rim = cv2.GaussianBlur(rim, (0, 0), 2.2)[..., None] * (0.35 + 1.4 * rad)
    c = over(c, body * fa, fa)
    c = screen(c, rim * 1.6 * np.float32([0.95, 0.97, 1.0]))
    pulse = 0.75 + 0.5 * env(t, [(WORDS[w], 1.0) for w in ("SOME", "ACROSS", "THE", "SEA", "OF")], 0.25)
    c = flare(c, lx, ly, 0.9 * pulse, size=1.1)
    c = bokeh(c, t, 0.28, 1.0, (0.85, 0.9, 1.0))
    # White bokeh burst that opens the shot.
    burst = ease_out(ramp(t, 5.78, 6.02)) * (1 - ease_in_out(ramp(t, 6.05, 6.75)))
    if burst > 0:
        c = bokeh(c, t * 3, burst * 1.6, 1.0 + 2.2 * ease_out(ramp(t, 5.78, 6.6)), seed=9, n=18)
        c = c + (1 - c) * burst * 0.75
    c *= ease_out(ramp(t, 5.72, 5.95))
    words = [("SOMEWHERE", WORDS["SOME"]), ("ACROSS", WORDS["ACROSS"]), ("THE", WORDS["THE"]),
             ("SEA", WORDS["SEA"]), ("OF", WORDS["OF"])]
    cur = word_seq(t, words)
    if cur:
        w, ti = cur
        if w == "SOMEWHERE":
            letters(c, t, w, "cinzel", 50, H * 0.64, ti, 0.045, 6, glow_amt=0.55, shadow=0.5)
        else:
            o, b, r = reveal(t, ti, 0.22)
            text(c, w, "cinzel", 54, C[0], H * 0.64 + r * 8, o, lerp(1.08, 1.0, o), blur=b,
                 tracking=6, glow_amt=0.55, shadow=0.5)
    return c


@lru_cache(maxsize=None)
def shard_geometry():
    rng = np.random.default_rng(21)
    pts = [(-200, -200), (W + 200, -200), (-200, H + 200), (W + 200, H + 200)]
    for x in np.linspace(-200, W + 200, 7):
        pts += [(x + rng.uniform(-60, 60), -200), (x + rng.uniform(-60, 60), H + 200)]
    for y in np.linspace(-200, H + 200, 5):
        pts += [(-200, y + rng.uniform(-60, 60)), (W + 200, y + rng.uniform(-60, 60))]
    for _ in range(26):
        pts.append((rng.uniform(0, W), rng.uniform(0, H)))
    sub = cv2.Subdiv2D((-400, -400, W + 800, H + 800))
    for p in pts:
        sub.insert((float(p[0]), float(p[1])))
    tris = []
    for tr in sub.getTriangleList():
        poly = tr.reshape(3, 2).astype(np.float32)
        if (poly < -350).any() or (poly[:, 0] > W + 350).any() or (poly[:, 1] > H + 350).any():
            continue
        cen = poly.mean(0)
        dn = np.hypot((cen[0] - C[0]) / (W / 2), (cen[1] - C[1]) / (H / 2))
        tris.append(dict(poly=poly, cen=cen, dn=dn, spin=rng.uniform(-1, 1),
                         tex=int(rng.integers(0, 3)), off=rng.uniform(-0.2, 0.2, 2),
                         z=rng.uniform(0.6, 1.4)))
    return tris


SHARD_TEX = ["sky_boruto_bg.png", "hands_jpeg.png", "silhouette_bg.png"]


def draw_shards(c, t, lt, alpha=1.0):
    """Edge shards linger and drift; centre shards blow out of the flash."""
    for s in shard_geometry():
        edge = s["dn"] > 0.78
        if edge:
            dist = 30 + 45 * lt + 90 * (1 - ease_out(lt / 0.5))
            ang = s["spin"] * (4 + 3 * lt)
            sc = 1 + 0.03 * lt * s["z"]
            op = alpha * (1 - ease_in(ramp(t, 11.3, 11.9)))
        else:
            e = ease_out(lt / 0.55, 2)
            dist = 900 * e * s["z"]
            ang = s["spin"] * 50 * e
            sc = 1 + 0.9 * e * s["z"]
            op = alpha * (1 - ease_in(lt / 0.5))
        if op <= 0.01:
            continue
        cen = s["cen"]
        dvec = (cen - C) / (np.linalg.norm(cen - C) + 1e-3)
        r = math.radians(ang)
        a, b = sc * math.cos(r), sc * math.sin(r)
        T = cen + dvec * dist
        M = np.float32([[a, -b, T[0] - (a * cen[0] - b * cen[1])],
                        [b, a, T[1] - (b * cen[0] + a * cen[1])]])
        poly = (s["poly"] @ M[:, :2].T + M[:, 2]).astype(np.int32)
        x0, y0 = max(poly[:, 0].min() - 2, 0), max(poly[:, 1].min() - 2, 0)
        x1, y1 = min(poly[:, 0].max() + 3, W), min(poly[:, 1].max() + 3, H)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.fillConvexPoly(mask, poly - (x0, y0), 1.0, cv2.LINE_AA)
        if edge:
            name = SHARD_TEX[s["tex"]]
            im = img(name)
            Mt = layer_matrix(im, fit="cover", s=1.25, fx=0.5 + s["off"][0], fy=0.5 + s["off"][1])
            Mt = np.vstack([M, [0, 0, 1]]) @ np.vstack([Mt, [0, 0, 1]])
            Mt = Mt[:2].astype(np.float32)
            Mt[:, 2] -= (x0, y0)
            tex = cv2.warpAffine(im, Mt, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_REFLECT)
            tex = tex[..., :3].astype(np.float32) / 255
            tex = mono(tex, (0.93, 0.96, 1.0), contrast=1.2) * 0.8 + 0.18
        else:
            tex = np.ones((y1 - y0, x1 - x0, 3), np.float32) * 0.96
        edge_line = np.zeros_like(mask)
        cv2.polylines(edge_line, [poly - (x0, y0)], True, 1.0, 2, cv2.LINE_AA)
        region = c[y0:y1, x0:x1]
        m = mask[..., None] * op
        region *= 1 - m
        region += tex * m
        region[:] = screen(region, edge_line[..., None] * 0.7 * op)
    return c


def sh_title(lt, d, t):
    """TIME - shattered glass around B/W Boruto; chrome name card with a callout line."""
    cam = Cam(t)
    k = ease_in_out(lt / d)
    zoom_in = ease_in_out(ramp(t, 11.42, 11.95))
    sky, _ = layer("sky_boruto_bg.png", blur=6, fit="cover", s=lerp(1.2, 1.3, k), fx=0.55, fy=0.4,
                   x=lerp(30, -30, k), cam=cam, depth=0.4)
    c = mono(sky, (0.9, 0.94, 1.0), contrast=1.3) * 0.85 + 0.08
    face, fa = layer("face_jpeg.png", fit="height", s=lerp(1.08, 1.16, k) * (1 + 0.5 * zoom_in),
                     fx=0.5, fy=0.47, x=lerp(250, 215, k), cam=cam, depth=1.0)
    face = mono(face, (0.92, 0.95, 1.0), contrast=1.35, pivot=0.45)
    yy, xx = grid()
    # Feather the still's edges into the cloud plate.
    fmask = np.clip((xx - (W * 0.18 + 250)) / 260, 0, 1)[..., None] * fa
    c = c * (1 - fmask) + face * fmask
    birds_x = lerp(W * 0.1, W * 0.5, k)
    put(c, "flock.png", fit="px", s=0.55, fx=0.5, fy=0.5, x=birds_x - C[0], y=-H * 0.33, alpha=0.8, cam=cam, depth=0.5)
    c = draw_shards(c, t, lt)
    c = flare(c, W * 0.22, H * 0.2, 0.7 + 0.3 * math.sin(t * 3), size=0.9, streak=0.8)
    flick = 1 - 0.75 * (env(t, [(10.97, 1.0)], 0.03) + env(t, [(11.37, 1.0)], 0.03))
    c *= flick

    # Name card (camera pushes into it before the flash).
    zs = 1 + 0.9 * zoom_in
    zx, zy = W * 0.3, H * 0.42

    def zp(x, y):
        return zx + (x - zx) * zs, zy + (y - zy) * zs

    o = ease_out(ramp(t, 9.15, 9.6))
    nx, ny = zp(W * 0.07, H * 0.37)
    text(c, "Boruto Uzumaki", "cinzel", 64, nx, ny, o, zs, anchor="l", fill="chrome",
         glow_col=(0.85, 0.92, 1.0), glow_amt=0.35, shadow=0.55, blur=(1 - o) * 8)
    line_k = ease_out(ramp(t, 9.35, 9.9))
    if line_k > 0:
        lay = np.zeros((H, W), np.float32)
        p0, p1 = zp(W * 0.07, H * 0.415), zp(W * 0.07 + W * 0.4 * line_k, H * 0.415)
        cv2.line(lay, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1.0, max(1, int(2 * zs)), cv2.LINE_AA)
        q0 = zp(W * 0.23, H * 0.43)
        q1 = zp(W * 0.23 + W * 0.2 * line_k, H * 0.43 + H * 0.12 * line_k)
        cv2.line(lay, (int(q0[0]), int(q0[1])), (int(q1[0]), int(q1[1])), 0.9, max(1, int(2 * zs)), cv2.LINE_AA)
        cv2.circle(lay, (int(q1[0]), int(q1[1])), int(6 * zs), 1.0, -1, cv2.LINE_AA)
        lay = lay + cv2.GaussianBlur(lay, (0, 0), 4) * 0.8
        c = screen(c, lay[..., None] * 0.9)
    o2 = ease_out(ramp(t, 9.75, 10.2))
    # Chrome needs a darker plate than the shattered glass to read.
    pc = zp(W * 0.5, H * 0.72)
    nc = zp(W * 0.2, H * 0.37)
    pool = np.exp(-(((xx - pc[0]) / (W * 0.3 * zs)) ** 2 + ((yy - pc[1]) / (H * 0.2 * zs)) ** 2))
    pool = np.maximum(pool, 0.8 * np.exp(-(((xx - nc[0]) / (W * 0.24 * zs)) ** 2
                                           + ((yy - nc[1]) / (H * 0.09 * zs)) ** 2)))
    c = c * (1 - pool[..., None] * 0.5 * max(o2, ramp(t, WORDS["TIME"] - 0.1, WORDS["TIME"])))
    for i, line in enumerate(("THE BOY THE", "WORLD FORGOT")):
        lx, ly = zp(W * 0.64, H * (0.6 + 0.075 * i))
        text(c, line, "cinzel", 38, lx, ly, o2, zs, anchor="r", fill="chrome", tracking=2,
             glow_col=(0.85, 0.92, 1.0), glow_amt=0.3, shadow=0.6, blur=(1 - o2) * 6)
    # Glitch plate flashing across the card.
    gk = env(t, [(9.35, 1.0), (9.62, 0.7)], 0.05)
    if gk > 0.05:
        x0, y0 = zp(W * 0.2, H * 0.46)
        cv2.rectangle(c, (int(x0), int(y0)), (int(x0 + 150 * zs), int(y0 + 30 * zs)), (gk, gk, gk), -1)
    tk = ramp(t, WORDS["TIME"], WORDS["TIME"] + 0.3)
    if tk > 0:
        s_in = 1 + 0.6 * (1 - ease_out_expo(tk))
        tx, ty = zp(W * 0.5, H * 0.83)
        text(c, "TIME", "cinzel", 96, tx, ty, ease_out(tk) * (1 - zoom_in), s_in * zs, fill="chrome",
             glow_col=(0.9, 0.95, 1.0), glow_amt=0.45, shadow=0.6, tracking=6,
             blur=(1 - ease_out(tk)) * 5)
    c = sprockets(c, 0.9 * (1 - zoom_in), bottom=False, t=t)
    c = defocus(c, 7 * zoom_in)
    return c


def sh_sky(lt, d, t):
    """A LOVE IMMORTAL / SUCH AS - Boruto in the clouds, then the negative."""
    cam = Cam(t, drift=0.7)
    k = ease_in_out(lt / d)
    kw = dict(fit="cover", s=lerp(1.08, 1.22, k), fx=lerp(0.42, 0.4, k), fy=lerp(0.5, 0.47, k))
    c = blank()
    put(c, "sky_boruto_bg.png", x=lerp(20, -50, k), cam=cam, depth=0.5, **kw)
    # 'a love immortal' is set between the sky and Boruto so it slips behind his face.
    tx, ty = W * 0.45, H * 0.47
    words = [("A", WORDS["A2"]), ("LOVE", WORDS["LOVE"]), ("IMMORTAL", WORDS["IMMORTAL"])]
    shown = " ".join(w for w, ti in words if t >= ti)
    out = 1 - ease_in(ramp(t, 14.45, 14.75))
    if shown:
        o = ease_out(ramp(t, WORDS["A2"], WORDS["A2"] + 0.3)) * out
        mask = blank()
        text(mask, shown, "cinzel", 60, tx, ty, 1.0, tracking=4, glow_amt=0.0, shadow=0.0,
             anchor="l", fill=(1, 1, 1))
        m = mask[..., :1]
        ink = np.float32([0.07, 0.17, 0.26])
        c = c * (1 - m * o) + ink * m * o
        band = m[int(ty - 44):int(ty + 38)][::-1]
        y0 = int(ty + 42)
        h = min(band.shape[0], H - y0)
        grad = np.linspace(0.35, 0.0, h, dtype=np.float32)[:, None, None]
        rm = cv2.GaussianBlur(band[:h], (0, 0), 2.0)[..., None] * grad * o
        c[y0:y0 + h] = c[y0:y0 + h] * (1 - rm) + ink * rm
    put(c, "sky_boruto_fg.png", x=lerp(20, -20, k), cam=cam, depth=1.1, **kw)
    c = np.clip((c - 0.04) * 1.08, 0, 1)
    c = saturate(c, 1.08)
    c = dust(c, t, 0.35, (1.0, 0.9, 0.95), size=2.0, seed=12, n=70)
    c = glow(c, 0.25, 0.7, (1.0, 0.92, 0.95))
    neg = ease_in_out(ramp(t, 13.72, 14.05))
    if neg > 0:
        inv = 1 - c
        inv = np.clip((inv - 0.5) * 1.25 + 0.5, 0, 1)
        inv = inv * np.float32([1.05, 1.0, 1.02])
        c = c * (1 - neg) + inv * neg
    words2 = [("SUCH", WORDS["SUCH"]), ("AS", 15.02)]
    cur = word_seq(t, words2)
    if cur:
        o, b, r = reveal(t, WORDS["SUCH"], 0.25)
        text(c, cur[0], "cinzel", 44, W * 0.2, H * 0.62 + r * 8, o * (1 - ease_in(ramp(t, 15.1, 15.3))),
             blur=b, tracking=4, glow_amt=0.5, shadow=0.4, anchor="l", glow_col=(0.7, 0.95, 1.0))
    c *= 1 - ease_in(ramp(t, 15.12, 15.3))
    c = c + (1 - c) * (1 - ease_out(ramp(t, 12.0, 12.35))) * np.float32([1.0, 0.93, 0.86])
    return c


def sh_mine(lt, d, t):
    """MINE - a hand reaching into the light for Sasuke's scratched headband."""
    cam = Cam(t, drift=0.6)
    k = ease_in_out(lt / d)
    hi = ease_in_out(ramp(t, 17.95, 18.05))
    yy, xx = grid()
    bx = C[0] + cam.dx * 0.3
    beam_w = lerp(90, 150, k) + 20 * math.sin(t * 2.3)
    beam = np.exp(-((xx - bx) / beam_w) ** 2) * np.clip(1.15 - yy / H, 0, 1)
    halo = np.exp(-(((xx - bx) / 380) ** 2 + ((yy - H * 0.3) / 300) ** 2))
    c = np.float32([0.02, 0.022, 0.03]) + (beam * 0.55 + halo * 0.35)[..., None] * np.float32([0.9, 0.94, 1.0])
    clouds, _ = layer("sky_boruto_bg.png", blur=3, fit="width", s=1.35, fx=0.5, fy=0.72,
                      x=lerp(40, -40, k), y=H * 0.45, cam=cam, depth=0.4)
    cl = mono(clouds, contrast=1.6, pivot=0.62)
    cmask = np.clip((yy - H * 0.62) / (H * 0.12), 0, 1)[..., None]
    c = screen(c, cl * cmask * 0.55 * (0.5 + halo[..., None] + beam[..., None]))
    c = fog(c, t, 0.18, (0.8, 0.85, 0.95))
    # Hand rises into frame.
    rise = ease_out(ramp(t, 15.3, 16.4))
    hand, ha = layer("hand_up_fg.png", fit="height", s=1.0, fx=0.44, fy=0.28, x=-15,
                     y=lerp(620, 210, rise) - 25 * k, rot=lerp(5, 1, k), cam=cam, depth=1.1)
    hand = mono(hand / np.maximum(ha, 1e-3), contrast=1.1) * (0.8 + 0.4 * np.clip(1 - yy / H, 0, 1))[..., None]
    c = over(c, hand * ha, ha)
    # Floating headband: bob, roll and a slow turn on its vertical axis.
    hb = img("headband_cut.png")
    turn = math.cos(t * 1.15 + 0.6)
    hx, hy = bx + 10 * math.sin(t * 0.8), H * 0.36 + 12 * math.sin(t * 1.3) + (1 - rise) * 60
    M = sprite_matrix((hb.shape[1] * 0.5, hb.shape[0] * 0.5), hx, hy, 0.24, 8 * math.sin(t * 0.9) - 6,
                      sx=0.55 + 0.45 * abs(turn))
    hrgb = hb[..., :3].astype(np.float32) / 255
    hal = hb[..., 3].astype(np.float32) / 255
    hrgb = mono(hrgb / np.maximum(hal[..., None], 1e-3), contrast=1.3) * (1.0 if turn > 0 else 0.7)
    blit_affine(c, (hrgb * hal[..., None]).astype(np.float32), hal, M, ease_out(ramp(t, 15.35, 15.9)))
    for i, (sx, sy, ph) in enumerate(((-150, -40, 0.0), (140, 30, 1.7), (-60, 90, 3.1), (90, -95, 4.2))):
        tw = max(0.0, math.sin(t * 2.4 + ph)) ** 3 * ease_out(ramp(t, 15.8, 16.3))
        c = sparkle(c, hx + sx, hy + sy, 7 + 3 * (i % 2), tw)
    # Gull crossing the beam.
    gk = ramp(t, 15.4, 18.4)
    flap = 0.55 + 0.45 * math.sin(t * 7.5)
    g = img("gull.png")
    Mg = sprite_matrix((g.shape[1] / 2, g.shape[0] / 2), lerp(W * 1.1, W * 0.25, gk), H * 0.13 + 25 * math.sin(gk * 5),
                       0.13, -8 + 6 * math.sin(t * 7.5))
    Mg[1, :2] *= flap
    Mg[1, 2] = (H * 0.13 + 25 * math.sin(gk * 5)) - flap * (0.13 * g.shape[0] / 2)
    grgb = g[..., :3].astype(np.float32) / 255
    gal = g[..., 3].astype(np.float32) / 255
    blit_affine(c, mono(grgb / np.maximum(gal[..., None], 1e-3), contrast=1.2) * gal[..., None], gal, Mg, 0.95)
    c = dust(c, t, 0.9, dark=True, size=2.6, seed=31, n=40)
    c = dust(c, t, 0.35, (0.9, 0.95, 1.0), size=1.6, seed=32, n=60)
    letters(c, t, "MINE", "cinzel", 60, H * 0.17, WORDS["MINE"], 0.13, 34, glow_amt=0.6, shadow=0.55,
            glow_col=(0.85, 0.92, 1.0))
    # Over-exposure into the high-key beat, then the black wipe.
    expo = ease_in(ramp(t, 17.72, 18.0))
    c = c + (1 - c) * expo * (1 - hi)
    if hi > 0:
        sky = np.float32([0.72, 0.84, 0.95]) * (1 - yy / H * 0.25)[..., None] + (beam * 0.3)[..., None]
        hk = over(sky.astype(np.float32), hand * ha * 1.05, ha)
        blit_affine(hk, (hrgb * hal[..., None] * 0.85).astype(np.float32), hal, M, 1.0)
        blit_affine(hk, mono(grgb / np.maximum(gal[..., None], 1e-3), (0.55, 0.6, 0.66)) * gal[..., None], gal, Mg, 1.0)
        g2 = sprite_matrix((g.shape[1] / 2, g.shape[0] / 2), W * 0.3 + 120 * lt, H * 0.25, 0.16, 5)
        blit_affine(hk, mono(grgb / np.maximum(gal[..., None], 1e-3), (0.5, 0.55, 0.62)) * gal[..., None], gal, g2, 1.0)
        c = c * (1 - hi) + hk * hi
        c = c + (1 - c) * (1 - ease_out(ramp(t, 18.0, 18.2))) * 0.8
    wipe = ramp(t, 18.28, 18.53)
    if wipe > 0:
        edge = lerp(-0.4, 1.2, ease_in_out(wipe)) * W
        m = np.clip((edge - xx) / 380, 0, 1)[..., None]
        c = c * (1 - m)
    return c


def sh_team(lt, d, t):
    """WILL COME TO - Team 7 under a moving sky."""
    cam = Cam(t, drift=0.7)
    k = ease_in_out(lt / d)
    kw = dict(fit="cover", s=lerp(1.1, 1.2, k), fx=lerp(0.5, 0.49, k), fy=lerp(0.5, 0.52, k))
    c = blank()
    put(c, "team7_bg.png", x=lerp(-30, 30, k), cam=cam, depth=0.5, **kw)
    put(c, "flock.png", fit="px", s=0.5, fx=0.5, fy=0.5, x=lerp(-380, -150, k), y=-300, alpha=0.5,
        cam=cam, depth=0.6)
    put(c, "team7_fg.png", x=lerp(-10, 10, k), cam=cam, depth=1.0, **kw)
    # Warm, lifted, slightly faded painterly grade.
    c = saturate(c, 0.82)
    c = c * np.float32([1.04, 0.98, 0.9]) + np.float32([0.03, 0.025, 0.035])
    yy, xx = grid()
    leak = np.exp(-(((xx - W * 0.05) / 520) ** 2 + ((yy + H * 0.05) / 420) ** 2))[..., None]
    c = screen(c, leak * 0.35 * np.float32([1.0, 0.8, 0.55]))
    c = dust(c, t, 0.25, (1.0, 0.95, 0.85), size=1.8, seed=41, n=60)
    # Reveal from black, right side first.
    rv = ease_in_out(ramp(t, 18.53, 19.15))
    m = np.clip((xx - lerp(W * 1.3, -W * 0.6, rv)) / 520, 0, 1)[..., None]
    c = c * m
    cur = word_seq(t, [("WILL", WORDS["WILL"]), ("COME", WORDS["COME"]), ("TO", WORDS["TO"])])
    if cur:
        o, b, r = reveal(t, cur[1], 0.22)
        text(c, cur[0], "cinzel", 40, W * 0.6, H * 0.83 + r * 6, o, blur=b, tracking=5,
             glow_amt=0.55, shadow=0.55, glow_col=(1.0, 0.95, 0.85))
    return c


GALLERY = [  # name, x, y (px from centre), height (px), rot, depth
    ("portrait_hashirama.png", -470, -250, 330, 7, 0.7),
    ("portrait_tobirama.png", 470, -265, 320, -9, 0.75),
    ("portrait_hiruzen.png", -500, 215, 360, -5, 1.25),
    ("portrait_minato.png", 505, 200, 370, 10, 1.2),
    ("naruto_portrait_jpeg.png", 0, -70, 560, -2, 1.0),
]


def gallery(t, pan=0.0, zoom=1.0, rot=0.0, dim=1.0, cam=NO_CAM):
    yy, xx = grid()
    wall = noise_tex(17, 540, 720, ((40, 1.0), (6, 0.35)))
    wall = cv2.resize(wall, (W, H))[..., None]
    c = (0.03 + 0.05 * wall) * np.float32([0.92, 0.95, 1.0]) * np.ones((H, W, 3), np.float32)
    shafts = light_shafts(t, amount=0.22)
    for name, x, y, h, r, dpt in GALLERY:
        im = img(name) if name.startswith("portrait_") else framed(name)
        z = zoom ** dpt
        s = h / im.shape[0] * z
        cx = C[0] + (x + pan * dpt) * z + cam.dx * dpt
        cy = C[1] + y * z + cam.dy * dpt
        rr = r + rot * dpt + cam.rot
        M = sprite_matrix((im.shape[1] / 2, im.shape[0] / 2), cx, cy, s, rr)
        # Drop shadow onto the wall, then the frame.
        Ms = M.copy()
        Ms[:, 2] += (18 * dpt, 22 * dpt)
        sh = np.ones(im.shape[:2], np.float32)
        blit_affine(c, np.zeros(im.shape[:2] + (3,), np.float32), sh, Ms, 0.55, blur=14)
        rgb = im[..., :3].astype(np.float32) / 255
        blit_affine(c, rgb * 0.85, np.ones(im.shape[:2], np.float32), M, 1.0)
    c = c * (0.55 + 0.9 * shafts)
    c = screen(c, shafts * 0.25 * np.float32([1.0, 0.95, 0.85]))
    return c * dim


def sh_gallery(lt, d, t):
    """TO BE - the Hokage portraits, drifting in a dark hall."""
    cam = Cam(t, drift=0.8)
    k = ease_in_out(lt / d)
    c = gallery(t, pan=lerp(50, -40, k), zoom=lerp(1.0, 1.08, k), rot=lerp(-2.5, 1.5, k), cam=cam)
    # Boruto seen from behind, in the foreground, looking up at them.
    fg, fa = layer("silhouette_fg.png", fit="height", s=0.95, fx=0.5, fy=0.2, x=lerp(-470, -440, k),
                   y=330, cam=cam, depth=1.6)
    body = mono(fg / np.maximum(fa, 1e-3), (0.8, 0.85, 0.95), contrast=1.2) * 0.2
    c = over(c, body * fa, fa)
    c = dust(c, t, 0.5, (1.0, 0.95, 0.85), size=1.7, seed=51, n=80)
    c = c * np.float32([1.0, 0.97, 0.92])
    to = ease_out(ramp(t, 21.40, 21.6))
    text(c, "TO", "cinzel_b", 110, W * 0.5 - 120, H * 0.82, to, lerp(1.2, 1.0, to), fill="chrome",
         glow_amt=0.35, shadow=0.7, tracking=4)
    bk = ramp(t, WORDS["BE"], WORDS["BE"] + 0.25)
    if bk > 0:
        text(c, "BE", "cinzel_b", 110, W * 0.5 + 125, H * 0.82, ease_out(bk),
             1 + 0.5 * (1 - ease_out_expo(bk)), fill="chrome", glow_amt=0.35, shadow=0.7, tracking=4)
    return c


def sh_portrait(lt, d, t):
    cam = Cam(t, drift=0.5)
    k = ease_out(lt / d, 2)
    im = framed("naruto_portrait_jpeg.png")
    c = gallery(t, pan=0, zoom=1.9 + 0.25 * k, rot=-1, dim=0.9)
    M = sprite_matrix((im.shape[1] / 2, im.shape[0] * 0.36), C[0] + cam.dx, C[1] + cam.dy,
                      lerp(1.02, 1.12, k) * H / im.shape[0] * 1.5, -1.5 + cam.rot)
    blit_affine(c, im[..., :3].astype(np.float32) / 255, np.ones(im.shape[:2], np.float32), M)
    c = c * (0.7 + 0.8 * light_shafts(t, amount=0.5))
    return np.clip(c * 1.1, 0, 1)


def sh_smear(lt, d, t):
    k = lt / d
    c = gallery(t, pan=lerp(500, -900, ease_in_out(k)), zoom=1.25, rot=lerp(4, -6, k), dim=0.8)
    return motion_blur_x(c, 240) * 0.8


def sh_bright(lt, d, t):
    k = ease_in_out(lt / d)
    c = blank()
    put(c, "father_son_clean.png", fit="cover", s=lerp(1.08, 1.16, k), fx=0.52, fy=0.55)
    c = saturate(c, 0.5)
    over_layer = blank()
    put(over_layer, "boruto_portrait_jpeg.png", fit="cover", s=lerp(1.2, 1.1, k), fx=0.45, fy=0.4)
    c = screen(c * 0.9, mono(over_layer) * 0.55)
    c = np.clip(c * 1.25 + 0.1, 0, 1)
    fl = (1 - ease_out(ramp(t, 23.93, 24.3)))
    c = c + (1 - c) * fl
    c = c + (1 - c) * ease_in(ramp(t, 24.5, 24.7))
    return c


def sh_fade(lt, d, t):
    v = lerp(1.0, 0.5, ease_in_out(ramp(t, 24.75, 25.15)))
    v = lerp(v, 0.0, ease_in_out(ramp(t, 25.15, 25.75)))
    return blank(v)


def sh_eternity(lt, d, t):
    c = blank(0.0)
    c = dust(c, t, 0.12, (0.9, 0.9, 1.0), size=1.4, seed=61, n=50)
    o = ease_in_out(ramp(t, 25.8, 26.8)) * (1 - ease_in_out(ramp(t, 28.5, 29.5)))
    k = ease_out(ramp(t, 25.8, 28.8), 2)
    s = lerp(1.0, 0.9, k)
    tr = int(round(lerp(150, 14, k)))
    text(c, "ETERNITY", "cinzel_l", 62, C[0], C[1], o, s, tracking=tr, fill=(0.88, 0.88, 0.9),
         glow_amt=0.35, shadow=0.0, glow_col=(0.8, 0.85, 1.0), blur=(1 - o) * 2)
    return c


SHOTS = [
    (0.00, 5.30, sh_hands, dict(glow=0.45, sprockets=0.9, grain=0.05)),
    (5.30, 9.00, sh_silhouette, dict(glow=0.55, sprockets=0.9, grain=0.05)),
    (9.00, 12.00, sh_title, dict(glow=0.4, grain=0.05)),
    (12.00, 15.30, sh_sky, dict(glow=0.2, grain=0.035)),
    (15.30, 18.53, sh_mine, dict(glow=0.55, grain=0.05)),
    (18.53, 21.40, sh_team, dict(glow=0.25, grain=0.035)),
    (21.40, 22.97, sh_gallery, dict(glow=0.4, grain=0.05)),
    (22.97, 23.37, sh_portrait, dict(glow=0.35, grain=0.05)),
    (23.37, 23.83, sh_smear, dict(glow=0.3, grain=0.05)),
    (23.83, 24.70, sh_bright, dict(glow=0.3, grain=0.04)),
    (24.70, 25.80, sh_fade, dict(glow=0.0, grain=0.04, vignette=False)),
    (25.80, DUR, sh_eternity, dict(glow=0.3, grain=0.03)),
]

FLASHES = [(9.03, 0.14, 0.28, (1.0, 1.0, 1.0)), (12.07, 0.2, 0.0, (1.0, 0.93, 0.86)),
           (23.93, 0.1, 0.0, (1.0, 0.97, 0.9))]


def shot_at(t):
    for i, s in enumerate(SHOTS):
        if s[0] <= t < s[1]:
            return i, s
    return len(SHOTS) - 1, SHOTS[-1]


def frame(t):
    i, (t0, t1, fn, fx) = shot_at(t)
    c = np.clip(fn(t - t0, t1 - t0, t), 0, None)
    for tp, rise, fall, col in FLASHES:
        if tp - rise <= t < tp:
            v = ease_in(ramp(t, tp - rise, tp))
        elif fall and tp <= t < tp + fall * 4:
            v = math.exp(-(t - tp) / fall)
        else:
            continue
        c = c + (np.float32(col) - c) * v
    c = glow(np.clip(c, 0, 1), fx.get("glow", 0.3))
    c = chroma(c, 1.2 + 9 * env(t, PUNCH, 0.08))
    if fx.get("sprockets"):
        c = sprockets(c, fx["sprockets"], t=t)
    if fx.get("vignette", True):
        c = c * vignette()
    c = c + grain_frames()[int(t * FPS) % 10] * fx.get("grain", 0.04)
    return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- output

def render_frame(n):
    return frame(n / FPS).tobytes()


def audio(out_path):
    fc = (f"[0:a]atrim={SONG_START}:{SONG_START + DUR},asetpts=PTS-STARTPTS,"
          f"afade=t=in:d=0.05,afade=t=out:st={DUR - 2.4}:d=2.4[a]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SONG, "-filter_complex", fc, "-map", "[a]",
                    "-ac", "2", "-ar", "48000", "-c:a", "aac", "-b:a", "256k", out_path], check=True)


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
                           "-pix_fmt", "yuv420p", silent], stdin=subprocess.PIPE)
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
