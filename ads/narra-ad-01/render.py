"""Narra ad 01 - "Written for you". Fully procedural 9:16 render.

No app UI and no photographic/stock sources: every image is generated here with
numpy/OpenCV/Pillow (shelves, paper, book, light paths, constellation, dust) and
all type is set in OFL fonts shipped in ./fonts.

  python3 render.py stills 0.5 2.6 ...    PNG frames -> OUT/stills
  python3 render.py video                 full MP4 (needs score.wav from audio.py)
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

from timeline import (BOOK_FORMED, COLLAPSE, DURATION, FPS, H, PICKS, SCENES, STRIKE,
                      TAP, TRANSITIONS, TYPE_END, TYPE_START, W)

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
OUT = os.environ.get("NARRA_OUT", "/tmp/narra_ad")
cv2.setNumThreads(1)


# ------------------------------------------------------------------ utilities

def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


BG = hexc("#070b11")
INK = hexc("#05070b")
CREAM = hexc("#f7f2e8")
GOLD = hexc("#e8b36d")
GOLD_HI = hexc("#f6dcab")
GOLD_LO = hexc("#b8844a")
MUTED = hexc("#949ba5")
COOL = hexc("#9fb2d6")
PURPLE = hexc("#b08ef6")


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def prog(t, t0, d):
    return clamp((t - t0) / d)


def eo3(p):
    return 1 - (1 - p) ** 3


def eo5(p):
    return 1 - (1 - p) ** 5


def ei3(p):
    return p ** 3


def eio3(p):
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def expo(p):
    return 1.0 if p >= 1 else 1 - 2 ** (-10 * p)


def back(p, s=1.7):
    p -= 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def lerp(a, b, p):
    return a + (b - a) * p


def smooth_noise(h, w, scale, seed):
    r = np.random.default_rng(seed)
    small = r.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


# ----------------------------------------------------------------------- type

FONT_FILES = {
    "display": "PlayfairDisplay[wght].ttf",
    "display_i": "PlayfairDisplay-Italic[wght].ttf",
    "serif": "Lora[wght].ttf",
    "serif_i": "Lora-Italic[wght].ttf",
    "sans": "DMSans[opsz,wght].ttf",
}


@lru_cache(maxsize=None)
def font(kind, size, wght):
    f = ImageFont.truetype(os.path.join(FONTS, FONT_FILES[kind]), size,
                           layout_engine=ImageFont.Layout.RAQM)
    vals = []
    for ax in f.get_variation_axes():
        name = ax.get("name", b"")
        name = name.decode() if isinstance(name, bytes) else str(name)
        if "eight" in name or name.lower().startswith("w"):
            vals.append(clamp(wght, ax["minimum"], ax["maximum"]))
        else:  # optical size
            vals.append(clamp(min(size, 40), ax["minimum"], ax["maximum"]))
    f.set_variation_by_axes(vals)
    return f


def _fill(alpha, color):
    h = alpha.shape[0]
    if isinstance(color, str) and color == "gold":
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        top, mid, bot = GOLD_HI, GOLD, GOLD_LO
        rgb = np.where(y < 0.5, lerp(top, mid, y * 2), lerp(mid, bot, (y - 0.5) * 2))
        rgb = np.broadcast_to(rgb, alpha.shape + (3,))
    else:
        rgb = np.broadcast_to(np.asarray(color, np.float32), alpha.shape + (3,))
    return np.dstack([rgb * alpha[..., None], alpha]).astype(np.float32)


@lru_cache(maxsize=4096)
def text_sprite(s, kind, size, wght, color="cream", tracking=0.0):
    """Premultiplied RGBA sprite; vertical center = middle of ascent+descent."""
    f = font(kind, size, wght)
    asc, desc = f.getmetrics()
    pad = int(size * 0.45) + 8
    if tracking:
        widths = [f.getlength(c) for c in s]
        total = sum(widths) + tracking * size * (len(s) - 1)
    else:
        total = f.getlength(s)
    im = Image.new("L", (int(total + 2 * pad), asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(im)
    if tracking:
        x = pad
        for c, cw in zip(s, widths):
            d.text((x, pad), c, font=f, fill=255, anchor="la")
            x += cw + tracking * size
    else:
        d.text((pad, pad), s, font=f, fill=255, anchor="la")
    a = np.asarray(im, np.float32) / 255.0
    col = {"cream": CREAM, "muted": MUTED, "ink": INK, "soft": CREAM * 0.85}.get(color, color)
    return _fill(a, col)


def text_width(s, kind, size, wght, tracking=0.0):
    f = font(kind, size, wght)
    if tracking:
        return sum(f.getlength(c) for c in s) + tracking * size * (len(s) - 1)
    return f.getlength(s)


@lru_cache(maxsize=1024)
def shadow_sprite(s, kind, size, wght, color, tracking, sigma):
    spr = text_sprite(s, kind, size, wght, color, tracking)
    p = int(sigma * 3)
    a = cv2.copyMakeBorder(spr[..., 3], p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    out = np.zeros(a.shape + (4,), np.float32)
    out[..., 3] = cv2.GaussianBlur(a, (0, 0), sigma)
    return out


def blit(dst, spr, cx, cy, scale=1.0, alpha=1.0, rot=0.0, blur=0.0, add=False):
    if alpha <= 0.004 or scale <= 0.01:
        return
    if scale < 0.6:
        f = max(scale / 0.8, 0.02)
        spr = cv2.resize(spr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        scale /= f
    if blur > 0.4:
        b = blur / scale
        k = int(b * 3) | 1
        p = k
        spr = cv2.copyMakeBorder(spr, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        spr = cv2.GaussianBlur(spr, (0, 0), b)
    h, w = spr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
    M[0, 2] += cx - w / 2
    M[1, 2] += cy - h / 2
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ M.T
    x0 = max(0, int(math.floor(corners[:, 0].min())))
    y0 = max(0, int(math.floor(corners[:, 1].min())))
    x1 = min(dst.shape[1], int(math.ceil(corners[:, 0].max())) + 1)
    y1 = min(dst.shape[0], int(math.ceil(corners[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    M[0, 2] -= x0
    M[1, 2] -= y0
    out = cv2.warpAffine(spr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    roi = dst[y0:y1, x0:x1]
    if add:
        roi += out[..., :3] * alpha
    else:
        roi *= 1 - out[..., 3:4] * alpha
        roi += out[..., :3] * alpha


def glint(spr, pos, width=0.12, strength=1.4):
    """Diagonal specular sweep across a sprite; pos in 0..1."""
    h, w = spr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx / w) + 0.35 * (yy / h)
    band = np.exp(-((u - lerp(-0.3, 1.5, pos)) / width) ** 2) * strength
    out = spr.copy()
    out[..., :3] += band[..., None] * spr[..., 3:4] * np.array([1.0, 0.95, 0.85], np.float32)
    return out


def kline(dst, t, text, kind, size, wght, color, cy, t_in, t_out=None, cx=W / 2,
          stagger=0.07, dur=0.5, style="rise", shadow=0.0, tracking=0.0, glint_at=None):
    """Kinetic line: per-word staggered reveal and blur-out exit. Returns x extents."""
    words = text.split(" ")
    f = font(kind, size, wght)
    space = f.getlength(" ") + (tracking * size if tracking else 0)
    widths = [text_width(wd, kind, size, wght, tracking) for wd in words]
    total = sum(widths) + space * (len(words) - 1)
    x = cx - total / 2
    for i, wd in enumerate(words):
        wc = x + widths[i] / 2
        x += widths[i] + space
        p = prog(t, t_in + i * stagger, dur)
        if p <= 0:
            continue
        if style == "slam":
            e = expo(p)
            sc, dy, bl, al = 1 + (1 - e) * 0.55, 0.0, (1 - e) * 16, min(1, p * 3)
        elif style == "fade":
            e = eo3(p)
            sc, dy, bl, al = 1 + (1 - e) * 0.04, 0.0, (1 - e) * 18, e
        else:
            e = eo5(p)
            sc, dy, bl, al = 1 + (1 - e) * 0.06, (1 - e) * size * 0.5, (1 - e) * 12, min(1, p * 2.2)
        if t_out is not None:
            q = prog(t, t_out + i * stagger * 0.4, 0.28)
            if q >= 1:
                continue
            q = ei3(q)
            al *= 1 - q
            bl += q * 18
            dy -= q * size * 0.35
            sc *= 1 + q * 0.05
        spr = text_sprite(wd, kind, size, wght, color, tracking)
        if glint_at is not None:
            gp = prog(t, glint_at + i * 0.08, 0.7)
            if 0 < gp < 1:
                spr = glint(spr, eio3(gp))
        if shadow > 0:
            sh = shadow_sprite(wd, kind, size, wght, color, tracking, size * 0.16)
            blit(dst, sh, wc, cy + dy + size * 0.05, sc,
                 al * shadow, blur=bl)
        blit(dst, spr, wc, cy + dy, sc, al, blur=bl)
    return cx - total / 2, cx + total / 2


# -------------------------------------------------------------------- effects

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
VIGNETTE = (1 - 0.55 * np.clip(((XX - W / 2) / (W * 0.75)) ** 2 + ((YY - H / 2) / (H * 0.62)) ** 2, 0, 1) ** 1.3)[..., None]
QH, QW = H // 4, W // 4
QYY, QXX = np.mgrid[0:QH, 0:QW].astype(np.float32)


def vgrad(top, bot, power=1.0):
    y = (np.linspace(0, 1, H, dtype=np.float32) ** power)[:, None, None]
    return np.broadcast_to(lerp(top, bot, y), (H, W, 3)).copy()


def glow(dst, cx, cy, radius, color, strength, power=2.0):
    """Additive radial light, computed at quarter res."""
    d = np.sqrt((QXX - cx / 4) ** 2 + (QYY - cy / 4) ** 2) / (radius / 4)
    g = np.clip(1 - d, 0, 1) ** power * strength
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    dst += g[..., None] * color


def splat(points, weights, scale=2, sigma=1.2, shape=(H, W)):
    """Accumulate weighted points into a blurred float map (at 1/scale res)."""
    h, w = shape[0] // scale, shape[1] // scale
    buf = np.zeros((h, w), np.float32)
    if len(points):
        xs = (points[:, 0] / scale).astype(np.int32)
        ys = (points[:, 1] / scale).astype(np.int32)
        m = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        np.add.at(buf, (ys[m], xs[m]), weights[m])
    if sigma > 0:
        buf = cv2.GaussianBlur(buf, (0, 0), sigma)
    return cv2.resize(buf, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


class Dust:
    def __init__(self, n, seed, speed=(-8, -30), size=1.0):
        r = np.random.default_rng(seed)
        self.p = r.random((n, 2)).astype(np.float32) * [W, H]
        self.z = r.uniform(0.25, 1.0, n).astype(np.float32)
        self.v = np.stack([r.uniform(-1, 1, n) * speed[0], r.uniform(0.3, 1, n) * speed[1]], 1)
        self.ph = r.random(n) * 6.28
        self.size = size

    def draw(self, dst, t, color=GOLD_HI, strength=1.0, parallax=(0, 0)):
        pos = self.p + self.v * self.z[:, None] * t
        pos[:, 0] += np.sin(t * 0.9 + self.ph) * 12 * self.z + parallax[0] * self.z
        pos[:, 1] += parallax[1] * self.z
        pos[:, 0] %= W
        pos[:, 1] %= H
        tw = 0.55 + 0.45 * np.sin(t * 3.1 + self.ph * 3)
        near = self.z > 0.8
        fine = splat(pos[~near], (tw * self.z * 2.2)[~near].astype(np.float32), 2, 0.9)
        bokeh = splat(pos[near], (tw * 30)[near].astype(np.float32), 4, 4.5 * self.size)
        dst += (fine * 0.9 + bokeh * 0.5)[..., None] * color * strength


def bloom(img, thr=0.62, amount=0.4):
    small = cv2.resize(img, (QW, QH), interpolation=cv2.INTER_AREA)
    b = np.clip(small - thr, 0, None)
    b = cv2.GaussianBlur(b, (0, 0), 6) + 0.6 * cv2.GaussianBlur(b, (0, 0), 22)
    return img + cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR) * amount


def motion_blur(img, length, axis):
    k = int(length) | 1
    if k < 3:
        return img
    return cv2.blur(img, (k, 1) if axis == 0 else (1, k))


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def zoom(img, s):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def radial_blur(img, amount, steps=6):
    acc = img.copy()
    for i in range(1, steps):
        acc += zoom(img, 1 + amount * i / steps)
    return acc / steps


def warp_quad(dst, tex, quad, alpha=1.0, add=False):
    """Map an RGBA (premultiplied) texture onto a projected quad (TL,TR,BR,BL)."""
    h, w = tex.shape[:2]
    quad = np.asarray(quad, np.float32)
    x0 = max(0, int(quad[:, 0].min()) - 1)
    y0 = max(0, int(quad[:, 1].min()) - 1)
    x1 = min(W, int(quad[:, 0].max()) + 2)
    y1 = min(H, int(quad[:, 1].max()) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, quad - np.float32([x0, y0]))
    out = cv2.warpPerspective(tex, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    roi = dst[y0:y1, x0:x1]
    if add:
        roi += out[..., :3] * alpha
    else:
        roi *= 1 - out[..., 3:4] * alpha
        roi += out[..., :3] * alpha


def rot(rx, ry, rz=0.0):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project(P, F=1100.0, cx=W / 2, cy=H / 2):
    P = np.atleast_2d(P)
    z = np.maximum(P[:, 2], 1e-3)
    return np.stack([cx + F * P[:, 0] / z, cy + F * P[:, 1] / z], 1)


def rgba(rgb, a=None):
    rgb = rgb.astype(np.float32)
    if a is None:
        a = np.ones(rgb.shape[:2], np.float32)
    return np.dstack([rgb * a[..., None], a]).astype(np.float32)


# ------------------------------------------------------------ static textures

def make_shelves(TW=5800, TH=1352, seed=4):
    r = np.random.default_rng(seed)
    tex = np.zeros((TH, TW, 3), np.float32)
    tex[:] = hexc("#0a0705")
    pal = [hexc(c) for c in ["#6a2420", "#2b3d5e", "#25463a", "#7a5530", "#40305a", "#8a6a3c",
                             "#223040", "#5a2236", "#665a48", "#303034", "#7c4628", "#1e3a3a"]]
    row_h, board = 225, 20
    for y0 in range(0, TH - row_h + 1, row_h):
        yb = y0 + row_h - board
        wood = np.linspace(1.0, 0.45, board, dtype=np.float32)[:, None, None] * hexc("#3a2616")
        tex[yb:yb + board] = wood
        tex[yb:yb + 2] = hexc("#7a5636")
        x = int(r.integers(0, 30))
        while x < TW - 70:
            if r.random() < 0.035:
                x += int(r.integers(30, 90))
                continue
            bw = int(r.integers(20, 56))
            bh = int(r.integers(int(row_h * 0.58), row_h - board - 10))
            col = pal[r.integers(len(pal))] * r.uniform(0.55, 1.1)
            xs = np.linspace(-1, 1, bw, dtype=np.float32)
            shade = (1 - 0.6 * xs ** 2)[None, :, None]
            vy = np.linspace(0.85, 1.0, bh, dtype=np.float32)[:, None, None]
            block = np.broadcast_to(col, (bh, bw, 3)) * shade * vy
            block = block.copy()
            if r.random() < 0.7:
                for _ in range(int(r.integers(1, 4))):
                    yy = int(r.integers(8, max(9, bh - 14)))
                    block[yy:yy + 3] = GOLD * 0.75 * shade[0]
            if r.random() < 0.55:
                a0, a1 = int(bh * r.uniform(0.25, 0.35)), int(bh * r.uniform(0.5, 0.65))
                block[a0:a1, bw // 2 - 2:bw // 2 + 2] = GOLD * 0.55
            tex[yb - bh:yb, x:x + bw] = block
            x += bw + int(r.integers(0, 4))
    grain = smooth_noise(TH, TW, 3, seed + 1)[..., None] * 0.18 + 0.91
    return (tex * grain).astype(np.float32)


def make_paper(w=1000, h=1300, seed=9):
    base = np.broadcast_to(hexc("#efe5d0"), (h, w, 3)).astype(np.float32).copy()
    n = smooth_noise(h, w, 2, seed) * 0.05 + smooth_noise(h, w, 40, seed + 1) * 0.03
    fib = cv2.GaussianBlur(np.random.default_rng(seed).random((h, w)).astype(np.float32), (0, 0), 0.7)
    base *= (0.9 + n - 0.03 * fib)[..., None]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = np.clip(((xx - w / 2) / (w / 2)) ** 6 + ((yy - h / 2) / (h / 2)) ** 6, 0, 1)
    base *= (1 - 0.35 * edge)[..., None]
    base[..., 2] *= 0.94
    return base


def make_cover(w=800, h=1136):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    leather = lerp(hexc("#2c1813"), hexc("#140b08"), (yy / h)[..., None])
    n = smooth_noise(h, w, 3, 21)[..., None] * 0.22 + smooth_noise(h, w, 60, 22)[..., None] * 0.25
    leather = leather * (0.8 + n)
    im = Image.fromarray((np.clip(leather, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    g = (232, 179, 109)
    gs = (90, 60, 30)
    for off, col in ((3, gs), (0, g)):
        d.rectangle([40 + off, 40 + off, w - 40 + off, h - 40 + off], outline=col, width=4)
        d.rectangle([58 + off, 58 + off, w - 58 + off, h - 58 + off], outline=col, width=2)
        for (cx, cy) in ((58, 58), (w - 58, 58), (58, h - 58), (w - 58, h - 58)):
            d.polygon([(cx + off, cy - 16 + off), (cx + 16 + off, cy + off), (cx + off, cy + 16 + off),
                       (cx - 16 + off, cy + off)], fill=col)
        ex, ey = w / 2 + off, h * 0.38 + off
        for rr in (150, 132):
            d.ellipse([ex - rr, ey - rr, ex + rr, ey + rr], outline=col, width=3 if rr == 150 else 2)
        for k in range(16):
            a = k * math.pi / 8
            r0, r1 = (20, 128) if k % 2 == 0 else (30, 86)
            d.line([(ex + r0 * math.cos(a), ey + r0 * math.sin(a)),
                    (ex + r1 * math.cos(a), ey + r1 * math.sin(a))], fill=col, width=3 if k % 2 == 0 else 2)
        d.ellipse([ex - 14, ey - 14, ex + 14, ey + 14], fill=col)
        ft = font("display", 74, 600)
        title = "YOUR STORY"
        tw = sum(ft.getlength(c) for c in title) + 0.14 * 74 * (len(title) - 1)
        x = w / 2 - tw / 2 + off
        for c in title:
            d.text((x, h * 0.70 + off), c, font=ft, fill=col, anchor="lm")
            x += ft.getlength(c) + 0.14 * 74
        d.line([(w / 2 - 90 + off, h * 0.77 + off), (w / 2 + 90 + off, h * 0.77 + off)], fill=col, width=2)
        d.text((w / 2 + off, h * 0.82 + off), "Book One", font=font("serif_i", 44, 450), fill=col, anchor="mm")
    a = np.asarray(im, np.float32) / 255.0
    # raised foil sheen
    foil = (np.abs(a - np.array(g) / 255.0).sum(2) < 0.25).astype(np.float32)
    a = a * (1 + 0.25 * foil[..., None] * (0.5 + 0.5 * np.sin(xx / 40 + yy / 70))[..., None])
    return rgba(a)


def make_page_edge(w, h, horizontal=False):
    """Stacked page edges (cream with fine lines)."""
    base = np.broadcast_to(hexc("#e6d9bd"), (h, w, 3)).astype(np.float32).copy()
    r = np.random.default_rng(5)
    if horizontal:
        lines = r.random(h).astype(np.float32)[:, None]
    else:
        lines = r.random(w).astype(np.float32)[None, :]
    base *= (0.82 + 0.18 * lines)[..., None] if lines.ndim == 2 else 1
    return rgba(base)


def make_spine(w, h):
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :, None]
    s = lerp(hexc("#2c1813"), hexc("#140b08"), yy) * (1 - 0.5 * xs ** 2)
    s = np.broadcast_to(s, (h, w, 3)).copy()
    for f in (0.12, 0.15, 0.85, 0.88):
        y = int(f * h)
        s[y:y + 4] = GOLD * 0.8
    return rgba(s)


def make_nebula(seed=31):
    n1 = smooth_noise(H, W, 220, seed)
    n2 = smooth_noise(H, W, 90, seed + 1)
    neb = np.clip(n1 * 0.7 + n2 * 0.5 - 0.45, 0, 1) ** 1.6
    col = lerp(PURPLE * 0.35, GOLD * 0.25, smooth_noise(H, W, 300, seed + 2)[..., None])
    return (neb[..., None] * col * 0.55).astype(np.float32)


TEX = {}


def init():
    if TEX:
        return
    TEX["shelves"] = make_shelves()
    TEX["paper"] = make_paper()
    TEX["cover"] = make_cover()
    TEX["pages_side"] = make_page_edge(120, 1136)
    TEX["pages_top"] = make_page_edge(800, 120, horizontal=True)
    TEX["spine"] = make_spine(120, 1136)
    TEX["nebula"] = make_nebula()
    TEX["dust_a"] = Dust(380, 1)
    TEX["dust_b"] = Dust(260, 2, speed=(-6, -14))
    TEX["dust_end"] = Dust(420, 3, speed=(-6, -22), size=1.3)
    TEX["rays"] = smooth_noise(1, 720, 9, 44)[0]


# --------------------------------------------------------------------- scenes

def library(t):
    """Infinite library corridor; camera dollies forward, decelerating."""
    tex = TEX["shelves"]
    F, X0, Y0, zn, zf = 1000.0, 1.0, 2.6, 0.32, 14.0
    Wc = int(260 * (zf - zn))
    off = 1150 * (1 - math.exp(-t / 0.6)) + 160 * t
    off = int(min(off, tex.shape[1] - Wc - 1))
    cam_y = 0.12 * math.sin(t * 0.7)
    sway = 0.05 * math.sin(t * 0.9)
    img = vgrad(hexc("#0b0806"), hexc("#050403"))
    z = zn + np.arange(Wc, dtype=np.float32) / Wc * (zf - zn)
    zt = z + off / 260.0
    lamps = 0.55 + 0.75 * np.exp(-(((zt % 2.4) - 1.2) ** 2) / 0.12)
    fog = (np.exp(-(z - zn) / 5.0) * lamps)[None, :, None]
    fog_col = hexc("#1a120b")
    for side in (-1, 1):
        crop = tex[:, off:off + Wc] if side < 0 else tex[::-1, ::-1][:, 300 + off:300 + off + Wc][::-1]
        crop = crop * fog + fog_col * (1 - fog)
        crop = cv2.resize(crop, (Wc // 2, crop.shape[0] // 2), interpolation=cv2.INTER_AREA)
        if side > 0:
            pass
        x = side * X0 + sway
        P = np.array([[x, -Y0 - cam_y, zn], [x, -Y0 - cam_y, zf], [x, Y0 - cam_y, zf], [x, Y0 - cam_y, zn]])
        q = project(P, F)
        warp_quad(img, rgba(crop), q)
    # floor reflection sheen + far glow (the end of the corridor)
    vx, vy = W / 2 + F * sway / zf, H / 2
    glow(img, vx, vy, 520, hexc("#c58a4a"), 0.22, 2.2)
    glow(img, vx, vy, 180, hexc("#f2c88a"), 0.25, 2.0)
    # depth of field: center (far) is soft, edges (near) slightly soft too
    blurred = cv2.GaussianBlur(img, (0, 0), 5)
    d = np.clip(1 - np.sqrt(((XX - vx) / 420) ** 2 + ((YY - vy) / 620) ** 2), 0, 1)[..., None]
    img = img * (1 - d) + blurred * d
    return img


def scene_hook(t):
    img = library(t)
    # darken the middle band so type reads over the shelves
    band = np.exp(-((YY - 900) / 330) ** 2)[..., None]
    img *= 1 - 0.55 * band
    TEX["dust_a"].draw(img, t, strength=0.5)
    kline(img, t, "Every story", "display", 150, 700, "cream", 820, -0.14,
          t_out=1.68, stagger=0.09, dur=0.32, style="slam", shadow=0.85)
    kline(img, t, "you’ve ever read", "serif_i", 100, 500, "gold", 985, 0.42,
          t_out=1.70, stagger=0.1, dur=0.45, shadow=0.85)
    return img


def scene_someone_else(t):
    img = library(t)
    drain = eio3(prog(t, 1.8, 0.8))
    lum = img @ np.array([0.3, 0.55, 0.15], np.float32)
    img = lerp(img, np.repeat(lum[..., None], 3, 2) * 0.9, drain) * (1 - 0.55 * drain)
    band = np.exp(-((YY - 900) / 330) ** 2)[..., None]
    img *= 1 - 0.55 * band
    TEX["dust_a"].draw(img, t, color=MUTED, strength=0.35 * (1 - drain * 0.6))
    kline(img, t, "was written for", "serif_i", 96, 500, "cream", 800, 1.84,
          t_out=2.86, stagger=0.08, dur=0.45, shadow=0.85)
    x0, x1 = kline(img, t, "someone else.", "display", 140, 700, "muted", 965, 2.02,
                   t_out=2.88, stagger=0.1, dur=0.4, shadow=0.85)
    p = eo3(prog(t, STRIKE, 0.22))
    if p > 0:
        q = 1 - ei3(prog(t, 2.88, 0.2))
        y = 978
        xa, xb = x0 - 14, lerp(x0 - 14, x1 + 14, p)
        m = np.zeros((H, W), np.uint8)
        cv2.line(m, (int(xa), y), (int(xb), y - 6), 255, 7, cv2.LINE_AA)
        a = (m.astype(np.float32) / 255)[..., None] * q
        img[:] = img * (1 - a) + GOLD * a
    return img


def scene_for_you(t):
    lt = t - 3.0
    img = vgrad(hexc("#06080c"), hexc("#0b0a0a"))
    breathe = eo3(prog(t, 3.0, 1.0))
    glow(img, W / 2, 1060, 900, hexc("#a8743c"), 0.18 * breathe, 2.2)
    TEX["dust_b"].draw(img, t, strength=0.5 + 0.6 * prog(t, 3.4, 0.4))
    kline(img, t, "This one", "serif_i", 96, 500, "cream", 640, 3.04, stagger=0.08, dur=0.45)
    kline(img, t, "is written for", "serif_i", 96, 500, "cream", 760, 3.16, stagger=0.08, dur=0.45)
    p = prog(t, 3.40, 0.32)
    if p > 0:
        spr = text_sprite("you.", "display_i", 330, 800, "gold")
        gp = prog(t, 3.62, 0.6)
        if 0 < gp < 1:
            spr = glint(spr, eio3(gp), 0.1, 1.8)
        e = expo(p)
        sc = 1 + (1 - e) * 0.7
        flash = math.exp(-max(0.0, t - 3.42) / 0.18) * min(1, p * 4)
        glow(img, W / 2, 1060, 700, GOLD, 0.35 * flash + 0.1, 2.0)
        blit(img, spr, W / 2, 1060, sc * (1 + 0.03 * lt), min(1, p * 3), blur=(1 - e) * 14)
    return img


TASTES = ["Fantasy", "Romance", "Thriller", "Sci-fi", "Cultivation", "LitRPG", "Apocalypse",
          "Villain stories", "Progression", "Fast pacing", "First person",
          "Dark tone", "Enemies to lovers", "Found family", "Mystery", "Magic academy",
          "Second chances", "Dragons", "Revenge", "Rivals", "Space opera",
          "Heists", "Court intrigue", "Underdog", "Witty banter", "Epic battles"]
PICKED = [("Dark fantasy", 905), ("Slow burn", 1040), ("Morally grey lead", 1175)]


def taste_field():
    r = np.random.default_rng(12)
    n = len(TASTES)
    pos = np.stack([r.uniform(-1.5, 1.5, n), r.uniform(-0.25, 1.9, n), r.uniform(1.2, 7.0, n)], 1)
    return pos


TASTE_POS = taste_field()


def scene_tastes(t):
    lt = t - 4.2
    img = vgrad(hexc("#070a12"), hexc("#0b0a10"))
    img += TEX["nebula"] * (0.7 + 0.3 * math.sin(lt * 0.8))
    glow(img, W / 2, 1050, 800, PURPLE, 0.06, 2.0)
    TEX["dust_b"].draw(img, t, color=hexc("#d8c8ff"), strength=0.35, parallax=(0, -lt * 60))
    collapse = eio3(prog(t, COLLAPSE, 0.42))
    order = np.argsort(-((TASTE_POS[:, 2] - 1.6 * lt - 0.2) % 6.0))
    for i in order:
        x, y, z0 = TASTE_POS[i]
        z = 0.6 + (z0 - 1.6 * lt - 0.6) % 6.0
        P = np.array([[x * (1 - collapse), y * (1 - collapse) + 0.25 * collapse, z]])
        sx, sy = project(P, 900, W / 2, 1020)[0]
        sc = 2.1 / z
        depth_blur = abs(z - 2.6) * 2.4
        al = clamp((6.4 - z) / 1.5) * clamp((z - 0.7) / 0.6) * 0.7 * (1 - collapse)
        if 860 < sy < 1240 and t > PICKS[0]:
            al *= 0.45
        al *= prog(t, 4.16 + 0.01 * i, 0.25)
        spr = text_sprite(TASTES[i], "sans", 64, 500, "muted")
        blit(img, spr, sx, sy, sc, al, blur=min(depth_blur, 14))
    for k, (word, y) in enumerate(PICKED):
        tp = PICKS[k]
        p = prog(t, tp, 0.5)
        if p <= 0:
            continue
        e = eo5(p)
        tx, ty = lerp(W / 2, W / 2, collapse), lerp(y, 1040, collapse)
        sx = lerp((-260, 300, -200)[k] + W / 2, tx, e)
        sy = lerp(y + (-140, 120, 160)[k], ty, e)
        sc = lerp(0.35, 1.0, e) * (1 - 0.85 * collapse)
        spr = text_sprite(word, "serif_i", 92, 500, "gold")
        bl = (1 - e) * 16 + collapse * 10
        glow(img, sx, sy, 380, GOLD, 0.12 * math.exp(-max(0.0, t - tp) / 0.3), 2.0)
        sh = shadow_sprite(word, "serif_i", 92, 500, "gold", 0.0, 16)
        blit(img, sh, sx, sy + 6, sc, 0.8 * min(1, p * 2) * (1 - collapse ** 3), blur=bl)
        blit(img, spr, sx, sy, sc, min(1, p * 2) * (1 - collapse ** 3), blur=bl)
        if e > 0.6 and collapse < 0.2:
            w = text_width(word, "serif_i", 92, 500) * sc
            ul = eo3(prog(t, tp + 0.25, 0.3))
            m = np.zeros((H, W), np.uint8)
            cv2.line(m, (int(sx - w / 2), int(sy + 58)), (int(sx - w / 2 + w * ul), int(sy + 58)), 255, 2, cv2.LINE_AA)
            img[:] += (m.astype(np.float32) / 255)[..., None] * GOLD * 0.8 * (1 - collapse * 5)
    if collapse > 0:
        s = math.sin(collapse * math.pi) ** 0.7
        glow(img, W / 2, 1040, 300 + 500 * collapse, GOLD_HI, 1.1 * collapse ** 2, 2.5)
        glow(img, W / 2, 1040, 90, CREAM, 2.0 * s * collapse, 1.5)
    kline(img, t, "Tell Narra", "display", 124, 700, "cream", 420, 4.32, t_out=6.9,
          stagger=0.09, dur=0.5, shadow=0.6)
    kline(img, t, "what you love.", "serif_i", 104, 500, "gold", 565, 4.55, t_out=6.95,
          stagger=0.08, dur=0.5, shadow=0.6)
    return img


PAGE_LINES = [
    "You were the only one awake",
    "when the first star fell.",
    "By morning, the whole kingdom",
    "would know your name.",
]


def page_texture(t):
    paper = TEX["paper"]
    ph, pw = paper.shape[:2]
    im = Image.new("L", (pw, ph), 0)
    gold_im = Image.new("L", (pw, ph), 0)
    d = ImageDraw.Draw(im)
    dg = ImageDraw.Draw(gold_im)
    head = prog(t, 7.3, 0.4)
    fc = font("sans", 30, 600)
    label = "CHAPTER ONE"
    tw = sum(fc.getlength(c) for c in label) + 0.3 * 30 * (len(label) - 1)
    x = pw / 2 - tw / 2
    for c in label:
        dg.text((x, 150), c, font=fc, fill=int(255 * head), anchor="lm")
        x += fc.getlength(c) + 9
    d.text((pw / 2, 228), "The Night the Stars Fell", font=font("display", 60, 600),
           fill=int(255 * head), anchor="mm")
    dg.line([(pw / 2 - 60, 300), (pw / 2 + 60, 300)], fill=int(255 * head), width=2)
    total = sum(len(s) for s in PAGE_LINES)
    n = int(total * prog(t, TYPE_START, TYPE_END - TYPE_START))
    fb = font("serif", 53, 430)
    fi = font("serif_i", 53, 520)
    caret = None
    y = 410
    LM = 92
    for line in PAGE_LINES:
        shown = line[:max(0, n)]
        n -= len(line)
        if shown:
            if line.endswith("your name."):
                pre = "would know "
                d.text((LM, y), shown[:len(pre)], font=fb, fill=255, anchor="ls")
                if len(shown) > len(pre):
                    dg.text((LM + fb.getlength(pre), y), shown[len(pre):], font=fi, fill=255, anchor="ls")
                end_x = LM + fb.getlength(shown[:len(pre)]) + (fi.getlength(shown[len(pre):]) if len(shown) > len(pre) else 0)
            else:
                d.text((LM, y), shown, font=fb, fill=255, anchor="ls")
                end_x = LM + fb.getlength(shown)
            if len(shown) < len(line) or n <= 0:
                caret = (end_x + 6, y - 16)
        if n <= 0 and caret is None and shown:
            caret = (end_x + 6, y - 16)
        y += 84
        if n <= 0:
            break
    ink = np.asarray(im, np.float32)[..., None] / 255.0
    gink = np.asarray(gold_im, np.float32)[..., None] / 255.0
    ink_col = hexc("#2a1d14")
    page = paper * (1 - ink * 0.92) + ink_col * ink * 0.92
    page = page * (1 - gink) + hexc("#9a6a2c") * gink
    return page, caret


def scene_writing(t):
    lt = t - 7.2
    img = vgrad(hexc("#0a0807"), hexc("#060505"))
    page, caret = page_texture(t)
    ph, pw = page.shape[:2]
    # candle light pooled on the page
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    light = np.clip(1.1 - np.sqrt(((xx - pw * 0.55) / (pw * 0.95)) ** 2 + ((yy - ph * 0.4) / (ph * 0.9)) ** 2), 0.3, 1.0)
    page = page * (light * 0.74)[..., None] * hexc("#fff1dc")
    p = eo3(prog(t, 7.2, 1.6))
    R = rot(math.radians(lerp(30, 20, p)), math.radians(lerp(-9, -3, p)), math.radians(lerp(-6, -3, p)))
    pwu, phu = 1.0, 1.3
    corners = np.array([[-pwu / 2, -phu / 2, 0], [pwu / 2, -phu / 2, 0], [pwu / 2, phu / 2, 0], [-pwu / 2, phu / 2, 0]])
    Z = lerp(1.72, 1.86, p)
    Pw = corners @ R.T + [0, 0.05, Z]
    q = project(Pw, 1500, W / 2, 1150)
    sh = np.zeros((ph // 4, pw // 4, 4), np.float32)
    sh[..., 3] = 0.85
    sh = cv2.copyMakeBorder(sh, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=0)
    sh = cv2.GaussianBlur(sh, (0, 0), 14)
    qs = q + [18, 34]
    c = qs.mean(0)
    warp_quad(img, sh, c + (qs - c) * 1.12)
    warp_quad(img, rgba(page), q)
    glow(img, W * 0.62, 900, 900, hexc("#d08a3c"), 0.07, 2.0)
    if caret is not None:
        u, v = caret[0] / pw - 0.5, caret[1] / ph - 0.5
        cp = np.array([[u * pwu, v * phu, 0]]) @ R.T + [0, 0.05, Z]
        sx, sy = project(cp, 1500, W / 2, 1150)[0]
        blink = 0.75 + 0.25 * math.sin(t * 40)
        glow(img, sx, sy, 70, GOLD_HI, 0.5 * blink, 2.4)
        glow(img, sx, sy, 16, CREAM, 0.8 * blink, 1.5)
    TEX["dust_b"].draw(img, t, strength=0.45, parallax=(lt * 20, 0))
    kline(img, t, "Narra writes it.", "display", 116, 700, "cream", 380, 7.28, stagger=0.08,
          dur=0.5, shadow=0.7, t_out=9.45)
    kline(img, t, "Just for you.", "serif_i", 100, 500, "gold", 520, 7.62, stagger=0.09,
          dur=0.5, shadow=0.7, t_out=9.47)
    return img


def book_faces(t):
    lt = t - 9.6
    p = eo3(prog(t, 9.6, 2.4))
    ry = math.radians(lerp(-50, -28, p) + 2 * math.sin(lt * 1.3))
    rx = math.radians(8 + 2 * math.sin(lt * 0.9))
    R = rot(rx, ry, math.radians(-3))
    center = np.array([0.0, 0.02 * math.sin(lt * 1.6), lerp(3.15, 2.9, p)])
    return R, center


def scene_book(t):
    lt = t - 9.6
    img = vgrad(hexc("#0a0910"), hexc("#07060a"))
    glow(img, W / 2, 980, 900, hexc("#7a5030"), 0.25, 2.0)
    R, C = book_faces(t)
    w, h, d = 1.0, 1.42, 0.2
    F, cx, cy = 1700, W / 2, 1000

    def P(pts):
        return project(np.asarray(pts) @ R.T + C, F, cx, cy)

    fz = -d / 2
    faces = {
        "front": ([[-w / 2, -h / 2, fz], [w / 2, -h / 2, fz], [w / 2, h / 2, fz], [-w / 2, h / 2, fz]], TEX["cover"]),
        "right": ([[w / 2, -h / 2, fz], [w / 2, -h / 2, -fz], [w / 2, h / 2, -fz], [w / 2, h / 2, fz]], TEX["pages_side"]),
        "left": ([[-w / 2, -h / 2, -fz], [-w / 2, -h / 2, fz], [-w / 2, h / 2, fz], [-w / 2, h / 2, -fz]], TEX["spine"]),
        "top": ([[-w / 2, -h / 2, -fz], [w / 2, -h / 2, -fz], [w / 2, -h / 2, fz], [-w / 2, -h / 2, fz]], TEX["pages_top"]),
    }
    normals = {"front": [0, 0, -1], "right": [1, 0, 0], "left": [-1, 0, 0], "top": [0, -1, 0]}
    form = eio3(prog(t, BOOK_FORMED - 0.55, 0.65))
    vis = []
    for k, (pts, tex) in faces.items():
        n = R @ np.array(normals[k])
        ctr = np.mean(np.asarray(pts) @ R.T + C, 0)
        if n @ ctr < 0:
            vis.append((ctr[2], k, pts, tex))
    # shadow under the book
    if form > 0:
        glow(img, W / 2 + 40, 1600, 420, -np.ones(3, np.float32) * 0.04, form, 1.5)
    for _, k, pts, tex in sorted(vis, key=lambda v: -v[0]):
        q = P(pts)
        tx = tex
        if k == "front":
            gp = prog(t, 10.9, 0.9)
            if 0 < gp < 1:
                tx = glint(tex, eio3(gp), 0.07, 0.4)
        shade = {"front": 1.0, "right": 0.8, "left": 0.55, "top": 0.9}[k]
        tt = tx.copy()
        tt[..., :3] *= shade
        warp_quad(img, tt, q, alpha=form)
    # gold dust assembling into the cover
    if t < BOOK_FORMED + 0.6:
        r = np.random.default_rng(77)
        n = 3600
        cov = TEX["cover"]
        gm = (np.abs(cov[..., :3] - GOLD).sum(2) < 0.35)
        gy, gx = np.nonzero(gm)
        pick = r.integers(0, len(gx), int(n * 0.7))
        uv = np.concatenate([np.stack([gx[pick] / cov.shape[1], gy[pick] / cov.shape[0]], 1),
                             r.random((n - len(pick), 2))])
        tgt = np.stack([(uv[:, 0] - 0.5) * w, (uv[:, 1] - 0.5) * h, np.full(n, fz)], 1) @ R.T + C
        ang = r.uniform(0, 2 * np.pi, n)
        rad = r.uniform(0.45, 1.5, n)
        start = np.stack([np.cos(ang) * rad, np.sin(ang) * rad * 1.3, C[2] + r.uniform(-0.6, 0.9, n)], 1)
        delay = r.uniform(0, 0.45, n)
        pp = np.clip((lt - delay) / 0.75, 0, 1)
        pe = 1 - (1 - pp) ** 3
        swirl = (1 - pe) * 1.2
        sp = start * (1 - pe[:, None]) + tgt * pe[:, None]
        sw = ang + swirl * 2.2
        sp[:, 0] = sp[:, 0] * (1 - swirl * 0.3) + np.cos(sw) * swirl * 0.35
        sp[:, 1] = sp[:, 1] * (1 - swirl * 0.3) + np.sin(sw) * swirl * 0.35
        xy = project(sp, F, cx, cy)
        wts = (1.2 * np.clip(lt / 0.2, 0, 1) * (1 - form) * (0.5 + 0.5 * r.random(n))).astype(np.float32)
        m = splat(xy, wts * 1.8, 2, 0.8) + splat(xy, wts * 6, 4, 3.0) * 0.5
        img += m[..., None] * GOLD_HI
    if form > 0:
        flare = math.exp(-max(0.0, t - BOOK_FORMED) / 0.35) * form
        glow(img, W / 2, 990, 700, GOLD, 0.35 * flare, 2.0)
    TEX["dust_b"].draw(img, t, strength=0.35)
    kline(img, t, "A novel that", "display", 112, 700, "cream", 330, 9.66, stagger=0.08,
          dur=0.5, shadow=0.7, t_out=11.82)
    kline(img, t, "didn’t exist", "display", 112, 700, "cream", 460, 9.82, stagger=0.08,
          dur=0.5, shadow=0.7, t_out=11.84)
    kline(img, t, "until you asked.", "serif_i", 104, 500, "gold", 1470, 10.75, stagger=0.09,
          dur=0.5, shadow=0.8, t_out=11.86)
    return img


def draw_polyline_mask(mask, pts, widths, closed=False):
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        cv2.line(mask, (int(a[0] * 4), int(a[1] * 4)), (int(b[0] * 4), int(b[1] * 4)), 255,
                 max(1, int(widths[i])), cv2.LINE_AA, shift=2)


def light_layer(mask, color, core=1.0, halo=1.0):
    m = mask.astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(m, (0, 0), 6)
    small = cv2.resize(m, (QW, QH), interpolation=cv2.INTER_AREA)
    g2 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 8), (W, H))
    return (m * core + g1 * 1.2 * halo + g2 * 2.2 * halo)[..., None] * color


def ground_proj(x, z, camx, camz, hcam, H0, F=900.0):
    zz = np.asarray(z) - camz
    zz = np.where(zz > 0.05, zz, np.nan)
    return np.stack([W / 2 + F * (np.asarray(x) - camx) / zz, H0 + F * hcam / zz], -1), zz


def sky(H0):
    img = vgrad(hexc("#05070d"), hexc("#0c0d14"))
    band = np.exp(-((YY - H0) / 140) ** 2)[..., None]
    img += band * hexc("#5a4030") * 0.35
    below = (YY > H0)[..., None]
    img = np.where(below, img * 0.6, img)
    return img


def scene_choice(t):
    lt = t - 12.0
    sel = eio3(prog(t, TAP + 0.1, 1.1))
    camx = lerp(0, -1.9, sel)
    camz = lerp(0, 6.0, sel) + 0.6 * lt
    H0, hcam = 900, 1.0
    img = sky(H0)
    TEX["dust_a"].draw(img, t, color=hexc("#cfd8ff"), strength=0.25)
    # faint perspective grid
    grid = np.zeros((H, W), np.uint8)
    for gx in np.arange(-12, 13, 1.5):
        zs = np.linspace(0.3, 60, 40) + camz
        pts, _ = ground_proj(np.full_like(zs, gx), zs, camx, camz, hcam, H0)
        draw_polyline_mask(grid, pts, np.ones(len(pts)))
    for gz in np.arange(math.floor(camz), camz + 40, 1.5):
        xs = np.linspace(-14, 14, 30)
        pts, _ = ground_proj(xs, np.full_like(xs, gz), camx, camz, hcam, H0)
        draw_polyline_mask(grid, pts, np.ones(len(pts)))
    img += (grid.astype(np.float32) / 255)[..., None] * COOL * 0.06
    grow = eo3(prog(t, 12.0, 0.7))
    s = np.linspace(0, 1, 120)
    z_main = 0.3 + s * 5.2 * grow
    x_main = 0.08 * np.sin(z_main * 1.3)
    gold_m = np.zeros((H, W), np.uint8)
    dim_m = np.zeros((H, W), np.uint8)
    pts, zz = ground_proj(x_main, z_main, camx, camz, hcam, H0)
    draw_polyline_mask(gold_m, pts, np.clip(34 / np.nan_to_num(zz, nan=99), 2, 24))
    bg = eo3(prog(t, 12.45, 0.6))
    ends = {}
    for side in (-1, 1):
        sb = np.linspace(0, 1, 120) * bg
        xb = x_main[-1] + side * 4.0 * sb ** 1.4
        zb = 5.5 + 9.0 * sb
        pts, zz = ground_proj(xb, zb, camx, camz, hcam, H0)
        widths = np.clip(34 / np.nan_to_num(zz, nan=99), 2, 24)
        chosen = side < 0
        if chosen or sel < 0.01:
            draw_polyline_mask(gold_m, pts, widths)
        else:
            draw_polyline_mask(dim_m, pts, widths)
        ends[side] = ground_proj([xb[-1]], [zb[-1]], camx, camz, hcam, H0)[0][0]
    if sel > 0:
        right = dim_m
        img += light_layer(right, COOL * 0.5, 0.6, 0.4)
    img += light_layer(gold_m, GOLD, 1.0, 0.9)
    # light motes travelling along the chosen path
    r = np.random.default_rng(5)
    ph = (r.random(60) + lt * 0.35) % 1.0
    zz = 0.3 + ph * 14.5
    xx = np.where(zz < 5.5, 0.08 * np.sin(zz * 1.3), -4.0 * np.clip((zz - 5.5) / 9.0, 0, 1) ** 1.4)
    reach = np.where(zz < 5.5, zz <= 0.3 + 5.2 * grow, zz <= 5.5 + 9.0 * bg)
    xx, zz = xx[reach], zz[reach]
    mp, _ = ground_proj(xx, zz, camx, camz, hcam, H0)
    mp = mp[np.isfinite(mp).all(1)]
    img += splat(mp, np.full(len(mp), 9.0, np.float32), 2, 1.5)[..., None] * GOLD_HI * grow
    labels = [("Trust her", -1), ("Walk away", 1)]
    for name, side in labels:
        ex, ey = ends[side]
        if not np.isfinite(ex):
            continue
        lp = prog(t, 12.8 + (0.1 if side > 0 else 0), 0.45)
        e = eo5(lp)
        chosen = side < 0
        al = e * (1 - 0.7 * sel if not chosen else 1)
        col = "gold" if (chosen and t >= TAP) else "cream"
        sc = 1 + 0.12 * math.exp(-max(0.0, t - TAP) / 0.25) * (t >= TAP and chosen)
        blit(img, text_sprite(name, "serif_i", 72, 500, col), ex, ey - 70 - (1 - e) * 30,
             sc, al, blur=(1 - e) * 10)
        if chosen and t >= TAP:
            rp = prog(t, TAP, 0.6)
            m = np.zeros((H, W), np.uint8)
            cv2.circle(m, (int(ex), int(ey)), int(20 + 160 * eo3(rp)), 255, 3, cv2.LINE_AA)
            img += (m.astype(np.float32) / 255)[..., None] * GOLD_HI * (1 - rp)
            glow(img, ex, ey, 260, GOLD, 0.5 * (1 - rp), 2.0)
    kline(img, t, "Every choice", "display", 124, 700, "cream", 380, 12.06, stagger=0.08,
          dur=0.5, shadow=0.6, t_out=14.22)
    kline(img, t, "changes the story.", "serif_i", 100, 500, "gold", 520, 12.28, stagger=0.08,
          dur=0.5, shadow=0.6, t_out=14.24)
    return img


def make_tree(seed=8):
    r = np.random.default_rng(seed)
    segs = []  # (level, pts[N,2] in ground (x,z), on_path)

    def grow(x, z, ang, length, level, on_path):
        if level > 7:
            return
        n = 2 if level < 2 or r.random() < 0.75 else 3
        spread = lerp(0.62, 0.32, level / 7)
        angs = [ang + spread * (k - (n - 1) / 2) * (1.2 if n == 2 else 0.9) + r.uniform(-0.08, 0.08)
                for k in range(n)]
        pick = int(r.integers(0, n)) if on_path else -1
        for k, a in enumerate(angs):
            s = np.linspace(0, 1, 14)[:, None]
            bend = r.uniform(-0.25, 0.25)
            aa = ang + (a - ang) * s[:, 0] + bend * s[:, 0] * (1 - s[:, 0])
            dx = np.cumsum(np.sin(aa)) * length / 14
            dz = np.cumsum(np.cos(aa)) * length / 14
            pts = np.stack([x + np.r_[0, dx[:-1]], z + np.r_[0, dz[:-1]]], 1)
            segs.append((level, pts, on_path and k == pick))
            grow(pts[-1, 0] + math.sin(a) * length / 14, pts[-1, 1] + math.cos(a) * length / 14,
                 a, length * 0.82, level + 1, on_path and k == pick)

    trunk = np.stack([np.zeros(14), np.linspace(-1.5, 1.0, 14)], 1)
    segs.append((-1, trunk, True))
    grow(0.0, 1.0, 0.0, 2.1, 0, True)
    return segs


TREE = make_tree()


def scene_paths(t):
    lt = t - 14.4
    p = eio3(prog(t, 14.4, 2.4))
    H0 = 820
    hcam = lerp(1.4, 3.6, p)
    camz = lerp(-1.0, 0.6, p)
    camx = 0.0
    img = sky(H0)
    TEX["dust_a"].draw(img, t, color=hexc("#cfd8ff"), strength=0.25)
    gold_m = np.zeros((H, W), np.uint8)
    dim_m = np.zeros((H, W), np.uint8)
    nodes = []
    for level, pts, on in TREE:
        g = prog(t, 14.45 + (level + 1) * 0.14, 0.3)
        if g <= 0:
            continue
        k = max(2, int(len(pts) * g))
        sp, zz = ground_proj(pts[:k, 0], pts[:k, 1], camx, camz, hcam, H0)
        widths = np.clip(26 / np.nan_to_num(zz, nan=99), 1, 14)
        draw_polyline_mask(gold_m if on else dim_m, sp, widths)
        if g >= 1 and np.isfinite(sp[-1]).all():
            nodes.append((sp[-1], on))
    img += light_layer(dim_m, COOL * 0.55, 0.7, 0.45)
    img += light_layer(gold_m, GOLD, 1.1, 1.0)
    if nodes:
        pts = np.array([n[0] for n in nodes])
        wts = np.array([14.0 if n[1] else 5.0 for n in nodes], np.float32)
        img += splat(pts, wts, 2, 1.6)[..., None] * GOLD_HI
    kline(img, t, "Endless paths.", "display", 124, 700, "cream", 380, 14.48, stagger=0.08,
          dur=0.5, shadow=0.6, t_out=16.62)
    kline(img, t, "One is yours.", "serif_i", 104, 500, "gold", 520, 14.95, stagger=0.09,
          dur=0.5, shadow=0.6, t_out=16.64, glint_at=15.6)
    return img


MEM_LABELS = ["EVERY NAME", "EVERY SECRET", "EVERY CHOICE", "THE WORLD", "OLD RIVALS",
              "ALLIES", "BROKEN PROMISES", "LOOSE THREADS", "WHAT YOU LOVE"]


def make_constellation(seed=17):
    r = np.random.default_rng(seed)
    n_lab = len(MEM_LABELS)
    n = 46
    v = r.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= r.uniform(0.75, 1.0, (n, 1))
    # spread labeled nodes evenly in latitude so labels don't stack
    for i in range(n_lab):
        yv = lerp(-0.85, 0.85, i / (n_lab - 1))
        a = i * 2.4
        rr = math.sqrt(1 - yv ** 2)
        v[i] = [rr * math.cos(a), yv, rr * math.sin(a)]
    d = np.linalg.norm(v[:, None] - v[None], axis=2)
    edges = set()
    for i in range(n):
        for j in np.argsort(d[i])[1:4]:
            edges.add((min(i, j), max(i, j)))
    return v, sorted(edges)


CONST, EDGES = make_constellation()


def scene_memory(t):
    lt = t - 16.8
    img = vgrad(hexc("#06070d"), hexc("#0a0a10"))
    img += TEX["nebula"] * 0.6
    collapse = eio3(prog(t, 18.75, 0.45))
    ang = 0.5 + 0.42 * lt
    R = rot(math.radians(14), ang)
    rad = 1.0 * (1 - collapse * 0.95)
    P = CONST @ R.T * rad * [1.05, 1.25, 1.05] + [0, 0, 3.2]
    xy = project(P, 1150, W / 2, 1110)
    depth = np.clip((4.3 - P[:, 2]) / 2.2, 0, 1)
    lines = np.zeros((H, W), np.uint8)
    lp = prog(t, 16.95, 1.2)
    for k, (i, j) in enumerate(EDGES):
        if k / len(EDGES) > lp:
            continue
        cv2.line(lines, tuple((xy[i] * 4).astype(int)), tuple((xy[j] * 4).astype(int)),
                 int(60 + 140 * (depth[i] + depth[j]) / 2), 2, cv2.LINE_AA, shift=2)
    img += light_layer(lines, GOLD, 0.6, 0.35)
    pop = np.array([prog(t, 16.9 + 0.025 * i, 0.3) for i in range(len(CONST))], np.float32)
    w = (4 + 10 * depth) * pop
    w[:len(MEM_LABELS)] *= 1.8
    img += splat(xy, w.astype(np.float32), 2, 1.3)[..., None] * GOLD_HI
    for i, lab in enumerate(MEM_LABELS):
        lp2 = prog(t, 17.25 + 0.09 * i, 0.4)
        if lp2 <= 0:
            continue
        al = eo3(lp2) * (0.3 + 0.7 * depth[i]) * (1 - collapse)
        col = "gold" if lab == "WHAT YOU LOVE" else "cream"
        spr = text_sprite(lab, "sans", 38, 600, col, 0.14)
        side = 1 if xy[i, 0] < W / 2 else -1
        wlab = spr.shape[1] * 0.85
        blit(img, spr, xy[i, 0] + side * (wlab / 2 + 4), xy[i, 1], 0.85 + 0.25 * depth[i], al,
             blur=(1 - eo3(lp2)) * 8 + (1 - depth[i]) * 1.5)
    if collapse > 0:
        glow(img, W / 2, 1110, 200 + 700 * collapse, GOLD_HI, 1.4 * collapse ** 2, 2.4)
    kline(img, t, "It remembers", "display", 124, 700, "cream", 380, 16.88, stagger=0.08,
          dur=0.5, shadow=0.6, t_out=18.95)
    kline(img, t, "everything.", "serif_i", 116, 500, "gold", 530, 17.15, stagger=0.09,
          dur=0.5, shadow=0.6, t_out=18.97)
    return img


def scene_endcard(t):
    lt = t - 19.2
    img = vgrad(hexc("#0b0907"), hexc("#050404"))
    # volumetric rays from above
    ang = np.arctan2(QXX - QW / 2, QYY + 60)
    idx = ((ang / np.pi + 1) * 360 + lt * 6).astype(np.int32) % 720
    rays = TEX["rays"][idx] ** 3
    dist = np.sqrt((QXX - QW / 2) ** 2 + (QYY + 60) ** 2) / QH
    rays = rays * np.exp(-dist * 1.6) * 0.6
    img += cv2.resize(rays.astype(np.float32), (W, H))[..., None] * hexc("#d9a45c")
    glow(img, W / 2, 1150, 820, hexc("#a06a34"), 0.25, 2.0)
    TEX["dust_end"].draw(img, t, strength=0.75)
    push = 1 + 0.035 * eo3(prog(t, 19.2, 4.8))
    kline(img, t, "Your story", "display", 150, 700, "cream", 660 + (1 - push) * 300, 19.28,
          stagger=0.1, dur=0.5, style="rise", shadow=0.6)
    kline(img, t, "starts now.", "serif_i", 132, 500, "gold", 832 + (1 - push) * 200, 19.55,
          stagger=0.12, dur=0.55, shadow=0.6)
    p = prog(t, 20.25, 0.7)
    if p > 0:
        e = eo5(p)
        spr = text_sprite("Narra", "display", 290, 500, "gold")
        gp = prog(t, 20.75, 0.9)
        if 0 < gp < 1:
            spr = glint(spr, eio3(gp), 0.09, 1.6)
        flare = math.exp(-max(0.0, t - 20.4) / 0.4) * min(1, p * 3)
        glow(img, W / 2, 1190, 600, GOLD, 0.3 * flare + 0.09, 2.0)
        blit(img, spr, W / 2, 1190, (1.06 - 0.06 * e) * push, min(1, p * 1.8), blur=(1 - e) * 18)
        lp = eo3(prog(t, 20.9, 0.6))
        tag = text_sprite("STORIES MADE FOR YOU", "sans", 38, 500, "soft", 0.32)
        tw = tag.shape[1] * 0.95
        blit(img, tag, W / 2, 1390, 0.95, lp, blur=(1 - lp) * 8)
        m = np.zeros((H, W), np.uint8)
        L = 70 * lp
        for sgn in (-1, 1):
            xa = W / 2 + sgn * (tw / 2 - 10)
            cv2.line(m, (int(xa), 1390), (int(xa + sgn * L), 1390), 255, 2, cv2.LINE_AA)
        img += (m.astype(np.float32) / 255)[..., None] * GOLD * 0.9
    return img


SCENE_FN = {
    "hook": scene_hook, "someone_else": scene_someone_else, "for_you": scene_for_you,
    "tastes": scene_tastes, "writing": scene_writing, "book": scene_book,
    "choice": scene_choice, "paths": scene_paths, "memory": scene_memory,
    "endcard": scene_endcard,
}


def scene_at(t):
    for s, e, name in SCENES:
        if s <= t < e:
            return name
    return SCENES[-1][2]


def frame(t):
    init()
    img = SCENE_FN[scene_at(t)](t)
    # transitions around cuts
    for c, kind in TRANSITIONS.items():
        d = t - c
        win = 0.2
        if abs(d) > win or kind in ("none", "cut"):
            continue
        a = 1 - abs(d) / win
        a2 = a * a
        if kind == "whip_x":
            img = motion_blur(shift(img, (-1 if d < 0 else 1) * a2 * W * 0.45 * (1 if d < 0 else 1), 0), 10 + 160 * a2, 0)
        elif kind == "whip_y":
            img = motion_blur(shift(img, 0, (-1 if d < 0 else 1) * a2 * H * 0.3), 10 + 200 * a2, 1)
        elif kind == "zoom":
            img = radial_blur(zoom(img, 1 + (0.25 if d < 0 else -0.12) * a2), 0.18 * a2)
        if kind in ("flash", "zoom"):
            fl = a ** 2.2 * (0.9 if kind == "flash" else 0.18)
            img = img + fl * hexc("#fff0d8")
    if 4.0 <= t < 4.2:  # "you." zooms through camera
        p = ei3(prog(t, 4.0, 0.2))
        spr = text_sprite("you.", "display_i", 330, 800, "gold")
        img *= 1 - p
        blit(img, spr, W / 2, 1060, 1 + 7 * p, 1 - p * 0.6, blur=p * 12)
    if t < 0.08:
        img = img + (1 - t / 0.08) * 0.12 * hexc("#fff0d8")
    img = bloom(img)
    img *= VIGNETTE
    r = np.random.default_rng(int(t * FPS) + 1)
    g = r.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    img += g[..., None] * 0.016
    img = np.clip(img, 0, 1)
    return (img * 255 + 0.5).astype(np.uint8)


def render_index(i):
    return frame(i / FPS).tobytes()


def stills(times):
    os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
    for t in times:
        p = os.path.join(OUT, "stills", f"t{float(t):05.2f}.png")
        Image.fromarray(frame(float(t))).save(p)
        print(p)


def video():
    init()
    os.makedirs(OUT, exist_ok=True)
    score = os.path.join(OUT, "score.wav")
    if not os.path.exists(score):
        subprocess.run([sys.executable, os.path.join(HERE, "audio.py"), score], check=True)
    out = os.path.join(OUT, "narra-ad-01.mp4")
    n = int(round(DURATION * FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", score,
           "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-maxrate", "4200k",
           "-bufsize", "8400k", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(int(os.environ.get("JOBS", os.cpu_count() or 4))) as pool:
        for k, buf in enumerate(pool.imap(render_index, range(n), chunksize=2)):
            ff.stdin.write(buf)
            if k % 30 == 0:
                print(f"frame {k}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("video ->", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        stills(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "video":
        video()
    else:
        print(__doc__)
