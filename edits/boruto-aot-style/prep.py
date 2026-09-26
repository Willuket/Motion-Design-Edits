"""Asset fixes that need hand-tuned geometry (run after upscale.py and cutout.py)."""
import os

import cv2
import numpy as np
from PIL import Image

UP = os.environ.get("UPSCALE_DIR", "/tmp/aot/up")
OUT = os.environ.get("LAYER_DIR", "/tmp/aot/layers")


def headband():
    """Solid plate mask from the hull of the rembg mask (bright metal confuses it), plus
    Sasuke's scratch through the leaf."""
    rgb = np.asarray(Image.open(os.path.join(UP, "headband_jpeg.png")).convert("RGB")).copy()
    m = np.asarray(Image.open(os.path.join(OUT, "headband_fg.png")))[..., 3]
    cnts, _ = cv2.findContours((m > 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    mask = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    mask = cv2.GaussianBlur(cv2.erode(mask, np.ones((9, 9), np.uint8)), (0, 0), 2.5)
    p0, p1 = (800, 1150), (1480, 1330)
    cv2.line(rgb, (p0[0], p0[1] + 9), (p1[0], p1[1] + 9), (235, 235, 238), 7, cv2.LINE_AA)
    cv2.line(rgb, p0, p1, (38, 38, 42), 13, cv2.LINE_AA)
    Image.fromarray(np.dstack([rgb, mask]), "RGBA").save(os.path.join(OUT, "headband_cut.png"))


def portraits():
    im = Image.open(os.path.join(UP, "hokages4_jpeg.png")).convert("RGB")
    boxes = {"hashirama": (0, 0, 362, 489), "tobirama": (398, 0, 758, 489),
             "minato": (0, 512, 362, 1000), "hiruzen": (398, 512, 758, 1000)}
    for name, (x0, y0, x1, y1) in boxes.items():
        im.crop((x0 * 4, y0 * 4, x1 * 4, y1 * 4)).save(os.path.join(OUT, f"portrait_{name}.png"))


def father_son():
    from simple_lama_inpainting import SimpleLama
    im = Image.open(os.path.join(UP, "father_son_jpeg.png")).convert("RGB")
    small = im.resize((im.width // 4, im.height // 4), Image.LANCZOS)
    hole = np.zeros((small.height, small.width), np.uint8)
    hole[0:60, 0:195] = 255
    fixed = SimpleLama()(small, Image.fromarray(hole)).crop((0, 0) + small.size)
    big = np.asarray(fixed.resize(im.size, Image.LANCZOS)).astype(np.float32)
    a = cv2.GaussianBlur(cv2.resize(hole, im.size, interpolation=cv2.INTER_NEAREST), (0, 0), 20)
    a = (a.astype(np.float32) / 255)[..., None]
    src = np.asarray(im).astype(np.float32)
    rng = np.random.default_rng(0)
    big += rng.normal(0, 3.0, big.shape).astype(np.float32)
    out = src * (1 - a) + big * a
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(os.path.join(OUT, "father_son_clean.png"))


if __name__ == "__main__":
    headband()
    portraits()
    father_son()
    print("prep done")
