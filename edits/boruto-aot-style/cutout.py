"""Background removal (rembg) + LaMa inpainted background plates for parallax layers."""
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
UP = os.environ.get("UPSCALE_DIR", "/tmp/aot/up")
OUT = os.environ.get("LAYER_DIR", "/tmp/aot/layers")


def source(name):
    """Prefer the 4x-upscaled version when one exists."""
    up = os.path.join(UP, name.replace(".", "_") + ".png")
    return Image.open(up if os.path.exists(up) else os.path.join(SRC, name)).convert("RGB")


def cut(name, model="isnet-anime", max_side=1600):
    from rembg import new_session, remove
    im = source(name)
    small = im.copy()
    small.thumbnail((max_side, max_side), Image.LANCZOS)
    m = remove(small, session=new_session(model), only_mask=True, post_process_mask=True)
    m = np.asarray(m.resize(im.size, Image.LANCZOS)).astype(np.float32) / 255
    return im, m


def plate(im, mask, grow=25):
    """Fill the region behind the cutout so the background layer can move independently."""
    from simple_lama_inpainting import SimpleLama
    hole = (cv2.dilate((mask > 0.1).astype(np.uint8), np.ones((grow, grow), np.uint8)) * 255)
    small = im.copy()
    scale = min(1.0, 1400 / max(im.size))
    if scale < 1:
        small = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        hole = cv2.resize(hole, small.size, interpolation=cv2.INTER_NEAREST)
    res = SimpleLama()(small, Image.fromarray(hole))
    res = res.crop((0, 0) + small.size).resize(im.size, Image.LANCZOS)
    return res


def main(jobs):
    os.makedirs(OUT, exist_ok=True)
    for job in jobs:
        name, _, model = job.partition(":")
        model = model or "isnet-anime"
        stem = os.path.splitext(name)[0]
        im, m = cut(name, model)
        rgba = np.dstack([np.asarray(im), (m * 255).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(os.path.join(OUT, stem + "_fg.png"))
        if os.environ.get("PLATES", "1") == "1":
            plate(im, m).save(os.path.join(OUT, stem + "_bg.png"))
        print("layered", name, model, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
