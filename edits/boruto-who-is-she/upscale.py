"""4x anime upscale of every source panel (RGB and alpha upscaled separately)."""
import os
import sys

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "Edit test images and song")
DST = os.environ.get("UPSCALE_DIR", "/tmp/w/up")
MODEL = os.environ.get("ESRGAN_MODEL", "/tmp/w/models/anime6b.pth")
TILE = 256
PAD = 16

torch.set_num_threads(os.cpu_count() or 4)
model = ModelLoader().load_from_file(MODEL).model.eval()


def run(rgb):
    """rgb: float32 HxWx3 in [0,1] -> 4x upscaled, tiled to bound memory."""
    h, w, _ = rgb.shape
    out = np.zeros((h * 4, w * 4, 3), np.float32)
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            y0, x0 = max(y - PAD, 0), max(x - PAD, 0)
            y1, x1 = min(y + TILE + PAD, h), min(x + TILE + PAD, w)
            t = torch.from_numpy(rgb[y0:y1, x0:x1].transpose(2, 0, 1))[None]
            with torch.no_grad():
                r = model(t)[0].clamp(0, 1).numpy().transpose(1, 2, 0)
            ty, tx = (y - y0) * 4, (x - x0) * 4
            th, tw = min(TILE, h - y) * 4, min(TILE, w - x) * 4
            out[y * 4:y * 4 + th, x * 4:x * 4 + tw] = r[ty:ty + th, tx:tx + tw]
    return out


def main(names):
    os.makedirs(DST, exist_ok=True)
    for name in names:
        out_path = os.path.join(DST, name.replace(".", "_") + ".png")
        if os.path.exists(out_path):
            continue
        im = Image.open(os.path.join(SRC, name))
        has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        im = im.convert("RGBA")
        a = np.asarray(im).astype(np.float32) / 255
        rgb = run(np.ascontiguousarray(a[..., :3]))
        if has_alpha:
            al = run(np.repeat(a[..., 3:4], 3, axis=2)).mean(axis=2, keepdims=True)
            res = np.concatenate([rgb, al], axis=2)
            Image.fromarray((res * 255 + 0.5).astype(np.uint8), "RGBA").save(out_path)
        else:
            Image.fromarray((rgb * 255 + 0.5).astype(np.uint8), "RGB").save(out_path)
        print("upscaled", name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
