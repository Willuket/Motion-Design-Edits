"""Align each hand-cut character PNG to its full panel and build a clean background plate.

Outputs (in UPSCALE_DIR):
  <n>_fg.png  character cutout placed in the panel's coordinate space (RGBA)
  <n>_bg.png  panel with the character inpainted out (RGB)
"""
import os

import cv2
import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "Edit test images and song")
UP = os.environ.get("UPSCALE_DIR", "/tmp/w/up")
PAIRS = [7, 8, 9, 10, 11, 12, 16, 17]


def read(path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def align(panel, cut):
    """Best (scale, x, y) placing `cut` (RGBA) inside `panel` (BGR); small originals."""
    pg = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY).astype(np.float32)
    best = (-1, None)
    for s in np.linspace(0.8, 1.25, 46):
        c = cv2.resize(cut, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if c.shape[0] > pg.shape[0] or c.shape[1] > pg.shape[1]:
            continue
        cg = cv2.cvtColor(c[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = (c[..., 3] > 200).astype(np.float32)
        r = cv2.matchTemplate(pg, cg, cv2.TM_SQDIFF, mask=m)
        _, _, loc, _ = cv2.minMaxLoc(r)
        err = r[loc[1], loc[0]] / max(m.sum(), 1)
        score = -err
        if score > best[0] or best[1] is None:
            best = (score, (s, loc[0], loc[1]))
    return best[1], -best[0]


def main():
    for n in PAIRS:
        panel_s = read(f"{SRC}/images ({n}).jpeg", cv2.IMREAD_COLOR)
        cut_s = read(f"{SRC}/images ({n}).png")
        (s, x, y), err = align(panel_s, cut_s)
        panel = read(f"{UP}/images ({n})_jpeg.png", cv2.IMREAD_COLOR)
        cut = read(f"{UP}/images ({n})_png.png")
        k = panel.shape[1] / panel_s.shape[1]
        sc = s * k * cut_s.shape[1] / cut.shape[1]
        M = np.float32([[sc, 0, x * k], [0, sc, y * k]])
        fg = cv2.warpAffine(cut, M, (panel.shape[1], panel.shape[0]),
                            flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))

        # The hand cutouts have stray transparent holes inside the silhouette; fill them
        # from the panel so the character stays solid when it separates from the plate.
        solid = (fg[..., 3] > 20).astype(np.uint8) * 255
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(solid)
        cv2.drawContours(filled_mask, cnts, -1, 255, cv2.FILLED)
        inner = cv2.erode(filled_mask, np.ones((7, 7), np.uint8))
        holes = (inner > 0) & (fg[..., 3] < 250)
        fg[holes, :3] = panel[holes]
        fg[holes, 3] = 255
        cv2.imwrite(f"{UP}/{n}_fg.png", fg)

        # Inpaint at low resolution (fast, smoother fill) then soften the filled region.
        mask = filled_mask
        mask = cv2.dilate(mask, np.ones((25, 25), np.uint8))
        f = 0.25
        ps = cv2.resize(panel, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        ms = cv2.resize(mask, (ps.shape[1], ps.shape[0]), interpolation=cv2.INTER_NEAREST)
        filled = cv2.inpaint(ps, ms, 9, cv2.INPAINT_TELEA)
        filled = cv2.GaussianBlur(filled, (0, 0), 3)
        filled = cv2.resize(filled, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_CUBIC)
        a = cv2.GaussianBlur(mask, (0, 0), 12).astype(np.float32)[..., None] / 255
        bg = (panel * (1 - a) + filled * a).astype(np.uint8)
        cv2.imwrite(f"{UP}/{n}_bg.png", bg)
        print(n, f"scale={s:.3f} x={x} y={y} err={err:.1f}", flush=True)


if __name__ == "__main__":
    main()
