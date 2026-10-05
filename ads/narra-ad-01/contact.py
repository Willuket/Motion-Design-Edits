"""Contact sheet of rendered stills with safe-zone guides (for frame review).

python3 contact.py out.png a.png b.png ...
"""
import sys

from PIL import Image, ImageDraw

out, files = sys.argv[1], sys.argv[2:]
tw, th = 360, 640
cols = min(4, len(files))
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * (th + 24)), (30, 30, 30))
d = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB").resize((tw, th), Image.LANCZOS)
    g = ImageDraw.Draw(im)
    for y in (250, 1920 - 350):
        g.line([(0, y * th / 1920), (tw, y * th / 1920)], fill=(255, 0, 80), width=1)
    x, y = (i % cols) * tw, (i // cols) * (th + 24)
    sheet.paste(im, (x, y + 24))
    d.text((x + 6, y + 5), f.split("/")[-1], fill=(220, 220, 220))
sheet.save(out)
