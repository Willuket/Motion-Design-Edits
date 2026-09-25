# Boruto: Two Blue Vortex — "Who Is She?" (test edit)

`Boruto - Who Is She (test edit).mp4` — 1080x1920, 30 fps, 39.9 s.
Song: I Monster – "Who Is She?" (15.08 s → 54.98 s), cold open voice line from the Boruto voice pack.

Everything is generated from the files in `Edit test images and song/`:

1. `upscale.py` — 4x Real-ESRGAN (anime 6B) upscale of every panel/render.
2. `prep_layers.py` — aligns each hand-cut PNG to its full panel and inpaints a background plate
   (`<n>_fg.png` / `<n>_bg.png`) for the parallax shots.
3. `render.py` — the edit: shot list, camera moves, hit-synced punches/flashes/shake, grading,
   fog/dust, glow, chromatic aberration, glitch, lyric typography, audio mix.

```bash
pip install -r requirements.txt   # plus a CPU torch build for upscale.py
curl -L -o /tmp/w/models/anime6b.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
# fonts (Google Fonts): Cinzel, Cormorant Garamond (+Italic), Noto Serif JP -> /tmp/w/fonts
cd "Edit test images and song" && python3 ../edits/boruto-who-is-she/upscale.py *.png *.jpeg *.jpg *.webp
cd ../edits/boruto-who-is-she && python3 prep_layers.py
python3 render.py stills 12.9 24.9   # preview frames
python3 render.py video              # -> /tmp/w/out/edit.mp4
```

Timing lives at the top of `render.py` (`BIG`/`SOFT` hit times, `SHOTS`, `TEXTS`), so re-cutting to a
different song section is a matter of editing those lists.
