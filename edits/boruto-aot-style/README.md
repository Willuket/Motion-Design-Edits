# Boruto — "A Love Immortal" (AoT "My best work yet" recreation)

`Boruto - A Love Immortal (AoT style edit).mp4` — 1440x1080 (4:3), 30 fps, 31.4 s.
Song: I Monster – "Who Is She?" (163.54 s → 194.97 s, the "lost embrace … eternity" verse), the
same section and cut structure as the Attack on Titan reference edit in the repo root.

Shot-for-shot map of the reference, retold with Boruto / Naruto:

| time | reference (AoT) | this edit |
|---|---|---|
| 0.0–5.3 | reaching hands, gold slot bar, "EMBRACE" | Naruto & Boruto hands, chrome red "EMBRACE" glitch |
| 5.3–9.0 | rim-lit silhouette, flare, "ACROSS THE SEA" | Boruto back view, anamorphic flare + bokeh |
| 9.0–12.0 | glass shatter, "Eren Yeager" chrome card, "TIME" | shatter into Boruto's eye, "Boruto Uzumaki" card |
| 12.0–15.3 | Eren in the clouds, "A LOVE IMMORTAL", negative | Boruto in the clouds, same text-behind-subject + negative flip |
| 15.3–18.5 | hand reaching for the key, gull, "MINE" | hand reaching for Sasuke's scratched headband, gull, "MINE" |
| 18.5–21.4 | Eren/Mikasa/Armin painting, "WILL COME TO" | Team 7 (Sarada/Boruto/Mitsuki) |
| 21.4–24.7 | memory gallery wall, whip, double exposure | Hokage portrait wall, Naruto close-up, father/son flash |
| 24.7–31.4 | white → grey → black, "ETERNITY" | same, tracking-in "ETERNITY" |

Everything is procedural (numpy/OpenCV, piped into ffmpeg): lens flares, glass-shard
Delaunay shatter, light shafts, bokeh, film sprockets, 3D gallery projection, chrome text, grain.

## Pipeline

Source art in `src/` was pulled with `fetch.py` (DuckDuckGo image search).

```bash
pip install -r requirements.txt   # CPU torch is enough
curl -L -o /tmp/w/models/anime6b.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
python3 upscale.py                  # 4x Real-ESRGAN        -> /tmp/aot/up
python3 cutout.py silhouette.jpeg sky_boruto.jpeg team7.png hands.jpeg hand_up.jpeg \
        headband.jpeg naruto_back.jpeg boruto_sword.jpeg   # rembg + LaMa plates -> /tmp/aot/layers
python3 prep.py                     # headband scratch, Hokage portrait crops, cleanup
python3 render.py stills 9.6 13.5   # preview frames -> /tmp/aot/out
python3 render.py video             # -> /tmp/aot/out/edit.mp4
```

Timing lives at the top of `render.py` (`WORDS` sung-word onsets, `PUNCH` hits) and in `SHOTS`.
