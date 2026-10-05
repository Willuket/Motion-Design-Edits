# Narra — Ad 01 "Written for you"

`narra-ad-01.mp4`: 1080x1920 (9:16), 30 fps, 24.0 s, H.264 High / yuv420p + AAC stereo, ~13 MB.
Made for TikTok, Reels and Shorts. It works with the sound off: every key line is on screen.

## Concept

**One message: every story you've read was written for someone else. Narra writes one for you.**

This comes straight from the product idea in the Narra docs (`NARRA_MASTER_SPEC.md`). Readers
"select the kind of story they want", "Narra creates and continuously generates that story for
them", and the story "remembers" its world, characters, secrets and the reader's decisions, so it
feels like "a living novel". The spec's brand section says not to lead with "AI". Narra should feel
like entertainment, not a utility, so the word "AI" never appears.

The Narra app UI is being rebuilt, so **the ad shows no app UI, screens or mockups**. Everything
is concept-driven motion design: an endless library, words drifting in depth, ink writing itself,
a book forming out of gold dust, paths of light that fork, and a constellation of memories. The
literary gold-on-black look follows the spec's direction ("dark-first, premium, immersive"; deep
dark background, one distinctive accent colour).

## Script (beat by beat)

Cuts sit on a 100 BPM grid (one beat = 0.6 s). `timeline.py` is the shared cut sheet for picture
and sound.

| time | picture | on-screen type |
|---|---|---|
| 0.0–1.8 | **Hook.** A camera rushes down an endless library corridor (procedural shelves, lamp pools, depth of field). The first line is already on screen in frame 1. | **Every story** / *you've ever read* |
| 1.8–3.0 | The library drains to grey and dims. A gold strike-through cuts across "someone else." | *was written for* / **someone else.** |
| 3.0–4.2 | Darkness, one warm light and drifting dust. "you." slams in with a glint, then zooms through the camera. | *This one* / *is written for* / ***you.*** |
| 4.2–7.2 | A field of tastes (genres and tropes) floats toward the camera in depth. Three light up gold and gather in the centre, then collapse into a spark. | **Tell Narra** / *what you love.* · *Dark fantasy · Slow burn · Morally grey lead* |
| 7.2–9.6 | A candlelit page writes itself with a glowing nib: "You were the only one awake when the first star fell. By morning, the whole kingdom would know *your name.*" | **Narra writes it.** / *Just for you.* |
| 9.6–12.0 | Gold dust spirals into a 3D leather-bound book, "YOUR STORY", that catches the light. | **A novel that** / **didn't exist** / *until you asked.* |
| 12.0–14.4 | A path of light runs to the horizon and forks: *Trust her* / *Walk away*. One is chosen, and the camera follows it. | **Every choice** / *changes the story.* |
| 14.4–16.8 | The camera cranes up as the path branches into a whole tree of possible stories. One golden route runs through it. | **Endless paths.** / *One is yours.* |
| 16.8–19.2 | A rotating constellation of memories: EVERY NAME, EVERY SECRET, EVERY CHOICE, THE WORLD, OLD RIVALS, ALLIES, BROKEN PROMISES, LOOSE THREADS, WHAT YOU LOVE. It collapses into a flash. | **It remembers** / *everything.* |
| 19.2–24.0 | **End card.** God rays, gold dust, the wordmark and the tagline. | **Your story** / *starts now.* · **Narra** · STORIES MADE FOR YOU |

The tagline "Stories made for you." comes from the spec's brand section. The repo has no URL or
social handle, so none is shown.

## Sound

The score is fully synthesized in `audio.py` (numpy/scipy), with no samples and no licensed music.
- **Hook:** an A-minor drone with a ticking pulse, and sub impacts on frame 0 and on "you.".
- **Under the product idea:** Karplus-Strong harp arpeggios and detuned-saw pads (Am–F–C–G), plus a kick and hats.
- **Effects:** whooshes on cuts, chimes when tastes light up, soft pen ticks while the page writes, and a bell when the book forms and when a path is chosen.
- **End card:** a riser into the card, which resolves to C.

The mix measures about -13 LUFS integrated with a -1 dBFS peak.

## Typography and palette

- Playfair Display (headlines, wordmark), Lora (literary italics) and DM Sans (labels). These are the typefaces the Narra web app ships (`apps/web/app/fonts.ts`). All are SIL OFL 1.1, and the licences are in `fonts/`.
- Gold `#e8b36d` (gradient `#f6dcab` → `#b8844a`), ink background `#070b11`, cream `#f7f2e8`, and a faint purple `#b08ef6` haze in the taste field.

## Safe zones

All critical text sits between y = 280 and y = 1520 (it clears the top 250 px and bottom 350 px)
and stays horizontally centred, away from the right-side action rail.

## Re-render

```bash
pip install -r requirements.txt          # numpy, scipy, opencv-python-headless, pillow
python3 audio.py /tmp/narra_ad/score.wav # score (≈2 s)
python3 render.py stills 1.4 3.9 11.4 22.5   # preview PNGs -> /tmp/narra_ad/stills
python3 render.py video                  # -> /tmp/narra_ad/narra-ad-01.mp4 (≈5 min on 4 cores)
python3 contact.py sheet.png /tmp/narra_ad/stills/*.png   # review sheet with safe-zone guides
```

`NARRA_OUT` changes the output directory and `JOBS` sets the number of worker processes. If
`score.wav` is missing, `render.py video` builds it first. Delete it after changing cues in
`timeline.py`.

## Files

- `timeline.py`: the cut sheet, scene windows, transition styles and sound cues
- `render.py`: the procedural picture: textures, scenes, kinetic type, transitions, bloom, grain and the ffmpeg encode
- `audio.py`: the synthesized score and sound design
- `contact.py`: builds review contact sheets
- `fonts/`: the OFL fonts
