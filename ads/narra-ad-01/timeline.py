"""Shared cut sheet for picture (render.py) and score (audio.py).

Everything is locked to a 100 BPM grid (one beat = 0.6 s) so every cut lands on
a beat of the synthesized score.
"""

W, H, FPS = 1080, 1920, 30
BPM = 100
BEAT = 60.0 / BPM
DURATION = 24.0

# (start, end, scene id)
SCENES = [
    (0.0, 1.8, "hook"),          # Every story you've ever read      (infinite library dolly)
    (1.8, 3.0, "someone_else"),  # was written for someone else.     (library drains to grey)
    (3.0, 4.2, "for_you"),       # This one is written for you.      (darkness, one warm light)
    (4.2, 7.2, "tastes"),        # Tell Narra what you love.         (field of genres/tastes)
    (7.2, 9.6, "writing"),       # Narra writes it. Just for you.    (ink writing itself)
    (9.6, 12.0, "book"),         # A novel that didn't exist until you asked. (book from dust)
    (12.0, 14.4, "choice"),      # Every choice changes the story.   (path of light forks)
    (14.4, 16.8, "paths"),       # Endless paths. One is yours.      (branching tree of paths)
    (16.8, 19.2, "memory"),      # It remembers everything.          (memory constellation)
    (19.2, 24.0, "endcard"),     # Your story starts now. + Narra + Stories made for you.
]

# Transition style at each cut.
TRANSITIONS = {
    1.8: "none",
    3.0: "cut",
    4.2: "zoom",
    7.2: "flash",
    9.6: "whip_x",
    12.0: "whip_y",
    14.4: "whip_x",
    16.8: "zoom",
    19.2: "flash",
}

# Score cues (seconds).
IMPACTS = [0.0, 3.45, 19.2]
WHOOSHES = [1.8, 4.2, 9.6, 12.0, 14.4, 16.8]
PICKS = [5.4, 5.8, 6.2]            # tastes light up
COLLAPSE = 6.75                    # tastes collapse into a spark
TYPE_START, TYPE_END = 7.55, 9.25  # ink writing
BOOK_FORMED = 10.5
TAP = 13.15                        # a path is chosen
STRIKE = 2.5
LOGO = 20.4
GROOVE_START = 4.2
GROOVE_END = 19.2
