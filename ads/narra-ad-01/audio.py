"""Synthesized score + sound design for Narra ad 01 (no samples, no licensed music).

A minor cinematic bed at 100 BPM: sub impacts, a ticking tension pulse under the
hook, a Karplus-Strong harp arpeggio and detuned-saw pads once the product
enters, FM bell shimmer on "you.", the finished book and the logo, whooshes on
cuts, soft pen ticks while the page writes itself.

Usage: python3 audio.py [out.wav]
"""
import sys

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, fftconvolve

from timeline import (BEAT, BOOK_FORMED, COLLAPSE, DURATION, GROOVE_END,
                      GROOVE_START, IMPACTS, LOGO, PICKS, STRIKE, TAP, TYPE_END,
                      TYPE_START, WHOOSHES)

SR = 48000
N = int(DURATION * SR)
rng = np.random.default_rng(7)


def t_arr(dur):
    return np.arange(int(dur * SR)) / SR


def note(name):
    names = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7,
             "G#": 8, "A": 9, "A#": 10, "B": 11}
    pitch, octave = name[:-1], int(name[-1])
    return 440.0 * 2 ** ((names[pitch] + 12 * (octave + 1) - 69) / 12)


def place(buf, sig, at, gain=1.0, pan=0.0):
    i = int(at * SR)
    if i >= buf.shape[1] or i + len(sig) <= 0:
        return
    j0 = max(0, -i)
    sig = sig[j0:]
    i = max(i, 0)
    n = min(len(sig), buf.shape[1] - i)
    left = np.cos((pan + 1) * np.pi / 4)
    right = np.sin((pan + 1) * np.pi / 4)
    buf[0, i:i + n] += sig[:n] * gain * left * 1.414
    buf[1, i:i + n] += sig[:n] * gain * right * 1.414


def lp(x, hz, order=2):
    return sosfilt(butter(order, hz, "low", fs=SR, output="sos"), x)


def hp(x, hz, order=2):
    return sosfilt(butter(order, hz, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def sweep_lp(x, f0, f1, block=256):
    """One-pole lowpass with an exponentially swept cutoff."""
    out = np.empty_like(x)
    zi = np.zeros(1)
    nb = int(np.ceil(len(x) / block))
    for b in range(nb):
        f = f0 * (f1 / f0) ** (b / max(nb - 1, 1))
        a = 1 - np.exp(-2 * np.pi * f / SR)
        seg = x[b * block:(b + 1) * block]
        y, zi = lfilter([a], [1, a - 1], seg, zi=zi)
        out[b * block:(b + 1) * block] = y
    return out


# ---------------------------------------------------------------- instruments

def kick(dur=0.45):
    t = t_arr(dur)
    f = 45 + 95 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.16)
    click = hp(rng.standard_normal(len(t)), 2500) * np.exp(-t / 0.004) * 0.25
    return np.tanh((body + click) * 1.6) * 0.9


def impact(dur=3.2):
    t = t_arr(dur)
    f = 30 + 40 * np.exp(-t / 0.08)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9)
    crack = lp(rng.standard_normal(len(t)), 3800) * np.exp(-t / 0.07)
    rumble = lp(rng.standard_normal(len(t)), 180) * np.exp(-t / 0.7) * 2.5
    boom = np.tanh(1.8 * sub) + 0.5 * crack + 0.6 * rumble
    return boom * 0.9


def tick(dur=0.09):
    t = t_arr(dur)
    s = bp(rng.standard_normal(len(t)), 1800, 5200) * np.exp(-t / 0.012)
    s += np.sin(2 * np.pi * 1100 * t) * np.exp(-t / 0.02) * 0.3
    return s * 0.5


def hat(dur=0.08, open_=False):
    t = t_arr(dur if not open_ else 0.25)
    s = hp(rng.standard_normal(len(t)), 7000) * np.exp(-t / (0.015 if not open_ else 0.08))
    return s * 0.35


def whoosh(dur=0.7, peak=0.62):
    t = t_arr(dur)
    n = rng.standard_normal(len(t))
    env = np.where(t < dur * peak, (t / (dur * peak)) ** 2.2,
                   np.exp(-(t - dur * peak) / 0.07))
    s = sweep_lp(n, 300, 7000) * env
    return hp(s, 150) * 0.9


def riser(dur=1.2):
    t = t_arr(dur)
    n = sweep_lp(rng.standard_normal(len(t)), 200, 9000)
    tone = np.sin(2 * np.pi * np.cumsum(220 * 2 ** (2 * t / dur)) / SR) * 0.15
    env = (t / dur) ** 2.5
    return (hp(n, 200) * 0.7 + tone) * env


def reverse_swell(dur=0.9):
    t = t_arr(dur)
    s = sweep_lp(rng.standard_normal(len(t)), 400, 6000) * (t / dur) ** 3
    return hp(s, 300) * 0.6


def scratch(dur=0.28):
    t = t_arr(dur)
    n = rng.standard_normal(len(t))
    s = sweep_lp(n, 6000, 900) * np.minimum(1, t / 0.01) * np.exp(-t / 0.12)
    return hp(s, 600) * 0.7


def tap(dur=0.12):
    t = t_arr(dur)
    s = np.sin(2 * np.pi * 1650 * t) * np.exp(-t / 0.018)
    s += np.sin(2 * np.pi * 2900 * t) * np.exp(-t / 0.008) * 0.4
    return s * 0.45


def pop(freq, dur=0.25):
    t = t_arr(dur)
    f = freq * (1 + 0.6 * np.exp(-t / 0.01))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.07)
    return s * 0.35


def bell(freq, dur=3.0, index=2.2):
    t = t_arr(dur)
    mod = np.sin(2 * np.pi * freq * 3.5 * t) * index * np.exp(-t / 0.6)
    s = np.sin(2 * np.pi * freq * t + mod) * np.exp(-t / 1.1)
    s += 0.35 * np.sin(2 * np.pi * freq * 2.001 * t) * np.exp(-t / 0.5)
    return s * np.minimum(1, t / 0.004) * 0.4


def pluck(freq, dur=1.6, bright=0.5):
    n = int(dur * SR)
    period = int(SR / freq)
    buf = lp(rng.uniform(-1, 1, period), 1500 + 6000 * bright, 1)
    out = np.zeros(n)
    out[:period] = buf
    # Karplus-Strong: vectorized one period at a time.
    for i in range(period, n, period):
        prev = out[i - period:i]
        nxt = 0.5 * (prev + np.roll(prev, 1)) * 0.996
        out[i:i + period] = nxt[:min(period, n - i)]
    t = np.arange(n) / SR
    return out * np.exp(-t / 0.9) * 0.6


def pad(freqs, dur, attack=0.5, release=0.9, cutoff=1400):
    t = t_arr(dur + release)
    s = np.zeros(len(t))
    for f in freqs:
        for det in (-0.08, 0.0, 0.07):
            ff = f * 2 ** (det / 12)
            ph = rng.uniform(0, 1)
            s += 2 * ((ff * t + ph) % 1.0) - 1
    s = lp(s / (len(freqs) * 3), cutoff, 2)
    env = np.minimum(1, t / attack)
    env *= np.where(t > dur, np.exp(-(t - dur) / (release / 3)), 1.0)
    return s * env


def drone(dur, f=note("A1")):
    t = t_arr(dur)
    s = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.003 * t)
    s += 0.3 * lp(2 * ((f * 1.498 * t) % 1.0) - 1, 600)
    lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.35 * t)
    return s * lfo * 0.35


def reverb(x, seconds=2.4, damp=5000, seed=3):
    r = np.random.default_rng(seed)
    t = t_arr(seconds)
    ir = r.standard_normal(len(t)) * np.exp(-t / (seconds / 5.5))
    ir = lp(ir, damp)
    ir[: int(0.012 * SR)] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    return fftconvolve(x, ir)[: len(x)]


# ---------------------------------------------------------------------- score

CHORDS = [  # (start, root-position voicing for pad, arpeggio tones)
    ("Am", ["A2", "E3", "A3", "C4"], ["A4", "C5", "E5", "A5", "E5", "C5", "B4", "C5"]),
    ("F", ["F2", "C3", "A3", "C4"], ["F4", "A4", "C5", "F5", "C5", "A4", "G4", "A4"]),
    ("C", ["C3", "G3", "C4", "E4"], ["C5", "E5", "G5", "C6", "G5", "E5", "D5", "E5"]),
    ("G", ["G2", "D3", "G3", "B3"], ["G4", "B4", "D5", "G5", "D5", "B4", "A4", "B4"]),
]


def build():
    dry = np.zeros((2, N))
    wet_send = np.zeros((2, N))

    # Hook: drone + ticking pulse, dropping out right before "you."
    d = drone(3.3)
    d *= np.minimum(1, t_arr(3.3) / 0.05) * np.where(t_arr(3.3) > 2.75,
                                                    np.exp(-(t_arr(3.3) - 2.75) / 0.12), 1)
    place(dry, d, 0.0, 0.55)
    for k in range(int(2.7 / (BEAT / 2))):
        at = k * BEAT / 2
        place(dry, tick(), at, 0.35 if k % 2 else 0.55, pan=-0.3 if k % 2 else 0.3)
    place(dry, kick(), 1.8, 0.7)
    place(dry, scratch(), STRIKE, 0.5, pan=0.2)
    place(wet_send, reverse_swell(), 3.45 - 0.9, 0.7)

    for at in IMPACTS:
        place(dry, impact(), at, 0.95 if at != 3.45 else 0.7)
        place(wet_send, impact(), at, 0.35)

    # "you." shimmer
    for f, g in ((note("A5"), 0.5), (note("E6"), 0.3), (note("C6"), 0.25)):
        place(wet_send, bell(f, 3.0), 3.45, g)
        place(dry, bell(f, 3.0), 3.45, g * 0.4)
    place(dry, pad([note("A2"), note("E3"), note("C4")], 0.75, attack=0.3), 3.45, 0.35)

    # Groove: pads + harp arpeggio + kick/hat from the product reveal to the end card.
    chord_len = 4 * BEAT
    k = 0
    at = GROOVE_START
    while at < GROOVE_END - 1e-6:
        _, voicing, arp = CHORDS[k % 4]
        dur = min(chord_len, GROOVE_END - at)
        p = pad([note(n) for n in voicing], dur, attack=0.35, release=0.8,
                cutoff=900 + 500 * min(1, (at - GROOVE_START) / 8))
        place(dry, p, at, 0.42)
        place(wet_send, p, at, 0.3)
        for i, n in enumerate(arp):
            when = at + i * BEAT / 2
            if when >= GROOVE_END - 0.05:
                break
            pl = pluck(note(n), bright=0.35 + 0.3 * (i % 2))
            place(dry, pl, when, 0.32, pan=-0.45 if i % 2 else 0.45)
            place(wet_send, pl, when, 0.25)
        place(dry, kick(), at, 0.0)
        k += 1
        at += chord_len

    b = GROOVE_START
    i = 0
    while b < GROOVE_END - 0.01:
        if b >= GROOVE_START + 2 * BEAT:  # let the reveal breathe before the beat drops
            place(dry, kick(), b, 0.75 if i % 2 == 0 else 0.55)
            place(dry, hat(), b + BEAT / 2, 0.55, pan=0.25)
            if i % 4 == 3:
                place(dry, hat(open_=True), b + BEAT * 0.75, 0.35, pan=-0.25)
        b += BEAT
        i += 1

    for at in WHOOSHES:
        w = whoosh()
        place(dry, w, at - 0.62 * 0.7, 0.55, pan=0.0)
        place(wet_send, w, at - 0.62 * 0.7, 0.35)
    place(dry, tap(), TAP, 0.7)
    place(wet_send, tap(), TAP, 0.35)
    place(wet_send, bell(note("E6"), 2.0, 1.2), TAP, 0.2)
    for i, at in enumerate(PICKS):
        place(dry, pop(700 + 140 * i), at, 0.6, pan=(-0.4, 0.4, 0.0)[i % 3])
        place(wet_send, bell(note(("A5", "C6", "E6")[i]), 1.6, 1.0), at, 0.18)
    place(wet_send, reverse_swell(0.5), COLLAPSE - 0.05, 0.6)
    place(dry, whoosh(0.5), 7.2 - 0.3, 0.4)
    place(dry, impact(1.6), 7.2, 0.35)
    r = np.random.default_rng(11)
    at = TYPE_START
    while at < TYPE_END:
        place(dry, tick(0.05), at, 0.12 + 0.08 * r.random(), pan=r.uniform(-0.3, 0.3))
        at += r.uniform(0.045, 0.11)
    for i, n in enumerate(["E5", "A5", "C6"]):
        place(wet_send, bell(note(n), 2.6, 1.6), BOOK_FORMED + i * 0.06, 0.22)
        place(dry, bell(note(n), 2.6, 1.6), BOOK_FORMED + i * 0.06, 0.08)
    place(dry, riser(0.9), BOOK_FORMED - 0.9, 0.3)

    # Into the end card.
    place(dry, riser(1.3), 19.2 - 1.3, 0.55)
    end_pad = pad([note("F2"), note("C3"), note("A3"), note("E4")], 2.4, attack=0.15,
                  release=1.0, cutoff=1600)
    place(dry, end_pad, 19.2, 0.5)
    place(wet_send, end_pad, 19.2, 0.4)
    final = pad([note("C3"), note("G3"), note("E4"), note("C4")], 2.0, attack=0.6,
                release=1.4, cutoff=1300)
    place(dry, final, 21.6, 0.48)
    place(wet_send, final, 21.6, 0.4)
    for i, n in enumerate(["C5", "E5", "G5", "C6"]):
        place(wet_send, bell(note(n), 2.5, 1.4), LOGO + i * 0.09, 0.22)
        place(dry, bell(note(n), 2.5, 1.4), LOGO + i * 0.09, 0.1)
    for i, n in enumerate(["A4", "C5", "E5", "G5", "E5", "C5"]):
        pl = pluck(note(n), bright=0.3)
        place(dry, pl, 19.2 + i * BEAT / 2, 0.25, pan=-0.4 if i % 2 else 0.4)
        place(wet_send, pl, 19.2 + i * BEAT / 2, 0.3)

    wet = np.stack([reverb(wet_send[0], seed=3), reverb(wet_send[1], seed=4)])
    mix = dry + 0.8 * wet
    mix = hp(mix, 28)

    # Gentle bus glue, fade tail, peak-normalize to -1 dBFS.
    mix = np.tanh(mix * 0.9) / 0.9
    t = np.arange(N) / SR
    mix *= np.where(t > DURATION - 1.4, np.clip((DURATION - t) / 1.4, 0, 1) ** 1.5, 1)
    mix *= 10 ** (-1 / 20) / np.max(np.abs(mix))
    return mix.T.astype(np.float32)


def write_wav(path, data):
    import wave
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/narra_ad/score.wav"
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_wav(out, build())
    print("score ->", out)
