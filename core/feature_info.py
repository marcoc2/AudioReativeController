"""What each audio feature a knob can read is, said for a person: what it measures, its
range, and what it listens to. A feature is read from an audio source: the song (the mix)
or any audio file of the project (a stem: ``audio:`` in the knob), and every source has the
same features (``FEATURES``): its level, its sub-bands, its texture, its brightness and
change. ``stems.<name>`` is the older way to a stem's level (the song's own analysis, with a
slow fall).

``describe(name, stems)`` -> {name, group, label, about, range: [lo, hi], unit, heard}, where
``heard`` is {kind: stem, stem} | {kind: band, lo, hi} (Hz) | {kind: mix}: what the feature
listens to inside its source.
"""
from __future__ import annotations

from typing import Optional

from core.texture import TEXTURE_NAMES, TREMOLO_RANGE

# the groups, in the order the editor lists them (by what one wants to follow)
GROUPS = ("how loud", "a band of it", "its texture", "its colour", "a stem's level (the older way)")

BANDS = {        # the sub-band: (lo Hz, hi Hz, what lives there)
    "sub_bass": (20, 60, "the felt low end: the kick's thump, the deepest bass"),
    "bass": (60, 250, "the kick's body, the bass line"),
    "low_mid": (250, 500, "warmth: the low of guitars, piano, voices"),
    "mid": (500, 2000, "the body of most instruments and the voice"),
    "high_mid": (2000, 4000, "attack and presence: the pick, consonants"),
    "presence": (4000, 6000, "edge and clarity"),
    "brilliance": (6000, 20000, "air: cymbals, breath, sparkle"),
}

TEXTURE = {      # name: (lo, hi, unit, what it measures)
    "loudness": (0.0, 1.0, "", "how loud, on a dB scale (quiet passages keep their dynamics)"),
    "swell": (-1.0, 1.0, "", "growing (+) or fading (-), and how fast (6 dB a second reads as 1)"),
    "harmonic_change": (0.0, 1.0, "", "how different the next second's harmony is from the last: peaks where the chord changes"),
    "percussive": (0.0, 1.0, "", "the share of the sound that is attack rather than sustain (0 a pad, ~1 drums alone)"),
    "noisiness": (0.0, 1.0, "", "tonal (0) to noise-like (1): a sine against wind or cymbals"),
    "tremolo_depth": (0.0, 1.0, "", "how strongly the volume pulsates"),
    "tremolo_rate": (0.0, float(TREMOLO_RANGE[1]), "Hz", "how fast the volume pulsates (0 when it does not)"),
}

# what every source has, in the editor's order
FEATURES = (["level"] + [f"subbands.{b}" for b in BANDS] + [f"texture.{n}" for n in TEXTURE_NAMES]
            + ["centroid", "flux"])


def names(stems=()) -> list:
    """Every feature a knob can read from a source, then the older stems.<name>."""
    return list(FEATURES) + [f"stems.{s}" for s in stems]


def describe(name: str, stems=()) -> Optional[dict]:
    """What the feature ``name`` is (None for one this table does not know)."""
    head, _, tail = name.partition(".")
    if name == "level":
        return {"name": name, "group": GROUPS[0], "label": "level",
                "about": "how loud it is: its RMS against its own loudest moment (follows every attack)",
                "range": [0.0, 1.0], "unit": "", "heard": {"kind": "mix"}}
    if head == "stems" and tail:
        return {"name": name, "group": GROUPS[4], "label": f"{tail}: its level, falling slowly",
                "about": f"the {tail} stem's level as the song's analysis reads it: it rises at once and falls slowly",
                "range": [0.0, 1.0], "unit": "", "heard": {"kind": "stem", "stem": tail}}
    if head == "subbands" and tail in BANDS:
        lo, hi, what = BANDS[tail]
        return {"name": name, "group": GROUPS[1], "label": f"{tail.replace('_', ' ')} ({_hz(lo)}-{_hz(hi)})",
                "about": f"how much of it is in {_hz(lo)}-{_hz(hi)}: {what} (against its own loudest moment)",
                "range": [0.0, 1.0], "unit": "", "heard": {"kind": "band", "lo": lo, "hi": hi}}
    if head == "texture" and tail in TEXTURE:
        lo, hi, unit, what = TEXTURE[tail]
        return {"name": name, "group": GROUPS[0] if tail in ("loudness", "swell") else GROUPS[2],
                "label": tail.replace("_", " "), "about": what, "range": [lo, hi], "unit": unit, "heard": {"kind": "mix"}}
    if name == "centroid":
        return {"name": name, "group": GROUPS[3], "label": "brightness",
                "about": "where the sound's weight sits from low (0) to high (1): dark against bright",
                "range": [0.0, 1.0], "unit": "", "heard": {"kind": "mix"}}
    if name == "flux":
        return {"name": name, "group": GROUPS[3], "label": "change",
                "about": "how much the spectrum changed since the frame before: high on attacks and new sounds",
                "range": [0.0, 1.0], "unit": "", "heard": {"kind": "mix"}}
    return None


def _hz(f: float) -> str:
    return f"{f / 1000:g} kHz" if f >= 1000 else f"{f:g} Hz"
