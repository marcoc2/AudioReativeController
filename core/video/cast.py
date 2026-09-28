"""Cast — each instrument gets a body on screen: the objects of a mask track, cast by role.

The objects of a mask track change from shot to shot, so the cast is written by role,
not by id: in every shot the roles are filled again from what that shot holds.

    - source: cast
      track: render_output/x_base_segments     # or the render's --masks
      roles:
        - {who: largest,    plays: {track: kick},     act: punch, amount: 0.12}
        - {who: second,     plays: {track: snare},    act: shake, amount: 10}
        - {who: third,      plays: {track: ez-hihat}, act: glow, color: [255, 240, 180]}
        - {who: background, plays: {track: bass},     act: tint, color: chord,
           harmony: {score: input/song/sheetsage, snap: auto}}
        - {who: largest,    plays: {track: synth-voice}, act: presence, hold: 2.0}

who     largest | second | third | fourth (by their mean size in the shot) |
        leftmost | rightmost | topmost | bottommost (by where they sit) |
        all (every object) | background (what no object covers) | [ids]
plays   the instrument: the usual trigger spec (track / notes / audio / score)
act     punch     the object jumps out of the paper on each hit, bigger, with a shadow
                  (amount: extra scale at the hit)
        shake     it jolts sideways on each hit, leaving its ghost behind (amount: pixels at 720p)
        tint      it takes a colour on each hit, keeping its shading (amount: 0..1)
        glow      a halo round its outline on each hit (width: pixels at 720p)
        presence  it is a flat silhouette until its instrument plays, and goes back to one
                  when it has been quiet for ``hold`` seconds (fade: seconds to change)
envelope  seconds a hit takes to die away (default by act)
color     [r, g, b] | chord (the chord's hue, needs harmony:)

Roles are played in order, each on the picture the previous left; a post-op, it sees the
whole composite below it. CPU (numpy/OpenCV) for now.
"""
from __future__ import annotations

import colorsys
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from core.segment.mask_track import MaskTrack

ACTS = ("punch", "shake", "tint", "glow", "presence")
RANKS = {"largest": 0, "second": 1, "third": 2, "fourth": 3}
PLACES = {"leftmost": (1, 1), "rightmost": (1, -1), "topmost": (2, 1), "bottommost": (2, -1)}
ENVELOPE = {"punch": 0.25, "shake": 0.15, "tint": 0.4, "glow": 0.3, "presence": 0.0}


class _Role:
    def __init__(self, spec: dict, notes, onset_loader, grid, scale: float, index: int):
        from core.video.layers import EnvelopeOpacity, VeilsLayer, _layer_events
        self.who = spec.get("who", "largest")
        if not (isinstance(self.who, list) or self.who in RANKS or self.who in PLACES
                or self.who in ("all", "background")):
            raise ValueError(f"cast role {index}: who {self.who!r} (use {sorted(RANKS)}, "
                             f"{sorted(PLACES)}, all, background or a list of ids)")
        self.act = spec.get("act", "punch")
        if self.act not in ACTS:
            raise ValueError(f"cast role {index}: act {self.act!r} (use one of {ACTS})")
        if "plays" not in spec:
            raise ValueError(f"cast role {index}: plays: names its instrument ({{track: kick}})")
        if self.act in ("punch", "shake") and self.who == "background":
            raise ValueError(f"cast role {index}: the background cannot {self.act}")
        events = _layer_events(spec["plays"], notes, onset_loader, grid)
        self.times = np.array([e.time for e in events], float)
        self.env = EnvelopeOpacity(self.times, float(spec.get("envelope", ENVELOPE[self.act]) or 1e-3))
        default = {"punch": 0.12, "shake": 10.0, "tint": 0.7, "glow": 1.0, "presence": 1.0}[self.act]
        self.amount = float(spec.get("amount", default))
        if self.act == "shake":
            self.amount *= scale
        self.width = float(spec.get("width", 8)) * scale
        self.hold = float(spec.get("hold", 2.0))
        self.fade = max(1e-3, float(spec.get("fade", 0.3)))
        self.color = spec.get("color", [255, 220, 120] if self.act == "glow" else [255, 60, 60])
        self._chords, self._chord_times, self._hue_of_root = [], np.array([]), VeilsLayer.hue_of_root
        if self.color == "chord":
            h = spec.get("harmony")
            if not h:
                raise ValueError(f"cast role {index}: color: chord needs harmony:")
            self._chords = _layer_events(h, notes, onset_loader, grid)
            self._chord_times = np.array([e.time for e in self._chords], float)
        self._seed = 1000 + index

    def colour(self, t: float) -> np.ndarray:
        if self.color == "chord":
            i = int(np.searchsorted(self._chord_times, t, side="right")) - 1
            hue = self._hue_of_root(self._chords[i].pitch) if i >= 0 else 0.0
            return np.array(colorsys.hsv_to_rgb(hue, 0.85, 1.0), np.float32) * 255.0
        return np.array(self.color, np.float32)

    def last_hit(self, t: float) -> Optional[int]:
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        return i if i >= 0 else None

    def presence(self, t: float) -> float:
        """1 while the instrument is playing (a hit within ``hold`` s), fading in/out over ``fade``."""
        i = self.last_hit(t)
        if i is None:
            return 0.0
        since = t - self.times[i]
        fade_in = min(1.0, since / self.fade) if since < self.fade else 1.0
        # how long it had been quiet before this hit decides whether it fades in again
        if i > 0 and self.times[i] - self.times[i - 1] < self.hold:
            fade_in = 1.0
        if since <= self.hold:
            return fade_in
        return max(0.0, 1.0 - (since - self.hold) / self.fade)


class CastLayer:
    """Post-op: the roles above, played on the objects of a mask track."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int, fps: float,
                 onset_loader=None, grid=None, default_track: Optional[str] = None,
                 tracks: Optional[Dict[str, MaskTrack]] = None):
        folder = spec.get("track") or default_track
        if not folder:
            raise ValueError("cast: needs a mask track (track: <folder>) or the render's --masks")
        tracks = tracks if tracks is not None else {}
        if folder not in tracks:
            tracks[folder] = MaskTrack(folder)
        self.track = tracks[folder]
        self.W, self.H = width, height
        scale = height / 720.0
        roles = spec.get("roles") or []
        if not roles:
            raise ValueError("cast: roles: is empty — who plays what?")
        self.roles = [_Role(r, notes, onset_loader, grid, scale, i + 1) for i, r in enumerate(roles)]
        self._ranking = {}

    # ------------------------------------------------------------ who is who
    def _shot_objects(self, shot: int) -> List[dict]:
        """The shot's objects with their mean size and place over the frames they are seen."""
        if shot not in self._ranking:
            s = self.track.shots[shot]
            rows = []
            for oid, o in self.track.objects.items():
                if o["shot"] != shot:
                    continue
                st = self.track.stats[s["first"]:s["last"] + 1, oid]
                seen = st[:, 0] > 0
                if not seen.any():
                    continue
                rows.append({"id": oid, "share": float(st[seen, 0].mean()),
                             "cx": float(st[seen, 1].mean()), "cy": float(st[seen, 2].mean())})
            self._ranking[shot] = rows
        return self._ranking[shot]

    def _shot_of(self, i: int) -> int:
        for k, s in enumerate(self.track.shots):
            if s["first"] <= i <= s["last"]:
                return k
        return len(self.track.shots) - 1

    def _cast(self, who, i: int) -> Optional[List[int]]:
        """The ids playing ``who`` on frame ``i`` (None: the background)."""
        if who == "background":
            return None
        if isinstance(who, list):
            return [int(w) for w in who]
        objs = self._shot_objects(self._shot_of(i))
        if who == "all":
            return [o["id"] for o in objs]
        if who in RANKS:
            ranked = sorted(objs, key=lambda o: -o["share"])
            k = RANKS[who]
            return [ranked[k]["id"]] if k < len(ranked) else []
        axis, sign = PLACES[who]
        key = "cx" if axis == 1 else "cy"
        ranked = sorted(objs, key=lambda o: sign * o[key])
        return [ranked[0]["id"]] if ranked else []

    def _mask(self, labels: np.ndarray, ids: Optional[List[int]]) -> np.ndarray:
        m = (labels == 0) if ids is None else np.isin(labels, ids)
        if m.shape != (self.H, self.W):
            m = cv2.resize(m.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST) > 0
        return m

    # ----------------------------------------------------------------- acts
    @staticmethod
    def _paste(out: np.ndarray, src: np.ndarray, m: np.ndarray, M: np.ndarray, shadow: float):
        """Warp the object (``src`` where ``m``) by the 2×3 affine ``M`` and lay it on ``out``,
        with a soft drop shadow under it."""
        H, W = m.shape
        warped_m = cv2.warpAffine(m.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
        warped = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR)
        res = out.astype(np.float32)
        if shadow > 0:
            off = max(2, H // 90)
            sh = cv2.GaussianBlur(np.roll(warped_m, (off, off), axis=(0, 1)), (0, 0), off)
            res *= (1.0 - shadow * sh)[..., None]
        a = cv2.GaussianBlur(warped_m, (0, 0), 0.8)[..., None]
        res = res * (1.0 - a) + warped.astype(np.float32) * a
        return np.clip(res + 0.5, 0, 255).astype(np.uint8)

    def _play(self, role: _Role, out: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        if not m.any():
            return out
        if role.act == "presence":
            p = role.presence(t)
            if p >= 1.0:
                return out
            flat = np.array([28, 24, 34], np.float32)
            res = out.astype(np.float32)
            res[m] = res[m] * p + flat * (1.0 - p)
            return (res + 0.5).astype(np.uint8)
        e = role.env(t)
        if e <= 0.0:
            return out
        if role.act == "punch":
            ys, xs = np.nonzero(m)
            M = cv2.getRotationMatrix2D((float(xs.mean()), float(ys.mean())), 0.0, 1.0 + role.amount * e)
            return self._paste(out, out.copy(), m, M, 0.45 * e)
        if role.act == "shake":
            k = role.last_hit(t) or 0
            ang = np.random.default_rng(role._seed * 7919 + k).uniform(0, 2 * np.pi)
            d = role.amount * e
            M = np.float32([[1, 0, d * np.cos(ang)], [0, 1, d * np.sin(ang)]])
            return self._paste(out, out.copy(), m, M, 0.0)
        c = role.colour(t)
        res = out.astype(np.float32)
        if role.act == "tint":
            L = res[m].dot(np.array([0.299, 0.587, 0.114], np.float32))[:, None] / 255.0
            tinted = np.clip(c[None, :] * (0.25 + 1.1 * L), 0, 255)
            a = min(1.0, role.amount) * e
            res[m] = res[m] * (1.0 - a) + tinted * a
        else:  # glow
            w = max(1, int(round(role.width)))
            ring = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w + 1,) * 2))
            halo = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), w / 2.0) * (~m)
            res += c[None, None, :] * (halo * e * role.amount)[..., None]
        return np.clip(res + 0.5, 0, 255).astype(np.uint8)

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        i = self.track.frame_at(t)
        if i is None:
            return frame
        labels = self.track.labels(i)
        out = frame
        for role in self.roles:
            ids = self._cast(role.who, i)
            if ids is not None and not ids:
                continue                     # nobody for that role in this shot
            out = self._play(role, out, self._mask(labels, ids), t)
        return out
