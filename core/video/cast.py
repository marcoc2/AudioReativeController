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
whole composite below it. On the GPU when the compositor has one (the picture never leaves
it); the numpy path is the same maths, kept as the reference and for machines without one.
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
FLAT = (28, 24, 34)                      # a silhouette's colour, for presence

# --- the acts on the GPU: the numpy ones in _play, pixel for pixel (within a level)

# the role's objects (0/1 at the track's size) stretched to the frame as cv2.resize
# INTER_NEAREST does: through its own lookup of which track column/row each pixel takes
_NEAR_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_sel, u_xo, u_yo;
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 q = ivec2(texelFetch(u_xo, ivec2(p.x, 0), 0).r, texelFetch(u_yo, ivec2(p.y, 0), 0).r);
    f_color = vec4(texelFetch(u_sel, q, 0).r, 0.0, 0.0, 1.0);
}
"""

# cv2.warpAffine(INTER_LINEAR, BORDER_CONSTANT 0): each pixel samples the source at the
# inverse map, taps outside it read 0 (OpenCV 5 does this in float, as here)
_WARP = """
uniform vec3 u_r0, u_r1;            // the inverse affine map, by rows
vec4 tap(sampler2D s, ivec2 q, float scale){
    ivec2 n = textureSize(s, 0);
    if (q.x < 0 || q.y < 0 || q.x >= n.x || q.y >= n.y) return vec4(0.0);
    vec4 v = texelFetch(s, q, 0);
    return scale > 1.0 ? floor(v * scale + 0.5) : v;
}
vec4 warp(sampler2D s, ivec2 p, float scale){
    vec2 x = vec2(dot(u_r0, vec3(vec2(p), 1.0)), dot(u_r1, vec3(vec2(p), 1.0)));
    vec2 f = floor(x);
    ivec2 i = ivec2(f);
    f = x - f;
    return (tap(s, i, scale) * (1.0 - f.x) + tap(s, i + ivec2(1, 0), scale) * f.x) * (1.0 - f.y)
         + (tap(s, i + ivec2(0, 1), scale) * (1.0 - f.x) + tap(s, i + ivec2(1, 1), scale) * f.x) * f.y;
}
"""

# the object's mask carried by the map; shifted (np.roll) for its shadow
_WARP_MASK_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_m;
uniform ivec2 u_shift;
""" + _WARP + """
void main(){
    ivec2 n = textureSize(u_m, 0);
    ivec2 p = (ivec2(gl_FragCoord.xy) - u_shift + n) % n;
    f_color = vec4(warp(u_m, p, 1.0).r, 0.0, 0.0, 1.0);
}
"""

# CastLayer._paste: the shadow darkens, then the carried object is laid over it
_PASTE_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_src, u_alpha, u_sh;
uniform float u_shadow;
""" + _WARP + """
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 v = floor(texelFetch(u_src, p, 0).rgb * 255.0 + 0.5);
    if (u_shadow > 0.0) v *= 1.0 - u_shadow * texelFetch(u_sh, p, 0).r;
    vec3 w = floor(warp(u_src, p, 255.0).rgb + 0.5);
    float a = texelFetch(u_alpha, p, 0).r;
    v = v * (1.0 - a) + w * a;
    f_color = vec4(floor(clamp(v + 0.5, 0.0, 255.0)) / 255.0, 1.0);
}
"""

_PRESENCE_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_src, u_m;
uniform float u_p; uniform vec3 u_flat;
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 v = floor(texelFetch(u_src, p, 0).rgb * 255.0 + 0.5);
    if (texelFetch(u_m, p, 0).r > 0.5) v = floor(v * u_p + u_flat * (1.0 - u_p) + 0.5);
    f_color = vec4(v / 255.0, 1.0);
}
"""

_TINT_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_src, u_m;
uniform vec3 u_c; uniform float u_a;
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 v = floor(texelFetch(u_src, p, 0).rgb * 255.0 + 0.5);
    if (texelFetch(u_m, p, 0).r > 0.5){
        float L = dot(v, vec3(0.299, 0.587, 0.114)) / 255.0;
        vec3 tinted = clamp(u_c * (0.25 + 1.1 * L), 0.0, 255.0);
        v = floor(clamp(v * (1.0 - u_a) + tinted * u_a + 0.5, 0.0, 255.0));
    }
    f_color = vec4(v / 255.0, 1.0);
}
"""

_GLOW_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_src, u_m, u_halo;
uniform vec3 u_c; uniform float u_k;
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 v = floor(texelFetch(u_src, p, 0).rgb * 255.0 + 0.5);
    float h = texelFetch(u_m, p, 0).r > 0.5 ? 0.0 : texelFetch(u_halo, p, 0).r;
    v = floor(clamp(v + u_c * (h * u_k) + 0.5, 0.0, 255.0));
    f_color = vec4(v / 255.0, 1.0);
}
"""


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
        self._lookups = {}
        self.on_gpu = True

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

    def _lookup(self, shape):
        """For the track's size: the track column each frame column takes and the track row
        each frame row takes (cv2.resize INTER_NEAREST's own choice), plus how many frame
        pixels each track column/row stands for and the sum of their coordinates."""
        if shape not in self._lookups:
            h, w = shape
            xo = cv2.resize(np.arange(w, dtype=np.float32)[None, :], (self.W, 1),
                            interpolation=cv2.INTER_NEAREST)[0].astype(np.int64)
            yo = cv2.resize(np.arange(h, dtype=np.float32)[:, None], (1, self.H),
                            interpolation=cv2.INTER_NEAREST)[:, 0].astype(np.int64)
            self._lookups[shape] = {
                "xo": xo, "yo": yo, "cols": np.unique(xo), "rows": np.unique(yo),
                "xn": np.bincount(xo, minlength=w).astype(float),
                "xs": np.bincount(xo, weights=np.arange(self.W), minlength=w),
                "yn": np.bincount(yo, minlength=h).astype(float),
                "ys": np.bincount(yo, weights=np.arange(self.H), minlength=h)}
        return self._lookups[shape]

    # ----------------------------------------------------------------- acts
    def _step(self, role: _Role, t: float, centre) -> Optional[tuple]:
        """What ``role`` does to its object at ``t``, whichever side paints it: (kind, params),
        or None when it leaves the picture alone. ``centre()`` gives the object's centre
        (asked for by a punch only)."""
        if role.act == "presence":
            p = role.presence(t)
            return None if p >= 1.0 else ("presence", {"p": p})
        e = role.env(t)
        if e <= 0.0:
            return None
        if role.act == "punch":
            return "paste", {"M": cv2.getRotationMatrix2D(centre(), 0.0, 1.0 + role.amount * e),
                             "shadow": 0.45 * e}
        if role.act == "shake":
            k = role.last_hit(t) or 0
            ang = np.random.default_rng(role._seed * 7919 + k).uniform(0, 2 * np.pi)
            d = role.amount * e
            return "paste", {"M": np.float32([[1, 0, d * np.cos(ang)], [0, 1, d * np.sin(ang)]]),
                             "shadow": 0.0}
        if role.act == "tint":
            return "tint", {"c": role.colour(t), "a": min(1.0, role.amount) * e}
        return "glow", {"c": role.colour(t), "w": max(1, int(round(role.width))), "k": e * role.amount}

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

        def centre():
            ys, xs = np.nonzero(m)
            return float(xs.mean()), float(ys.mean())

        step = self._step(role, t, centre)
        if step is None:
            return out
        kind, a = step
        if kind == "paste":
            return self._paste(out, out.copy(), m, a["M"], a["shadow"])
        res = out.astype(np.float32)
        if kind == "presence":
            p = a["p"]
            res[m] = res[m] * p + np.array(FLAT, np.float32) * (1.0 - p)
            return (res + 0.5).astype(np.uint8)
        if kind == "tint":
            L = res[m].dot(np.array([0.299, 0.587, 0.114], np.float32))[:, None] / 255.0
            tinted = np.clip(a["c"][None, :] * (0.25 + 1.1 * L), 0, 255)
            res[m] = res[m] * (1.0 - a["a"]) + tinted * a["a"]
        else:  # glow
            w = a["w"]
            ring = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w + 1,) * 2))
            halo = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), w / 2.0) * (~m)
            res += a["c"][None, None, :] * (halo * a["k"])[..., None]
        return np.clip(res + 0.5, 0, 255).astype(np.uint8)

    def _play_gpu(self, g, kind: str, a: dict, out, m):
        """``_play`` on the GPU: ``out`` the Frame below, ``m`` the objects' mask texture."""
        from core.video.gpu_compose import Frame
        src = out.tex()
        if kind == "presence":
            tex = g.run("cast_presence", g.texture(), {"u_src": src, "u_m": m}, fs=_PRESENCE_FS,
                        u_p=float(a["p"]), u_flat=tuple(float(c) for c in FLAT))
        elif kind == "tint":
            tex = g.run("cast_tint", g.texture(), {"u_src": src, "u_m": m}, fs=_TINT_FS,
                        u_c=tuple(float(c) for c in a["c"]), u_a=float(a["a"]))
        elif kind == "glow":
            halo = g.blur(g.dilate(m, a["w"]), a["w"] / 2.0)
            tex = g.run("cast_glow", g.texture(), {"u_src": src, "u_m": m, "u_halo": halo}, fs=_GLOW_FS,
                        u_c=tuple(float(c) for c in a["c"]), u_k=float(a["k"]))
        else:  # paste
            inv = cv2.invertAffineTransform(np.float64(a["M"]))
            rows = {"u_r0": tuple(float(v) for v in inv[0]), "u_r1": tuple(float(v) for v in inv[1])}
            carried = g.run("cast_warp_mask", g.texture(1, "f4"), {"u_m": m}, fs=_WARP_MASK_FS,
                            u_shift=(0, 0), **rows)
            alpha = g.blur(carried, 0.8)
            sh = alpha
            if a["shadow"] > 0:
                off = max(2, self.H // 90)
                sh = g.blur(g.run("cast_warp_mask", g.texture(1, "f4"), {"u_m": m}, u_shift=(off, off), **rows), off)
            tex = g.run("cast_paste", g.texture(), {"u_src": src, "u_alpha": alpha, "u_sh": sh}, fs=_PASTE_FS,
                        u_shadow=float(a["shadow"]), **rows)
        return Frame(g, tex=tex)

    def process_gpu(self, frame, t: float, g):
        """``process`` with the picture on the GPU: only the objects' masks, at the track's
        size, go up."""
        i = self.track.frame_at(t)
        if i is None:
            return frame
        labels = self.track.labels(i)
        lk = self._lookup(labels.shape)
        tables = None
        out = frame
        for role in self.roles:
            ids = self._cast(role.who, i)
            if ids is not None and not ids:
                continue
            sel = (labels == 0) if ids is None else np.isin(labels, ids)
            if not sel[np.ix_(lk["rows"], lk["cols"])].any():   # nothing of it lands on the frame
                continue

            def centre():
                s = sel.astype(float)
                n = lk["yn"] @ s @ lk["xn"]
                return float(lk["yn"] @ s @ lk["xs"] / n), float(lk["ys"] @ s @ lk["xn"] / n)

            step = self._step(role, t, centre)
            if step is None:
                continue
            if tables is None:
                tables = {"u_xo": g.small(lk["xo"][None, :].astype(np.float32), "cast_xo"),
                          "u_yo": g.small(lk["yo"][None, :].astype(np.float32), "cast_yo")}
            m = g.run("cast_near", g.texture(1, "f4"),
                      {"u_sel": g.small(sel.astype(np.uint8) * 255, "cast_sel"), **tables}, fs=_NEAR_FS)
            out = self._play_gpu(g, step[0], step[1], out, m)
        return out

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
