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
        - {who: second,     plays: {track: acid},     act: rings, speed: 300}
        - {who: largest,    plays: {track: lead},     act: echo, trail: 3}
        - {who: largest,    plays: {track: snare},    act: carry}

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
        rings     each note sends a ring out from its outline, like sonar: low notes thick,
                  loud ones bright (speed: pixels/s at 720p, life: seconds, width: pixels
                  at 720p; color defaults to the note's own hue)
        echo      while a note sounds it leaves ghosts of itself behind, one every ``every``
                  seconds, each fading over ``trail`` times the note's length: a long note
                  leaves a long trail, staccato a short one (amount: the ghosts' opacity;
                  color: tint them, onion-skin style)
        carry     at a cut it stays: the object as last seen in the shot that ended, stuck
                  over the new one like a sticker (with a shadow), until its instrument's
                  next hit, then it dissolves (hold: seconds at most, fade: seconds to go,
                  shadow: 0..1)
envelope  seconds a hit takes to die away (default by act)
color     [r, g, b] | chord (the chord's hue, needs harmony:) | pitch (the note's hue)

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

ACTS = ("punch", "shake", "tint", "glow", "presence", "rings", "echo", "carry")
RANKS = {"largest": 0, "second": 1, "third": 2, "fourth": 3}
PLACES = {"leftmost": (1, 1), "rightmost": (1, -1), "topmost": (2, 1), "bottommost": (2, -1)}
ENVELOPE = {"punch": 0.25, "shake": 0.15, "tint": 0.4, "glow": 0.3, "presence": 0.0,
            "rings": 0.0, "echo": 0.0, "carry": 0.0}
FLAT = (28, 24, 34)                      # a silhouette's colour, for presence
MAX_RINGS = 32                           # rings out at once, per role
MAX_GHOSTS = 16                          # ghosts kept, per role
LUMA = (0.299, 0.587, 0.114)
IDENTITY = np.float32([[1, 0, 0], [0, 1, 0]])

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

# CastLayer._paste: the shadow darkens, then the object (from u_obj: the same picture, or
# one kept from before) is carried by the map and laid over it
_PASTE_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_src, u_obj, u_alpha, u_sh;
uniform float u_shadow;
""" + _WARP + """
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 v = floor(texelFetch(u_src, p, 0).rgb * 255.0 + 0.5);
    if (u_shadow > 0.0) v *= 1.0 - u_shadow * texelFetch(u_sh, p, 0).r;
    vec3 w = floor(warp(u_obj, p, 255.0).rgb + 0.5);
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

# the object's distance field (at the track's size) stretched as cv2.resize INTER_LINEAR
# does, and the rings riding it, outside the object only: each ring mixes its colour in
# (not added: on white paper an added colour would vanish)
_RINGS_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_src, u_m, u_dist;
uniform vec2 u_size; uniform float u_scale;
uniform int u_n;
uniform vec4 u_ring[%d];            // radius, thickness, strength
uniform vec4 u_col[%d];
float at(vec2 q){
    ivec2 n = textureSize(u_dist, 0);
    vec2 s = clamp((q + 0.5) * vec2(n) / u_size - 0.5, vec2(0.0), vec2(n - 1));
    ivec2 i0 = ivec2(floor(s)), i1 = min(i0 + 1, n - 1);
    vec2 f = s - vec2(i0);
    float a = texelFetch(u_dist, i0, 0).r, b = texelFetch(u_dist, ivec2(i1.x, i0.y), 0).r;
    float c = texelFetch(u_dist, ivec2(i0.x, i1.y), 0).r, d = texelFetch(u_dist, i1, 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 v = floor(texelFetch(u_src, p, 0).rgb * 255.0 + 0.5);
    if (texelFetch(u_m, p, 0).r < 0.5){
        float d = at(floor(gl_FragCoord.xy)) * u_scale;
        for (int i = 0; i < u_n; i++){
            float x = (d - u_ring[i].x) / u_ring[i].y;
            float a = min(1.0, u_ring[i].z * exp(-x * x));
            v = v * (1.0 - a) + u_col[i].rgb * a;
        }
    }
    f_color = vec4(floor(clamp(v + 0.5, 0.0, 255.0)) / 255.0, 1.0);
}
""" % (MAX_RINGS, MAX_RINGS)

# one ghost laid over what is built so far (kept unrounded, in a float texture)
_GHOST_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_acc, u_ghost, u_gm;
uniform int u_first, u_tint;
uniform float u_a; uniform vec3 u_c;
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 v = texelFetch(u_acc, p, 0).rgb;
    if (u_first == 1) v = floor(v * 255.0 + 0.5);
    vec3 g = floor(texelFetch(u_ghost, p, 0).rgb * 255.0 + 0.5);
    if (u_tint == 1) g = clamp(u_c * (0.25 + 1.1 * dot(g, vec3(0.299, 0.587, 0.114)) / 255.0), 0.0, 255.0);
    float k = u_a * texelFetch(u_gm, p, 0).r;
    f_color = vec4(v * (1.0 - k) + g * k, 1.0);
}
"""

# and the object itself back on top of its ghosts
_ECHO_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_acc, u_src, u_am;
uniform int u_first;                // no ghost laid: u_acc is the picture itself
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 v = texelFetch(u_acc, p, 0).rgb;
    if (u_first == 1) v = floor(v * 255.0 + 0.5);
    vec3 s = floor(texelFetch(u_src, p, 0).rgb * 255.0 + 0.5);
    float a = texelFetch(u_am, p, 0).r;
    v = v * (1.0 - a) + s * a;
    f_color = vec4(floor(clamp(v + 0.5, 0.0, 255.0)) / 255.0, 1.0);
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
        if self.act in ("punch", "shake", "rings", "echo", "carry") and self.who == "background":
            raise ValueError(f"cast role {index}: the background cannot {self.act}")
        events = _layer_events(spec["plays"], notes, onset_loader, grid)
        self.events = events
        self.times = np.array([e.time for e in events], float)
        self.env = EnvelopeOpacity(self.times, float(spec.get("envelope", ENVELOPE[self.act]) or 1e-3))
        default = {"punch": 0.12, "shake": 10.0, "tint": 0.7, "glow": 1.0, "presence": 1.0,
                   "rings": 1.0, "echo": 0.6, "carry": 1.0}[self.act]
        self.amount = float(spec.get("amount", default))
        if self.act == "shake":
            self.amount *= scale
        self.width = float(spec.get("width", 3 if self.act == "rings" else 8)) * scale
        self.hold = float(spec.get("hold", 4.0 if self.act == "carry" else 2.0))
        self.fade = max(1e-3, float(spec.get("fade", 0.4 if self.act == "carry" else 0.3)))
        self.shadow = float(spec.get("shadow", 0.35))
        self.speed = float(spec.get("speed", 300)) * scale
        self.life = max(1e-3, float(spec.get("life", 1.0)))
        self.every = float(spec.get("every", 0.08))
        self.trail = float(spec.get("trail", 3.0))
        colour = {"glow": [255, 220, 120], "rings": "pitch", "echo": None}.get(self.act, [255, 60, 60])
        self.color = spec.get("color", colour)
        self.ghosts: List[dict] = []         # echo: what it has left behind
        self._spare: List[tuple] = []        # echo: GPU textures of ghosts gone, for new ones
        self._caught, self._caught_at, self._shot, self._last_t = None, 0.0, None, None
        self.snap: Optional[dict] = None     # carry: the object as last seen in this shot
        self.carried: Optional[dict] = None  # carry: the one brought over the cut
        self._cut_t = 0.0
        self._chords, self._chord_times, self._hue_of_root = [], np.array([]), VeilsLayer.hue_of_root
        if self.color == "chord":
            h = spec.get("harmony")
            if not h:
                raise ValueError(f"cast role {index}: color: chord needs harmony:")
            self._chords = _layer_events(h, notes, onset_loader, grid)
            self._chord_times = np.array([e.time for e in self._chords], float)
        self._seed = 1000 + index

    def colour(self, t: float, pitch: Optional[int] = None) -> np.ndarray:
        if self.color == "pitch":
            hue = self._hue_of_root(60 if pitch is None else pitch)
            return np.array(colorsys.hsv_to_rgb(hue, 0.8, 1.0), np.float32) * 255.0
        if self.color == "chord":
            i = int(np.searchsorted(self._chord_times, t, side="right")) - 1
            hue = self._hue_of_root(self._chords[i].pitch) if i >= 0 else 0.0
            return np.array(colorsys.hsv_to_rgb(hue, 0.85, 1.0), np.float32) * 255.0
        return np.array(self.color, np.float32)

    def last_hit(self, t: float) -> Optional[int]:
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        return i if i >= 0 else None

    def rings(self, t: float) -> List[tuple]:
        """The rings out at ``t``: (radius, thickness, strength, colour) for each note
        younger than ``life``."""
        lo = int(np.searchsorted(self.times, t - self.life, side="right"))
        hi = int(np.searchsorted(self.times, t, side="right"))
        out = []
        for e in self.events[max(lo, hi - MAX_RINGS):hi]:
            age = t - e.time
            pitch = int(getattr(e, "pitch", 60))
            thick = self.width * float(np.clip(2.0 ** ((60 - pitch) / 24.0), 0.5, 3.0))
            k = self.amount * getattr(e, "velocity", 100) / 127.0 * (1.0 - age / self.life) ** 2
            out.append((self.speed * age, thick, k, self.colour(t, pitch)))
        return out

    def forget(self, t: float, shot: int) -> None:
        """Echo: ghosts past their life go, and all of them at a cut or a jump back in time."""
        if shot != self._shot or (self._last_t is not None and t < self._last_t):
            gone, self.ghosts, self._caught = self.ghosts, [], None
        else:
            gone = [g for g in self.ghosts if t - g["born"] >= g["life"]]
            self.ghosts = [g for g in self.ghosts if t - g["born"] < g["life"]]
        self._spare += [g["data"] for g in gone if g["side"] == "gpu"]
        self._shot, self._last_t = shot, t

    def capture(self, t: float) -> Optional[dict]:
        """Echo: a ghost to leave now, or None. While a note sounds, one every ``every`` s
        (and always one per note, however short)."""
        i = self.last_hit(t)
        if i is None:
            return None
        e = self.events[i]
        dur = max(float(getattr(e, "duration", 0.0) or 0.0), 0.05)
        if t - e.time > dur or (self._caught == i and t - self._caught_at < self.every):
            return None
        self._caught, self._caught_at = i, t
        return {"born": t, "life": float(np.clip(dur * self.trail, 0.15, 3.0)),
                "a0": self.amount * getattr(e, "velocity", 100) / 127.0}

    def keep(self, ghost: dict) -> None:
        self.ghosts.append(ghost)
        while len(self.ghosts) > MAX_GHOSTS:
            old = self.ghosts.pop(0)
            if old["side"] == "gpu":
                self._spare.append(old["data"])

    def _drop(self, kept: Optional[dict]) -> None:
        if kept is not None and kept["side"] == "gpu":
            self._spare.append(kept["data"])

    def carrying(self, t: float, shot: int) -> float:
        """Carry: at a cut, what was last seen of the object becomes the carried one; how
        much of it shows at ``t`` (1 until the next hit after the cut, or ``hold`` s, then
        fading out over ``fade``)."""
        if self._last_t is not None and t < self._last_t:          # a jump back: start over
            self._drop(self.snap)
            self._drop(self.carried)
            self.snap = self.carried = self._shot = None
        if self._shot is not None and shot != self._shot:
            self._drop(self.carried)
            self.carried = self.snap if self.snap is not None and self.snap["shot"] == self._shot else None
            if self.carried is None:
                self._drop(self.snap)
            self.snap, self._cut_t = None, t
        self._shot, self._last_t = shot, t
        if self.carried is None:
            return 0.0
        k = int(np.searchsorted(self.times, self._cut_t + 0.05))    # a hit on the cut itself doesn't count
        end = min(self.times[k] if k < len(self.times) else np.inf, self._cut_t + self.hold)
        a = 1.0 if t < end else 1.0 - (t - end) / self.fade
        if a <= 0.0:
            self._drop(self.carried)
            self.carried = None
            return 0.0
        return a

    def spare_for(self, g) -> tuple:
        """Carry/echo: a picture and a mask texture on ``g`` for something to keep."""
        for d in self._spare:
            if d[2] is g:
                self._spare.remove(d)
                return d
        return g.keep(3, "f1"), g.keep(1, "f4"), g

    @staticmethod
    def fading(ghost: dict, t: float) -> float:
        return ghost["a0"] * max(0.0, 1.0 - (t - ghost["born"]) / ghost["life"])

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

    def _mask(self, sel: np.ndarray) -> np.ndarray:
        """The objects picked at the track's size, stretched to the frame."""
        if sel.shape != (self.H, self.W):
            return cv2.resize(sel.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST) > 0
        return sel

    @staticmethod
    def _distance(sel: np.ndarray) -> np.ndarray:
        """Rings: how far each pixel is from the objects, at the track's size."""
        return cv2.distanceTransform((~sel).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

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
    def _step(self, role: _Role, t: float, centre, shot: int) -> Optional[tuple]:
        """What ``role`` does to its object at ``t``, whichever side paints it: (kind, params),
        or None when it leaves the picture alone. ``centre()`` gives the object's centre
        (asked for by a punch only)."""
        if role.act == "rings":
            live = role.rings(t)
            return ("rings", {"rings": live}) if live else None
        if role.act == "echo":
            role.forget(t, shot)
            cap = role.capture(t)
            if cap is None and not role.ghosts:
                return None
            return "echo", {"capture": cap, "role": role,
                            "tint": None if role.color is None else role.colour(t)}
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

    def _play(self, role: _Role, out: np.ndarray, sel: np.ndarray, t: float, shot: int) -> np.ndarray:
        m = self._mask(sel)
        if not m.any():
            return out

        def centre():
            ys, xs = np.nonzero(m)
            return float(xs.mean()), float(ys.mean())

        step = self._step(role, t, centre, shot)
        if step is None:
            return out
        kind, a = step
        if kind == "paste":
            return self._paste(out, out.copy(), m, a["M"], a["shadow"])
        res = out.astype(np.float32)
        if kind == "rings":
            d = self._distance(sel)
            if d.shape != (self.H, self.W):
                d = cv2.resize(d, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            d = d * np.float32(self.H / sel.shape[0])
            away = ~m
            for r, thick, k, c in a["rings"]:
                x = (d - np.float32(r)) / np.float32(thick)
                k = np.minimum(np.float32(1.0), np.float32(k) * np.exp(-x * x))[..., None]
                res = np.where(away[..., None], res * (1.0 - k) + c[None, None, :] * k, res)
            return np.clip(res + 0.5, 0, 255).astype(np.uint8)
        if kind == "echo":
            am = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)
            if a["capture"] is not None:
                role.keep({**a["capture"], "side": "cpu", "data": (out.copy(), am)})
            for g in role.ghosts:
                if g["side"] != "cpu":
                    continue
                img, gm = g["data"]
                img = img.astype(np.float32)
                if a["tint"] is not None:
                    L = img.dot(np.array(LUMA, np.float32))[..., None] / 255.0
                    img = np.clip(a["tint"][None, None, :] * (0.25 + 1.1 * L), 0, 255)
                k = (np.float32(role.fading(g, t)) * gm)[..., None]
                res = res * (1.0 - k) + img * k
            res = res * (1.0 - am[..., None]) + out.astype(np.float32) * am[..., None]
            return np.clip(res + 0.5, 0, 255).astype(np.uint8)
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

    def _play_gpu(self, g, kind: str, a: dict, out, m, sel, t: float):
        """``_play`` on the GPU: ``out`` the Frame below, ``m`` the objects' mask texture."""
        from core.video.gpu_compose import Frame
        src = out.tex()
        if kind == "rings":
            ring = np.zeros((MAX_RINGS, 4), np.float32)
            col = np.zeros((MAX_RINGS, 4), np.float32)
            for n, (r, thick, k, c) in enumerate(a["rings"]):
                ring[n, :3] = r, thick, k
                col[n, :3] = c
            tex = g.run("cast_rings", g.texture(), {"u_src": src, "u_m": m,
                        "u_dist": g.small(self._distance(sel), "cast_dist")}, fs=_RINGS_FS,
                        u_size=(float(self.W), float(self.H)), u_scale=float(np.float32(self.H / sel.shape[0])),
                        u_n=len(a["rings"]), u_ring=ring, u_col=col)
            return Frame(g, tex=tex)
        if kind == "echo":
            role = a["role"]
            am = g.blur(m, 0.8)
            if a["capture"] is not None:
                spare = [d for d in role._spare if d[2] is g]
                if spare:
                    data = spare[0]
                    role._spare.remove(data)
                else:
                    data = (g.keep(3, "f1"), g.keep(1, "f4"), g)
                g.copy(src, data[0])
                g.copy(am, data[1])
                role.keep({**a["capture"], "side": "gpu", "data": data})
            acc = None
            tint = {"u_tint": int(a["tint"] is not None),
                    "u_c": tuple(float(c) for c in (a["tint"] if a["tint"] is not None else (0, 0, 0)))}
            for gh in role.ghosts:
                if gh["side"] != "gpu" or gh["data"][2] is not g:
                    continue
                layers = {"u_acc": src if acc is None else acc, "u_ghost": gh["data"][0], "u_gm": gh["data"][1]}
                acc = g.run("cast_ghost", g.texture(4, "f4"), layers, fs=_GHOST_FS, u_first=int(acc is None),
                            u_a=float(np.float32(role.fading(gh, t))), **tint)
            tex = g.run("cast_echo", g.texture(), {"u_acc": src if acc is None else acc, "u_src": src, "u_am": am},
                        fs=_ECHO_FS, u_first=int(acc is None))
            return Frame(g, tex=tex)
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
        else:
            tex = self._paste_gpu(g, src, src, m, a["M"], a["shadow"])
        return Frame(g, tex=tex)

    def _paste_gpu(self, g, below, obj, m, M: np.ndarray, shadow: float):
        """``_paste`` on the GPU: the object of picture ``obj`` (mask ``m``) laid on ``below``."""
        inv = cv2.invertAffineTransform(np.float64(M))
        rows = {"u_r0": tuple(float(v) for v in inv[0]), "u_r1": tuple(float(v) for v in inv[1])}
        carried = g.run("cast_warp_mask", g.texture(1, "f4"), {"u_m": m}, fs=_WARP_MASK_FS,
                        u_shift=(0, 0), **rows)
        alpha = g.blur(carried, 0.8)
        sh = alpha
        if shadow > 0:
            off = max(2, self.H // 90)
            shifted = g.run("cast_warp_mask", g.texture(1, "f4"), {"u_m": m}, u_shift=(off, off), **rows)
            sh = g.blur(shifted, off)
        return g.run("cast_paste", g.texture(), {"u_src": below, "u_obj": obj, "u_alpha": alpha, "u_sh": sh},
                     fs=_PASTE_FS, u_shadow=float(shadow), **rows)

    def _carry_gpu(self, g, role: _Role, out, m, t: float, shot: int):
        """``_carry`` on the GPU (``m`` None: the object is not in this frame)."""
        from core.video.gpu_compose import Frame
        a = role.carrying(t, shot)
        res = out
        if a > 0.0 and role.carried["side"] == "gpu" and role.carried["data"][2] is g:
            img, cm, _ = role.carried["data"]
            pasted = self._paste_gpu(g, out.tex(), img, cm, IDENTITY, role.shadow)
            res = Frame(g, tex=g.blend(out.tex(), pasted, "normal", a))
        if m is not None:
            snap = role.snap
            data = snap["data"] if snap is not None and snap["side"] == "gpu" and snap["data"][2] is g \
                else role.spare_for(g)
            g.copy(out.tex(), data[0])
            g.copy(m, data[1])
            role.snap = {"shot": shot, "side": "gpu", "data": data}
        return res

    def process_gpu(self, frame, t: float, g):
        """``process`` with the picture on the GPU: only the objects' masks, at the track's
        size, go up."""
        i = self.track.frame_at(t)
        if i is None:
            return frame
        labels = self.track.labels(i)
        lk = self._lookup(labels.shape)
        shot = self._shot_of(i)
        tables = None

        def mask(sel):
            nonlocal tables
            if tables is None:
                tables = {"u_xo": g.small(lk["xo"][None, :].astype(np.float32), "cast_xo"),
                          "u_yo": g.small(lk["yo"][None, :].astype(np.float32), "cast_yo")}
            return g.run("cast_near", g.texture(1, "f4"),
                         {"u_sel": g.small(sel.astype(np.uint8) * 255, "cast_sel"), **tables}, fs=_NEAR_FS)

        out = frame
        for role in self.roles:
            ids = self._cast(role.who, i)
            sel = None if ids is not None and not ids else                 (labels == 0) if ids is None else np.isin(labels, ids)
            if sel is not None and not sel[np.ix_(lk["rows"], lk["cols"])].any():
                sel = None                                        # nothing of it lands on the frame
            if role.act == "carry":
                out = self._carry_gpu(g, role, out, None if sel is None else mask(sel), t, shot)
                continue
            if sel is None:
                continue

            def centre():
                s = sel.astype(float)
                n = lk["yn"] @ s @ lk["xn"]
                return float(lk["yn"] @ s @ lk["xs"] / n), float(lk["ys"] @ s @ lk["xn"] / n)

            step = self._step(role, t, centre, shot)
            if step is None:
                continue
            out = self._play_gpu(g, step[0], step[1], out, mask(sel), sel, t)
        return out

    def _carry(self, role: _Role, out: np.ndarray, sel: Optional[np.ndarray], t: float,
               shot: int) -> np.ndarray:
        """Carry: the object brought over the cut laid on the picture, and the object as it
        is now kept for the next cut (``sel`` None: not in this frame)."""
        from core.video.layers import blend_frames
        a = role.carrying(t, shot)
        res = out
        if a > 0.0:
            img, cm = role.carried["data"]
            res = blend_frames(out, self._paste(out, img, cm, IDENTITY, role.shadow), "normal", a)
        if sel is not None:
            m = self._mask(sel)
            if m.any():
                role.snap = {"shot": shot, "side": "cpu", "data": (out.copy(), m)}
        return res

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        i = self.track.frame_at(t)
        if i is None:
            return frame
        labels = self.track.labels(i)
        shot = self._shot_of(i)
        out = frame
        for role in self.roles:
            ids = self._cast(role.who, i)
            sel = None if ids is not None and not ids else \
                (labels == 0) if ids is None else np.isin(labels, ids)
            if role.act == "carry":
                out = self._carry(role, out, sel, t, shot)
            elif sel is not None:            # else nobody plays that role in this shot
                out = self._play(role, out, sel, t, shot)
        return out
