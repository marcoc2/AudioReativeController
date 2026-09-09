"""Layer compositor — stack visual sources with blend modes.

A composition is a bottom-up list of layers. The base layer supplies the
canvas (normally the clip compositor); each further layer renders its own
RGB frame and is blended in with an opacity that may vary per frame
(trigger envelopes now; audio-driven modulation next).

Scene YAML:

    video:
      layers:
        - source: clips            # base — uses the video: section itself
        - source: solid
          color: [255, 255, 255]
          blend: add               # normal | add | screen | multiply
          triggers:
            snare: {notes: [38, 40], envelope: 0.1}   # flash decay (s)
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence
import math
import collections
from scipy.ndimage import affine_transform

import numpy as np

BLENDS = {"normal", "add", "screen", "multiply"}


def draw_line_numpy(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: tuple):
    steps = int(max(abs(x1 - x0), abs(y1 - y0)) * 1.5)
    steps = max(2, steps)
    t_vals = np.linspace(0, 1, steps)
    xs = x0 + t_vals * (x1 - x0)
    ys = y0 + t_vals * (y1 - y0)
    H, W = img.shape[:2]
    for x, y in zip(xs, ys):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            img[iy, ix] = color


def draw_circle_numpy(img: np.ndarray, cx: float, cy: float, r: float, color: tuple):
    H, W = img.shape[:2]
    x_min = max(0, int(math.floor(cx - r)))
    x_max = min(W - 1, int(math.ceil(cx + r)))
    y_min = max(0, int(math.floor(cy - r)))
    y_max = min(H - 1, int(math.ceil(cy + r)))
    
    if x_max < x_min or y_max < y_min:
        return
        
    yy, xx = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
    img[y_min : y_max + 1, x_min : x_max + 1][mask] = color


def scale_rotate_image(img: np.ndarray, scale: float, angle: float) -> np.ndarray:
    H, W, C = img.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    inv_s = 1.0 / max(1e-4, scale)
    M = np.array([
        [inv_s * cos_a, inv_s * sin_a],
        [-inv_s * sin_a, inv_s * cos_a]
    ])
    
    center = np.array([cy, cx])
    offset = center - M.dot(center)
    
    out = np.empty_like(img)
    for c in range(C):
        out[:, :, c] = affine_transform(
            img[:, :, c],
            matrix=M,
            offset=offset,
            order=1,
            mode='constant',
            cval=0
        )
    return out


def hue_shift_channels(img: np.ndarray, shift: float) -> np.ndarray:
    if shift <= 0.0:
        return img
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    
    k = shift % 1.0
    nr = (1.0 - k) * r + k * g
    ng = (1.0 - k) * g + k * b
    nb = (1.0 - k) * b + k * r
    
    out = np.stack([nr, ng, nb], axis=2)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def blend_frames(base: np.ndarray, top: np.ndarray, mode: str, opacity: float) -> np.ndarray:
    """Blend ``top`` over ``base`` (both HxWx3 uint8) with 0..1 opacity."""
    if opacity <= 0.0:
        return base
    a = base.astype(np.float32)
    b = top.astype(np.float32) * float(min(1.0, opacity))
    if mode == "add":
        out = a + b
    elif mode == "screen":
        out = 255.0 - (255.0 - a) * (255.0 - b) / 255.0
    elif mode == "multiply":
        k = float(min(1.0, opacity))
        out = a * ((1.0 - k) + k * top.astype(np.float32) / 255.0)
    else:  # normal
        k = float(min(1.0, opacity))
        out = a * (1.0 - k) + top.astype(np.float32) * k
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


class EnvelopeOpacity:
    """Opacity from trigger hits: 1.0 at each hit, linear decay over ``dur``."""

    def __init__(self, times: Sequence[float], dur: float = 0.1):
        self.times = np.asarray(sorted(times), dtype=float)
        self.dur = max(1e-3, float(dur))

    def __call__(self, t: float) -> float:
        idx = int(np.searchsorted(self.times, t, side="right")) - 1
        if idx < 0:
            return 0.0
        dt = t - float(self.times[idx])
        return max(0.0, 1.0 - dt / self.dur)


class SolidLayer:
    """Constant-color frame; pair with EnvelopeOpacity for drum flashes."""

    def __init__(self, width: int, height: int, color=(255, 255, 255)):
        self._frame = np.empty((height, width, 3), dtype=np.uint8)
        self._frame[:] = np.asarray(color, dtype=np.uint8)

    def frame_at(self, t: float) -> np.ndarray:
        return self._frame


class Compositor:
    """Bottom-up layer stack. First layer is the base canvas."""

    def __init__(self):
        self._layers: List[tuple] = []

    def add(self, source, blend: str = "normal",
            opacity: Optional[Callable[[float], float]] = None) -> None:
        if blend not in BLENDS:
            raise ValueError(f"unknown blend {blend!r}; expected {sorted(BLENDS)}")
        if not self._layers and hasattr(source, "process"):
            raise ValueError("a post-op cannot be the base layer")
        self._layers.append((source, blend, opacity))

    def __len__(self) -> int:
        return len(self._layers)

    def frame_at(self, t: float) -> np.ndarray:
        if not self._layers:
            raise RuntimeError("compositor has no layers")
        src0, _, _ = self._layers[0]
        out = src0.frame_at(t)
        for src, blend, opacity in self._layers[1:]:
            if hasattr(src, "process"):
                # post-op: transforms the composite built so far
                out = src.process(out, t)
                continue
            op = 1.0 if opacity is None else float(opacity(t))
            if op <= 0.0:
                continue
            top = src.frame_at(t)
            if top is not None:
                out = blend_frames(out, top, blend, op)
        return out


def _layer_hits(spec: dict, notes: Sequence, onset_loader=None) -> list:
    """Hit times for a layer trigger: MIDI ``notes`` or audio ``audio``
    (same spec shape as composer triggers)."""
    if "audio" in spec:
        if onset_loader is None:
            from core.video.composer import _default_onset_loader
            onset_loader = _default_onset_loader
        src = onset_loader(spec)
    else:
        pitches = set(spec.get("notes", []))
        src = [n for n in notes if n.pitch in pitches]
    min_vel = int(spec.get("min_velocity", 0))
    return [n.time for n in src if n.velocity >= min_vel]


class CellsLayer:
    """8-bit cells (core.cells) rendered per frame and nearest-upscaled.

    ``features_at(t) -> dict`` supplies audio controls (chroma/flux/...);
    ``mitosis`` trigger spec (notes or audio) splits cells on hits.
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None):
        from core.cells import CellSystem
        self.sys = CellSystem(
            n_base=int(spec.get("n_base", 12)),
            n_max=int(spec.get("n_max", 48)),
            grid=int(spec.get("resolution", 160)),
            seed=spec.get("seed"),
        )
        self.features_at = features_at
        self.fps = fps
        self.W, self.H = width, height
        mit = spec.get("mitosis") or {}
        self.hits = sorted(_layer_hits(mit, notes, onset_loader)) if mit else []
        self._ptr = 0

    def frame_at(self, t: float) -> np.ndarray:
        while self._ptr < len(self.hits) and self.hits[self._ptr] <= t:
            self.sys.mitosis()
            self._ptr += 1
        controls = (self.features_at(t) if self.features_at else None) or {}
        self.sys.step(1.0 / self.fps, controls)
        img = self.sys.render()
        g = img.shape[0]
        iy = np.arange(self.H) * g // self.H
        ix = np.arange(self.W) * g // self.W
        return img[iy][:, ix]


class JuliaLayer:
    """Audio-driven Julia set (core.fractal), nearest-upscaled.

    ``zoom_pulse`` and ``invert`` trigger specs (notes or audio) drive
    envelopes: kick sucks the camera in, snare flips the palette.
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None):
        from core.fractal import JuliaSystem
        self.sys = JuliaSystem(grid=int(spec.get("resolution", 256)),
                               iters=int(spec.get("iters", 48)),
                               aspect=height / width)
        self.features_at = features_at
        self.fps = fps
        self.W, self.H = width, height
        def env(key, default_dur):
            tspec = spec.get(key) or {}
            if not tspec:
                return None
            hits = _layer_hits(tspec, notes, onset_loader)
            return EnvelopeOpacity(hits, tspec.get("envelope", default_dur))
        self._zoom = env("zoom_pulse", 0.25)
        self._invert = env("invert", 0.1)

    def frame_at(self, t: float) -> np.ndarray:
        controls = (self.features_at(t) if self.features_at else None) or {}
        self.sys.step(1.0 / self.fps, controls)
        img = self.sys.render(
            zoom=self._zoom(t) if self._zoom else 0.0,
            invert=self._invert(t) if self._invert else 0.0,
        )
        gh, gw = img.shape[:2]
        iy = np.arange(self.H) * gh // self.H
        ix = np.arange(self.W) * gw // self.W
        return img[iy][:, ix]


class MandelbulbLayer:
    """GPU ray-marched Mandelbulb (core.mandelbulb) at native resolution."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None):
        from core.mandelbulb import MandelbulbSystem
        self.sys = MandelbulbSystem(width, height,
                                    supersample=int(spec.get("supersample", 2)))
        self.features_at = features_at
        self.fps = fps
        tspec = spec.get("zoom_pulse") or {}
        self._zoom = (EnvelopeOpacity(_layer_hits(tspec, notes, onset_loader),
                                      tspec.get("envelope", 0.3))
                      if tspec else None)

    def frame_at(self, t: float) -> np.ndarray:
        controls = (self.features_at(t) if self.features_at else None) or {}
        self.sys.step(1.0 / self.fps, controls)
        return self.sys.render(zoom=self._zoom(t) if self._zoom else 0.0)


class MandelboxLayer:
    """Infinite-zoom KIFS (core.mandelbox); loop period locked to bars."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None, grid=None):
        from core.mandelbox import MandelboxSystem
        self.sys = MandelboxSystem(width, height,
                                   supersample=int(spec.get("supersample", 2)))
        self.sys.hue_spread = float(spec.get("hue_spread", 0.35))
        self.features_at = features_at
        self.fps = fps
        if grid is not None:
            self.loop_s = float(spec.get("loop_bars", 4)) * grid.bar_duration
        else:
            self.loop_s = float(spec.get("loop_seconds", 8.0))
        tspec = spec.get("zoom_pulse") or {}
        self._pulse = (EnvelopeOpacity(_layer_hits(tspec, notes, onset_loader),
                                       tspec.get("envelope", 0.3))
                       if tspec else None)

    def frame_at(self, t: float) -> np.ndarray:
        controls = (self.features_at(t) if self.features_at else None) or {}
        self.sys.step(1.0 / self.fps, controls)
        return self.sys.render(phase=(t / self.loop_s) % 1.0,
                               pulse=self._pulse(t) if self._pulse else 0.0)


class OrbitersLayer:
    """Nested orbiting mandalas (layer source).

    Orbit speed of parents tied to bar_phase + bass, satellites tied to beat_phase + flux.
    Radii are modulated by centroid and flux.
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None, grid=None):
        self.W, self.H = width, height
        self.fps = fps
        self.features_at = features_at
        self.grid = grid
        self.seed = spec.get("seed", 7)
        self.rng = np.random.default_rng(self.seed)
        self.n_parents = int(spec.get("n_parents", 6))
        self.n_satellites = int(spec.get("n_satellites", 4))
        self.parent_radius_spec = spec.get("parent_radius_spec", "centroid")
        self.satellite_radius_spec = spec.get("satellite_radius_spec", "flux")
        self.grid_res = int(spec.get("resolution", 160))
        
        # Base scale values (as fractions of grid size)
        self.parent_r_base = float(spec.get("parent_r_base", 0.22))
        self.sat_r_base = float(spec.get("sat_r_base", 0.08))
        
        # Angle accumulators for audio-reactive speed
        self.p_angle_accum = 0.0
        self.s_angle_accum = 0.0

    def frame_at(self, t: float) -> np.ndarray:
        g = self.grid_res
        img = np.zeros((g, g, 3), dtype=np.uint8)
        controls = (self.features_at(t) if self.features_at else None) or {}
        
        # Feature inputs
        p_mod = float(controls.get(self.parent_radius_spec, 0.5) or 0.5)
        s_mod = float(controls.get(self.satellite_radius_spec, 0.5) or 0.5)
        
        # Dynamic rotation speed based on bass & flux
        dt = 1.0 / self.fps
        bass = float(controls.get("bass", controls.get("bass_energy", 0.5)) or 0.5)
        flux = float(controls.get("flux", 0.5) or 0.5)
        
        self.p_angle_accum += dt * (1.0 + 8.0 * bass)
        self.s_angle_accum += dt * (2.0 + 12.0 * flux)
        
        # Chroma feature for colors (fallback to all 0.5 if not present)
        chroma = controls.get("chroma", [0.5] * 12)
        if len(chroma) < 12:
            chroma = [0.5] * 12
            
        cx_c, cy_c = g / 2.0, g / 2.0
        
        from core.cells import PALETTE
        
        # Draw connections and nodes
        for i in range(self.n_parents):
            # Map parent to a chroma hue
            p_bin = (i * 12 // self.n_parents) % 12
            p_energy = float(chroma[p_bin])
            
            # Parent color based on chroma
            col_float = PALETTE[p_bin] * (0.35 + 0.65 * p_energy)
            color = tuple(int(c) for c in col_float)
            
            # Parent angle (accumulated + offset) and radius (scaled up modulation)
            p_angle = self.p_angle_accum + (2.0 * math.pi * i / self.n_parents)
            p_r = g * (self.parent_r_base + 0.35 * p_mod)
            
            px = cx_c + p_r * math.cos(p_angle)
            py = cy_c + p_r * math.sin(p_angle)
            
            # Draw line from center to parent (dimmed color)
            draw_line_numpy(img, cx_c, cy_c, px, py, tuple(int(c * 0.25) for c in color))
            
            # Draw satellites
            for j in range(self.n_satellites):
                s_angle = -self.s_angle_accum + (2.0 * math.pi * j / self.n_satellites)
                s_r = g * (self.sat_r_base + 0.15 * s_mod)
                
                sx = px + s_r * math.cos(s_angle)
                sy = py + s_r * math.sin(s_angle)
                
                # Draw line from parent to satellite
                draw_line_numpy(img, px, py, sx, sy, tuple(int(c * 0.45) for c in color))
                
                # Draw satellite circle
                draw_circle_numpy(img, sx, sy, g * 0.02, color)
                
            # Draw parent circle
            draw_circle_numpy(img, px, py, g * 0.035, tuple(int(c * 0.8) for c in color))
            
        # Draw central hub circle
        draw_circle_numpy(img, cx_c, cy_c, g * 0.05, (255, 255, 255))
        
        # Nearest-neighbor upscale to H x W
        iy = np.arange(self.H) * g // self.H
        ix = np.arange(self.W) * g // self.W
        return img[iy][:, ix]


class ParticlesLayer:
    """Unifies ParticleSystem and emits particles from a central rotating mutant polygon.

    Emissions are triggered on MIDI or audio onset triggers. Physics forces (bass, flux)
    continuously deform the simulation.
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None):
        self.W, self.H = width, height
        self.fps = fps
        self.features_at = features_at
        self.seed = spec.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)
        
        self.grid_res = int(spec.get("resolution", 200))
        self.n_particles = int(spec.get("n_particles", 10000))
        
        # Core particle arrays
        self.pos = self.rng.random((self.n_particles, 2), dtype=np.float32)
        self.vel = (self.rng.random((self.n_particles, 2), dtype=np.float32) - 0.5) * 0.02
        self.life = self.rng.random(self.n_particles, dtype=np.float32)
        self.life_max = self.rng.random(self.n_particles, dtype=np.float32) * 1.5 + 0.5
        self.hue_off = self.rng.random(self.n_particles, dtype=np.float32)
        
        # Central emitter polygon
        self.n_vertices = int(spec.get("n_vertices", 5))
        self.poly_r = float(spec.get("emitter_radius", 0.2)) # fraction of grid
        self.rot_speed = float(spec.get("rotation_speed", 1.5)) # rad/s
        
        # Trigger config
        tspec = spec.get("triggers", {}).get("burst") or spec.get("mitosis") or {}
        self.hits = sorted(_layer_hits(tspec, notes, onset_loader)) if tspec else []
        self.ptr_trigger = 0
        self.burst_qty = int(tspec.get("quantity", 250))
        
        # Current index to cycle through for respawning particles
        self.respawn_idx = 0

    def frame_at(self, t: float) -> np.ndarray:
        g = self.grid_res
        dt = 1.0 / self.fps
        
        # Process trigger hits
        bursts_to_trigger = 0
        while self.ptr_trigger < len(self.hits) and self.hits[self.ptr_trigger] <= t:
            bursts_to_trigger += 1
            self.ptr_trigger += 1
            
        controls = (self.features_at(t) if self.features_at else None) or {}
        
        # Rotate the emitter polygon
        theta = t * self.rot_speed
        
        # Calculate polygon vertices (normalized 0..1 coordinates)
        vertices = []
        for i in range(self.n_vertices):
            angle = theta + 2.0 * math.pi * i / self.n_vertices
            vx = 0.5 + self.poly_r * math.cos(angle)
            vy = 0.5 + self.poly_r * math.sin(angle)
            vertices.append((vx, vy, angle))
            
        # If triggers occurred, respawn particles at vertices and shoot them out
        if bursts_to_trigger > 0 and self.n_vertices > 0:
            qty_per_vertex = max(1, self.burst_qty // self.n_vertices)
            for vx, vy, angle in vertices:
                for _ in range(qty_per_vertex):
                    idx = self.respawn_idx
                    self.pos[idx] = [vx, vy]
                    # Velocity pointing outwards + random spray
                    spray = self.rng.normal(0, 0.05, 2)
                    vel_dir = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
                    self.vel[idx] = vel_dir * self.rng.uniform(0.12, 0.28) + spray
                    self.life[idx] = 0.0
                    self.life_max[idx] = self.rng.uniform(0.8, 2.0)
                    self.respawn_idx = (self.respawn_idx + 1) % self.n_particles
                    
        # Update physics
        center = np.array([0.5, 0.5], dtype=np.float32)
        diff = self.pos - center
        dist = np.linalg.norm(diff, axis=1, keepdims=True).clip(1e-6)
        radial = diff / dist
        
        # Audio-reactive continuous forces
        bass = float(controls.get("bass", controls.get("bass_energy", 0.0)) or 0.0)
        flux = float(controls.get("flux", 0.0) or 0.0)
        
        # Bass -> radial explosion from center
        self.vel += radial * (bass * 0.02)
        
        # Flux -> turbulence (noise)
        if flux > 0.05:
            noise = self.rng.random((self.n_particles, 2), dtype=np.float32) - 0.5
            self.vel += noise * (flux * 0.008)
            
        # Attract to center attractor
        to_center = center - self.pos
        self.vel += to_center * 0.02 * dt
        
        # Drag
        self.vel *= 0.90
        
        # Move
        self.pos += self.vel * (dt * 60.0)
        self.pos %= 1.0  # wrap at edges
        
        # Age
        self.life += dt / self.life_max
        
        # Respawn naturally dead particles randomly
        dead = self.life >= 1.0
        n_dead = int(dead.sum())
        if n_dead > 0:
            self.pos[dead] = self.rng.random((n_dead, 2), dtype=np.float32)
            self.vel[dead] = (self.rng.random((n_dead, 2), dtype=np.float32) - 0.5) * 0.01
            self.life[dead] = 0.0
            
        # Draw particles additive buffer
        px = (self.pos[:, 0] * (g - 1)).astype(np.int32)
        py = (self.pos[:, 1] * (g - 1)).astype(np.int32)
        brightness = np.sin(self.life * np.pi).astype(np.float32)
        
        centroid = float(controls.get("centroid", 0.5) or 0.5)
        bar_phase = float(t / 4.0) % 1.0
        
        base_hue = centroid * 0.67
        hue = (base_hue + self.hue_off * 0.25 + bar_phase * 0.08) % 1.0
        
        from core.particles import _hsv_to_rgb_vec
        r_f, g_f, b_f = _hsv_to_rgb_vec(hue, 0.85, brightness)
        
        buf = np.zeros((g, g, 3), dtype=np.float32)
        np.add.at(buf[:, :, 0], (py, px), r_f)
        np.add.at(buf[:, :, 1], (py, px), g_f)
        np.add.at(buf[:, :, 2], (py, px), b_f)
        
        np.clip(buf, 0.0, 1.0, out=buf)
        buf_u8 = (buf * 255).astype(np.uint8)
        
        # Draw the central polygon outline on the buffer
        poly_pts = [(int(vx * g), int(vy * g)) for vx, vy, _ in vertices]
        for i in range(len(poly_pts)):
            x0, y0 = poly_pts[i]
            x1, y1 = poly_pts[(i + 1) % len(poly_pts)]
            draw_line_numpy(buf_u8, x0, y0, x1, y1, (100, 200, 255))
            
        # Nearest-neighbor upscale to output resolution
        iy = np.arange(self.H) * g // self.H
        ix = np.arange(self.W) * g // self.W
        return buf_u8[iy][:, ix]


class FeedbackLayer:
    """Post-op: accumulates feedback by blending the previous frame,

    slightly scaled and rotated, under the current frame.
    """

    def __init__(self, spec: dict, notes: Sequence, fps: int,
                 features_at=None, onset_loader=None):
        self.fps = fps
        self.features_at = features_at
        self.decay = float(spec.get("decay", 0.85))
        self.max_scale = float(spec.get("max_scale", 1.05))
        self.base_scale = float(spec.get("base_scale", 0.96))
        self.max_rotate = float(spec.get("max_rotate", 0.1)) # rad
        
        # Trigger for zoom (e.g. kick)
        zspec = spec.get("zoom_pulse")
        self._zoom_env = (EnvelopeOpacity(_layer_hits(zspec, notes, onset_loader),
                                          zspec.get("envelope", 0.25))
                          if zspec else None)
                          
        # Smooth state for rotation (centroid lowpass)
        self.rot_smooth = 0.0
        self._prev = None

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        if self.decay <= 0.0:
            return frame
            
        if self._prev is None or self._prev.shape != frame.shape:
            self._prev = frame.copy()
            return frame
            
        # Compute scale
        if self._zoom_env is not None:
            env_val = self._zoom_env(t)
            scale = self.base_scale + (self.max_scale - self.base_scale) * env_val
        else:
            scale = self.base_scale
            
        # Compute rotation (angle)
        dt = 1.0 / self.fps
        x = 0.5
        if self.features_at is not None:
            feats = self.features_at(t) or {}
            x = float(feats.get("centroid", 0.5) or 0.5)
            
        tau = 0.15
        self.rot_smooth += (x - self.rot_smooth) * (1.0 - math.exp(-dt / tau))
        angle = self.max_rotate * (self.rot_smooth - 0.5)
        
        # Scale and rotate self._prev
        warped = scale_rotate_image(self._prev, scale, angle)
        
        # Blend: current frame composed over warped history using screen mode
        out = blend_frames(frame, warped, "screen", self.decay)
        
        # Save output for next iteration
        self._prev = out.copy()
        return out


class EchoesLayer:
    """Post-op: dynamically blends previous frames to create temporal echoes

    with Doppler scaling and hue shifting.
    """

    def __init__(self, spec: dict, notes: Sequence, fps: int,
                 features_at=None, onset_loader=None):
        self.fps = fps
        self.features_at = features_at
        self.depth = max(0, min(16, int(spec.get("depth", 5)))) # cap at 16 frames
        self.opacity_decay = float(spec.get("opacity_decay", 0.75))
        self.scale_decay = float(spec.get("scale_decay", 0.95))
        self.hue_shift = float(spec.get("hue_shift", 0.08))
        self.history = collections.deque(maxlen=self.depth)

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        if self.depth <= 0 or self.opacity_decay <= 0.0:
            return frame
            
        # Add copy of current frame to history
        self.history.append(frame.copy())
        
        if len(self.history) < 2:
            return frame
            
        out = frame.copy()
        n = len(self.history)
        
        # Get current flux to modulate hue shift
        flux = 0.5
        if self.features_at is not None:
            flux = float((self.features_at(t) or {}).get("flux", 0.5) or 0.5)
            
        # Draw history from oldest to newest (index 0 to n-2)
        for i in range(n - 1):
            past_img = self.history[i]
            age = n - 1 - i
            
            scale = self.scale_decay ** age
            opacity = self.opacity_decay ** age
            
            # Scale past frame
            scaled = scale_rotate_image(past_img, scale, angle=0.0)
            
            # Apply hue shift
            shift_val = self.hue_shift * age * (0.3 + 0.7 * flux)
            shifted = hue_shift_channels(scaled, shift_val)
            
            # Additive/screen blend over current composite
            out = blend_frames(out, shifted, "screen", opacity)
            
        return out


class RgbSplit:
    """Post-op reference implementation: chromatic aberration on impact.

    Shifts the R and B channels apart by up to ``amount`` pixels, driven
    by a trigger envelope (e.g. snare hits) — the classic "camera hit"
    glitch that resolves with the drum's decay. Stateless; identity when
    the envelope is at zero, so scenes without hits are untouched.
    """

    def __init__(self, spec: dict, notes: Sequence, fps: int,
                 features_at=None, onset_loader=None):
        tspec = spec.get("trigger") or {}
        self._env = (EnvelopeOpacity(_layer_hits(tspec, notes, onset_loader),
                                     tspec.get("envelope", 0.12))
                     if tspec else None)
        self.amount = float(spec.get("amount", 8.0))
        self.features_at = features_at

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        if self._env is not None:
            e = self._env(t)
        elif self.features_at is not None:
            e = float((self.features_at(t) or {}).get("flux", 0.0) or 0.0)
        else:
            e = 0.0
        px = int(round(self.amount * e))
        if px <= 0:
            return frame
        out = frame.copy()
        out[:, :, 0] = np.roll(frame[:, :, 0], px, axis=1)
        out[:, :, 2] = np.roll(frame[:, :, 2], -px, axis=1)
        return out


class CubesLayer:
    """Stage/scenography layer: reactive clips painted on a lattice of cubes
    that a camera flies through, over an otherwise empty solid-color space.

    This is a *stage*, not a generated world: it takes clip footage — the
    content — and shows it on the faces of cubes floating in space. It owns
    its own clip playback (``ClipLibrary`` + ``ClipComposer``), so it is
    self-contained and never touches the global clip base.

    Scene YAML (``source: cubes``)::

        - source: cubes
          clips_dir: C:/path/to/clips     # own clip pool (required)
          bg_color: [0, 0, 0]             # empty-space color
          n_cubes: 40                     # cubes in the periodic corridor
          faces_with_clips: 3             # 0..6 faces showing the clip screen
          face_resolution: 256            # per-face texture size (RAM/CPU)
          crop_faces: true                # center-crop 16:9 clips to fill (no bars)
          one_clip_per_face: false        # false: same screen on all faces;
                                          # true: an independent clip per face-index
          loop_bars: 2                    # camera advances 1 corridor / N bars
          zoom_pulse: {notes: [36], envelope: 0.25}   # kick = forward lurch
          clip_reactivity:                # a ClipComposer config (the "screen")
            clip_per_bar: true
            clip_order: shuffle           # per-face variety comes from seeded streams
            triggers:
              kick: {notes: [36], actions: [reverse]}

    ``one_clip_per_face`` toggles the depth of the recursion: false = a single
    internal ``ClipComposer`` shown on every clip face (Stage 0); true = one
    ``ClipComposer`` per face-index (0..``faces_with_clips``-1), each an
    independent reactive stream (Stage 1). All composers share one decode
    library, so per-face variety costs O(faces) RAM, not O(cubes). Distinct
    faces come from per-face seeds; use ``clip_order: shuffle`` or ``random``
    (``sequential`` keeps faces in lock-step by design).
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, grid=None, features_at=None, onset_loader=None):
        from core.video.clip_library import ClipLibrary
        from core.video.composer import ClipComposer
        self.W, self.H = width, height
        self.fps = fps
        clips_dir = spec.get("clips_dir")
        if not clips_dir:
            raise ValueError("cubes layer requires 'clips_dir'")
        face_res = int(spec.get("face_resolution", 256))

        # one reactive screen shared by all faces (Stage 0), or one per
        # face-index when one_clip_per_face is set (Stage 1) — bounded to the
        # 0..6 face count, never to the cube count.
        k = max(0, min(6, int(spec.get("faces_with_clips", 3))))
        n_screens = k if (spec.get("one_clip_per_face") and k > 0) else 1

        # a single shared library (a decode cache): size it to hold every
        # screen's current clip at once, so per-face variety costs O(faces),
        # not O(cubes), in RAM.
        cache = max(int(spec.get("cache_size", 4)), n_screens + 2)
        # cover = center-crop 16:9 footage to fill the square faces (no bars)
        fit = "cover" if spec.get("crop_faces") else "contain"
        self.lib = ClipLibrary(clips_dir, face_res, face_res, fps,
                               cache_size=cache, fit=fit)

        base_react = dict(spec.get("clip_reactivity") or {})
        base_seed = spec.get("seed")
        self.composers = []
        for i in range(n_screens):
            react = dict(base_react)
            # distinct stream per face (shuffle/random spread apart by seed);
            # keep Stage 0's single-composer seed exactly as before.
            react["seed"] = base_react.get("seed", base_seed) if n_screens == 1 else (
                None if base_seed is None else int(base_seed) + i * 9973)
            comp = ClipComposer(self.lib, grid, notes, react, onset_loader=onset_loader)
            if n_screens > 1 and len(self.lib) > 1:
                comp._first_selection = False           # don't force clip 0 on bar 0
                comp.transport.set_clip(i % len(self.lib))   # distinct start per face
            self.composers.append(comp)

        self.bg = spec.get("bg_color", [0, 0, 0])
        if grid is not None:
            self.loop_s = float(spec.get("loop_bars", 2)) * grid.bar_duration
        else:
            self.loop_s = float(spec.get("loop_seconds", 8.0))
        zspec = spec.get("zoom_pulse") or {}
        self._pulse = (EnvelopeOpacity(_layer_hits(zspec, notes, onset_loader),
                                       zspec.get("envelope", 0.25))
                       if zspec else None)
        self._pulse_gain = float(spec.get("pulse_gain", 0.12))
        from core.cubes import CubeField
        self.field = CubeField(width, height, spec, seed=base_seed, n_screens=n_screens)
        self._seeked = False

    def frame_at(self, t: float) -> np.ndarray:
        if not self._seeked:
            for comp in self.composers:
                comp.seek(t)          # don't dump pre-start events on frame 0
            self._seeked = True
        screens = [comp.frame_at(t) for comp in self.composers]   # one per face-index
        phase = (t / self.loop_s) % 1.0
        if self._pulse is not None:                     # kick = transient lurch
            phase = (phase + self._pulse_gain * self._pulse(t)) % 1.0
        return self.field.render(phase, screens, self.bg)


def build_compositor(base, video_cfg: dict, notes: Sequence,
                     width: int, height: int, onset_loader=None,
                     fps: int = 24, features_at=None, grid=None) -> "Compositor":
    """Compose ``base`` (ClipComposer) with the scene's extra layers.

    Layers with ``source: clips`` map to the base; unknown sources raise.
    Without a ``layers:`` section, the result is just the base (legacy).
    Layer triggers accept MIDI ``notes`` or ``audio`` onset sources.
    """
    comp = Compositor()
    # no layers: section -> legacy single clips base; a layers: list is
    # used as-is (first layer = canvas), so pure-generative scenes work
    layers_cfg = video_cfg.get("layers") or [{"source": "clips"}]
    for spec in layers_cfg:
        src_name = spec.get("source", "clips")
        blend = spec.get("blend", "normal")
        static_op = spec.get("opacity")
        op_fn = (lambda t, v=float(static_op): v) if isinstance(static_op, (int, float)) else None
        if src_name == "clips":
            comp.add(base, blend if len(comp) else "normal", op_fn)
            continue
        if src_name == "solid":
            src = SolidLayer(width, height, spec.get("color", [255, 255, 255]))
            opacity = op_fn
            trig = spec.get("triggers") or {}
            for name, tspec in trig.items():
                hits = _layer_hits(tspec, notes, onset_loader)
                opacity = EnvelopeOpacity(hits, tspec.get("envelope", 0.1))
            comp.add(src, blend, opacity)
            continue
        if src_name == "cells":
            comp.add(CellsLayer(spec, notes, width, height, fps,
                                features_at=features_at,
                                onset_loader=onset_loader),
                     blend, op_fn)
            continue
        if src_name == "orbiters":
            comp.add(OrbitersLayer(spec, notes, width, height, fps,
                                   features_at=features_at,
                                   onset_loader=onset_loader, grid=grid),
                     blend, op_fn)
            continue
        if src_name == "particles":
            comp.add(ParticlesLayer(spec, notes, width, height, fps,
                                    features_at=features_at,
                                    onset_loader=onset_loader),
                     blend, op_fn)
            continue
        if src_name == "feedback":
            comp.add(FeedbackLayer(spec, notes, fps,
                                   features_at=features_at,
                                   onset_loader=onset_loader),
                     "normal", None)
            continue
        if src_name == "echoes":
            comp.add(EchoesLayer(spec, notes, fps,
                                 features_at=features_at,
                                 onset_loader=onset_loader),
                     "normal", None)
            continue
        if src_name == "rgb_split":
            comp.add(RgbSplit(spec, notes, fps, features_at=features_at,
                              onset_loader=onset_loader), "normal", None)
            continue
        if src_name == "mandelbox":
            comp.add(MandelboxLayer(spec, notes, width, height, fps,
                                    features_at=features_at,
                                    onset_loader=onset_loader, grid=grid),
                     blend, op_fn)
            continue
        if src_name == "mandelbulb":
            comp.add(MandelbulbLayer(spec, notes, width, height, fps,
                                     features_at=features_at,
                                     onset_loader=onset_loader),
                     blend, op_fn)
            continue
        if src_name == "julia":
            comp.add(JuliaLayer(spec, notes, width, height, fps,
                                features_at=features_at,
                                onset_loader=onset_loader),
                     blend, op_fn)
            continue
        if src_name == "cubes":
            comp.add(CubesLayer(spec, notes, width, height, fps,
                                grid=grid, features_at=features_at,
                                onset_loader=onset_loader),
                     blend if len(comp) else "normal", op_fn)
            continue
        raise ValueError(f"unknown layer source {src_name!r}")
    return comp
