"""Quantum Kaleidoscope particle system — ultimate visual fidelity.

Features:
  • Flow field physics (curl-noise simulation) for organic swarming
  • Kaleidoscopic 3-fold radial symmetry (multiplies visual density)
  • Dynamic parametric audio rings (oscilloscopes) that deform with audio
  • Chromatic aberration on particle accumulation
  • Anamorphic bloom (horizontal lens flares) on kicks
  • Cinematic vignette & HDR Reinhard tonemapping
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pygame
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _hsv_to_rgb_vec(
    h: np.ndarray,
    s: float | np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised HSV → RGB.  All inputs in 0..1, outputs in 0..1."""
    h6 = h * 6.0
    i  = np.floor(h6).astype(np.int32) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1.0 - s)
    q  = v * (1.0 - s * f)
    t_ = v * (1.0 - s * (1.0 - f))
    r  = np.choose(i, [v,  q,  p,  p,  t_, v ])
    g  = np.choose(i, [t_, v,  v,  q,  p,  p ])
    b  = np.choose(i, [p,  p,  t_, v,  v,  q ])
    return r, g, b


# ---------------------------------------------------------------------------
# Quantum Kaleidoscope System
# ---------------------------------------------------------------------------

class ParticleSystemV3:
    """
    V3 introduces geometric symmetry, flow fields, and parametric rendering.
    """

    def __init__(self, n: int = 25_000, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.n = n
        self._rng = rng

        # Physics state
        self.pos = rng.random((n, 2), dtype=np.float32)
        self.vel = (rng.random((n, 2), dtype=np.float32) - 0.5) * 0.002
        self.life = rng.random(n, dtype=np.float32)
        self.life_max = rng.random(n, dtype=np.float32) * 2.0 + 1.0
        self.hue_off = rng.random(n, dtype=np.float32)
        
        self.prev_pos = self.pos.copy()
        
        # Parametric state smoothers
        self._time = 0.0
        self._smooth_bass = 0.0
        self._smooth_kick = 0.0
        self._smooth_snare = 0.0
        self._smooth_flux = 0.0
        
        # Vignette cache
        self._vignette = None
        self._vignette_size = (0, 0)

    # ------------------------------------------------------------------
    def step(
        self,
        dt: float,
        bass: float,
        flux: float,
        beat_phase: float,
        kick: float = 0.0,
        snare: float = 0.0,
        solo: float = 0.0,
    ) -> None:
        """Advance physics using flow fields and orbital mechanics."""
        self._time += dt
        
        # Smooth audio features for parametric animations
        self._smooth_bass  += (bass - self._smooth_bass) * 0.15
        self._smooth_kick  += (kick - self._smooth_kick) * 0.25
        self._smooth_snare += (snare - self._smooth_snare) * 0.25
        self._smooth_flux  += (flux - self._smooth_flux) * 0.10

        np.copyto(self.prev_pos, self.pos)
        
        center = np.array([0.5, 0.5], dtype=np.float32)
        diff = self.pos - center
        dist = np.linalg.norm(diff, axis=1, keepdims=True).clip(1e-6)
        radial = diff / dist

        # 1. Flow Field (Curl noise simulation)
        fx = self.pos[:, 0] * 12.0
        fy = self.pos[:, 1] * 12.0
        phase = self._time * 1.5
        
        # Cross-sine waves create swirling vortices
        vx =  np.sin(fy + phase) * np.cos(fx * 0.6)
        vy = -np.sin(fx + phase) * np.cos(fy * 0.6)
        
        flow = np.stack([vx, vy], axis=1)
        # Flux intensifies the turbulence
        self.vel += flow * (0.0008 + flux * 0.01)

        # 2. Central Attractor / Black Hole
        # Normally pulls particles in, but kicks blast them outward
        pull = 0.001 - kick * 0.015
        self.vel -= radial * pull
        
        # 3. Tangential Orbit
        # Particles spiral around the center
        tang = np.stack([-diff[:, 1], diff[:, 0]], axis=1)
        vortex_speed = 0.0015
        self.vel += tang * vortex_speed

        # Integration & Drag
        self.vel *= 0.94
        self.pos += self.vel * (dt * 60.0)
        self.pos %= 1.0

        # Aging & Respawn
        self.life += dt / self.life_max
        dead = self.life >= 1.0
        n_dead = int(dead.sum())
        
        if n_dead:
            # Respawn in a dense ring around center for better symmetry
            angle = self._rng.random(n_dead, dtype=np.float32) * 2 * np.pi
            r = 0.1 + self._rng.random(n_dead, dtype=np.float32) * 0.3
            self.pos[dead, 0] = 0.5 + np.cos(angle) * r
            self.pos[dead, 1] = 0.5 + np.sin(angle) * r
            self.vel[dead] = 0.0
            self.life[dead] = 0.0

    # ------------------------------------------------------------------
    def _get_rotated(self, pos: np.ndarray, angle: float) -> np.ndarray:
        """Rotates a batch of positions around the center (0.5, 0.5)."""
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (pos - 0.5) @ R + 0.5

    def _ensure_vignette(self, W: int, H: int) -> np.ndarray:
        if self._vignette is not None and self._vignette_size == (W, H):
            return self._vignette
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = W / 2.0, H / 2.0
        max_dist = math.sqrt(cx * cx + cy * cy)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_dist
        # Sharper vignette for V3
        self._vignette = np.clip(1.0 - d ** 1.8, 0.05, 1.0)
        self._vignette_size = (W, H)
        return self._vignette

    # ------------------------------------------------------------------
    def render(
        self,
        surface: pygame.Surface,
        centroid: float,
        bar_phase: float,
        bass_bg: float = 0.0,
        solo: float = 0.0,
    ) -> None:
        """Draw particles with symmetry, chromatic aberration, and parametric rings."""
        W, H = surface.get_size()
        buf = np.zeros((W, H, 3), dtype=np.float32)

        # Base properties
        brightness = np.sin(self.life * np.pi).astype(np.float32) * 1.5
        base_hue = centroid * 0.8
        
        # Psychedelic LUT (Purple, Blue, Green)
        # Map each particle to one of the 3 colors
        color_choice = (self.hue_off * 3).astype(np.int32) % 3
        lut_hues = np.array([0.80, 0.60, 0.35], dtype=np.float32)
        hue = lut_hues[color_choice]
        
        # Swirl the colors slightly over time when solo is active
        hue = (hue + self._time * 0.1 * solo) % 1.0
        
        # Saturation controlled by solo (0.0 = Grayscale/Monochromatic, 1.0 = Psychedelic)
        saturation = np.clip(solo * 2.0, 0.0, 1.0)
        
        r_f, g_f, b_f = _hsv_to_rgb_vec(hue, saturation, brightness)

        # Chromatic Aberration offset based on kick
        aberration = int(self._smooth_kick * min(W, H) * 0.04)

        # Kaleidoscope Symmetry
        mirrors = 3
        global_rot = self._time * 0.2
        
        for m in range(mirrors):
            angle = m * (2 * np.pi / mirrors) + global_rot
            
            # Rotate positions and wrap to create an infinite repeating pattern
            r_pos = self._get_rotated(self.pos, angle) % 1.0
            r_prev = self._get_rotated(self.prev_pos, angle) % 1.0
            
            # Chromatic direction (radial from center)
            dir_x = r_pos[:, 0] - 0.5
            dir_x_sign = np.sign(dir_x).astype(np.int32)
            
            px = (r_pos[:, 0] * (W - 1)).astype(np.int32)
            py = (r_pos[:, 1] * (H - 1)).astype(np.int32)
            ppx = (r_prev[:, 0] * (W - 1)).astype(np.int32)
            ppy = (r_prev[:, 1] * (H - 1)).astype(np.int32)
            mpx = ((px + ppx) // 2).clip(0, W - 1)
            mpy = ((py + ppy) // 2).clip(0, H - 1)
            
            # Apply chromatic aberration to X coordinate
            px_R = (px + aberration * dir_x_sign).clip(0, W - 1)
            px_B = (px - aberration * dir_x_sign).clip(0, W - 1)
            
            mpx_R = (mpx + aberration * dir_x_sign).clip(0, W - 1)
            mpx_B = (mpx - aberration * dir_x_sign).clip(0, W - 1)

            # Accumulate Midpoints (trails)
            np.add.at(buf[:, :, 0], (mpx_R, mpy), r_f * 0.4)
            np.add.at(buf[:, :, 1], (mpx, mpy),   g_f * 0.4)
            np.add.at(buf[:, :, 2], (mpx_B, mpy), b_f * 0.4)
            
            # Accumulate Current
            np.add.at(buf[:, :, 0], (px_R, py), r_f)
            np.add.at(buf[:, :, 1], (px, py),   g_f)
            np.add.at(buf[:, :, 2], (px_B, py), b_f)

        # Render Parametric Audio Rings
        self._render_rings(buf, W, H, base_hue)

        # Anamorphic Bloom (Horizontal streak flares on kick)
        if self._smooth_kick > 0.05:
            bloom_x = max(2, int(min(W, H) * 0.08 * self._smooth_kick))
            bloom_y = 1
            bloom = gaussian_filter(buf, sigma=(bloom_x, bloom_y, 0))
            buf += bloom * 0.9
            
        # Global soft bloom to blend the geometry
        bloom_soft = gaussian_filter(buf, sigma=(3, 3, 0))
        buf += bloom_soft * 0.5

        # Vignette
        vignette = self._ensure_vignette(W, H)
        buf *= vignette.T[:, :, np.newaxis]
        
        # Tone mapping (Reinhard + Gamma)
        buf = buf / (1.0 + buf)
        buf = np.power(buf, 0.85)

        buf_u8 = (np.clip(buf, 0.0, 1.0) * 255).astype(np.uint8)
        pygame.surfarray.blit_array(surface, buf_u8)

    # ------------------------------------------------------------------
    def _render_rings(self, buf: np.ndarray, W: int, H: int, base_hue: float):
        """Draws waveform rings that deform based on audio features."""
        ring_surf = pygame.Surface((W, H)) # Black by default
        cx, cy = W // 2, H // 2
        
        # 3 Rings: Inner (Snare), Mid (Flux), Outer (Kick)
        configs = [
            (0.12, self._smooth_snare * 0.4, 3, (base_hue + 0.0) % 1.0),
            (0.22, self._smooth_flux  * 0.4, 5, (base_hue + 0.33) % 1.0),
            (0.32, self._smooth_kick  * 0.4, 7, (base_hue + 0.66) % 1.0),
        ]
        
        max_r = min(W, H)
        for base_r, amp, waves, hue in configs:
            pts = []
            steps = 120
            for i in range(steps):
                theta = i * 2 * math.pi / steps
                # Deform radius with audio amplitude and time
                r_mod = math.sin(theta * waves - self._time * 5.0) * amp
                r = (base_r + r_mod) * max_r
                x = cx + r * math.cos(theta + self._time * 0.5)
                y = cy + r * math.sin(theta + self._time * 0.5)
                pts.append((x, y))
            
            # Convert HSV to RGB for the Pygame draw call
            h_arr = np.array([hue], dtype=np.float32)
            r_c, g_c, b_c = _hsv_to_rgb_vec(h_arr, 0.8, 1.0)
            color = (int(r_c[0]*255), int(g_c[0]*255), int(b_c[0]*255))
            
            pygame.draw.polygon(ring_surf, color, pts, width=3)

        # Extract ring pixels and add to main float buffer
        ring_arr = pygame.surfarray.pixels3d(ring_surf).astype(np.float32) / 255.0
        buf += ring_arr * 2.5 # Boost intensity so they glow via bloom
