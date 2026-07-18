"""Enhanced NumPy particle system — visually rich audio-reactive renderer.

Compared to the original particles.py this version adds:
  • Particle *trails* (velocity-based streaks)
  • Gaussian bloom / glow pass
  • Kick-triggered expanding shockwave rings
  • Multiple particle populations (core, ambient, sparks)
  • Nebula background that breathes with sub-bass
  • Richer HSV palette cycling driven by centroid + chroma
  • Cinematic vignette
  • Soft tonemapping (Reinhard) instead of hard clamp

Physics API is identical to ParticleSystem so it's a drop-in replacement.
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
# Shockwave ring
# ---------------------------------------------------------------------------

class _Shockwave:
    """An expanding ring triggered by a kick."""
    __slots__ = ("cx", "cy", "radius", "max_radius", "speed", "life", "hue")

    def __init__(self, cx: float, cy: float, intensity: float, hue: float):
        self.cx = cx
        self.cy = cy
        self.radius = 0.0
        self.max_radius = 0.25 + intensity * 0.30
        self.speed = 0.35 + intensity * 0.25
        self.life = 1.0
        self.hue = hue

    def step(self, dt: float) -> bool:
        """Advance ring; return False when dead."""
        self.radius += self.speed * dt
        self.life = max(0.0, 1.0 - self.radius / self.max_radius)
        return self.life > 0.0


# ---------------------------------------------------------------------------
# Enhanced particle system
# ---------------------------------------------------------------------------

class ParticleSystemV2:
    """
    Three particle populations:
      • core   (60%) — attracted to beat phase, bass-driven radial push
      • ambient (30%) — slow drift, flux turbulence, very soft glow
      • sparks  (10%) — short-lived high-velocity, spawned on kicks
    """

    def __init__(self, n: int = 20_000, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.n = n
        self._rng = rng

        # Population splits
        self.n_core    = int(n * 0.60)
        self.n_ambient = int(n * 0.30)
        self.n_sparks  = n - self.n_core - self.n_ambient

        # Core state
        self.pos     = rng.random((n, 2), dtype=np.float32)
        self.vel     = (rng.random((n, 2), dtype=np.float32) - 0.5) * 0.004
        self.life    = rng.random(n, dtype=np.float32)
        self.life_max = rng.random(n, dtype=np.float32) * 3.0 + 1.0

        # Sparks get shorter lifetimes
        spark_start = self.n_core + self.n_ambient
        self.life_max[spark_start:] = rng.random(self.n_sparks, dtype=np.float32) * 0.6 + 0.2

        # Per-particle hue offset
        self.hue_off = rng.random(n, dtype=np.float32)

        # Previous position for trail rendering
        self.prev_pos = self.pos.copy()

        # Shockwave list
        self.shockwaves: List[_Shockwave] = []

        # Pre-compute vignette mask (computed lazily on first render)
        self._vignette: np.ndarray | None = None
        self._vignette_size: tuple = (0, 0)

        # For kick cooldown
        self._kick_cooldown = 0.0

        # Smoothed bass for nebula
        self._smooth_bass = 0.0

    # Population slices
    @property
    def _core_sl(self):   return slice(0, self.n_core)
    @property
    def _amb_sl(self):    return slice(self.n_core, self.n_core + self.n_ambient)
    @property
    def _spark_sl(self):  return slice(self.n_core + self.n_ambient, self.n)

    # ------------------------------------------------------------------
    def step(
        self,
        dt: float,
        bass: float,
        flux: float,
        beat_phase: float,
        kick: float = 0.0,
        solo: float = 0.0,
    ) -> None:
        """Advance physics — same signature as original ParticleSystem."""
        rng = self._rng
        center = np.array([0.5, 0.5], dtype=np.float32)

        # Store previous positions for trails
        np.copyto(self.prev_pos, self.pos)

        diff = self.pos - center
        dist = np.linalg.norm(diff, axis=1, keepdims=True).clip(1e-6)
        radial = diff / dist

        # ---- Core particles -------------------------------------------
        cs = self._core_sl

        # Bass → radial explosion from centre (stronger than original)
        self.vel[cs] += radial[cs] * (bass * 0.022)

        # Rotating attractor follows beat phase
        angle = beat_phase * 2.0 * np.pi
        attr = center + 0.25 * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        to_attr = attr - self.pos[cs]
        self.vel[cs] += to_attr * 0.004

        # Solo → spiral (tangential force around attractor)
        if solo > 0.05:
            tang = np.stack([-to_attr[:, 1], to_attr[:, 0]], axis=1)
            self.vel[cs] += tang * (solo * 0.014)

        # ---- Ambient particles ----------------------------------------
        amb = self._amb_sl

        # Gentle orbital drift
        amb_diff = self.pos[amb] - center
        amb_tang = np.stack([-amb_diff[:, 1], amb_diff[:, 0]], axis=1)
        self.vel[amb] += amb_tang * 0.0008

        # Flux → turbulence (ambient is more sensitive)
        if flux > 0.03:
            noise = rng.random((self.n_ambient, 2), dtype=np.float32) - 0.5
            self.vel[amb] += noise * (flux * 0.012)

        # Weak gravity toward center
        self.vel[amb] += (center - self.pos[amb]) * 0.001

        # ---- Spark particles ------------------------------------------
        sp = self._spark_sl

        # Sparks are mainly kick-driven
        self._kick_cooldown = max(0.0, self._kick_cooldown - dt)

        if kick > 0.3 and self._kick_cooldown <= 0.0:
            self._kick_cooldown = 0.12  # min gap between kicks
            # Burst impulse
            burst = rng.random((self.n_sparks, 2), dtype=np.float32) - 0.5
            self.vel[sp] += burst * (kick * 0.045)
            # Respawn sparks near center with radial velocity
            self.pos[sp] = center + (rng.random((self.n_sparks, 2), dtype=np.float32) - 0.5) * 0.08
            self.life[sp] = 0.0

            # Spawn shockwave
            self.shockwaves.append(_Shockwave(0.5, 0.5, kick, beat_phase * 0.67))

        # Light gravity away from center for sparks (they fly outward)
        self.vel[sp] += radial[sp] * (bass * 0.008)

        # ---- Global forces -------------------------------------------
        # Flux turbulence on all particles (lighter)
        if flux > 0.05:
            noise = rng.random((self.n, 2), dtype=np.float32) - 0.5
            self.vel += noise * (flux * 0.004)

        # Drag (different per population)
        self.vel[cs]  *= 0.935
        self.vel[amb] *= 0.965
        self.vel[sp]  *= 0.90

        # Integrate
        self.pos += self.vel * (dt * 60.0)
        self.pos %= 1.0

        # Age particles
        self.life += dt / self.life_max

        # Respawn dead
        dead = self.life >= 1.0
        n_dead = int(dead.sum())
        if n_dead:
            self.pos[dead]  = rng.random((n_dead, 2), dtype=np.float32)
            self.vel[dead]  = (rng.random((n_dead, 2), dtype=np.float32) - 0.5) * 0.002
            self.life[dead] = 0.0

        # Advance shockwaves
        self.shockwaves = [sw for sw in self.shockwaves if sw.step(dt)]

        # Smooth bass for nebula
        self._smooth_bass += (bass - self._smooth_bass) * 0.08

    # ------------------------------------------------------------------
    def _ensure_vignette(self, W: int, H: int) -> np.ndarray:
        if self._vignette is not None and self._vignette_size == (W, H):
            return self._vignette
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = W / 2.0, H / 2.0
        max_dist = math.sqrt(cx * cx + cy * cy)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_dist
        # smooth vignette: 1 at center, ~0.15 at corners
        self._vignette = np.clip(1.0 - d ** 1.6 * 0.85, 0.15, 1.0)
        self._vignette_size = (W, H)
        return self._vignette

    # ------------------------------------------------------------------
    def render(
        self,
        surface: pygame.Surface,
        centroid: float,
        bar_phase: float,
        bass_bg: float = 0.0,
    ) -> None:
        """Draw particles with trails, bloom, shockwaves, and vignette."""
        W, H = surface.get_size()

        # --- Pixel coordinates (current + previous for trails) ---
        px  = (self.pos[:, 0] * (W - 1)).astype(np.int32)
        py  = (self.pos[:, 1] * (H - 1)).astype(np.int32)
        ppx = (self.prev_pos[:, 0] * (W - 1)).astype(np.int32)
        ppy = (self.prev_pos[:, 1] * (H - 1)).astype(np.int32)

        # --- Brightness: smooth fade-in / fade-out over lifetime ---
        brightness = np.sin(self.life * np.pi).astype(np.float32)

        # Sparks are brighter
        sp = self._spark_sl
        brightness[sp] *= 2.5

        # --- Hue computation ---
        base_hue = centroid * 0.67
        hue = (base_hue + self.hue_off * 0.30 + bar_phase * 0.12) % 1.0

        # Ambient particles shift toward complementary
        amb = self._amb_sl
        hue[amb] = (hue[amb] + 0.45) % 1.0

        # Sparks shift toward warm (orange/yellow)
        hue[sp] = (base_hue + 0.08 + self.hue_off[sp] * 0.06) % 1.0

        # Saturation varies by population
        sat = np.full(self.n, 0.85, dtype=np.float32)
        sat[amb] = 0.5   # ambient is more pastel
        sat[sp]  = 0.95  # sparks are vivid

        r_f, g_f, b_f = _hsv_to_rgb_vec(hue, sat, brightness)

        # === ACCUMULATION BUFFER ===
        buf = np.zeros((W, H, 3), dtype=np.float32)

        # --- Nebula background ---
        self._render_nebula(buf, W, H, bar_phase, centroid)

        # --- Trail layer (previous frame positions, dimmer) ---
        trail_weight = 0.25
        np.add.at(buf[:, :, 0], (ppx, ppy), r_f * trail_weight)
        np.add.at(buf[:, :, 1], (ppx, ppy), g_f * trail_weight)
        np.add.at(buf[:, :, 2], (ppx, ppy), b_f * trail_weight)

        # --- Interpolated trail midpoints for smoother streaks ---
        mpx = ((px + ppx) // 2).clip(0, W - 1)
        mpy = ((py + ppy) // 2).clip(0, H - 1)
        mid_weight = 0.4
        np.add.at(buf[:, :, 0], (mpx, mpy), r_f * mid_weight)
        np.add.at(buf[:, :, 1], (mpx, mpy), g_f * mid_weight)
        np.add.at(buf[:, :, 2], (mpx, mpy), b_f * mid_weight)

        # --- Main particle positions ---
        np.add.at(buf[:, :, 0], (px, py), r_f)
        np.add.at(buf[:, :, 1], (px, py), g_f)
        np.add.at(buf[:, :, 2], (px, py), b_f)

        # --- Shockwave rings ---
        self._render_shockwaves(buf, W, H, base_hue)

        # --- Bloom pass (Gaussian blur on bright areas) ---
        bloom_radius = max(2, int(min(W, H) * 0.012))
        bloom = gaussian_filter(buf, sigma=(bloom_radius, bloom_radius, 0))
        buf += bloom * 0.6

        # --- Vignette ---
        vignette = self._ensure_vignette(W, H)
        # vignette is (H, W) but buf is (W, H, 3)
        buf *= vignette.T[:, :, np.newaxis]

        # --- Tonemap (Reinhard) ---
        buf = buf / (1.0 + buf)
        # Boost contrast slightly
        buf = np.power(buf, 0.9)

        buf_u8 = (np.clip(buf, 0.0, 1.0) * 255).astype(np.uint8)
        pygame.surfarray.blit_array(surface, buf_u8)

    # ------------------------------------------------------------------
    def _render_nebula(
        self,
        buf: np.ndarray,
        W: int, H: int,
        bar_phase: float,
        centroid: float,
    ) -> None:
        """Soft coloured background glow that breathes with bass."""
        intensity = self._smooth_bass * 0.18 + 0.02

        # Two soft radial gradients offset by bar_phase
        cx1 = int(W * (0.5 + 0.2 * math.cos(bar_phase * 2.0 * math.pi)))
        cy1 = int(H * (0.5 + 0.2 * math.sin(bar_phase * 2.0 * math.pi)))
        cx2 = int(W * (0.5 - 0.15 * math.cos(bar_phase * 2.0 * math.pi + 1.0)))
        cy2 = int(H * (0.5 - 0.15 * math.sin(bar_phase * 2.0 * math.pi + 1.0)))

        # Create coordinate grids (transposed for W,H buffer layout)
        yy, xx = np.mgrid[0:H, 0:W]
        xx_t = xx.T.astype(np.float32)
        yy_t = yy.T.astype(np.float32)

        # Gradient 1 — warm hue
        d1 = np.sqrt((xx_t - cx1) ** 2 + (yy_t - cy1) ** 2)
        falloff1 = np.exp(-d1 ** 2 / (2 * (min(W, H) * 0.25) ** 2))
        h1 = (centroid * 0.67 + 0.0) % 1.0
        r1, g1, b1 = _hsv_to_rgb_vec(
            np.full_like(falloff1, h1), 0.6,
            falloff1 * intensity
        )

        # Gradient 2 — complementary hue
        d2 = np.sqrt((xx_t - cx2) ** 2 + (yy_t - cy2) ** 2)
        falloff2 = np.exp(-d2 ** 2 / (2 * (min(W, H) * 0.30) ** 2))
        h2 = (centroid * 0.67 + 0.5) % 1.0
        r2, g2, b2 = _hsv_to_rgb_vec(
            np.full_like(falloff2, h2), 0.5,
            falloff2 * intensity * 0.6
        )

        buf[:, :, 0] += r1 + r2
        buf[:, :, 1] += g1 + g2
        buf[:, :, 2] += b1 + b2

    # ------------------------------------------------------------------
    def _render_shockwaves(
        self,
        buf: np.ndarray,
        W: int, H: int,
        base_hue: float,
    ) -> None:
        """Draw expanding ring shockwaves into the buffer."""
        if not self.shockwaves:
            return

        yy, xx = np.mgrid[0:H, 0:W]
        xx_t = xx.T.astype(np.float32)
        yy_t = yy.T.astype(np.float32)

        for sw in self.shockwaves:
            cx_px = sw.cx * (W - 1)
            cy_px = sw.cy * (H - 1)
            r_px  = sw.radius * max(W, H)

            d = np.sqrt((xx_t - cx_px) ** 2 + (yy_t - cy_px) ** 2)
            ring_width = max(2.0, r_px * 0.08)

            # Smooth ring shape
            ring = np.exp(-((d - r_px) ** 2) / (2 * ring_width ** 2))
            ring *= sw.life ** 2  # fade with lifetime

            hue_val = (base_hue + sw.hue) % 1.0
            r_c, g_c, b_c = _hsv_to_rgb_vec(
                np.full_like(ring, hue_val), 0.7, ring * 0.8
            )

            buf[:, :, 0] += r_c
            buf[:, :, 1] += g_c
            buf[:, :, 2] += b_c
