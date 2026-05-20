"""NumPy-based particle system for audio-reactive offline rendering.

Physics runs entirely on CPU with vectorised NumPy operations.
Rendering uses an additive float32 accumulation buffer written to a
pygame Surface via surfarray — duplicate-index collisions are handled
correctly by np.add.at.
"""
from __future__ import annotations

import numpy as np
import pygame


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
# Particle system
# ---------------------------------------------------------------------------

class ParticleSystem:
    """
    Maintains state for *n* particles: position, velocity, lifetime.

    Positions are normalised 0..1 (wrap at edges).
    Call step() then render() each frame.
    """

    def __init__(self, n: int = 20_000, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.n = n
        self._rng = rng

        # Core state — float32 for memory efficiency
        self.pos     = rng.random((n, 2), dtype=np.float32)
        self.vel     = (rng.random((n, 2), dtype=np.float32) - 0.5) * 0.004
        self.life    = rng.random(n, dtype=np.float32)              # 0..1
        self.life_max = rng.random(n, dtype=np.float32) * 3.0 + 1.0  # 1–4 s

        # Per-particle hue offset so particles have individual colour variation
        self.hue_off = rng.random(n, dtype=np.float32)

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
        """Advance physics by dt seconds given current audio feature values."""
        rng = self._rng
        center = np.array([0.5, 0.5], dtype=np.float32)

        diff = self.pos - center
        dist = np.linalg.norm(diff, axis=1, keepdims=True).clip(1e-6)
        radial = diff / dist

        # Bass → radial explosion from centre
        self.vel += radial * (bass * 0.018)

        # Kick → random burst impulse
        if kick > 0.3:
            burst = rng.random((self.n, 2), dtype=np.float32) - 0.5
            self.vel += burst * (kick * 0.025)

        # Rotating attractor follows beat phase
        angle = beat_phase * 2.0 * np.pi
        attr = center + 0.25 * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        to_attr = attr - self.pos
        self.vel += to_attr * 0.003

        # Solo → spiral (tangential force around attractor)
        if solo > 0.05:
            tang = np.stack([-to_attr[:, 1], to_attr[:, 0]], axis=1)
            self.vel += tang * (solo * 0.010)

        # Flux → turbulence
        if flux > 0.05:
            noise = rng.random((self.n, 2), dtype=np.float32) - 0.5
            self.vel += noise * (flux * 0.006)

        # Drag
        self.vel *= 0.94

        # Integrate (normalised to 60 fps so tuning is fps-independent)
        self.pos += self.vel * (dt * 60.0)
        self.pos %= 1.0  # wrap at edges

        # Age particles
        self.life += dt / self.life_max

        # Respawn dead particles
        dead = self.life >= 1.0
        n_dead = int(dead.sum())
        if n_dead:
            self.pos[dead]  = rng.random((n_dead, 2), dtype=np.float32)
            self.vel[dead]  = (rng.random((n_dead, 2), dtype=np.float32) - 0.5) * 0.002
            self.life[dead] = 0.0

    # ------------------------------------------------------------------
    def render(
        self,
        surface: pygame.Surface,
        centroid: float,
        bar_phase: float,
        bass_bg: float = 0.0,
    ) -> None:
        """Draw particles onto surface using additive accumulation blending."""
        W, H = surface.get_size()

        # --- Pixel coordinates ---
        px = (self.pos[:, 0] * (W - 1)).astype(np.int32)
        py = (self.pos[:, 1] * (H - 1)).astype(np.int32)

        # --- Brightness: smooth fade-in / fade-out over lifetime ---
        brightness = np.sin(self.life * np.pi).astype(np.float32)

        # --- Hue: centroid sets warm/cool base, bar_phase and per-particle
        #     offset add variation so particles don't all match exactly ---
        base_hue = centroid * 0.67          # 0 = red (bass), 0.67 = blue (treble)
        hue = (base_hue + self.hue_off * 0.25 + bar_phase * 0.08) % 1.0

        r_f, g_f, b_f = _hsv_to_rgb_vec(hue, 0.85, brightness)

        # --- Accumulation buffer (additive blending) ---
        # Background colour pulses dark red on bass
        bg_r = bass_bg * 0.16
        bg_g = bass_bg * 0.02

        buf = np.zeros((W, H, 3), dtype=np.float32)
        buf[:, :, 0] = bg_r
        buf[:, :, 1] = bg_g

        np.add.at(buf[:, :, 0], (px, py), r_f)
        np.add.at(buf[:, :, 1], (px, py), g_f)
        np.add.at(buf[:, :, 2], (px, py), b_f)

        # Tonemap: simple clamp (additive can exceed 1.0 where many particles overlap)
        np.clip(buf, 0.0, 1.0, out=buf)
        buf_u8 = (buf * 255).astype(np.uint8)

        pygame.surfarray.blit_array(surface, buf_u8)
