"""
ARC Debug Dashboard v2 — 4K feature inspector.

Shows all extracted audio features in a 4-column × 2-row grid:
  Row 1: FFT Spectrum | Mel Bands | Sub-Band Energy | AI Stems
  Row 2: Spectral Centroid | Flux + Onsets | Chroma / Piano | HUD

Usage:
    .venv/Scripts/python.exe visualizer/visualizer_debug_v2.py --file audio.mp3

Keys:
    S        toggle temporal smoothing
    N        toggle normalization
    K / L    contrast −/+
    + / −    mel bands +/−
    SPACE    pause / resume
    ESC      quit
"""
import sys
import argparse
import numpy as np
import pygame
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.feature_extractor import AudioFeatureExtractor

# ── palette ────────────────────────────────────────────────────────────────────
BG        = (8,   8,  12)
PANEL_BG  = (16,  16, 22)
BORDER    = (40,  40, 55)
WHITE     = (220, 220, 230)
GREY      = (110, 110, 130)
PINK      = (236,  72, 153)
BLUE      = ( 59, 130, 246)
CYAN      = (  0, 210, 190)
GOLD      = (255, 190,  50)
GREEN     = ( 60, 215, 100)
RED       = (220,  60,  60)
ORANGE    = (255, 140,  50)
PURPLE    = (180,  80, 220)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SUBBAND_ORDER = ["sub_bass", "bass", "low_mid", "mid", "high_mid", "presence", "brilliance"]
SUBBAND_COLORS = {
    "sub_bass":   (180,  40, 255),
    "bass":       (220,  60,  60),
    "low_mid":    (255, 130,  50),
    "mid":        (255, 220,  50),
    "high_mid":   ( 80, 220, 100),
    "presence":   ( 60, 180, 255),
    "brilliance": (200, 120, 255),
}

STEM_ORDER  = ["vocals", "drums", "bass", "guitar", "piano", "other"]
STEM_COLORS = {
    "vocals": GOLD,
    "drums":  RED,
    "bass":   PINK,
    "guitar": GREEN,
    "piano":  BLUE,
    "other":  GREY,
}

HISTORY_LEN = 300  # ~5 s at 60 fps
SPEC_H      = 200  # rows in the scrolling spectrogram texture


# ── Dashboard ──────────────────────────────────────────────────────────────────
class Dashboard4K:
    def __init__(self, screen: pygame.Surface, extractor: AudioFeatureExtractor):
        self.screen    = screen
        self.extractor = extractor
        self.W, self.H = screen.get_size()

        # Fonts — scale relative to 1080p baseline
        scale  = self.W / 1920
        self.fs = pygame.font.SysFont("monospace", max(12, int(14 * scale)))
        self.fl = pygame.font.SysFont("monospace", max(16, int(20 * scale)), bold=True)
        self.ft = pygame.font.SysFont("monospace", max(10, int(11 * scale)))

        # Layout constants
        M   = 10   # outer margin
        HH  = 70   # header height
        GAP = 8    # gap between panels

        body_y = HH + M
        body_h = self.H - body_y - M
        body_w = self.W - 2 * M

        row_h = (body_h - GAP) // 2
        col_w = (body_w - 3 * GAP) // 4

        row1_y = body_y
        row2_y = body_y + row_h + GAP

        self.header_rect = pygame.Rect(M, M, self.W - 2 * M, HH - 2 * M)

        panel_names_r1 = ["fft",      "bands", "subbands", "stems"]
        panel_names_r2 = ["centroid", "flux",  "chroma",   "hud"]
        self.panels: dict[str, pygame.Rect] = {}
        for ci, name in enumerate(panel_names_r1):
            x = M + ci * (col_w + GAP)
            self.panels[name] = pygame.Rect(x, row1_y, col_w, row_h)
        for ci, name in enumerate(panel_names_r2):
            x = M + ci * (col_w + GAP)
            self.panels[name] = pygame.Rect(x, row2_y, col_w, row_h)

        # Rolling histories
        self.hist: dict[str, deque] = {}
        for key in ("bass", "mid", "high", "centroid", "flux", "vocal"):
            self.hist[key] = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.hist["onset"] = deque([False] * HISTORY_LEN, maxlen=HISTORY_LEN)
        for n in SUBBAND_ORDER:
            self.hist[f"sb_{n}"] = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        for s in STEM_ORDER:
            self.hist[f"st_{s}"] = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)

        # Scrolling spectrogram texture: (SPEC_H × HISTORY_LEN × 3)
        self._spec_arr  = np.zeros((SPEC_H, HISTORY_LEN, 3), dtype=np.uint8)
        self._spec_surf = pygame.Surface((HISTORY_LEN, SPEC_H))

        # UI state (toggled via keyboard from main loop)
        self.use_smoothing     = True
        self.use_normalization = True

    # ── helpers ────────────────────────────────────────────────────────────────

    def _txt(self, text: str, pos: tuple, color=WHITE, large=False, tiny=False):
        font = self.fl if large else (self.ft if tiny else self.fs)
        self.screen.blit(font.render(text, True, color), pos)

    def _panel_bg(self, rect: pygame.Rect, title: str, title_color=CYAN) -> pygame.Rect:
        """Draw panel background + title; return inner content rect."""
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, BORDER,   rect, width=1, border_radius=6)
        self._txt(title, (rect.x + 10, rect.y + 8), color=title_color, large=True)
        return pygame.Rect(rect.x + 8, rect.y + 38, rect.w - 16, rect.h - 46)

    def _h_bar(self, value: float, rect: pygame.Rect, color: tuple,
               label: str = "", show_val: bool = True):
        """Horizontal filled-bar meter."""
        pygame.draw.rect(self.screen, (28, 28, 38), rect, border_radius=3)
        fw = int(rect.w * max(0.0, min(1.0, value)))
        if fw > 1:
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(rect.x, rect.y, fw, rect.h), border_radius=3)
        if label:
            self._txt(label, (rect.x + 5, rect.y + rect.h // 2 - 7), WHITE)
        if show_val:
            vs = self.ft.render(f"{value:.2f}", True, GREY)
            self.screen.blit(vs, (rect.right - vs.get_width() - 4,
                                  rect.y + rect.h // 2 - vs.get_height() // 2))

    def _line_hist(self, key: str, area: pygame.Rect, color: tuple, lw: int = 2):
        """Draw rolling line graph from a named history deque."""
        pts = list(self.hist[key])
        n   = len(pts)
        if n < 2:
            return
        step   = area.w / (n - 1)
        points = [
            (int(area.x + i * step),
             int(area.bottom - float(pts[i]) * area.h))
            for i in range(n)
        ]
        pygame.draw.lines(self.screen, color, False, points, lw)

    # ── history push ──────────────────────────────────────────────────────────

    def _push(self, features: dict):
        self.hist["bass"].append(float(features.get("bass",     0)))
        self.hist["mid"].append( float(features.get("mid",      0)))
        self.hist["high"].append(float(features.get("high",     0)))
        self.hist["centroid"].append(float(features.get("centroid", 0)))
        self.hist["flux"].append(float(features.get("flux",     0)))
        self.hist["onset"].append(bool(features.get("onset",    False)))
        self.hist["vocal"].append(float(features.get("vocal",   0)))
        sb = features.get("subbands", {})
        for n in SUBBAND_ORDER:
            self.hist[f"sb_{n}"].append(float(sb.get(n, 0)))
        st = features.get("stems", {})
        for s in STEM_ORDER:
            self.hist[f"st_{s}"].append(float(st.get(s, 0)))
        self._push_spec(features)

    def _push_spec(self, features: dict):
        """Append one column to the scrolling spectrogram texture."""
        spectrum = np.asarray(features.get("spectrum", np.zeros(SPEC_H)))
        h = SPEC_H
        col = np.zeros(h, dtype=float)
        n   = min(len(spectrum), h)
        col[:n] = spectrum[:n]
        col = col[::-1]   # flip: low-freq at bottom

        r = np.clip(col * 60,  0, 255).astype(np.uint8)
        g = np.clip(col * 210, 0, 255).astype(np.uint8)
        b = np.clip(col * 255, 20, 255).astype(np.uint8)

        self._spec_arr[:, :-1] = self._spec_arr[:, 1:]
        self._spec_arr[:, -1, 0] = r
        self._spec_arr[:, -1, 1] = g
        self._spec_arr[:, -1, 2] = b
        pygame.surfarray.blit_array(self._spec_surf,
                                    self._spec_arr.transpose(1, 0, 2))

    # ── panel draws ───────────────────────────────────────────────────────────

    def _draw_header(self, features: dict, t: float):
        r = self.header_rect
        pygame.draw.rect(self.screen, PANEL_BG, r, border_radius=6)
        pygame.draw.rect(self.screen, BORDER,   r, width=1, border_radius=6)

        pitch = NOTE_NAMES[int(features.get("dominant_pitch", 0))]
        info  = (f"  ARC DEBUG v2   |   t={t:.2f}s   "
                 f"frame={features.get('frame_idx', 0)}   "
                 f"pitch={pitch}   "
                 f"centroid={features.get('centroid', 0):.3f}   "
                 f"flux={features.get('flux', 0):.3f}   "
                 f"onset={'YES' if features.get('onset') else ' no'}   "
                 f"pulse={features.get('pulse', 1):.3f}")
        self._txt(info, (r.x + 12, r.y + r.h // 2 - 11), WHITE, large=True)

        bx = r.right - 440
        sm_c = GREEN if self.use_smoothing     else RED
        nm_c = GREEN if self.use_normalization else RED
        self._txt(f"[S] smooth : {'ON ' if self.use_smoothing     else 'OFF'}",
                  (bx, r.y + 6),  sm_c)
        self._txt(f"[N] norm   : {'ON ' if self.use_normalization else 'OFF'}",
                  (bx, r.y + 28), nm_c)
        self._txt("[K/L] contrast  [+/−] bands  [SPACE] pause  [ESC] quit",
                  (bx - 520, r.bottom - 20), GREY, tiny=True)

    # panel 1 — FFT
    def _draw_fft(self, features: dict):
        inner = self._panel_bg(self.panels["fft"], "FFT SPECTRUM", CYAN)
        spectrum = np.asarray(features.get("spectrum", np.zeros(512)))
        n_bins   = min(len(spectrum), 512)

        bar_h = inner.h * 55 // 100
        bw    = max(1, inner.w // n_bins)

        for i in range(n_bins):
            v  = float(spectrum[i])
            vh = int(v * bar_h)
            if vh < 1:
                continue
            t_c = i / n_bins
            c   = (int(255 * (1 - t_c)), int(80 + 120 * (1 - abs(t_c - 0.5) * 2)), int(255 * t_c))
            pygame.draw.rect(self.screen, c,
                             (inner.x + i * bw, inner.y + bar_h - vh, max(1, bw - 1), vh))

        # Scrolling spectrogram
        spec_y = inner.y + bar_h + 4
        spec_h = inner.bottom - spec_y
        if spec_h > 10:
            scaled = pygame.transform.scale(self._spec_surf, (inner.w, spec_h))
            self.screen.blit(scaled, (inner.x, spec_y))
            pygame.draw.rect(self.screen, BORDER,
                             pygame.Rect(inner.x, spec_y, inner.w, spec_h), 1)
            self._txt("▼ scrolling spectrogram (low freq at bottom)",
                      (inner.x + 4, spec_y + 4), GREY, tiny=True)

    # panel 2 — Mel bands
    def _draw_bands(self, features: dict):
        inner = self._panel_bg(self.panels["bands"], "MEL FREQUENCY BANDS", BLUE)
        bands = np.asarray(features.get("bands", np.zeros(32)))
        n     = len(bands)

        short_labels_3 = ["BASS", "MID", "HIGH"]
        short_labels_6 = ["SUB", "BASS", "LO-M", "MID", "HI-M", "PRES"]

        bh = max(4, (inner.h - n * 3) // max(1, n))
        for i, v in enumerate(bands):
            t_c = i / max(1, n - 1)
            c   = (int(220 * (1 - t_c) + 59  * t_c),
                   int(60  * (1 - t_c) + 130 * t_c),
                   int(80  * (1 - t_c) + 246 * t_c))
            if n <= 3:
                lbl = short_labels_3[i]
            elif n <= 6:
                lbl = short_labels_6[i] if i < 6 else f"B{i+1}"
            else:
                lbl = f"B{i+1}"
            rect = pygame.Rect(inner.x, inner.y + i * (bh + 3), inner.w, bh)
            self._h_bar(float(v), rect, c, label=lbl, show_val=(n <= 32))

    # panel 3 — Sub-bands
    def _draw_subbands(self, features: dict):
        inner   = self._panel_bg(self.panels["subbands"], "SUB-BAND ENERGY (Hz RANGES)", ORANGE)
        sb      = features.get("subbands", {})
        n       = len(SUBBAND_ORDER)
        bar_h   = max(8, (inner.h * 55 // 100 - n * 4) // max(1, n))
        bar_w   = inner.w * 78 // 100

        for i, name in enumerate(SUBBAND_ORDER):
            v = float(sb.get(name, 0))
            c = SUBBAND_COLORS[name]
            r = pygame.Rect(inner.x, inner.y + i * (bar_h + 4), bar_w, bar_h)
            self._h_bar(v, r, c, label=name.replace("_", " ").upper())

        hist_y = inner.y + n * (bar_h + 4) + 8
        ha     = pygame.Rect(inner.x, hist_y, inner.w, inner.bottom - hist_y)
        if ha.h > 24:
            pygame.draw.rect(self.screen, (20, 20, 28), ha, border_radius=4)
            self._txt("energy history", (ha.x + 4, ha.y + 2), GREY, tiny=True)
            for name in SUBBAND_ORDER:
                self._line_hist(f"sb_{name}", ha, SUBBAND_COLORS[name], lw=1)

    # panel 4 — AI stems
    def _draw_stems(self, features: dict):
        no_service = (self.extractor.stem_service is None)
        title_color = (140, 140, 60) if no_service else GOLD
        inner = self._panel_bg(self.panels["stems"], "AI STEMS ENERGY", title_color)
        st    = features.get("stems", {})
        n     = len(STEM_ORDER)
        bar_h = max(10, (inner.h * 55 // 100 - n * 4) // max(1, n))
        bar_w = inner.w * 78 // 100

        for i, name in enumerate(STEM_ORDER):
            v = float(st.get(name, 0))
            c = STEM_COLORS[name] if not no_service else (50, 50, 60)
            r = pygame.Rect(inner.x, inner.y + i * (bar_h + 4), bar_w, bar_h)
            self._h_bar(v, r, c, label=name.upper())

        hist_y = inner.y + n * (bar_h + 4) + 8
        ha     = pygame.Rect(inner.x, hist_y, inner.w, inner.bottom - hist_y)
        if ha.h > 24:
            pygame.draw.rect(self.screen, (20, 20, 28), ha, border_radius=4)
            if no_service:
                # Centre-align a two-line explanation
                msg1 = "StemService unavailable"
                msg2 = "Install demucs or use --mode vocals"
                msg3 = "Stems default to 0 until separation runs"
                for j, msg in enumerate((msg1, msg2, msg3)):
                    surf = self.fs.render(msg, True, (160, 160, 80))
                    self.screen.blit(surf, (ha.x + (ha.w - surf.get_width()) // 2,
                                            ha.y + 20 + j * 28))
            else:
                self._txt("energy history", (ha.x + 4, ha.y + 2), GREY, tiny=True)
                for name in STEM_ORDER:
                    self._line_hist(f"st_{name}", ha, STEM_COLORS[name], lw=1)

    # panel 5 — Spectral centroid
    def _draw_centroid(self, features: dict):
        inner    = self._panel_bg(self.panels["centroid"], "SPECTRAL CENTROID", PURPLE)
        centroid = float(features.get("centroid", 0))
        nyquist  = self.extractor.sample_rate / 2.0
        hz       = centroid * nyquist

        self._txt(f"{centroid:.4f}   ({hz:.0f} Hz)", (inner.x, inner.y), PURPLE, large=True)

        # Frequency gradient ruler
        ruler_y = inner.y + 46
        ruler_h = max(20, inner.h // 14)
        for px in range(inner.w):
            t_c = px / inner.w
            c   = (int(180 * (1 - t_c) + 60  * t_c),
                   int(60  * (1 - t_c) + 80  * t_c),
                   int(80  * (1 - t_c) + 255 * t_c))
            pygame.draw.line(self.screen, c,
                             (inner.x + px, ruler_y),
                             (inner.x + px, ruler_y + ruler_h))
        # Cursor
        cx = inner.x + int(centroid * inner.w)
        pygame.draw.rect(self.screen, WHITE, (cx - 2, ruler_y - 6, 4, ruler_h + 12))
        # Freq tick labels
        for freq in [50, 200, 500, 1000, 2000, 5000, 10000, 20000]:
            tx = inner.x + int(min(freq / nyquist, 1.0) * inner.w)
            pygame.draw.line(self.screen, GREY, (tx, ruler_y + ruler_h), (tx, ruler_y + ruler_h + 6))
            self._txt(f"{freq}", (tx - 14, ruler_y + ruler_h + 8), GREY, tiny=True)

        # History
        ha = pygame.Rect(inner.x, ruler_y + ruler_h + 28, inner.w, inner.bottom - ruler_y - ruler_h - 36)
        if ha.h > 24:
            pygame.draw.rect(self.screen, (20, 20, 28), ha, border_radius=4)
            self._txt("centroid (purple)  bass (red)  high (blue)",
                      (ha.x + 4, ha.y + 2), GREY, tiny=True)
            self._line_hist("bass",     ha, RED,    lw=1)
            self._line_hist("high",     ha, BLUE,   lw=1)
            self._line_hist("centroid", ha, PURPLE, lw=2)

    # panel 6 — Flux + onsets
    def _draw_flux(self, features: dict):
        inner  = self._panel_bg(self.panels["flux"], "SPECTRAL FLUX + ONSETS", GREEN)
        flux   = float(features.get("flux",  0))
        onset  = bool(features.get("onset", False))

        flux_c   = RED if flux > 0.7 else (GOLD if flux > 0.3 else GREEN)
        onset_c  = RED if onset else GREY
        self._txt(f"FLUX: {flux:.4f}", (inner.x, inner.y), flux_c, large=True)
        self._txt(f"ONSET: {'■ DETECTED' if onset else '○ none'}",
                  (inner.x + inner.w // 2, inner.y), onset_c, large=True)

        ha = pygame.Rect(inner.x, inner.y + 42, inner.w, inner.h - 42)
        pygame.draw.rect(self.screen, (20, 20, 28), ha, border_radius=4)
        self._txt("flux (green) — red ticks = onset frames",
                  (ha.x + 4, ha.y + 2), GREY, tiny=True)

        # Onset ticks (draw before flux line so line sits on top)
        step = ha.w / max(1, HISTORY_LEN - 1)
        onset_list = list(self.hist["onset"])
        for i, has_onset in enumerate(onset_list):
            if has_onset:
                ox = int(ha.x + i * step)
                pygame.draw.line(self.screen, (200, 50, 50),
                                 (ox, ha.y + 16), (ox, ha.bottom), 1)

        self._line_hist("flux", ha, GREEN, lw=2)

        # Current onset flash: bright border pulse
        if onset:
            pygame.draw.rect(self.screen, RED, self.panels["flux"], width=3, border_radius=6)

    # panel 7 — Chroma / piano
    def _draw_chroma(self, features: dict):
        inner     = self._panel_bg(self.panels["chroma"], "CHROMA + DOMINANT PITCH", PINK)
        chroma    = np.asarray(features.get("chroma", np.zeros(12)))
        dom_pitch = int(features.get("dominant_pitch", 0))

        bw         = inner.w // 12
        bar_area_h = inner.h * 52 // 100

        for i in range(12):
            v      = float(chroma[i])
            is_dom = i == dom_pitch
            c      = GOLD if is_dom else (
                int(59  + (236 - 59)  * v),
                int(130 - 80          * v),
                int(246 - 90          * v + 50 * v),
            )
            bh = int(v * bar_area_h)
            bx = inner.x + i * bw

            pygame.draw.rect(self.screen, (24, 24, 34),
                             (bx + 1, inner.y, bw - 2, bar_area_h))
            if bh > 0:
                pygame.draw.rect(self.screen, c,
                                 (bx + 1, inner.y + bar_area_h - bh, bw - 2, bh))
            # Dominant-pitch marker
            if is_dom:
                pygame.draw.rect(self.screen, GOLD,
                                 (bx + 1, inner.y, bw - 2, bar_area_h), width=2,
                                 border_radius=2)
            # Note name
            nc = GOLD if is_dom else (GREY if "#" in NOTE_NAMES[i] else WHITE)
            self._txt(NOTE_NAMES[i],
                      (bx + bw // 2 - 7, inner.y + bar_area_h + 4), nc)

        dp_y = inner.y + bar_area_h + 28
        self._txt(f"DOMINANT PITCH:  {NOTE_NAMES[dom_pitch]}  (class {dom_pitch})",
                  (inner.x, dp_y), GOLD, large=True)

        piano_y = dp_y + 44
        piano_h = inner.bottom - piano_y - 4
        if piano_h > 20:
            self._draw_piano(inner.x, piano_y, inner.w, piano_h, chroma, dom_pitch)

    def _draw_piano(self, x: int, y: int, w: int, h: int,
                    chroma: np.ndarray, dom_pitch: int):
        white_notes = [0, 2, 4, 5, 7, 9, 11]
        black_offsets = [0.65, 1.65, 3.65, 4.65, 5.65]
        black_notes   = [1, 3, 6, 8, 10]
        wk_w = w // 7

        # White keys
        for i, note in enumerate(white_notes):
            v      = float(chroma[note])
            is_dom = note == dom_pitch
            base   = GOLD if is_dom else (240, 240, 245)
            tint   = (
                int(base[0] * (1 - v * 0.35)),
                int(base[1] * (1 - v * 0.35)),
                int(base[2] * (1 - v * 0.55) + 200 * v * 0.55),
            ) if not is_dom else base
            kx = x + i * wk_w
            pygame.draw.rect(self.screen, tint, (kx + 1, y, wk_w - 2, h), border_radius=3)
            pygame.draw.rect(self.screen, BORDER, (kx + 1, y, wk_w - 2, h), width=1, border_radius=3)

        # Black keys
        bk_w = int(wk_w * 0.55)
        bk_h = int(h * 0.62)
        for offset, note in zip(black_offsets, black_notes):
            v      = float(chroma[note])
            is_dom = note == dom_pitch
            c      = GOLD if is_dom else (int(15 + 220 * v), int(15 + 170 * v), int(15 + 80 * v))
            kx     = int(x + offset * wk_w - bk_w // 2)
            pygame.draw.rect(self.screen, c, (kx, y, bk_w, bk_h), border_radius=2)

    # panel 8 — HUD
    def _draw_hud(self, features: dict, t: float):
        inner = self._panel_bg(self.panels["hud"], "CURRENT VALUES + CONTROLS", WHITE)

        rows = [
            ("TIME",     f"{t:.3f} s"),
            ("FRAME",    f"{features.get('frame_idx', 0)}"),
            ("PULSE",    f"{features.get('pulse', 1):.4f}"),
            ("BASS",     f"{features.get('bass',  0):.4f}"),
            ("MID",      f"{features.get('mid',   0):.4f}"),
            ("HIGH",     f"{features.get('high',  0):.4f}"),
            ("VOCAL",    f"{features.get('vocal', 0):.4f}"),
            ("CENTROID", f"{features.get('centroid', 0):.4f}"),
            ("FLUX",     f"{features.get('flux',     0):.4f}"),
            ("ONSET",    f"{'YES' if features.get('onset') else 'no'}"),
            ("PITCH",    f"{NOTE_NAMES[int(features.get('dominant_pitch', 0))]} "
                         f"({features.get('dominant_pitch', 0)})"),
        ]
        sb = features.get("subbands", {})
        for n in SUBBAND_ORDER:
            rows.append((n.upper().replace("_", " "), f"{sb.get(n, 0):.4f}"))
        st = features.get("stems", {})
        for s in STEM_ORDER:
            rows.append((f"STEM/{s.upper()}", f"{st.get(s, 0):.4f}"))

        # Adaptive row height
        lh  = max(16, inner.h // (len(rows) + 5))
        cx2 = inner.x + inner.w // 2

        for i, (k, v) in enumerate(rows):
            ry = inner.y + i * lh
            if ry + lh > inner.bottom - 58:
                break
            vc = GREEN if (k == "ONSET" and v == "YES") else WHITE
            self._txt(k, (inner.x + 4, ry), GREY)
            self._txt(v, (cx2, ry), vc)

        # Controls legend at bottom
        ctrl_lines = [
            f"contrast={self.extractor.contrast_level:.2f}  "
            f"bands={self.extractor.num_bands}  "
            f"mode={self.extractor.separation_mode}",
            "[K/L] contrast  [+/−] bands  [S] smooth  [N] norm",
            "[SPACE] pause  [ESC] quit",
        ]
        for i, line in enumerate(ctrl_lines):
            self._txt(line, (inner.x + 4, inner.bottom - (len(ctrl_lines) - i) * 18), GREY, tiny=True)

    # ── main render ────────────────────────────────────────────────────────────

    def render(self, features: dict, t: float):
        self._push(features)
        self.screen.fill(BG)
        self._draw_header(features, t)
        self._draw_fft(features)
        self._draw_bands(features)
        self._draw_subbands(features)
        self._draw_stems(features)
        self._draw_centroid(features)
        self._draw_flux(features)
        self._draw_chroma(features)
        self._draw_hud(features, t)


# ── entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ARC Debug Dashboard v2")
    parser.add_argument("--file", "-f", required=True,
                        help="Audio file to analyse")
    parser.add_argument("--fps",    type=int, default=60)
    parser.add_argument("--mode",   default="demucs",
                        choices=["vocals", "demucs", "roformer"])
    parser.add_argument("--width",  type=int, default=3840,
                        help="Window width  (default 3840 for 4K)")
    parser.add_argument("--height", type=int, default=2160,
                        help="Window height (default 2160 for 4K)")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption(f"ARC Debug Dashboard v2 — {file_path.name}")
    clock  = pygame.time.Clock()

    print(f"Loading: {file_path.name}")
    extractor = AudioFeatureExtractor(str(file_path), fps=args.fps,
                                      separation_mode=args.mode)
    dashboard = Dashboard4K(screen, extractor)

    pygame.mixer.music.load(str(file_path))
    pygame.mixer.music.play()

    use_smoothing     = True
    use_normalization = True
    running           = True

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    use_smoothing = not use_smoothing
                    dashboard.use_smoothing = use_smoothing
                elif event.key == pygame.K_n:
                    use_normalization = not use_normalization
                    dashboard.use_normalization = use_normalization
                elif event.key == pygame.K_k:
                    extractor.update_contrast(-0.02)
                elif event.key == pygame.K_l:
                    extractor.update_contrast(0.02)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    extractor.update_num_bands(extractor.num_bands + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    extractor.update_num_bands(extractor.num_bands - 1)
                elif event.key == pygame.K_SPACE:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.pause()
                    else:
                        pygame.mixer.music.unpause()

        pos_ms = pygame.mixer.music.get_pos()
        if pos_ms == -1:
            running = False
            continue

        features = extractor.get_features_at_time(
            pos_ms / 1000.0,
            use_smoothing=use_smoothing,
            use_normalization=use_normalization,
        )
        if features:
            dashboard.render(features, pos_ms / 1000.0)
            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
