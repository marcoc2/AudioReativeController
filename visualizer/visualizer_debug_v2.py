"""
ARC Debug Dashboard v2 — 4K feature inspector.

Shows all extracted audio features in a 4-column × 2-row grid:
  Row 1: FFT Spectrum | Mel Bands | Sub-Band Energy | AI Stems
  Row 2: Spectral Centroid | Flux + Onsets | Chroma / Piano | HUD
  Texture: Loudness + Swell | Harmonic Change | Percussive / Noisiness | Tremolo
  Rhythm:  Rhythm Grid | MIDI Notes (2 cols) | MIDI Automation Lanes  (only with a grid)
  Score:   Harmony Now | Score Timeline (2 cols) | Song Map    (only with a score)

The rhythm row follows the bar/beat grid that drives triggers. The grid comes
from --midi (with --midi-offset), else from the score, else from --bpm.

The texture row plots core/texture — what moves in the sound when nothing
attacks — over a window of -15 s .. +5 s around the playhead.

Row 3 shows what SheetSage2 transcribed — key, chords, sections, beats and
melody (see core/score). It appears when --score points at a SheetSage2
output folder, or when a "sheetsage" folder sits next to the audio file.

Usage:
    .venv/Scripts/python.exe visualizer/visualizer_debug_v2.py --file audio.mp3
    ... --score path/to/sheetsage_output     explicit transcription folder
    ... --no-stems                           skip AI stem separation (no GPU work)
    ... --midi drums.mid --midi-offset 0.059 grid + notes + automation lanes
    ... --stem solo=solo_guitarra.mp3        extra original stem (repeatable)

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
import colorsys
import numpy as np
import pygame
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataclasses import dataclass, field

from core.feature_extractor import AudioFeatureExtractor
from core.rhythm import RhythmGrid, read_midi, shift_in_time
from core.rhythm.midi_automation import MidiAutomationReader
from core.score import Score, has_score, parse_chord, read_score


@dataclass
class RhythmInput:
    """The bar/beat grid the dashboard follows, and the MIDI it came with (if any)."""
    grid: RhythmGrid
    source: str                                   # "MIDI", "score (SheetSage2)", "--bpm"
    notes: list = field(default_factory=list)     # MidiNote, already shifted to audio time
    automation: MidiAutomationReader | None = None
    offset: float = 0.0                           # t_audio = t_midi + offset

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

SECTION_COLORS = {
    "intro": (70, 110, 160), "outro": (70, 110, 160), "fade-out": (60, 80, 110),
    "verse": (60, 150, 110), "pre-chorus": (150, 160, 60), "chorus": (220, 120, 50),
    "post-chorus": (200, 150, 70), "bridge": (160, 80, 190), "interlude": (90, 90, 140),
    "instrumental": (90, 90, 140), "solo": (220, 70, 120), "silence": (35, 35, 45),
}
GM_DRUMS = {   # General MIDI percussion names; a kit mapped differently will not match
    35: "kick", 36: "kick", 37: "rim", 38: "snare", 39: "clap", 40: "snare 2",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    42: "hat", 44: "hat ped", 46: "hat open", 49: "crash", 57: "crash 2",
    51: "ride", 59: "ride 2", 53: "bell", 52: "china", 55: "splash", 56: "cowbell",
}
MAX_NOTE_LANES = 14      # busiest pitches shown in the MIDI notes panel

TEXTURE_HUD_LABELS = {   # short enough for the HUD's key column at 1080p
    "loudness": "TX LOUD", "swell": "TX SWELL", "harmonic_change": "TX HARM CHG",
    "percussive": "TX PERC", "noisiness": "TX NOISE",
    "tremolo_depth": "TX TREM", "tremolo_rate": "TX TREM Hz",
}
TEXTURE_PAST    = 15.0   # texture is slow: a long look back ...
TEXTURE_FUTURE  = 5.0    # ... and a short look ahead of the playhead
TREMOLO_MAX_HZ  = 12.0   # top of the tremolo-rate curve (core.texture.TREMOLO_RANGE)
SCORE_PAST   = 4.0    # seconds of score shown behind the playhead
SCORE_FUTURE = 12.0   # ... and ahead of it: what is about to be played

HISTORY_LEN = 300  # ~5 s at 60 fps
SPEC_H      = 200  # rows in the scrolling spectrogram texture


# ── Dashboard ──────────────────────────────────────────────────────────────────
class Dashboard4K:
    def __init__(self, screen: pygame.Surface, extractor: AudioFeatureExtractor,
                 score: Score | None = None, rhythm: RhythmInput | None = None):
        self.screen    = screen
        self.extractor = extractor
        self.score     = score
        self.rhythm    = rhythm
        extra = [s for s in getattr(extractor, "stems_energy", {}) if s not in STEM_ORDER]
        self.stem_names = STEM_ORDER + sorted(extra)
        self.W, self.H = screen.get_size()

        # Fonts — scale relative to 1080p baseline
        scale  = self.W / 1920
        self.fs = pygame.font.SysFont("monospace", max(12, int(14 * scale)))
        self.fl = pygame.font.SysFont("monospace", max(16, int(20 * scale)), bold=True)
        self.ft = pygame.font.SysFont("monospace", max(10, int(11 * scale)))
        self.fx = pygame.font.SysFont("monospace", max(28, int(54 * scale)), bold=True)

        # Layout constants
        M   = 10   # outer margin
        HH  = 70   # header height
        GAP = 8    # gap between panels

        body_y = HH + M
        body_h = self.H - body_y - M
        body_w = self.W - 2 * M

        # Rows share the body height by weight. Texture curves are slow and
        # need less height than the spectral panels; the score row only exists
        # when there is a transcription to show.
        row_weights = {"features_1": 1.0, "features_2": 1.0, "texture": 0.6}
        if rhythm is not None:
            row_weights["rhythm"] = 0.8
        if score is not None:
            row_weights["score"] = 1.0
        usable  = body_h - (len(row_weights) - 1) * GAP
        total_w = sum(row_weights.values())
        rows, y = {}, body_y
        for name, weight in row_weights.items():
            h = int(usable * weight / total_w)
            rows[name] = (y, h)
            y += h + GAP
        col_w = (body_w - 3 * GAP) // 4

        self.header_rect = pygame.Rect(M, M, self.W - 2 * M, HH - 2 * M)

        row_panels = {
            "features_1": ["fft",       "bands",    "subbands",    "stems"],
            "features_2": ["centroid",  "flux",     "chroma",      "hud"],
            "texture":    ["tex_loud",  "tex_harm", "tex_surface", "tex_trem"],
        }
        self.panels: dict[str, pygame.Rect] = {}
        for row, names in row_panels.items():
            ry, rh = rows[row]
            for ci, name in enumerate(names):
                self.panels[name] = pygame.Rect(M + ci * (col_w + GAP), ry, col_w, rh)

        if rhythm is not None:
            ry, rh = rows["rhythm"]
            self.panels["rhythm_grid"]  = pygame.Rect(M, ry, col_w, rh)
            self.panels["rhythm_notes"] = pygame.Rect(M + col_w + GAP, ry, 2 * col_w + GAP, rh)
            self.panels["rhythm_lanes"] = pygame.Rect(M + 3 * (col_w + GAP), ry, col_w, rh)
            counts: dict[int, int] = {}
            for n in rhythm.notes:
                counts[n.pitch] = counts.get(n.pitch, 0) + 1
            busiest = sorted(counts, key=counts.get, reverse=True)[:MAX_NOTE_LANES]
            self._note_lanes = sorted(busiest)            # low pitch at the bottom
            self._note_times = np.array([n.time for n in rhythm.notes])

        if score is not None:
            row3_y, row_h = rows["score"]
            self.panels["harmony"]  = pygame.Rect(M, row3_y, col_w, row_h)
            self.panels["timeline"] = pygame.Rect(M + col_w + GAP, row3_y, 2 * col_w + GAP, row_h)
            self.panels["songmap"]  = pygame.Rect(M + 3 * (col_w + GAP), row3_y, col_w, row_h)
            pitches = [n.pitch for n in score.melody_vocal + score.melody_instrumental]
            self._pitch_lo = (min(pitches) if pitches else 48) - 2
            self._pitch_hi = (max(pitches) if pitches else 84) + 2
        self._songmap_surf: pygame.Surface | None = None

        # Rolling histories
        self.hist: dict[str, deque] = {}
        for key in ("bass", "mid", "high", "centroid", "flux", "vocal"):
            self.hist[key] = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.hist["onset"] = deque([False] * HISTORY_LEN, maxlen=HISTORY_LEN)
        for n in SUBBAND_ORDER:
            self.hist[f"sb_{n}"] = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        for s in self.stem_names:
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
        top = max(38, 8 + self.fl.get_height() + 8)   # title font grows with the window
        return pygame.Rect(rect.x + 8, rect.y + top, rect.w - 16, rect.h - top - 8)

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
        for s in self.stem_names:
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
        n     = len(self.stem_names)
        bar_h = max(10, (inner.h * 55 // 100 - n * 4) // max(1, n))
        bar_w = inner.w * 78 // 100

        for i, name in enumerate(self.stem_names):
            v = float(st.get(name, 0))
            extra = name not in STEM_COLORS        # a stem the user supplied: always real data
            c = ORANGE if extra else (STEM_COLORS[name] if not no_service else (50, 50, 60))
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
                for name in self.stem_names:
                    self._line_hist(f"st_{name}", ha, STEM_COLORS.get(name, ORANGE), lw=1)

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
        if self.score is not None:
            key, chord = self.score.key_at(t), self.score.chord_at(t)
            section    = self.score.section_at(t)
            bar, beat  = self.score.bar_beat_at(t)
            rows += [
                ("KEY",      key.label if key else "-"),
                ("CHORD",    chord.label if chord else "-"),
                ("SECTION",  section.label if section else "-"),
                ("BAR.BEAT", f"{bar}.{beat}"),
            ]
        for name, value in features.get("texture", {}).items():
            rows.append((TEXTURE_HUD_LABELS.get(name, name.upper()),
                         f"{value:+.4f}" if name == "swell" else f"{value:.4f}"))
        sb = features.get("subbands", {})
        for n in SUBBAND_ORDER:
            rows.append((n.upper().replace("_", " "), f"{sb.get(n, 0):.4f}"))
        st = features.get("stems", {})
        for s in self.stem_names:
            rows.append((f"STEM/{s.upper()}", f"{st.get(s, 0):.4f}"))

        # Rows never overlap: line height follows the font, and when one column
        # cannot hold them all (a third panel row makes this panel shorter) they
        # wrap into a second key/value column.
        lh       = self.fs.get_height() + 2
        legend_h = 3 * (self.ft.get_height() + 4) + 6
        per_col  = max(1, (inner.h - legend_h) // lh)
        n_cols   = 1 if len(rows) <= per_col else 2
        col_w    = inner.w // n_cols

        # When not everything fits, the last slot says so instead of dropping rows
        # silently (every value is also drawn in its own panel).
        capacity = per_col * n_cols
        if len(rows) > capacity:
            capacity -= 1
        for i, (k, v) in enumerate(rows[:capacity]):
            cx = inner.x + (i // per_col) * col_w
            ry = inner.y + (i % per_col) * lh
            vc = GREEN if (k == "ONSET" and v == "YES") else WHITE
            self._txt(k, (cx + 4, ry), GREY)
            self._txt(v, (cx + col_w // 2, ry), vc)
        if len(rows) > capacity:
            self._txt(f"+{len(rows) - capacity} more (see panels)",
                      (inner.x + (n_cols - 1) * col_w + 4, inner.y + (per_col - 1) * lh), GREY)

        # Controls legend at bottom
        ctrl_lines = [
            f"contrast={self.extractor.contrast_level:.2f}  "
            f"bands={self.extractor.num_bands}  "
            f"mode={self.extractor.separation_mode}",
            "[K/L] contrast  [+/−] bands  [S] smooth  [N] norm",
            "[SPACE] pause  [ESC] quit",
        ]
        ctrl_lh = self.ft.get_height() + 4
        for i, line in enumerate(ctrl_lines):
            self._txt(line, (inner.x + 4, inner.bottom - (len(ctrl_lines) - i) * ctrl_lh), GREY, tiny=True)

    # ── rhythm row — the grid and the MIDI that drives triggers ───────────────

    def _draw_rhythm_grid(self, t: float):
        rh    = self.rhythm
        grid  = rh.grid
        inner = self._panel_bg(self.panels["rhythm_grid"], f"RHYTHM GRID  ({rh.source})", GREEN)
        num, den  = grid.time_signature
        bar_phase, beat_phase = grid.phase(t), grid.beat_phase(t)
        started   = t >= grid.start_offset
        bar       = int((t - grid.start_offset) // grid.bar_duration) + 1 if started else 0
        beat      = int(bar_phase * num) + 1 if started else 0

        self._txt(f"{grid.bpm:.2f} BPM   {num}/{den}   offset {rh.offset:+.3f}s",
                  (inner.x, inner.y), WHITE)
        big   = self.fx.render(f"{bar}.{beat}", True, GOLD if beat == 1 else WHITE)
        big_y = inner.y + self.fs.get_height() + 4
        self.screen.blit(big, (inner.x, big_y))

        bar_h = self.fs.get_height() + 4
        y     = big_y + big.get_height() + 6
        if y + 2 * bar_h + 6 <= inner.bottom:
            self._h_bar(bar_phase,  pygame.Rect(inner.x, y, inner.w, bar_h), GREEN, label="BAR PHASE")
            self._h_bar(beat_phase, pygame.Rect(inner.x, y + bar_h + 6, inner.w, bar_h), CYAN,
                        label="BEAT PHASE")
        # beat lamps: one per beat of the bar, the current one lit
        lamp = max(10, big.get_height() // 2)
        for i in range(num):
            on = started and i == beat - 1
            color = (GOLD if i == 0 else CYAN) if on else (40, 40, 55)
            pygame.draw.circle(self.screen, color,
                               (inner.right - (num - i) * (lamp + 8) + lamp // 2, big_y + big.get_height() // 2),
                               lamp // 2)
        if started and beat == 1 and beat_phase < 0.25:          # downbeat flash
            pygame.draw.rect(self.screen, GOLD, self.panels["rhythm_grid"], width=3, border_radius=6)

    def _draw_rhythm_notes(self, t: float):
        rh    = self.rhythm
        title = "MIDI NOTES  (GM names, -{:.0f}s ... +{:.0f}s)".format(SCORE_PAST, SCORE_FUTURE)
        inner = self._panel_bg(self.panels["rhythm_notes"], title, GREEN)
        t0, t1   = t - SCORE_PAST, t + SCORE_FUTURE
        label_w  = self.ft.size("127 hat open ")[0]
        area     = pygame.Rect(inner.x + label_w, inner.y, inner.w - label_w, inner.h - 14)
        px_per_s = area.w / (t1 - t0)
        pygame.draw.rect(self.screen, (20, 20, 28), area, border_radius=3)

        grid = rh.grid                                   # bar lines, numbered
        first = int(np.floor((t0 - grid.start_offset) / grid.bar_duration))
        for b in range(max(0, first), int((t1 - grid.start_offset) / grid.bar_duration) + 1):
            x = area.x + int((grid.start_offset + b * grid.bar_duration - t0) * px_per_s)
            if area.x <= x <= area.right:
                pygame.draw.line(self.screen, (70, 70, 95), (x, area.y), (x, area.bottom), 1)
                self._txt(str(b + 1), (x + 3, area.bottom + 1), GREY, tiny=True)

        if not self._note_lanes:
            self._txt("no MIDI notes loaded (--midi)", (area.x + 8, area.y + 8), GREY)
        lane_h = area.h / max(1, len(self._note_lanes))
        lane_y = {p: area.bottom - (i + 1) * lane_h for i, p in enumerate(self._note_lanes)}
        for p, y in lane_y.items():
            self._txt(f"{p:3d} {GM_DRUMS.get(p, '')}", (inner.x, int(y + lane_h / 2 - 6)), GREY, tiny=True)
        lo, hi = np.searchsorted(self._note_times, [t0, t1]) if len(self._note_times) else (0, 0)
        for n in rh.notes[lo:hi]:
            if n.pitch not in lane_y:
                continue
            x      = area.x + int((n.time - t0) * px_per_s)
            level  = n.velocity / 127.0
            recent = 0.0 <= t - n.time < 0.12                      # just hit: flash white
            color  = WHITE if recent else (int(60 + 176 * level), int(215 * level + 30), int(100 * level + 40))
            h      = max(3, int(lane_h * (0.35 + 0.6 * level)))
            pygame.draw.rect(self.screen, color,
                             (x, int(lane_y[n.pitch] + (lane_h - h) / 2), max(3, int(0.03 * px_per_s)), h))
        px = area.x + int(SCORE_PAST * px_per_s)
        pygame.draw.line(self.screen, WHITE, (px, area.y), (px, area.bottom), 2)

    def _draw_rhythm_lanes(self, t: float):
        rh    = self.rhythm
        inner = self._panel_bg(self.panels["rhythm_lanes"], "MIDI AUTOMATION LANES (now)", GREEN)
        if rh.automation is None:
            self._txt("no MIDI loaded (--midi)", (inner.x, inner.y), GREY)
            return
        # The reader indexes by frame of MIDI time; undo the offset to look it up.
        frame = max(0, int((t - rh.offset) * self.extractor.fps))
        lanes = [l for l in rh.automation.available_lanes if not l.startswith("ch")]
        lh    = self.fs.get_height() + 6            # _h_bar labels use the small font
        shown = lanes[: max(1, inner.h // lh)]
        for i, lane in enumerate(shown):
            self._h_bar(rh.automation.get(lane, frame),
                        pygame.Rect(inner.x, inner.y + i * lh, inner.w, lh - 3),
                        ORANGE if lane.startswith("cc") else GREEN, label=lane)
        if len(lanes) > len(shown):
            self._txt(f"+{len(lanes) - len(shown)} more", (inner.right - 80, inner.bottom - lh), GREY, tiny=True)

    # ── texture row — what moves when nothing attacks (core/texture) ──────────

    def _texture_window(self, name: str, t: float, n: int):
        """``n`` samples of a texture array across [t - TEXTURE_PAST, t + TEXTURE_FUTURE].

        Read from the precomputed arrays rather than a rolling history: texture
        is slow (a swell can last several seconds), so the window has to be long,
        and the stretch ahead of the playhead shows what is about to happen.
        """
        times = self.extractor.times
        grid  = np.linspace(t - TEXTURE_PAST, t + TEXTURE_FUTURE, n)
        vals  = np.interp(grid, times, self.extractor.texture[name], left=0.0, right=0.0)
        return vals

    def _texture_plot(self, rect: pygame.Rect, t: float, curves, bipolar: bool = False):
        """Plot texture curves in ``rect``. curves: [(name, colour, scale)], value*scale -> 0..1."""
        pygame.draw.rect(self.screen, (20, 20, 28), rect, border_radius=4)
        n   = max(2, rect.w // 2)
        xs  = np.linspace(rect.x, rect.right - 1, n).astype(int)
        mid = rect.centery
        if bipolar:
            pygame.draw.line(self.screen, (50, 50, 66), (rect.x, mid), (rect.right, mid), 1)
        for name, color, scale in curves:
            vals = np.clip(self._texture_window(name, t, n) * scale, -1.0, 1.0)
            if bipolar:
                ys = (mid - vals * (rect.h / 2 - 2)).astype(int)
            else:
                ys = (rect.bottom - 2 - np.clip(vals, 0.0, 1.0) * (rect.h - 4)).astype(int)
            pygame.draw.lines(self.screen, color, False, list(zip(xs.tolist(), ys.tolist())), 2)
        px = rect.x + int(TEXTURE_PAST / (TEXTURE_PAST + TEXTURE_FUTURE) * rect.w)
        pygame.draw.line(self.screen, WHITE, (px, rect.y), (px, rect.bottom), 1)

    def _texture_panel(self, key: str, title: str, color, readout: str, t: float,
                       curves, bipolar: bool = False):
        inner = self._panel_bg(self.panels[key], title, color)
        self._txt(readout, (inner.x, inner.y), color)
        top = inner.y + self.fs.get_height() + 4
        if inner.bottom - top > 12:
            self._texture_plot(pygame.Rect(inner.x, top, inner.w, inner.bottom - top), t, curves, bipolar)

    def _draw_texture(self, features: dict, t: float):
        tex = features.get("texture", {})
        v   = lambda k: float(tex.get(k, 0.0))
        self._texture_panel(
            "tex_loud", "LOUDNESS (dB) + SWELL", GOLD,
            f"loud {v('loudness'):.2f}   swell {v('swell'):+.2f}  (gold / green, -{TEXTURE_PAST:.0f}s..+{TEXTURE_FUTURE:.0f}s)",
            t, [("loudness", GOLD, 1.0), ("swell", GREEN, 1.0)], bipolar=True)
        self._texture_panel(
            "tex_harm", "HARMONIC CHANGE", PINK,
            f"change {v('harmonic_change'):.2f}   (peaks = the harmony moved)",
            t, [("harmonic_change", PINK, 1.0)])
        self._texture_panel(
            "tex_surface", "SURFACE: PERCUSSIVE / NOISINESS", CYAN,
            f"percussive {v('percussive'):.2f} (red)   noisiness {v('noisiness'):.2f} (cyan)",
            t, [("percussive", RED, 1.0), ("noisiness", CYAN, 1.0)])
        self._texture_panel(
            "tex_trem", "TREMOLO (volume pulsation)", PURPLE,
            f"depth {v('tremolo_depth'):.2f} (purple)   rate {v('tremolo_rate'):.2f} Hz (grey, /{TREMOLO_MAX_HZ:.0f})",
            t, [("tremolo_rate", GREY, 1.0 / TREMOLO_MAX_HZ), ("tremolo_depth", PURPLE, 1.0)])

    # ── row 3 — the score (SheetSage2) ────────────────────────────────────────

    @staticmethod
    def _chord_color(label: str) -> tuple:
        """Hue = root around the circle of fifths (neighbours in key look alike);
        minor-family chords are darker; no-chord is a neutral grey."""
        root, quality, _ = parse_chord(label)
        if root is None:
            return (45, 45, 58)
        hue = ((root * 7) % 12) / 12.0
        val = 0.62 if quality.startswith(("min", "dim", "hdim")) else 0.92
        r, g, b = colorsys.hsv_to_rgb(hue, 0.70, val)
        return (int(r * 255), int(g * 255), int(b * 255))

    @staticmethod
    def _section_color(label: str) -> tuple:
        return SECTION_COLORS.get(label, (100, 100, 120))

    def _spans_lane(self, spans, lane: pygame.Rect, t0: float, t1: float,
                    color_of, labels: bool = True):
        """Draw labelled spans as blocks on a time lane covering [t0, t1]."""
        pygame.draw.rect(self.screen, (20, 20, 28), lane, border_radius=3)
        px_per_s = lane.w / (t1 - t0)
        for s in spans:
            if s.end <= t0 or s.start >= t1:
                continue
            x0 = lane.x + int((max(s.start, t0) - t0) * px_per_s)
            x1 = lane.x + int((min(s.end, t1) - t0) * px_per_s)
            if x1 - x0 < 1:
                continue
            pygame.draw.rect(self.screen, color_of(s.label),
                             (x0, lane.y, max(1, x1 - x0 - 1), lane.h), border_radius=3)
            if labels:
                txt = self.fs.render(s.label, True, (10, 10, 14))
                if txt.get_width() + 8 < x1 - x0:
                    self.screen.blit(txt, (x0 + 5, lane.y + (lane.h - txt.get_height()) // 2))

    # panel 9 — what is written right now
    def _draw_harmony(self, features: dict, t: float):
        inner = self._panel_bg(self.panels["harmony"], "HARMONY NOW (SheetSage2)", GOLD)
        sc    = self.score
        key, chord, nxt = sc.key_at(t), sc.chord_at(t), sc.next_chord(t)
        section   = sc.section_at(t)
        bar, beat = sc.bar_beat_at(t)
        label     = chord.label if chord else "-"
        root, _, tones = parse_chord(label)

        self._txt(f"KEY {key.label if key else '-'}   bar {bar}.{beat}",
                  (inner.x, inner.y), WHITE, large=True)
        sec_lbl = section.label if section else "-"
        sec_txt = self.fl.render(sec_lbl.upper(), True, self._section_color(sec_lbl))
        self.screen.blit(sec_txt, (inner.right - sec_txt.get_width(), inner.y))

        big   = self.fx.render(label, True, self._chord_color(label) if root is not None else GREY)
        big_y = inner.y + self.fl.get_height() + 6
        self.screen.blit(big, (inner.x, big_y))
        if nxt is not None:
            self._txt(f"next  {nxt.label}  in {nxt.start - t:4.1f}s",
                      (inner.x + big.get_width() + 24, big_y + big.get_height() // 2 - 10),
                      self._chord_color(nxt.label), large=True)

        # How much of the *heard* chroma sits on the *written* chord tones:
        # a cheap check that transcription and audio agree (and are in sync).
        heard = np.asarray(features.get("chroma", np.zeros(12)), dtype=float)
        match = float(heard[list(tones)].sum() / heard.sum()) if tones and heard.sum() > 0 else 0.0
        bar_y = big_y + big.get_height() + 8
        bar_h = max(14, inner.h // 12)
        self._h_bar(match, pygame.Rect(inner.x, bar_y, inner.w, bar_h), GREEN,
                    label="HEARD CHROMA ON CHORD TONES")

        melody = "  ".join(
            f"{name}:{NOTE_NAMES[n.pitch % 12]}{n.pitch // 12 - 1}"
            for name, n in (("voc", sc.melody_at(t, "vocal")),
                            ("inst", sc.melody_at(t, "instrumental")))
            if n is not None)
        mel_y = bar_y + bar_h + 6
        self._txt(f"melody  {melody or '-'}", (inner.x, mel_y), CYAN)

        piano_y = mel_y + self.fs.get_height() + 8
        piano_h = inner.bottom - piano_y - 4
        if piano_h > 20:
            written = np.zeros(12)
            written[list(tones)] = 1.0
            self._draw_piano(inner.x, piano_y, inner.w, piano_h, written,
                             root if root is not None else -1)

    # panel 10 — a window of the score around the playhead
    def _draw_timeline(self, t: float):
        inner = self._panel_bg(self.panels["timeline"],
                               f"SCORE TIMELINE  (-{SCORE_PAST:.0f}s ... +{SCORE_FUTURE:.0f}s)", GOLD)
        sc       = self.score
        t0, t1   = t - SCORE_PAST, t + SCORE_FUTURE
        px_per_s = inner.w / (t1 - t0)

        def x_of(time: float) -> int:
            return inner.x + int((time - t0) * px_per_s)

        lane_h     = max(18, inner.h // 9)
        sec_lane   = pygame.Rect(inner.x, inner.y, inner.w, lane_h)
        chord_lane = pygame.Rect(inner.x, sec_lane.bottom + 4, inner.w, int(lane_h * 1.5))
        roll       = pygame.Rect(inner.x, chord_lane.bottom + 4, inner.w,
                                 inner.bottom - chord_lane.bottom - 4 - 18)
        self._spans_lane(sc.sections, sec_lane, t0, t1, self._section_color)
        self._spans_lane(sc.chords, chord_lane, t0, t1, self._chord_color)

        # beats through the piano roll; downbeats brighter, with bar numbers
        pygame.draw.rect(self.screen, (20, 20, 28), roll, border_radius=3)
        lo, hi = np.searchsorted(sc.beats, [t0, t1])
        for i in range(lo, hi):
            down = sc.beat_numbers[i] == 1
            x    = x_of(sc.beats[i])
            pygame.draw.line(self.screen, (70, 70, 95) if down else (34, 34, 46),
                             (x, roll.y), (x, roll.bottom), 2 if down else 1)
            if down:
                bar = int(np.searchsorted(sc.downbeats, sc.beats[i], side="right"))
                self._txt(str(bar), (x + 3, roll.bottom + 2), GREY, tiny=True)

        # melody: vocal gold, instrumental cyan; the sounding note turns white
        span   = max(1, self._pitch_hi - self._pitch_lo)
        note_h = max(3, roll.h // span)
        for notes, color in ((sc.melody_instrumental, CYAN), (sc.melody_vocal, GOLD)):
            for n in notes:
                if n.time + n.duration <= t0:
                    continue
                if n.time >= t1:
                    break
                x0 = max(roll.x, x_of(n.time))
                x1 = min(roll.right, x_of(n.time + n.duration))
                y  = roll.bottom - int((n.pitch - self._pitch_lo) / span * roll.h) - note_h
                sounding = n.time <= t < n.time + n.duration
                pygame.draw.rect(self.screen, WHITE if sounding else color,
                                 (x0, y, max(2, x1 - x0 - 1), note_h), border_radius=2)
        self._txt("melody: vocal (gold)  instrumental (cyan)",
                  (roll.x + 4, roll.y + 2), GREY, tiny=True)

        px = x_of(t)
        pygame.draw.line(self.screen, WHITE, (px, inner.y - 2), (px, roll.bottom), 2)

    # panel 11 — the whole song at a glance
    def _draw_songmap(self, t: float):
        inner = self._panel_bg(self.panels["songmap"], "SONG MAP", GOLD)
        sc    = self.score
        dur   = max(sc.duration, float(getattr(self.extractor, "duration", 0.0)), 1e-6)
        map_h = inner.h * 45 // 100

        if self._songmap_surf is None:    # static: drawn once, blitted every frame
            screen, self.screen = self.screen, pygame.Surface((inner.w, map_h))
            self.screen.fill(PANEL_BG)
            lane_h = max(12, map_h // 6)
            self._spans_lane(sc.sections, pygame.Rect(0, 0, inner.w, lane_h), 0.0, dur,
                             self._section_color, labels=False)
            self._spans_lane(sc.chords, pygame.Rect(0, lane_h + 3, inner.w, lane_h), 0.0, dur,
                             self._chord_color, labels=False)
            dots = pygame.Rect(0, 2 * lane_h + 6, inner.w, map_h - 2 * lane_h - 6)
            pygame.draw.rect(self.screen, (20, 20, 28), dots, border_radius=3)
            span = max(1, self._pitch_hi - self._pitch_lo)
            for notes, color in ((sc.melody_instrumental, CYAN), (sc.melody_vocal, GOLD)):
                for n in notes:
                    x0 = int(n.time / dur * inner.w)
                    x1 = int((n.time + n.duration) / dur * inner.w)
                    y  = dots.bottom - 2 - int((n.pitch - self._pitch_lo) / span * (dots.h - 4))
                    pygame.draw.line(self.screen, color, (x0, y), (max(x0 + 1, x1), y), 2)
            self._songmap_surf, self.screen = self.screen, screen

        self.screen.blit(self._songmap_surf, (inner.x, inner.y))
        px = inner.x + int(min(t / dur, 1.0) * inner.w)
        pygame.draw.line(self.screen, WHITE, (px, inner.y - 2), (px, inner.y + map_h), 2)

        # section list, current one highlighted
        list_y = inner.y + map_h + 8
        lh     = self.fs.get_height() + 2
        for i, s in enumerate(sc.sections):
            y = list_y + i * lh
            if y + lh > inner.bottom:
                break
            current = s.start <= t < s.end
            m0, s0  = divmod(int(s.start), 60)
            self._txt(f"{'>' if current else ' '} {m0}:{s0:02d}  {s.label:<14} {s.duration:5.1f}s",
                      (inner.x + 4, y), self._section_color(s.label) if current else GREY)

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
        self._draw_texture(features, t)
        if self.rhythm is not None:
            self._draw_rhythm_grid(t)
            self._draw_rhythm_notes(t)
            self._draw_rhythm_lanes(t)
        if self.score is not None:
            self._draw_harmony(features, t)
            self._draw_timeline(t)
            self._draw_songmap(t)


# ── entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ARC Debug Dashboard v2")
    parser.add_argument("--file", "-f", required=True,
                        help="Audio file to analyse")
    parser.add_argument("--fps",    type=int, default=60)
    parser.add_argument("--mode",   default="demucs",
                        choices=["vocals", "demucs", "roformer"])
    parser.add_argument("--midi", default=None,
                        help="MIDI file: gives the bar/beat grid, the notes and the automation lanes")
    parser.add_argument("--midi-offset", type=float, default=0.0,
                        help="Seconds to move the MIDI later so it lines up with the audio")
    parser.add_argument("--bpm", type=float, default=None,
                        help="Fixed-tempo grid when there is no MIDI (bar 1 at t=0, 4/4)")
    parser.add_argument("--stem", action="append", default=[], metavar="NAME=FILE",
                        help="Extra original stem to show beside the AI stems (repeatable)")
    parser.add_argument("--score", default=None,
                        help="SheetSage2 output folder (default: 'sheetsage' next to the audio)")
    parser.add_argument("--no-stems", action="store_true",
                        help="Skip AI stem separation (no GPU work; stems panel stays empty)")
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
    prebuilt = {}
    for spec in args.stem:
        name, sep, path = spec.partition("=")
        if not sep or not Path(path).exists():
            print(f"[ERROR] --stem expects NAME=FILE with an existing file, got: {spec}")
            sys.exit(1)
        prebuilt[name] = path

    extractor = AudioFeatureExtractor(str(file_path), fps=args.fps,
                                      separation_mode=args.mode,
                                      skip_separation=args.no_stems,
                                      prebuilt_stems=prebuilt)

    score_dir = Path(args.score) if args.score else file_path.parent / "sheetsage"
    score = None
    if has_score(score_dir):
        score = read_score(score_dir)
        print(f"Score: {score_dir}  ({len(score.chords)} chords, {len(score.sections)} sections, "
              f"{len(score.melody_vocal) + len(score.melody_instrumental)} melody notes)")
    elif args.score:
        print(f"[ERROR] No SheetSage2 output in: {score_dir}")
        sys.exit(1)
    # The grid comes from the most trustworthy source available. No beat tracker
    # here on purpose: a debug view showing a guessed grid misleads more than
    # it helps — declare the tempo with --bpm instead.
    rhythm = None
    if args.midi:
        grid, notes = read_midi(args.midi, fps=args.fps)
        shift_in_time(grid, notes, args.midi_offset)
        automation = MidiAutomationReader(args.midi, fps=args.fps, duration=extractor.duration)
        rhythm = RhythmInput(grid, "MIDI", notes, automation, args.midi_offset)
        print(f"MIDI: {args.midi}  ({len(notes)} notes, {grid.bpm:.2f} BPM, offset {args.midi_offset:+.3f}s)")
    elif score is not None and score.grid(args.fps) is not None:
        rhythm = RhythmInput(score.grid(args.fps), "score (SheetSage2)")
    elif args.bpm:
        rhythm = RhythmInput(RhythmGrid(bpm=args.bpm, fps=args.fps), "--bpm")
    dashboard = Dashboard4K(screen, extractor, score, rhythm)

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
