#!/usr/bin/env python3
"""ARC Studio — GUI for configuring, previewing, and rendering ARC projects.

Usage:
    python arc_studio.py
"""

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import dearpygui.dearpygui as dpg
import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
PREV_W, PREV_H   = 596, 336     # preview pane (16:9)
TIMELINE_W       = 1060         # drawlist pixel width
TRACK_H          = 52           # pixels per waveform track
BAR_H            = 22           # bar-number header height
EVENT_H          = 16           # trigger-events lane height
WIN_W, WIN_H     = 1110, 1020

EVENT_COLORS = [
    (255, 160, 80),    # 1st trigger (alphabetical) — orange
    (210, 100, 255),   # 2nd — purple
    (255, 230, 80),    # 3rd — yellow
    (100, 210, 255),   # 4th — blue
]

# render mode -> generator script (clips mode is handled separately)
_MODE_SCRIPTS = {
    "particles":    "generators/particle_generator.py",
    "particles_v2": "generators/particle_generator_new.py",
    "particles_v3": "generators/particle_generator_v3.py",
    "geometry":     "generators/animation_generator.py",
}

TRACK_COLORS = [
    (100, 210, 255),   # mix — blue
    (100, 255, 150),   # bass / custom stem 1 — green
    (255, 160, 80),    # drums / custom stem 2 — orange
    (210, 100, 255),   # solo / custom stem 3 — purple
    (255, 230, 80),    # other — yellow
]

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class _State:
    audio_path: Optional[str] = None
    midi_path:  Optional[str] = None
    scene_path: str           = str(Path(__file__).parent / "scenes" / "default.yaml")

    extractor       = None
    grid            = None
    pipeline        = None
    midi_automation = None
    midi_notes: list = []    # MidiNote list from read_midi (clips mode triggers)
    player          = None   # AudioPlayer

    # stems panel: list of {"name": str, "path": str}
    stem_rows: list = []
    skip_separation: bool = False
    _stem_row_counter: int = 0  # monotonic id for unique tags

    loaded:   bool = False
    loading:   bool = False
    rendering: bool = False

    # waveform sources: name → ("audio", raw samples) | ("energy", env array);
    # per-view envelopes are computed at draw time (zoom-aware)
    _wave_sources: dict = {}

    # timeline zoom window
    view_start: float = 0.0
    view_dur: Optional[float] = None   # None = whole song

    # trigger events from the scene (clips mode), for timeline display
    trigger_events:  list = []   # ClipComposer TriggerEvent list
    handover_points: list = []   # (trigger, until_trigger, time) from `until:`

    # scene trigger editor (authors the scene YAML's video: section)
    trig_rows: list = []
    _trig_counter: int = 0
    scene_clip_per_bar: bool = True
    scene_clip_order: str = "shuffle"
    _scene_extra_video: dict = {}   # video-level keys the editor doesn't manage
    # layers editor: one dict per scene layer (see _layers_to_rows)
    layer_rows: list = []
    _layer_counter: int = 0

    # preview playback
    preview_frames:   list  = []
    preview_idx:      int   = 0
    preview_playing:  bool  = False
    preview_start_t:  float = 0.0   # song time of the preview's first frame
    _preview_audio:   bool  = False # preview started the audio player
    _preview_last_t:  float = 0.0   # wall-clock time of last frame advance

    # render settings
    bars:   int = 8
    fps:    int = 24
    width:  int = 854
    height: int = 480
    mode:   str = "particles"
    preview_bars: int = 1   # bars rendered by the Preview button (RAM-bound)
    codec: str = "x264"     # clips-mode encoder; nvenc = GPU, ideal p/ 4K
    start_time: float = 0.0     # where the render starts in the song (0 = first downbeat)
    render_stems: bool = False  # separate stems for the layers' features (--stems; slow)
    clip_fit: str = "contain"   # how clips fill the frame: contain (bars) or cover (crop)
    last_render: str = ""       # the last full render's MP4 (the repaint's default input)
    last_render_start: float = 0.0

    # repaint: the render through a ComfyUI workflow (see repaint.py)
    repaint_video: str = ""
    repaint_start: float = 0.0          # where that video starts in the song
    repaint_workflow: str = "reference/workflow_api_rundiff.json"
    repaint_scene: str = "(none)"       # a scene with a repaint: block (denoise/kick/prompts)
    repaint_prompt: str = ""            # one prompt for every frame (overrides the scene's)
    repaint_fps: float = 12.0
    repaint_seconds: float = 0.0        # 0 = the whole video
    _repaint_proc = None

    # clips mode settings
    clips_dir: Optional[str] = None
    clip_order: str  = "(scene)"   # "(scene)" = keep the YAML's clip_order
    full_song:  bool = False       # render --bars 0
    cache_size: int  = 8
    grav_enable: bool  = False     # override the scene's gravity blocks
    grav_peak:   float = 3.0
    grav_floor:  float = 0.3
    grav_radius: float = 0.45
    grav_curve:  float = 2.0
    clips_seed: Optional[int] = None   # persisted; makes lane == preview == render
    clip_overrides: dict = {}          # {bar_index: clip name} — manual pins
    _deck_thumbs: list = []
    _deck_ready: bool = False

    # resolved output arrangement (dry-run of the composer; no decoding)
    resolved_segments: list = []
    resolved_times = None    # np arrays: speed curve samples
    resolved_speeds = None

    # UI/audio sync calibration: added to the displayed time while playing.
    # Positive values push the UI forward (use when the audio sounds ahead).
    sync_offset_ms: int = 0
    # the MIDI's place against the audio (s, added to every MIDI time) and meter
    # changes ("22:6/4,27:5/4"), the same as clip_generator's --midi-offset / --meter
    midi_offset: float = 0.0
    meter: str = ""
    _last_controls_t: float = 0.0   # throttle for the controls tables

    # internal flag: stop scrubber → seek feedback loop
    _syncing: bool = False

S = _State()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _surface_to_dpg(surf: pygame.Surface) -> np.ndarray:
    arr  = pygame.surfarray.array3d(surf).transpose(1, 0, 2).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    rgba = np.ones((H, W, 4), dtype=np.float32)
    rgba[:, :, :3] = arr
    return rgba.flatten()


def _blank_frame() -> np.ndarray:
    return np.full(PREV_W * PREV_H * 4, 0.08, dtype=np.float32)


def _np_to_dpg(arr: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> flat float32 RGBA for a dpg raw texture."""
    rgb = arr.astype(np.float32) / 255.0
    H, W = rgb.shape[:2]
    rgba = np.ones((H, W, 4), dtype=np.float32)
    rgba[:, :, :3] = rgb
    return rgba.flatten()


def _set_status(msg: str):
    try:
        dpg.set_value("status_text", msg)
    except Exception:
        pass


def _log(msg: str):
    try:
        prev  = dpg.get_value("log_box")
        lines = (prev + "\n" + msg).split("\n")
        dpg.set_value("log_box", "\n".join(lines[-30:]))
    except Exception:
        print(msg)


def _refresh_controls(t: Optional[float] = None):
    if not S.loaded or S.pipeline is None:
        return
    if t is None:
        t = dpg.get_value("scrubber") or 0.0
    fi = int(t * S.fps)
    sl = S.pipeline.query(t, frame_idx=fi)
    if sl is None:
        return

    dpg.set_value("bar_beat_text",
                  f"Bar {(sl.bar or 0) + 1}   Beat {(sl.beat or 0) % 4 + 1}"
                  f"   phase {sl.bar_phase:.2f}")

    audio_rows, midi_rows = [], []
    for k, v in sorted(sl.controls.items()):
        src = S.pipeline._mappings.get(k, {}).get("source", "")
        (midi_rows if src.startswith("midi.") else audio_rows).append((k, v))

    _fill_table("audio_table", audio_rows)
    _fill_table("midi_table",  midi_rows)


def _fill_table(tag: str, rows: list):
    if not dpg.does_item_exist(tag):
        return
    dpg.delete_item(tag, children_only=True)
    for name, val in rows:
        with dpg.table_row(parent=tag):
            dpg.add_text(name)
            bar = "█" * max(0, min(10, int(val * 10)))
            dpg.add_text(f"{bar:<10}  {val:.3f}")

# ---------------------------------------------------------------------------
# Timeline drawing
# ---------------------------------------------------------------------------

OUT_H, SPEED_H = 30, 26   # output-arrangement and speed-curve lane heights
MIDI_ROW_H = 11           # one mini-row per MIDI track (note ticks + name)


def _midi_lane_tracks() -> list:
    """Track names in the MIDI lane, in the file's order."""
    if not S.midi_notes:
        return []
    seen = {}
    for n in S.midi_notes:
        if n.track and n.track not in seen:
            seen[n.track] = n.track_index
    return sorted(seen, key=lambda t: seen[t])


def _midi_lane_h() -> int:
    return len(_midi_lane_tracks()) * MIDI_ROW_H


def _timeline_height() -> int:
    h = BAR_H + max(1, len(S._wave_sources)) * TRACK_H + _midi_lane_h()
    if S.trigger_events:
        h += EVENT_H
    if S.resolved_segments:
        h += OUT_H + SPEED_H
    return h


def _clip_color(idx: int) -> tuple:
    """Stable, distinct color per clip index (golden-ratio hue walk)."""
    h = (idx * 0.61803) % 1.0
    i = int(h * 6); f = h * 6 - i
    v, s = 0.82, 0.55
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return int(r * 255), int(g * 255), int(b * 255)


def _full_duration() -> float:
    return float(S.extractor.duration) if S.extractor else 0.0


def _view_window() -> tuple:
    """Current zoom window as (start, length) in seconds, clamped."""
    dur = _full_duration()
    if dur <= 0:
        return 0.0, 1.0
    vd = min(S.view_dur, dur) if S.view_dur else dur
    v0 = max(0.0, min(S.view_start, dur - vd))
    return v0, vd


def _t_to_x(t: float, v0: float, vd: float) -> int:
    return int((t - v0) / vd * TIMELINE_W)


def _window_envelope(kind: str, data, v0: float, vd: float) -> np.ndarray:
    """TIMELINE_W-point envelope of the zoom window, from the raw source."""
    from core.audio_player import rms_envelope
    if kind == "audio":
        sr  = S.extractor.sample_rate
        seg = data[int(v0 * sr): int((v0 + vd) * sr)]
        if len(seg) < 2:
            return np.zeros(TIMELINE_W, dtype=np.float32)
        return rms_envelope(seg, TIMELINE_W)
    # "energy": per-feature-frame array spanning the whole song
    n   = len(data)
    dur = _full_duration()
    xs  = np.linspace(v0 / dur * (n - 1), (v0 + vd) / dur * (n - 1), TIMELINE_W)
    return np.interp(xs, np.arange(n), data).astype(np.float32)


def _draw_timeline_static():
    """Draw waveforms + bar/beat markers for the current zoom window."""
    if not dpg.does_item_exist("timeline_canvas"):
        return
    dpg.delete_item("timeline_canvas", children_only=True)

    v0, vd = _view_window()
    v1 = v0 + vd
    H  = _timeline_height()
    dpg.configure_item("timeline_canvas", height=H)

    # Beat markers (subtle; skipped when denser than ~6px)
    beats = S.grid.beats if S.grid.beats is not None else []
    if len(beats) and vd / max(1e-6, S.grid.beat_duration) < TIMELINE_W / 6:
        for bt in beats:
            if v0 <= bt <= v1:
                x = _t_to_x(bt, v0, vd)
                dpg.draw_line([x, BAR_H], [x, H],
                              color=(55, 55, 75, 160), thickness=1,
                              parent="timeline_canvas")

    # Bar markers + numbers
    for i, dt in enumerate(S.grid.downbeats if S.grid.downbeats is not None else []):
        if not (v0 <= dt <= v1):
            continue
        x = _t_to_x(dt, v0, vd)
        dpg.draw_line([x, 0], [x, H],
                      color=(90, 120, 200, 200), thickness=1,
                      parent="timeline_canvas")
        dpg.draw_text([x + 3, 3], str(i + 1), size=11,
                      color=(180, 180, 180, 220),
                      parent="timeline_canvas")

    # Waveform envelopes (recomputed for the window — real zoom resolution)
    for track_i, (name, (kind, data)) in enumerate(S._wave_sources.items()):
        env   = _window_envelope(kind, data, v0, vd)
        color = TRACK_COLORS[track_i % len(TRACK_COLORS)]
        fill  = (*color, 70)
        line  = (*color, 200)
        yc    = BAR_H + track_i * TRACK_H + TRACK_H // 2
        half  = TRACK_H * 0.42

        top_pts = [[i, yc - int(env[i] * half)] for i in range(TIMELINE_W)]
        bot_pts = [[i, yc + int(env[i] * half)] for i in range(TIMELINE_W - 1, -1, -1)]
        dpg.draw_polygon(top_pts + bot_pts,
                         fill=fill, color=(0, 0, 0, 0),
                         parent="timeline_canvas")
        dpg.draw_polyline(top_pts, color=line, thickness=1,
                          parent="timeline_canvas")

        # Track label
        dpg.draw_text([4, BAR_H + track_i * TRACK_H + 3], name,
                      size=11, color=(220, 220, 220, 180),
                      parent="timeline_canvas")

    # View info (zoom window)
    if S.view_dur:
        dpg.draw_text([TIMELINE_W - 190, 3],
                      f"zoom: {v0:.1f}s – {v1:.1f}s", size=11,
                      color=(255, 200, 120, 220), parent="timeline_canvas")

    # MIDI lane: one mini-row per track — where each instrument plays (what track: selects)
    tracks = _midi_lane_tracks()
    if tracks:
        ym = BAR_H + len(S._wave_sources) * TRACK_H
        cols: dict = {t: set() for t in tracks}
        for n in S.midi_notes:
            if n.track in cols and v0 <= n.time <= v1:
                cols[n.track].add(_t_to_x(n.time, v0, vd))
        for ti, name in enumerate(tracks):
            y = ym + ti * MIDI_ROW_H
            color = TRACK_COLORS[(ti + 1) % len(TRACK_COLORS)]
            dpg.draw_rectangle([0, y], [TIMELINE_W, y + MIDI_ROW_H],
                               fill=(24, 26, 36, 150) if ti % 2 else (30, 32, 44, 150),
                               color=(0, 0, 0, 0), parent="timeline_canvas")
            for x in cols[name]:
                dpg.draw_line([x, y + 2], [x, y + MIDI_ROW_H - 2], color=(*color, 210),
                              thickness=1, parent="timeline_canvas")
            dpg.draw_text([4, y], name, size=10, color=(220, 220, 220, 170),
                          parent="timeline_canvas")

    # Trigger events lane + hand-over markers (clips scenes)
    if S.trigger_events:
        y0 = BAR_H + len(S._wave_sources) * TRACK_H + _midi_lane_h()
        dpg.draw_rectangle([0, y0], [TIMELINE_W, y0 + EVENT_H],
                           fill=(30, 30, 42, 140), color=(0, 0, 0, 0),
                           parent="timeline_canvas")
        names = sorted({e.name for e in S.trigger_events})
        for e in S.trigger_events:
            if not (v0 <= e.time <= v1):
                continue
            x = _t_to_x(e.time, v0, vd)
            color = EVENT_COLORS[names.index(e.name) % len(EVENT_COLORS)]
            if "next_clip" in e.actions or "random_clip" in e.actions:
                # clip switch: full-height tick
                dpg.draw_line([x, y0 + 2], [x, y0 + EVENT_H - 2],
                              color=(*color, 235), thickness=2,
                              parent="timeline_canvas")
            else:
                # other actions (reverse, restart): half-height tick
                dpg.draw_line([x, y0 + EVENT_H // 2], [x, y0 + EVENT_H - 2],
                              color=(*color, 180), thickness=1,
                              parent="timeline_canvas")
        dpg.draw_text([4, y0 + 2], "TRIGGERS", size=11,
                      color=(220, 220, 220, 190), parent="timeline_canvas")
        for name, until, t in S.handover_points:
            if not (v0 <= t <= v1):
                continue
            x = _t_to_x(t, v0, vd)
            dpg.draw_line([x, 0], [x, H], color=(255, 230, 80, 235),
                          thickness=2, parent="timeline_canvas")
            dpg.draw_text([x + 5, BAR_H + 2], f"{until} assume", size=12,
                          color=(255, 230, 80, 255), parent="timeline_canvas")

    # Output lane: resolved clip blocks + direction, then the speed curve
    if S.resolved_segments:
        y0 = (BAR_H + len(S._wave_sources) * TRACK_H + _midi_lane_h()
              + (EVENT_H if S.trigger_events else 0))
        for seg in S.resolved_segments:
            if seg.t1 < v0 or seg.t0 > v1:
                continue
            xa, xb = max(0, _t_to_x(seg.t0, v0, vd)), min(TIMELINE_W, _t_to_x(seg.t1, v0, vd))
            if xb - xa < 1:
                continue
            r, g, b = _clip_color(seg.clip_idx)
            if seg.direction < 0:
                r, g, b = int(r * 0.55), int(g * 0.55), int(b * 0.55)
            dpg.draw_rectangle([xa, y0 + 1], [xb, y0 + OUT_H - 1],
                               fill=(r, g, b, 235), color=(13, 15, 20, 255),
                               parent="timeline_canvas")
            arrow = "◀" if seg.direction < 0 else "▶"
            if xb - xa > 14:
                dpg.draw_text([xa + 2, y0 + OUT_H - 14], arrow, size=10,
                              color=(13, 15, 20, 230), parent="timeline_canvas")
            max_chars = int((xb - xa - 16) / 6)
            if max_chars >= 4:
                dpg.draw_text([xa + 13, y0 + 3], seg.clip_name[:max_chars], size=10,
                              color=(13, 15, 20, 255), parent="timeline_canvas")
        # pinned bars: amber outline
        for seg in S.resolved_segments:
            if seg.t1 < v0 or seg.t0 > v1:
                continue
            if _bar_at(seg.t0 + 1e-3) in S.clip_overrides:
                xa = max(0, _t_to_x(seg.t0, v0, vd))
                xb = min(TIMELINE_W, _t_to_x(seg.t1, v0, vd))
                dpg.draw_rectangle([xa, y0 + 1], [xb, y0 + OUT_H - 1],
                                   color=(255, 180, 84, 255), thickness=2,
                                   parent="timeline_canvas")
        dpg.draw_text([4, y0 + OUT_H - 12], "SAIDA", size=10,
                      color=(255, 255, 255, 130), parent="timeline_canvas")

        # speed curve (gravity made visible)
        ys = y0 + OUT_H
        if S.resolved_times is not None and len(S.resolved_times):
            mask = (S.resolved_times >= v0) & (S.resolved_times <= v1)
            ts, vs = S.resolved_times[mask], S.resolved_speeds[mask]
            if len(ts) > 1:
                vmax = max(2.0, float(vs.max()))
                step = max(1, len(ts) // TIMELINE_W)
                pts = [[_t_to_x(float(ts[i]), v0, vd),
                        ys + SPEED_H - 3 - (float(vs[i]) / vmax) * (SPEED_H - 6)]
                       for i in range(0, len(ts), step)]
                y1x = ys + SPEED_H - 3 - (1.0 / vmax) * (SPEED_H - 6)
                dpg.draw_line([0, y1x], [TIMELINE_W, y1x],
                              color=(70, 76, 95, 160), thickness=1,
                              parent="timeline_canvas")
                dpg.draw_polyline(pts, color=(255, 180, 84, 235), thickness=1,
                                  parent="timeline_canvas")
                dpg.draw_text([4, ys + 1], f"speed (max {vmax:.1f}x)", size=10,
                              color=(255, 180, 84, 160), parent="timeline_canvas")

    # Playhead (created last so it renders on top; tag allows configure later)
    dpg.draw_line([0, 0], [0, H],
                  color=(255, 80, 80, 255), thickness=2,
                  tag="playhead_line", parent="timeline_canvas")


def _update_playhead(t: float):
    if not dpg.does_item_exist("playhead_line") or not S.extractor:
        return
    v0, vd = _view_window()
    x = int(np.clip((t - v0) / vd, 0.0, 1.0) * TIMELINE_W)
    H = _timeline_height()
    dpg.configure_item("playhead_line", p1=[x, 0], p2=[x, H])

# ---------------------------------------------------------------------------
# Stems panel (dynamic rows)
# ---------------------------------------------------------------------------

def _add_stem_row(name: str = "", path: str = "") -> None:
    rid  = S._stem_row_counter
    S._stem_row_counter += 1
    tag  = f"stem_row_{rid}"
    row  = {"name": name or f"stem{rid}", "path": path, "_rid": rid, "_tag": tag}
    S.stem_rows.append(row)

    with dpg.group(horizontal=True, tag=tag, parent="stems_panel"):
        dpg.add_input_text(
            default_value=row["name"], width=80,
            callback=lambda s, v, u: u.update({"name": v}),
            user_data=row,
        )
        dpg.add_text(Path(path).name if path else "(no file)",
                     tag=f"stem_path_label_{rid}")
        dpg.add_button(label="Browse",
                       callback=lambda s, a, u: _browse_stem(*u),
                       user_data=(row, rid))
        dpg.add_button(label="x", width=26,
                       callback=lambda s, a, u: _remove_stem_row(u[0], u[1]),
                       user_data=(tag, row))


_stem_dialog_target: dict = {}   # row and rid waiting for the shared dialog

def _browse_stem(row: dict, rid: int):
    _stem_dialog_target["row"] = row
    _stem_dialog_target["rid"] = rid
    dpg.show_item("dlg_stem_shared")


def _pick_stem_shared(s, a):
    p = _first_selection(a)
    if not p:
        return
    row = _stem_dialog_target.get("row")
    rid = _stem_dialog_target.get("rid")
    if row is not None:
        row["path"] = p
    if rid is not None and dpg.does_item_exist(f"stem_path_label_{rid}"):
        dpg.set_value(f"stem_path_label_{rid}", Path(p).name)


def _remove_stem_row(tag: str, row: dict):
    if row in S.stem_rows:
        S.stem_rows.remove(row)
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)


def _clear_stem_rows():
    for row in list(S.stem_rows):
        t = row.get("_tag", "")
        if t and dpg.does_item_exist(t):
            dpg.delete_item(t)
    S.stem_rows.clear()


# ---------------------------------------------------------------------------
# Scene trigger editor (UI <-> scene YAML "video:" section)
# ---------------------------------------------------------------------------

_TRIG_KNOWN_KEYS = {"notes", "audio", "actions", "until", "min_velocity",
                    "gravity", "threshold", "min_gap",
                    "track", "score", "events", "snap", "voice"}
TRIG_SOURCES = ["notes", "track", "audio", "score"]
SCORE_EVENTS_UI = ["chord_changes", "section_changes", "melody"]
SCORE_SNAPS_UI = ["auto", "bar", "beat", "off"]


def _trig_source(spec: dict) -> str:
    """Where a trigger's hits come from (the same precedence the composer uses)."""
    if "audio" in spec:
        return "audio"
    if "score" in spec:
        return "score"
    if "track" in spec:
        return "track"
    return "notes"


def _parse_tracks(text: str):
    """"kick, kick-2" -> ["kick", "kick-2"]; one name stays a plain string; digits are indexes."""
    items = [t.strip() for t in str(text).split(",") if t.strip()]
    items = [int(t) if t.isdigit() else t for t in items]
    return items[0] if len(items) == 1 else items


def _video_cfg_to_rows(video_cfg: dict) -> list:
    """Scene video config -> editor row dicts (unknown keys preserved)."""
    rows = []
    for name, spec in (video_cfg.get("triggers") or {}).items():
        g = spec.get("gravity") or {}
        track = spec.get("track", "")
        rows.append({
            "name": name,
            "source": _trig_source(spec),
            "notes": ",".join(str(n) for n in spec.get("notes", [])),
            "track": ", ".join(str(t) for t in track) if isinstance(track, (list, tuple))
                     else str(track),
            "score": str(spec.get("score", "")),
            "events": spec.get("events", "chord_changes"),
            "snap": spec.get("snap", "auto"),
            "voice": str(spec.get("voice", "")),
            "audio": spec.get("audio", ""),
            "threshold": float(spec.get("threshold", 0.3)),
            "min_gap": float(spec.get("min_gap", 0.05)),
            "actions": ",".join(spec.get("actions", [])),
            "until": spec.get("until", "") or "",
            "min_vel": int(spec.get("min_velocity", 0)),
            "grav_on": bool(g),
            "peak": float(g.get("peak", 3.0)),
            "floor": float(g.get("floor", 0.3)),
            "radius": float(g.get("radius", 0.45)),
            "curve": float(g.get("curve", 2.0)),
            "_extra": {k: v for k, v in spec.items() if k not in _TRIG_KNOWN_KEYS},
        })
    return rows


def _rows_to_video_cfg(rows: list, clip_per_bar: bool, clip_order: str,
                       extra: Optional[dict] = None) -> dict:
    """Editor row dicts -> scene video config (round-trip safe)."""
    video = dict(extra or {})
    video["clip_per_bar"] = bool(clip_per_bar)
    video["clip_order"]   = clip_order
    triggers = {}
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        spec = dict(r.get("_extra") or {})
        source = r.get("source", "notes")
        notes = [
            int(x) for x in str(r.get("notes", "")).replace(";", ",").split(",")
            if x.strip().lstrip("-").isdigit()
        ]
        if source == "audio" and r.get("audio"):
            spec["audio"]     = r["audio"]
            spec["threshold"] = float(r.get("threshold", 0.3))
            spec["min_gap"]   = float(r.get("min_gap", 0.05))
        elif source == "track" and str(r.get("track", "")).strip():
            spec["track"] = _parse_tracks(r["track"])
            if notes:                              # optional: only these pitches of the track
                spec["notes"] = notes
        elif source == "score" and str(r.get("score", "")).strip():
            spec["score"] = str(r["score"]).strip()
            events = r.get("events", "chord_changes")
            if events != "chord_changes":          # the composer's default
                spec["events"] = events
            if r.get("snap", "auto") != "auto":
                spec["snap"] = r["snap"]
            if events == "melody" and str(r.get("voice", "")).strip():
                spec["voice"] = str(r["voice"]).strip()
            if notes:                              # optional: e.g. only changes into G#
                spec["notes"] = notes
        else:
            spec["notes"] = notes
        spec["actions"] = [a.strip() for a in str(r.get("actions", "")).split(",")
                           if a.strip()]
        if str(r.get("until", "")).strip():
            spec["until"] = str(r["until"]).strip()
        if int(r.get("min_vel", 0)) > 0:
            spec["min_velocity"] = int(r["min_vel"])
        if r.get("grav_on"):
            spec["gravity"] = {"peak": float(r["peak"]), "floor": float(r["floor"]),
                               "radius": float(r["radius"]), "curve": float(r["curve"])}
        triggers[name] = spec
    video["triggers"] = triggers
    return video


# canonical display/apply order; engine actions not listed here are appended
_ACTION_ORDER = ["next_clip", "random_clip", "restart", "reverse"]


def _action_options() -> list:
    from core.video.composer import ACTIONS
    return _ACTION_ORDER + sorted(a for a in ACTIONS if a not in _ACTION_ORDER)


def _toggle_action(row: dict, action: str, on: bool) -> None:
    """Checkbox callback: rebuild the row's actions string in canonical order."""
    acts = {a.strip() for a in str(row.get("actions", "")).split(",") if a.strip()}
    if on:
        acts.add(action)
    else:
        acts.discard(action)
    row["actions"] = ",".join(a for a in _action_options() if a in acts)


_trig_audio_target: dict = {}


def _pick_trig_audio(s, a):
    p = _first_selection(a)
    row = _trig_audio_target.get("row")
    if p and row is not None:
        row["audio"] = p
        rid = row.get("_rid")
        if rid is not None and dpg.does_item_exist(f"trig_audio_label_{rid}"):
            dpg.set_value(f"trig_audio_label_{rid}", Path(p).name)


_track_names_cache: dict = {}


def _midi_track_names() -> list:
    """The MIDI's track names (what ``track:`` triggers select), for the editor's menu."""
    if S.midi_notes:
        return sorted({n.track for n in S.midi_notes if n.track})
    if not S.midi_path or not Path(S.midi_path).is_file():
        return []
    if S.midi_path not in _track_names_cache:
        try:
            from core.rhythm.midi_reader import read_midi
            _, notes = read_midi(S.midi_path)
            _track_names_cache[S.midi_path] = sorted({n.track for n in notes if n.track})
        except Exception:
            _track_names_cache[S.midi_path] = []
    return _track_names_cache[S.midi_path]


def _default_score_path() -> str:
    """input/<song>/sheetsage next to the audio, when SheetSage has run for it."""
    if S.audio_path:
        cand = Path(S.audio_path).parent / "sheetsage"
        if cand.is_dir():
            return cand.as_posix()
    return ""


def _add_trigger_row(cfg: Optional[dict] = None) -> None:
    rid = S._trig_counter
    S._trig_counter += 1
    tag = f"trig_row_{rid}"
    row = {"name": f"trig{rid}", "source": "notes", "notes": "36", "audio": "",
           "track": "", "score": _default_score_path(), "events": "chord_changes",
           "snap": "auto", "voice": "",
           "threshold": 0.3, "min_gap": 0.05, "actions": "next_clip", "until": "",
           "min_vel": 0, "grav_on": False, "peak": 3.0, "floor": 0.3,
           "radius": 0.45, "curve": 2.0, "_extra": {}}
    if cfg:
        row.update(cfg)
    row["_rid"] = rid
    row["_tag"] = tag
    S.trig_rows.append(row)

    def _set(key):
        return lambda s, v, u=row: u.update({key: v})

    def _on_source(s, v, u=row):
        u["source"] = v
        for src in TRIG_SOURCES:
            if dpg.does_item_exist(f"trig_{src}_{rid}"):
                dpg.configure_item(f"trig_{src}_{rid}", show=(src == v))

    def _add_track(s, v, u=row):
        names = [t.strip() for t in u["track"].split(",") if t.strip()]
        if v and v not in names:
            names.append(v)
        u["track"] = ", ".join(names)
        dpg.set_value(f"trig_track_text_{rid}", u["track"])

    with dpg.group(tag=tag, parent="triggers_panel"):
        with dpg.group(horizontal=True):
            dpg.add_input_text(default_value=row["name"], width=70,
                               callback=_set("name"))
            dpg.add_combo(TRIG_SOURCES, default_value=row["source"], width=70,
                          callback=_on_source)
            dpg.add_text("notes:")
            dpg.add_input_text(default_value=row["notes"], width=60, hint="36,38",
                               callback=_set("notes"))
            # per-source fields; only the chosen source's group shows
            with dpg.group(horizontal=True, tag=f"trig_notes_{rid}",
                           show=row["source"] == "notes"):
                pass
            with dpg.group(horizontal=True, tag=f"trig_track_{rid}",
                           show=row["source"] == "track"):
                dpg.add_text("track:")
                dpg.add_input_text(default_value=row["track"], width=130, hint="kick, kick-2",
                                   tag=f"trig_track_text_{rid}", callback=_set("track"))
                dpg.add_combo(_midi_track_names(), width=90, default_value="",
                              callback=_add_track)
            with dpg.group(horizontal=True, tag=f"trig_audio_{rid}",
                           show=row["source"] == "audio"):
                dpg.add_button(label="Audio…", width=60,
                               callback=lambda s, a, u=row: (
                                   _trig_audio_target.update({"row": u}),
                                   dpg.show_item("dlg_trig_audio")))
                dpg.add_text(Path(row["audio"]).name if row["audio"] else "(none)",
                             tag=f"trig_audio_label_{rid}")
            with dpg.group(horizontal=True, tag=f"trig_score_{rid}",
                           show=row["source"] == "score"):
                dpg.add_input_text(default_value=row["score"], width=150,
                                   hint="input/música/sheetsage", callback=_set("score"))
                dpg.add_combo(SCORE_EVENTS_UI, default_value=row["events"], width=110,
                              callback=_set("events"))
                dpg.add_text("snap")
                dpg.add_combo(SCORE_SNAPS_UI, default_value=row["snap"], width=55,
                              callback=_set("snap"))
                dpg.add_input_text(default_value=row["voice"], width=80,
                                   hint="voice (melody)", callback=_set("voice"))
            dpg.add_button(label="x", width=24,
                           callback=lambda s, a, u=(tag, row): _remove_trigger_row(*u))
        active = {a.strip() for a in str(row["actions"]).split(",") if a.strip()}
        with dpg.group(horizontal=True):
            dpg.add_text("   actions:")
            for act in _action_options():
                dpg.add_checkbox(label=act, default_value=act in active,
                                 callback=lambda s, v, u=(row, act):
                                     _toggle_action(u[0], u[1], v))
            dpg.add_text("until:")
            dpg.add_input_text(default_value=row["until"], width=60,
                               callback=_set("until"))
            dpg.add_text("min_vel:")
            dpg.add_input_int(default_value=row["min_vel"], width=70,
                              callback=_set("min_vel"))
        with dpg.group(horizontal=True):
            dpg.add_text("   ")
            dpg.add_checkbox(label="gravity", default_value=row["grav_on"],
                             callback=_set("grav_on"))
            for key, lbl in (("peak", "peak"), ("floor", "floor"),
                             ("radius", "radius"), ("curve", "curve")):
                dpg.add_text(lbl)
                dpg.add_input_float(default_value=row[key], width=52, step=0,
                                    format="%.2f", callback=_set(key))
        dpg.add_spacer(height=3)


def _remove_trigger_row(tag, row):
    if row in S.trig_rows:
        S.trig_rows.remove(row)
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)


def _clear_trigger_rows():
    for row in list(S.trig_rows):
        t = row.get("_tag", "")
        if t and dpg.does_item_exist(t):
            dpg.delete_item(t)
    S.trig_rows.clear()

# ---------------------------------------------------------------------------
# Scene layer editor (dynamic rows)
# ---------------------------------------------------------------------------

# the keys a layer row edits with its own widgets; everything else is the row's params YAML
_LAYER_KEYS = ("source", "enabled", "file", "blend", "opacity", "bars")
_BLEND_ORDER = ["normal", "add", "screen", "multiply"]
_NEW_LAYER_PARAMS = {"solid": "color: [0, 0, 0]\n"}


def _shader_choices() -> list:
    """The Shadertoy shaders in the project: single .glsl files and multipass folders."""
    root = Path("shaders/shadertoy")
    if not root.is_dir():
        return []
    out = [f.as_posix() for f in root.glob("*.glsl")]
    out += [d.as_posix() for d in root.iterdir() if d.is_dir() and (d / "image.glsl").is_file()]
    return sorted(out)


def _yaml_flow(value) -> str:
    import yaml
    return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()


def _layers_to_rows(video_cfg: dict) -> list:
    """Scene ``layers:`` -> editor row dicts (params YAML keeps what the row has no widget for)."""
    import yaml
    rows = []
    for spec in video_cfg.get("layers") or []:
        spec = spec or {}
        params = {k: v for k, v in spec.items() if k not in _LAYER_KEYS}
        op = spec.get("opacity", 1.0)
        rows.append({
            "source":  spec.get("source", "clips"),
            "enabled": spec.get("enabled", True) is not False,
            "file":    str(spec.get("file", "") or ""),
            "blend":   spec.get("blend", "normal"),
            "opacity": float(op) if isinstance(op, (int, float)) else 1.0,
            "bars":    _yaml_flow(spec["bars"]) if "bars" in spec else "",
            "params":  yaml.safe_dump(params, allow_unicode=True, sort_keys=False)
                       if params else "",
        })
    return rows


def _rows_to_layers(rows: list) -> list:
    """Editor rows -> scene ``layers:``; raises ValueError naming the row whose YAML is wrong."""
    import yaml
    layers = []
    for i, r in enumerate(rows, 1):
        spec = {"source": r["source"]}
        if not r.get("enabled", True):
            spec["enabled"] = False
        if r.get("file"):
            spec["file"] = r["file"]
        if r.get("blend", "normal") != "normal":
            spec["blend"] = r["blend"]
        if abs(float(r.get("opacity", 1.0)) - 1.0) > 1e-6:
            spec["opacity"] = round(float(r["opacity"]), 3)
        bars = str(r.get("bars", "")).strip()
        if bars:
            try:
                spec["bars"] = yaml.safe_load(bars)
            except yaml.YAMLError as exc:
                raise ValueError(f"layer {i} ({r['source']}): bars is not valid YAML: {exc}")
        text = str(r.get("params", "")).strip()
        if text:
            try:
                params = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise ValueError(f"layer {i} ({r['source']}): params are not valid YAML: {exc}")
            if not isinstance(params, dict):
                raise ValueError(f"layer {i} ({r['source']}): params must be 'key: value' lines")
            spec.update({k: v for k, v in params.items() if k not in _LAYER_KEYS})
        layers.append(spec)
    return layers


def _add_layer_row(cfg: Optional[dict] = None) -> None:
    row = {"source": "solid", "enabled": True, "file": "", "blend": "normal",
           "opacity": 1.0, "bars": "", "params": ""}
    if cfg:
        row.update(cfg)
    row["_rid"] = S._layer_counter
    S._layer_counter += 1
    S.layer_rows.append(row)


def _draw_layer_rows() -> None:
    """(Re)build the layer panel from S.layer_rows — the order on screen is the stack order."""
    if not dpg.does_item_exist("layers_panel"):
        return
    dpg.delete_item("layers_panel", children_only=True)
    shaders = _shader_choices()
    n = len(S.layer_rows)
    for i, row in enumerate(S.layer_rows):
        def _set(key, row=row):
            return lambda s, v: row.update({key: v})
        with dpg.group(parent="layers_panel"):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(default_value=row["enabled"], callback=_set("enabled"))
                dpg.add_text(f"{i + 1}. {row['source']}",
                             color=(230, 210, 140) if row["enabled"] else (110, 110, 110))
                if i == 0:
                    dpg.add_text("(base)", color=(120, 120, 140))
                else:
                    dpg.add_combo(_BLEND_ORDER, default_value=row["blend"], width=80,
                                  callback=_set("blend"))
                dpg.add_text("opacity")
                dpg.add_slider_float(default_value=row["opacity"], min_value=0.0,
                                     max_value=1.0, width=80, format="%.2f",
                                     callback=_set("opacity"))
                dpg.add_text("bars")
                dpg.add_input_text(default_value=row["bars"], width=130,
                                   hint="[[21, 33]]", callback=_set("bars"))
                dpg.add_button(label="^", width=20, enabled=i > 0,
                               callback=lambda s, a, u=i: _move_layer(u, -1))
                dpg.add_button(label="v", width=20, enabled=i < n - 1,
                               callback=lambda s, a, u=i: _move_layer(u, +1))
                dpg.add_button(label="x", width=20,
                               callback=lambda s, a, u=i: _remove_layer(u))
            if row["source"] == "shadertoy":
                with dpg.group(horizontal=True):
                    dpg.add_text("     shader")
                    choices = shaders if row["file"] in shaders or not row["file"] \
                        else [row["file"]] + shaders
                    dpg.add_combo(choices, default_value=row["file"], width=360,
                                  callback=_set("file"))
            lines = max(2, min(14, str(row["params"]).count("\n") + 2))
            with dpg.tree_node(label="     params (YAML)", default_open=False):
                dpg.add_input_text(default_value=row["params"], multiline=True,
                                   width=560, height=lines * 17, tab_input=True,
                                   callback=_set("params"))
            dpg.add_spacer(height=2)


def _move_layer(i: int, step: int) -> None:
    j = i + step
    if 0 <= j < len(S.layer_rows):
        S.layer_rows[i], S.layer_rows[j] = S.layer_rows[j], S.layer_rows[i]
        _draw_layer_rows()


def _remove_layer(i: int) -> None:
    if 0 <= i < len(S.layer_rows):
        S.layer_rows.pop(i)
        _draw_layer_rows()


def btn_add_layer() -> None:
    src = dpg.get_value("new_layer_source") or "solid"
    cfg = {"source": src, "params": _NEW_LAYER_PARAMS.get(src, "")}
    if src == "shadertoy":
        shaders = _shader_choices()
        cfg["file"] = shaders[0] if shaders else ""
    _add_layer_row(cfg)
    _draw_layer_rows()


def _leading_comments(path: str) -> str:
    """The comment block at the top of a scene file (credits, how to render), kept on save."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, UnicodeDecodeError):
        return ""
    head = []
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            head.append(ln)
        else:
            break
    return "\n".join(head).rstrip() + "\n" if any(l.startswith("#") for l in head) else ""



def btn_scene_to_editor():
    import yaml
    try:
        with open(S.scene_path, encoding="utf-8") as f:
            scene = yaml.safe_load(f) or {}
    except FileNotFoundError:
        scene = {}
    video = scene.get("video", {}) or {}
    S.scene_clip_per_bar = bool(video.get("clip_per_bar", True))
    S.scene_clip_order   = video.get("clip_order", "shuffle")
    S._scene_extra_video = {k: v for k, v in video.items()
                            if k not in ("clip_per_bar", "clip_order", "triggers", "layers")}
    dpg.set_value("scene_cpb_check", S.scene_clip_per_bar)
    dpg.set_value("scene_order_combo", S.scene_clip_order)
    _clear_trigger_rows()
    for cfg in _video_cfg_to_rows(video):
        _add_trigger_row(cfg)
    S.layer_rows = []
    for cfg in _layers_to_rows(video):
        _add_layer_row(cfg)
    _draw_layer_rows()
    if dpg.does_item_exist("scene_editor_label"):
        n = len(S.trig_rows)
        dpg.set_value("scene_editor_label",
                      f"editando: {Path(S.scene_path).name} — "
                      f"{n} trigger(s) ativos" if n else
                      f"editando: {Path(S.scene_path).name} — sem triggers (adicione)")
    _log(f"Editor <- {Path(S.scene_path).name}: {len(S.trig_rows)} trigger(s), "
         f"{len(S.layer_rows)} layer(s)")


def btn_editor_to_scene():
    import yaml
    video = _rows_to_video_cfg(S.trig_rows, S.scene_clip_per_bar,
                               S.scene_clip_order, S._scene_extra_video)
    try:
        layers = _rows_to_layers(S.layer_rows)
    except ValueError as exc:
        _set_status(f"Scene NÃO salva — {exc}")
        _log(f"Scene não salva: {exc}")
        return
    if layers:                                   # no layers: the legacy single clips base
        video["layers"] = layers
    try:
        with open(S.scene_path, encoding="utf-8") as f:
            scene = yaml.safe_load(f) or {}
    except FileNotFoundError:
        scene = {}
    scene["video"] = video
    header = _leading_comments(S.scene_path)   # the file's credits / how-to survive the rewrite
    with open(S.scene_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(scene, f, allow_unicode=True, sort_keys=False)
    _log(f"Scene salva -> {S.scene_path} ({len(video['triggers'])} triggers, "
         f"{len(layers)} layers)")
    _set_status(f"Scene salva: {Path(S.scene_path).name}")
    if dpg.does_item_exist("scene_editor_label"):
        dpg.set_value("scene_editor_label",
                      f"editando: {Path(S.scene_path).name} — "
                      f"{len(video['triggers'])} trigger(s) salvos ✓")
    if S.loaded:
        # refresh the timeline TRIGGERS + output lanes with the new scene
        _load_trigger_events()
        _resolve_output_lane()
        _draw_timeline_static()


# ---------------------------------------------------------------------------
# Project save / load
# ---------------------------------------------------------------------------

def _project_dict() -> dict:
    """Serialise current GUI state to a plain dict."""
    return {
        "audio": S.audio_path or "",
        "midi":  S.midi_path  or "",
        "scene": S.scene_path,
        "skip_separation": S.skip_separation,
        "sync_offset_ms": S.sync_offset_ms,
        "midi_offset": S.midi_offset,
        "meter": S.meter,
        "stems": {r["name"]: r["path"] for r in S.stem_rows if r["name"] and r["path"]},
        "render": {
            "bars":       S.bars,
            "fps":        S.fps,
            "resolution": f"{S.width}x{S.height}",
            "mode":       S.mode,
            "start_time": S.start_time,
            "stems":      S.render_stems,
            "clip_fit":   S.clip_fit,
        },
        "repaint": {
            "video":    S.repaint_video,
            "start":    S.repaint_start,
            "workflow": S.repaint_workflow,
            "scene":    S.repaint_scene,
            "prompt":   S.repaint_prompt,
            "fps":      S.repaint_fps,
            "seconds":  S.repaint_seconds,
        },
        "clips": {
            "dir":        S.clips_dir or "",
            "order":      S.clip_order,
            "seed":       S.clips_seed,
            "overrides":  {int(k): v for k, v in S.clip_overrides.items()},
            "full_song":  S.full_song,
            "cache_size": S.cache_size,
            "gravity": {
                "enabled": S.grav_enable,
                "peak":    S.grav_peak,
                "floor":   S.grav_floor,
                "radius":  S.grav_radius,
                "curve":   S.grav_curve,
            },
        },
    }


def btn_save_project():
    import yaml
    if not S.audio_path:
        _set_status("Select an audio file before saving.")
        return
    stem = Path(S.audio_path).stem
    out  = Path("projects") / f"{stem}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(_project_dict(), f, allow_unicode=True, sort_keys=False)
    dpg.set_value("project_label", out.name)
    _set_status(f"Project saved → {out}")
    _log(f"Saved: {out}")


def _apply_project(data: dict):
    """Populate GUI state and widgets from a loaded project dict."""
    import yaml

    S.audio_path      = data.get("audio") or None
    S.midi_path       = data.get("midi")  or None
    S.scene_path      = data.get("scene", S.scene_path)
    S.skip_separation = bool(data.get("skip_separation", False))
    S.sync_offset_ms  = int(data.get("sync_offset_ms", S.sync_offset_ms))
    S.midi_offset     = float(data.get("midi_offset", 0.0))
    S.meter           = str(data.get("meter", "") or "")

    render = data.get("render", {})
    S.bars   = int(render.get("bars",   S.bars))
    S.fps    = int(render.get("fps",    S.fps))
    S.mode   =     render.get("mode",   S.mode)
    S.start_time   = float(render.get("start_time", 0.0))
    S.render_stems = bool(render.get("stems", False))
    S.clip_fit     = render.get("clip_fit", "contain")
    rp = data.get("repaint") or {}
    S.repaint_video    = rp.get("video", "")
    S.repaint_start    = float(rp.get("start", 0.0))
    S.repaint_workflow = rp.get("workflow", S.repaint_workflow)
    S.repaint_scene    = rp.get("scene", "(none)")
    S.repaint_prompt   = rp.get("prompt", "")
    S.repaint_fps      = float(rp.get("fps", 12.0))
    S.repaint_seconds  = float(rp.get("seconds", 0.0))
    res      =     render.get("resolution", f"{S.width}x{S.height}")
    S.width, S.height = (int(x) for x in res.split("x"))

    # Rebuild stem rows
    _clear_stem_rows()
    for name, path in (data.get("stems") or {}).items():
        _add_stem_row(name, path)

    clips = data.get("clips", {})
    S.clips_dir  = clips.get("dir") or None
    S.clip_order = clips.get("order", S.clip_order)
    S.clip_overrides = {int(k): v for k, v in (clips.get("overrides") or {}).items()}
    S.clips_seed = clips.get("seed")
    if S.clips_seed is None:
        import random as _random
        S.clips_seed = _random.randint(1, 2**31 - 1)   # saved on Save Project
    S.full_song  = bool(clips.get("full_song", S.full_song))
    S.cache_size = int(clips.get("cache_size", S.cache_size))
    grav = clips.get("gravity", {})
    S.grav_enable = bool(grav.get("enabled", S.grav_enable))
    S.grav_peak   = float(grav.get("peak",   S.grav_peak))
    S.grav_floor  = float(grav.get("floor",  S.grav_floor))
    S.grav_radius = float(grav.get("radius", S.grav_radius))
    S.grav_curve  = float(grav.get("curve",  S.grav_curve))

    # Update widgets
    dpg.set_value("audio_label",  Path(S.audio_path).name  if S.audio_path  else "(none)")
    dpg.set_value("midi_label",   Path(S.midi_path).name   if S.midi_path   else "(none)")
    dpg.set_value("scene_label",  Path(S.scene_path).name)
    dpg.configure_item("skip_sep_check", default_value=S.skip_separation)
    dpg.set_value("mode_combo", S.mode)
    dpg.set_value("res_input",  f"{S.width}x{S.height}")
    dpg.set_value("clips_label", Path(S.clips_dir).name if S.clips_dir else "(none)")
    dpg.set_value("clip_order_combo", S.clip_order)
    dpg.set_value("full_song_check", S.full_song)
    dpg.set_value("cache_input", S.cache_size)
    dpg.set_value("grav_check", S.grav_enable)
    for attr in ("grav_peak", "grav_floor", "grav_radius", "grav_curve"):
        dpg.set_value(attr, getattr(S, attr))
    dpg.set_value("sync_input", S.sync_offset_ms)
    for tag, val in (("start_time_input", S.start_time), ("stems_check", S.render_stems),
                     ("clip_fit_combo", S.clip_fit),
                     ("repaint_video_label", Path(S.repaint_video).name or "(nenhum)"),
                     ("repaint_start_input", S.repaint_start),
                     ("repaint_workflow_combo", S.repaint_workflow),
                     ("repaint_scene_combo", S.repaint_scene),
                     ("repaint_prompt_input", S.repaint_prompt),
                     ("repaint_fps_input", S.repaint_fps),
                     ("repaint_seconds_input", S.repaint_seconds)):
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, val)
    dpg.set_value("midi_offset_input", S.midi_offset)
    dpg.set_value("meter_input", S.meter)
    dpg.configure_item("clips_group", show=(S.mode == "clips"))
    if dpg.does_item_exist("mode_script_label"):
        dpg.set_value("mode_script_label", f"→ {_mode_script(S.mode)}")
    btn_scene_to_editor()   # editor always mirrors the project's scene

    _set_status(f"Project loaded from file — click Load Project to extract features.")


def _pick_project(s, a):
    import yaml
    p = _first_selection(a)
    if not p:
        return
    try:
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _apply_project(data)
        dpg.set_value("project_label", Path(p).name)
        _log(f"Opened: {p}")
    except Exception as exc:
        _set_status(f"Failed to open project: {exc}")
        _log(str(exc))


# ---------------------------------------------------------------------------
# Load project
# ---------------------------------------------------------------------------

def _load_trigger_events():
    """Build trigger events from the scene YAML for timeline display.

    Uses the real ClipComposer event pipeline (min_velocity, exclude,
    until), so the timeline shows exactly what a render would fire.
    Audio triggers run onset detection here — a few seconds per stem.
    """
    import yaml
    from core.video.composer import ClipComposer

    S.trigger_events  = []
    S.handover_points = []
    try:
        with open(S.scene_path, encoding="utf-8") as f:
            scene = yaml.safe_load(f) or {}
        video_cfg = scene.get("video", {})
        triggers  = video_cfg.get("triggers", {})
        if not triggers:
            return

        class _StubLib:  # events don't need decoded clips
            def __len__(self): return 1
            def get(self, i): raise RuntimeError("display-only composer")

        _set_status("Resolving scene triggers (audio onsets may take a moment)…")
        comp = ClipComposer(_StubLib(), S.grid, S.midi_notes, video_cfg)
        S.trigger_events = comp.events

        for name, spec in triggers.items():
            until = spec.get("until")
            if until:
                first = next((e.time for e in comp.events if e.name == until), None)
                if first is not None:
                    S.handover_points.append((name, until, float(first)))

        names = sorted({e.name for e in S.trigger_events})
        legend = ", ".join(
            f"{n}={EVENT_COLORS[i % len(EVENT_COLORS)]}" for i, n in enumerate(names)
        )
        _log(f"Triggers: {len(S.trigger_events)} events ({legend})")
        for name, until, t in S.handover_points:
            _log(f"  hand-over: {name} -> {until} em {t:.2f}s")
    except Exception as exc:
        _log(f"Trigger events indisponiveis: {exc}")


def _do_load():
    from core.audio_player import AudioPlayer, rms_envelope
    from core.feature_extractor import AudioFeatureExtractor
    from core.pipeline import ARCPipeline
    from core.rhythm.midi_automation import MidiAutomationReader
    from core.rhythm.midi_reader import parse_meter_changes, read_midi, shift_in_time
    from core.rhythm.analyzer import analyze

    S.loading = True
    try:
        prebuilt = {r["name"]: r["path"] for r in S.stem_rows if r["name"] and r["path"]}
        skip     = S.skip_separation
        msg = "Extracting features (AI separation skipped)…" if skip else "Extracting features + separating stems (may take a few minutes)…"
        _set_status(msg)
        _log(msg)
        if prebuilt:
            _log(f"Prebuilt stems: {', '.join(f'{k}={Path(v).name}' for k,v in prebuilt.items())}")

        S.extractor = AudioFeatureExtractor(
            S.audio_path, fps=S.fps,
            prebuilt_stems=prebuilt or None,
            skip_separation=skip,
        )
        S.extractor.update_num_bands(32)

        if S.midi_path:
            _set_status("Reading MIDI…")
            S.grid, S.midi_notes = read_midi(
                S.midi_path, fps=S.fps,
                meter_changes=parse_meter_changes(S.meter) if S.meter.strip() else None)
            shift_in_time(S.grid, S.midi_notes, S.midi_offset)
            S.midi_automation = MidiAutomationReader(S.midi_path, fps=S.fps)
            lanes = S.midi_automation.available_lanes
            if lanes:
                _log(f"MIDI lanes: {', '.join(lanes)}")
        else:
            _set_status("Analysing rhythm…")
            S.grid = analyze(S.extractor.y, sr=S.extractor.sample_rate, fps=S.fps)
            S.midi_automation = None
            S.midi_notes = []

        _set_status("Building pipeline…")
        S.pipeline = ARCPipeline.from_yaml(
            S.extractor, S.grid, S.scene_path,
            midi_automation=S.midi_automation,
        )

        _load_trigger_events()
        _resolve_output_lane()

        # Waveform sources (envelopes are computed per zoom window at draw time)
        _set_status("Computing waveforms…")
        S._wave_sources = {"mix": ("audio", S.extractor.y)}
        # one track per stem with real energy (prebuilt or AI-separated)
        for name, energy in S.extractor.stems_energy.items():
            if energy is not None and len(energy) and float(np.max(energy)) > 0:
                S._wave_sources[name] = ("energy", np.asarray(energy, dtype=np.float32))
        S.view_start, S.view_dur = 0.0, None

        # Audio player
        if S.player:
            S.player.stop()
        S.player = AudioPlayer(S.extractor.sample_rate)
        S.player.add_track("mix", S.extractor.y)

        dur = S.extractor.duration
        dpg.configure_item("scrubber", max_value=dur)
        dpg.set_value("scrubber", 0.0)
        sig = S.grid.time_signature
        dpg.set_value("bpm_text",
                      f"BPM {S.grid.bpm:.1f}   {sig[0]}/{sig[1]}   "
                      f"bar {S.grid.bar_duration:.2f}s   audio {dur:.1f}s")

        S.loaded = True
        _set_status("Project loaded — click timeline to seek, ▶ to play.")
        _log(f"Loaded: {Path(S.audio_path).name}")
        _refresh_controls(0.0)
        _draw_timeline_static()

    except Exception as exc:
        import traceback
        _set_status(f"Error: {exc}")
        _log(traceback.format_exc())
    finally:
        S.loading = False


def btn_load_project():
    if S.loading:
        return
    if not S.audio_path:
        _set_status("Select an audio file first.")
        return
    threading.Thread(target=_do_load, daemon=True).start()

# ---------------------------------------------------------------------------
# Playback controls
# ---------------------------------------------------------------------------

def btn_play():
    if not S.loaded or S.player is None:
        return
    t = dpg.get_value("scrubber") or 0.0
    S.player.play(start_t=t)
    lat = S.player.output_latency
    if lat > 0:
        _log(f"Audio output latency: {lat*1000:.0f}ms (referencia para o Sync)")
    _set_status("Playing…")


def btn_pause():
    if S.player:
        S.player.pause()
    _set_status("Paused.")


def btn_stop():
    if S.player:
        S.player.stop()
    dpg.set_value("scrubber", 0.0)
    _update_playhead(0.0)
    _refresh_controls(0.0)
    _set_status("Stopped.")


_pan_state = {"last": 0.0, "draw": 0.0}


def on_timeline_wheel(sender, delta):
    """Mouse wheel over the timeline: zoom keeping the time under the cursor."""
    if not S.loaded or not dpg.is_item_hovered("timeline_canvas"):
        return
    v0, vd = _view_window()
    dur = _full_duration()
    mx, _ = dpg.get_mouse_pos(local=False)
    rx, _ = dpg.get_item_rect_min("timeline_canvas")
    frac = min(1.0, max(0.0, (mx - rx) / TIMELINE_W))
    t_mouse = v0 + frac * vd
    nvd = vd * (0.8 if delta > 0 else 1.25)
    if nvd >= dur:
        btn_zoom_full()
        return
    nvd = max(1.0, nvd)
    S.view_dur   = nvd
    S.view_start = max(0.0, min(t_mouse - frac * nvd, dur - nvd))
    _draw_timeline_static()
    _update_playhead(dpg.get_value("scrubber") or 0.0)


def on_timeline_pan(sender, app_data):
    """Middle-button drag over the timeline: pan the zoom window."""
    if not S.loaded or S.view_dur is None:
        return
    if not dpg.is_item_hovered("timeline_canvas"):
        return
    dx = float(app_data[1])          # cumulative drag delta
    inc = dx - _pan_state["last"]
    _pan_state["last"] = dx
    if inc == 0.0:
        return
    v0, vd = _view_window()
    dur = _full_duration()
    S.view_start = max(0.0, min(v0 - inc / TIMELINE_W * vd, dur - vd))
    now = time.time()
    if now - _pan_state["draw"] >= 0.03:   # ~30 Hz redraw cap while dragging
        _draw_timeline_static()
        _update_playhead(dpg.get_value("scrubber") or 0.0)
        _pan_state["draw"] = now


def on_timeline_pan_end(sender, app_data):
    _pan_state["last"] = 0.0
    if S.loaded and S.view_dur is not None:
        _draw_timeline_static()
        _update_playhead(dpg.get_value("scrubber") or 0.0)


def _center_view(center: float, vd: float):
    dur = _full_duration()
    if dur <= 0:
        return
    vd = max(1.0, vd)
    S.view_dur   = None if vd >= dur else vd
    S.view_start = max(0.0, min(center - vd / 2, max(0.0, dur - vd)))
    _draw_timeline_static()
    _update_playhead(dpg.get_value("scrubber") or 0.0)


def btn_zoom_in():
    if not S.loaded:
        return
    v0, vd = _view_window()
    center = float(dpg.get_value("scrubber") or (v0 + vd / 2))
    _center_view(center, vd / 2)


def btn_zoom_out():
    if not S.loaded:
        return
    v0, vd = _view_window()
    _center_view(v0 + vd / 2, vd * 2)


def btn_zoom_full():
    if not S.loaded:
        return
    S.view_dur, S.view_start = None, 0.0
    _draw_timeline_static()
    _update_playhead(dpg.get_value("scrubber") or 0.0)


def on_scrubber_drag(sender, value):
    if S._syncing:
        return
    if S.player:
        S.player.seek(float(value))
    _update_playhead(float(value))
    _refresh_controls(float(value))


def on_timeline_click(sender, app_data):
    if not S.loaded or S.extractor is None:
        return
    mx, my = dpg.get_mouse_pos(local=False)
    rx, _  = dpg.get_item_rect_min("timeline_canvas")
    local_x = mx - rx
    if 0 <= local_x <= TIMELINE_W:
        v0, vd = _view_window()
        t = v0 + local_x / TIMELINE_W * vd
        S._syncing = True
        dpg.set_value("scrubber", t)
        S._syncing = False
        if S.player:
            S.player.seek(t)
        _update_playhead(t)
        _refresh_controls(t)

# ---------------------------------------------------------------------------
# Preview render
# ---------------------------------------------------------------------------

def _effective_video_cfg() -> dict:
    """Scene video config + GUI overrides (order, gravity, seed).

    Single source of truth so the output lane, the preview and the full
    render all resolve the exact same arrangement.
    """
    import yaml
    with open(S.scene_path, encoding="utf-8") as f:
        scene = yaml.safe_load(f) or {}
    video_cfg = dict(scene.get("video", {}))
    if S.clip_order != "(scene)":
        video_cfg["clip_order"] = S.clip_order
    if S.grav_enable:
        for spec in video_cfg.get("triggers", {}).values():
            if "gravity" in spec:
                spec["gravity"] = dict(spec["gravity"])
                spec["gravity"].update({
                    "peak": S.grav_peak, "floor": S.grav_floor,
                    "radius": S.grav_radius, "curve": S.grav_curve,
                })
    if "seed" not in video_cfg and S.clips_seed is not None:
        video_cfg["seed"] = S.clips_seed
    if S.clip_overrides:
        merged = dict(video_cfg.get("overrides") or {})
        merged.update(S.clip_overrides)
        video_cfg["overrides"] = merged
    return video_cfg


def _bar_at(t: float) -> int:
    db = S.grid.downbeats if S.grid is not None else None
    if db is None or not len(db):
        return 0
    return max(0, int(np.searchsorted(db, t, side="right")) - 1)


# ---------------------------------------------------------------------------
# Clip deck (thumbnails + drag-to-pin)
# ---------------------------------------------------------------------------

def _build_deck():
    """Extract cached thumbnails and populate the deck grid (worker thread)."""
    import subprocess as sp
    if not S.clips_dir:
        _set_status("Clips mode: selecione a pasta de clipes antes.")
        return
    folder = Path(S.clips_dir)
    tdir = folder / ".arc_thumbs"
    tdir.mkdir(exist_ok=True)
    from core.video.clip_library import VIDEO_EXTS
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    _set_status(f"Deck: gerando thumbnails ({len(paths)} clipes)…")
    thumbs = []
    for p in paths:
        out = tdir / (p.stem + ".png")
        if not out.exists():
            sp.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(p),
                    "-frames:v", "1", "-vf", "scale=72:72", str(out)],
                   capture_output=True)
        if out.exists():
            thumbs.append((p.stem, out))
    S._deck_thumbs = thumbs
    _set_status(f"Deck pronto: {len(thumbs)} clipes. (arraste um thumb até a timeline)")


def _populate_deck_ui():
    if not getattr(S, "_deck_thumbs", None):
        return
    dpg.delete_item("deck_panel", children_only=True)
    row = None
    for i, (name, png) in enumerate(S._deck_thumbs):
        if i % 12 == 0:
            row = dpg.add_group(horizontal=True, parent="deck_panel")
        try:
            w, h, _, data = dpg.load_image(str(png))
            tex = dpg.add_static_texture(w, h, data, parent="deck_textures")
            btn = dpg.add_image_button(tex, width=64, height=64, parent=row,
                                       user_data=name)
            with dpg.drag_payload(parent=btn, drag_data=name, payload_type="CLIP"):
                dpg.add_text(f"pin: {name}")
            with dpg.tooltip(btn):
                dpg.add_text(name)
        except Exception:
            pass


def btn_build_deck():
    def _run():
        _build_deck()
        S._deck_ready = True   # main loop populates UI (textures need main thread)
    threading.Thread(target=_run, daemon=True).start()


def on_timeline_drop(sender, app_data):
    """Drop a deck thumbnail on the timeline: pin that bar to the clip."""
    if not S.loaded:
        return
    v0, vd = _view_window()
    mx, _ = dpg.get_mouse_pos(local=False)
    rx, _ = dpg.get_item_rect_min("pin_strip")
    t = v0 + min(1.0, max(0.0, (mx - rx) / TIMELINE_W)) * vd
    bar = _bar_at(t)
    S.clip_overrides[bar] = str(app_data)
    _log(f"Pin: compasso {bar + 1} -> {app_data}")
    _resolve_output_lane()
    _draw_timeline_static()


def btn_clear_pins():
    S.clip_overrides = {}
    _log("Pins removidos.")
    if S.loaded:
        _resolve_output_lane()
        _draw_timeline_static()


def _resolve_output_lane():
    """Dry-run the composer over the whole song -> output lane data."""
    from core.video.resolver import MetaLibrary, resolve_song

    S.resolved_segments = []
    S.resolved_times = S.resolved_speeds = None
    if S.mode != "clips" or not S.clips_dir or S.grid is None:
        return
    try:
        _set_status("Resolving output arrangement (dry-run)…")
        lib = MetaLibrary(S.clips_dir, S.fps)
        start = float(S.grid.start_offset or 0.0)
        end = _full_duration() or (start + 60.0)
        segs, times, speeds = resolve_song(
            lib, S.grid, S.midi_notes, _effective_video_cfg(), S.fps, start, end)
        S.resolved_segments = segs
        S.resolved_times, S.resolved_speeds = times, speeds
        _log(f"Output lane: {len(segs)} segmentos, seed={S.clips_seed}")
    except Exception as exc:
        _log(f"Output lane indisponivel: {exc}")


def _clip_preview_frames(start_t: float, n_frames: int) -> list:
    """Render preview frames with the same stack a full render uses: the ClipComposer
    (when the scene has clips) under the scene's layers — generators, shaders, post-ops."""
    from core.video.clip_library import ClipLibrary
    from core.video.composer import ClipComposer
    from core.video.layers import build_compositor

    video_cfg = _effective_video_cfg()
    layers = video_cfg.get("layers")
    uses_clips = not layers or any((l or {}).get("source", "clips") == "clips" for l in layers)
    comp = None
    if uses_clips:
        if not S.clips_dir:
            raise RuntimeError("Clips mode: this scene uses clips — select a clips folder first.")
        _log(f"Clips preview: loading library from {S.clips_dir}")
        lib  = ClipLibrary(S.clips_dir, PREV_W, PREV_H, S.fps, cache_size=4)
        comp = ClipComposer(lib, S.grid, S.midi_notes, video_cfg)
        comp.seek(start_t)

    features_at = None
    if S.extractor is not None:
        features_at = lambda t: S.extractor.get_features_at_time(t, apply_gate=False)
    stack = build_compositor(comp, video_cfg, S.midi_notes, PREV_W, PREV_H,
                             fps=S.fps, features_at=features_at, grid=S.grid)
    if len(stack) > 1:
        _log(f"Preview: {len(stack)} layers")

    frames = []
    for fi in range(n_frames):
        arr = stack.frame_at(start_t + fi / S.fps)
        frames.append(_np_to_dpg(arr))
        _set_status(f"Rendering preview… {int((fi+1)/n_frames*100)}%")
    return frames


def _do_preview():
    import traceback
    from core.particles import ParticleSystem

    S.rendering = True
    S.preview_playing = False
    if S._preview_audio and S.player:
        S.player.pause()
        S._preview_audio = False
    _set_status("Rendering preview…")

    try:
        pygame.init()
        pygame.mixer.quit()  # yield audio device to sounddevice
        surf   = pygame.Surface((PREV_W, PREV_H))
        parts  = ParticleSystem(n=8_000)
        frames = []

        # Start at the bar that contains the current scrubber position
        scrub_t = float(dpg.get_value("scrubber") or 0.0)
        downbeats = S.grid.downbeats if S.grid.downbeats is not None else []
        if len(downbeats) > 0:
            # most recent downbeat <= scrub_t
            past = [d for d in downbeats if float(d) <= scrub_t + 1e-3]
            start_t = float(past[-1]) if past else float(downbeats[0])
        else:
            start_t = scrub_t

        n_bars   = max(1, int(S.preview_bars))
        n_frames = int(S.pipeline.bar_duration * n_bars * S.fps)
        dt       = 1.0 / S.fps

        _log(f"Preview: {n_bars} bar(s), {n_frames} frames @ {S.fps}fps  "
             f"start={start_t:.2f}s (scrubber={scrub_t:.2f}s)")

        if S.mode == "clips":
            frames = _clip_preview_frames(start_t, n_frames)
        else:
            for fi in range(n_frames):
                t  = start_t + fi * dt
                sl = S.pipeline.query(t, frame_idx=fi)
                if sl is None:
                    _log(f"  pipeline returned None at fi={fi}")
                    break
                c = sl.controls
                parts.step(dt=dt,
                           bass=float(c.get("bass_energy", 0)),
                           flux=float(c.get("flux", 0)),
                           beat_phase=sl.beat_phase,
                           kick=float(c.get("kick_intensity", 0)),
                           solo=float(c.get("solo_intensity", 0)))
                bass_bg = float(c.get("sub_bass", 0)) * 0.65 + float(c.get("bass_energy", 0)) * 0.35
                parts.render(surf, float(c.get("centroid", 0.5)), sl.bar_phase, bass_bg)
                frames.append(_surface_to_dpg(surf))
                _set_status(f"Rendering preview… {int((fi+1)/n_frames*100)}%")

        if not frames:
            _set_status("Preview FAILED — no frames generated.")
            _log("No frames generated. Check pipeline / grid / fps.")
            S.rendering = False
            return

        _log(f"Preview: generated {len(frames)} frames, frame size {len(frames[0])} floats")
        S.preview_frames  = frames
        S.preview_idx     = 0
        S.preview_start_t = start_t
        S._preview_last_t = 0.0
        if S.player is not None:
            S.player.play(start_t=start_t)   # audio drives the frame clock
            S._preview_audio = True
        S.preview_playing = True   # main loop takes over from here
        _set_status(f"Preview ready — {len(frames)} frames. Playing…")

    except Exception:
        _log("Preview crashed:\n" + traceback.format_exc())
        _set_status("Preview crashed — see log.")
    finally:
        S.rendering = False


def btn_preview():
    if not S.loaded:
        _set_status("Load a project first.")
        return
    if S.rendering:
        return
    threading.Thread(target=_do_preview, daemon=True).start()


def btn_stop_preview():
    S.preview_playing = False
    S.preview_idx = 0
    if S._preview_audio and S.player:
        S.player.pause()
        S._preview_audio = False
    _set_status("Preview stopped.")

# ---------------------------------------------------------------------------
# Full render
# ---------------------------------------------------------------------------

def _build_render_cmd() -> list:
    """Full-render command line for the currently selected mode."""
    if S.mode == "clips":
        cmd = [sys.executable, "clip_generator.py",
               "--file",  S.audio_path,
               *(["--clips", S.clips_dir] if S.clips_dir else []),   # generative scenes need none
               "--bars",  "0" if S.full_song else str(S.bars),
               "--cache-size", str(S.cache_size),
               "--fps",   str(S.fps),
               "--resolution", f"{S.width}x{S.height}",
               "--scene", S.scene_path]
        if S.clip_order != "(scene)":
            cmd += ["--clip-order", S.clip_order]
        if S.clips_seed is not None:
            cmd += ["--seed", str(S.clips_seed)]
        cmd += ["--codec", S.codec]
        if S.start_time > 0:
            cmd += ["--start-time", f"{S.start_time:.3f}"]
        if S.render_stems:
            cmd += ["--stems"]
        if S.clip_fit != "contain":
            cmd += ["--clip-fit", S.clip_fit]
        cmd += ["--output", _render_output_path()]
        if S.midi_path and S.midi_offset:
            cmd += ["--midi-offset", str(S.midi_offset)]
        if S.midi_path and S.meter.strip():
            cmd += ["--meter", S.meter.strip()]
        if S.grav_enable:
            cmd += ["--gravity-peak",   str(S.grav_peak),
                    "--gravity-floor",  str(S.grav_floor),
                    "--gravity-radius", str(S.grav_radius),
                    "--gravity-curve",  str(S.grav_curve)]
    else:
        script = _MODE_SCRIPTS.get(S.mode, "generators/particle_generator.py")
        cmd = [sys.executable, script,
               "--file",  S.audio_path,
               "--bars",  str(S.bars),
               "--fps",   str(S.fps),
               "--resolution", f"{S.width}x{S.height}",
               "--scene", S.scene_path]
    if S.midi_path:
        cmd += ["--midi", S.midi_path]
    return cmd


def _render_output_path() -> str:
    """render_output/<song>_<scene>[_from<start>].mp4 — one file per song, scene and start."""
    song = Path(S.audio_path or "render").stem
    scene = Path(S.scene_path or "scene").stem
    tail = f"_from{S.start_time:.0f}s" if S.start_time > 0 else ""
    return f"render_output/{song}_{scene}{tail}.mp4"


def btn_start_from_scrubber():
    """Start the render at the bar under the scrubber."""
    t = float(dpg.get_value("scrubber") or 0.0)
    if S.grid is not None and S.grid.downbeats is not None and len(S.grid.downbeats):
        past = [float(d) for d in S.grid.downbeats if float(d) <= t + 1e-3]
        t = past[-1] if past else float(S.grid.downbeats[0])
    S.start_time = round(t, 3)
    dpg.set_value("start_time_input", S.start_time)
    _set_status(f"Render começa em {S.start_time:.3f}s (compasso {_bar_at(t + 1e-3) + 1})")


def btn_render_full():
    if S.rendering:
        return
    if S.mode == "clips":
        # clips mode reads MIDI + clips itself; no feature extraction needed
        if not S.audio_path:
            _set_status("Clips mode: select an audio file first.")
            return
        needs_clips = True
        try:
            import yaml
            with open(S.scene_path, encoding="utf-8") as f:
                vc = (yaml.safe_load(f) or {}).get("video", {})
            layers = vc.get("layers")
            needs_clips = not layers or any(
                (l or {}).get("source", "clips") == "clips" for l in layers)
        except Exception:
            pass
        if needs_clips and not S.clips_dir:
            _set_status("Clips mode: select a clips folder (esta cena usa clipes).")
            return
    elif not S.loaded:
        return
    cmd = _build_render_cmd()
    _log("$ " + " ".join(Path(c).name if c.endswith(".py") else c for c in cmd))
    _set_status("Rendering (see log)…")

    out = cmd[cmd.index("--output") + 1] if "--output" in cmd else ""
    start = S.start_time

    def _run():
        S.rendering = True
        try:
            subprocess.run(cmd, check=True)
            _set_status("Render complete.")
            _log(f"Done. -> {out}" if out else "Done.")
            if out:                       # the repaint picks up where the render left it
                S.last_render, S.last_render_start = out, start
                S.repaint_video, S.repaint_start = out, start
                if dpg.does_item_exist("repaint_video_label"):
                    dpg.set_value("repaint_video_label", Path(out).name)
                    dpg.set_value("repaint_start_input", start)
        except subprocess.CalledProcessError as e:
            _set_status(f"Render failed (exit {e.returncode}).")
        finally:
            S.rendering = False

    threading.Thread(target=_run, daemon=True).start()

# ---------------------------------------------------------------------------
# Repaint (ComfyUI)
# ---------------------------------------------------------------------------

def _repaint_scenes() -> list:
    """Scenes with a repaint: block (denoise, kick, seed, prompts), for the repaint menu."""
    out = ["(none)"]
    for f in sorted(Path("examples").glob("*.yaml")):
        try:
            if "\nrepaint:" in "\n" + f.read_text(encoding="utf-8"):
                out.append(f.as_posix())
        except UnicodeDecodeError:
            pass
    return out


def _repaint_workflows() -> list:
    return sorted(f.as_posix() for f in Path("reference").glob("*.json")) if Path("reference").is_dir() else []


def _repaint_output_path() -> str:
    v = Path(S.repaint_video)
    return (v.parent / f"{v.stem}_repaint.mp4").as_posix()


def _build_repaint_cmd() -> list:
    """repaint.py's command line for the current repaint settings."""
    cmd = [sys.executable, "repaint.py", "--video", S.repaint_video]
    if S.repaint_scene and S.repaint_scene != "(none)":
        cmd += ["--scene", S.repaint_scene]
    if S.repaint_workflow:
        cmd += ["--workflow", S.repaint_workflow]
    if S.repaint_prompt.strip():
        cmd += ["--prompt", S.repaint_prompt.strip()]
    if S.midi_path:
        cmd += ["--midi", S.midi_path]
        if S.midi_offset:
            cmd += ["--midi-offset", str(S.midi_offset)]
        if S.meter.strip():
            cmd += ["--meter", S.meter.strip()]
    if S.repaint_start > 0:
        cmd += ["--start-time", f"{S.repaint_start:.3f}"]
    cmd += ["--fps", f"{S.repaint_fps:g}"]
    if S.repaint_seconds > 0:
        cmd += ["--seconds", f"{S.repaint_seconds:g}"]
    cmd += ["--output", _repaint_output_path()]
    return cmd


def _comfy_report() -> str:
    """ComfyUI's state and the GPU's, said before anything is sent to it."""
    from core.video.h3_client import ComfyH3Client
    from core.video.repaint import queue_status
    client = ComfyH3Client()
    if not client.is_available():
        return f"ComfyUI fora do ar em {client.server_url}"
    q = queue_status(client)
    msg = (f"ComfyUI ok — fila: {q['running']} rodando, {q['pending']} esperando "
           f"({q['ours']} nossos)")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        util, used, total = (x.strip() for x in out.splitlines()[0].split(","))
        msg += f" · GPU {util}% · {int(used) / 1024:.1f}/{int(total) / 1024:.0f} GB"
    except Exception:
        pass
    return msg


def btn_repaint_check():
    def _run():
        try:
            msg = _comfy_report()
        except Exception as exc:
            msg = f"ComfyUI: {exc}"
        _set_status(msg)
        _log(msg)
    threading.Thread(target=_run, daemon=True).start()


def btn_repaint():
    if S._repaint_proc is not None and S._repaint_proc.poll() is None:
        _set_status("Repaint já está rodando.")
        return
    if not S.repaint_video or not Path(S.repaint_video).is_file():
        _set_status("Repaint: escolha um vídeo (ou renderize antes).")
        return
    if not S.repaint_workflow and S.repaint_scene == "(none)":
        _set_status("Repaint: escolha um workflow ou uma cena com repaint:.")
        return
    cmd = _build_repaint_cmd()

    def _run():
        try:
            report = _comfy_report()
        except Exception as exc:
            report = f"ComfyUI: {exc}"
        _log(report)
        if "fora do ar" in report:
            _set_status(report)
            return
        if " 0 rodando, 0 esperando" not in report:
            _log("Repaint: a fila da ComfyUI tem jobs de outros — os nossos entram intercalados "
                 "(mais lento para todos). Stop Repaint tira só os nossos.")
        _log("$ " + " ".join(Path(c).name if c.endswith(".py") else c for c in cmd))
        _set_status("Repaint rodando na ComfyUI (ver log)…")
        S._repaint_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace")
        for line in S._repaint_proc.stdout:
            line = line.rstrip()
            if line:
                _log(line)
        code = S._repaint_proc.wait()
        _set_status("Repaint pronto: " + Path(_repaint_output_path()).name if code == 0
                    else f"Repaint parou (exit {code}) — ver log.")

    threading.Thread(target=_run, daemon=True).start()


def btn_repaint_stop():
    """Stop the repaint and take only our jobs out of ComfyUI's queue (others stay)."""
    proc = S._repaint_proc
    if proc is not None and proc.poll() is None:
        proc.kill()

    def _run():
        from core.video.h3_client import ComfyH3Client
        from core.video.repaint import cancel_jobs
        try:
            n = cancel_jobs(ComfyH3Client())
            msg = f"Repaint parado — {n} job(s) nossos tirados da fila; os quadros prontos ficam (roda de novo para continuar)."
        except Exception as exc:
            msg = f"Repaint parado; não consegui limpar a fila: {exc}"
        _set_status(msg)
        _log(msg)
    threading.Thread(target=_run, daemon=True).start()


def _pick_repaint_video(s, a):
    p = _first_selection(a)
    if p:
        S.repaint_video = p
        dpg.set_value("repaint_video_label", Path(p).name)


# ---------------------------------------------------------------------------
# File dialog callbacks
# ---------------------------------------------------------------------------

def _first_selection(app_data: dict) -> str:
    """Return a single file path regardless of how many were selected."""
    selections = app_data.get("selections", {})
    if selections:
        return next(iter(selections.values()))
    return app_data.get("file_path_name", "")


def _pick_audio(s, a):
    p = _first_selection(a)
    if p:
        S.audio_path = p
        dpg.set_value("audio_label", Path(p).name)
        _set_status(f"Audio selecionado: {Path(p).name} — clique em Load Project para extrair features.")

def _pick_midi(s, a):
    p = _first_selection(a)
    if p:
        S.midi_path = p
        dpg.set_value("midi_label", Path(p).name)
        _set_status(f"MIDI selecionado: {Path(p).name}")

def _pick_scene(s, a):
    p = _first_selection(a)
    if p:
        S.scene_path = p
        dpg.set_value("scene_label", Path(p).name)
        btn_scene_to_editor()   # editor always mirrors the active scene

def _pick_clips_dir(s, a):
    p = a.get("file_path_name", "")
    if p:
        S.clips_dir = p
        dpg.set_value("clips_label", Path(p).name or p)

def _mode_script(mode: str) -> str:
    return "clip_generator.py" if mode == "clips" else _MODE_SCRIPTS.get(mode, "?")


def _on_res_change(s, v):
    """Parse LARGURAxALTURA (applied on Enter or preset pick)."""
    try:
        w, h = (int(x) for x in str(v).lower().replace(" ", "").split("x"))
        if w < 16 or h < 16 or w % 2 or h % 2:
            raise ValueError
        S.width, S.height = w, h
        _set_status(f"Resolução de render: {w}x{h}")
    except Exception:
        _set_status(f"Resolução inválida: {v!r} — use LARGURAxALTURA com valores pares (ex.: 3840x2160)")


def _on_mode_change(s, v):
    S.mode = v
    if dpg.does_item_exist("clips_group"):
        dpg.configure_item("clips_group", show=(v == "clips"))
    if dpg.does_item_exist("mode_script_label"):
        dpg.set_value("mode_script_label", f"→ {_mode_script(v)}")

# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def _build_ui():
    from core.video.layers import LAYER_SOURCES
    dpg.create_context()

    with dpg.texture_registry():
        dpg.add_raw_texture(PREV_W, PREV_H, _blank_frame(),
                            tag="preview_tex",
                            format=dpg.mvFormat_Float_rgba)
    dpg.add_texture_registry(tag="deck_textures")

    with dpg.file_dialog(show=False, callback=_pick_audio, tag="dlg_audio",
                         width=620, height=420):
        dpg.add_file_extension(".mp3"); dpg.add_file_extension(".wav")
        dpg.add_file_extension(".flac"); dpg.add_file_extension(".*")

    with dpg.file_dialog(show=False, callback=_pick_midi, tag="dlg_midi",
                         width=620, height=420):
        dpg.add_file_extension(".mid"); dpg.add_file_extension(".midi")
        dpg.add_file_extension(".*")

    with dpg.file_dialog(show=False, callback=_pick_project, tag="dlg_project",
                         width=620, height=420):
        dpg.add_file_extension(".yaml"); dpg.add_file_extension(".yml")
        dpg.add_file_extension(".*")

    with dpg.file_dialog(show=False, callback=_pick_scene, tag="dlg_scene",
                         width=620, height=420):
        dpg.add_file_extension(".yaml"); dpg.add_file_extension(".yml")
        dpg.add_file_extension(".*")

    with dpg.file_dialog(show=False, callback=_pick_stem_shared, tag="dlg_stem_shared",
                         width=620, height=420):
        dpg.add_file_extension(".wav"); dpg.add_file_extension(".mp3")
        dpg.add_file_extension(".flac"); dpg.add_file_extension(".*")

    dpg.add_file_dialog(show=False, directory_selector=True,
                        callback=_pick_clips_dir, tag="dlg_clips",
                        width=620, height=420)

    with dpg.file_dialog(show=False, callback=_pick_repaint_video, tag="dlg_repaint_video",
                         width=620, height=420):
        dpg.add_file_extension(".mp4"); dpg.add_file_extension(".mov")
        dpg.add_file_extension(".*")

    with dpg.file_dialog(show=False, callback=_pick_trig_audio, tag="dlg_trig_audio",
                         width=620, height=420):
        dpg.add_file_extension(".wav"); dpg.add_file_extension(".mp3")
        dpg.add_file_extension(".flac"); dpg.add_file_extension(".*")

    with dpg.item_handler_registry(tag="timeline_handler"):
        dpg.add_item_clicked_handler(callback=on_timeline_click)

    with dpg.handler_registry():
        dpg.add_mouse_wheel_handler(callback=on_timeline_wheel)
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle,
                                   threshold=1, callback=on_timeline_pan)
        dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Middle,
                                      callback=on_timeline_pan_end)

    # -------------------------------------------------------------------
    with dpg.window(label="ARC Studio", tag="main_win",
                    no_close=True, no_move=True, no_resize=True,
                    pos=(0, 0), width=WIN_W, height=WIN_H):

        # ── Project open/save ───────────────────────────────────────────
        with dpg.group(horizontal=True):
            dpg.add_button(label="Open Project", width=100,
                           callback=lambda: dpg.show_item("dlg_project"))
            dpg.add_button(label="Save Project", width=100,
                           callback=btn_save_project)
            dpg.add_text("(unsaved)", tag="project_label", color=(160, 160, 100))
        dpg.add_separator()

        # ── File pickers ────────────────────────────────────────────────
        with dpg.group(horizontal=True):
            dpg.add_button(label="Audio", width=55,
                           callback=lambda: dpg.show_item("dlg_audio"))
            dpg.add_text("(none)", tag="audio_label")
        with dpg.group(horizontal=True):
            dpg.add_button(label="MIDI", width=55,
                           callback=lambda: dpg.show_item("dlg_midi"))
            dpg.add_text("(none)", tag="midi_label")
        with dpg.group(horizontal=True):
            dpg.add_text("  offset (s):")
            dpg.add_input_float(default_value=S.midi_offset, width=80, step=0,
                                format="%.3f", tag="midi_offset_input",
                                callback=lambda s, v: setattr(S, "midi_offset", float(v)))
            dpg.add_text("meter:")
            dpg.add_input_text(default_value=S.meter, width=110, hint="22:6/4,27:5/4",
                               tag="meter_input",
                               callback=lambda s, v: setattr(S, "meter", str(v)))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Scene", width=55,
                           callback=lambda: dpg.show_item("dlg_scene"))
            dpg.add_text(Path(S.scene_path).name, tag="scene_label")

        dpg.add_separator()

        # ── Stems panel ─────────────────────────────────────────────────
        with dpg.group(horizontal=True):
            dpg.add_text("STEMS", color=(160, 160, 160))
            dpg.add_button(label="+ Add Stem", callback=lambda s, a: _add_stem_row(), width=90)
            dpg.add_checkbox(
                label="Skip AI separation",
                tag="skip_sep_check",
                default_value=False,
                callback=lambda s, v: setattr(S, "skip_separation", v),
            )
        dpg.add_text(
            "  name        file                            ",
            color=(120, 120, 120),
        )
        with dpg.group(tag="stems_panel"):
            pass   # rows appended dynamically by _add_stem_row()

        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="Load Project", callback=btn_load_project, width=120)
            dpg.add_text("", tag="bpm_text", color=(160, 200, 160))
        dpg.add_text("Idle.", tag="status_text", color=(220, 200, 60))
        dpg.add_separator()

        # ── Two-column: controls + preview ──────────────────────────────
        with dpg.group(horizontal=True):

            # LEFT — scrubber + control tables
            with dpg.child_window(width=446, height=420, border=True):
                dpg.add_text("TIMELINE", color=(160, 160, 160))
                dpg.add_slider_float(
                    tag="scrubber", label="t (s)",
                    default_value=0.0, min_value=0.0, max_value=300.0,
                    width=420, callback=on_scrubber_drag,
                )
                dpg.add_text("", tag="bar_beat_text", color=(120, 210, 120))
                dpg.add_separator()
                dpg.add_text("AUDIO CONTROLS", color=(160, 160, 160))
                with dpg.table(tag="audio_table", header_row=True,
                               borders_innerH=True, borders_outerH=True,
                               borders_outerV=True, resizable=True):
                    dpg.add_table_column(label="name",  width_fixed=True, init_width_or_weight=150)
                    dpg.add_table_column(label="value")
                dpg.add_separator()
                dpg.add_text("MIDI AUTOMATION", color=(160, 160, 160))
                with dpg.table(tag="midi_table", header_row=True,
                               borders_innerH=True, borders_outerH=True,
                               borders_outerV=True, resizable=True):
                    dpg.add_table_column(label="lane",  width_fixed=True, init_width_or_weight=150)
                    dpg.add_table_column(label="value")

            # RIGHT — preview + render
            with dpg.child_window(width=650, height=420, border=True):
                dpg.add_text("PREVIEW", color=(160, 160, 160))
                dpg.add_image("preview_tex", width=PREV_W, height=PREV_H)
                with dpg.group(horizontal=True):
                    dpg.add_text("Frame 0/0", tag="frame_label")
                    dpg.add_button(label="Preview",
                                   callback=btn_preview, width=90)
                    dpg.add_text("Bars:")
                    dpg.add_input_int(default_value=S.preview_bars, width=75,
                                      tag="preview_bars_input",
                                      min_value=1, max_value=8,
                                      min_clamped=True, max_clamped=True,
                                      callback=lambda s, v: setattr(S, "preview_bars", v))
                    dpg.add_button(label="Stop",
                                   callback=btn_stop_preview, width=50)
                dpg.add_separator()
                dpg.add_text("RENDER / SYNTH", color=(160, 160, 160))
                with dpg.group(horizontal=True):
                    dpg.add_text("Mode:")
                    dpg.add_combo(["particles", "particles_v2", "particles_v3",
                                   "geometry", "clips"],
                                  default_value=S.mode, tag="mode_combo",
                                  width=130,
                                  callback=_on_mode_change)
                    dpg.add_text(f"→ {_mode_script(S.mode)}",
                                 tag="mode_script_label", color=(140, 170, 140))
                    dpg.add_text("Enc:")
                    dpg.add_combo(["x264", "nvenc"], default_value=S.codec,
                                  tag="codec_combo", width=80,
                                  callback=lambda s, v: setattr(S, "codec", v))
                    dpg.add_text("Bars:")
                    dpg.add_input_int(default_value=S.bars, width=60,
                                      callback=lambda s, v: setattr(S, "bars", v))
                    dpg.add_text("FPS:")
                    dpg.add_input_int(default_value=S.fps, width=55,
                                      callback=lambda s, v: setattr(S, "fps", v))
                with dpg.group(horizontal=True):
                    dpg.add_text("Resolution:")
                    dpg.add_input_text(default_value=f"{S.width}x{S.height}",
                                       tag="res_input", width=110,
                                       hint="LxA", on_enter=True,
                                       callback=_on_res_change)
                    dpg.add_combo(["854x480", "1280x720", "1920x1080",
                                   "2560x1440", "3840x2160",
                                   "480x480", "720x720", "1080x1080",
                                   "2160x2160"],
                                  default_value="", tag="res_preset",
                                  width=110,
                                  callback=lambda s, v: (
                                      dpg.set_value("res_input", v),
                                      _on_res_change(s, v)))
                with dpg.group(tag="clips_group", show=(S.mode == "clips")):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Clips Folder", width=95,
                                       callback=lambda: dpg.show_item("dlg_clips"))
                        dpg.add_text("(none)", tag="clips_label")
                    with dpg.group(horizontal=True):
                        dpg.add_text("Order:")
                        dpg.add_combo(["(scene)", "sequential", "random", "shuffle"],
                                      default_value=S.clip_order,
                                      tag="clip_order_combo", width=100,
                                      callback=lambda s, v: setattr(S, "clip_order", v))
                        dpg.add_checkbox(label="Full song", tag="full_song_check",
                                         default_value=S.full_song,
                                         callback=lambda s, v: setattr(S, "full_song", v))
                        dpg.add_text("Cache:")
                        dpg.add_input_int(default_value=S.cache_size, width=70,
                                          tag="cache_input",
                                          callback=lambda s, v: setattr(S, "cache_size", v))
                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(label="Gravity", tag="grav_check",
                                         default_value=S.grav_enable,
                                         callback=lambda s, v: setattr(S, "grav_enable", v))
                        for attr, label in (("grav_peak", "peak"), ("grav_floor", "floor"),
                                            ("grav_radius", "radius"), ("grav_curve", "curve")):
                            dpg.add_text(label)
                            dpg.add_input_float(default_value=getattr(S, attr), width=52,
                                                tag=attr, step=0, format="%.2f",
                                                callback=lambda s, v, u: setattr(S, u, v),
                                                user_data=attr)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Start (s):")
                        dpg.add_input_float(default_value=S.start_time, width=80, step=0,
                                            format="%.3f", tag="start_time_input",
                                            callback=lambda s, v: setattr(S, "start_time", float(v)))
                        dpg.add_button(label="<- scrubber", width=85,
                                       callback=btn_start_from_scrubber)
                        dpg.add_text("Fit:")
                        dpg.add_combo(["contain", "cover"], default_value=S.clip_fit,
                                      width=75, tag="clip_fit_combo",
                                      callback=lambda s, v: setattr(S, "clip_fit", v))
                        dpg.add_checkbox(label="Stems (features)", tag="stems_check",
                                         default_value=S.render_stems,
                                         callback=lambda s, v: setattr(S, "render_stems", v))
                dpg.add_button(label="Render Full",
                               callback=btn_render_full, width=120)

        dpg.add_separator()

        # ── Scene trigger editor ────────────────────────────────────────
        with dpg.collapsing_header(label="SCENE — CLIP TRIGGERS (editor)",
                                   default_open=True):
            dpg.add_text("(nenhuma cena carregada)", tag="scene_editor_label",
                         color=(160, 200, 160))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reload Scene", width=110,
                               callback=btn_scene_to_editor)
                dpg.add_button(label="Save Scene", width=100,
                               callback=btn_editor_to_scene)
                dpg.add_button(label="+ Add Trigger", width=100,
                               callback=lambda: _add_trigger_row())
                dpg.add_checkbox(label="clip per bar", tag="scene_cpb_check",
                                 default_value=S.scene_clip_per_bar,
                                 callback=lambda s, v: setattr(S, "scene_clip_per_bar", v))
                dpg.add_text("order:")
                dpg.add_combo(["sequential", "random", "shuffle"], width=100,
                              tag="scene_order_combo",
                              default_value=S.scene_clip_order,
                              callback=lambda s, v: setattr(S, "scene_clip_order", v))
            with dpg.group(tag="triggers_panel"):
                pass

        # ── Scene layer editor ──────────────────────────────────────────
        with dpg.collapsing_header(label="SCENE — LAYERS (editor)", default_open=True):
            dpg.add_text("a pilha de cima para baixo: a 1ª é a base, as seguintes vão por cima",
                         color=(120, 120, 140))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reload Scene", width=110,
                               callback=btn_scene_to_editor)
                dpg.add_button(label="Save Scene", width=100,
                               callback=btn_editor_to_scene)
                dpg.add_combo(list(LAYER_SOURCES), default_value="shadertoy", width=120,
                              tag="new_layer_source")
                dpg.add_button(label="+ Add Layer", width=100, callback=btn_add_layer)
            with dpg.group(tag="layers_panel"):
                pass

        # ── Repaint through ComfyUI ─────────────────────────────────────
        with dpg.collapsing_header(label="REPAINT — ComfyUI (img2img / edit)"):
            dpg.add_text("repinta cada quadro de um render num workflow da ComfyUI; usa a GPU "
                         "(e a fila da ComfyUI é compartilhada) — confira antes com Check",
                         color=(120, 120, 140))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Video…", width=60,
                               callback=lambda: dpg.show_item("dlg_repaint_video"))
                dpg.add_text(Path(S.repaint_video).name or "(nenhum — renderize ou escolha)",
                             tag="repaint_video_label")
                dpg.add_text("começa na música em (s):")
                dpg.add_input_float(default_value=S.repaint_start, width=80, step=0,
                                    format="%.3f", tag="repaint_start_input",
                                    callback=lambda s, v: setattr(S, "repaint_start", float(v)))
            with dpg.group(horizontal=True):
                dpg.add_text("Workflow:")
                wfs = _repaint_workflows()
                dpg.add_combo(wfs, default_value=S.repaint_workflow, width=330,
                              tag="repaint_workflow_combo",
                              callback=lambda s, v: setattr(S, "repaint_workflow", v))
                dpg.add_text("Cena:")
                dpg.add_combo(_repaint_scenes(), default_value=S.repaint_scene, width=250,
                              tag="repaint_scene_combo",
                              callback=lambda s, v: setattr(S, "repaint_scene", v))
            with dpg.group(horizontal=True):
                dpg.add_text("Prompt:")
                dpg.add_input_text(default_value=S.repaint_prompt, width=420,
                                   hint="vazio = os prompts da cena (ex.: make it claymation)",
                                   tag="repaint_prompt_input",
                                   callback=lambda s, v: setattr(S, "repaint_prompt", v))
                dpg.add_text("FPS:")
                dpg.add_input_float(default_value=S.repaint_fps, width=60, step=0,
                                    format="%.0f", tag="repaint_fps_input",
                                    callback=lambda s, v: setattr(S, "repaint_fps", float(v)))
                dpg.add_text("Segundos:")
                dpg.add_input_float(default_value=S.repaint_seconds, width=60, step=0,
                                    format="%.1f", tag="repaint_seconds_input",
                                    callback=lambda s, v: setattr(S, "repaint_seconds", float(v)))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Check ComfyUI / GPU", width=150, callback=btn_repaint_check)
                dpg.add_button(label="Repaint", width=90, callback=btn_repaint)
                dpg.add_button(label="Stop Repaint", width=100, callback=btn_repaint_stop)

        # ── Clip deck ───────────────────────────────────────────────────
        with dpg.collapsing_header(label="CLIP DECK — thumbnails + pin (arraste até a timeline)"):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Build Deck", width=100, callback=btn_build_deck)
                dpg.add_button(label="Clear Pins", width=100, callback=btn_clear_pins)
                dpg.add_text("arraste um thumbnail até um compasso da timeline para pinar",
                             color=(120, 120, 140))
            with dpg.child_window(tag="deck_panel", height=160, border=True):
                pass

        dpg.add_separator()

        # ── Timeline / DAW view ─────────────────────────────────────────
        dpg.add_text("DAW TIMELINE", color=(160, 160, 160))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play",  callback=btn_play,  width=50)
            dpg.add_button(label="Pause", callback=btn_pause, width=50)
            dpg.add_button(label="Stop",  callback=btn_stop,  width=50)
            dpg.add_text("Sync (ms):")
            dpg.add_input_int(default_value=S.sync_offset_ms, width=90,
                              tag="sync_input", step=10,
                              min_value=-500, max_value=500,
                              min_clamped=True, max_clamped=True,
                              callback=lambda s, v: setattr(S, "sync_offset_ms", v))
            dpg.add_button(label="Zoom +", callback=btn_zoom_in,   width=60)
            dpg.add_button(label="Zoom -", callback=btn_zoom_out,  width=60)
            dpg.add_button(label="Full",   callback=btn_zoom_full, width=45)

        dpg.add_button(
            label="▼ solte aqui o thumbnail para pinar o compasso sob o cursor ▼",
            tag="pin_strip", width=TIMELINE_W, height=18,
            drop_callback=on_timeline_drop, payload_type="CLIP",
        )
        dpg.add_drawlist(
            tag="timeline_canvas",
            width=TIMELINE_W,
            height=BAR_H + TRACK_H,   # grows after load
        )
        dpg.bind_item_handler_registry("timeline_canvas", "timeline_handler")

        dpg.add_separator()
        dpg.add_text("LOG", color=(160, 160, 160))
        dpg.add_input_text(tag="log_box", multiline=True,
                           default_value="", readonly=True,
                           width=1080, height=90)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="ARC Studio GUI")
    ap.add_argument("--audio",  default=None, help="Audio file path")
    ap.add_argument("--midi",   default=None, help="MIDI file path")
    ap.add_argument("--scene",  default=None, help="Scene YAML path")
    ap.add_argument("--stems",  nargs="*", default=[], metavar="name=path",
                    help="Pre-built stems, e.g. solo=input/solo.mp3")
    ap.add_argument("--skip-separation", action="store_true",
                    help="Skip AI stem separation")
    ap.add_argument("--project", default=None, help="Open a project YAML file")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    pygame.mixer.quit()

    args = _parse_args()

    # Build an initial project dict from CLI args (if any provided)
    initial_project = None
    if not args.project and not any([args.audio, args.midi, args.scene,
                                     args.stems, args.skip_separation]):
        # DEV default (temporário): sem argumentos, abre o projeto enxame.
        # Remover quando o fluxo de desenvolvimento estabilizar.
        dev_default = Path(__file__).parent / "projects" / "enxame.yaml"
        if dev_default.exists():
            args.project = str(dev_default)
            print(f"[dev] projeto padrao: {dev_default}")
    if args.project:
        import yaml
        with open(args.project, encoding="utf-8") as f:
            initial_project = yaml.safe_load(f)
    elif any([args.audio, args.midi, args.scene, args.stems, args.skip_separation]):
        stems = {}
        for item in (args.stems or []):
            name, path = item.split("=", 1)
            stems[name.strip()] = path.strip()
        initial_project = {
            "audio": args.audio or "",
            "midi":  args.midi  or "",
            "scene": args.scene or S.scene_path,
            "skip_separation": args.skip_separation,
            "stems": stems,
        }

    _build_ui()

    if initial_project:
        _apply_project(initial_project)
        if args.project:
            dpg.set_value("project_label", Path(args.project).name)

    dpg.create_viewport(title="ARC Studio", width=WIN_W + 10, height=WIN_H + 10)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_win", True)

    while dpg.is_dearpygui_running():
        # Deck thumbnails ready -> load textures (must run on main thread)
        if S._deck_ready:
            S._deck_ready = False
            _populate_deck_ui()

        # Preview playback — runs entirely on main thread (OpenGL texture update)
        if S.preview_playing and S.preview_frames:
            if S._preview_audio and S.player and S.player.playing:
                # frame index driven by the audio clock (A/V locked)
                t_a = S.player.current_time + S.sync_offset_ms / 1000.0
                idx = int((t_a - S.preview_start_t) * S.fps)
                if idx >= len(S.preview_frames):
                    S.player.seek(S.preview_start_t)   # loop audio with video
                    idx = 0
                    _set_status(f"Preview loop — {len(S.preview_frames)} frames. Stop to pause.")
                idx = max(0, idx)
                if idx != S.preview_idx:
                    dpg.set_value("preview_tex", S.preview_frames[idx])
                    dpg.set_value("frame_label",
                                  f"Frame {idx + 1} / {len(S.preview_frames)}")
                    S.preview_idx = idx
            else:
                now = time.time()
                if now - S._preview_last_t >= 1.0 / max(1, S.fps):
                    dpg.set_value("preview_tex", S.preview_frames[S.preview_idx])
                    dpg.set_value("frame_label",
                                  f"Frame {S.preview_idx + 1} / {len(S.preview_frames)}")
                    S.preview_idx += 1
                    S._preview_last_t = now
                    if S.preview_idx >= len(S.preview_frames):
                        S.preview_idx = 0   # loop
                        _set_status(f"Preview loop — {len(S.preview_frames)} frames. Stop to pause.")

        # Sync playhead + scrubber when audio is playing
        if S.player and S.player.playing:
            t = S.player.current_time + S.sync_offset_ms / 1000.0
            S._syncing = True
            dpg.set_value("scrubber", t)
            S._syncing = False
            # auto-scroll the zoom window to keep the playhead visible
            if S.view_dur:
                v0, vd = _view_window()
                if t > v0 + 0.95 * vd or t < v0:
                    S.view_start = max(0.0, t - 0.05 * vd)
                    _draw_timeline_static()
            _update_playhead(t)
            # tables are expensive to rebuild — throttle to ~10 Hz so the
            # playhead itself stays smooth
            now = time.time()
            if now - S._last_controls_t >= 0.1:
                _refresh_controls(t)
                S._last_controls_t = now

        dpg.render_dearpygui_frame()

    if S.player:
        S.player.stop()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
