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
    # the scene file as loaded (core.scene_doc: saving patches it, comments and anchors
    # stay) and what the editor said right after loading it (what a save diffs against)
    scene_doc = None
    _scene_video_base: dict = {}
    # the knob editor: {"row", "path" (keys to the knob in the row's params), "state"
    # (core.knob_forms), "shared" (the anchor it aliases, if any)}
    knob_ctx = None

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
    clip_turn_wide: bool = False  # portrait render: turn landscape clips a quarter (--clip-turn-wide)
    masks_dir: str = ""         # the mask track (segment_video.py's *_segments) for cast / mask:
    profile: bool = False       # time each layer and print the table (--profile)
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
    """(H, W, 3) uint8 RGB -> flat float32 RGBA for a dpg raw texture; a frame smaller
    than the preview pane is centred on black (a portrait render between bars)."""
    rgb = arr.astype(np.float32) / 255.0
    H, W = rgb.shape[:2]
    if (W, H) != (PREV_W, PREV_H) and W <= PREV_W and H <= PREV_H:
        y0, x0 = (PREV_H - H) // 2, (PREV_W - W) // 2
        rgba = np.zeros((PREV_H, PREV_W, 4), dtype=np.float32)
        rgba[:, :, 3] = 1.0
        rgba[y0:y0 + H, x0:x0 + W, :3] = rgb
        return rgba.flatten()
    rgba = np.ones((H, W, 4), dtype=np.float32)
    rgba[:, :, :3] = rgb
    return rgba.flatten()


def _preview_size(width: int, height: int) -> tuple:
    """The render's frame scaled to fit the preview pane, its aspect kept (even sides)."""
    k = min(PREV_W / max(1, width), PREV_H / max(1, height))
    return (max(2, int(width * k) // 2 * 2), max(2, int(height * k) // 2 * 2))


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
                    "gravity", "threshold", "min_gap", "exclude",
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
    from core.knob_forms import split_tracks
    items = [int(t) if t.isdigit() else t for t in split_tracks(text)]
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
            "exclude": str((spec.get("exclude") or {}).get("trigger", "") or ""),
            "exclude_win": float((spec.get("exclude") or {}).get("window", 0.04)),
            "_extra": {k: v for k, v in spec.items() if k not in _TRIG_KNOWN_KEYS},
        })
    return rows


def _rows_to_video_cfg(rows: list, clip_per_bar: bool, clip_order: str,
                       extra: Optional[dict] = None) -> dict:
    """Editor row dicts -> scene video config (round-trip safe)."""
    extra = dict(extra or {})
    video = {"clip_per_bar": bool(clip_per_bar)}          # in the order a new key lands
    if "bars_per_clip" in extra:
        video["bars_per_clip"] = extra.pop("bars_per_clip")
    video["clip_order"] = clip_order
    for k in ("clip_end", "seed"):
        if k in extra:
            video[k] = extra.pop(k)
    video.update(extra)
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
        if str(r.get("exclude", "")).strip():
            spec["exclude"] = {"trigger": str(r["exclude"]).strip(),
                               "window": float(r.get("exclude_win", 0.04))}
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
           "radius": 0.45, "curve": 2.0, "exclude": "", "exclude_win": 0.04, "_extra": {}}
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
        from core.knob_forms import split_tracks
        names = split_tracks(u["track"])
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
            dpg.add_text("  drop hits near:")
            dpg.add_input_text(default_value=row["exclude"], width=70, hint="kick",
                               callback=_set("exclude"))
            dpg.add_text("within (s)")
            dpg.add_input_float(default_value=row["exclude_win"], width=52, step=0,
                                format="%.3f", callback=_set("exclude_win"))
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
    return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True,
                          sort_keys=False).strip()


def _layers_to_rows(video_cfg: dict, doc=None) -> list:
    """Scene ``layers:`` -> editor row dicts (params YAML keeps what the row has no widget for).

    With the scene's ``doc`` (core.scene_doc.SceneDoc) a row's params are the file's own
    text: its comments, and the anchors other layers share shown as ``*name``."""
    import yaml
    rows = []
    nodes = doc.layers() if doc is not None else []
    for i, spec in enumerate(video_cfg.get("layers") or []):
        spec = spec or {}
        params = {k: v for k, v in spec.items() if k not in _LAYER_KEYS}
        op = spec.get("opacity", 1.0)
        node = nodes[i] if i < len(nodes) else None
        rows.append({
            "source":  spec.get("source", "clips"),
            "enabled": spec.get("enabled", True) is not False,
            "file":    str(spec.get("file", "") or ""),
            "blend":   spec.get("blend", "normal"),
            "opacity": float(op) if isinstance(op, (int, float)) else 1.0,
            "bars":    _yaml_flow(spec["bars"]) if "bars" in spec else "",
            "params":  doc.layer_text(i) if node is not None else
                       yaml.safe_dump(params, allow_unicode=True, sort_keys=False) if params else "",
            "_node":   node,
        })
    return rows


def _flow(value):
    """Plain lists/dicts as flow-style ruamel containers ([[21, 33]] stays on one line)."""
    from core.scene_doc import flow
    return flow(value)


def _rows_to_layer_items(rows: list, doc=None) -> list:
    """Editor rows -> one ``(node, base, layer)`` per row, in stack order: the layer the row
    says now (ruamel types, aliases tied to the anchors they name), the file's node it came
    from and what it said when loaded (``SceneDoc.set_layers`` patches the difference).
    Raises ValueError naming the row whose YAML is wrong."""
    import yaml
    from ruamel.yaml.comments import CommentedMap

    from core.scene_doc import SceneDoc, anchors_in
    doc = doc if doc is not None else SceneDoc()
    built: dict = {}
    extra: dict = {}                # anchors the rows' texts define, for the others' aliases
    pending = list(enumerate(rows, 1))
    while pending:
        unresolved = []
        for i, r in pending:
            text = str(r.get("params", "")).strip()
            try:
                params = doc.parse_text(text, extra)
            except KeyError as exc:
                unresolved.append((i, r, exc))
                continue
            except ValueError as exc:
                raise ValueError(f"layer {i} ({r['source']}): params must be 'key: value' lines") from exc
            except Exception as exc:          # ruamel's parser and scanner errors
                raise ValueError(f"layer {i} ({r['source']}): params are not valid YAML: {exc}")
            extra.update(anchors_in(params))
            spec = CommentedMap([("source", r["source"])])
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
                    spec["bars"] = _flow(yaml.safe_load(bars))
                except yaml.YAMLError as exc:
                    raise ValueError(f"layer {i} ({r['source']}): bars is not valid YAML: {exc}")
            for k, v in params.items():
                if k not in _LAYER_KEYS:
                    spec[k] = v
            built[i] = spec
        if len(unresolved) == len(pending):
            i, r, exc = unresolved[0]
            raise ValueError(f"layer {i} ({r['source']}): no anchor for {exc.args[0]}")
        pending = [(i, r) for i, r, _ in unresolved]
    return [(r.get("_node"), r.get("_base"), built[i]) for i, r in enumerate(rows, 1)]


def _rows_to_layers(rows: list, doc=None) -> list:
    """Editor rows -> scene ``layers:`` as plain dicts; raises ValueError naming the row
    whose YAML is wrong."""
    from core.scene_doc import plain
    return [plain(layer) for _, _, layer in _rows_to_layer_items(rows, doc)]


def _add_layer_row(cfg: Optional[dict] = None) -> None:
    row = {"source": "solid", "enabled": True, "file": "", "blend": "normal",
           "opacity": 1.0, "bars": "", "params": ""}
    if cfg:
        row.update(cfg)
    row["_rid"] = S._layer_counter
    S._layer_counter += 1
    S.layer_rows.append(row)


def _row_params(row: dict) -> dict:
    """What a layer row's params text says (plain), {} while it does not parse."""
    from core.scene_doc import SceneDoc, plain
    try:
        return plain((S.scene_doc or SceneDoc()).parse_text(str(row.get("params", ""))))
    except Exception:
        return {}


def _text_with_uniform(text: str, name: str, value) -> str:
    """A layer's params text with ``name: value`` added at the end of its ``uniforms:``
    block (the block is made when there is none). Raises ValueError for a one-line
    ``uniforms: {...}``, which is left for the text editor."""
    import re
    lines = str(text or "").rstrip("\n").splitlines() if str(text or "").strip() else []
    head = next((j for j, ln in enumerate(lines) if re.match(r"^uniforms\s*:", ln)), None)
    if head is None:
        return "\n".join(lines + ["uniforms:", f"  {name}: {value}"]) + "\n"
    if any(re.match(rf"^\s+{re.escape(name)}\s*:", ln) for ln in lines[head + 1:]):
        raise ValueError("already in the scene")
    if not re.match(r"^uniforms\s*:\s*(#.*)?$", lines[head]):
        raise ValueError("uniforms is written on one line: add it in the text")
    end, indent = head + 1, "  "
    for j in range(head + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and not ln[0].isspace():
            break
        if ln.strip():
            if end == head + 1:
                indent = ln[:len(ln) - len(ln.lstrip())]
            end = j + 1
    lines.insert(end, f"{indent}{name}: {value}")
    return "\n".join(lines) + "\n"


def _add_uniform(row: dict, name: str, value) -> None:
    try:
        row["params"] = _text_with_uniform(row.get("params", ""), name, value)
    except ValueError as exc:
        _set_status(f"{name}: {exc}")
        return
    row["_open_params"] = True
    _draw_layer_rows()


def _draw_uniforms(row: dict) -> None:
    """The shader's knobs: each uniform, what the scene gives it (or its default) and
    what it does, from the shader's notes; + puts one in the scene at its default."""
    from core.shadertoy import uniforms_of
    try:
        unis = uniforms_of(row["file"]) if row.get("file") else []
    except Exception as exc:
        dpg.add_text(f"     (shader not read: {exc})", color=(200, 120, 120))
        return
    if not unis:
        return
    given = _row_params(row).get("uniforms") or {}
    with dpg.tree_node(label=f"     uniforms ({len(unis)}: {len(given)} set)",
                       default_open=row.get("_open_unis", False), tag=f"layer_unis_{row['_rid']}"):
        for u in unis:
            with dpg.group(horizontal=True):
                name, set_ = u["name"], u["name"] in given
                dpg.add_button(label="knob", width=40,
                               callback=lambda s, a, u=(row, ("uniforms", name), u["default"]):
                                   _open_knob_editor(*u))
                dpg.add_text(f"{name:<10}", color=(230, 210, 140) if set_ else (150, 150, 150))
                if set_:
                    v = given[name]
                    dpg.add_text(_yaml_flow(v)[:70] if isinstance(v, (dict, list)) else str(v),
                                 color=(170, 210, 170))
                else:
                    dpg.add_button(label="+", width=20,
                                   callback=lambda s, a, u=(row, name, u["default"]): _add_uniform(*u))
                    dpg.add_text(f"default {u['default']:g}", color=(110, 110, 110))
            if u["about"]:
                dpg.add_text(f"   {u['about']}", wrap=560, color=(120, 120, 140))


# ---------------------------------------------------------------------------
# Knob editor: what in the music moves a number (core/knob_forms.py)
# ---------------------------------------------------------------------------

def _feature_names() -> list:
    """The audio features a knob can follow (features_at's keys)."""
    names = ["centroid", "flux"]
    try:
        from core.feature_extractor import AudioFeatureExtractor
        from core.texture import TEXTURE_NAMES
        names += [f"subbands.{n}" for n in AudioFeatureExtractor.SUBBAND_RANGES]
        names += [f"texture.{n}" for n in TEXTURE_NAMES]
    except Exception:
        names += ["subbands.bass", "subbands.mid", "subbands.presence"]
    if S.extractor is not None:
        names += [f"stems.{n}" for n in getattr(S.extractor, "stems_energy", {}) or {}]
    return names


_TRIGGER_KEYS = {"track", "notes", "audio", "threshold", "min_gap", "score", "events", "snap", "voice"}


def _signal_kind(value) -> str:
    """What a shared value is: a knob, a trigger (who plays), a score, or other."""
    from core.knob_forms import form_of
    if isinstance(value, dict) and value and set(value) <= _TRIGGER_KEYS:
        return "score" if "score" in value and "events" not in value else "trigger"
    return "knob" if form_of(value) is not None else "other"


def _rows_anchors() -> dict:
    """name -> object for every anchor the layer rows hold now (their texts as edited), the
    first by stack order; the scene file's where the rows do not parse."""
    from core.scene_doc import SceneDoc, anchors_in
    doc = S.scene_doc or SceneDoc()
    out = dict(anchors_in(doc.data))
    try:
        items = _rows_to_layer_items(S.layer_rows, doc)
    except ValueError:
        return out
    found: dict = {}
    for _, _, layer in items:
        for name, obj in anchors_in(layer).items():
            found.setdefault(name, obj)
    out.update(found)
    return out


def _scene_signals() -> list:
    """The scene's shared values (its YAML anchors), as the editor shows them: one
    {name, kind, value, def: (row, path), uses: [(row, path)]} each, in stack order. The
    first place one appears defines it; every other place uses it."""
    from core.scene_doc import SceneDoc, _anchor_name, occurrences, plain
    try:
        items = _rows_to_layer_items(S.layer_rows, S.scene_doc or SceneDoc())
    except ValueError:
        return []
    sig: dict = {}
    for row, (_, _, layer) in zip(S.layer_rows, items):
        for path, obj in occurrences(layer):
            name = _anchor_name(obj)
            if not name:
                continue
            if name not in sig:
                sig[name] = {"name": name, "value": plain(obj), "def": (row, path), "uses": []}
            else:
                sig[name]["uses"].append((row, path))
    for v in sig.values():
        v["kind"] = _signal_kind(v["value"])
    return list(sig.values())


def _where(row: dict, path: tuple) -> str:
    i = S.layer_rows.index(row) + 1 if row in S.layer_rows else "?"
    return f"{i}. {row['source']} > " + " > ".join(map(str, path))


def _knob_value(row: dict, path: tuple):
    """(the knob at ``path`` in a row's params as plain, "&name" when the row defines that
    anchor there, "*name" when it uses one defined elsewhere, else None)."""
    from core.scene_doc import SceneDoc, _anchor_name, get_path, plain
    extra = _rows_anchors()
    node = get_path((S.scene_doc or SceneDoc()).parse_text(str(row.get("params", "")), extra), path)
    shared = None
    if node is not None and _anchor_name(node):
        outside = any(o is node for o in extra.values())
        shared = ("*" if outside else "&") + _anchor_name(node)
    return plain(node), shared


def _open_knob_editor(row: dict, path: tuple, default=0.0, mode: str = "knob", target=None) -> None:
    """Edit the knob at ``path`` in a layer row (``mode`` trigger / score: only a source,
    the who-plays or the transcription a signal shares). ``target``, in place of a row:
    another owner of the layer (the Arrange window's plugin), with ``title``,
    ``value(path)`` -> (plain, shared), ``anchors()`` -> {name: object} and
    ``apply(path, spec, anchor, links)`` -> an error or None."""
    from core.knob_forms import new_state, state_of
    try:
        value, shared = target.value(path) if target is not None else _knob_value(row, path)
    except Exception as exc:
        _set_status(f"knob: the layer's params do not parse ({exc})")
        return
    if mode == "trigger":
        state = state_of({"hits": value or {"notes": [36]}})
    elif mode == "score":
        state = state_of({"harmony": value or {"score": _default_score_path()}})
    else:
        state = state_of(value) if value is not None else new_state("constant")
        if value is None:
            state["main"] = float(default)
    link = shared[1:] if shared and shared[0] == "*" else None
    S.knob_ctx = {"row": row, "path": tuple(path), "state": state, "shared": shared,
                  "mode": mode, "link": link, "anchor": "", "target": target}
    where = " > ".join(map(str, path))
    dpg.set_value("knob_title", f"{target.title if target is not None else row['source']} · {where}")
    if shared and shared[0] == "*":
        note = (f"this is {shared}, a signal another layer defines: unlink it to give this "
                f"layer its own")
    elif shared:
        note = f"this knob is {shared}: the layers that use *{shared[1:]} follow what you set here"
    else:
        note = ""
    dpg.set_value("knob_shared", note)
    _knob_draw()
    dpg.show_item("knob_win")
    dpg.focus_item("knob_win")


def _knob_draw() -> None:
    if not dpg.does_item_exist("knob_form") or S.knob_ctx is None:
        return
    dpg.delete_item("knob_form", children_only=True)
    ctx = S.knob_ctx
    if ctx["mode"] == "knob":
        _draw_knob_link("knob_form", ctx)
    if not ctx.get("link"):
        _draw_knob_state("knob_form", ctx["state"], 0, only_source=ctx["mode"] != "knob")
    _knob_show_yaml()


def _knob_signals() -> list:
    """The signals the open knob can use: the layer rows', or its target's."""
    from core.scene_doc import plain
    target = (S.knob_ctx or {}).get("target")
    if target is None:
        return _scene_signals()
    return [{"name": n, "value": plain(o), "kind": _signal_kind(plain(o))} for n, o in target.anchors().items()]


def _draw_knob_link(parent, ctx: dict) -> None:
    """Use a signal (a knob another place defines) or name this one so others can."""
    from core.knob_forms import state_of
    sigs = {v["name"]: v for v in _knob_signals() if v["kind"] == "knob"}
    own = ctx["shared"][1:] if ctx["shared"] and ctx["shared"][0] == "&" else None
    others = [n for n in sigs if n != own]

    def use(s, v, ctx=ctx):
        if v in sigs:
            ctx["link"] = v
            ctx["state"] = state_of(sigs[v]["value"])
            _knob_draw()

    def unlink(s, a, ctx=ctx):
        ctx["link"] = None
        _knob_draw()

    with dpg.group(parent=parent, horizontal=True):
        if ctx.get("link"):
            dpg.add_text(f"uses the signal *{ctx['link']}", color=(230, 210, 140))
            dpg.add_button(label="unlink (its own copy)", callback=unlink)
        else:
            dpg.add_text("use a signal")
            dpg.add_combo(others, width=120, callback=use)
            dpg.add_text("or name this one")
            dpg.add_input_text(default_value=own or ctx.get("anchor", ""), width=110,
                               hint="kick", enabled=own is None,
                               callback=lambda s, v, ctx=ctx: ctx.update({"anchor": v}))


def _knob_spec(ctx: dict):
    """What Apply writes: the knob (or, in trigger/score mode, its source)."""
    from core.knob_forms import spec_of
    spec = spec_of(ctx["state"])
    if ctx["mode"] == "trigger":
        return spec["hits"]
    if ctx["mode"] == "score":
        return spec["harmony"]
    return spec


def _knob_links(st: dict, prefix: tuple, anchors: dict) -> dict:
    """{path in the knob: anchored object} for the sources tied to a signal."""
    out = {}
    src = st.get("source")
    if isinstance(src, dict) and src.get("link") in anchors:
        out[prefix + (st["form"],)] = anchors[src["link"]]
    if st.get("from"):
        out.update(_knob_links(st["from"], prefix + ("from",), anchors))
    return out


def _knob_show_yaml() -> None:
    ctx = S.knob_ctx
    try:
        text = f"*{ctx['link']}" if ctx.get("link") else _yaml_flow(_knob_spec(ctx))
    except Exception as exc:
        text = f"({exc})"
    dpg.set_value("knob_yaml", text)


def _draw_knob_source_link(st: dict, kind: str, pad: str) -> bool:
    """A source tied to a signal (who plays, the transcription): shows it and an unlink;
    else a menu of the signals of that kind. True when it is tied (no fields to draw)."""
    from core.knob_forms import _trigger_state
    src = st["source"]
    sigs = {v["name"]: v for v in _knob_signals() if v["kind"] == kind}

    def use(s, v, src=src):
        if v not in sigs:
            return
        value = sigs[v]["value"]
        if kind == "trigger":
            src.update(_trigger_state(value))
        else:
            src.update({"score": str(value.get("score", "")), "snap": value.get("snap", "auto"),
                        "keep_snap": "snap" in value,
                        "extra": {k: x for k, x in value.items() if k not in ("score", "snap")}})
        src["link"] = v
        _knob_draw()

    def unlink(s, a, src=src):
        src.pop("link", None)
        _knob_draw()

    if src.get("link"):
        with dpg.group(horizontal=True):
            dpg.add_text(f"{pad}  from the signal *{src['link']}", color=(230, 210, 140))
            dpg.add_button(label="unlink", callback=unlink)
        return True
    if sigs:
        with dpg.group(horizontal=True):
            dpg.add_text(f"{pad}  or from a signal")
            dpg.add_combo(list(sigs), width=120, callback=use)
    return False


def _draw_knob_state(parent, st: dict, depth: int, only_source: bool = False) -> None:
    """The widgets for one knob state (and, for ``rate``, its ``from`` knob under it);
    ``only_source``: just where it comes from (a trigger or score signal)."""
    from core.knob_forms import FORMS, SHAPING, SNAPS, TRIGGER_KINDS, choices, new_state
    pad = "    " * depth
    labels = dict(choices())
    by_label = {v: k for k, v in labels.items()}

    def upd(d, key):
        def cb(s, v):
            d[key] = v
            _knob_show_yaml()
        return cb

    def on_form(s, v, st=st):
        st.update(new_state(by_label[v], like=st))
        _knob_draw()

    with dpg.group(parent=parent):
        if st["form"] == "custom":
            dpg.add_text(f"{pad}(a form the editor does not know: edit it in the text)")
            return
        if not only_source:
            with dpg.group(horizontal=True):
                dpg.add_text(f"{pad}what in the music")
                dpg.add_combo(list(labels.values()), default_value=labels[st["form"]], width=440,
                              callback=on_form)
        f = FORMS[st["form"]]
        if "main" in f and not only_source:
            name, _, kind, about = f["main"]
            with dpg.group(horizontal=True):
                dpg.add_text(f"{pad}  {name}")
                _knob_setting(st, "main", st["main"], kind, about)
        src = st["source"]
        tied = (f["source"] in ("trigger", "score") and not only_source
                and _draw_knob_source_link(st, f["source"], pad))
        if tied:
            pass
        elif f["source"] == "trigger":
            def on_kind(s, v, src=src):
                src["kind"] = v
                _knob_draw()
            with dpg.group(horizontal=True):
                dpg.add_text(f"{pad}  from")
                dpg.add_combo(TRIGGER_KINDS, default_value=src["kind"], width=80, callback=on_kind)
                if src["kind"] == "track":
                    def add_track(s, v, src=src):
                        from core.knob_forms import split_tracks
                        names = split_tracks(src["track"])
                        if v and v not in names:
                            names.append(v)
                        src["track"] = ", ".join(names)
                        _knob_draw()
                    dpg.add_input_text(default_value=src["track"], width=150, hint="kick, kick-2",
                                       callback=upd(src, "track"))
                    dpg.add_combo(_midi_track_names(), width=110, callback=add_track)
                elif src["kind"] == "audio":
                    dpg.add_input_text(default_value=src["audio"], width=300, hint="stems/snare.wav",
                                       callback=upd(src, "audio"))
                elif src["kind"] == "melody":
                    dpg.add_input_text(default_value=src["score"] or _default_score_path(), width=200,
                                       hint="input/song/sheetsage", callback=upd(src, "score"))
                    dpg.add_input_text(default_value=src["voice"], width=70, hint="voice",
                                       callback=upd(src, "voice"))
                if src["kind"] != "audio":
                    dpg.add_text("notes")
                    dpg.add_input_text(default_value=src["notes"], width=80, hint="36, 38 (all)",
                                       callback=upd(src, "notes"))
        elif f["source"] == "score":
            if not src.get("score"):
                src["score"] = _default_score_path()
            with dpg.group(horizontal=True):
                dpg.add_text(f"{pad}  score")
                dpg.add_input_text(default_value=src["score"], width=250, hint="input/song/sheetsage",
                                   callback=upd(src, "score"))
                dpg.add_text("snap")
                dpg.add_combo(SNAPS, default_value=src.get("snap", "auto"), width=60,
                              callback=upd(src, "snap"))
        elif f["source"] == "feature":
            def on_feature(s, v, st=st):
                st["source"] = v
                _knob_show_yaml()
            with dpg.group(horizontal=True):
                dpg.add_text(f"{pad}  feature")
                feats = _feature_names()
                dpg.add_combo(feats if st["source"] in feats else [st["source"]] + feats,
                              default_value=st["source"], width=220, callback=on_feature)
        if only_source:
            return
        for name, default, kind, about in f["params"]:
            with dpg.group(horizontal=True):
                dpg.add_text(f"{pad}  {name}")
                _knob_setting(st["params"], name, st["params"].get(name, default), kind, about)
        if st["form"] != "constant":
            with dpg.group(horizontal=True):
                dpg.add_text(f"{pad}  value = bias + gain x it:")
                for name, default, kind, _ in SHAPING:
                    dpg.add_text(name)
                    _knob_setting(st, name, st.get(name, default), kind, "")
        if st["form"] == "rate":
            if st.get("from") is None:
                def follow(s, a, st=st):
                    st["from"] = new_state("hits")
                    _knob_draw()
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{pad}  rate x 1 (a steady drift)")
                    dpg.add_button(label="make the rate follow the music", callback=follow)
            else:
                def steady(s, a, st=st):
                    st["from"] = None
                    _knob_draw()
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{pad}  rate x (what follows):")
                    dpg.add_button(label="steady instead", callback=steady)
                _draw_knob_state(parent, st["from"], depth + 1)


def _knob_setting(d: dict, key: str, value, kind, about: str) -> None:
    """One setting's widget: a number, a switch, a word or one of a few."""
    def cb(s, v):
        d[key] = v
        _knob_show_yaml()
    if isinstance(kind, tuple):
        w = dpg.add_combo(kind, default_value=str(value), width=110, callback=cb)
    elif kind == "bool":
        w = dpg.add_checkbox(default_value=bool(value), callback=cb)
    elif kind == "int":
        w = dpg.add_input_int(default_value=int(value), width=90, callback=cb)
    elif kind == "text":
        w = dpg.add_input_text(default_value=str(value), width=90, callback=cb)
    else:
        w = dpg.add_input_float(default_value=float(value), width=90, step=0, format="%.3f",
                                callback=cb)
    if about:
        dpg.add_text(about, color=(120, 120, 140))
    return w


def _knob_cut_times():
    segs = S.resolved_segments or []
    return [float(seg.t0) for seg in segs[1:]] if len(segs) > 1 else None


def _knob_span() -> tuple:
    """The stretch a knob curve covers: what the timeline shows (its zoom), else 16 s
    around the scrubber."""
    t = float(dpg.get_value("scrubber") or 0.0)
    if _full_duration() > 0:
        v0, vd = _view_window()
        if vd > 1.0:
            return v0, v0 + vd
    return t - 4.0, t + 12.0


def btn_knob_curve() -> None:
    """Draw the knob over the stretch of song the timeline shows."""
    from core.knob_forms import sample, spec_of
    if S.knob_ctx is None:
        return
    if not S.loaded:
        _set_status("knob curve: Load Project first (it needs the MIDI, the grid and the audio)")
        return
    t = float(dpg.get_value("scrubber") or 0.0)
    features_at = None
    if S.extractor is not None:
        features_at = lambda u: S.extractor.get_features_at_time(u, apply_gate=False)  # as a render
    t0, t1 = _knob_span()
    if S.knob_ctx["mode"] != "knob":
        _set_status("a trigger or a score has no curve: draw the knobs that use it")
        return
    try:
        ts, vs = sample(spec_of(S.knob_ctx["state"]), t0, t1, 600,
                        notes=S.midi_notes, features_at=features_at, grid=S.grid,
                        cuts=_knob_cut_times())
    except Exception as exc:
        _set_status(f"knob curve: {exc}")
        return
    dpg.set_value("knob_series", [list(map(float, ts)), list(map(float, vs))])
    db = S.grid.downbeats if S.grid is not None and S.grid.downbeats is not None else []
    bars = [float(d) for d in db if t0 <= d <= t1]
    dpg.set_value("knob_bars", [bars])
    dpg.set_value("knob_now", [[t]])
    dpg.fit_axis_data("knob_x")
    dpg.fit_axis_data("knob_y")
    _set_status(f"knob: {vs.min():.3f} .. {vs.max():.3f} between {t0:.1f}s and {t1:.1f}s")


def btn_knob_apply() -> None:
    from core.knob_forms import spec_of
    from core.scene_doc import SceneDoc, set_knob
    ctx = S.knob_ctx
    if ctx is None:
        return
    import re
    row = ctx["row"]
    target = ctx.get("target")
    anchors = target.anchors() if target is not None else _rows_anchors()
    anchor = str(ctx.get("anchor", "") or "").strip() or None
    if anchor and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", anchor):
        _set_status(f"signal name {anchor!r}: letters, digits, - and _ (a letter first)")
        return
    if anchor and anchor in anchors:
        _set_status(f"there is a signal {anchor!r} already: pick it under 'use a signal'")
        return
    if ctx.get("link"):
        links = {(): anchors[ctx["link"]]} if ctx["link"] in anchors else {}
        if not links:
            _set_status(f"no signal *{ctx['link']} any more")
            return
        anchor = None
    elif ctx["mode"] == "knob":
        links = _knob_links(ctx["state"], (), anchors)
    else:
        links = {}
    if target is not None:
        err = target.apply(ctx["path"], _knob_spec(ctx), anchor, links)
        if err:
            _set_status(f"knob not set: {err}")
            return
        dpg.hide_item("knob_win")
        S.knob_ctx = None
        _set_status("knob set")
        return
    try:
        row["params"] = set_knob(S.scene_doc or SceneDoc(), str(row.get("params", "")),
                                 ctx["path"], _knob_spec(ctx), extra=anchors, anchor=anchor,
                                 links=links)
    except Exception as exc:
        _set_status(f"knob not set: {exc}")
        return
    row["_open_unis"] = True
    dpg.hide_item("knob_win")
    S.knob_ctx = None
    _draw_layer_rows()
    _set_status("knob set — Save Scene to write it")


def _draw_signals() -> None:
    """The scene's signals: each shared value, what it is, where it is defined and who uses
    it; edit changes it for all of them."""
    if not dpg.does_item_exist("signals_panel"):
        return
    dpg.delete_item("signals_panel", children_only=True)
    sigs = _scene_signals()
    if not sigs:
        dpg.add_text("   (none yet: name a knob in its editor to share it)", parent="signals_panel",
                     color=(120, 120, 140))
        return
    for v in sigs:
        row, path = v["def"]
        with dpg.group(parent="signals_panel"):
            with dpg.group(horizontal=True):
                dpg.add_button(label="edit", width=40, enabled=v["kind"] != "other",
                               callback=lambda s, a, u=(row, path, v["kind"]):
                                   _open_knob_editor(u[0], u[1], mode=u[2]))
                dpg.add_text(f"&{v['name']}", color=(230, 210, 140))
                dpg.add_text(v["kind"], color=(150, 180, 220))
                dpg.add_text(_yaml_flow(v["value"])[:90], color=(170, 210, 170))
            uses = "; ".join(_where(r, p) for r, p in v["uses"]) or "nobody else yet"
            dpg.add_text(f"      defined in {_where(row, path)} · used by {uses}", wrap=1000,
                         color=(120, 120, 140))


def _remember_open_nodes() -> None:
    for row in S.layer_rows:
        for key, tag in (("_open_params", f"layer_params_{row['_rid']}"),
                         ("_open_unis", f"layer_unis_{row['_rid']}")):
            if dpg.does_item_exist(tag):
                row[key] = bool(dpg.get_value(tag))


def _draw_layer_rows() -> None:
    """(Re)build the layer panel from S.layer_rows — the order on screen is the stack order."""
    if not dpg.does_item_exist("layers_panel"):
        return
    _remember_open_nodes()
    dpg.delete_item("layers_panel", children_only=True)
    _draw_signals()
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
                                  callback=lambda s, v, row=row: (row.update({"file": v}),
                                                                  _draw_layer_rows()))
                _draw_uniforms(row)
            lines = max(2, min(14, str(row["params"]).count("\n") + 2))
            with dpg.tree_node(label="     params (YAML)", tag=f"layer_params_{row['_rid']}",
                               default_open=row.get("_open_params", False)):
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


_VIDEO_EDITOR_KEYS = ("clip_per_bar", "clip_order", "triggers", "layers")


def _scene_editor_rows(doc) -> tuple:
    """The editor's view of a scene: (clip_per_bar, clip_order, the video keys it has no
    widget for, trigger rows, layer rows). Each layer row keeps its node in the file and
    what it said (``_base``), so a save patches only what changed."""
    from core.scene_doc import plain
    video = doc.plain_video()
    cpb = bool(video.get("clip_per_bar", True))
    order = video.get("clip_order", "shuffle")
    extra = {k: v for k, v in video.items() if k not in _VIDEO_EDITOR_KEYS}
    layer_rows = _layers_to_rows(video, doc)
    for row, (_, _, layer) in zip(layer_rows, _rows_to_layer_items(layer_rows, doc)):
        row["_base"] = plain(layer)
    return cpb, order, extra, _video_cfg_to_rows(video), layer_rows


def _apply_scene_editor(doc, video_base: dict, trig_rows: list, clip_per_bar: bool,
                        clip_order: str, extra: dict, layer_rows: list) -> dict:
    """Patch the scene ``doc`` with what the editor's rows changed since ``video_base``
    (what they said when loaded); returns what they say now. Raises ValueError (the row
    whose YAML is wrong) before touching the doc."""
    video = _rows_to_video_cfg(trig_rows, clip_per_bar, clip_order, extra)
    items = _rows_to_layer_items(layer_rows, doc)
    doc.apply_video(video_base, video)
    doc.set_layers(items)
    return video


# video-level keys with a widget of their own (kept in S._scene_extra_video): default, kind
_SCENE_VIDEO_WIDGETS = {"bars_per_clip": (1, int), "clip_end": ("bounce", str), "seed": (None, int)}


def _set_scene_video(key: str, value) -> None:
    """A video-level widget changed: at its default the key goes (unless the file had it)."""
    default, kind = _SCENE_VIDEO_WIDGETS[key]
    if value in ("", None):
        value = None
    else:
        try:
            value = kind(value)
        except (TypeError, ValueError):
            _set_status(f"{key}: {value!r} is not a {kind.__name__}")
            return
    if value == default and key not in S._scene_video_base:
        S._scene_extra_video.pop(key, None)
    elif value is None:
        S._scene_extra_video.pop(key, None)
    else:
        S._scene_extra_video[key] = value


def _show_scene_video_widgets() -> None:
    ex = S._scene_extra_video
    for key, tag in (("bars_per_clip", "scene_bpc_input"), ("clip_end", "scene_end_combo"),
                     ("seed", "scene_seed_input")):
        if dpg.does_item_exist(tag):
            v = ex.get(key, _SCENE_VIDEO_WIDGETS[key][0])
            dpg.set_value(tag, "" if v is None else (str(v) if key == "seed" else v))


def btn_scene_to_editor():
    from core.scene_doc import SceneDoc
    try:
        S.scene_doc = SceneDoc.load(S.scene_path)
        cpb, order, extra, trig_cfgs, layer_cfgs = _scene_editor_rows(S.scene_doc)
    except Exception as exc:                 # a hand-edited scene the editor can't read
        S.scene_doc = None
        _set_status(f"Scene não carregada no editor — {exc}")
        _log(f"Editor: {Path(S.scene_path).name} não carregada: {exc}")
        return
    S.scene_clip_per_bar = cpb
    S.scene_clip_order   = order
    S._scene_extra_video = extra
    dpg.set_value("scene_cpb_check", S.scene_clip_per_bar)
    dpg.set_value("scene_order_combo", S.scene_clip_order)
    _clear_trigger_rows()
    for cfg in trig_cfgs:
        _add_trigger_row(cfg)
    S.layer_rows = []
    for cfg in layer_cfgs:
        _add_layer_row(cfg)
    S._scene_video_base = _rows_to_video_cfg(S.trig_rows, cpb, order, extra)
    S._scene_extra_video = dict(extra)       # the widgets edit a copy; the base stays
    _show_scene_video_widgets()
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
    from core.scene_doc import SceneDoc
    if S.scene_doc is None:                  # never loaded (or unreadable): start it afresh
        S.scene_doc, S._scene_video_base = SceneDoc(), {}
    try:
        video = _apply_scene_editor(S.scene_doc, S._scene_video_base, S.trig_rows,
                                    S.scene_clip_per_bar, S.scene_clip_order,
                                    S._scene_extra_video, S.layer_rows)
    except ValueError as exc:
        _set_status(f"Scene NÃO salva — {exc}")
        _log(f"Scene não salva: {exc}")
        return
    S.scene_doc.save(S.scene_path)           # comments, anchors and layout as they were
    _log(f"Scene salva -> {S.scene_path} ({len(video['triggers'])} triggers, "
         f"{len(S.layer_rows)} layers)")
    btn_scene_to_editor()                    # the rows now point at the saved file
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
            "turn_wide":  S.clip_turn_wide,
            "masks":      S.masks_dir,
            "profile":    S.profile,
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
    S.clip_turn_wide = bool(render.get("turn_wide", False))
    S.masks_dir    = str(render.get("masks", "") or "")
    S.profile      = bool(render.get("profile", False))
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
                     ("turn_wide_check", S.clip_turn_wide), ("profile_check", S.profile),
                     ("masks_label", _masks_label()),
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
    import hashlib                           # never inside the user's clip folder
    tdir = Path("render_output") / ".thumbs" / "deck" / hashlib.sha1(
        str(folder.resolve()).encode()).hexdigest()[:12]
    tdir.mkdir(parents=True, exist_ok=True)
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
    pw, ph = _preview_size(S.width, S.height)     # the render's aspect, letterboxed
    layers = video_cfg.get("layers")
    uses_clips = not layers or any((l or {}).get("source", "clips") == "clips" for l in layers)
    comp = None
    if uses_clips:
        if not S.clips_dir:
            raise RuntimeError("Clips mode: this scene uses clips — select a clips folder first.")
        _log(f"Clips preview: loading library from {S.clips_dir}")
        lib  = ClipLibrary(S.clips_dir, pw, ph, S.fps, cache_size=4, fit=S.clip_fit,
                           turn_wide=S.clip_turn_wide)
        comp = ClipComposer(lib, S.grid, S.midi_notes, video_cfg)
        comp.seek(start_t)

    features_at = None
    if S.extractor is not None:
        features_at = lambda t: S.extractor.get_features_at_time(t, apply_gate=False)
    stack = build_compositor(comp, video_cfg, S.midi_notes, pw, ph,
                             fps=S.fps, features_at=features_at, grid=S.grid,
                             masks=S.masks_dir or None)
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
        from core import gpu
        gpu.release()          # this thread's GPU context (and all the layers made in it) goes


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
        if S.clip_turn_wide:
            cmd += ["--clip-turn-wide"]
        if S.masks_dir:
            cmd += ["--masks", S.masks_dir]
        if S.profile:
            cmd += ["--profile"]
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


def _bar_beat_to_time(text: str, grid) -> float:
    """"18.3" (bar 18, beat 3, as the DAW and the timeline count them) -> song seconds."""
    parts = [p for p in str(text).strip().split(".") if p != ""]
    if not parts or len(parts) > 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"{text!r}: write bar.beat, e.g. 18.3")
    bar, beat = int(parts[0]), int(parts[1]) if len(parts) > 1 else 1
    db = getattr(grid, "downbeats", None)
    if db is None or not len(db):
        raise ValueError("load the project first (no bars yet)")
    if not 1 <= bar <= len(db):
        raise ValueError(f"bar {bar}: the song has bars 1..{len(db)}")
    t0 = float(db[bar - 1])
    t1 = float(db[bar]) if bar < len(db) else t0 + float(grid.bar_duration)
    beats = getattr(grid, "beats", None)
    inside = [float(b) for b in (beats if beats is not None else []) if t0 - 1e-3 <= b < t1 - 1e-3]
    if not inside:                                   # no beat markers: split the bar evenly
        n = max(1, int(round((t1 - t0) / max(1e-6, float(grid.beat_duration)))))
        inside = [t0 + k * (t1 - t0) / n for k in range(n)]
    if not 1 <= beat <= len(inside):
        raise ValueError(f"bar {bar} has beats 1..{len(inside)}")
    return inside[beat - 1]


def _time_to_bar_beat(t: float, grid) -> str:
    """Song seconds -> "bar.beat" (the beat it falls in)."""
    db = getattr(grid, "downbeats", None)
    if db is None or not len(db):
        return ""
    i = max(0, int(np.searchsorted(db, t + 1e-3, side="right")) - 1)
    beats = getattr(grid, "beats", None)
    t0 = float(db[i])
    if beats is not None and len(beats):
        k = int(np.sum((beats >= t0 - 1e-3) & (beats <= t + 1e-3)))
    else:
        k = int((t - t0) // max(1e-6, float(grid.beat_duration))) + 1
    return f"{i + 1}.{max(1, k)}"


def _on_start_bar(s, v):
    try:
        t = _bar_beat_to_time(v, S.grid)
    except ValueError as exc:
        _set_status(f"Início: {exc}")
        return
    S.start_time = round(t, 3)
    dpg.set_value("start_time_input", S.start_time)
    _set_status(f"Render começa em {S.start_time:.3f}s (compasso {v})")


def btn_start_from_scrubber():
    """Start the render at the bar under the scrubber."""
    t = float(dpg.get_value("scrubber") or 0.0)
    if S.grid is not None and S.grid.downbeats is not None and len(S.grid.downbeats):
        past = [float(d) for d in S.grid.downbeats if float(d) <= t + 1e-3]
        t = past[-1] if past else float(S.grid.downbeats[0])
    S.start_time = round(t, 3)
    dpg.set_value("start_time_input", S.start_time)
    if dpg.does_item_exist("start_bar_input"):
        dpg.set_value("start_bar_input", _time_to_bar_beat(t, S.grid))
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

NEW_SCENE = """# {name}: a new scene. The clips layer is the base (and the cut schedule the cut knobs
# read); add the layers on top in the editor.
video:
  clip_per_bar: true
  clip_order: shuffle
  layers:
    - source: clips
"""


def _new_scene_path(name: str, folder: str = "scenes") -> Path:
    """scenes/<name>.yaml for a name typed in the GUI (letters, digits, - and _)."""
    import re
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip()).strip("_")
    if not stem:
        raise ValueError("give the scene a name")
    path = Path(folder) / f"{stem}.yaml"
    if path.exists():
        raise ValueError(f"{path.as_posix()} already exists")
    return path


def btn_new_scene():
    try:
        path = _new_scene_path(dpg.get_value("new_scene_name"))
    except ValueError as exc:
        _set_status(f"New Scene: {exc}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(NEW_SCENE.format(name=path.stem))
    dpg.hide_item("dlg_new_scene")
    S.scene_path = path.as_posix()
    dpg.set_value("scene_label", path.name)
    btn_scene_to_editor()
    _set_status(f"Cena nova: {path.as_posix()}")


def _pick_scene(s, a):
    p = _first_selection(a)
    if p:
        S.scene_path = p
        dpg.set_value("scene_label", Path(p).name)
        btn_scene_to_editor()   # editor always mirrors the active scene

def _masks_label() -> str:
    return Path(S.masks_dir).name if S.masks_dir else "(nenhuma — cast e mask: precisam)"


def _pick_masks_dir(s, a):
    p = a.get("file_path_name") or _first_selection(a)
    if p:
        S.masks_dir = Path(p).as_posix()
        dpg.set_value("masks_label", _masks_label())


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

class _ArrangeHost:
    """What the Arrange window (arc_arrange.py) needs from the studio."""

    def scene_path(self):
        return S.scene_path

    def grid(self):
        return S.grid

    def notes(self):
        return S.midi_notes

    def clips_dir(self):
        return S.clips_dir

    def duration(self):
        return _full_duration() if S.loaded else 0.0

    def log(self, msg):
        _log(msg)

    def status(self, msg):
        _set_status(msg)

    def edit_knob(self, target, path, default=0.0):
        """The studio's knob editor on a parameter the Arrange window owns."""
        _open_knob_editor(None, tuple(path), default, "knob", target)

    def reload_editor(self):
        btn_scene_to_editor()
        if S.loaded:
            _load_trigger_events()
            _resolve_output_lane()
            _draw_timeline_static()


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
    dpg.add_file_dialog(show=False, directory_selector=True, default_path="render_output",
                        callback=_pick_masks_dir, tag="dlg_masks",
                        width=620, height=420)

    with dpg.file_dialog(show=False, callback=_pick_repaint_video, tag="dlg_repaint_video",
                         width=620, height=420):
        dpg.add_file_extension(".mp4"); dpg.add_file_extension(".mov")
        dpg.add_file_extension(".*")

    with dpg.file_dialog(show=False, callback=_pick_trig_audio, tag="dlg_trig_audio",
                         width=620, height=420):
        dpg.add_file_extension(".wav"); dpg.add_file_extension(".mp3")
        dpg.add_file_extension(".flac"); dpg.add_file_extension(".*")

    with dpg.window(label="Knob — what in the music moves it", show=False, tag="knob_win",
                    width=760, height=600, pos=(180, 60), on_close=lambda: setattr(S, "knob_ctx", None)):
        dpg.add_text("", tag="knob_title", color=(230, 210, 140))
        dpg.add_text("", tag="knob_shared", color=(220, 160, 100), wrap=740)
        dpg.add_group(tag="knob_form")
        dpg.add_separator()
        dpg.add_text("", tag="knob_yaml", color=(150, 190, 150), wrap=740)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Draw curve", width=100, callback=btn_knob_curve)
            dpg.add_text("over what the timeline shows (zoom it to look closer); the bars and the "
                         "scrubber as lines", color=(120, 120, 140))
        with dpg.plot(height=170, width=-1, tag="knob_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="song (s)", tag="knob_x")
            with dpg.plot_axis(dpg.mvYAxis, tag="knob_y"):
                dpg.add_line_series([], [], tag="knob_series")
                dpg.add_inf_line_series([], tag="knob_bars")
                dpg.add_inf_line_series([], tag="knob_now")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply", width=90, callback=btn_knob_apply)
            dpg.add_button(label="Cancel", width=90,
                           callback=lambda: (dpg.hide_item("knob_win"), setattr(S, "knob_ctx", None)))

    import arc_arrange
    arc_arrange.build(_ArrangeHost())

    with dpg.window(label="New Scene", modal=True, show=False, tag="dlg_new_scene",
                    width=360, height=90, pos=(300, 120), no_resize=True):
        with dpg.group(horizontal=True):
            dpg.add_text("scenes/")
            dpg.add_input_text(tag="new_scene_name", width=180, hint="me_diz_ideia",
                               on_enter=True, callback=lambda s, a: btn_new_scene())
            dpg.add_text(".yaml")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Create", width=80, callback=btn_new_scene)
            dpg.add_button(label="Cancel", width=80,
                           callback=lambda: dpg.hide_item("dlg_new_scene"))

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
            dpg.add_button(label="New…", width=45,
                           callback=lambda: dpg.show_item("dlg_new_scene"))
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
                                   "2160x2160", "480x854", "720x1280", "1080x1920"],
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
                        dpg.add_text("= bar.beat")
                        dpg.add_input_text(default_value="", width=50, hint="18.3",
                                           tag="start_bar_input", on_enter=True,
                                           callback=_on_start_bar)
                        dpg.add_button(label="<- scrubber", width=85,
                                       callback=btn_start_from_scrubber)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Fit:")
                        dpg.add_combo(["contain", "cover"], default_value=S.clip_fit,
                                      width=75, tag="clip_fit_combo",
                                      callback=lambda s, v: setattr(S, "clip_fit", v))
                        dpg.add_checkbox(label="turn wide", tag="turn_wide_check",
                                         default_value=S.clip_turn_wide,
                                         callback=lambda s, v: setattr(S, "clip_turn_wide", v))
                        dpg.add_checkbox(label="Stems (features)", tag="stems_check",
                                         default_value=S.render_stems,
                                         callback=lambda s, v: setattr(S, "render_stems", v))
                        dpg.add_checkbox(label="profile", tag="profile_check",
                                         default_value=S.profile,
                                         callback=lambda s, v: setattr(S, "profile", v))
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Masks…", width=60,
                                       callback=lambda: dpg.show_item("dlg_masks"))
                        dpg.add_button(label="x", width=20,
                                       callback=lambda: (setattr(S, "masks_dir", ""),
                                                         dpg.set_value("masks_label", _masks_label())))
                        dpg.add_text(_masks_label(), tag="masks_label")
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
            with dpg.group(horizontal=True):
                dpg.add_text("a new clip every")
                dpg.add_input_int(default_value=1, width=80, min_value=1, min_clamped=True,
                                  tag="scene_bpc_input",
                                  callback=lambda s, v: _set_scene_video("bars_per_clip", v))
                dpg.add_text("bar(s) · a clip shorter than its turn:")
                dpg.add_combo(["bounce", "loop"], default_value="bounce", width=80,
                              tag="scene_end_combo",
                              callback=lambda s, v: _set_scene_video("clip_end", v))
                dpg.add_text("seed:")
                dpg.add_input_text(default_value="", width=90, hint="(project's)",
                                   tag="scene_seed_input", decimal=True,
                                   callback=lambda s, v: _set_scene_video("seed", v))
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
                dpg.add_button(label="Arrange...", width=100,
                               callback=lambda: __import__("arc_arrange").open_window())
            with dpg.tree_node(label="signals — values the layers share (kick, organ, the chords)",
                               default_open=True):
                dpg.add_button(label="refresh", width=70, callback=lambda: _draw_signals())
                with dpg.group(tag="signals_panel"):
                    pass
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
        __import__("arc_arrange").tick()       # the Arrange window's thumbnails and redraws

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
