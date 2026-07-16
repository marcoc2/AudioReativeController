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
WIN_W, WIN_H     = 1110, 1020

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

    # waveforms: ordered dict name → np.ndarray (TIMELINE_W,) rms envelope
    waveforms: dict = {}

    # preview playback
    preview_frames:   list  = []
    preview_idx:      int   = 0
    preview_playing:  bool  = False
    _preview_last_t:  float = 0.0   # wall-clock time of last frame advance

    # render settings
    bars:   int = 8
    fps:    int = 24
    width:  int = 854
    height: int = 480
    mode:   str = "particles"

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

def _timeline_height() -> int:
    return BAR_H + max(1, len(S.waveforms)) * TRACK_H


def _draw_timeline_static():
    """Draw waveforms + bar/beat markers. Called once after project loads."""
    if not dpg.does_item_exist("timeline_canvas"):
        return
    dpg.delete_item("timeline_canvas", children_only=True)

    dur = S.extractor.duration
    H   = _timeline_height()

    # Beat markers (subtle)
    for bt in (S.grid.beats if S.grid.beats is not None else []):
        x = int(bt / dur * TIMELINE_W)
        dpg.draw_line([x, BAR_H], [x, H],
                      color=(55, 55, 75, 160), thickness=1,
                      parent="timeline_canvas")

    # Bar markers + numbers
    for i, dt in enumerate(S.grid.downbeats if S.grid.downbeats is not None else []):
        x = int(dt / dur * TIMELINE_W)
        dpg.draw_line([x, 0], [x, H],
                      color=(90, 120, 200, 200), thickness=1,
                      parent="timeline_canvas")
        dpg.draw_text([x + 3, 3], str(i + 1), size=11,
                      color=(180, 180, 180, 220),
                      parent="timeline_canvas")

    # Waveform envelopes
    for track_i, (name, env) in enumerate(S.waveforms.items()):
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

    # Playhead (created last so it renders on top; tag allows configure later)
    dpg.draw_line([0, 0], [0, H],
                  color=(255, 80, 80, 255), thickness=2,
                  tag="playhead_line", parent="timeline_canvas")


def _update_playhead(t: float):
    if not dpg.does_item_exist("playhead_line") or not S.extractor:
        return
    dur = S.extractor.duration
    x   = int(np.clip(t / dur, 0.0, 1.0) * TIMELINE_W)
    H   = _timeline_height()
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
# Project save / load
# ---------------------------------------------------------------------------

def _project_dict() -> dict:
    """Serialise current GUI state to a plain dict."""
    return {
        "audio": S.audio_path or "",
        "midi":  S.midi_path  or "",
        "scene": S.scene_path,
        "skip_separation": S.skip_separation,
        "stems": {r["name"]: r["path"] for r in S.stem_rows if r["name"] and r["path"]},
        "render": {
            "bars":       S.bars,
            "fps":        S.fps,
            "resolution": f"{S.width}x{S.height}",
            "mode":       S.mode,
        },
        "clips": {
            "dir":        S.clips_dir or "",
            "order":      S.clip_order,
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

    render = data.get("render", {})
    S.bars   = int(render.get("bars",   S.bars))
    S.fps    = int(render.get("fps",    S.fps))
    S.mode   =     render.get("mode",   S.mode)
    res      =     render.get("resolution", f"{S.width}x{S.height}")
    S.width, S.height = (int(x) for x in res.split("x"))

    # Rebuild stem rows
    _clear_stem_rows()
    for name, path in (data.get("stems") or {}).items():
        _add_stem_row(name, path)

    clips = data.get("clips", {})
    S.clips_dir  = clips.get("dir") or None
    S.clip_order = clips.get("order", S.clip_order)
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
    dpg.set_value("res_combo",  f"{S.width}x{S.height}")
    dpg.set_value("clips_label", Path(S.clips_dir).name if S.clips_dir else "(none)")
    dpg.set_value("clip_order_combo", S.clip_order)
    dpg.set_value("full_song_check", S.full_song)
    dpg.set_value("cache_input", S.cache_size)
    dpg.set_value("grav_check", S.grav_enable)
    for attr in ("grav_peak", "grav_floor", "grav_radius", "grav_curve"):
        dpg.set_value(attr, getattr(S, attr))
    dpg.configure_item("clips_group", show=(S.mode == "clips"))

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

def _do_load():
    from core.audio_player import AudioPlayer, rms_envelope
    from core.feature_extractor import AudioFeatureExtractor
    from core.pipeline import ARCPipeline
    from core.rhythm.midi_automation import MidiAutomationReader
    from core.rhythm.midi_reader import read_midi
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
            S.grid, S.midi_notes = read_midi(S.midi_path, fps=S.fps)
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

        # Waveforms
        _set_status("Computing waveforms…")
        S.waveforms = {}
        S.waveforms["mix"] = rms_envelope(S.extractor.y, TIMELINE_W)

        # Audio player
        if S.player:
            S.player.stop()
        S.player = AudioPlayer(S.extractor.sample_rate)
        S.player.add_track("mix", S.extractor.y)

        dur = S.extractor.duration
        dpg.configure_item("scrubber", max_value=dur)
        dpg.set_value("scrubber", 0.0)
        dpg.set_value("bpm_text",
                      f"BPM {S.grid.bpm:.1f}   bar {S.grid.bar_duration:.2f}s   "
                      f"audio {dur:.1f}s")

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
        t = local_x / TIMELINE_W * S.extractor.duration
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

def _clip_preview_frames(start_t: float, n_frames: int) -> list:
    """Render preview frames with the real ClipComposer (clips mode)."""
    import yaml
    from core.video.clip_library import ClipLibrary
    from core.video.composer import ClipComposer

    if not S.clips_dir:
        raise RuntimeError("Clips mode: select a clips folder first.")

    with open(S.scene_path, encoding="utf-8") as f:
        scene = yaml.safe_load(f) or {}
    video_cfg = dict(scene.get("video", {}))
    if S.clip_order != "(scene)":
        video_cfg["clip_order"] = S.clip_order
    if S.grav_enable:
        for spec in video_cfg.get("triggers", {}).values():
            if "gravity" in spec:
                spec["gravity"].update({
                    "peak": S.grav_peak, "floor": S.grav_floor,
                    "radius": S.grav_radius, "curve": S.grav_curve,
                })

    _log(f"Clips preview: loading library from {S.clips_dir}")
    lib  = ClipLibrary(S.clips_dir, PREV_W, PREV_H, S.fps, cache_size=4)
    comp = ClipComposer(lib, S.grid, S.midi_notes, video_cfg)
    comp.seek(start_t)

    frames = []
    for fi in range(n_frames):
        arr = comp.frame_at(start_t + fi / S.fps)
        frames.append(_np_to_dpg(arr))
        _set_status(f"Rendering preview… {int((fi+1)/n_frames*100)}%")
    return frames


def _do_preview():
    import traceback
    from core.particles import ParticleSystem

    S.rendering = True
    S.preview_playing = False
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

        n_frames = int(S.pipeline.bar_duration * S.fps)
        dt       = 1.0 / S.fps

        _log(f"Preview: {n_frames} frames @ {S.fps}fps  start={start_t:.2f}s "
             f"(scrubber={scrub_t:.2f}s)")

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
        S._preview_last_t = 0.0
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
    _set_status("Preview stopped.")

# ---------------------------------------------------------------------------
# Full render
# ---------------------------------------------------------------------------

def _build_render_cmd() -> list:
    """Full-render command line for the currently selected mode."""
    if S.mode == "clips":
        cmd = [sys.executable, "clip_generator.py",
               "--file",  S.audio_path,
               "--clips", S.clips_dir,
               "--bars",  "0" if S.full_song else str(S.bars),
               "--cache-size", str(S.cache_size),
               "--fps",   str(S.fps),
               "--resolution", f"{S.width}x{S.height}",
               "--scene", S.scene_path]
        if S.clip_order != "(scene)":
            cmd += ["--clip-order", S.clip_order]
        if S.grav_enable:
            cmd += ["--gravity-peak",   str(S.grav_peak),
                    "--gravity-floor",  str(S.grav_floor),
                    "--gravity-radius", str(S.grav_radius),
                    "--gravity-curve",  str(S.grav_curve)]
    else:
        script = "particle_generator.py" if S.mode == "particles" else "animation_generator.py"
        cmd = [sys.executable, script,
               "--file",  S.audio_path,
               "--bars",  str(S.bars),
               "--fps",   str(S.fps),
               "--resolution", f"{S.width}x{S.height}",
               "--scene", S.scene_path]
    if S.midi_path:
        cmd += ["--midi", S.midi_path]
    return cmd


def btn_render_full():
    if S.rendering:
        return
    if S.mode == "clips":
        # clips mode reads MIDI + clips itself; no feature extraction needed
        if not (S.audio_path and S.clips_dir):
            _set_status("Clips mode: select an audio file and a clips folder first.")
            return
    elif not S.loaded:
        return
    cmd = _build_render_cmd()
    _log("$ " + " ".join(Path(c).name if c.endswith(".py") else c for c in cmd))
    _set_status("Rendering (see log)…")

    def _run():
        S.rendering = True
        try:
            subprocess.run(cmd, check=True)
            _set_status("Render complete.")
            _log("Done.")
        except subprocess.CalledProcessError as e:
            _set_status(f"Render failed (exit {e.returncode}).")
        finally:
            S.rendering = False

    threading.Thread(target=_run, daemon=True).start()

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

def _pick_clips_dir(s, a):
    p = a.get("file_path_name", "")
    if p:
        S.clips_dir = p
        dpg.set_value("clips_label", Path(p).name or p)

def _on_mode_change(s, v):
    S.mode = v
    if dpg.does_item_exist("clips_group"):
        dpg.configure_item("clips_group", show=(v == "clips"))

# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def _build_ui():
    dpg.create_context()

    with dpg.texture_registry():
        dpg.add_raw_texture(PREV_W, PREV_H, _blank_frame(),
                            tag="preview_tex",
                            format=dpg.mvFormat_Float_rgba)

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

    with dpg.item_handler_registry(tag="timeline_handler"):
        dpg.add_item_clicked_handler(callback=on_timeline_click)

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
                    dpg.add_button(label="Preview 1 Bar",
                                   callback=btn_preview, width=120)
                    dpg.add_button(label="Stop",
                                   callback=btn_stop_preview, width=50)
                dpg.add_separator()
                dpg.add_text("RENDER", color=(160, 160, 160))
                with dpg.group(horizontal=True):
                    dpg.add_text("Mode:")
                    dpg.add_combo(["particles", "geometry", "clips"],
                                  default_value=S.mode, tag="mode_combo",
                                  width=110,
                                  callback=_on_mode_change)
                    dpg.add_text("Bars:")
                    dpg.add_input_int(default_value=S.bars, width=60,
                                      callback=lambda s, v: setattr(S, "bars", v))
                    dpg.add_text("FPS:")
                    dpg.add_input_int(default_value=S.fps, width=55,
                                      callback=lambda s, v: setattr(S, "fps", v))
                with dpg.group(horizontal=True):
                    dpg.add_text("Resolution:")
                    dpg.add_combo(["854x480", "1280x720", "1920x1080",
                                   "480x480", "720x720", "1080x1080"],
                                  default_value=f"{S.width}x{S.height}",
                                  tag="res_combo", width=120,
                                  callback=lambda s, v: (
                                      setattr(S, "width",  int(v.split("x")[0])),
                                      setattr(S, "height", int(v.split("x")[1])),
                                  ))
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
                dpg.add_button(label="Render Full",
                               callback=btn_render_full, width=120)

        dpg.add_separator()

        # ── Timeline / DAW view ─────────────────────────────────────────
        dpg.add_text("DAW TIMELINE", color=(160, 160, 160))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play",  callback=btn_play,  width=50)
            dpg.add_button(label="Pause", callback=btn_pause, width=50)
            dpg.add_button(label="Stop",  callback=btn_stop,  width=50)

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
        # Preview playback — runs entirely on main thread (OpenGL texture update)
        if S.preview_playing and S.preview_frames:
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
            t = S.player.current_time
            S._syncing = True
            dpg.set_value("scrubber", t)
            S._syncing = False
            _update_playhead(t)
            _refresh_controls(t)

        dpg.render_dearpygui_frame()

    if S.player:
        S.player.stop()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
