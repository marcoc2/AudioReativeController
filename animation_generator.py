import os
import sys
import colorsys
import numpy as np
import pygame
from pathlib import Path
from collections import deque
import subprocess
import shutil

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from core.feature_extractor import AudioFeatureExtractor
    from core.motion import create_zigzag_preset
    from core.pipeline import ARCPipeline, FrameSlice
    from core.rhythm.analyzer import analyze
    from core.rhythm.grid import SUBDIVISIONS
    from core.rhythm.midi_automation import MidiAutomationReader
    from core.rhythm.midi_reader import read_midi
    from core.visual import Frame, VisualObject, render_frame
except ImportError:
    print("[Animation Gen] Error: Required modules not found.")
    sys.exit(1)


def _hue_color(hue: float, saturation: float = 1.0, value: float = 1.0) -> tuple:
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def _centroid_color(centroid: float, value: float = 1.0) -> tuple:
    """Map spectral centroid (0..1) to RGB. Low=warm (red), high=cool (cyan/blue)."""
    hue = centroid * 0.67  # red(0.0) → green(0.33) → blue(0.67)
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def _velocity_envelope(times: np.ndarray, vels: np.ndarray, t: float, decay: float = 0.3) -> float:
    """Linear-decay envelope for the most recent note at or before t."""
    if len(times) == 0:
        return 0.0
    idx = int(np.searchsorted(times, t, side="right")) - 1
    if idx < 0:
        return 0.0
    age = t - float(times[idx])
    if age > decay:
        return 0.0
    return float(vels[idx]) / 127.0 * (1.0 - age / decay)


def build_frame(
    sl: FrameSlice,
    midi_notes: list,
    kick_times: np.ndarray,
    kick_vels: np.ndarray,
    snare_times: np.ndarray,
    snare_vels: np.ndarray,
    ride_times: np.ndarray,
    ride_vels: np.ndarray,
    l_y_px: int,
    r_y_px: int,
    resolution: tuple,
    fps: int,
    bar_duration: float,
) -> Frame:
    """Produce a Frame from a FrameSlice and MIDI state.

    Discrete rendering decisions (downbeat ring, etc.) are derived from
    bar_phase — no boolean flags stored on Frame.
    """
    W, H = resolution
    bar_phase  = sl.bar_phase
    beat_phase = sl.beat_phase
    breath     = 1.0 + 0.15 * np.cos(2 * np.pi * bar_phase)

    if midi_notes:
        l_amp    = _velocity_envelope(kick_times,  kick_vels,  sl.t)
        r_amp    = _velocity_envelope(snare_times, snare_vels, sl.t)
        ride_amp = _velocity_envelope(ride_times,  ride_vels,  sl.t, decay=0.12)
    else:
        l_amp    = min(1.0, sl.controls.get("kick_intensity", 0))
        r_amp    = sl.controls.get("snare_intensity", 0)
        ride_amp = sl.controls.get("centroid", 0)

    l_r = l_amp * breath
    r_r = r_amp * breath

    lx, ly = 0.25, l_y_px / H
    rx, ry = 0.75, r_y_px / H

    centroid       = sl.controls.get("centroid", 0.5)
    flux           = sl.controls.get("flux", 0.0)
    dominant_pitch = int(sl.controls.get("dominant_pitch", 0))

    c_full = _centroid_color(centroid, value=1.0)
    c_core = _centroid_color(centroid, value=0.55)
    c_ring = _centroid_color(centroid, value=0.85)

    # Polygon parameters driven by pitch and flux
    n_verts  = 3 + (dominant_pitch % 6)          # 3..8 sides
    rotation = beat_phase * 2 * np.pi / n_verts  # rotates once per beat
    jitter   = flux * 0.4                         # jagged when spectral flux is high

    objects: list[VisualObject] = []

    if l_r > 0.01:
        objects.append(VisualObject(id="kick", x=lx, y=ly, radius=l_r, color=c_full,
                                    shape="polygon", n_vertices=n_verts,
                                    rotation=rotation, vertex_jitter=jitter))
        objects.append(VisualObject(id="kick_core", x=lx, y=ly, radius=l_r * 0.3,
                                    color=c_core, in_trail=False,
                                    shape="polygon", n_vertices=n_verts,
                                    rotation=rotation + np.pi / n_verts,
                                    vertex_jitter=jitter * 0.5))

    if r_r > 0.01:
        objects.append(VisualObject(id="snare", x=rx, y=ry, radius=r_r, color=c_ring,
                                    shape="polygon", n_vertices=n_verts,
                                    rotation=-rotation, vertex_jitter=jitter))
        objects.append(VisualObject(id="snare_core", x=rx, y=ry, radius=r_r * 0.3,
                                    color=c_core, in_trail=False,
                                    shape="polygon", n_vertices=n_verts,
                                    rotation=-rotation + np.pi / n_verts,
                                    vertex_jitter=jitter * 0.5))

    # Solo mandala: 6 polygon petals orbiting the center, hue-cycling
    solo_intensity = sl.controls.get("solo_intensity", 0.0)
    if solo_intensity > 0.05:
        n_petals = 6
        orbit_r  = 0.18 + solo_intensity * 0.12   # how far from center
        petal_r  = 0.12 + solo_intensity * 0.35   # size of each petal
        for i in range(n_petals):
            angle   = bar_phase * 2 * np.pi + (2 * np.pi * i / n_petals)
            px      = 0.5 + orbit_r * np.cos(angle)
            py      = 0.5 + orbit_r * np.sin(angle)
            hue     = (bar_phase + i / n_petals) % 1.0
            color   = _hue_color(hue, saturation=0.9, value=1.0)
            rot     = angle + beat_phase * 2 * np.pi
            objects.append(VisualObject(
                id=f"solo_{i}", x=px, y=py, radius=petal_r,
                color=color, alpha=int(solo_intensity * 210),
                shape="polygon", n_vertices=n_verts,
                rotation=rot, vertex_jitter=jitter * 1.8,
                in_trail=True,
            ))

    # Ride: thin expanding ring at center, fast decay
    if ride_amp > 0.02:
        ride_r = 0.15 + ride_amp * 0.6
        objects.append(VisualObject(id="ride", x=0.5, y=0.5, radius=ride_r,
                                    color=c_full, filled=False, ring_width=2,
                                    alpha=int(ride_amp * 200)))

    # Downbeat ring: derived from bar_phase (continuous), not a stored boolean.
    # Threshold ≈ half a frame's worth of bar, so it fires for ~1 frame.
    downbeat_thresh = 0.5 / fps / max(bar_duration, 1e-6)
    if bar_phase < downbeat_thresh or bar_phase > (1.0 - downbeat_thresh):
        objects.append(VisualObject(id="ring_left",  x=lx, y=ly, radius=1.1, color=(255, 255, 255), filled=False, ring_width=4, in_trail=False))
        objects.append(VisualObject(id="ring_right", x=rx, y=ry, radius=1.1, color=(255, 255, 255), filled=False, ring_width=4, in_trail=False))

    return Frame(frame_idx=sl.frame, t=sl.t,
                 bar_phase=bar_phase, beat_phase=beat_phase,
                 objects=objects)


def generate_animation_mp4(audio_path, output_mp4, start_sec=0.0, duration_sec=10.0, fps=24,
                           resolution=(1024, 1024), contrast=0.45, mode="demucs",
                           preset="none", speed=0.25, trail_count=0, scale=1.0,
                           bars=None, subdivision="quarter", time_signature=(4, 4),
                           midi_path=None, midi_offset=0.0, scene_path=None,
                           prebuilt_stems=None):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        return
    output_path = Path(output_mp4)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n[Animation Gen] STARTING OFFLINE RENDER\n  Input: {audio_path.name}\n{'='*60}\n")
    extractor = AudioFeatureExtractor(str(audio_path), fps=fps, separation_mode=mode,
                                      prebuilt_stems=prebuilt_stems)
    extractor.contrast_level = contrast
    extractor.update_num_bands(32)

    midi_notes = []
    midi_automation = None
    if midi_path:
        grid, midi_notes = read_midi(midi_path, time_signature=time_signature, fps=fps)
        midi_automation = MidiAutomationReader(midi_path, fps=fps)
        if midi_offset != 0.0:
            for n in midi_notes:
                n.time += midi_offset
            if grid.beats is not None:
                grid.beats = grid.beats + midi_offset
            if grid.downbeats is not None:
                grid.downbeats = grid.downbeats + midi_offset
            grid.start_offset += midi_offset
        lanes = midi_automation.available_lanes
        print(f"[Rhythm/MIDI] {Path(midi_path).name}  notes={len(midi_notes)}"
              + (f"  offset={midi_offset:+.3f}s" if midi_offset != 0.0 else ""))
        if lanes:
            print(f"[MIDI Automation] {len(lanes)} lanes: {', '.join(lanes[:8])}"
                  + (" …" if len(lanes) > 8 else ""))
    else:
        grid = analyze(extractor.y, sr=extractor.sample_rate,
                       time_signature=time_signature, fps=fps)
    subdiv_step = grid.subdivision_duration(subdivision)
    print(f"[Rhythm] BPM={grid.bpm:.1f}  bar={grid.bar_duration:.2f}s  "
          f"subdivision={subdivision} ({subdiv_step:.3f}s)  "
          f"beats={len(grid.beats) if grid.beats is not None else 0}")

    _scene_path = scene_path or (Path(__file__).parent / "scenes" / "default.yaml")
    pipeline = ARCPipeline.from_yaml(extractor, grid, str(_scene_path),
                                     midi_automation=midi_automation)
    print(f"[Pipeline] scene={Path(_scene_path).name}  controls={list(pipeline._mappings)}")

    KICK, SNARE, RIDE = 36, 38, 51
    _empty = np.array([], dtype=float)
    kick_times  = np.array([n.time     for n in midi_notes if n.pitch == KICK],  dtype=float) if midi_notes else _empty
    kick_vels   = np.array([n.velocity for n in midi_notes if n.pitch == KICK],  dtype=float) if midi_notes else _empty
    snare_times = np.array([n.time     for n in midi_notes if n.pitch == SNARE], dtype=float) if midi_notes else _empty
    snare_vels  = np.array([n.velocity for n in midi_notes if n.pitch == SNARE], dtype=float) if midi_notes else _empty
    ride_times  = np.array([n.time     for n in midi_notes if n.pitch == RIDE],  dtype=float) if midi_notes else _empty
    ride_vels   = np.array([n.velocity for n in midi_notes if n.pitch == RIDE],  dtype=float) if midi_notes else _empty

    if bars is not None:
        duration_sec = bars * grid.bar_duration
        print(f"[Rhythm] --bars {bars} -> duration {duration_sec:.2f}s")

    if start_sec + duration_sec > extractor.duration:
        duration_sec = max(0, extractor.duration - start_sec)

    temp_dir = Path("render_temp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    pygame.init()
    surface     = pygame.Surface(resolution)
    W, H        = resolution
    total_frames = int(duration_sec * fps)
    center_y    = H // 2
    max_r       = int((W // 4 - 50) * scale)

    m_left, m_right = None, None
    if preset == "zigzag":
        m_left, m_right = create_zigzag_preset(H, speed_ratio=speed)

    trail_history: deque[list[VisualObject]] = (
        deque(maxlen=trail_count) if trail_count > 0 else deque(maxlen=0)
    )

    for frame_idx in range(total_frames):
        time_sec = start_sec + (frame_idx / fps)

        l_y_px = m_left.update(1.0 / fps)  if m_left  else center_y
        r_y_px = m_right.update(1.0 / fps) if m_right else center_y

        sl = pipeline.query(time_sec, frame_idx)
        if sl is None:
            break

        frame = build_frame(
            sl, midi_notes, kick_times, kick_vels, snare_times, snare_vels,
            ride_times, ride_vels,
            l_y_px, r_y_px, resolution, fps, pipeline.bar_duration,
        )

        bass_pulse     = (sl.controls.get("sub_bass", 0) * 0.65
                          + sl.controls.get("bass_energy", 0) * 0.35)
        solo_intensity = sl.controls.get("solo_intensity", 0.0)

        if solo_intensity > 0.05:
            solo_hue = (sl.bar_phase + sl.beat_phase * 0.25) % 1.0
            sr, sg, sb = colorsys.hsv_to_rgb(solo_hue, 0.75, solo_intensity * 0.28)
            bg_r = int(max(bass_pulse * 50, sr * 255))
            bg_g = int(max(bass_pulse *  9, sg * 255))
            bg_b = int(max(bass_pulse * 14, sb * 255))
        else:
            bg   = int(bass_pulse * 50)
            bg_r, bg_g, bg_b = bg, int(bg * 0.18), int(bg * 0.28)

        surface.fill((bg_r, bg_g, bg_b))
        render_frame(surface, frame, max_r,
                     trail_frames=list(trail_history) if trail_count > 0 else None)

        if trail_count > 0:
            trail_history.append([obj for obj in frame.objects if obj.in_trail])

        pygame.image.save(surface, str(temp_dir / f"{frame_idx:04d}.png"))
        if frame_idx % 24 == 0:
            sys.stdout.write(f"\r  Progress: {frame_idx}/{total_frames}")
            sys.stdout.flush()

    pygame.quit()
    subprocess.run(
        ["ffmpeg", "-y",
         "-framerate", str(fps),
         "-i", str(temp_dir / "%04d.png"),
         "-ss", str(start_sec), "-i", str(audio_path),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k",
         "-shortest",
         str(output_mp4)],
        capture_output=True,
    )
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--start",    type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--bars",     type=int,   default=None,
                        help="Render N full bars (overrides --duration)")
    parser.add_argument("--subdivision", default="quarter", choices=list(SUBDIVISIONS),
                        help="Minimum grid resolution for rhythm-aware primitives")
    parser.add_argument("--time-sig", default="4/4",
                        help="Time signature, e.g. 4/4, 3/4, 6/8")
    parser.add_argument("--midi",        default=None,
                        help="MIDI file: overrides audio-based beat tracking")
    parser.add_argument("--midi-offset", type=float, default=0.0,
                        help="Shift MIDI events by N seconds (positive = MIDI starts later)")
    parser.add_argument("--scene",       default=None,
                        help="Scene YAML file (default: scenes/default.yaml)")
    parser.add_argument("--stems",       nargs="*", default=None,
                        help="Pre-built stems: name=path [name=path ...] e.g. guitar=input/guitar.wav")
    parser.add_argument("--out",     default=None)
    parser.add_argument("--mode",    default="demucs")
    parser.add_argument("--preset",  default="none", choices=["none", "zigzag"])
    parser.add_argument("--speed",   type=float, default=0.25)
    parser.add_argument("--trail",   type=int,   default=0)
    parser.add_argument("--scale",   type=float, default=1.0)
    args = parser.parse_args()
    if args.out is None:
        args.out = f"render_output/{Path(args.file).stem}_seed.mp4"
    num, den = (int(x) for x in args.time_sig.split("/"))
    generate_animation_mp4(
        args.file, args.out,
        start_sec=args.start, duration_sec=args.duration,
        mode=args.mode, preset=args.preset, speed=args.speed,
        trail_count=args.trail, scale=args.scale,
        bars=args.bars, subdivision=args.subdivision,
        time_signature=(num, den), midi_path=args.midi,
        midi_offset=args.midi_offset, scene_path=args.scene,
        prebuilt_stems=dict(s.split("=", 1) for s in args.stems) if args.stems else None,
    )
