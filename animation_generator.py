import os
import sys
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
    from core.rhythm.analyzer import analyze
    from core.rhythm.grid import SUBDIVISIONS
    from core.rhythm.midi_reader import read_midi
    from core.visual import Frame, VisualObject, render_frame
except ImportError:
    print("[Animation Gen] Error: Required modules not found.")
    sys.exit(1)


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
    frame_idx: int,
    time_sec: float,
    features: dict,
    grid,
    midi_notes: list,
    kick_times: np.ndarray,
    kick_vels: np.ndarray,
    snare_times: np.ndarray,
    snare_vels: np.ndarray,
    l_y_px: int,
    r_y_px: int,
    resolution: tuple,
    fps: int,
) -> Frame:
    """Produce a Frame from current audio features and grid state.

    All continuous quantities drive the VisualObjects produced here.
    Discrete rendering decisions (downbeat ring, etc.) are derived from
    bar_phase — no boolean flags stored on Frame.
    """
    W, H = resolution
    bar_phase  = grid.phase(time_sec)
    beat_phase = grid.beat_phase(time_sec)
    breath     = 1.0 + 0.15 * np.cos(2 * np.pi * bar_phase)

    if midi_notes:
        l_amp = _velocity_envelope(kick_times,  kick_vels,  time_sec)
        r_amp = _velocity_envelope(snare_times, snare_vels, time_sec)
    else:
        l_amp = min(1.0, features.get("bass",  0) * 1.2)
        r_amp = features["stems"].get("vocals", 0)

    l_r = l_amp * breath
    r_r = r_amp * breath

    lx, ly = 0.25, l_y_px / H
    rx, ry = 0.75, r_y_px / H

    objects: list[VisualObject] = []

    if l_r > 0.01:
        objects.append(VisualObject(id="kick",      x=lx, y=ly, radius=l_r,       color=(255,  50,  50)))
        objects.append(VisualObject(id="kick_core", x=lx, y=ly, radius=l_r * 0.3, color=(255, 150, 150), in_trail=False))

    if r_r > 0.01:
        objects.append(VisualObject(id="snare",      x=rx, y=ry, radius=r_r,       color=(255, 180,  50)))
        objects.append(VisualObject(id="snare_core", x=rx, y=ry, radius=r_r * 0.3, color=(255, 230, 200), in_trail=False))

    # Downbeat ring: derived from bar_phase (continuous), not a stored boolean.
    # Threshold ≈ half a frame's worth of bar, so it fires for ~1 frame.
    downbeat_thresh = 0.5 / fps / max(grid.bar_duration, 1e-6)
    if bar_phase < downbeat_thresh or bar_phase > (1.0 - downbeat_thresh):
        objects.append(VisualObject(id="ring_left",  x=lx, y=ly, radius=1.1, color=(255, 255, 255), filled=False, ring_width=4, in_trail=False))
        objects.append(VisualObject(id="ring_right", x=rx, y=ry, radius=1.1, color=(255, 255, 255), filled=False, ring_width=4, in_trail=False))

    return Frame(frame_idx=frame_idx, t=time_sec,
                 bar_phase=bar_phase, beat_phase=beat_phase,
                 objects=objects)


def generate_animation_mp4(audio_path, output_mp4, start_sec=0.0, duration_sec=10.0, fps=24,
                           resolution=(1024, 1024), contrast=0.45, mode="demucs",
                           preset="none", speed=0.25, trail_count=0, scale=1.0,
                           bars=None, subdivision="quarter", time_signature=(4, 4),
                           midi_path=None):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        return
    output_path = Path(output_mp4)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n[Animation Gen] STARTING OFFLINE RENDER\n  Input: {audio_path.name}\n{'='*60}\n")
    extractor = AudioFeatureExtractor(str(audio_path), fps=fps, separation_mode=mode)
    extractor.contrast_level = contrast
    extractor.update_num_bands(32)

    midi_notes = []
    if midi_path:
        grid, midi_notes = read_midi(midi_path, time_signature=time_signature, fps=fps)
        print(f"[Rhythm/MIDI] {Path(midi_path).name}  notes={len(midi_notes)}")
    else:
        grid = analyze(extractor.y, sr=extractor.sample_rate,
                       time_signature=time_signature, fps=fps)
    subdiv_step = grid.subdivision_duration(subdivision)
    print(f"[Rhythm] BPM={grid.bpm:.1f}  bar={grid.bar_duration:.2f}s  "
          f"subdivision={subdivision} ({subdiv_step:.3f}s)  "
          f"beats={len(grid.beats) if grid.beats is not None else 0}")

    KICK, SNARE = 36, 38
    kick_times  = np.array([n.time     for n in midi_notes if n.pitch == KICK],  dtype=float)
    kick_vels   = np.array([n.velocity for n in midi_notes if n.pitch == KICK],  dtype=float)
    snare_times = np.array([n.time     for n in midi_notes if n.pitch == SNARE], dtype=float)
    snare_vels  = np.array([n.velocity for n in midi_notes if n.pitch == SNARE], dtype=float)

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

        features = extractor.get_features_at_time(time_sec, apply_gate=False)
        if not features:
            break

        frame = build_frame(
            frame_idx, time_sec, features, grid,
            midi_notes, kick_times, kick_vels, snare_times, snare_vels,
            l_y_px, r_y_px, resolution, fps,
        )

        surface.fill((0, 0, 0))
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
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(temp_dir / "%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
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
    parser.add_argument("--midi",    default=None,
                        help="MIDI file: overrides audio-based beat tracking")
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
    )
