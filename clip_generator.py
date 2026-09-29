#!/usr/bin/env python3
"""Clip-based audio-reactive video generator.

Composes pre-rendered mini-clips (mp4s in a folder) on the musical grid:
by default each bar picks a new clip, and drum hits from the MIDI trigger
transport operations (e.g. kick -> reverse playback, snare -> next clip).
Behaviour is configured in the ``video:`` section of the scene YAML.

Usage examples
--------------
  python clip_generator.py --file audio.mp3 --midi song.mid \
      --clips input/clips --bars 8 --scene scenes/clips_kick_reverse.yaml
"""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import parse_meter_changes, read_midi
from core.video.clip_library import ClipLibrary
from core.video.composer import ClipComposer
from core.video.layers import build_compositor


@dataclass
class Render:
    """Everything a render needs, set up: ``stack.frame_at(start_sec + i / fps)`` for i < n_frames."""
    stack: object
    start_sec: float
    n_frames: int
    fps: int
    width: int
    height: int
    output_path: str
    composer: Optional[object] = None
    library: Optional[object] = None


def span_of_bars(grid, start_sec: float, bars: int, fps: float) -> float:
    """Seconds that ``bars`` bars take from ``start_sec`` (bars are not all the same length
    when the meter changes). A start within half a frame of a downbeat counts as that bar:
    --start-time 13.351 for a bar at 13.3513 would otherwise spend one of the bars on the
    0.3 ms left of the bar before."""
    first_bar = grid.bar_index(start_sec + 0.5 / fps)
    return grid.bar_start(first_bar + bars) - start_sec


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ARC clip compositor")
    ap.add_argument("--file",        required=True,  help="Audio file (mp3/wav/flac)")
    ap.add_argument("--clips",       default=None,   help="Folder of pre-rendered video clips (optional for pure-generative scenes)")
    ap.add_argument("--midi",        default=None,   help="MIDI file for rhythm grid + drum triggers")
    ap.add_argument("--scene",       default="scenes/clips_kick_reverse.yaml")
    ap.add_argument("--bars",        type=int,   default=8,
                    help="Bars to render; 0 = whole song (until audio ends)")
    ap.add_argument("--cache-size",  type=int,   default=4,
                    help="Decoded clips kept in RAM (raise for long renders that cycle many clips)")
    ap.add_argument("--start-time",  type=float, default=0.0, help="Start time in seconds (default: first downbeat)")
    ap.add_argument("--fps",         type=int,   default=24)
    ap.add_argument("--resolution",  default="854x480", help="WxH pixels")
    ap.add_argument("--output",      default=None,   help="Output MP4 path")
    ap.add_argument("--midi-offset", type=float, default=0.0)
    ap.add_argument("--clip-fit", choices=["contain", "cover"], default="contain",
                    help="contain: letterbox clips of another aspect; cover: fill the frame and crop")
    ap.add_argument("--clip-turn-wide", action="store_true",
                    help="Turn landscape clips 90 degrees clockwise (for a portrait render)")
    ap.add_argument("--stems", action="store_true",
                    help="Separate stems (demucs, GPU; cached in stems_output/) so layers can read features['stems']")
    ap.add_argument("--meter", default=None, metavar="BAR:N/D,...",
                    help="Meter changes the MIDI file does not carry, by DAW bar number, e.g. 22:6/4,27:5/4")
    ap.add_argument("--gravity-peak",   type=float, default=None,
                    help="Override gravity peak speed for all triggers in the scene")
    ap.add_argument("--gravity-floor",  type=float, default=None,
                    help="Override gravity floor speed")
    ap.add_argument("--gravity-radius", type=float, default=None,
                    help="Override gravity influence radius (seconds)")
    ap.add_argument("--gravity-curve",  type=float, default=None,
                    help="Override gravity falloff curve exponent")
    ap.add_argument("--clip-order", choices=["sequential", "random", "shuffle"],
                    default=None, help="Override the scene's clip selection order")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override the scene's shuffle/random seed (reproducible order)")
    ap.add_argument("--masks", default=None, metavar="FOLDER",
                    help="Mask track (segment_video.py) for layers whose mask: names none")
    ap.add_argument("--codec", choices=["x264", "nvenc"], default="x264",
                    help="Encoder: nvenc = GPU (NVIDIA), RAM baixa e rapido em 4K")
    ap.add_argument("--profile", action="store_true",
                    help="Print how many ms each layer takes per frame (work, blend, mask)")
    return ap


def prepare(args) -> Render:
    """Read the song, the MIDI and the scene; build the layer stack. No frame is rendered."""
    W, H = (int(x) for x in args.resolution.split("x"))
    fps = args.fps

    midi_notes = []
    if args.midi:
        print(f"Reading MIDI: {args.midi}")
        grid, midi_notes = read_midi(
            args.midi, fps=fps,
            meter_changes=parse_meter_changes(args.meter) if args.meter else None)
        if args.midi_offset != 0.0:
            shift = args.midi_offset
            if grid.beats is not None:
                grid.beats = grid.beats + shift
            if grid.downbeats is not None:
                grid.downbeats = grid.downbeats + shift
            grid.start_offset += shift
            for n in midi_notes:
                n.time += shift
    else:
        grid = RhythmGrid(bpm=120.0, fps=fps)

    with open(args.scene, "r", encoding="utf-8") as f:
        scene = yaml.safe_load(f) or {}
    video_cfg = scene.get("video", {})

    if args.clip_order:
        video_cfg["clip_order"] = args.clip_order
        print(f"clip order override: {args.clip_order}")
    if args.seed is not None:
        video_cfg["seed"] = args.seed
        print(f"seed override: {args.seed}")

    overrides = {"peak": args.gravity_peak, "floor": args.gravity_floor,
                 "radius": args.gravity_radius, "curve": args.gravity_curve}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        for name, spec in video_cfg.get("triggers", {}).items():
            if "gravity" in spec:
                spec["gravity"].update(overrides)
                print(f"gravity override on {name!r}: {spec['gravity']}")

    layers_cfg = video_cfg.get("layers")
    uses_clips = not layers_cfg or any(
        (l or {}).get("source", "clips") == "clips" for l in layers_cfg)
    library = composer = None
    if uses_clips:
        if not args.clips:
            raise SystemExit("--clips is required (this scene uses clip layers)")
        print(f"Loading clips from {args.clips}")
        library = ClipLibrary(args.clips, W, H, fps, cache_size=args.cache_size, fit=args.clip_fit,
                              turn_wide=args.clip_turn_wide)
        composer = ClipComposer(library, grid, midi_notes, video_cfg)

    # generator and post-op layers need per-frame audio features
    features_at = None
    if any((l or {}).get("source", "clips") not in ("clips", "solid")
           for l in video_cfg.get("layers") or []):
        from core.feature_extractor import AudioFeatureExtractor
        print("Extracting audio features for layers…")
        extractor = AudioFeatureExtractor(args.file, fps=fps, skip_separation=not args.stems)
        features_at = lambda t: extractor.get_features_at_time(t, apply_gate=False)

    stack = build_compositor(composer, video_cfg, midi_notes, W, H,
                             fps=fps, features_at=features_at, grid=grid, masks=args.masks)
    if len(stack) > 1:
        print(f"layers: {len(stack)} (compositing enabled)")

    start_sec = float(grid.start_offset) if grid.start_offset else 0.0
    if args.start_time > 0.0:
        start_sec = args.start_time
    if composer is not None:
        composer.seek(start_sec)

    if args.bars > 0:
        total_dur = span_of_bars(grid, start_sec, args.bars, fps)
    else:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(args.file)],
            capture_output=True, text=True, check=True,
        )
        total_dur = float(probe.stdout.strip()) - start_sec
        print(f"Full-song mode: rendering {total_dur:.1f}s of audio")
    n_frames  = int(total_dur * fps)
    triggers  = list(video_cfg.get("triggers", {}).keys())

    sig = grid.time_signature
    print(
        f"BPM={grid.bpm:.1f}  sig={sig[0]}/{sig[1]}  bar={grid.bar_duration:.2f}s  "
        f"start={start_sec:.2f}s  frames={n_frames}  "
        f"clips={len(library) if library else 0}  triggers={triggers}  "
        f"events={len(composer.events) if composer else 0}"
    )

    stem        = Path(args.file).stem
    output_path = args.output or f"render_output/{stem}_clips.mp4"
    return Render(stack, start_sec, n_frames, fps, W, H, output_path, composer, library)


def main() -> None:
    args = build_parser().parse_args()
    r = prepare(args)
    stack, composer, library = r.stack, r.composer, r.library
    W, H, fps, n_frames, start_sec, output_path = r.width, r.height, r.fps, r.n_frames, r.start_sec, r.output_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if args.profile:
        stack.profile = {}
    t_render = t_write = 0.0
    cuts, shown = [], None            # frames where the clip changes: written beside the video

    # Encode by piping raw frames straight into ffmpeg (no temp files).
    enc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-ss", str(start_sec), "-i", str(args.file),
            "-map", "0:v", "-map", "1:a",
            *(["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
               "-cq", "19", "-b:v", "0"] if args.codec == "nvenc"
              else ["-c:v", "libx264", "-crf", "18"]),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            output_path,
        ],
        stdin=subprocess.PIPE,
    )

    try:
        for fi in range(n_frames):
            t = start_sec + fi / fps
            t0 = time.perf_counter()
            frame = stack.frame_at(t)
            if composer is not None:
                if shown is not None and composer.transport.clip_idx != shown:
                    cuts.append(fi)
                shown = composer.transport.clip_idx
            t1 = time.perf_counter()
            enc.stdin.write(frame.tobytes())
            t_render += t1 - t0
            t_write += time.perf_counter() - t1
            if fi % fps == 0:
                if composer is not None:
                    tp = composer.transport
                    print(
                        f"  {fi:5d}/{n_frames}  t={t:.1f}s  bar={composer._bar_index(t)}  "
                        f"clip={tp.clip_idx} ({library.paths[tp.clip_idx].name})  "
                        f"dir={'>>' if tp.direction > 0 else '<<'}  pos={tp.pos:.0f}"
                    )
                else:
                    print(f"  {fi:5d}/{n_frames}  t={t:.1f}s")
    finally:
        enc.stdin.close()
        enc.wait()

    if enc.returncode != 0:
        raise SystemExit(f"ffmpeg encoding failed (exit {enc.returncode})")
    if args.profile and n_frames:
        print(stack.profile_report(n_frames))
        print(f"  whole frame: {1000 * t_render / n_frames:8.1f} ms   "
              f"to the encoder: {1000 * t_write / n_frames:6.1f} ms")
    print(f"\nDone! -> {output_path}")
    if composer is not None:
        write_cuts(output_path, cuts, fps, start_sec, n_frames)


def cuts_path(video: str) -> Path:
    """Where a render keeps its cuts: ``<video stem>_cuts.json`` beside it."""
    p = Path(video)
    return p.with_name(p.stem + "_cuts.json")


def write_cuts(video: str, cuts, fps: float, start_sec: float, n_frames: int) -> None:
    """The frames where the render changed clip, so segment_video.py knows the shots
    without guessing them from the picture (a strobing clip looks like a cut every flash)."""
    cuts_path(video).write_text(json.dumps({"fps": fps, "start_time": start_sec, "frames": n_frames,
                                            "cuts": [int(c) for c in cuts]}), encoding="utf-8")


if __name__ == "__main__":
    main()
