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
import subprocess
from pathlib import Path

import yaml

from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import read_midi
from core.video.clip_library import ClipLibrary
from core.video.composer import ClipComposer


def main() -> None:
    ap = argparse.ArgumentParser(description="ARC clip compositor")
    ap.add_argument("--file",        required=True,  help="Audio file (mp3/wav/flac)")
    ap.add_argument("--clips",       required=True,  help="Folder of pre-rendered video clips")
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
    args = ap.parse_args()

    W, H = (int(x) for x in args.resolution.split("x"))
    fps = args.fps

    midi_notes = []
    if args.midi:
        print(f"Reading MIDI: {args.midi}")
        grid, midi_notes = read_midi(args.midi, fps=fps)
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

    overrides = {"peak": args.gravity_peak, "floor": args.gravity_floor,
                 "radius": args.gravity_radius, "curve": args.gravity_curve}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        for name, spec in video_cfg.get("triggers", {}).items():
            if "gravity" in spec:
                spec["gravity"].update(overrides)
                print(f"gravity override on {name!r}: {spec['gravity']}")

    print(f"Loading clips from {args.clips}")
    library = ClipLibrary(args.clips, W, H, fps, cache_size=args.cache_size)
    composer = ClipComposer(library, grid, midi_notes, video_cfg)

    start_sec = float(grid.start_offset) if grid.start_offset else 0.0
    if args.start_time > 0.0:
        start_sec = args.start_time
    composer.seek(start_sec)

    if args.bars > 0:
        total_dur = args.bars * grid.bar_duration
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

    print(
        f"BPM={grid.bpm:.1f}  bar={grid.bar_duration:.2f}s  start={start_sec:.2f}s  "
        f"frames={n_frames}  clips={len(library)}  triggers={triggers}  "
        f"events={len(composer.events)}"
    )

    stem        = Path(args.file).stem
    output_path = args.output or f"render_output/{stem}_clips.mp4"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Encode by piping raw frames straight into ffmpeg (no temp files).
    enc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-ss", str(start_sec), "-i", str(args.file),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            output_path,
        ],
        stdin=subprocess.PIPE,
    )

    try:
        for fi in range(n_frames):
            t = start_sec + fi / fps
            frame = composer.frame_at(t)
            enc.stdin.write(frame.tobytes())
            if fi % fps == 0:
                tp = composer.transport
                print(
                    f"  {fi:5d}/{n_frames}  t={t:.1f}s  bar={composer._bar_index(t)}  "
                    f"clip={tp.clip_idx} ({library.paths[tp.clip_idx].name})  "
                    f"dir={'>>' if tp.direction > 0 else '<<'}  pos={tp.pos:.0f}"
                )
    finally:
        enc.stdin.close()
        enc.wait()

    if enc.returncode != 0:
        raise SystemExit(f"ffmpeg encoding failed (exit {enc.returncode})")
    print(f"\nDone! -> {output_path}")


if __name__ == "__main__":
    main()
