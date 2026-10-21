"""Repaint an ARC render frame by frame through a ComfyUI img2img workflow, driven by the music.

  python repaint.py --video render_output/x.mp4 --scene examples/esfolado_repaint.yaml \\
      --midi input/esfolado/esfolado.mid --midi-offset 0.018 --start-time 10.685 \\
      --output render_output/x_repaint.mp4

Without a song (any video): ``--workflow`` and ``--prompt`` alone are enough, e.g.

  python repaint.py --video input/clip.mp4 --workflow reference/workflow_api_edit.json \
      --prompt "make it claymation" --fps 12

``--start-time`` is where the source render starts in the song (the same value given to
clip_generator.py), so the kicks and bars line up. The scene's ``repaint:`` block holds the
workflow, fps, denoise/kick, seed and prompts (see core/video/repaint.py).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from core.rhythm.midi_reader import parse_meter_changes, read_midi, shift_in_time
from core.video.repaint import Workflow, plan_frames, repaint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="The ARC render to repaint")
    ap.add_argument("--scene", default=None, help="YAML with a repaint: block")
    ap.add_argument("--midi", default=None, help="The song's MIDI (bars and kicks)")
    ap.add_argument("--workflow", default=None, help="Override the scene's ComfyUI workflow")
    ap.add_argument("--prompt", default=None, help="One prompt for every frame")
    ap.add_argument("--midi-offset", type=float, default=0.0)
    ap.add_argument("--meter", default=None, metavar="BAR:N/D,...")
    ap.add_argument("--start-time", type=float, default=0.0, help="Where the video starts in the song (s)")
    ap.add_argument("--seconds", type=float, default=None, help="Only the first N seconds of the video")
    ap.add_argument("--fps", type=float, default=None, help="Override the scene's fps")
    ap.add_argument("--output", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Print the plan, send nothing")
    args = ap.parse_args()

    cfg = {}
    if args.scene:
        cfg = (yaml.safe_load(Path(args.scene).read_text(encoding="utf-8")) or {})["repaint"]
    if args.workflow:
        cfg["workflow"] = args.workflow
    if args.prompt:
        cfg["prompts"] = [args.prompt]
    if "workflow" not in cfg:
        ap.error("give --scene or --workflow")
    if args.fps:
        cfg["fps"] = args.fps
    fps = float(cfg.get("fps", 12))

    bar_of, notes = (lambda t: 0), []
    if args.midi:
        grid, notes = read_midi(args.midi, meter_changes=parse_meter_changes(args.meter) if args.meter else None)
        shift_in_time(grid, notes, args.midi_offset)
        bar_of = grid.bar_index
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", args.video],
        capture_output=True, text=True, check=True).stdout.strip())
    if args.seconds:
        duration = min(duration, args.seconds)

    plan = plan_frames(duration, fps, args.start_time, bar_of, cfg, notes)
    if args.dry_run:
        last = None
        for p in plan:
            if p.prompt != last:
                print(f"bar {p.bar + 1}: {p.prompt}")
                last = p.prompt
            print(f"  f{p.index:04d}  t={p.t_song:7.3f}  denoise={p.denoise:.2f}  seed={p.seed}")
        return

    workflow = Workflow.load_file(cfg["workflow"])
    out = args.output or str(Path(args.video).with_name(Path(args.video).stem + "_repaint.mp4"))
    repaint(args.video, out, workflow, plan, cfg)


if __name__ == "__main__":
    main()
