"""Drops that follow what they fell on (SAM2) — a two-pass render.

  pass 1  clip_generator.py renders the scene with its drops layer switched off
  pass 2  the drops land on that video; SAM2 follows each drop's region and the paint
          rides along with it (core/video/drops_follow.py)

  python follow_drops.py --scene examples/esfolado_gotas_curadoria.yaml \\
      --file input/esfolado/esfolado_mix_20_09_2026.mp3 \\
      --midi input/esfolado/esfolado.mid --midi-offset 0.018 --start-time 13.351 --fps 30 \\
      --sam2-weights F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/models/sam2 \\
      --clips C:/Users/marco/Videos/ltx2/curadoria --clip-fit cover --seed 3 --bars 4 \\
      --resolution 854x480 --output render_output/esfolado_gotas_follow_c6-9.mp4

Every argument this script does not know goes to clip_generator.py as it is. With
``--base`` pass 1 is skipped and that video is used (it must start at ``--start-time``).
Needs the extras in requirements-segment.txt; ``--sam2-weights`` may also come from the
ARC_SAM2_WEIGHTS environment variable.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from core.rhythm.midi_reader import parse_meter_changes, read_midi, shift_in_time


def _drops_layer(scene: dict, index) -> int:
    layers = (scene.get("video") or {}).get("layers") or []
    found = [i for i, l in enumerate(layers) if (l or {}).get("source") == "drops"]
    if not found:
        raise SystemExit("the scene has no drops layer")
    if index is None:
        return found[0]
    if index not in found:
        raise SystemExit(f"layer {index} is not a drops layer (drops layers: {found})")
    return index


def _probe(video: str):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "csv=p=0", video],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    num, den = out[2].split("/")
    return int(out[0]), int(out[1]), float(num) / float(den)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--file", required=True, help="The song (also goes to clip_generator)")
    ap.add_argument("--midi", default=None)
    ap.add_argument("--midi-offset", type=float, default=0.0)
    ap.add_argument("--meter", default=None, metavar="BAR:N/D,...")
    ap.add_argument("--start-time", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--output", default=None)
    ap.add_argument("--base", default=None, help="Skip pass 1: the scene already rendered without drops")
    ap.add_argument("--layer", type=int, default=None, help="Which drops layer (index in layers:)")
    ap.add_argument("--sam2-weights", default=os.environ.get("ARC_SAM2_WEIGHTS"),
                    help="Folder with the SAM 2.1 checkpoint (.pt or .safetensors)")
    ap.add_argument("--sam2-model", default="base_plus", choices=["tiny", "small", "base_plus", "large"])
    args, passthrough = ap.parse_known_args()
    if not args.sam2_weights:
        ap.error("give --sam2-weights (or set ARC_SAM2_WEIGHTS)")

    scene = yaml.safe_load(Path(args.scene).read_text(encoding="utf-8")) or {}
    li = _drops_layer(scene, args.layer)
    spec = dict(scene["video"]["layers"][li])
    out = Path(args.output or f"render_output/{Path(args.scene).stem}_follow.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: the scene without its drops
    base = args.base
    if base is None:
        base = str(out.with_name(out.stem + "_base.mp4"))
        scene["video"]["layers"][li] = {**spec, "enabled": False}
        base_scene = out.with_name(out.stem + "_base_scene.yaml")
        base_scene.write_text(yaml.safe_dump(scene, allow_unicode=True, sort_keys=False), encoding="utf-8")
        cmd = [sys.executable, "clip_generator.py", "--scene", str(base_scene), "--file", args.file,
               "--midi-offset", str(args.midi_offset), "--start-time", str(args.start_time),
               "--fps", str(args.fps), "--output", base, *passthrough]
        if args.midi:
            cmd += ["--midi", args.midi]
        if args.meter:
            cmd += ["--meter", args.meter]
        print("pass 1:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    # ---- pass 2: the drops, followed by SAM2
    notes, grid = [], None
    if args.midi:
        grid, notes = read_midi(args.midi, meter_changes=parse_meter_changes(args.meter) if args.meter else None)
        shift_in_time(grid, notes, args.midi_offset)
    start = args.start_time or (float(grid.start_offset) if grid is not None and grid.start_offset else 0.0)
    W, H, fps = _probe(base)

    from core.segment import RegionTracker
    from core.video.drops_follow import FollowedDrops
    print(f"pass 2: SAM2 {args.sam2_model} on {base} ({W}x{H} @ {fps:g})", flush=True)
    tracker = RegionTracker(args.sam2_weights, args.sam2_model)
    drops = FollowedDrops(spec, notes, W, H, fps, tracker, grid=grid)

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", base, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-", "-i", base,
                            "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
                            "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(out)],
                           stdin=subprocess.PIPE)
    import numpy as np
    size, i, t0 = W * H * 3, 0, time.time()
    try:
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            enc.stdin.write(drops.process(frame, start + i / fps).tobytes())
            i += 1
            if i % int(round(fps)) == 0:
                print(f"  {i:5d} frames  t={start + i / fps:7.2f}s  live regions={tracker.live}  "
                      f"{i / (time.time() - t0):.1f} fps", flush=True)
    finally:
        enc.stdin.close()
        enc.wait()
        dec.stdout.close()
        dec.wait()
    s = drops.stats
    print(f"done: {out}  ({i} frames, {s['drops']} drops: {s['mask_prompts']} from their region, "
          f"{s['point_prompts']} from a point)")


if __name__ == "__main__":
    main()
