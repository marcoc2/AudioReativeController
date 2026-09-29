"""Split a video into its objects with SAM2 — the mask track (core/segment/mask_track.py).

  python segment_video.py --video render_output/esfolado_gotas_follow_c6-9_base.mp4 \\
      --start-time 13.351 --sam2-weights F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/models/sam2 \\
      --preview

writes ``render_output/esfolado_gotas_follow_c6-9_base_segments/`` (and, with --preview,
``..._segments_preview.mp4``: every object tinted and numbered). ``--start-time`` is where
the video starts in the song (the value given to clip_generator.py), so effects can ask
for the masks by song time. ``--sam2-weights`` may also come from ARC_SAM2_WEIGHTS.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.segment.mask_track import default_folder, write_mask_track
from core.video.frames import iter_frames, open_encoder, probe_video


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--output", default=None, help="Folder (default: <video stem>_segments next to it)")
    ap.add_argument("--start-time", type=float, default=0.0, help="Where the video starts in the song (s)")
    ap.add_argument("--sam2-weights", default=os.environ.get("ARC_SAM2_WEIGHTS"))
    ap.add_argument("--sam2-model", default="base_plus", choices=["tiny", "small", "base_plus", "large"])
    ap.add_argument("--max-objects", type=int, default=8, help="Objects followed at once")
    ap.add_argument("--min-share", type=float, default=0.005, help="Smallest object (share of the frame)")
    ap.add_argument("--max-share", type=float, default=0.6, help="Largest object (share of the frame)")
    ap.add_argument("--granularity", choices=["whole", "parts"], default="whole",
                    help="whole: characters and things; parts: their pieces (eyes, hands...)")
    ap.add_argument("--reseed", type=float, default=2.0, help="Look for new objects every N s (0: only at cuts)")
    ap.add_argument("--cut", type=float, default=40.0, help="Picture change that starts a new shot (0..255)")
    ap.add_argument("--cuts", default="auto",
                    help="auto: the cuts the render wrote beside the video (<stem>_cuts.json), if any, "
                         "else found from the picture | detect: always from the picture | a _cuts.json path")
    ap.add_argument("--points", type=int, default=32, help="SAM2 automatic grid: points per side")
    ap.add_argument("--min-iou", type=float, default=0.6,
                    help="SAM2's own score a found object needs (lower: more, rougher objects)")
    ap.add_argument("--min-stability", type=float, default=0.7,
                    help="How steady its outline must be (lower: more, rougher objects)")
    ap.add_argument("--preview", action="store_true", help="Also write a video with the objects tinted")
    args = ap.parse_args()
    if not args.sam2_weights:
        ap.error("give --sam2-weights (or set ARC_SAM2_WEIGHTS)")

    from core.segment.sam2_tracker import RegionTracker
    W, H, fps = probe_video(args.video)
    out = Path(args.output) if args.output else default_folder(args.video)
    print(f"SAM2 {args.sam2_model} on {args.video} ({W}x{H} @ {fps:g}) -> {out}", flush=True)
    cuts = None
    if args.cuts != "detect":
        path = Path(args.video).with_name(Path(args.video).stem + "_cuts.json") if args.cuts == "auto" \
            else Path(args.cuts)
        if path.is_file():
            cuts = json.loads(path.read_text(encoding="utf-8"))["cuts"]
            print(f"cuts from {path}: {len(cuts)}", flush=True)
        elif args.cuts != "auto":
            ap.error(f"no cuts file {path}")
    tracker = RegionTracker(args.sam2_weights, args.sam2_model)
    preview = None
    if args.preview:
        preview = open_encoder(str(out) + "_preview.mp4", W, H, fps, audio_from=args.video, crf=20)
    try:
        write_mask_track(args.video, lambda: iter_frames(args.video), tracker, str(out), fps,
                         start_time=args.start_time, max_objects=args.max_objects,
                         min_share=args.min_share, max_share=args.max_share,
                         granularity=args.granularity, reseed=args.reseed, cut=args.cut,
                         points_per_side=args.points, min_iou=args.min_iou,
                         min_stability=args.min_stability, preview=preview, model=args.sam2_model,
                         log=lambda m: print(m, flush=True), cuts=cuts)
    finally:
        if preview is not None:
            preview.stdin.close()
            preview.wait()


if __name__ == "__main__":
    main()
