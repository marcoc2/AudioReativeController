#!/usr/bin/env python3
"""H3 Scene Generator — Procedural Scene & Clip-Farm Generator for ARC.

Analyzes an audio file and musical grid, builds mathematically aligned
MiniMax H3 prompts (cuts on downbeats, camera energy mapped to tempo/features),
and generates a batch of cinematic video clips via ComfyUI for use in ARC's
ClipComposer (`clip_generator.py`).

Usage:
  python h3_scene_generator.py --file audio.mp3 --midi song.mid --theme "cyberpunk crystal jungle" --scenes 4 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import read_midi
from core.video.h3_client import ComfyH3Client, find_workflow_path
from core.video.h3_prompt_builder import (
    build_h3_fl2va_prompt,
    build_h3_t2va_prompt,
    format_h3_timestamp,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="ARC H3 Scene & Clip-Farm Generator")
    ap.add_argument("--file", required=True, help="Audio file (mp3/wav/flac)")
    ap.add_argument("--midi", default=None, help="MIDI file for exact downbeats and tempo")
    ap.add_argument("--bpm", type=float, default=120.0, help="Fallback BPM if no MIDI is provided")
    ap.add_argument("--theme", default="psychedelic abstract bioluminescent ecosystem", help="Visual subject theme")
    ap.add_argument("--style", default="Cinematic", help="Visual style specification (Cinematic, 2D-animated, etc.)")
    ap.add_argument("--scenes", type=int, default=4, help="Number of scene clips to generate")
    ap.add_argument("--bars-per-scene", type=int, default=4, help="Musical bars per scene (determines shot cuts)")
    ap.add_argument("--energy", type=float, default=0.6, help="Visual/motion energy level (0.0 to 1.0)")
    ap.add_argument("--clips-out", default="input/clips/h3_pool", help="Folder to save generated clips")
    ap.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    ap.add_argument("--workflow", default=None, help="Custom workflow_api_h3_t2v_turbo.json path")
    ap.add_argument("--dry-run", action="store_true", help="Print generated prompts and timing without running ComfyUI")
    ap.add_argument("--flf-bridge", action="store_true", help="Generate FL2VA morphing bridges between consecutive clips")

    args = ap.parse_args()

    # 1. Read musical rhythm grid
    if args.midi and Path(args.midi).is_file():
        print(f"[H3 Generator] Reading MIDI: {args.midi}")
        grid, _ = read_midi(args.midi, fps=24)
    else:
        print(f"[H3 Generator] Building fixed RhythmGrid with BPM={args.bpm}")
        grid = RhythmGrid(bpm=args.bpm, fps=24)

    bar_sec = grid.bar_duration
    scene_sec = args.bars_per_scene * bar_sec
    print(f"[H3 Generator] BPM: {grid.bpm:.1f} | 1 Bar: {bar_sec:.3f}s | Scene Duration: {scene_sec:.2f}s ({args.bars_per_scene} bars)")

    client = ComfyH3Client(server_url=args.server)
    out_dir = Path(args.clips_out)

    if not args.dry_run:
        if not client.is_available():
            print(f"[H3 Generator] ERROR: ComfyUI server at {args.server} is not reachable.", file=sys.stderr)
            print("[H3 Generator] Run with --dry-run to test prompt generation without a running ComfyUI.", file=sys.stderr)
            sys.exit(1)
        out_dir.mkdir(parents=True, exist_ok=True)

    generated_clips = []

    # 2. Generate prompts and clips per scene
    for s_idx in range(args.scenes):
        start_bar = s_idx * args.bars_per_scene
        prompt_text = build_h3_t2va_prompt(
            grid=grid,
            start_bar=start_bar,
            num_bars=args.bars_per_scene,
            theme=args.theme,
            energy=args.energy,
            style=args.style,
        )

        print(f"\n==================================================")
        print(f"--- SCENE {s_idx + 1}/{args.scenes} (Bars {start_bar} to {start_bar + args.bars_per_scene}) ---")
        print(f"==================================================")
        print(prompt_text)

        if args.dry_run:
            continue

        target_file = out_dir / f"h3_scene_{s_idx + 1:02d}.mp4"
        print(f"\n[H3 Generator] Enqueuing to ComfyUI (H3 T2V Turbo) -> {target_file}...")
        res_path = client.generate_t2v_scene(
            prompt_text=prompt_text,
            duration_sec=scene_sec,
            seed=1000 + s_idx * 17,
            workflow_template_path=args.workflow,
            output_path=target_file,
        )
        print(f"[H3 Generator] Saved clip: {res_path}")
        generated_clips.append(res_path)

    if args.dry_run:
        print("\n[H3 Generator] Dry run complete. All prompts and cut calculations verified.")
        return

    print(f"\n[H3 Generator] Successfully generated {len(generated_clips)} scene clips in '{out_dir}'.")
    print(f"[H3 Generator] You can now composite them directly with ARC:")
    print(f"  python clip_generator.py --file {args.file} --clips {out_dir} --bars {args.scenes * args.bars_per_scene}")


if __name__ == "__main__":
    main()
