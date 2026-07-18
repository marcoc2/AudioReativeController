#!/usr/bin/env python3
"""Enhanced audio-reactive particle animation generator (V2).

Uses ParticleSystemV2 for richer visuals:
  • Particle trails / motion blur
  • Gaussian bloom / glow
  • Kick shockwave rings
  • Three particle populations (core, ambient, sparks)
  • Nebula background that breathes with sub-bass
  • Cinematic vignette & Reinhard tonemapping

Drives the same ARCPipeline and accepts the same CLI arguments as the original.

Usage examples
--------------
  python particle_generator_new.py --file audio.mp3 --midi song.mid --bars 16
  python particle_generator_new.py --file audio.mp3 --n-particles 40000 --bars 8
  python particle_generator_new.py --file audio.mp3 --stems solo=guitar.wav --bars 8
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import pygame

from core.feature_extractor import AudioFeatureExtractor
from core.particles_v2 import ParticleSystemV2
from core.pipeline import ARCPipeline
from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import read_midi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_stems(raw: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw:
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ARC particle animation generator (V2 — enhanced visuals)")
    ap.add_argument("--file",         required=True,  help="Audio file (mp3/wav/flac)")
    ap.add_argument("--midi",         default=None,   help="MIDI file for rhythm grid")
    ap.add_argument("--scene",        default="scenes/default.yaml")
    ap.add_argument("--bars",         type=int,   default=8)
    ap.add_argument("--fps",          type=int,   default=24)
    ap.add_argument("--resolution",   default="854x480", help="WxH pixels")
    ap.add_argument("--n-particles",  type=int,   default=20_000, dest="n_particles")
    ap.add_argument("--output",       default=None,   help="Output MP4 path")
    ap.add_argument("--stems",        nargs="*",  default=[], metavar="name=path")
    ap.add_argument("--midi-offset",  type=float, default=0.0,
                    help="Shift all MIDI events by N seconds (+ = MIDI late, - = MIDI early)")
    args = ap.parse_args()

    W, H = (int(x) for x in args.resolution.split("x"))
    fps  = args.fps

    # --- Feature extraction --------------------------------------------
    prebuilt = _parse_stems(args.stems)
    print("Extracting audio features…")
    extractor = AudioFeatureExtractor(
        args.file,
        fps=fps,
        prebuilt_stems=prebuilt or None,
    )
    extractor.update_num_bands(32)

    # --- Rhythm grid ---------------------------------------------------
    if args.midi:
        print(f"Reading MIDI: {args.midi}")
        grid, _ = read_midi(args.midi, fps=fps)
        if args.midi_offset != 0.0:
            shift = args.midi_offset
            if grid.beats is not None:
                grid.beats = grid.beats + shift
            if grid.downbeats is not None:
                grid.downbeats = grid.downbeats + shift
            grid.start_offset += shift
    else:
        grid = RhythmGrid(bpm=120.0, fps=fps)

    # --- Pipeline ------------------------------------------------------
    pipeline = ARCPipeline.from_yaml(extractor, grid, args.scene)

    bar_dur    = pipeline.bar_duration
    start_sec  = float(grid.start_offset) if grid.start_offset else 0.0
    total_dur  = args.bars * bar_dur
    n_frames   = int(total_dur * fps)

    print(
        f"BPM={grid.bpm:.1f}  bar={bar_dur:.2f}s  "
        f"start={start_sec:.2f}s  frames={n_frames}  "
        f"particles={args.n_particles:,}"
    )

    # --- Particle system (V2 enhanced) ---------------------------------
    particles = ParticleSystemV2(n=args.n_particles)

    # --- Render loop ---------------------------------------------------
    pygame.init()
    surface    = pygame.Surface((W, H))
    frame_dir  = Path(tempfile.mkdtemp(prefix="arc_particles_v2_"))
    print(f"Writing frames to {frame_dir}")

    for fi in range(n_frames):
        t  = start_sec + fi / fps
        sl = pipeline.query(t, frame_idx=fi)
        if sl is None:
            print(f"  pipeline returned None at t={t:.2f}s — stopping.")
            n_frames = fi
            break

        c  = sl.controls
        dt = 1.0 / fps

        particles.step(
            dt=dt,
            bass=float(c.get("bass_energy",   0.0)),
            flux=float(c.get("flux",           0.0)),
            beat_phase=sl.beat_phase,
            kick=float(c.get("kick_intensity", 0.0)),
            solo=float(c.get("solo_intensity", 0.0)),
        )

        bass_bg = (
            float(c.get("sub_bass",    0.0)) * 0.65
            + float(c.get("bass_energy", 0.0)) * 0.35
        )
        particles.render(
            surface,
            centroid=float(c.get("centroid", 0.5)),
            bar_phase=sl.bar_phase,
            bass_bg=bass_bg,
        )

        # BMP is uncompressed → much faster than PNG for temp frames
        pygame.image.save(surface, str(frame_dir / f"frame_{fi:05d}.bmp"))

        if fi % fps == 0:
            print(
                f"  {fi:5d}/{n_frames}  t={t:.1f}s  "
                f"bass={c.get('bass_energy', 0):.2f}  "
                f"flux={c.get('flux', 0):.2f}  "
                f"kick={c.get('kick_intensity', 0):.2f}  "
                f"shockwaves={len(particles.shockwaves)}"
            )

    # --- Assemble video ------------------------------------------------
    stem        = Path(args.file).stem
    output_path = args.output or f"render_output/{stem}_particles_v2.mp4"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("Assembling video with ffmpeg…")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%05d.bmp"),
            "-ss", str(start_sec),
            "-i", str(args.file),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            output_path,
        ],
        check=True,
    )
    print(f"\nDone! -> {output_path}")


if __name__ == "__main__":
    main()
