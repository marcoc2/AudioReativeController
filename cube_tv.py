"""Cube TV room — standalone generator: clips on floating cubes, heard from the camera.

This is deliberately **not** part of the audio-reactive pipeline. There, sound
is the cause and the picture is the consequence; here it is the other way
round. Each cube is a television carrying one clip, and what you hear is what
the camera sees: a screen filling more of the frame is louder, a screen off to
the left arrives on the left. Nothing about it is reactive, nothing about it
reads MIDI, and `clip_generator.py` neither knows nor cares that it exists.

It reuses the pure pieces the reactive side already owns — `core.cubes`
(the GPU stage), `core.video.clip_library` (decode + centre-crop) — and adds
`core.video.clip_audio` for the sound. The dependency arrow only ever points
inward: nothing in `core/` was taught anything about this script.

How loud is "most visible"? It is measured, not modelled. Every frame the
stage is drawn a second time into a small buffer where each cube paints its
own id; the pixels a cube owns *are* its share of the frame, with occlusion,
facing and off-screen already accounted for (`CubeField.visibility`). No
physics, no inverse-square law, no attempt at a real room — just "how much of
what I see is this TV".

    python cube_tv.py --clips C:/Users/marco/Videos/ltx2/2_3_samples \\
        --duration 20 --resolution 1280x720 --output render_output/cube_tv.mp4
"""
from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", required=True, help="folder of video clips (the TVs' content)")
    p.add_argument("--output", default="render_output/cube_tv.mp4")
    p.add_argument("--duration", type=float, default=20.0, help="seconds to render")
    p.add_argument("--resolution", default="1280x720")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--codec", choices=["libx264", "nvenc"], default="libx264")

    p.add_argument("--cubes", type=int, default=24,
                   help="number of TVs; each gets its own clip (max 254)")
    p.add_argument("--faces", type=int, default=6,
                   help="0..6 faces of each cube showing its clip")
    p.add_argument("--face-resolution", type=int, default=256)
    p.add_argument("--bg", default="0,0,0", help="empty-space colour, r,g,b")
    p.add_argument("--loop-seconds", type=float, default=10.0,
                   help="seconds per corridor pass = how long each TV keeps a clip")
    p.add_argument("--radius-min", type=float, default=1.4)
    p.add_argument("--radius-max", type=float, default=4.5)
    p.add_argument("--corridor", type=float, default=14.0,
                   help="depth of the corridor; shorter = TVs pass closer together")
    p.add_argument("--size-min", type=float, default=1.0)
    p.add_argument("--size-max", type=float, default=2.2)
    p.add_argument("--spin", type=int, default=2,
                   help="whole turns per corridor pass; 0 freezes the cubes")
    p.add_argument("--dwell-seconds", type=float, default=0.0,
                   help="how long a TV keeps a clip; 0 = one full corridor pass. "
                        "Shorter shows more clips without speeding up the camera")
    p.add_argument("--swap-below", type=float, default=0.004,
                   help="a TV may only change clip while its share of the frame "
                        "is under this — swaps stay out of sight")

    p.add_argument("--no-audio", action="store_true", help="render the picture only")
    p.add_argument("--gain-db", type=float, default=-1.0, help="peak of the final mix")
    p.add_argument("--attack-ms", type=float, default=40.0)
    p.add_argument("--release-ms", type=float, default=220.0)
    p.add_argument("--curve", type=float, default=0.6,
                   help="exponent on the visibility share; <1 lets distant TVs stay audible")
    p.add_argument("--pan-width", type=float, default=0.9, help="0 = mono, 1 = full width")
    return p.parse_args()


def smooth_envelope(w: np.ndarray, fps: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """One-pole attack/release per TV, so a screen entering or leaving the
    frame fades instead of clicking. ``w`` is (frames, cubes)."""
    a = math.exp(-1.0 / max(1e-6, fps * attack_ms / 1000.0))
    r = math.exp(-1.0 / max(1e-6, fps * release_ms / 1000.0))
    out = np.empty_like(w)
    cur = np.zeros(w.shape[1])
    for f in range(w.shape[0]):
        tgt = w[f]
        coef = np.where(tgt > cur, a, r)
        cur = tgt + coef * (cur - tgt)
        out[f] = cur
    return out


class Deck:
    """The running order: every clip dealt once before any is dealt again.

    A 3-minute render cannot show three thousand clips, so which ones it shows
    matters. Drawing without replacement from a seeded shuffle means no cube
    ever repeats what another already played, until the whole folder is spent.
    """

    def __init__(self, n_clips: int, seed=None):
        self.order = np.random.default_rng(seed).permutation(n_clips)
        self.i = 0
        self.laps = 0

    def draw(self) -> int:
        if self.i >= len(self.order):
            self.i, self.laps = 0, self.laps + 1
        v = int(self.order[self.i])
        self.i += 1
        return v


class Mixer:
    """Stereo bed, accumulated a segment at a time.

    A cube's turn with a clip is mixed in the moment that turn ends, which is
    what lets the clip be dropped from memory right after. The alternative —
    tracing the whole render and mixing at the end — would need every clip
    still resident, and that is exactly what does not fit.
    """

    def __init__(self, duration: float, fps: int, sample_rate: int, pan_width: float):
        self.sr = sample_rate
        self.fps = fps
        self.pan_width = pan_width
        self.buf = np.zeros((int(round(duration * sample_rate)), 2), dtype=np.float64)
        self.segments = 0
        self.heard = 0

    def add_segment(self, audio, env, pan, f0: int, f1: int) -> None:
        """Mix one cube showing one clip over frames ``[f0, f1)``."""
        self.segments += 1
        if len(audio) == 0 or f1 <= f0:
            return
        s0 = int(round(f0 / self.fps * self.sr))
        s1 = min(int(round(f1 / self.fps * self.sr)), len(self.buf))
        if s1 <= s0:
            return
        self.heard += 1
        ts = np.arange(s0, s1, dtype=np.float64) / self.sr
        # time since this clip started on this cube; wraps if the turn outlasts it
        idx = ((ts - f0 / self.fps) * self.sr).astype(np.int64) % len(audio)
        sig = audio[idx]
        ft = np.arange(f0, f1, dtype=np.float64) / self.fps
        g = np.interp(ts, ft, env[f0:f1])
        p = np.interp(ts, ft, pan[f0:f1]) * self.pan_width
        ang = (np.clip(p, -1.0, 1.0) + 1.0) * (math.pi / 4.0)   # equal-power pan
        self.buf[s0:s1, 0] += sig[:, 0] * g * np.cos(ang)
        self.buf[s0:s1, 1] += sig[:, 1] * g * np.sin(ang)

    def finish(self, gain_db: float) -> np.ndarray:
        m = float(np.abs(self.buf).max())
        if m > 0:
            self.buf *= (10.0 ** (gain_db / 20.0)) / m
        return self.buf.astype(np.float32)


def render_and_mix(args, field, lib, alib, deck, n, enc_stdin):
    """The single streaming pass: draw, measure, swap, mix, release.

    Each cube keeps one clip for exactly one trip down the corridor, then
    trades it for a fresh one from the deck. The swap is timed to the depth
    wrap, which the stage was already built to hide: a cube reaching the
    camera has left the frustum sideways, so it changes what it shows while
    nobody can see it. That is also the moment its clip can be freed, which is
    what keeps memory flat no matter how many thousand clips stream through.
    """
    n_frames = int(round(args.duration * args.fps))
    bg = [int(c) for c in args.bg.split(",")]
    layers = np.arange(n)
    env = np.zeros((n_frames, n))
    pan = np.zeros((n_frames, n))
    mixer = Mixer(args.duration, args.fps, SAMPLE_RATE, args.pan_width)

    showing = [deck.draw() for _ in range(n)]
    since = [0] * n                      # frame each cube's current turn began
    distinct = set(showing)
    a = math.exp(-1.0 / max(1e-6, args.fps * args.attack_ms / 1000.0))
    r = math.exp(-1.0 / max(1e-6, args.fps * args.release_ms / 1000.0))
    level = np.zeros(n)
    prev_d = None
    prev_share = np.zeros(n)

    for f in range(n_frames):
        t = f / args.fps
        u = t / args.loop_seconds        # unwrapped: no artificial reset at the seam
        d = (field.z0 - u) % 1.0         # 0 = at the camera, 1 = far end
        if prev_d is not None:
            due = d > prev_d + 0.5       # wrapped past the camera: always hidden
            if args.dwell_seconds > 0:
                # or: held it long enough *and* is currently too small to notice
                aged = (f - np.asarray(since)) / args.fps >= args.dwell_seconds
                due = due | (aged & (prev_share < args.swap_below))
            for i in np.flatnonzero(due):
                mixer.add_segment(alib.get(showing[i]), env[:, i], pan[:, i],
                                  since[i], f)
                alib.release(showing[i])
                showing[i] = deck.draw()
                since[i] = f
                distinct.add(showing[i])
        prev_d = d

        screens = []
        for i in range(n):
            clip = lib.get(showing[i])
            screens.append(clip.frame((f - since[i]) % len(clip)))
        enc_stdin.write(field.render(u, screens, bg, cube_layers=layers).tobytes())

        share, p = field.visibility(u)
        prev_share = share
        tgt = share ** args.curve        # peak-normalised later, so scale is free
        level = tgt + np.where(tgt > level, a, r) * (level - tgt)
        env[f], pan[f] = level, p

        if f % (args.fps * 5) == 0:
            print(f"  {f:5d}/{n_frames}  t={t:5.1f}s  na tela={share.sum() * 100:5.1f}%  "
                  f"clipes usados={len(distinct)}")

    for i in range(n):                   # flush the turns still running at the end
        mixer.add_segment(alib.get(showing[i]), env[:, i], pan[:, i], since[i], n_frames)
    return mixer, len(distinct)


def main() -> None:
    args = parse_args()
    W, H = (int(v) for v in args.resolution.lower().split("x"))
    n = max(1, min(254, args.cubes))

    from core.cubes import CubeField
    from core.video.clip_library import ClipLibrary
    from core.video.clip_audio import AudioLibrary

    # cover = centre-crop, so 16:9 footage fills a square screen with no bars
    lib = ClipLibrary(args.clips, args.face_resolution, args.face_resolution,
                      args.fps, cache_size=n + 4, fit="cover")
    alib = AudioLibrary(lib.paths, SAMPLE_RATE, cache_size=n + 4)
    deck = Deck(len(lib), args.seed)

    spec = {
        "n_cubes": n,
        "faces_with_clips": args.faces,
        "face_resolution": args.face_resolution,
        "radius_min": args.radius_min,
        "radius_max": args.radius_max,
        "corridor_depth": args.corridor,
        "size_min": args.size_min,
        "size_max": args.size_max,
        "spin_max": args.spin,
    }
    field = CubeField(W, H, spec, seed=args.seed, n_screens=n)

    turn = args.dwell_seconds if args.dwell_seconds > 0 else args.loop_seconds
    print(f"{n} TVs, pool de {len(lib)} clipes, {W}x{H} @ {args.fps}fps, "
          f"{args.duration:.0f}s\ncorredor a cada {args.loop_seconds:.0f}s, "
          f"cada TV segura um clipe ~{turn:.0f}s -> "
          f"~{int(n * args.duration / turn)} clipes distintos (sem repetir)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="cube_tv_"))
    silent = tmpdir / "video.mp4"

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         *(["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
           if args.codec == "nvenc" else ["-c:v", "libx264", "-crf", "18"]),
         "-pix_fmt", "yuv420p", str(silent)],
        stdin=subprocess.PIPE,
    )
    try:
        mixer, used = render_and_mix(args, field, lib, alib, deck, n, enc.stdin)
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        raise SystemExit(f"ffmpeg encoding failed (exit {enc.returncode})")
    print(f"{used} clipes distintos mostrados; {mixer.heard}/{mixer.segments} "
          f"turnos com áudio" + (f"; baralho deu {deck.laps} volta(s)" if deck.laps else ""))

    if args.no_audio:
        silent.replace(out)
        print(f"\nDone (sem áudio)! -> {out}")
        return

    mix = mixer.finish(args.gain_db)
    raw = tmpdir / "mix.f32"
    raw.write_bytes(mix.tobytes())

    mux = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-i", str(silent),
         "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-i", str(raw),
         "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)],
    )
    if mux.returncode != 0:
        raise SystemExit(f"ffmpeg muxing failed (exit {mux.returncode})")
    print(f"\nDone! -> {out}")


if __name__ == "__main__":
    main()
