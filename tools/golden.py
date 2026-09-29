"""Golden frames — the example scenes rendered once and kept, so a change in the compositor
can be checked against what the scenes looked like before it.

  python tools/golden.py make            # render every case, keep the frames
  python tools/golden.py check           # render again, compare with what was kept
  python tools/golden.py check --only esfolado_cast world_melts

Frames go to render_output/golden/<case>/NNNN.png (lossless) with cases.json beside them.
The cases use the songs, clips and mask tracks of this machine (not in the repository); a
case whose inputs are missing is skipped and said so. A check passes when the frames match
within ``--tolerance`` levels (0..255) on all but ``--allow`` of the pixels: GPU and CPU
maths round differently, so bit-identical is not the bar.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = Path("render_output/golden")
SIZE, FPS = "854x480", 24
KEEP = (0, 12, 24, 36, 47)     # the frames kept; all before them are rendered, in order
                               # (effects with memory need the run-up)

ESFOLADO = ["--file", "input/esfolado/esfolado_mix_20_09_2026.mp3", "--midi", "input/esfolado/esfolado.mid",
            "--midi-offset", "0.018", "--start-time", "13.351"]
CURADORIA = ["--clips", "C:/Users/marco/Videos/ltx2/curadoria", "--clip-fit", "cover", "--seed", "3"]
MASKS = ["--masks", "render_output/esfolado_gotas_follow_c6-9_base_segments"]
KISS = ["--file", "input/scott/kiss.mp3", "--midi", "input/scott/Instrumento.mid", "--midi-offset", "0.059",
        "--start-time", "100"]
ENXAME_DIR = "C:/Audio/recordings/projeto2024/ep/stems/enxame"
ENXAME = ["--file", f"{ENXAME_DIR}/mix_enxame_ep_15_07_2026.mp3", "--midi", f"{ENXAME_DIR}/enxame_drum_no_snare.mid",
          "--start-time", "30"]

# case -> (scene, arguments): together they touch every layer source the examples use
CASES = {
    "esfolado":              ("esfolado.yaml", ESFOLADO + ["--clips", "C:/Users/marco/Videos/ltx2", "--clip-fit", "cover", "--seed", "3"]),
    "esfolado_areia":        ("esfolado_areia_curadoria.yaml", ESFOLADO + CURADORIA),
    "esfolado_bocas":        ("esfolado_bocas.yaml", ESFOLADO),
    "esfolado_olhos_slitscan": ("esfolado_olhos_slitscan.yaml", ESFOLADO),
    "esfolado_ferrofluido":  ("esfolado_ferrofluido.yaml", ESFOLADO),
    "esfolado_gotas":        ("esfolado_gotas_curadoria.yaml", ESFOLADO + CURADORIA),
    "esfolado_gotas_espiral": ("esfolado_gotas_espiral_curadoria.yaml", ESFOLADO + CURADORIA),
    "esfolado_onda":         ("esfolado_onda_curadoria.yaml", ESFOLADO + CURADORIA),
    "esfolado_shaders":      ("esfolado_shaders_inteira.yaml", ESFOLADO),
    "esfolado_prism":        ("esfolado_prism_liquid_musica.yaml", ESFOLADO),
    "world_melts":           ("esfolado_world_melts.yaml", ESFOLADO + CURADORIA + MASKS),
    "shadertoy_lente":       ("shadertoy_exemplo_curadoria.yaml", ESFOLADO + CURADORIA),
    "esfolado_cast":         ("esfolado_cast.yaml", ESFOLADO + CURADORIA + MASKS),
    "esfolado_melodia":      ("esfolado_melodia.yaml", ESFOLADO + CURADORIA + MASKS),
    "esfolado_colagem":      ("esfolado_colagem.yaml", ESFOLADO + CURADORIA + MASKS),
    "kiss_veus":             ("kiss_veus.yaml", KISS),
    "kiss_areia":            ("kiss_areia.yaml", KISS),
    "kiss_chama":            ("kiss_chama.yaml", KISS),
    "enxame_cells":          ("enxame_cells.yaml", ENXAME + CURADORIA),
    "enxame_feedback":       ("enxame_feedback.yaml", ENXAME),
    "enxame_julia_echoes":   ("enxame_julia_echoes.yaml", ENXAME),
    "enxame_julia_orbiters": ("enxame_julia_orbiters.yaml", ENXAME),
    "enxame_particles":      ("enxame_particles.yaml", ENXAME),
    "enxame_voo":            ("enxame_voo.yaml", ENXAME),
    "enxame_cubos":          ("enxame_cubos.yaml", ENXAME),
}
# scenes that draw lots on every render (no seed): the case renders a copy with one fixed
SEEDED = {"enxame_cubos": 11}
# cases whose effect shows later than the usual frames: their own (the collage acts at the
# cut, 64 frames in, and lets go on the next snare)
OWN_KEEP = {"esfolado_colagem": (0, 64, 70, 76, 82)}


def _missing(args) -> list:
    paths = [args[i + 1] for i, a in enumerate(args) if a in ("--file", "--midi", "--clips", "--masks")]
    return [p for p in paths if not Path(p).exists()]


def _render(name: str, numpy_only: bool = False):
    """The kept frames of a case, {index: uint8 RGB}, and the seconds it took."""
    from clip_generator import build_parser, prepare
    scene, extra = CASES[name]
    scene = f"examples/{scene}"
    if name in SEEDED:
        import yaml
        cfg = yaml.safe_load(Path(scene).read_text(encoding="utf-8"))
        for layer in cfg["video"].get("layers") or []:
            layer.setdefault("seed", SEEDED[name])
        scene = OUT / "_scenes" / f"{name}.yaml"
        scene.parent.mkdir(parents=True, exist_ok=True)
        scene.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    args = build_parser().parse_args(["--scene", str(scene), "--resolution", SIZE,
                                      "--fps", str(FPS), "--bars", "0", *extra])
    t0 = time.time()
    r = prepare(args)
    if numpy_only:
        r.stack.use_gpu = False              # the compositor's numpy path: the reference
    kept = {}
    keep = OWN_KEEP.get(name, KEEP)
    for i in range(max(keep) + 1):
        frame = r.stack.frame_at(r.start_sec + i / r.fps)
        if i in keep:
            kept[i] = np.array(frame, copy=True)
    return kept, time.time() - t0


def _compare(a: np.ndarray, b: np.ndarray, tolerance: int) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return {"max": int(d.max()), "mean": float(d.mean()), "over": float((d > tolerance).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["make", "check", "list"])
    ap.add_argument("--only", nargs="*", default=None, help="Just these cases")
    ap.add_argument("--tolerance", type=int, default=2, help="Levels a pixel may differ by")
    ap.add_argument("--allow", type=float, default=0.005, help="Share of pixels allowed past the tolerance")
    ap.add_argument("--numpy", action="store_true",
                    help="Compose in numpy only (the reference path), e.g. to make a new case's frames")
    args = ap.parse_args()

    names = args.only or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        ap.error(f"unknown cases {unknown}; known: {list(CASES)}")
    if args.action == "list":
        for n in names:
            miss = _missing(CASES[n][1])
            print(f"{n:26s} {CASES[n][0]:40s} {'MISSING ' + str(miss) if miss else ''}")
        return

    failed, skipped = [], []
    for n in names:
        miss = _missing(CASES[n][1])
        if miss:
            print(f"{n:26s} skipped: missing {miss}")
            skipped.append(n)
            continue
        folder = OUT / n
        try:
            kept, secs = _render(n, args.numpy)
        except (Exception, SystemExit) as exc:          # a broken case is reported, the rest still run
            print(f"{n:26s} ERROR {type(exc).__name__}: {exc}")
            failed.append(n)
            continue
        if args.action == "make":
            folder.mkdir(parents=True, exist_ok=True)
            for i, f in kept.items():
                cv2.imwrite(str(folder / f"{i:04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            print(f"{n:26s} kept {len(kept)} frames ({secs:.0f} s)")
            continue
        worst = None
        for i, f in kept.items():
            ref = cv2.imread(str(folder / f"{i:04d}.png"))
            if ref is None:
                worst = {"max": 255, "mean": 255.0, "over": 1.0, "frame": i, "note": "no golden frame"}
                break
            c = _compare(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), f, args.tolerance)
            c["frame"] = i
            if worst is None or c["over"] > worst["over"]:
                worst = c
        ok = worst["over"] <= args.allow
        if not ok:
            failed.append(n)
        print(f"{n:26s} {'ok  ' if ok else 'FAIL'} worst frame {worst['frame']}: max {worst['max']}, "
              f"mean {worst['mean']:.3f}, {100 * worst['over']:.2f}% past +/-{args.tolerance}"
              f"{'  ' + worst['note'] if 'note' in worst else ''}  ({secs:.0f} s)")

    if args.action == "make":
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        (OUT / "cases.json").write_text(json.dumps(
            {"commit": rev, "size": SIZE, "fps": FPS, "keep": KEEP, "own_keep": OWN_KEEP,
             "cases": {n: {"scene": CASES[n][0], "args": CASES[n][1]} for n in CASES}}, indent=1),
            encoding="utf-8")
    print(f"\n{len(names) - len(failed) - len(skipped)} ok, {len(failed)} failed {failed or ''}, "
          f"{len(skipped)} skipped {skipped or ''}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
