"""Find where each piece of a stem comes from in a clip, and write the cue list a
``footage`` layer plays (core/stem_match.py explains how).

usage:
  python tools/match_stem.py --stem input/esfolado/esfolado_voz.mp3 \\
      --clip input/esfolado/MiniMax_H3_00039_.mp4 [--semitones 1] [--out cues.yaml]

Prints the cues as a YAML flow list, ready for ``cues:`` in a scene, and what the stem
sounds that no cue covers. ``--semitones`` skips finding the transposition.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.stem_match import find_cues  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True, help="The stem cut from the clip's sound")
    ap.add_argument("--clip", required=True, help="The clip (its sound is read)")
    ap.add_argument("--semitones", type=float, default=None,
                    help="How far the stem is transposed from the clip (default: found)")
    ap.add_argument("--min-score", type=float, default=0.45,
                    help="Likeness a second of the stem needs to count (0..1)")
    ap.add_argument("--out", default=None, help="Also write the cue file (YAML)")
    args = ap.parse_args()
    r = find_cues(args.stem, args.clip, semitones=args.semitones, min_score=args.min_score)
    print(f"# {args.stem} <- {args.clip}: transposed {r['semitones']:+g} semitones, {len(r['cues'])} cues")
    print("cues: [" + ",\n       ".join("[" + ", ".join(f"{v:g}" for v in c) + "]" for c in r["cues"]) + "]")
    if r["uncovered"]:
        print("# sounding with no cue: " + ", ".join(f"{a:g}-{b:g}" for a, b in r["uncovered"]))
    if args.out:
        import yaml
        Path(args.out).write_text(yaml.safe_dump(r, sort_keys=False, default_flow_style=None),
                                  encoding="utf-8")
        print(f"# written {args.out}")


if __name__ == "__main__":
    main()
