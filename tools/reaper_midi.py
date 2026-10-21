"""Write the notes of REAPER tracks to a MIDI file, added to the song's own MIDI export.

usage:
  python tools/reaper_midi.py --project C:/Audio/.../song.rpp --tracks piano \\
      --merge input/esfolado/esfolado.mid --out input/esfolado/esfolado_piano.mid

For a part that was never exported: its MIDI items are read from the .rpp (offsets,
loops and muted notes followed) and written as one more track, named as in REAPER.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.reaper_project import write_midi  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="The REAPER project (.rpp)")
    ap.add_argument("--tracks", nargs="+", required=True, help="Track names to take")
    ap.add_argument("--merge", default=None, help="A MIDI file whose tracks go first")
    ap.add_argument("--out", required=True, help="The MIDI file to write")
    args = ap.parse_args()
    if args.merge and Path(args.merge).resolve() == Path(args.out).resolve():
        ap.error("--out must not be the --merge file (it is read, not changed)")
    for name, n in write_midi(args.project, args.tracks, args.out, merge=args.merge).items():
        print(f"{name}: {n} notes")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
