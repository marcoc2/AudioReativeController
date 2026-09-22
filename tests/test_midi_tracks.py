"""Track-aware MIDI: a DAW export has one track per instrument, often all on channel 0 with clashing pitches."""
import mido
import pytest

from core.rhythm import read_midi, select_notes
from core.video.layers import _layer_hits


def _write(path, tracks, bpm=120.0, tpb=480):
    """``tracks`` = [(name, [(beat, pitch), ...]), ...]; every note lasts a quarter of a beat."""
    mid = mido.MidiFile(ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="song", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    mid.tracks.append(meta)
    for name, hits in tracks:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        events = sorted([(int(b * tpb), "note_on", p) for b, p in hits]
                        + [(int((b + 0.25) * tpb), "note_off", p) for b, p in hits])
        last = 0
        for tick, kind, pitch in events:
            tr.append(mido.Message(kind, note=pitch, velocity=100 if kind == "note_on" else 0,
                                   channel=0, time=tick - last))
            last = tick
        mid.tracks.append(tr)
    mid.save(path)
    return str(path)


SONG = [("kick", [(0, 36), (2, 36)]), ("kick-2", [(1, 36)]), ("hihat", [(0, 36), (0.5, 42)]),
        ("piano", [(0, 60)]), ("piano", [(4, 64)]), ("synth", [(4, 64)])]


def test_notes_remember_their_track(tmp_path):
    _, notes = read_midi(_write(tmp_path / "s.mid", SONG))
    assert {(n.track, n.track_index) for n in notes} == {("kick", 1), ("kick-2", 2), ("hihat", 3),
                                                          ("piano", 4), ("piano", 5), ("synth", 6)}
    # the same pitch at the same instant on two tracks is two notes, each with its own length
    at_zero = [n for n in notes if n.time == 0 and n.pitch == 36]
    assert len(at_zero) == 2 and all(n.duration == pytest.approx(0.125) for n in at_zero)


def test_select_by_track(tmp_path):
    _, notes = read_midi(_write(tmp_path / "s.mid", SONG))
    times = lambda spec: [n.time for n in select_notes(notes, spec)]
    assert times({"notes": [36]}) == [0.0, 0.0, 0.5, 1.0]                  # pitch alone mixes three instruments
    assert times({"track": "kick"}) == [0.0, 1.0]
    assert times({"track": ["kick", "kick-2"]}) == [0.0, 0.5, 1.0]         # two sample triggers summed into one
    assert times({"track": "Piano"}) == [0.0, 2.0]                         # same name: taken together
    assert times({"track": 5}) == [2.0]                                    # ... an index tells them apart
    assert times({"track": ["piano", "synth"]}) == [0.0, 2.0]              # a doubled part counts once
    assert times({"track": "hihat", "notes": [42]}) == [0.25]
    with pytest.raises(ValueError, match="tracks in the file"):
        select_notes(notes, {"track": "snare"})


def test_layer_triggers_accept_track(tmp_path):
    _, notes = read_midi(_write(tmp_path / "s.mid", SONG))
    assert _layer_hits({"track": ["kick", "kick-2"]}, notes) == [0.0, 0.5, 1.0]
    assert _layer_hits({"notes": [42]}, notes) == [0.25]
