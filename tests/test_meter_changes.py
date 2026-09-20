"""Meter changes: a few bars of 6/4 inside a 5/4 song must move the bar lines, not just the count."""
import mido
import numpy as np
import pytest

from core.rhythm import RhythmGrid, parse_meter_changes, read_midi


def _write_midi(path, bpm=120.0, signatures=((0, 5, 4),), beats=60, tpb=480):
    """One note per beat; ``signatures`` = (beat position, num, den) meta events."""
    mid = mido.MidiFile(ticks_per_beat=tpb)
    meta, notes = mido.MidiTrack(), mido.MidiTrack()
    mid.tracks += [meta, notes]
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    last = 0
    for beat, num, den in signatures:
        meta.append(mido.MetaMessage("time_signature", numerator=num, denominator=den,
                                     time=beat * tpb - last))
        last = beat * tpb
    for _ in range(beats):
        notes.append(mido.Message("note_on", note=36, velocity=100, channel=9, time=0))
        notes.append(mido.Message("note_off", note=36, velocity=0, channel=9, time=tpb))
    mid.save(path)
    return str(path)


def test_parse_meter_changes():
    assert parse_meter_changes("22:6/4, 27:5/4") == [(22, (6, 4)), (27, (5, 4))]
    assert parse_meter_changes("27:5/4,22:6/4")[0][0] == 22          # sorted by bar
    for bad in ("22-6/4", "x:6/4", "0:6/4", "3:6"):
        with pytest.raises(ValueError):
            parse_meter_changes(bad)


def test_declared_changes_move_the_bar_lines(tmp_path):
    # 5/4 at 120 BPM: bar = 2.5 s. Bars 3-4 in 6/4 (3.0 s each), back to 5/4 at bar 5.
    path = _write_midi(tmp_path / "song.mid")
    grid, _ = read_midi(path, meter_changes=[(3, (6, 4)), (5, (5, 4))])
    assert np.allclose(grid.downbeats[:6], [0.0, 2.5, 5.0, 8.0, 11.0, 13.5])
    assert grid.bar_beats[:6].tolist() == [5, 5, 6, 6, 5, 5]
    assert grid.time_signature == (5, 4)                              # the meter the song opens with

    assert grid.bar_index(7.9) == 2 and grid.beat_in_bar(7.9) == (6, 6)     # sixth beat of a 6/4 bar
    assert grid.bar_index(8.0) == 3 and grid.beat_in_bar(8.0) == (1, 6)
    assert grid.bar_index(11.0) == 4 and grid.beat_in_bar(11.2) == (1, 5)   # back in 5/4
    assert grid.is_downbeat(8.0) and not grid.is_downbeat(7.5)              # 7.5 s is where fixed 5/4 put it
    assert grid.phase(9.5) == pytest.approx(0.5)                            # half of a 3 s bar


def test_static_five_four_would_be_one_bar_ahead_afterwards(tmp_path):
    # two 6/4 bars add 2 beats, not a whole bar: the fixed grid is also out of phase after them
    path = _write_midi(tmp_path / "song.mid")
    fixed, _ = read_midi(path)
    real, _ = read_midi(path, meter_changes=[(3, (6, 4)), (5, (5, 4))])
    assert fixed.bar_index(11.0) == 4 and fixed.beat_in_bar(11.0)[0] == 3   # thinks: bar 5, beat 3
    assert real.bar_index(11.0) == 4 and real.beat_in_bar(11.0)[0] == 1     # really: bar 5, beat 1


def test_time_signature_events_in_the_file_are_all_honoured(tmp_path):
    path = _write_midi(tmp_path / "song.mid", signatures=((0, 5, 4), (10, 6, 4), (22, 5, 4)))
    grid, _ = read_midi(path)
    assert np.allclose(grid.downbeats[:6], [0.0, 2.5, 5.0, 8.0, 11.0, 13.5])
    assert grid.bar_beats[:6].tolist() == [5, 5, 6, 6, 5, 5]


def test_declared_changes_win_over_the_file(tmp_path):
    path = _write_midi(tmp_path / "song.mid", signatures=((0, 5, 4), (10, 6, 4)))
    grid, _ = read_midi(path, meter_changes=[(3, (7, 4))])
    assert grid.bar_beats[:3].tolist() == [5, 5, 7]


def test_bars_carry_on_past_the_last_marker_at_the_last_bars_length(tmp_path):
    path = _write_midi(tmp_path / "song.mid", beats=20)
    grid, _ = read_midi(path, meter_changes=[(2, (6, 4))])               # ends in 6/4: 3.0 s bars
    last = len(grid.downbeats) - 1
    assert grid.bar_start(last + 2) == pytest.approx(grid.downbeats[-1] + 6.0)
    assert grid.bar_index(grid.downbeats[-1] + 3.1) == last + 1
    assert grid.beats_in_bar(last + 5) == 6


def test_grid_without_markers_still_answers():
    grid = RhythmGrid(bpm=120.0, time_signature=(5, 4), start_offset=1.0)
    assert grid.bar_index(0.5) == -1 and grid.beat_in_bar(0.5) == (0, 5)
    assert grid.bar_index(1.0) == 0 and grid.beat_in_bar(3.4) == (5, 5)
    assert grid.bar_start(2) == pytest.approx(6.0)


def test_score_snaps_to_the_real_bar_lines(tmp_path):
    from core.score import read_score
    rows = [("0.0", "7.7", "E:min"), ("7.7", "13.6", "C:maj7")]
    (tmp_path / "chord.lab").write_text("".join(chr(9).join(r) + chr(10) for r in rows), encoding="utf-8")
    grid, _ = read_midi(_write_midi(tmp_path / "song.mid"), meter_changes=[(3, (6, 4)), (5, (5, 4))])
    snapped = read_score(tmp_path).snap_to(grid)
    assert [e.time for e in snapped.chord_changes()] == [8.0]            # bar 4 starts at 8.0 s, not 7.5 s
    assert snapped.bar_beat_at(7.9) == (3, 6)
