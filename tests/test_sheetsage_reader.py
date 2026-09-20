"""Score reader: SheetSage2's .lab folder -> chords, key, sections, beats, melody."""
import numpy as np
import pytest

from core.score import Span, has_score, parse_chord, read_score


def _write(folder, name, rows):
    (folder / name).write_text("\n".join("\t".join(str(c) for c in r) for r in rows) + "\n",
                               encoding="utf-8")


@pytest.fixture
def folder(tmp_path):
    # 2 s bars; SheetSage2 writes one chord line per bar even when nothing changes.
    _write(tmp_path, "chord.lab", [(0.0, 2.0, "N"), (2.0, 4.0, "A:maj"), (4.0, 6.0, "A:maj"),
                                   (6.0, 8.0, "F#:min7"), (8.0, 10.0, "N")])
    _write(tmp_path, "key.lab", [(0.0, 10.0, "A:major")])
    _write(tmp_path, "structure.lab", [(0.0, 4.0, "intro"), (4.0, 10.0, "verse")])
    _write(tmp_path, "beat.lab", [(i * 0.5, i % 4 + 1, 4, 4) for i in range(20)])
    _write(tmp_path, "downbeat.lab", [(i * 2.0,) for i in range(5)])
    _write(tmp_path, "melody_instrumental.lab", [(2.0, 2.5, 69), (3.0, 4.0, 73)])
    return tmp_path


def test_parse_chord():
    assert parse_chord("A:maj") == (9, "maj", (9, 1, 4))
    assert parse_chord("F#:min7") == (6, "min7", (6, 9, 1, 4))
    assert parse_chord("Bb:7/5")[0] == 10
    assert parse_chord("N") == (None, "", ())
    # unknown quality still yields a triad instead of raising
    assert parse_chord("C:min13")[2] == (0, 3, 7)


def test_same_chord_across_bars_is_one_span(folder):
    score = read_score(folder)
    assert [s.label for s in score.chords] == ["N", "A:maj", "F#:min7", "N"]
    assert score.chords[1] == Span(2.0, 6.0, "A:maj")


def test_lookup_at_time(folder):
    score = read_score(folder)
    assert score.chord_at(5.9).label == "A:maj"
    assert score.chord_at(6.0).label == "F#:min7"
    assert score.next_chord(5.9).label == "F#:min7"
    assert score.key_at(3.0).label == "A:major"
    assert score.section_at(4.0).label == "verse"
    assert score.chord_at(99.0) is None
    assert score.bar_beat_at(4.6) == (3, 2)
    assert score.bar_beat_at(-1.0) == (0, 0)


def test_melody(folder):
    score = read_score(folder)
    assert score.melody_at(2.2).pitch == 69
    assert score.melody_at(2.7) is None          # gap between the two notes
    assert score.melody_at(3.5).duration == pytest.approx(1.0)
    assert score.melody("vocal") == []
    with pytest.raises(ValueError):
        score.melody_at(0.0, voice="bass")


def test_changes_as_trigger_events(folder):
    score = read_score(folder)
    changes = score.chord_changes()
    # entering "N" is not an event; pitch carries the new root
    assert [(e.time, e.pitch) for e in changes] == [(2.0, 9), (6.0, 6)]
    assert changes[0].duration == pytest.approx(4.0)
    assert [e.time for e in score.section_changes()] == [4.0]        # the opening section is not a change


def test_grid_from_transcribed_beats(folder):
    grid = read_score(folder).grid(fps=30)
    assert grid.bpm == pytest.approx(120.0)
    assert grid.time_signature == (4, 4)
    assert np.allclose(grid.downbeats, [0, 2, 4, 6, 8])


def test_grid_tempo_survives_10ms_rounding(tmp_path):
    # SheetSage2 writes times rounded to 10 ms; at 130 BPM the median gap then says 130.43
    beats = np.round(np.arange(400) * 60.0 / 130.0, 2)
    _write(tmp_path, "beat.lab", [(f"{t:.2f}", i % 4 + 1, 4, 4) for i, t in enumerate(beats)])
    assert 60.0 / np.median(np.diff(beats)) == pytest.approx(130.43, abs=0.01)   # the trap
    assert read_score(tmp_path).grid().bpm == pytest.approx(130.0, abs=0.02)


def _trusted_grid(bpm=120.0, time_signature=(5, 4), seconds=12.0):
    from core.rhythm import RhythmGrid
    beat = 60.0 / bpm
    beats = np.arange(0.0, seconds, beat)
    return RhythmGrid(bpm=bpm, time_signature=time_signature,
                      beats=beats, downbeats=beats[:: time_signature[0]])


def test_snap_moves_boundaries_onto_the_trusted_grid(folder):
    # the transcription slipped ~0.18 s late, as SheetSage2 does for long stretches
    _write(folder, "chord.lab", [(0.0, 2.68, "A:maj"), (2.68, 5.17, "D:maj"), (5.17, 10.0, "A:maj")])
    snapped = read_score(folder).snap_to(_trusted_grid(), unit="beat")
    assert [(s.start, s.end, s.label) for s in snapped.chords] == [
        (0.0, 2.5, "A:maj"), (2.5, 5.0, "D:maj"), (5.0, 10.0, "A:maj")]
    assert [e.time for e in snapped.chord_changes()] == [2.5, 5.0]    # the opening chord is not a change


def test_snap_to_bar_and_adopts_the_grids_meter(folder):
    score = read_score(folder)
    assert score.time_signature == (4, 4)                      # what SheetSage2 always says
    snapped = score.snap_to(_trusted_grid(), unit="bar")       # 5/4 at 120 BPM: bars every 2.5 s
    assert snapped.time_signature == (5, 4)
    assert all(s.start % 2.5 == pytest.approx(0.0, abs=1e-9) for s in snapped.chords)
    assert snapped.bar_beat_at(2.5) == (2, 1)                  # bar 2 starts at 2.5 s, not at 2.0 s
    assert snapped.bar_beat_at(4.6) == (2, 5)                  # fifth beat of a 5/4 bar
    assert score.bar_beat_at(4.6) == (3, 2)                    # the original is left untouched


def test_snap_drops_spans_squeezed_to_nothing_and_rejoins_neighbours(folder):
    _write(folder, "chord.lab", [(0.0, 2.4, "A:maj"), (2.4, 2.6, "E:maj"), (2.6, 10.0, "A:maj")])
    snapped = read_score(folder).snap_to(_trusted_grid(), unit="bar")
    assert [(s.start, s.end, s.label) for s in snapped.chords] == [(0.0, 10.0, "A:maj")]


def test_snap_carries_the_grid_past_its_last_marker(folder):
    # a MIDI file often ends before its audio does
    short = _trusted_grid(seconds=3.0)
    snapped = read_score(folder).snap_to(short, unit="beat")
    assert snapped.beats[-1] >= 9.5 and snapped.downbeats[-1] >= 7.5
    assert np.allclose(np.diff(snapped.beats), 0.5)


def test_snap_leaves_melody_alone_unless_asked(folder):
    score = read_score(folder)
    _write(folder, "melody_instrumental.lab", [(2.13, 2.4, 69), (3.31, 4.0, 73)])
    score = read_score(folder)
    grid = _trusted_grid()
    assert [n.time for n in score.snap_to(grid).melody()] == [2.13, 3.31]
    assert [n.time for n in score.snap_to(grid, melody_division=4).melody()] == [2.125, 3.25]
    with pytest.raises(ValueError):
        score.snap_to(grid, unit="16th")


def test_partial_folder_and_detection(tmp_path):
    assert not has_score(tmp_path)
    _write(tmp_path, "chord.lab", [(0.0, 3.0, "C:maj")])
    assert has_score(tmp_path)
    score = read_score(tmp_path)
    assert score.duration == 3.0
    assert score.sections == [] and score.grid() is None
    with pytest.raises(FileNotFoundError):
        read_score(tmp_path / "nope")
