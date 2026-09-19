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
    assert [e.time for e in score.section_changes()] == [0.0, 4.0]


def test_grid_from_transcribed_beats(folder):
    grid = read_score(folder).grid(fps=30)
    assert grid.bpm == pytest.approx(120.0)
    assert grid.time_signature == (4, 4)
    assert np.allclose(grid.downbeats, [0, 2, 4, 6, 8])


def test_partial_folder_and_detection(tmp_path):
    assert not has_score(tmp_path)
    _write(tmp_path, "chord.lab", [(0.0, 3.0, "C:maj")])
    assert has_score(tmp_path)
    score = read_score(tmp_path)
    assert score.duration == 3.0
    assert score.sections == [] and score.grid() is None
    with pytest.raises(FileNotFoundError):
        read_score(tmp_path / "nope")
