import mido
import numpy as np
import pytest

from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import KICK_NOTE, MidiNote, read_midi


def _build_midi(tmp_path, bpm: float, kick_count: int, kick_every_beats: int = 4,
                ticks_per_beat: int = 480) -> str:
    """Write a small MIDI file with kicks every ``kick_every_beats`` beats."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    note_dur = ticks_per_beat // 4
    for i in range(kick_count):
        # subtract note_off duration so next kick lands exactly on the beat
        delta = (ticks_per_beat * kick_every_beats - note_dur) if i > 0 else 0
        track.append(mido.Message("note_on", note=KICK_NOTE, velocity=100, channel=9, time=delta))
        track.append(mido.Message("note_off", note=KICK_NOTE, velocity=0, channel=9, time=note_dur))
    path = tmp_path / "test.mid"
    mid.save(path)
    return str(path)


def test_read_midi_extracts_tempo_and_notes(tmp_path):
    path = _build_midi(tmp_path, bpm=120, kick_count=4, kick_every_beats=4)
    grid, notes = read_midi(path, time_signature=(4, 4))
    assert isinstance(grid, RhythmGrid)
    assert grid.bpm == pytest.approx(120, rel=0.05)
    assert len(notes) == 4
    assert all(n.pitch == KICK_NOTE for n in notes)
    assert all(n.velocity == 100 for n in notes)
    assert all(n.duration == pytest.approx(0.125, rel=0.1) for n in notes)


def test_read_midi_kicks_become_downbeats(tmp_path):
    # 4 kicks at downbeat (every 4 beats @ 120bpm = every 2.0s)
    path = _build_midi(tmp_path, bpm=120, kick_count=4, kick_every_beats=4)
    grid, _ = read_midi(path, time_signature=(4, 4))
    assert grid.downbeats is not None and len(grid.downbeats) == 4
    gaps = np.diff(grid.downbeats)
    assert float(np.median(gaps)) == pytest.approx(grid.bar_duration, rel=0.1)


def test_read_midi_interpolates_beats_between_kicks(tmp_path):
    path = _build_midi(tmp_path, bpm=120, kick_count=3, kick_every_beats=4)
    grid, _ = read_midi(path, time_signature=(4, 4))
    # 3 downbeats -> at least 2 bars worth of beats
    assert grid.beats is not None and len(grid.beats) >= 9
    assert grid.beat_duration == pytest.approx(0.5, rel=0.05)
    assert float(np.median(np.diff(grid.beats))) == pytest.approx(grid.beat_duration, rel=0.05)


def test_read_midi_dense_kicks_classified_per_bar(tmp_path):
    # Kicks every beat (four-on-the-floor): classifier should still produce
    # downbeats spaced one bar apart, not every beat.
    path = _build_midi(tmp_path, bpm=120, kick_count=8, kick_every_beats=1)
    grid, _ = read_midi(path, time_signature=(4, 4))
    assert grid.bpm == pytest.approx(120, rel=0.05)
    assert grid.downbeats is not None
    gap = float(np.median(np.diff(grid.downbeats)))
    assert gap == pytest.approx(grid.bar_duration, rel=0.1)


def test_read_midi_no_kicks_still_metronomic_with_tempo(tmp_path):
    # tempo meta present -> metronomic grid even without kicks
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(140)))
    for i in range(4):
        track.append(mido.Message("note_on", note=38, velocity=90, channel=9,
                                  time=mid.ticks_per_beat if i > 0 else 0))
        track.append(mido.Message("note_off", note=38, velocity=0, channel=9, time=mid.ticks_per_beat // 4))
    path = tmp_path / "snare.mid"
    mid.save(path)

    grid, notes = read_midi(str(path))
    assert grid.bpm == pytest.approx(140, rel=0.05)
    assert grid.downbeats is not None and float(grid.downbeats[0]) == 0.0
    assert float(np.median(np.diff(grid.downbeats))) == pytest.approx(grid.bar_duration, rel=0.01)
    assert len(notes) == 4 and all(n.pitch == 38 for n in notes)


def test_read_midi_grid_anchored_at_zero_with_tempo(tmp_path):
    path = _build_midi(tmp_path, bpm=120, kick_count=4, kick_every_beats=4)
    grid, _ = read_midi(path, time_signature=(4, 4))
    assert float(grid.downbeats[0]) == 0.0
    assert grid.start_offset == 0.0


def test_read_midi_reads_time_signature_from_file(tmp_path):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("time_signature", numerator=5, denominator=4))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(130)))
    for i in range(6):
        track.append(mido.Message("note_on", note=KICK_NOTE, velocity=100, channel=9,
                                  time=mid.ticks_per_beat * 5 if i > 0 else 0))
        track.append(mido.Message("note_off", note=KICK_NOTE, velocity=0, channel=9, time=0))
    path = tmp_path / "fivefour.mid"
    mid.save(path)

    grid, _ = read_midi(str(path))              # no explicit signature
    assert grid.time_signature == (5, 4)
    assert grid.bar_duration == pytest.approx(5 * 60.0 / 130.0, rel=0.01)
    assert float(np.median(np.diff(grid.downbeats))) == pytest.approx(grid.bar_duration, rel=0.01)


def test_read_midi_explicit_signature_overrides_file(tmp_path):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("time_signature", numerator=5, denominator=4))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)))
    track.append(mido.Message("note_on", note=KICK_NOTE, velocity=100, channel=9, time=0))
    track.append(mido.Message("note_off", note=KICK_NOTE, velocity=0, channel=9, time=120))
    path = tmp_path / "override.mid"
    mid.save(path)

    grid, _ = read_midi(str(path), time_signature=(3, 4))
    assert grid.time_signature == (3, 4)


def test_read_midi_velocity_preserved(tmp_path):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)))
    for vel in (40, 80, 127):
        track.append(mido.Message("note_on", note=KICK_NOTE, velocity=vel, channel=9,
                                  time=mid.ticks_per_beat))
        track.append(mido.Message("note_off", note=KICK_NOTE, velocity=0, channel=9,
                                  time=mid.ticks_per_beat // 4))
    path = tmp_path / "vel.mid"
    mid.save(path)

    _, notes = read_midi(str(path))
    assert [n.velocity for n in notes] == [40, 80, 127]


def test_read_midi_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_midi(str(tmp_path / "nope.mid"))
