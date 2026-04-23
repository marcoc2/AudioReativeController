import numpy as np
import pytest

from core.rhythm.grid import RhythmGrid, SUBDIVISIONS


def test_durations_fixed_grid():
    g = RhythmGrid(bpm=120, time_signature=(4, 4), fps=24)
    assert g.beat_duration == pytest.approx(0.5)
    assert g.bar_duration == pytest.approx(2.0)
    assert g.whole_duration == pytest.approx(2.0)


def test_durations_6_8():
    g = RhythmGrid(bpm=120, time_signature=(6, 8), fps=24)
    assert g.beat_duration == pytest.approx(0.5)
    assert g.bar_duration == pytest.approx(3.0)
    assert g.whole_duration == pytest.approx(4.0)


def test_phase_fixed_grid_wraps():
    g = RhythmGrid(bpm=120, time_signature=(4, 4), fps=24)
    assert g.phase(0.0) == pytest.approx(0.0)
    assert g.phase(1.0) == pytest.approx(0.5)
    assert g.phase(2.0) == pytest.approx(0.0)
    assert g.phase(3.0) == pytest.approx(0.5)


def test_beat_phase_fixed_grid():
    g = RhythmGrid(bpm=120, fps=24)
    assert g.beat_phase(0.0) == pytest.approx(0.0)
    assert g.beat_phase(0.25) == pytest.approx(0.5)
    assert g.beat_phase(0.5) == pytest.approx(0.0)


def test_is_beat_fixed_grid():
    g = RhythmGrid(bpm=120, fps=24)
    tol = 0.5 / 24
    assert g.is_beat(0.0)
    assert g.is_beat(0.5)
    assert g.is_beat(0.5 + tol * 0.5)
    assert not g.is_beat(0.25)


def test_is_downbeat_fixed_grid_only_once_per_bar():
    g = RhythmGrid(bpm=120, time_signature=(4, 4), fps=24)
    assert g.is_downbeat(0.0)
    assert not g.is_downbeat(0.5)
    assert not g.is_downbeat(1.0)
    assert not g.is_downbeat(1.5)
    assert g.is_downbeat(2.0)


def test_subdivision_duration():
    g = RhythmGrid(bpm=120, time_signature=(4, 4))
    assert g.subdivision_duration("whole") == pytest.approx(2.0)
    assert g.subdivision_duration("quarter") == pytest.approx(0.5)
    assert g.subdivision_duration("eighth") == pytest.approx(0.25)
    assert g.subdivision_duration("16th") == pytest.approx(0.125)


def test_subdivision_unknown_raises():
    g = RhythmGrid(bpm=120)
    with pytest.raises(ValueError):
        g.subdivision_duration("triplet")


def test_subdivision_index_counts_steps():
    g = RhythmGrid(bpm=120, fps=24)
    assert g.subdivision_index(0.0, "16th") == 0
    assert g.subdivision_index(0.125, "16th") == 1
    assert g.subdivision_index(0.25, "16th") == 2


def test_from_beats_infers_bpm():
    beats = np.arange(0, 10) * 0.5
    g = RhythmGrid.from_beats(beats, time_signature=(4, 4), fps=24)
    assert g.bpm == pytest.approx(120.0)
    assert g.start_offset == pytest.approx(0.0)
    assert len(g.downbeats) == -(-len(beats) // 4)  # ceil(n/4)


def test_from_beats_requires_two():
    with pytest.raises(ValueError):
        RhythmGrid.from_beats([1.0])


def test_variable_grid_phase_uses_markers():
    downbeats = np.array([0.0, 2.0, 4.1, 6.0])
    g = RhythmGrid(bpm=120, fps=24, downbeats=downbeats)
    assert g.phase(0.0) == pytest.approx(0.0)
    assert g.phase(1.0) == pytest.approx(0.5)
    assert g.phase(2.0) == pytest.approx(0.0)
    assert g.phase(2.0 + (4.1 - 2.0) / 2) == pytest.approx(0.5)


def test_is_beat_with_markers():
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    g = RhythmGrid(bpm=120, fps=24, beats=beats)
    assert g.is_beat(0.0)
    assert g.is_beat(1.5)
    assert not g.is_beat(0.25)


def test_phase_before_first_downbeat():
    g = RhythmGrid(bpm=120, fps=24, downbeats=np.array([2.0, 4.0]))
    assert g.phase(2.0) == pytest.approx(0.0)
    assert g.phase(1.0) == pytest.approx(0.5)


def test_start_offset_shifts_fixed_grid():
    g = RhythmGrid(bpm=120, fps=24, start_offset=0.25)
    assert g.beat_phase(0.25) == pytest.approx(0.0)
    assert g.is_beat(0.25)
    assert not g.is_beat(0.0 + 0.01)
