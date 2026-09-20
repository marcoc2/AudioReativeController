"""`score:` trigger source — a transcription's chord changes drive the composer like drum hits do."""
import numpy as np
import pytest

from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import MidiNote
from core.video.composer import ClipComposer


class _Library:
    """The composer only needs a clip count to build its events."""
    def __len__(self):
        return 3


def _midi_grid(bpm=120.0, time_signature=(4, 4), seconds=20.0):
    """A grid with real markers, as read_midi produces: bars every 2 s at 120 BPM in 4/4."""
    beats = np.arange(0.0, seconds, 60.0 / bpm)
    return RhythmGrid(bpm=bpm, time_signature=time_signature, fps=24,
                      beats=beats, downbeats=beats[:: time_signature[0]])


def _write(folder, name, rows):
    (folder / name).write_text("\n".join("\t".join(str(c) for c in r) for r in rows) + "\n",
                               encoding="utf-8")


@pytest.fixture
def transcription(tmp_path):
    # chord changes ~0.3 s late, the way SheetSage2 reports them
    _write(tmp_path, "chord.lab", [(0.0, 4.31, "A#:min"), (4.31, 6.28, "G#:sus4"),
                                   (6.28, 12.33, "A#:min"), (12.33, 14.3, "G#:sus4"),
                                   (14.3, 20.0, "A#:min")])
    _write(tmp_path, "structure.lab", [(0.0, 8.2, "intro"), (8.2, 20.0, "chorus")])
    _write(tmp_path, "melody_instrumental.lab", [(1.02, 1.4, 70), (1.27, 1.6, 73)])
    return tmp_path


def _composer(grid, cfg, notes=(), **loaders):
    return ClipComposer(_Library(), grid, list(notes), {"clip_per_bar": False, "triggers": cfg}, **loaders)


def test_chord_changes_become_events_snapped_to_the_midi_bars(transcription):
    comp = _composer(_midi_grid(), {"harmony": {"score": str(transcription), "actions": ["next_clip"]}})
    assert [e.time for e in comp.events] == [4.0, 6.0, 12.0, 14.0]       # the opening chord is not a change
    assert {e.name for e in comp.events} == {"harmony"}
    assert all(e.actions == ("next_clip",) for e in comp.events)


def test_auto_snap_leaves_times_alone_on_a_placeholder_grid(transcription):
    # no MIDI: clip_generator falls back to a bare tempo, which knows nothing about this song
    comp = _composer(RhythmGrid(bpm=120.0), {"harmony": {"score": str(transcription), "actions": ["next_clip"]}})
    assert [e.time for e in comp.events] == [4.31, 6.28, 12.33, 14.3]


def test_snap_can_be_forced_either_way(transcription):
    spec = {"score": str(transcription), "actions": ["next_clip"]}
    off = _composer(_midi_grid(), {"harmony": {**spec, "snap": "off"}})
    assert [e.time for e in off.events][0] == 4.31
    beat = _composer(_midi_grid(), {"harmony": {**spec, "snap": "beat"}})
    assert [e.time for e in beat.events] == [4.5, 6.5, 12.5, 14.5]        # nearest beat: the coin flip


def test_notes_filters_by_the_new_root(transcription):
    comp = _composer(_midi_grid(), {"to_g_sharp": {"score": str(transcription), "notes": [8],
                                                   "actions": ["next_clip"]}})
    assert [e.time for e in comp.events] == [4.0, 12.0]


def test_section_changes_and_melody(transcription):
    sections = _composer(_midi_grid(), {"form": {"score": str(transcription), "events": "section_changes",
                                                 "actions": ["random_clip"]}})
    assert [e.time for e in sections.events] == [8.0]
    melody = _composer(_midi_grid(), {"tune": {"score": str(transcription), "events": "melody",
                                               "actions": ["reverse"]}})
    assert [e.time for e in melody.events] == [1.0, 1.25]                # sixteenths of the grid, not bars


def test_chords_drive_until_the_drums_come_in(transcription):
    kicks = [MidiNote(time=t, pitch=36, velocity=110, channel=9, duration=0.1) for t in (10.0, 10.5, 11.0)]
    comp = _composer(_midi_grid(), {
        "harmony": {"score": str(transcription), "actions": ["next_clip"], "until": "kick"},
        "kick": {"notes": [36], "actions": ["next_clip"]},
    }, notes=kicks)
    assert [(e.name, e.time) for e in comp.events] == [
        ("harmony", 4.0), ("harmony", 6.0),
        ("kick", 10.0), ("kick", 10.5), ("kick", 11.0)]


def test_score_loader_is_injectable_and_gets_the_grid():
    seen = {}

    def fake_loader(spec, grid):
        seen["spec"], seen["grid"] = spec, grid
        return [MidiNote(time=2.0, pitch=0, velocity=100, channel=0, duration=1.0)]

    grid = _midi_grid()
    comp = _composer(grid, {"harmony": {"score": "anywhere", "actions": ["restart"]}}, score_loader=fake_loader)
    assert [e.time for e in comp.events] == [2.0]
    assert seen["spec"]["score"] == "anywhere" and seen["grid"] is grid


@pytest.mark.parametrize("bad", [{"events": "key_changes"}, {"snap": "16th"}])
def test_bad_options_are_rejected(transcription, bad):
    with pytest.raises(ValueError):
        _composer(_midi_grid(), {"harmony": {"score": str(transcription), "actions": ["next_clip"], **bad}})
