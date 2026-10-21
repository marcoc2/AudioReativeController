"""The web Arrange's session, driven headless: open a project, every edit, undo/redo, the save
round trip (an untouched save writes the scene as it was; an edit changes only its lines)."""
import time
from pathlib import Path

import mido
import pytest
import yaml

from core.web.api import Api
from core.web.server import byte_range
from core.web.session import Session

SCENE = """\
# a test scene
video:
  clip_per_bar: true
  clip_order: shuffle
  layers:
    - source: clips
      pins: {3: none}
    - source: group
      name: BG
      layers:
        # the bubbles, slow
        - {source: shadertoy, file: shaders/shadertoy/bolhas_arc.glsl, bars: [1, 8], uniforms: {u_s: 0.4}}
        - source: footage            # the river
          role: background
          file: river.mp4
          blocks: [{bars: [3, 6], file: a.mp4}, {bars: [9, 12], file: b.mp4, from: 2.0}]
        - {source: shadertoy, file: shaders/shadertoy/fases_arc.glsl, bars: [9, 12]}
      fx:
        - {source: grain, amount: 0.2}
    - source: group
      name: FG
      layers:
        - {source: shadertoy, file: shaders/shadertoy/bolhas_arc.glsl, bars: [5, 6]}
    - {source: chord_color, amount: 0.5}   # the chords tint it
"""


def _midi(path: Path, bars: int = 16, bpm: float = 120.0) -> Path:
    mid = mido.MidiFile(ticks_per_beat=480)
    meta, notes = mido.MidiTrack(), mido.MidiTrack()
    mid.tracks += [meta, notes]
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    notes.append(mido.MetaMessage("track_name", name="kick", time=0))
    for _ in range(bars * 4):
        notes.append(mido.Message("note_on", note=36, velocity=100, channel=9, time=0))
        notes.append(mido.Message("note_off", note=36, velocity=0, channel=9, time=480))
    mid.save(path)
    return path


@pytest.fixture
def project(tmp_path):
    scene = tmp_path / "scene.yaml"
    scene.write_text(SCENE, encoding="utf-8")
    midi = _midi(tmp_path / "song.mid")
    proj = {"audio": str(tmp_path / "missing.wav"), "midi": str(midi), "scene": str(scene),
            "render": {"fps": 30, "resolution": "854x480"}}
    path = tmp_path / "proj.yaml"
    path.write_text(yaml.safe_dump(proj), encoding="utf-8")
    return path


class Quiet:
    def __init__(self):
        self.got = []

    def emit(self, fn, data):
        self.got.append((fn, data))

    def log(self, kind, text):
        self.got.append(("log", {"kind": kind, "text": text}))


class Files:
    def url(self, path):
        return f"/media/{Path(path).name}"


def _session(project) -> Session:
    from core.rhythm.midi_reader import read_midi
    proj = yaml.safe_load(project.read_text(encoding="utf-8"))
    s = Session(Files(), Quiet())
    s.set_project(proj, project)
    s.load_scene()
    grid, notes = read_midi(proj["midi"], fps=30)
    s.set_bars(grid, notes, 32.0)
    return s


def _member(state, label):
    return next(m for m in state["lanes"] if m["label"] == label)


def test_the_lanes_and_their_names(project):
    st = _session(project).state()
    labels = {m["label"]: m["lane"] for m in st["lanes"]}
    assert labels["the bubbles"] == "bg"            # its name from the comment above it
    assert labels["the river"] == "bg"              # from the comment at the end of its line
    assert labels["the chords tint it"] == "master"
    assert any(m["lane"] == "bgfx" for m in st["lanes"]) and any(m["lane"] == "fg" for m in st["lanes"])
    assert st["pins"] == {"3": "none"} and not st["dirty"]


def test_an_untouched_save_writes_the_scene_as_it_was(project):
    from core.scene_doc import SceneDoc
    s = _session(project)
    before = SceneDoc.load(str(s.scene)).dumps()
    s.save()
    assert s.scene.read_text(encoding="utf-8") == before


def test_an_edit_changes_only_its_lines(project):
    s = _session(project)
    before = s.scene.read_text(encoding="utf-8").splitlines()
    bub = _member(s.state(), "the bubbles")
    s.spans(bub["ref"], [{"first": 2, "last": 9}])
    s.save()
    after = s.scene.read_text(encoding="utf-8").splitlines()
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after) and len(changed) == 1
    assert "bars: [[2, 9]]" in changed[0][1] and "name: the bubbles" in changed[0][1]


def test_blocks_keep_their_own_clips_when_one_goes(project):
    s = _session(project)
    river = _member(s.state(), "the river")
    assert [b["file"] for b in river["spans"]] == ["a.mp4", "b.mp4"]
    st = s.spans(river["ref"], river["spans"][1:])          # the first block deleted, as the page sends it
    river = _member(st, "the river")
    assert [(b["first"], b["file"], b["from"]) for b in river["spans"]] == [(9, "b.mp4", 2.0)]


def test_add_move_remove_and_undo_back_to_the_start(project):
    s = _session(project)
    start = s.state()["lanes"]
    st = s.add("bg", "shadertoy:fases_arc", 13, 16)
    assert len(st["lanes"]) == len(start) + 1
    new = [m for m in st["lanes"] if m["spans"] and m["spans"][0]["first"] == 13][0]
    st = s.relane(new["ref"], "fg", None)
    assert any(m["lane"] == "fg" and m["spans"][0]["first"] == 13 for m in st["lanes"])
    st = s.remove(_member(st, "the chords tint it")["ref"])
    assert all(m["label"] != "the chords tint it" for m in st["lanes"])
    st = s.set_layer(_member(st, "the river")["ref"], "enabled", False)
    assert not _member(st, "the river")["enabled"]
    for _ in range(4):
        st = s.step(True)
    assert [m["label"] for m in st["lanes"]] == [m["label"] for m in start] and st["redo"] == 4
    st = s.step(False)
    assert len(st["lanes"]) == len(start) + 1


def test_a_knob_follows_the_music_and_a_bad_one_is_refused(project):
    s = _session(project)
    bub = _member(s.state(), "the bubbles")
    st = s.set_param(bub["ref"], "u_s", "{hits: {track: kick}, envelope: 0.25}")
    p = next(p for p in _member(st, "the bubbles")["params"] if p["name"] == "u_s")
    vals = s.knob(p["knob"])["values"]
    assert max(vals) > 0.9 and min(vals) < 0.1          # a hit on every beat
    assert "error" in s.set_param(bub["ref"], "u_s", "{hits: [unclosed")
    s.save()
    text = s.scene.read_text(encoding="utf-8")
    assert "u_s: {hits: {track: kick}, envelope: 0.25}" in text


def test_pins_and_a_new_layer_round_trip(project):
    s = _session(project)
    s.pin(5, "river")
    s.pin(3, None)
    s.add("master", "grain", 1, 4)
    got = s.save()
    assert not got["dirty"] and got["saved_at"]
    data = yaml.safe_load(s.scene.read_text(encoding="utf-8"))
    layers = data["video"]["layers"]
    assert layers[0]["pins"] == {5: "river"}
    assert layers[-1]["source"] == "grain" and layers[-1]["bars"] == [[1, 4]]
    assert "# the bubbles, slow" in s.scene.read_text(encoding="utf-8")
    s.step(True)                                       # undo survives the save
    assert yaml.safe_load(yaml.safe_dump(s.state()["pins"])) == {"5": "river"}


def test_save_as_and_reload(project, tmp_path):
    s = _session(project)
    s.add("bg", "shadertoy:fases_arc", 13, 16)
    got = s.save(tmp_path / "copy.yaml")
    assert got["path"].endswith("copy.yaml") and (tmp_path / "copy.yaml").is_file()
    assert s.proj["scene"].endswith("copy.yaml")
    s.add("bg", "shadertoy:fases_arc", 1, 2)
    st = s.reload()
    assert st["undo"] == 0 and not st["dirty"]
    assert sum(1 for m in st["lanes"] if m["key"] == "shadertoy:fases_arc") == 2      # the saved one, not the later one


def test_a_project_whose_song_is_missing_fails_at_the_song_and_still_edits(project, tmp_path, monkeypatch):
    monkeypatch.setattr("core.web.api.PREFS", tmp_path / "prefs.json")       # not the user's recent projects
    api = Api()
    api.open(str(project))
    for _ in range(100):
        steps = {s["step"]: s for s in api.load_state()}
        if all(s["state"] in ("done", "failed", "skipped") for s in steps.values()):
            break
        time.sleep(0.1)
    assert steps["bars"]["state"] == "done"
    assert steps["song"]["state"] == "failed" and "missing.wav" in steps["song"]["message"]
    assert steps["features"]["state"] == "skipped"
    st = api.state()
    assert st["lanes"]
    assert "error" not in api.add("bg", "shadertoy:fases_arc", 1, 2)
    assert api.recent()[0]["path"].endswith("proj.yaml")


def test_an_api_call_that_breaks_answers_with_where():
    got = Api().spans(["BG", "layers", 99], [])
    assert "error" in got and "where" in got


def test_a_range_past_the_end_is_refused():
    assert byte_range("bytes=0-99", 50) == (0, 49, 206)
    assert byte_range("bytes=60-", 50)[2] == 416
    assert byte_range("bytes=-10", 50) == (40, 49, 206)
    assert byte_range("", 50) == (0, 49, 200)


def test_every_example_saves_unchanged_through_the_session():
    from core.scene_doc import SceneDoc
    checked = flat = 0
    for path in sorted(Path("examples").glob("*.yaml")) + sorted(Path("scenes").glob("*.yaml")):
        s = Session(Files(), Quiet())
        s.set_project({"scene": str(path)}, None)
        try:
            s.load_scene()
        except Exception:
            continue                                  # not a scene the Arrange opens
        before = SceneDoc.load(str(path)).dumps()
        s._patched(s.doc, s.items)
        if s.converted:                               # a flat scene is written grouped, saying the same
            from core.scene_doc import plain
            again = Session(Files(), Quiet())
            again.doc = SceneDoc.from_text(s.doc.dumps())
            assert plain(again.doc.layers()) == plain(s.specs()), path
            flat += 1
        else:
            assert s.doc.dumps() == before, path
            checked += 1
    assert checked >= 1 and checked + flat >= 90
