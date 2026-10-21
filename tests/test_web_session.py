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


# -- the Arrange's edits (items 7-16) ---------------------------------------------------------

def _path(st, label):
    return _member(st, label)["ref"]


def test_moving_a_block_takes_its_bars_from_the_others_and_a_resized_start_moves_its_from(project):
    s = _session(project)
    river = _path(s.state(), "the river")
    st = s.move(river, 1, "first", 10, 12)              # its second block's start: one bar later
    b = _member(st, "the river")["spans"]
    assert [(x["first"], x["last"]) for x in b] == [(3, 6), (10, 12)]
    assert b[1]["from"] == pytest.approx(2.0 + 2.0)     # a bar of 120 BPM 4/4 is 2 s
    st = s.move(river, 0, "move", 9, 12)                 # the first block over the second
    assert [(x["first"], x["last"]) for x in _member(st, "the river")["spans"]] == [(9, 12)]


def test_a_block_dragged_to_another_row_becomes_a_member_of_its_own(project):
    s = _session(project)
    river = _path(s.state(), "the river")
    st = s.move(river, 1, "move", 9, 12, "bg", 3)
    rivers = [m for m in st["lanes"] if m["file"] == "river.mp4"]
    assert len(rivers) == 2 and st["undo"] == 1
    moved = next(m for m in rivers if m["spans"][0]["first"] == 9)
    assert moved["spans"][0]["file"] == "b.mp4" and moved["spans"][0]["from"] == 2.0
    assert st["select"]["ref"] == moved["ref"]
    st = s.step(True)
    assert len([m for m in st["lanes"] if m["file"] == "river.mp4"]) == 1


def test_a_clip_dragged_to_the_fg_is_recast_as_a_character(project):
    s = _session(project)
    river = _path(s.state(), "the river")
    st = s.move(river, 0, "move", 3, 6, "fg", 0)
    fg = next(m for m in st["lanes"] if m["lane"] == "fg" and m["file"] == "river.mp4")
    spec = s.get(fg["ref"])
    assert spec["role"] == "foreground" and spec.get("fit", "contain") == "contain" and spec["scale"] == 0.6
    assert "MASTER rows only" in s.move(_path(st, "the chords tint it"), 0, "move", 1, 2, "bg", 0)["error"]


def test_new_members_go_where_they_are_put(project, tmp_path):
    s = _session(project)
    clip = tmp_path / "lion.mp4"
    clip.write_bytes(b"x")
    st = s.new_member("fg", str(clip), 2, 4, None, 1.5)
    lion = _member(st, "lion")
    assert lion["lane"] == "fg" and lion["spans"][0]["from"] == 1.5
    assert s.get(lion["ref"])["role"] == "foreground"
    st = s.new_member("bgfx", "grain", 5, 8)
    assert any(m["lane"] == "bgfx" and m["spans"] and m["spans"][0]["first"] == 5 for m in st["lanes"])
    assert "error" in s.new_member("bgfx", "shadertoy:fases_arc", 1, 2)          # a Generator on an FX track
    st = s.new_member("bg", "shadertoy:fases_arc", 1, 2, 2)
    new = st["select"]["ref"]
    assert s.get(new)["lane_row"] == 2


def test_hidden_bars_and_the_fields_that_set_them(project):
    s = _session(project)
    bub = _path(s.state(), "the bubbles")
    st = s.mute(bub, 3, 4)
    st = s.mute(bub, 6, 6)
    assert _member(st, "the bubbles")["mutes"] == [[3, 4], [6, 6]]
    st = s.mute(bub, 4, 4, False)
    assert _member(st, "the bubbles")["mutes"] == [[3, 3], [6, 6]]
    st = s.set_mutes(bub, "")
    assert _member(st, "the bubbles")["mutes"] == []
    assert "error" in s.set_mutes(bub, "x-3")
    st = s.set_windows(bub, "1-2, 7")
    assert _member(st, "the bubbles")["bars_text"] == "1-2, 7"


def test_fx_in_three_places(project):
    s = _session(project)
    river = _path(s.state(), "the river")
    st = s.add_fx(river, "grain")
    assert [f["label"] for f in _member(st, "the river")["fx"]] == ["grain"]
    st = s.add_fx(["BG"], "chord_color")
    assert sum(1 for m in st["lanes"] if m["lane"] == "bgfx") == 2
    st = s.add_fx(["master"], "grain")
    master = [m for m in st["lanes"] if m["lane"] == "master"]
    assert [m["label"] for m in master][-1] == "grain"
    st = s.reorder(master[-1]["ref"], -1)
    assert [m["label"] for m in st["lanes"] if m["lane"] == "master"] == ["grain", "the chords tint it"]
    st = s.add_fx(river, "chord_color")
    st = s.move_fx(_path(st, "the river") + ["fx", 1], 0)
    assert [f["label"] for f in _member(st, "the river")["fx"]][0] != "grain"
    st = s.set_layer(["FG"], "spill", True)
    assert st["groups"]["FG"]["spill"]


def test_copy_and_paste_links_and_fx(project):
    s = _session(project)
    bub = _path(s.state(), "the bubbles")
    s.set_param(bub, "u_s", "{hits: {track: kick}, envelope: 0.25}")
    assert "note" in s.copy_links(bub)
    fg = next(m for m in s.state()["lanes"] if m["lane"] == "fg")
    st = s.paste_links(fg["ref"])
    assert "u_s" in st["note"]
    assert s.get(fg["ref"])["uniforms"]["u_s"]["hits"]["track"] == "kick"
    river = _path(st, "the river")
    s.add_fx(river, "grain")
    s.copy_fx(river)
    st = s.paste_fx(bub)
    assert [f["label"] for f in _member(st, "the bubbles")["fx"]] == ["grain"]
    items = s.menu("bg", bub, 0, 3, None)
    labels = [i.get("label") or i.get("head") for i in items]
    assert any(l.startswith("Paste FX from") for l in labels if l)
    assert any(i.get("catalog") == "gen" for i in items)


def test_the_montage_bars_and_its_settings(project, tmp_path):
    s = _session(project)
    st = s.set_pins(5, 7, "none")
    assert st["pins"] == {"3": "none", "5": "none", "6": "none", "7": "none"}
    st = s.set_pins(3, 6, None)
    assert st["pins"] == {"7": "none"}
    clip = tmp_path / "sea.mp4"
    clip.write_bytes(b"x")
    st = s.put_in_montage(9, 10, str(clip), 0.0)          # not the montage's clip: a BG block
    assert any(m["label"] == "sea" and m["lane"] == "bg" for m in st["lanes"])
    st = s.set_video("bars_per_clip", 2)
    assert s.video_settings()["bars_per_clip"] == 2
    assert "error" in s.set_video("triggers", "- not a mapping")
    s.save()
    assert yaml.safe_load(s.scene.read_text(encoding="utf-8"))["video"]["bars_per_clip"] == 2
    s.step(True)
    assert s.video_settings()["bars_per_clip"] is None


def test_the_video_size_is_written_in_the_project(project):
    s = _session(project)
    assert "error" in s.set_resolution("1081x1920")
    got = s.set_resolution("1080x1920")
    assert got["resolution"] == "1080x1920"
    assert yaml.safe_load(project.read_text(encoding="utf-8"))["render"]["resolution"] == "1080x1920"


def test_the_preview_keeps_the_projects_shape():
    from core.web.util import preview_res
    assert preview_res({"resolution": "480x768"}) == "400x640"
    assert preview_res({"resolution": "854x480"}) == "640x360"
    assert preview_res({"resolution": "1080x1920"}) == "360x640"
    assert preview_res({"resolution": "300x200"}) == "300x200"      # never bigger than the video


def test_a_montage_clips_end_dragged_pins_the_bars(project):
    s = _session(project)
    db = s.bars()
    seg = lambda a, b, name: {"t0": db[a - 1], "t1": db[b], "name": name, "path": f"clips/{name}.mp4"}
    pins = lambda: {int(k): v for k, v in s.pins().items()}
    had = pins()
    s.montage = [seg(1, 2, "a"), seg(3, 4, "b"), seg(5, 8, "c")]
    s.resize_run(3, 4, 3, 6)                       # b takes bars 5-6 from c
    assert pins() == {**had, 3: "b", 4: "b", 5: "b", 6: "b"}
    s.montage = [seg(1, 2, "a"), seg(3, 6, "b"), seg(7, 8, "c")]
    s.resize_run(3, 6, 4, 6)                       # its first bar goes back to a
    assert pins() == {**had, 3: "a", 4: "b", 5: "b", 6: "b"}
    assert "error" in s.resize_run(3, 6, 7, 6)
    assert "error" in s.resize_run(9, 9, 9, 10)


def _empty(tmp_path, clips: bool = False) -> Session:
    from core.scene_doc import NEW_SCENE
    scene = tmp_path / "new.yaml"
    scene.write_text(NEW_SCENE.format(name="new"), encoding="utf-8")
    proj = {"audio": str(tmp_path / "missing.wav"), "midi": str(_midi(tmp_path / "song.mid")), "scene": str(scene),
            "render": {"fps": 30, "resolution": "854x480"}}
    if clips:
        (tmp_path / "clips").mkdir()
        proj["clips"] = {"dir": str(tmp_path / "clips")}
    path = tmp_path / "proj.yaml"
    path.write_text(yaml.safe_dump(proj), encoding="utf-8")
    return _session(path)


def test_a_new_scene_starts_empty_and_takes_a_generator_in_its_bars(tmp_path):
    s = _empty(tmp_path)
    st = s.state()
    assert st["lanes"] == [] and not st["has"]["clips"] and not s.dirty     # no montage, nothing to group
    s.new_member("bg", "chladni", 1, 5)
    layers = s.video_cfg()["layers"]
    assert [l["source"] for l in layers] == ["group", "group"]
    assert layers[0]["layers"][0]["source"] == "chladni" and layers[0]["layers"][0]["bars"] == [[1, 5]]
    assert "error" in s.set_pins(2, 2, "none")                  # no montage: nothing to pin


def test_the_montage_is_an_item_added_and_taken_out(tmp_path):
    s = _empty(tmp_path)
    got = s.add_montage()
    assert got["needs"] == "clips_dir"                          # it needs the clips folder
    (tmp_path / "with").mkdir()
    s = _empty(tmp_path / "with", clips=True)
    st = s.add_montage(3, 6)
    assert st["has"]["clips"] and st["select"] == {"clips": True}
    assert s.specs()[0] == {"source": "clips", "bars": [[3, 6]]} and s.video["clip_per_bar"] is True
    assert s.state()["montage_bars"] == [[3, 6]]
    assert "error" in s.add_montage()                           # one montage
    s.set_montage_bars()
    assert s.specs()[0] == {"source": "clips"}                  # all the song
    s.set_pins(2, 2, "none")
    st = s.remove_montage()
    assert not st["has"]["clips"] and all(x.get("source") == "group" for x in s.specs())
    s.step(True)                                                # undo: it is back, its pin too
    assert s.specs()[0] == {"source": "clips", "pins": {2: "none"}}


def test_without_a_montage_the_cuts_are_where_the_bg_blocks_begin(tmp_path):
    s = _empty(tmp_path)
    s.new_member("bg", "chladni", 1, 4)
    s.new_member("bg", "chladni", 5, 8)
    db = s.bars()
    assert s.cut_times() == [pytest.approx(db[4])]               # bar 5 begins


def test_each_stem_with_sound_is_a_timeline_lane():
    import numpy as np
    from core.web.load import STEM_HZ, stem_waves

    class Extractor:
        times = np.arange(0, 400) / 100.0                 # 4 s at 100 frames a second
        stems_energy = {"vocals": np.zeros(400),        # the AI's, skipped: no sound
                        "guitar": np.r_[np.zeros(200), np.full(200, 0.5)],
                        "piano": np.r_[np.full(100, 0.25), np.zeros(300)]}
    got = stem_waves(Extractor())
    assert [w["name"] for w in got] == ["guitar", "piano"] and got[0]["hz"] == STEM_HZ
    assert got[0]["values"][2 * STEM_HZ - 1] == 0.0 and got[0]["values"][2 * STEM_HZ] == 0.5
    assert max(got[1]["values"][:STEM_HZ]) == 0.25 and max(got[1]["values"][STEM_HZ:]) == 0.0


def test_who_plays_offers_the_projects_tracks_and_counts_the_hits(project):
    s = _session(project)
    midi = s.midi_tracks()
    assert midi[0]["track"] == "kick" and midi[0]["count"] == 64
    assert midi[0]["pitches"] == [{"pitch": 36, "name": "C2", "count": 64}]
    assert s.default_hits() == {"track": "kick"}                       # the most played track, not note 36
    st = s.knob_open(["BG", "layers", 0], ["zoom_pulse"], "trigger")["state"]
    assert st["source"]["kind"] == "track" and st["source"]["track"] == "kick"
    got = s.knob_check(st, "trigger")
    assert got["hits"] == 64 and got["first_bar"] == 1 and got["last_bar"] == 16


def test_a_track_with_no_name_is_written_by_its_index():
    from core import knob_forms as kf
    st = kf.state_of({"hits": {"track": 2}})
    assert kf.spec_of(st) == {"hits": {"track": 2}}


def test_a_new_who_plays_anywhere_is_the_projects_own_and_a_link_shows_its_hits(project):
    s = _session(project)
    st = s.knob_new("hits")                                  # a Link's "a pulse on each hit"
    assert st["source"]["kind"] == "track" and st["source"]["track"] == "kick"
    got = s.knob_check(st, "knob")
    assert got["hits"] == 64 and len(got["hit_times"]) == 64 and "values" in got
    rate = s.knob_new("rate")                                # what moves a rate: hits, the project's
    assert rate["from"]["source"]["track"] == "kick"
    assert s.knob_check(rate, "knob")["hits"] == 64           # the hits that move it
    assert s.knob_check(s.knob_new("constant"), "knob").get("hits") is None
    ref = ["BG", "layers", 0]
    s.add_trigger(ref, ["clip_reactivity", "triggers"])
    trs = s.get(ref)["clip_reactivity"]["triggers"]
    assert all(t.get("track") == "kick" and "notes" not in t for t in trs.values())
    s.add_gesture(ref, ["hits"], "spin")
    assert s.get(ref)["hits"][-1]["track"] == "kick"
