"""The catalog's demos (core/demos.py): a recipe for every entry, the hash that says when one
is out of date, the backdrops, the silent loop, one demo made end to end."""
import json

import numpy as np
import pytest

from core import catalog, demos


@pytest.fixture
def tmp_demos(tmp_path, monkeypatch):
    monkeypatch.setattr(demos, "DEMOS", tmp_path / "demos")
    return tmp_path / "demos"


def test_every_entry_has_a_recipe_its_needs_are_met_by():
    for e in catalog.entries():
        rec = demos.recipe(e)
        assert rec["backdrop"] in demos.BACKDROPS, e.key
        assert rec["layer"].get("source") == e.source, e.key
        if e.kind == catalog.GENERATOR and e.key not in demos._curated():
            assert rec["backdrop"] == "black", e.key
        if "masks" in e.needs:
            assert rec["backdrop"] == "figures", e.key
        if "cuts" in e.needs:
            assert rec["backdrop"] in ("cuts", "figures"), e.key
        if rec["backdrop"] != "figures":
            assert "mask" not in rec["layer"], e.key


def test_a_tuned_scene_is_used_with_the_loops_transcription():
    rec = demos.recipe(catalog.by_key("shadertoy:octagrams_arc"))
    assert rec["origin"] == "examples/me_diz_acorde.yaml"
    text = json.dumps(rec["layer"])
    assert "input/me_diz/sheetsage" not in text and "_loop/score" in text


def test_the_hash_follows_the_recipe_and_the_code(monkeypatch):
    e = catalog.by_key("shadertoy:droste_arc")
    h = demos.digest(e)
    assert demos.digest(e) == h
    monkeypatch.setattr(demos, "_code_of", lambda entry: "void main() {}  // edited")
    assert demos.digest(e) != h
    monkeypatch.undo()
    monkeypatch.setattr(demos, "_curated", lambda: {e.key: {"backdrop": "clip"}})
    assert demos.digest(e) != h


def test_a_demos_state(tmp_demos):
    e = catalog.by_key("grain")
    assert demos.status(e) == "missing"
    tmp_demos.mkdir(parents=True)
    demos.video_path(e.key).write_bytes(b"x")
    demos.info_path(e.key).write_text(json.dumps({"state": "ready", "hash": demos.digest(e)}))
    assert demos.status(e) == "ready"
    demos.info_path(e.key).write_text(json.dumps({"state": "ready", "hash": "an old one"}))
    assert demos.status(e) == "stale"
    demos.info_path(e.key).write_text(json.dumps({"state": "failed", "error": "boom"}))
    assert demos.status(e) == "failed"
    assert demos.file_of("shadertoy:droste_arc") == "shadertoy__droste_arc"


@pytest.mark.parametrize("kind", ["black", "grid", "figure"])
def test_the_drawn_backdrops(kind):
    b = demos.Backdrop(kind, 96, 64)
    a, c = b.frame_at(1.0), b.frame_at(1.5)
    assert a.shape == (64, 96, 3) and a.dtype == np.uint8
    assert (kind == "black") == (not np.any(a != c))      # the drawn ones move
    assert list(b.cut_times(17.0)) == [4.0, 8.0, 12.0, 16.0]   # a cut every two bars at 120 BPM


def test_the_loop_has_the_tracks_the_tuned_scenes_name(tmp_demos):
    names = {n.track for n in demos.loop_notes()}
    assert {"drums (midi)", "organ", "e-piano", "bass acid"} <= names
    kicks = [n.time for n in demos.loop_notes() if n.pitch == 36 and n.time < 2.0]
    assert kicks[:3] == [0.0, 0.75, 1.0]


def test_the_knobs_the_music_moves_are_the_meters_lanes():
    layer = {"source": "shadertoy", "file": "x.glsl",
             "uniforms": {"u_kick": {"hits": {"track": "drums (midi)", "notes": [36]}}, "u_size": 0.4},
             "amount": "subbands.bass", "direction": [1, 2]}
    assert [l["label"] for l in demos._knob_lanes(layer)] == ["u_kick", "amount"]


def test_a_demo_made_end_to_end(tmp_demos, monkeypatch):
    """Half a second of grain over the grid: the video, its info, then up to date."""
    import cv2
    monkeypatch.setattr(demos, "LOOP_SECONDS", 8.0)
    monkeypatch.setattr(demos, "START", 4.0)
    monkeypatch.setattr(demos, "SECONDS", 0.5)
    monkeypatch.setattr(demos, "PREROLL", 0.1)
    monkeypatch.setattr(demos, "_curated", lambda: {"grain": {"backdrop": "grid"}})
    e = catalog.by_key("grain")
    out = demos.render(e, demos.song(), log=lambda *a: None)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 15
    ok, frame = cap.read()
    assert ok and frame.shape == (demos.SIZE, demos.SIZE, 3)
    cap.release()
    assert demos.read_info("grain")["state"] == "ready" and demos.status(e) == "ready"
    assert demos.to_make({"grain"}) == []
