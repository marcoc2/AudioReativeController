"""The scene editor saves a scene without losing its comments, anchors or layout."""
from pathlib import Path

import pytest
import yaml

from arc_studio import _apply_scene_editor, _rows_to_video_cfg, _scene_editor_rows
from core.scene_doc import SceneDoc, plain

SCENE = """\
# header: how to render
video:
  clip_per_bar: true
  bars_per_clip: 2
  clip_order: shuffle
  layers:
    - source: clips                 # the cut schedule
    - source: shadertoy
      file: shaders/shadertoy/perolas_arc.glsl
      uniforms:
        u_kick: &kick {hits: {track: drums (midi), notes: [36]}, envelope: 0.3}
        u_depth: 9                  # how many inversions
    - source: meters
      title: pearls
      lanes:
        - {label: kick -> swell, knob: *kick, range: [0, 1]}
"""


def _editor(doc):
    cpb, order, extra, trigs, layers = _scene_editor_rows(doc)
    base = _rows_to_video_cfg(trigs, cpb, order, extra)
    return cpb, order, extra, trigs, layers, base


def _save(doc, editor):
    cpb, order, extra, trigs, layers, base = editor
    _apply_scene_editor(doc, base, trigs, cpb, order, extra, layers)
    return doc.dumps()


def test_an_untouched_save_writes_every_example_back_as_it_was():
    checked = 0
    for path in sorted(Path("examples").glob("*.yaml")) + sorted(Path("scenes").glob("*.yaml")):
        doc = SceneDoc.load(str(path))
        before = doc.dumps()           # ruamel's own round trip (3 files rewrap a long flow line)
        assert _save(doc, _editor(doc)) == before, path
        checked += 1
    assert checked >= 100


def test_a_layer_shows_the_anchors_other_layers_share():
    doc = SceneDoc.from_text(SCENE)
    *_, layers, _ = _editor(doc)
    assert "u_kick: &kick {hits:" in layers[1]["params"]
    assert "# how many inversions" in layers[1]["params"]
    assert "knob: *kick" in layers[2]["params"]


def test_editing_the_anchored_knob_moves_its_lane_too():
    doc = SceneDoc.from_text(SCENE)
    ed = _editor(doc)
    layers = ed[4]
    layers[1]["params"] = layers[1]["params"].replace("envelope: 0.3", "envelope: 0.5")
    out = _save(doc, ed)
    assert "u_kick: &kick {hits: {track: drums (midi), notes: [36]}, envelope: 0.5}" in out
    assert "knob: *kick" in out
    assert "# header: how to render" in out and "# the cut schedule" in out
    scene = yaml.safe_load(out)["video"]["layers"]
    assert scene[2]["lanes"][0]["knob"]["envelope"] == 0.5


def test_a_widget_changes_its_own_line_only():
    doc = SceneDoc.from_text(SCENE)
    ed = _editor(doc)
    ed[4][1]["opacity"] = 0.6
    out = _save(doc, ed)
    assert out.replace("      opacity: 0.6\n", "") == SCENE


def test_a_new_layer_can_use_an_anchor_and_a_removed_one_goes():
    doc = SceneDoc.from_text(SCENE)
    ed = _editor(doc)
    layers = ed[4]
    layers.append({"source": "rgb_split", "enabled": True, "file": "", "blend": "normal",
                   "opacity": 1.0, "bars": "[[3, 5]]", "params": "amount: 8\ntrigger: *kick\n"})
    del layers[2]                      # the meters strip
    out = _save(doc, ed)
    scene = yaml.safe_load(out)["video"]["layers"]
    assert [l["source"] for l in scene] == ["clips", "shadertoy", "rgb_split"]
    assert scene[2]["trigger"] == scene[1]["uniforms"]["u_kick"]
    assert scene[2]["bars"] == [[3, 5]]
    assert "trigger: *kick" in out and "bars: [[3, 5]]" in out


def test_an_unknown_alias_names_its_layer():
    doc = SceneDoc.from_text(SCENE)
    ed = _editor(doc)
    ed[4][2]["params"] = "lanes:\n  - {label: x, knob: *snare}\n"
    with pytest.raises(ValueError, match=r"layer 3 \(meters\): no anchor for \*snare"):
        _save(doc, ed)
    assert doc.dumps() == SceneDoc.from_text(SCENE).dumps()   # nothing touched


def test_a_new_trigger_is_added_and_the_rest_stays():
    doc = SceneDoc.from_text(SCENE)
    ed = _editor(doc)
    ed[3].append({"name": "snare", "source": "notes", "notes": "38", "actions": "next_clip",
                  "until": "", "min_vel": 0, "grav_on": False, "_extra": {}})
    out = _save(doc, ed)
    assert out.startswith(SCENE.split("  layers:")[0])
    video = yaml.safe_load(out)["video"]
    assert video["triggers"] == {"snare": {"notes": [38], "actions": ["next_clip"]}}
    assert plain(SceneDoc.from_text(SCENE).video()["layers"]) == video["layers"]


def test_comments_above_layers_follow_them_when_layers_come_and_go():
    from core.scene_doc import SceneDoc, plain
    text = ("video:\n  layers:\n    - source: clips\n    # the tint\n    - source: solid\n"
            "    # the grain, last\n    - source: grain\n")
    doc = SceneDoc.from_text(text)
    nodes = doc.layers()
    items = [(n, plain(n), plain(n)) for n in nodes]
    items.insert(1, (None, None, {"source": "footage", "fit": "cover"}))
    del items[2]                                                 # the tint goes
    doc.set_layers(items)
    out = doc.dumps()
    assert "# the tint" not in out
    assert out.index("# the grain, last") < out.index("source: grain")
    assert out.index("source: footage") < out.index("# the grain, last")


def test_plain_keeping_signals_writes_back_still_tied():
    from core.scene_doc import SceneDoc, plain, plain_keeping_signals
    doc = SceneDoc.from_text("video:\n  layers:\n    - source: grain\n      amount: &soft {hits: {track: kick}}\n"
                             "      bars: [[1, 4]]\n    - source: rgb_split\n      amount: *soft\n")
    nodes = doc.layers()
    items = [[n, plain(n), plain_keeping_signals(n)] for n in nodes]
    items[1][2] = {**items[1][2], "bars": [[5, 8]]}               # the second layer edited, its signal not
    doc.set_layers([tuple(it) for it in items])
    out = doc.dumps()
    assert "amount: &soft {hits: {track: kick}}" in out and "amount: *soft" in out
