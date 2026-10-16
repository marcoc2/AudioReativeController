"""The Arrange window, driven without a screen: the BG and FG lanes, blocks dragged, FX on a
block, a lane and the master, undo, save."""
import numpy as np
import pytest
import yaml

dpg = pytest.importorskip("dearpygui.dearpygui")

SCENE = """# a scene
video:
  layers:
    - source: clips
    - source: feedback
      bars: [[1, 2]]
    # the dancer
    - source: footage
      file: lion.mp4
      loop: true
      bars: [[3, 4]]
    - source: grain
"""


class _Host:
    def __init__(self, path):
        from core.rhythm.grid import RhythmGrid
        self.path = path
        self._grid = RhythmGrid(bpm=120.0, fps=30, beats=np.arange(64) * 0.5, downbeats=np.arange(16) * 2.0)
        self.saved = 0

    def scene_path(self):
        return str(self.path)

    def grid(self):
        return self._grid

    def notes(self):
        return []

    def clips_dir(self):
        return ""

    def duration(self):
        return 32.0

    def log(self, msg):
        pass

    def status(self, msg):
        pass

    def reload_editor(self):
        self.saved += 1


@pytest.fixture
def window(tmp_path):
    import arc_arrange as R
    path = tmp_path / "scene.yaml"
    path.write_text(SCENE, encoding="utf-8")
    dpg.create_context()
    host = _Host(path)
    R.build(host)
    R.open_window()
    yield R, host, path
    dpg.destroy_context()


def _x(R, bar):
    return R._x_of(bar) + R._bw() / 2                     # the middle of a bar


def _y(R, lane, sub=0):
    """The middle of a lane's sub-row (sub -1: its empty row, where new things go)."""
    rows = [r for r, row in enumerate(R.A.rows) if row.get("lane") == lane]
    return R._row_y(rows[sub]) + 10


def _layers(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["video"]["layers"]


def test_a_flat_scene_is_grouped_into_bg_fg_and_master(window):
    R, host, path = window
    assert R.A.converted
    assert [r["kind"] for r in R.A.rows] == ["master", "lane", "lane", "lane", "montage"]
    assert [m.label for m in R.A.members["FG"]] == ["lion"] and R.A.members["BG"] == []
    R.btn_save()
    layers = _layers(path)
    assert [(l["source"], l.get("name")) for l in layers] == [("clips", None), ("group", "BG"),
                                                              ("group", "FG"), ("grain", None)]
    bg, fg = layers[1], layers[2]
    assert bg["canvas"] == "below" and bg["fx"] == [{"source": "feedback", "bars": [[1, 2]]}]
    assert fg["layers"][0]["file"] == "lion.mp4" and fg["layers"][0]["bars"] == [[3, 4]]
    assert not R.A.converted and host.saved == 1                         # grouped now


def test_drag_makes_moves_and_resizes_blocks(window):
    R, host, path = window
    y = _y(R, "FG")
    lion = lambda: [(b.first, b.last) for b in R.A.members["FG"][0].blocks]
    R.press(_x(R, 3), y)                                  # move the lion two bars on
    R.release(_x(R, 5))
    assert lion() == [(5, 6)]
    R.press(R._x_of(6 + 1), y)                            # its end: one bar longer
    R.release(_x(R, 7))
    assert lion() == [(5, 7)]
    R.press(_x(R, 6), y)                                  # a click: it is chosen
    R.release(_x(R, 6))
    assert R.A.sel == ("block", "FG", 0, 0)
    R.btn_save()
    member = _layers(path)[2]["layers"][0]
    assert member["blocks"] == [{"bars": [5, 7]}] and "bars" not in member


def test_clips_and_generators_put_in_a_lane_stack_up(window):
    R, host, path = window
    R.A.content = ("clip", "C:/v/a/final.mp4", 3.5)        # as if picked in the browser
    R.press(_x(R, 1), _y(R, "BG", -1))
    R.release(_x(R, 4))
    R.A.content = ("gen", "shadertoy:nebulosa_arc")
    R.press(_x(R, 3), _y(R, "BG", -1))                    # at the same time: a row of its own
    R.release(_x(R, 6))
    bg = R.A.members["BG"]
    assert [(m.label, m.kind) for m in bg] == [("final", "clip"), ("nebulosa_arc", "generator")]
    assert bg[0].row != bg[1].row
    R.btn_save()
    layers = _layers(path)[1]["layers"]
    assert layers[0] == {"source": "footage", "role": "background", "fit": "cover", "loop": True,
                         "file": "C:/v/a/final.mp4", "blocks": [{"bars": [1, 4], "from": 3.5}]}
    assert layers[1] == {"source": "shadertoy", "file": "shaders/shadertoy/nebulosa_arc.glsl", "bars": [[3, 6]]}


def test_fx_on_a_block_a_lane_and_the_master(window):
    R, host, path = window
    fg = R._lane_item("FG")[3]
    R.add_fx(fg, ("layers", 0, "fx"), "pixel_sort")      # on the lion only
    R.add_fx(fg, ("layers", 0, "fx"), "rgb_split")
    R._set_fx_bars(fg, ("layers", 0, "fx", 0), "3-3")
    R._swap(fg, ("layers", 0, "fx"), 0, 1)               # rgb_split first
    R.add_fx(fg, ("fx",), "echoes")                      # on the whole FG
    R._edit(fg, (), lambda g: {**g, "spill": True})
    R.add_master("rgb_noise")
    R.open_plugin(fg, ("layers", 0, "fx", 0))
    assert "rgb_split - FX on lion" in dpg.get_item_label("arr_plugin")
    assert R.set_param(fg, ("layers", 0, "fx", 0, "mix"), 0.6) == ""
    grain = next(it[3] for it in R._master_items() if it[2]["source"] == "grain")
    R._toggle(grain, ())
    R.A.sel = ("block", "FG", 0, 0)
    R._draw_inspector()                                   # the block's FX list draws
    R.A.sel = ("master",)
    R._draw_inspector()
    R.context_menu(_x(R, 3), _y(R, "FG"))                 # the menus build
    R.context_menu(_x(R, 3), _y(R, "BG", -1))
    R.btn_save()
    layers = _layers(path)
    lion = layers[2]["layers"][0]
    assert lion["fx"] == [{"source": "rgb_split", "mix": 0.6}, {"source": "pixel_sort", "bars": [[3, 3]]}]
    assert layers[2]["fx"] == [{"source": "echoes"}] and layers[2]["spill"] is True
    assert [l["source"] for l in layers[3:]] == ["grain", "rgb_noise"] and layers[3]["enabled"] is False
    text = path.read_text(encoding="utf-8")
    assert "- {source: pixel_sort, bars: [[3, 3]]}" in text


def test_undo_and_redo(window):
    R, host, path = window
    R.A.content = ("gen", "julia")
    R.press(_x(R, 5), _y(R, "BG", -1))
    R.release(_x(R, 6))
    R.add_master("grain")
    assert len(R.A.members["BG"]) == 1 and len(R._master_items()) == 2
    R.btn_undo()
    assert len(R._master_items()) == 1
    R.btn_undo()
    assert R.A.members["BG"] == []
    R.btn_redo()
    assert [m.label for m in R.A.members["BG"]] == ["julia"]
    R.btn_save()                                          # the history goes on after a save
    R.btn_undo()
    assert R.A.members["BG"] == []


def test_deleting_the_last_block_takes_the_member(window):
    R, host, path = window
    R._delete_block("FG", 0, 0)
    assert R.A.members["FG"] == []                        # not left playing the whole song


def test_the_generators_and_fx_browser(window):
    R, host, path = window
    headers = dpg.get_item_children("arr_cat_generator", 1)     # drawn when the window opens
    assert headers and dpg.get_item_label(headers[0]).startswith("Fractals")
    dpg.set_value("arr_cat_q_fx", "grain")
    R._draw_catalog("fx")
    (header,) = dpg.get_item_children("arr_cat_fx", 1)
    (item,) = [i for i in dpg.get_item_children(header, 1) if dpg.get_item_type(i).endswith("Selectable")]
    R._on_pick_entry(item, True, dpg.get_item_user_data(item))
    assert R.A.entry == "grain"
    shown = " ".join(str(dpg.get_value(t)) for t in dpg.get_item_children("arr_cat_info_fx", 1))
    assert "FX · Colour & light" in shown and "amount = 0.06" in shown
    R._on_pick_entry(item, True, "shadertoy:nebulosa_arc")     # a Generator: what a drag puts
    assert R.A.content == ("gen", "shadertoy:nebulosa_arc")


def test_signals_named_in_one_plugin_used_in_another(window):
    R, host, path = window
    R.A.content = ("gen", "shadertoy:nebulosa_arc")
    R.press(_x(R, 1), _y(R, "BG", -1))
    R.release(_x(R, 2))
    bg, fg = R._lane_item("BG")[3], R._lane_item("FG")[3]
    kick = {"hits": {"track": "kick"}, "envelope": 0.3}
    defines = R.PluginTarget(bg, "nebulosa_arc")                       # the BG comes first in the file
    assert defines.apply(("layers", 0, "uniforms", "u_form"), kick, "kick", {}) is None
    assert defines.value(("layers", 0, "uniforms", "u_form")) == (kick, "&kick")
    R.add_fx(fg, ("layers", 0, "fx"), "grain")
    uses = R.PluginTarget(fg, "grain")
    uses.apply(("layers", 0, "fx", 0, "burst"), None, None, {(): uses.anchors()["kick"]})
    assert uses.value(("layers", 0, "fx", 0, "burst"))[1] == "*kick"
    defines.apply(("layers", 0, "uniforms", "u_form"), {"hits": {"track": "snare"}}, None, {})
    assert uses.value(("layers", 0, "fx", 0, "burst"))[0] == {"hits": {"track": "snare"}}
    R.btn_save()
    text = path.read_text(encoding="utf-8")
    assert "&kick" in text and "*kick" in text


def test_who_plays_a_trigger_keeps_its_envelope(window):
    R, host, path = window
    fg = R._lane_item("FG")[3]
    R.add_fx(fg, ("layers", 0, "fx"), "grain")
    R.set_param(fg, ("layers", 0, "fx", 0, "trigger"), {"track": "piano", "envelope": 0.4})
    assert R._is_trigger("trigger", {"track": "piano", "envelope": 0.4})
    assert not R._is_trigger("amount", {"hits": {"track": "kick"}}) and R._is_trigger("trigger", None)
    R.open_plugin(fg, ("layers", 0, "fx", 0))                            # draws its "Who plays"
    R.PluginTarget(fg, "grain").apply(("layers", 0, "fx", 0, "trigger"), {"track": "snare"}, None, {})
    assert R._item(fg)[2]["layers"][0]["fx"][0]["trigger"] == {"envelope": 0.4, "track": "snare"}


def test_zoom_shows_fewer_bars_and_keeps_the_mouse_bar(window):
    R, host, path = window
    assert R._shown() == 16 and R._bar_of_x(_x(R, 9)) == 9
    R.set_view(zoom=4.0, keep=9)                          # 4 bars across, bar 9 where it was
    assert R._shown() == 4 and R._bar_of_x(_x(R, 9)) == 9
    R.set_view(first=13)
    assert R.A.view0 == 13 and R._bar_of_x(R.LABEL_W + 1) == 13
    R.set_view(first=99)                                  # not past the end
    assert R.A.view0 == 13
    y = _y(R, "FG")
    R.set_view(zoom=2.0, first=3)                         # bars 3-10: the lion's block still drags
    R.press(_x(R, 3), y)
    R.release(_x(R, 4))
    assert [(b.first, b.last) for b in R.A.members["FG"][0].blocks] == [(4, 5)]
    R.set_view(zoom=1.0, first=1)
