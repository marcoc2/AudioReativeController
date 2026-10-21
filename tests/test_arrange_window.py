"""The Arrange window, driven without a screen: the BG and FG lanes, blocks dragged, FX on a
block, a lane and the master, undo, save."""
import sys
from pathlib import Path

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

    size = "480x854  9:16 portrait"

    def resolutions(self):
        return ["480x854  9:16 portrait", "1080x1920  9:16 portrait", "1920x1080  16:9 landscape"]

    def resolution(self):
        return self.size

    def set_resolution(self, v):
        self.size = v
        return ""


@pytest.fixture
def window(tmp_path):
    import arc_arrange as R
    path = tmp_path / "scene.yaml"
    path.write_text(SCENE, encoding="utf-8")
    dpg.create_context()
    host = _Host(path)
    R.STATE_FILE = tmp_path / "arrange.json"              # never the studio's own
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
    assert [(r["kind"], r.get("lane")) for r in R.A.rows] == [
        ("lane", "master"), ("lane", "FG"), ("lane", "FG"), ("lane", "FG fx"), ("lane", "BG"), ("lane", "BG fx"),
        ("montage", None)]                                # each track, a lane's with a row for new things
    assert [m.label for m in R.A.members["FG"]] == ["the dancer"] and R.A.members["BG"] == []  # its comment
    R.btn_save()
    layers = _layers(path)
    assert [(l["source"], l.get("name")) for l in layers] == [("clips", None), ("group", "BG"),
                                                              ("group", "FG"), ("grain", None)]
    bg, fg = layers[1], layers[2]
    assert bg["canvas"] == "below" and bg["fx"] == [{"source": "feedback", "bars": [[1, 2]]}]
    assert fg["layers"][0]["file"] == "lion.mp4" and fg["layers"][0]["bars"] == [[3, 4]]
    assert fg["layers"][0]["name"] == "the dancer"                      # kept, now the comment is gone
    R._rename("FG", 0, "Leo")
    assert R.A.members["FG"][0].label == "Leo"
    R._rename("FG", 0, "")
    assert R.A.members["FG"][0].label == "lion"                         # its file's
    assert not R.A.converted and host.saved == 1                         # grouped now


def test_drag_makes_moves_and_resizes_blocks(window, monkeypatch):
    R, host, path = window
    y = _y(R, "FG")
    lion = lambda: [(b.first, b.last) for b in R.A.members["FG"][0].blocks]
    R.press(_x(R, 3), y)                                  # move the lion two bars on
    R.release(_x(R, 5))
    assert lion() == [(5, 6)]
    R.press(R._x_of(6 + 1) - 2, y)                        # its end, no Shift: the block moves
    R.release(R._x_of(6 + 1) - 2 + R._bw())
    assert lion() == [(6, 7)]
    monkeypatch.setattr(R, "_shift", lambda: True)
    R.press(R._x_of(7 + 1), y)                            # Shift: its end, one bar longer
    R.release(_x(R, 8))
    assert lion() == [(6, 8)]
    R.press(R._x_of(6), y)                                # Shift: its start, one bar sooner
    R.release(_x(R, 5))
    assert lion() == [(5, 8)]
    monkeypatch.setattr(R, "_shift", lambda: False)
    R.press(_x(R, 6), y)                                  # a click: it is chosen
    R.release(_x(R, 6))
    assert R.A.sel == ("block", "FG", 0, 0)
    R.btn_save()
    member = _layers(path)[2]["layers"][0]
    assert member["blocks"] == [{"bars": [5, 8]}] and "bars" not in member


def test_where_two_blocks_meet_shift_takes_the_end_of_the_one_the_mouse_is_in(window, monkeypatch):
    R, host, path = window
    R._set_blocks("FG", 0, [R.arr.Block(3, 4), R.arr.Block(5, 6)])   # the lion: two blocks that meet
    y = _y(R, "FG")
    seam = R._x_of(5)
    monkeypatch.setattr(R, "_shift", lambda: True)
    assert R._edge_at(seam - 3, y)[1:3] == ("last", (0, 0))           # in the first: its end
    assert R._edge_at(seam + 3, y)[1:3] == ("first", (0, 1))          # in the second: its start
    assert R._edge_at(R._x_of(4) + 2, y) is None                       # mid-block: no end
    R.press(seam + 3, y)                                               # the second's start, later
    R.release(_x(R, 6))
    assert [(b.first, b.last) for b in R.A.members["FG"][0].blocks] == [(3, 4), (6, 6)]
    shapes = []
    monkeypatch.setattr(R.arc_window, "set_cursor", shapes.append)
    R.A.drag = (0, 6, "last", (0, 1))                                  # an end being dragged: the arrows
    R._show_edge()
    R.A.drag = None
    monkeypatch.setattr(R, "_shift", lambda: False)                    # no Shift: the plain cursor
    R._show_edge()
    assert shapes == ["resize", None]


def test_a_new_member_stays_on_the_sub_row_it_was_made_on(window):
    """A row with room in those bars would take it (the topmost free one): it stays where it
    was made, there after a reload too; one in the way of it falls to a free row."""
    R, host, path = window

    def by_file(stem):
        return next(m for m in R.A.members["FG"] if Path(str(m.spec.get("file"))).stem == stem)

    def rows_by_file():
        return {Path(str(m.spec.get("file"))).stem: m.row for m in R.A.members["FG"]}

    R.A.content = ("clip", "C:/v/a/one.mp4", 0.0)
    R.press(_x(R, 9), _y(R, "FG", -1))                    # under the lion's row: a row of its own
    R.release(_x(R, 10))
    rows = rows_by_file()
    assert rows["one"] == 1 and rows["lion"] == 0
    R.A.content = ("clip", "C:/v/a/two.mp4", 0.0)
    R.press(_x(R, 12), _y(R, "FG", 1))                    # on one's row, though the lion's has room
    R.release(_x(R, 13))
    assert rows_by_file()["two"] == 1
    R.btn_save()
    assert rows_by_file() == {"lion": 0, "one": 1, "two": 1}
    R.A.content = ("clip", "C:/v/a/three.mp4", 0.0)
    R.press(_x(R, 3), _y(R, "FG", -1))                    # a third row, then deleted: rows close up
    R.release(_x(R, 4))
    three = by_file("three")
    assert three.row == 2
    R._remove_member("FG", by_file("one").at)
    R._remove_member("FG", by_file("two").at)
    assert by_file("three").row == 1


def test_save_says_when_and_where_it_saved(window):
    R, host, path = window
    R._show_saved()
    assert R.dpg.get_value("arr_save_state") == "" and R.dpg.get_item_label("arr_save_btn") == "Save to scene"
    R.A.content = ("clip", "C:/v/a/one.mp4", 0.0)
    R.press(_x(R, 9), _y(R, "FG", -1))
    R.release(_x(R, 10))
    R._show_saved()
    assert R.dpg.get_value("arr_save_state") == "unsaved changes"
    assert R.dpg.get_item_label("arr_save_btn") == "Save to scene *"
    R.btn_save()
    assert R.dpg.get_item_label("arr_save_btn") == "Saved!"
    assert R.dpg.get_value("arr_save_state").startswith("saved at ") and path.name in R.dpg.get_value("arr_save_state")
    assert str(path) in R.dpg.get_value("arr_status")
    R.A.saved_flash -= R.SAVED_FLASH                     # a moment later
    R._show_saved()
    assert R.dpg.get_item_label("arr_save_btn") == "Save to scene"


def test_clips_and_generators_put_in_a_lane_stack_up(window):
    R, host, path = window
    R.A.content = ("clip", "C:/v/a/final.mp4", 3.5)        # as if picked in the browser
    R.press(_x(R, 1), _y(R, "BG", -1))
    R.release(_x(R, 4))
    R.A.content = ("gen", "shadertoy:nebulosa_arc")
    R.press(_x(R, 3), _y(R, "BG", -1))                    # at the same time, on the row under it
    R.release(_x(R, 6))
    bg = R.A.members["BG"]                                # (under it in the picture too: first in the lane)
    assert [(m.label, m.kind, m.row) for m in bg] == [("nebulosa_arc", "generator", 1), ("final", "clip", 0)]
    R.btn_save()
    layers = _layers(path)[1]["layers"]
    assert layers[1] == {"source": "footage", "role": "background", "fit": "cover", "loop": True,
                         "file": "C:/v/a/final.mp4", "blocks": [{"bars": [1, 4], "from": 3.5}], "lane_row": 0}
    assert layers[0] == {"source": "shadertoy", "file": "shaders/shadertoy/nebulosa_arc.glsl", "bars": [[3, 6]],
                         "lane_row": 1}


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


def test_ctrl_wheel_zooms_around_the_mouse_alt_moves_along_the_wheel_scrolls(window, monkeypatch):
    R, host, path = window
    held = {"mvKey_LControl"}
    monkeypatch.setattr(R.dpg, "is_item_hovered", lambda item: item in ("arr_canvas", "arr_tracks"))
    monkeypatch.setattr(R, "_down", lambda *keys: bool(held & set(keys)))
    monkeypatch.setattr(R, "_mouse_in_canvas", lambda: (_x(R, 9) + 1, 40))
    R.on_wheel(None, 1)                                   # Ctrl: zoom in, bar 9 stays put
    assert R._shown() < 16 and R._bar_of_x(_x(R, 9) + 1) == 9
    while R._shown() > 4:
        R.on_wheel(None, 1)
    assert R._shown() == 4 and R._bar_of_x(_x(R, 9) + 1) == 9
    first = R.A.view0
    held = {"mvKey_LAlt"}                                 # Alt: along the song, same zoom
    R.on_wheel(None, -1)
    assert R.A.view0 == first + 1 and R._shown() == 4
    scroll = [0.0]
    monkeypatch.setattr(R.dpg, "get_y_scroll", lambda item: scroll[0])
    monkeypatch.setattr(R.dpg, "get_y_scroll_max", lambda item: 200.0)
    monkeypatch.setattr(R.dpg, "set_y_scroll", lambda item, v: scroll.__setitem__(0, v))
    held = set()                                          # no key: the lanes scroll, nothing zooms
    R.on_wheel(None, -1)
    assert scroll[0] == 3 * R.A.lane_h and R._shown() == 4 and R.A.view0 == first + 1
    R.on_wheel(None, 1)
    R.on_wheel(None, 1)
    assert scroll[0] == 0.0
    held = {"mvKey_LControl"}
    for _ in range(20):
        R.on_wheel(None, -1)                              # zoom out: back to the whole song
    assert R._shown() == 16 and R.A.view0 == 1
    assert dpg_config(R, "arr_tracks", "no_scroll_with_mouse")   # the wheel is on_wheel's alone


def test_shift_wheel_makes_the_rows_taller_and_everything_follows(window, monkeypatch):
    R, host, path = window
    held = {"mvKey_LShift"}
    monkeypatch.setattr(R.dpg, "is_item_hovered", lambda item: item in ("arr_canvas", "arr_tracks"))
    monkeypatch.setattr(R, "_down", lambda *keys: bool(held & set(keys)))
    y = _y(R, "FG")
    monkeypatch.setattr(R, "_mouse_in_canvas", lambda: (_x(R, 3), y))
    shown = R._shown()
    R.on_wheel(None, 1)                                   # taller rows, the same bars across
    assert R.A.lane_h == 36 and R._shown() == shown
    assert R.A.scroll_to == pytest.approx((y - R._row_y(0)) * 0.2)   # the FG row stays under the mouse
    for _ in range(20):
        R.on_wheel(None, 1)
    assert R.A.lane_h == R.LANE_MAX
    R.press(_x(R, 3), _y(R, "FG"))                        # tall rows: the lion's block still drags
    R.release(_x(R, 4))
    assert [(b.first, b.last) for b in R.A.members["FG"][0].blocks] == [(4, 5)]
    for _ in range(30):
        R.on_wheel(None, -1)
    assert R.A.lane_h == R.LANE_MIN
    assert R._row_at(_y(R, "BG")) is not None and R.A.rows[R._row_at(_y(R, "BG"))]["lane"] == "BG"
    R.set_lane_height(R.LANE_H)
    assert R.dpg.get_value("arr_rows_text") == f"{R.LANE_H} px"


def test_the_ruler_stays_on_top_when_the_lanes_scroll(window, monkeypatch):
    R, host, path = window
    R.set_lane_height(R.LANE_MAX)
    monkeypatch.setattr(R.dpg, "get_y_scroll", lambda item: 150.0)
    R._pin_ruler()
    assert R.A.head_y == 150.0                            # the ruler drawn 150 px down: at the view's top
    y = 150 + 10                                          # on the ruler as it shows
    assert R._row_at(y) is None                           # not the row scrolled under it
    R.press(_x(R, 5), y)
    R.release(_x(R, 7))
    assert R.A.span == (5, 7)                             # its bars chosen, as when not scrolled
    assert R._row_at(150 + R._row_y(0) + 1) is not None   # just under it: a row again


def test_the_timeline_widens_with_the_window(window, monkeypatch):
    R, host, path = window
    pane = [1400]
    monkeypatch.setattr(R.dpg, "is_item_shown", lambda item: True)
    monkeypatch.setattr(R.dpg, "get_item_rect_size", lambda item: [pane[0], 600])
    narrow = R._bw()
    R._fit_canvas()                                       # a wider window: wider bars, same bars shown
    assert R.A.canvas_w == 1400 - R.SCROLLBAR_W and R._bw() > narrow and R._shown() == 16
    assert dpg_config(R, "arr_canvas", "width") == R.A.canvas_w
    assert R._bar_of_x(_x(R, 12) + 1) == 12
    pane[0] = 300                                         # never under the narrowest
    R._fit_canvas()
    assert R.A.canvas_w == R.CANVAS_W
    R._fit_canvas()


def test_the_final_videos_size_is_chosen_in_the_window(window):
    R, host, path = window
    assert R.dpg.get_value("arr_video_res") == "480x854  9:16 portrait"
    assert "1920x1080  16:9 landscape" in dpg_config(R, "arr_video_res", "items")
    R.on_video_res("arr_video_res", "1920x1080  16:9 landscape")
    assert host.size == "1920x1080  16:9 landscape"


def test_the_studios_size_choosers():
    """Every preset names its shape; a chooser's line or a typed size sets it, odd sizes do not."""
    import arc_studio as S
    for line in S.RES_CHOICES:
        w, h = (int(x) for x in line.split()[0].split("x"))
        shape = line.split("  ")[1]
        assert ("portrait" in shape) == (h > w) and ("square" in shape) == (h == w)
        a, b = (int(x) for x in shape.split()[0].split(":"))
        assert abs(w / h - a / b) < 0.002 and w % 2 == 0 and h % 2 == 0      # its shape, and even (x264)
    shapes = {line.split("  ")[1] for line in S.RES_CHOICES}
    assert {"4:5 portrait", "5:4 landscape", "10:16 portrait", "16:10 landscape"} <= shapes
    dpg.create_context()
    try:
        assert S.set_resolution("1080x1920  9:16 portrait") == "" and (S.S.width, S.S.height) == (1080, 1920)
        assert S.set_resolution("1000x1000") == "" and S._res_choice(1000, 1000) == "1000x1000"
        assert S.set_resolution("1081x1920") and (S.S.width, S.S.height) == (1000, 1000)
    finally:
        S.set_resolution("480x854")
        dpg.destroy_context()


def dpg_config(R, tag, key):
    return R.dpg.get_item_configuration(tag)[key]


def test_a_cued_member_is_chosen_by_a_click_not_dragged_over(window):
    R, host, path = window
    fg = R._lane_item("FG")[3]
    R._edit(fg, ("layers",), lambda ls: ls + [{"source": "footage", "file": "bottles.mp4",
                                               "cues": [[4.0, 0.0, 2.0], [10.0, 1.0, 2.0]]}])
    bottles = next(m for m in R.A.members["FG"] if m.label == "bottles")
    R.A.cues[R._cue_key(bottles.spec)] = [(4.0, 6.0), (10.0, 12.0)]   # as the reader leaves them
    R._relane()
    bottles = next(m for m in R.A.members["FG"] if m.label == "bottles")
    sub = bottles.row
    y = _y(R, "FG", sub)
    R.press(R._x_of(R._time_to_bar(10.5)), y)            # on its second cut
    assert R.A.drag is None and R.A.sel == ("block", "FG", bottles.at, None)
    assert len(R.A.members["FG"]) == 2                    # nothing new made
    shown = " ".join(str(dpg.get_value(t)) for t in dpg.get_item_children("arr_inspector", 1)
                     if dpg.get_item_type(t).endswith("Text"))
    assert "2 cue(s) written in the scene" in shown and "2 cuts, from bar 3" in shown
    R.context_menu(R._x_of(R._time_to_bar(10.5)), y)     # its own menu
    texts = [dpg.get_value(t) for t in dpg.get_item_children("arr_menu", 1) if dpg.get_item_type(t).endswith("Text")]
    labels = [dpg.get_item_label(t) for t in dpg.get_item_children("arr_menu", 1)]
    assert texts[0] == "FG · bottles" and "Delete bottles (Delete)" in labels


def test_a_cued_character_is_hidden_in_some_bars_and_stays_cued(window):
    """Right-click a cut: Hide it (its bars), or the chosen bars, or the bar; Show again; the
    inspector's field. The cues stay (it goes on in sync underneath): only mute: changes."""
    R, host, path = window
    fg = R._lane_item("FG")[3]
    R._edit(fg, ("layers",), lambda ls: ls + [{"source": "footage", "file": "bottles.mp4",
                                               "cues": [[4.0, 0.0, 2.0], [10.0, 1.0, 4.0]]}])
    bottles = next(m for m in R.A.members["FG"] if m.label == "bottles")
    R.A.cues[R._cue_key(bottles.spec)] = [(4.0, 6.0), (10.0, 14.0)]   # bars 3, and 6-7 (2 s bars)
    R._relane()
    y = _y(R, "FG", next(m for m in R.A.members["FG"] if m.label == "bottles").row)

    def menu_at(t):
        R.context_menu(R._x_of(R._time_to_bar(t)), y)
        return {dpg.get_item_label(i): i for i in dpg.get_item_children("arr_menu", 1)
                if dpg.get_item_type(i).endswith("MenuItem")}

    def bottles_spec():
        return next(m for m in R.A.members["FG"] if m.label == "bottles").spec

    items = menu_at(10.5)
    assert "Hide this cut (bars 6-7)" in items and "Hide in bar 6" in items
    _click(items["Hide this cut (bars 6-7)"])
    spec = bottles_spec()
    assert spec["mute"] == [[6, 7]] and spec["cues"] == [[4.0, 0.0, 2.0], [10.0, 1.0, 4.0]]
    assert "hidden in this bar" in R.hover_text(R._x_of(R._time_to_bar(12.5)), y)
    items = menu_at(12.5)                                 # inside it: shown again
    assert "Show again in bars 6-7" in items and "Show again in bar 7" in items
    _click(items["Show again in bar 7"])
    assert bottles_spec()["mute"] == [[6, 6]]
    R.A.span = (2, 4)                                     # bars chosen on the ruler
    _click(menu_at(4.5)["Hide in the chosen bars 2-4"])
    assert bottles_spec()["mute"] == [[2, 4], [6, 6]]
    R._set_mutes("FG", next(m for m in R.A.members["FG"] if m.label == "bottles").at, "9-10, 3")
    assert bottles_spec()["mute"] == [[3, 3], [9, 10]]
    R._set_mutes("FG", next(m for m in R.A.members["FG"] if m.label == "bottles").at, "")
    assert "mute" not in bottles_spec()
    R.btn_undo()
    assert bottles_spec()["mute"] == [[3, 3], [9, 10]]


def _with_a_montage(host):
    """The studio's montage, as a stub: 16 bars of 2 s, a clip a bar; the clips layer's pins win."""
    names = ["lofi_2", "tororo_0", "emojibots_2", "limonada_1"] * 4

    def montage(layers=None):
        pins = next(({int(k): v for k, v in (l.get("pins") or {}).items()} for l in layers or []
                     if l.get("source", "clips") == "clips"), {})
        out = []
        for b in range(16):
            name = pins.get(b + 1, names[b])
            out.append((b * 2.0, b * 2.0 + 2.0, name, "" if name == "none" else f"C:/c/{name}.mp4"))
        return out
    host.montage = montage
    host.montage_clips = lambda: [f"C:/c/{n}.mp4" for n in set(names)]


def _montage_y(R):
    return R._row_y(next(r for r, row in enumerate(R.A.rows) if row["kind"] == "montage")) + 10


def test_the_montage_row_shows_which_clip_plays_each_bar(window):
    R, host, path = window
    _with_a_montage(host)
    R._draw()
    assert [(a, b, name) for a, b, name, _ in R.A.runs[:2]] == [(1.0, 2.0, "lofi_2"), (2.0, 3.0, "tororo_0")]
    texts = [dpg.get_item_configuration(i)["text"] for i in dpg.get_item_children("arr_canvas", 2)
             if dpg.get_item_type(i) == "mvAppItemType::mvDrawText"]
    assert "lof~" in texts                                # named as far as it fits
    y = _montage_y(R)
    assert R.hover_text(_x(R, 2), y) == "bar 2: tororo_0\nC:/c/tororo_0.mp4"  # all of it
    assert R.hover_text(_x(R, 3), _y(R, "FG")) == "the dancer (clip), bars 3-4"
    R.press(_x(R, 1), y)                                  # a click: the clip there is chosen
    R.release(_x(R, 1))
    assert R.A.sel == ("montage", 1, 1)
    shown = [dpg.get_value(i) for i in dpg.get_item_children("arr_inspector", 1)
             if dpg.get_item_type(i) == "mvAppItemType::mvText"]
    assert any("lofi_2" in t for t in shown) and any(t == "C:/c/lofi_2.mp4" for t in shown)


def test_the_montage_bars_take_another_clip_nothing_or_go_back(window):
    R, host, path = window
    _with_a_montage(host)
    R._draw()
    y = _montage_y(R)
    R.press(_x(R, 1), y)                                  # bars 1-2 chosen by a drag
    R.release(_x(R, 2))
    assert R.A.sel == ("montage", 1, 2)
    R.A.content = ("clip", "C:/c/limonada_1.mp4", 0.0)    # as if picked on the left
    labels = [label for label, _ in R._montage_actions(1, 2)]
    assert labels[0] == "Put limonada_1 here (bars 1-2)"
    R._put_in_montage(1, 2)
    assert [(a, b, name) for a, b, name, _ in R.A.runs[:2]] == [(1.0, 3.0, "limonada_1"), (3.0, 4.0, "emojibots_2")]
    R.A.sel = ("montage", 3, 3)
    R.delete_selected()                                   # Delete: nothing there
    assert R.A.runs[1][2] == "none" and R._pins() == {1: "limonada_1", 2: "limonada_1", 3: "none"}
    R._set_pins(1, 1, None)                               # bar 1 back to the montage's choice
    assert R._pins() == {2: "limonada_1", 3: "none"}
    R.btn_undo()
    assert R._pins() == {1: "limonada_1", 2: "limonada_1", 3: "none"}
    R.btn_redo()
    R.context_menu(_x(R, 5), y)                           # its menu builds, on the clip there
    assert R.A.sel == ("montage", 5, 5)
    make = dict(R._montage_actions(7, 7))["Make emojibots_2 a BG block (move it, resize it, FX)"]
    make()                                                # a BG block, to edit like any
    R.A.content = ("clip", "D:/other/x.mp4", 0.0)         # not in the montage's folder: in the BG
    R._put_in_montage(9, 10)
    bg = R.A.members["BG"]
    assert [(m.label, [(b.first, b.last) for b in m.blocks]) for m in bg] == [("emojibots_2", [(7, 7)]),
                                                                              ("x", [(9, 10)])]
    R.btn_save()
    layers = _layers(path)
    assert layers[0] == {"source": "clips", "pins": {2: "limonada_1", 3: "none"}}
    assert "pins: {2: limonada_1, 3: none}" in path.read_text(encoding="utf-8")


def test_fx_are_blocks_on_their_tracks(window, monkeypatch):
    R, host, path = window
    assert [(m.label, [(b.first, b.last) for b in m.blocks]) for m in R.A.members["BG fx"]] == [("feedback", [(1, 2)])]
    y = _y(R, "BG fx")
    R.press(_x(R, 1), y)                                  # the feedback moved to bars 3-4
    R.release(_x(R, 3))
    monkeypatch.setattr(R, "_shift", lambda: True)
    R.press(R._x_of(5), y)                                # Shift on its end: to bar 6
    R.release(_x(R, 6))
    monkeypatch.setattr(R, "_shift", lambda: False)
    assert R.A.members["BG fx"][0].blocks[0].first == 3 and R.A.members["BG fx"][0].blocks[0].last == 6
    R.A.entry = "pixel_sort"                              # picked in FX: dragged in
    R.press(_x(R, 8), y)
    R.release(_x(R, 9))
    assert [m.label for m in R.A.members["BG fx"]] == ["feedback", "pixel_sort"]
    assert R.A.sel == ("block", "BG fx", 1, 0)
    R.delete_selected()                                   # Delete
    assert [m.label for m in R.A.members["BG fx"]] == ["feedback"]
    R.press(_x(R, 5), _y(R, "master"))                    # the grain (all the song) chosen by a click
    R.release(_x(R, 5))
    assert R.A.sel == ("block", "master", 0, None)
    R.context_menu(_x(R, 5), _y(R, "master"))           # its menu builds
    R.delete_selected()
    assert R._master_items() == []
    R._put_fx("FG fx", {"source": "echoes"})
    R._put_fx_at("FG fx", 2, "rgb_split")
    R._reorder("FG fx", 1, -1)                            # rgb_split first in the chain
    assert R.A.sel == ("block", "FG fx", 0, 0)
    R._draw_inspector()
    R.btn_save()
    layers = _layers(path)
    assert layers[1]["fx"] == [{"source": "feedback", "bars": [[3, 6]]}]
    assert layers[2]["fx"] == [{"source": "rgb_split", "bars": [[2, 5]]}, {"source": "echoes"}]
    assert len(layers) == 3                               # no master left


def test_bars_chosen_on_the_ruler_or_a_lane_are_previewed(window):
    R, host, path = window
    asked, progress = [], [(True, "Rendering preview… 40%", ""), (False, "Preview crashed — see log.", "OSError: no GPU")]
    host.preview = lambda at, bars, layers: asked.append((at, bars, len(layers))) or True
    host.preview_texture = lambda: ("arr_tex", 192, 192, (96, 170))   # a portrait frame in it
    host.preview_progress = lambda: progress[0]
    host.stop_preview = lambda: None
    R.press(_x(R, 5), 10)                                 # across the ruler: bars 5-7
    R.release(_x(R, 7))
    assert R.A.span == (5, 7) and R.A.sel is None
    assert dpg.get_item_label("arr_prev_btn") == "Preview bars 5-7 (Space)"
    R.A.content = None                                    # nothing picked: a drag on a lane chooses bars
    R.press(_x(R, 9), _y(R, "BG", -1))
    R.release(_x(R, 10))
    assert R.A.span == (9, 10) and R.A.members["BG"] == []
    R.preview_span()                                      # Space, or the button
    assert asked == [(16.0, 2, 4)]                        # bar 9 starts at 16 s; two bars; the layers as they are
    R.tick()
    assert dpg.get_value("arr_prev_text") == "Rendering preview... 40%"
    progress.pop(0)
    R.tick()                                              # over: why it failed, in the window
    assert dpg.get_value("arr_prev_text") == "the preview failed: OSError: no GPU" and R.A.previewing is None
    R.press(_x(R, 3), _y(R, "FG"))                        # choosing a block chooses its bars
    R.release(_x(R, 3))
    assert R.A.span == (3, 4)


def test_preview_never_does_nothing(window):
    """Not loaded: the project loads and the bars play once it is; busy: they come next;
    no bars chosen: the first bars in view. The window says which."""
    R, host, path = window
    asked, state, loads = [], ["unloaded"], []
    host.preview = lambda at, bars, layers: asked.append((at, bars)) or True
    host.preview_texture = lambda: ("arr_tex", 192, 192, (96, 170))
    host.preview_progress = lambda: (False, "", "")
    host.preview_state = lambda: state[0]
    host.load = lambda: loads.append(1) or ""
    R.A.span = None
    R.preview_span()                                      # nothing chosen: the bars in view
    assert R.A.span == (1, 8) and loads == [1] and asked == []
    assert dpg.get_value("arr_prev_text").startswith("loading the project")
    state[0] = "loading"
    R.tick()
    assert asked == []
    state[0] = "ready"
    R.tick()                                              # loaded: it plays
    assert asked == [(0.0, 8)] and R.A.queued is None
    state[0] = "busy"                                     # another preview: let go, this one next
    cancelled = []
    host.cancel_preview = lambda: cancelled.append(1) or True
    R.preview_bars(9, 10)
    assert dpg.get_value("arr_prev_text").endswith("starts right after...") and len(asked) == 1
    assert cancelled == [1]
    state[0] = "ready"
    R.tick()
    assert asked[-1] == (16.0, 2)
    state[0] = "busy"                                     # the video's render: not let go, waited for
    host.cancel_preview = lambda: False
    R.preview_bars(9, 10)
    assert dpg.get_value("arr_prev_text").endswith("comes when it is done...") and R.A.queued is not None
    state[0] = "ready"
    R.tick()
    host.load = lambda: "the project has no song (audio file): open the project first"
    state[0] = "unloaded"
    R.preview_bars(3, 4)
    assert dpg.get_value("arr_prev_text").startswith("the preview can't start: the project has no song")
    assert R.A.queued is None


def test_stop_lets_the_preview_being_made_go(monkeypatch):
    """Stop (or another preview) makes the preview being rendered give up at its next frame;
    a full render is never stopped so."""
    import threading
    import time
    import arc_studio as S
    frames, gave_up = [], threading.Event()

    def slow(*args):                                      # a preview's frames, one at a time
        try:
            for fi in range(1000):
                S._check_cancel()
                frames.append(fi)
                time.sleep(0.01)
        except S._PreviewCancelled:
            gave_up.set()
        finally:
            S.S.rendering = S.S.preview_making = S.S.preview_cancel = False
    monkeypatch.setattr(S, "_do_preview", slow)
    monkeypatch.setattr(S, "_set_status", lambda msg: None)
    S._start_preview(0.0, 2, [])
    assert S.S.rendering and S.S.preview_making
    time.sleep(0.05)
    assert S.cancel_preview() is True
    assert gave_up.wait(2) and len(frames) < 1000 and not S.S.rendering
    S.S.rendering = True                                  # the video's render
    try:
        assert S.cancel_preview() is False and not S.S.preview_cancel
    finally:
        S.S.rendering = False


def _click(item):
    """A click as DearPyGui makes it: the callback gets as many of (sender, app_data,
    user_data) as it takes positionally; a third positional (a default capturing a value)
    gets the item's user_data, None."""
    cb = dpg.get_item_callback(item)
    n = cb.__code__.co_argcount - (1 if hasattr(cb, "__self__") else 0)
    cb(*(item, None, dpg.get_item_user_data(item))[:n])


def _buttons(parent):
    out = {}
    for i in dpg.get_item_children(parent, 1):
        if dpg.get_item_type(i) == "mvAppItemType::mvButton":
            out[dpg.get_item_label(i)] = i
        elif dpg.get_item_type(i) == "mvAppItemType::mvGroup":
            out.update(_buttons(i))
    return out


def test_the_montage_buttons_work_when_clicked(window):
    """Its Preview, Nothing here and Back buttons, clicked the way DearPyGui clicks."""
    R, host, path = window
    _with_a_montage(host)
    asked = []
    host.preview = lambda at, bars, layers: asked.append((at, bars)) or True
    host.preview_texture = lambda: ("arr_tex", 192, 192, (96, 170))
    host.preview_progress = lambda: (True, "", "")
    R._draw()
    y = _montage_y(R)
    R.press(_x(R, 3), y)
    R.release(_x(R, 4))
    assert R.A.sel == ("montage", 3, 4)
    _click(_buttons("arr_inspector")["Preview"])
    assert asked == [(4.0, 2)]
    _click(_buttons("arr_inspector")["Nothing here"])
    assert R._pins() == {3: "none", 4: "none"}
    _click(_buttons("arr_inspector")["Back to the montage"])
    assert R._pins() == {}


def test_no_callback_takes_a_captured_value_as_its_third_positional():
    """DearPyGui gives a callback's third positional the item's user_data (None), so
    ``lambda s, a, fn=fn: fn()`` calls None: capture after ``*`` instead."""
    import re
    from pathlib import Path
    bad = re.compile(r"(lambda +\w+, *\w+, *\w+ *=|def \w+\(\s*\w+, *\w+, *\w+ *=)")
    root = Path(__file__).resolve().parents[1]
    hits = [f"{p.name}:{i}" for p in root.glob("arc_*.py")
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if bad.search(line)]
    assert hits == []


def test_a_catalog_entry_plays_its_demo_or_offers_to_make_it(window, tmp_path, monkeypatch):
    import json
    from core import catalog, demos
    R, host, path = window
    monkeypatch.setattr(demos, "DEMOS", tmp_path / "demos")
    made, popen = [], R.subprocess.Popen

    def fake(cmd, **kw):                                  # the demo's run; another thread's ffmpeg is its own
        if "core.demos" not in cmd:
            return popen(cmd, **kw)
        made.append(cmd)
        return type("P", (), {"poll": lambda self: None})()
    monkeypatch.setattr(R.subprocess, "Popen", fake)
    e = catalog.by_key("grain")
    R._show_entry(e)                                      # none yet: a button makes it
    tag = "arr_cat_info_fx"
    texts = [dpg.get_value(i) for i in dpg.get_item_children(tag, 1) if dpg.get_item_type(i).endswith("Text")]
    assert "no demo yet" in texts and R.A.demo_key is None
    _click(_buttons(tag)["Make demo"])
    assert made and made[0][-4:] == ["render", "--only", "grain", "--force"]
    texts = [dpg.get_value(i) for i in dpg.get_item_children(tag, 1) if dpg.get_item_type(i).endswith("Text")]
    assert "making its demo..." in texts
    R.A.demo_making.clear()
    (tmp_path / "demos").mkdir()                          # made: it plays
    demos.video_path("grain").write_bytes(b"")
    demos.info_path("grain").write_text(json.dumps({"state": "ready", "hash": demos.digest(e)}))
    R._show_entry(e)
    images = [i for i in dpg.get_item_children(tag, 1) if dpg.get_item_type(i).endswith("Image")]
    assert images and dpg.get_item_configuration(images[0])["texture_tag"] in ("arr_demo_tex",
                                                                               dpg.get_alias_id("arr_demo_tex"))
    assert R.A.demo_key == "grain"
    R._play_demo(None, None)


def _clip_with(tmp_path, name, sound):
    """A 3 s clip (a moving square), with a tone or silent, by ffmpeg."""
    import subprocess
    out = tmp_path / "clips" / name
    out.parent.mkdir(exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=3"]
    if sound:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest"]
    subprocess.run(cmd + ["-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True)
    return out


def test_a_clip_plays_muted_its_sound_turns_on_and_it_drags_onto_a_lane(window, tmp_path, monkeypatch):
    import time
    R, host, path = window
    loud, quiet = _clip_with(tmp_path, "loud.mp4", True), _clip_with(tmp_path, "quiet.mp4", False)
    played = []

    class Mixer:                                          # no sound device in the tests
        def __init__(self, sr): self.sr = sr
        def remove_track(self, name): pass
        def add_track(self, name, audio): played.append(len(audio))
        def play(self, start_t=None): played.append(("from", round(start_t, 1)))
        def pause(self): played.append("pause")
    import core.audio_player
    monkeypatch.setattr(core.audio_player, "AudioPlayer", Mixer)
    R.dpg.set_value("arr_folder", str(tmp_path / "clips"))
    R.btn_list()
    rows = {Path(p).name: row for row, p in R.A.list_items.items()}
    assert set(rows) == {"loud.mp4", "quiet.mp4"}
    R.on_list_press(None, [0, rows["loud.mp4"]])          # pressed: chosen, it plays, muted
    assert R.A.clip == str(loud) and R.A.clip_play == (str(loud), 0.0) and not R.A.clip_sound
    assert R.dpg.get_value(rows["loud.mp4"]) and not R.dpg.get_value(rows["quiet.mp4"])
    deadline = time.time() + 5
    while R.A.clip_frame is None and time.time() < deadline:
        time.sleep(0.05)
    assert R.A.clip_frame is not None and R.A.clip_frame.shape == (R.THUMB * R.THUMB * 4,)
    assert played == []                                   # muted: nothing to hear
    R.set_clip_sound(True)
    deadline = time.time() + 5
    while not played and time.time() < deadline:
        time.sleep(0.05)
    assert played and abs(played[0] - 3 * R.CLIP_SR) < R.CLIP_SR // 10   # its 3 s of sound
    assert R.dpg.get_item_label("arr_clip_sound") == "sound: on"
    assert R._clip_audio(str(quiet)) is None              # a silent clip has none to play
    R.set_clip_sound(False)
    assert played[-1] == "pause"

    # dragged onto the BG at bar 5: a block from there, as long as it plays (3 s: 1 bar at 2 s)
    monkeypatch.setattr(R, "_over_lanes", lambda: True)
    monkeypatch.setattr(R, "_mouse_in_canvas", lambda: (_x(R, 5), _y(R, "BG")))
    R.on_mouse_drag(None, None)
    assert R.dpg.get_item_configuration("arr_ghost")["show"]
    R.on_mouse_release(None, None)
    assert R.A.clip_drag is None
    bg = [m for m in R.A.members["BG"] if m.spec.get("file") == str(loud)]
    assert bg and bg[0].spec["role"] == "background"
    assert [(b.first, b.last) for b in bg[0].blocks] == [(5, 6)]
    R.on_list_press(None, [0, rows["quiet.mp4"]])         # dropped off the lanes: nothing
    monkeypatch.setattr(R, "_over_lanes", lambda: False)
    n = len(_layers_now(R))
    R.on_mouse_release(None, None)
    assert len(_layers_now(R)) == n and R.A.clip == str(quiet)
    R._play_clip(None)


def test_the_list_shows_each_clips_picture_and_length_and_keeps_its_folder(window, tmp_path):
    import time
    R, host, path = window
    _clip_with(tmp_path, "a_rather_long_clip_name_from_comfyui_00042_.mp4", False)
    R.dpg.set_value("arr_folder", str(tmp_path / "clips"))
    R.btn_list()
    (clip,) = [str(p) for p in R.A.clips]
    pics, names = R.A.list_pics[clip], [n for n, p in R.A.list_names.items() if p == clip]
    assert len(pics) == 1 and len(names) == 1
    assert dpg_config(R, pics[0], "texture_tag") == "arr_cover_blank"
    label = R.dpg.get_item_label(names[0])
    assert all(len(line) <= R.NAME_CHARS for line in label.split("\n")) and label.split("\n")[0].endswith("_")
    deadline = time.time() + 10                           # made in the background
    while not R.A.covers_ready and time.time() < deadline:
        time.sleep(0.05)
    R._show_covers()
    assert clip in R.A.covers and dpg_config(R, pics[0], "texture_tag") == R.A.covers[clip]
    assert R.dpg.get_item_label(names[0]).endswith("3.0 s")
    R.on_list_press(None, [0, pics[0]])                   # the picture picks it too
    assert R.A.clip == clip and R.A.clip_drag is not None
    R.A.clip_drag = None
    R._play_clip(None)
    # opened again with nothing typed: the folder listed last
    R.dpg.set_value("arr_folder", "")
    R.open_window()
    assert Path(R.dpg.get_value("arr_folder")) == (tmp_path / "clips").resolve()


def test_clips_dropped_from_the_explorer_go_on_the_lane_one_after_another(window, tmp_path, monkeypatch):
    R, host, path = window
    a, b = _clip_with(tmp_path, "a.mp4", False), _clip_with(tmp_path, "b.mp4", False)
    (tmp_path / "notes.txt").write_text("x")
    monkeypatch.setattr(R, "_over_lanes", lambda pos=None: True)
    monkeypatch.setattr(R, "_canvas_of", lambda pos: pos)
    R.drop_files([str(a), str(tmp_path / "notes.txt"), str(b)], (_x(R, 5) + 3, _y(R, "BG")))
    got = {m.spec.get("file"): [(k.first, k.last) for k in m.blocks] for m in R.A.members["BG"]}
    assert got[str(a)] == [(5, 6)] and got[str(b)] == [(7, 8)]      # 3 s: 2 bars at 2 s
    n = len(_layers_now(R))
    R.drop_files([str(tmp_path / "notes.txt")], (_x(R, 5), _y(R, "BG")))
    assert len(_layers_now(R)) == n
    # dropped on the browser: a folder is listed, a clip is chosen
    monkeypatch.setattr(R, "_over_lanes", lambda pos=None: False)
    R.drop_files([str(tmp_path / "clips")], (10, 10))
    assert R.dpg.get_value("arr_folder") == str(tmp_path / "clips") and len(R.A.clips) == 2
    R.drop_files([str(b)], (10, 10))
    assert R.A.clip == str(b)
    R._play_clip(None)


def test_a_clip_longer_than_30_s_asks_how_many_bars_it_takes(window, tmp_path, monkeypatch):
    """A long clip dropped in: a window asks its bars (or the whole video); the clips after
    it wait for the answer and follow it; Cancel puts none of them."""
    R, host, path = window
    a, b = _clip_with(tmp_path, "long.mp4", False), _clip_with(tmp_path, "short.mp4", False)
    seconds = {str(a): 75.0, str(b): 3.0}
    monkeypatch.setattr(R.arr, "duration", lambda p: seconds.get(str(p), 0.0))
    monkeypatch.setattr(R, "_over_lanes", lambda pos=None: True)
    monkeypatch.setattr(R, "_canvas_of", lambda pos: pos)

    def blocks():
        return {m.spec.get("file"): [(k.first, k.last) for k in m.blocks] for m in R.A.members["BG"]}

    R.drop_files([str(a), str(b)], (_x(R, 3) + 3, _y(R, "BG")))
    assert dpg_config(R, "arr_ask_win", "show") and blocks() == {}            # asked, nothing put yet
    assert "75 s: 38 bars" in R.dpg.get_value("arr_ask_text") and "from bar 3" in R.dpg.get_value("arr_ask_text")
    assert R.dpg.get_value("arr_ask_n") == 8
    on_n = dpg.get_item_callback("arr_ask_n")
    R.dpg.set_value("arr_ask_n", 7)
    on_n("arr_ask_n", 7)                                  # its "-": ImGui calls back, no Enter: only the number
    assert dpg_config(R, "arr_ask_win", "show") and blocks() == {} and R.A.asking is not None
    down = R._down
    monkeypatch.setattr(R, "_down", lambda *keys: "mvKey_Return" in keys)
    on_n("arr_ask_n", 6)                                  # Enter: 6 bars; the short one after
    monkeypatch.setattr(R, "_down", down)
    assert not dpg_config(R, "arr_ask_win", "show")
    assert blocks() == {str(a): [(3, 8)], str(b): [(9, 10)]}
    R.drop_clip(_x(R, 12) + 3, _y(R, "BG"), (str(a), 10.0, 65.0))            # from the list: asked too
    assert "from 10 s" in R.dpg.get_value("arr_ask_text")
    assert R.dpg.get_item_label("arr_ask_whole") == "The whole video (5 bars, to the song's end)"
    R.answer_bars("whole")                                                    # (the test song: 16 bars)
    got = [(k.first, k.last) for m in R.A.members["BG"] if m.spec.get("file") == str(a) for k in m.blocks]
    assert got == [(3, 8), (12, 16)]
    n = len(_layers_now(R))
    R.drop_files([str(a), str(b)], (_x(R, 1) + 3, _y(R, "BG")))
    R.answer_bars(None)                                                       # Cancel: neither
    assert len(_layers_now(R)) == n and "not put (nor the 1 after it)" in R.dpg.get_value("arr_status")
    R.drop_clip(_x(R, 1) + 3, _y(R, "BG"), (str(b), 0.0, 3.0))               # a short one: never asked
    assert not dpg_config(R, "arr_ask_win", "show")


@pytest.mark.skipif(sys.platform != "win32", reason="the Explorer's drop is Windows'")
def test_an_explorer_drop_is_read_where_it_let_go():
    import ctypes
    import arc_window
    k32 = ctypes.WinDLL("kernel32")
    k32.GlobalAlloc.restype, k32.GlobalLock.restype = ctypes.c_void_p, ctypes.c_void_p
    k32.GlobalLock.argtypes = k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    paths = ["C:\\clips\\a.mp4", "D:\\çãé\\b.mov"]
    body = ("\0".join(paths) + "\0\0").encode("utf-16-le")
    head = (ctypes.c_uint32 * 5)(20, 321, 123, 0, 1)      # DROPFILES: where, wide names
    h = k32.GlobalAlloc(0x0042, 20 + len(body))
    p = k32.GlobalLock(h)
    ctypes.memmove(p, head, 20)
    ctypes.memmove(p + 20, body, len(body))
    k32.GlobalUnlock(h)
    assert arc_window._read(arc_window._shell32(), h) == (paths, (321.0, 123.0))
    assert arc_window.install("no window has this title") is False


def _layers_now(R):
    """Every layer of the arrangement as it is now (saved or not), the lanes' members flat."""
    from core.scene_doc import plain
    out = []
    for item in R.A.items:
        spec = plain(item[2])
        out += spec.get("layers") or [spec]
    return out


def test_the_sam2_cutout_is_an_fx_whose_seeds_open_in_a_window(window):
    """Cut it out: the FX on the character, its window open with SAM2's own way (as always);
    seeds of one's own: a box, points (+ / -), more objects, the frame they are on; Cancel
    takes a new one away; its button opens it again."""
    R, host, path = window
    lion = R.A.members["FG"][0]
    R.A.clip = "lion.mp4"
    R.btn_segment()                                       # the Clips tab's "Cut it out"
    assert dpg_config(R, "arr_sam_win", "show") and R.A.sam["new"]
    assert R.dpg.get_value("arr_sam_mode") == R.SAM_MODES[0] and R.dpg.get_value("arr_sam_max") == 4
    assert R.dpg.get_value("arr_sam_keep") == R.SAM_KEEPS[0]
    R.sam_close(False)                                    # Cancel: not added
    assert "fx" not in R.A.members["FG"][0].spec or not R.A.members["FG"][0].spec["fx"]
    R.btn_segment()
    R.sam_close(True)                                     # OK: SAM2's own way
    fx = R.A.members["FG"][0].spec["fx"]
    assert fx == [{"source": "sam_cutout", "seeds": "auto", "max_objects": 4, "granularity": "whole",
                   "reseed": 2.0, "keep": "figures", "feather": 2}]
    assert not dpg_config(R, "arr_sam_win", "show")
    R.btn_segment()                                       # there already: its window, not a second one
    assert not R.A.sam["new"] and len(R.A.members["FG"][0].spec["fx"]) == 1
    R.dpg.set_value("arr_sam_mode", R.SAM_MODES[1])
    R.on_sam_mode()
    assert R.dpg.get_value("arr_sam_keep") == R.SAM_KEEPS[1]                # seeded: every object kept
    R.sam_close(True)                                     # seeding, nothing drawn: not yet
    assert dpg_config(R, "arr_sam_win", "show") and "draw a box" in R.dpg.get_value("arr_status")
    R.sam_press(0.2, 0.3)                                 # a box, dragged
    R.sam_drag(0.4, 0.5)
    R.sam_drag(0.6, 0.9, done=True)
    R.dpg.set_value("arr_sam_tool", R.SAM_TOOLS[1])
    R.sam_press(0.4, 0.6)                                 # + a point on it
    R.sam_new_object()
    R.dpg.set_value("arr_sam_tool", R.SAM_TOOLS[2])
    R.sam_press(0.8, 0.2)                                 # object 2: a - point, then another taken away
    R.sam_press(0.1, 0.1)
    R.sam_press(0.11, 0.1, button=1)
    R.dpg.set_value("arr_sam_at", 1.5)
    R.dpg.set_value("arr_sam_keep", R.SAM_KEEPS[1])
    R.sam_close(True)
    fx = R.A.members["FG"][0].spec["fx"][0]
    assert fx == {"source": "sam_cutout", "seeds": {"at": 1.5, "objects": [
        {"box": [0.2, 0.3, 0.6, 0.9], "points": [[0.4, 0.6, 1]]}, {"points": [[0.8, 0.2, 0]]}]},
        "keep": "all", "feather": 2}
    R.A.sel = ("block", "FG", lion.at, None)
    R._draw_inspector()
    shown = " ".join(str(dpg.get_value(t)) for t in _all_children("arr_inspector") if dpg.get_item_type(t).endswith("Text"))
    assert "cut out by SAM2 (2 object(s) seeded): made at the next preview or render, once" in shown
    R.open_plugin(*R._where("FG", lion.at)[:1], R._where("FG", lion.at)[1] + ("fx", 0))   # its "plugin": the seeds
    assert R.A.sam is not None and len(R.A.sam["objects"]) == 2 and R.dpg.get_value("arr_sam_at") == 1.5
    R.sam_clear(True)
    R.sam_close(False)
    assert R.A.members["FG"][0].spec["fx"][0]["seeds"]["at"] == 1.5         # Cancel: as it was


def _all_children(tag):
    out = []
    for c in dpg.get_item_children(tag, 1) or []:
        out.append(c)
        out += _all_children(c)
    return out


def test_a_block_dragged_onto_another_row_goes_there(window):
    """Onto the BG: the lion goes there (the whole frame, behind); one Ctrl+Z puts it back.
    One block of several: that block alone, a member of its own on the row it was let go on.
    An FX to the other lane's FX track; a clip onto an FX track: it only moves along."""
    R, host, path = window

    def lanes():
        return {k: [(m.label, [(b.first, b.last) for b in m.blocks], m.row) for m in R.A.members[k]]
                for k in ("FG", "BG", "FG fx", "BG fx")}
    lion = R.A.members["FG"][0]
    name = lion.label
    R.press(_x(R, 3), _y(R, "FG"))
    R.release(_x(R, 5), _y(R, "BG", -1))                  # down onto the BG, two bars on
    got = lanes()
    assert got["FG"] == [] and [(l, b) for l, b, _ in got["BG"]] == [(name, [(5, 6)])]
    bg = R.A.members["BG"][0].spec
    assert bg["role"] == "background" and bg["fit"] == "cover" and bg["file"] == "lion.mp4"
    assert "x" not in bg and R.A.sel == ("block", "BG", 0, 0)
    R.btn_undo()                                          # one step: all of it
    assert [(l, b) for l, b, _ in lanes()["FG"]] == [(name, [(3, 4)])] and lanes()["BG"] == []
    R._set_blocks("FG", lion.at, [R.arr.Block(3, 4), R.arr.Block(8, 9)])
    R.press(_x(R, 8), _y(R, "FG"))
    R.release(_x(R, 8), _y(R, "FG", -1))                  # its second block, onto the row under it
    fg = lanes()["FG"]
    assert sorted((l, b) for l, b, _ in fg) == [(name, [(3, 4)]), (name, [(8, 9)])]
    split = next(m for m in R.A.members["FG"] if m.blocks[0].first == 8)
    assert split.spec["lane_row"] == 1 and split.row == 1 and split.spec["loop"] is True
    R.press(_x(R, 8), _y(R, "FG", 1))
    R.release(_x(R, 8), _y(R, "FG", 0))                  # back up: the row it is on changes, only
    assert next(m for m in R.A.members["FG"] if m.blocks[0].first == 8).spec["lane_row"] == 0
    R.press(_x(R, 1), _y(R, "BG fx"))                     # the BG's feedback onto the FG's FX track
    R.release(_x(R, 1), _y(R, "FG fx"))
    assert lanes()["BG fx"] == [] and [l for l, _, _ in lanes()["FG fx"]] == ["feedback"]
    assert _layers_now(R) and R._lane_item("FG")[2]["fx"][0]["source"] == "feedback"
    R.press(_x(R, 3), _y(R, "FG"))                        # a clip onto an FX track: it moves along only
    R.release(_x(R, 4), _y(R, "FG fx"))
    assert any(b == [(4, 5)] for l, b, _ in lanes()["FG"])


def test_the_preview_window_says_what_the_studio_says_now(window):
    """Not a line from before: once the preview is over, the window follows the studio's
    status (it plays, it loops); a failure stays said until the next preview."""
    R, host, path = window
    progress = [(True, "SAM2: cutting x.mp4 out: 10/714 frames", "")]
    host.preview = lambda at, bars, layers: True
    host.preview_texture = lambda: ("arr_tex", 192, 192, (96, 170))
    host.preview_progress = lambda: progress[0]
    host.preview_state = lambda: "ready"
    R.preview_bars(1, 2)
    R.tick()
    assert R.dpg.get_value("arr_prev_text") == "SAM2: cutting x.mp4 out: 10/714 frames"
    progress[0] = (False, "Preview ready - 120 frames. Playing...", "")
    R.tick()
    progress[0] = (False, "Preview loop - 120 frames. Stop to pause.", "")
    R.tick()                                              # over, and still followed
    assert R.A.previewing is None and R.dpg.get_value("arr_prev_text") == "Preview loop - 120 frames. Stop to pause."
    R.preview_bars(1, 2)
    progress[0] = (False, "Preview crashed - see log.", "RuntimeError: SAM2 could not cut x.mp4 out")
    R.A.previewing = (0.0, True)
    R.tick()
    progress[0] = (False, "Ready.", "")
    R.tick()                                              # the failure stays said
    assert R.dpg.get_value("arr_prev_text") == "the preview failed: RuntimeError: SAM2 could not cut x.mp4 out"


def test_callbacks_run_on_the_main_thread_and_one_that_fails_is_only_told(monkeypatch, capsys):
    """The studio runs dearpygui's callbacks itself, between frames: each gets as many of
    (sender, app_data, user_data) as it takes positionally (a keyword-only one keeps its
    default, as dearpygui does); one that raises is told once, the rest still run."""
    import functools
    import arc_studio as S
    got = []

    class Owner:
        def method(self, sender, app_data):
            got.append(("method", sender, app_data))

    def boom(sender):
        raise KeyError("arr_edge")
    jobs = [(lambda s, a, *, here=(1, 2): got.append(("kw", s, a, here)), "btn", 5, "ud"),
            (boom, "x", None, None),
            (Owner().method, "m", 7, "ud"),
            (functools.partial(lambda tag, s: got.append(("partial", tag, s)), "p"), "q", 1, 2),
            (lambda: got.append(("none",)), "z", 0, 0),
            (None, "n", 0, 0)]
    monkeypatch.setattr(S.dpg, "get_callback_queue", lambda: jobs)
    S.run_callbacks()
    S.run_callbacks()                                     # the same failure: told once
    assert got.count(("kw", "btn", 5, (1, 2))) == 2 and ("method", "m", 7) in got
    assert ("partial", "p", "q") in got and ("none",) in got
    assert capsys.readouterr().out.count("[boom] went wrong, the studio goes on") == 1


def test_a_characters_place_linked_to_the_music_is_said_edited_and_unlinked(window):
    """y following the snare: said in words, its Link opens the knob editor on it, "fixed"
    puts it back to a number (where the music held it); a number gets its Link too."""
    R, host, path = window
    lion = R.A.members["FG"][0]
    R._set_member("FG", lion.at, {**lion.spec, "y": {"hits": {"track": "snare"}, "envelope": 0.25,
                                                       "gain": -0.04, "bias": 0.7}})
    assert R._knob_text(R.A.members["FG"][0].spec["y"]) == \
        "a jump on each snare hit, fading in 0.25 s; around 0.7, by -0.04"
    assert R._knob_text({"lfo": "sine", "period": "4/1", "gain": 0.1, "bias": 0.25}) == \
        "a sine wave, one every 4/1; around 0.25, by 0.1"
    R.A.sel = ("block", "FG", lion.at, 0)
    R._draw_inspector()
    shown = " ".join(str(dpg.get_value(t)) for t in _all_children("arr_inspector") if dpg.get_item_type(t).endswith("Text"))
    assert "follows the music: a jump on each snare hit" in shown
    opened = []
    host.edit_knob = lambda target, p, default=0.0, mode="knob": opened.append((target.uid, tuple(p), default, mode))
    links = [b for b in _all_children("arr_inspector") if dpg.get_item_type(b).endswith("Button")
             and dpg.get_item_label(b) == "Link..."]
    assert len(links) == 3                                # x, y, scale
    _click(links[1])
    uid, here = R._where("FG", lion.at)
    assert opened == [(uid, here + ("y",), 0.5, "knob")]
    fixed = next(b for b in _all_children("arr_inspector") if dpg.get_item_type(b).endswith("Button")
                 and dpg.get_item_label(b) == "fixed")
    _click(fixed)
    assert R.A.members["FG"][0].spec["y"] == 0.7 and "y fixed at 0.7" in R.dpg.get_value("arr_status")


def test_a_blocks_music_links_and_fx_are_copied_onto_another(window):
    """Right-click: copy a block's links (what the music moves) or its FX; right-click another:
    paste them (the links its kind takes; the FX after its own, over all its bars; seeds of
    another clip's SAM2 cutout become SAM2's own). One undo each."""
    R, host, path = window
    lion = R.A.members["FG"][0]
    snare = {"hits": {"track": "snare"}, "envelope": 0.25, "gain": -0.04, "bias": 0.7}
    R._set_member("FG", lion.at, {**lion.spec, "y": snare, "fx": [
        {"source": "rgb_split", "bars": [[3, 3]], "trigger": {"track": "snare", "envelope": 0.15}},
        {"source": "sam_cutout", "seeds": {"at": 0, "objects": [{"box": [0, 0, 1, 1]}]}}]})
    R._insert_member("FG", {"source": "footage", "file": "two.mp4", "blocks": [{"bars": [9, 10]}]}, None, 9, 10)
    R._insert_member("BG", {"source": "shadertoy", "file": "shaders/shadertoy/nebulosa_arc.glsl",
                            "bars": [[1, 2]]}, None, 1, 2)

    def menu(key, bar, sub=0):
        R.context_menu(_x(R, bar), _y(R, key, sub))
        return {dpg.get_item_label(i): i for i in dpg.get_item_children("arr_menu", 1)
                if dpg.get_item_type(i).endswith("MenuItem")}

    def spec(file):
        return next(m.spec for k in ("FG", "BG") for m in R.A.members[k] if m.spec.get("file") == file)
    lion_row = next(m.row for m in R.A.members["FG"] if m.spec.get("file") == "lion.mp4")
    items = menu("FG", 3, lion_row)
    assert "Copy its music links (y)" in items and "Copy its FX (2): rgb_split, sam_cutout" in items
    _click(items["Copy its music links (y)"])
    _click(menu("FG", 3, lion_row)["Copy its FX (2): rgb_split, sam_cutout"])
    two_row = next(m.row for m in R.A.members["FG"] if m.spec.get("file") == "two.mp4")
    items = menu("FG", 9, two_row)
    n = len(R.A.undo)
    _click(items["Paste music links from the dancer (y)"])
    assert spec("two.mp4")["y"] == snare and len(R.A.undo) == n + 1
    items = menu("FG", 9, two_row)
    _click(next(i for label, i in items.items() if label.startswith("Paste FX from the dancer")))
    fx = spec("two.mp4")["fx"]
    assert fx == [{"source": "rgb_split", "trigger": {"track": "snare", "envelope": 0.15}},
                  {"source": "sam_cutout", "seeds": "auto"}]                 # its bars, the other's seeds: gone
    assert "SAM2 seeds were the other clip's" in R.dpg.get_value("arr_status")
    R.btn_undo()
    assert "fx" not in spec("two.mp4") and spec("two.mp4")["y"] == snare
    items = menu("BG", 1)                                 # a shader: no y of a clip to take
    assert not any(label.startswith("Paste music links") for label in items)
    assert any(label.startswith("Paste FX from the dancer") for label in items)
    items = menu("BG fx", 1)                              # an FX track's block: itself, copied
    _click(items["Copy this FX: feedback"])
    items = menu("FG", 9, two_row)
    _click(items["Paste FX from feedback (feedback)"])
    assert spec("two.mp4")["fx"] == [{"source": "feedback"}]


def test_every_parameter_of_a_plugin_is_edited_in_its_window(window, tmp_path):
    """Not only numbers: a folder (typed or picked), a flag, a colour, a structure in YAML."""
    R, host, path = window
    clips = tmp_path / "clips"
    clips.mkdir()
    dpg.set_value("arr_folder", str(clips))
    R.A.content = ("gen", "cubes")
    spec = R._content_spec("FG", 3, 4)
    assert spec["clips_dir"] == clips.resolve().as_posix()          # the browser's folder to begin with
    fg = R._lane_item("FG")[3]
    R._edit(fg, ("layers",), lambda ls: list(ls) + [{"source": "cubes", "bars": [[3, 4]]}])
    at = ("layers", 1)
    R.open_plugin(fg, at)

    def items(kind):
        out, todo = [], list(dpg.get_item_children("arr_plugin_body", 1) or [])
        while todo:
            i = todo.pop()
            if dpg.get_item_type(i) == kind:
                out.append(i)
            for slot in (dpg.get_item_children(i) or {}).values():
                todo += slot
        return out

    folder = [i for i in items("mvAppItemType::mvInputText") if "folder" in dpg.get_item_configuration(i)["hint"]]
    assert len(folder) == 1 and "required" in dpg.get_item_configuration(folder[0])["hint"]
    assert len(items("mvAppItemType::mvCheckbox")) >= 3                 # on, crop_faces, one_clip_per_face
    assert items("mvAppItemType::mvColorEdit")                          # bg_color
    R._set_text(fg, at + ("clips_dir",), "  D:/my clips ", True)
    R._set_text(fg, at + ("clip_reactivity",), "{clip_per_bar: true, clip_order: shuffle}", False)
    R._set_text(fg, at + ("zoom_pulse",), "{notes: [36", False)          # not YAML: nothing changes
    R.set_param(fg, at + ("crop_faces",), False)
    R.A.param_pick = (fg, at + ("clips_dir",))
    R._picked_param(None, {"file_path_name": str(clips)})
    R.btn_save()
    cubes = _layers(path)[2]["layers"][1]
    assert cubes["clips_dir"] == clips.as_posix() and cubes["crop_faces"] is False
    assert cubes["clip_reactivity"] == {"clip_per_bar": True, "clip_order": "shuffle"}
    assert "zoom_pulse" not in cubes
    fg = R._lane_item("FG")[3]                                          # (saved: a new one)
    R._set_text(fg, at + ("clips_dir",), "", True)                      # empty: taken out
    assert "clips_dir" not in R._item(fg)[2]["layers"][1]


def test_a_folder_is_picked_with_the_systems_own_dialog(window, tmp_path, monkeypatch):
    """Any drive: Windows' dialog (on a thread), its answer handed over on the main loop."""
    import threading
    import tkinter
    from tkinter import filedialog
    import arc_window
    R, host, path = window
    clips = tmp_path / "other drive clips"
    clips.mkdir()

    class _Root:
        def withdraw(self): pass
        def attributes(self, *a): pass
        def destroy(self): pass

    asked = {}
    monkeypatch.setattr(tkinter, "Tk", _Root)
    monkeypatch.setattr(filedialog, "askdirectory", lambda **kw: asked.update(kw) or str(clips))
    R._choose_folder()
    for t in threading.enumerate():
        if t.name == "arc-pick":
            t.join(5)
    assert dpg.get_value("arr_folder") != str(clips)                    # not yet: on the main loop
    R.tick()
    assert dpg.get_value("arr_folder") == str(clips) and asked["mustexist"]


def _widgets(kind, root="arr_plugin_body"):
    out, todo = [], list(dpg.get_item_children(root, 1) or [])
    while todo:
        i = todo.pop(0)
        if dpg.get_item_type(i) == f"mvAppItemType::{kind}":
            out.append(i)
        for slot in (dpg.get_item_children(i) or {}).values():
            todo += slot
    return out


def _fire(item, value=None):
    """A widget used: its callback, as dearpygui calls it."""
    cb = dpg.get_item_callback(item)
    n = cb.__code__.co_argcount
    cb(*(item, value, None)[:n])


def test_a_layers_own_clips_are_set_with_controls_not_yaml(window, monkeypatch):
    """cubes' clip_reactivity: when a clip changes, its order, its triggers (who plays, what
    it does, until, not near, gravity), each a control; zoom_pulse: who plays, its envelope."""
    R, host, path = window
    asked = []
    monkeypatch.setattr(R, "_link", lambda uid, p, d, mode="knob": asked.append((p[-1], mode)))
    fg = R._lane_item("FG")[3]
    R._edit(fg, ("layers",), lambda ls: list(ls) + [{"source": "cubes", "clips_dir": "C:/c", "bars": [[3, 4]],
                                                     "zoom_pulse": {"notes": [36], "envelope": 0.25}}])
    at = ("layers", 1)
    R.open_plugin(fg, at)
    from core.scene_doc import plain
    cubes = lambda: plain(R._item(fg)[2]["layers"][1])                 # noqa: E731
    order = next(c for c in _widgets("mvCombo") if dpg.get_item_configuration(c)["items"] == list(R.CLIP_ORDERS))
    _fire(order, "shuffle")
    assert cubes()["clip_reactivity"] == {"clip_order": "shuffle"}
    plus = next(b for b in _widgets("mvButton") if dpg.get_item_label(b) == "+ trigger")
    _fire(plus)
    _fire(next(b for b in _widgets("mvButton") if dpg.get_item_label(b) == "+ trigger"))
    assert asked == [("kick", "trigger"), ("snare", "trigger")]         # who plays: the studio's editor
    trs = lambda: cubes()["clip_reactivity"]["triggers"]                # noqa: E731
    assert trs() == {"kick": R.NEW_TRIGGER, "snare": R.NEW_TRIGGER}
    reverse = [c for c in _widgets("mvCheckbox") if dpg.get_item_label(c) == "reverse"]
    _fire(reverse[0], True)                                              # the kick reverses too
    assert trs()["kick"]["actions"] == ["next_clip", "reverse"]
    nxt = next(c for c in _widgets("mvCheckbox") if dpg.get_item_label(c) == "next clip")
    _fire(nxt, False)
    assert trs()["kick"]["actions"] == ["reverse"]
    until = lambda: [c for c in _widgets("mvCombo") if dpg.get_item_configuration(c)["items"] == ["-", "snare"]]
    _fire(until()[0], "snare")                                           # the kick's until
    _fire(until()[1], "snare")                                           # and not near the snare
    assert trs()["kick"]["until"] == "snare" and trs()["kick"]["exclude"] == {"trigger": "snare", "window": 0.04}
    names = [t for t in _widgets("mvInputText") if dpg.get_value(t) == "snare"]
    _fire(names[0], "caixa")                                             # renamed: the kick follows
    assert list(trs()) == ["kick", "caixa"] and trs()["kick"]["until"] == "caixa"
    grav = [c for c in _widgets("mvCheckbox") if not dpg.get_item_label(c)]
    _fire(grav[-1], True)                                                # the caixa's gravity
    assert trs()["caixa"]["gravity"]["peak"] == 3.0
    R._drop_trigger(fg, at + ("clip_reactivity", "triggers"), "caixa")   # its x
    assert list(trs()) == ["kick"] and "until" not in trs()["kick"] and "exclude" not in trs()["kick"]
    every = next(c for c in _widgets("mvCheckbox") if dpg.get_item_label(c) == "every")
    _fire(every, False)
    seed = next(t for t in _widgets("mvInputText") if dpg.get_item_configuration(t)["hint"] == "none")
    _fire(seed, "7")
    assert cubes()["clip_reactivity"]["clip_per_bar"] is False and cubes()["clip_reactivity"]["seed"] == 7
    env = next(f for f in _widgets("mvInputFloat") if dpg.get_value(f) == pytest.approx(0.25))
    _fire(env, 0.4)
    assert cubes()["zoom_pulse"] == {"notes": [36], "envelope": 0.4}
    R.btn_save()
    assert _layers(path)[2]["layers"][1]["clip_reactivity"]["triggers"]["kick"]["actions"] == ["reverse"]


def test_eyes_gestures_chords_and_features_are_controls(window, monkeypatch):
    """eyes: its hits as gesture rows (which, who plays, how long), harmony with its own
    editor, pupil_from linked like a knob, iris_hues two numbers, rows fixed (no Link)."""
    R, host, path = window
    asked = []
    monkeypatch.setattr(R, "_link", lambda uid, p, d, mode="knob": asked.append((p[-2:], mode)))
    fg = R._lane_item("FG")[3]
    R._edit(fg, ("layers",), lambda ls: list(ls) + [{"source": "eyes", "bars": [[3, 4]]}])
    at = ("layers", 1)
    R.open_plugin(fg, at)
    from core.scene_doc import plain
    eyes = lambda: plain(R._item(fg)[2]["layers"][1])                  # noqa: E731
    buttons = lambda label: [b for b in _widgets("mvButton") if dpg.get_item_label(b) == label]   # noqa: E731
    _fire(buttons("+ gesture")[0])
    _fire(buttons("+ gesture")[0])
    assert asked == [(("hits", 0), "trigger"), (("hits", 1), "trigger")]
    which = [c for c in _widgets("mvCombo") if dpg.get_item_configuration(c)["items"] == ["saccade", "dilate", "blink"]]
    _fire(which[1], "blink")
    env = [f for f in _widgets("mvInputFloat") if dpg.get_value(f) == pytest.approx(0.2)]
    _fire(env[1], 0.12)
    assert eyes()["hits"] == [{"notes": [36], "gesture": "saccade", "envelope": 0.2},
                              {"notes": [36], "gesture": "blink", "envelope": 0.12}]
    R._edit(fg, at + ("hits",), lambda ls: ls[1:])                       # (the first one's x)
    assert [h["gesture"] for h in eyes()["hits"]] == ["blink"]
    _fire(buttons("Chords from")[0])
    assert asked[-1] == ((1, "harmony"), "score")
    labels = [dpg.get_value(t) for t in _widgets("mvText")]
    assert not any(l.startswith("~ rows") or l.startswith("~ seed") for l in labels)
    assert "~ pupil_from" in labels and "~ wander" in labels
    hues = [f for f in _widgets("mvInputFloat") if dpg.get_value(f) in (pytest.approx(0.08), pytest.approx(0.6))]
    _fire(hues[1], 0.7)
    assert eyes()["iris_hues"] == [0.08, 0.7]
    pupil = next(t for t in _widgets("mvInputText") if dpg.get_item_configuration(t)["hint"] == "texture.loudness")
    row = dpg.get_item_parent(dpg.get_item_parent(pupil))               # its row (the text in a group)
    link = next(b for b in buttons("Link") if dpg.get_item_parent(dpg.get_item_parent(b)) == row)
    _fire(link)
    assert eyes()["pupil_from"] == "texture.loudness" and asked[-1] == ((1, "pupil_from"), "knob")


def test_the_final_video_is_rendered_from_the_arrange(window):
    """Render video...: saved first, the whole song or some bars, its progress, and the video
    to open when it is done."""
    R, host, path = window
    asked, state = [], {}
    host.render_output = lambda first=None, last=None: f"render_output/x{'' if first is None else f'_bars{first}-{last}'}.mp4"
    host.render = lambda first=None, last=None: asked.append((first, last)) or ""
    host.render_state = lambda: dict(state) if state else None
    host.stop_render = lambda: state.update(running=False, result="stopped")
    R._rename("FG", 0, "Leo")                                           # unsaved
    R.open_render()
    assert dpg.get_value("arr_render_out") == "-> render_output/x.mp4"
    R.btn_render()
    assert asked == [(None, None)] and not R.A.dirty and host.saved == 1  # saved before
    state.update(running=True, done=240, total=960, started=__import__("time").time() - 10, out="render_output/x.mp4")
    R.tick()
    assert dpg.get_value("arr_render_bar") == pytest.approx(0.25) and "240/960" in dpg.get_value("arr_render_state")
    assert "left" in dpg.get_value("arr_render_state") and dpg.get_item_label("arr_render_btn") == "Rendering 25%..."
    state.update(running=False, done=960, result="done")
    R.tick()
    assert dpg.is_item_shown("arr_render_done") and dpg.get_value("arr_render_state").startswith("done")
    dpg.set_value("arr_render_first", 5)
    dpg.set_value("arr_render_last", 3)
    R._render_where(bars=True)
    assert dpg.get_value("arr_render_out") == "-> render_output/x_bars3-5.mp4"
    R.btn_render()
    assert asked[-1] == (3, 5)
