"""The Arrange window, driven without a screen: drag to make, move and resize blocks; save."""
import numpy as np
import pytest
import yaml

dpg = pytest.importorskip("dearpygui.dearpygui")

SCENE = """# a scene
video:
  layers:
    - source: clips
    # the dancer
    - source: footage
      file: lion.mp4
      loop: true
      bars: [[3, 4]]
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


def test_drag_makes_moves_and_resizes_blocks(window):
    R, host, path = window
    lion = next(k for k, l in enumerate(R.A.lanes) if l.label == "lion")
    y = R._lane_y(lion) + 10
    assert [(b.first, b.last) for b in R.A.lanes[lion].blocks] == [(3, 4)]     # from bars:
    R.press(_x(R, 8), y)                                  # a new block, 8 to 10
    R.release(_x(R, 10))
    R.press(_x(R, 9), y)                                  # move it two bars on
    R.release(_x(R, 11))
    blocks = R.A.lanes[lion].blocks
    assert [(b.first, b.last) for b in blocks] == [(3, 4), (10, 12)]
    R.press(R._x_of(12 + 1), y)                           # its end: one bar longer
    R.release(_x(R, 13))
    R.press(R._x_of(3), y)                                # the first block's start, a bar later
    R.release(_x(R, 4))
    blocks = R.A.lanes[lion].blocks
    assert [(b.first, b.last, b.start) for b in blocks] == [(4, 4, 2.0), (10, 13, 0.0)]
    R.press(_x(R, 11), y)                                 # a click: it is chosen
    R.release(_x(R, 11))
    assert R.A.sel == (lion, 1)
    R.btn_save()
    out = path.read_text(encoding="utf-8")
    assert "# the dancer" in out and host.saved == 1
    layer = yaml.safe_load(out)["video"]["layers"][1]
    assert "bars" not in layer
    assert layer["blocks"] == [{"bars": [4, 4], "from": 2.0}, {"bars": [10, 13]}]


def test_a_background_lane_takes_the_chosen_clip(window, tmp_path):
    R, host, path = window
    R.A.clip, R.A.start = "C:/v/a/final.mp4", 3.5          # as if picked in the browser
    R.btn_add_background()
    bg = next(k for k, l in enumerate(R.A.lanes) if l.role == "background")
    R.press(_x(R, 1), R._lane_y(bg) + 10)
    R.release(_x(R, 2))
    R.btn_save()
    layers = yaml.safe_load(path.read_text(encoding="utf-8"))["video"]["layers"]
    assert layers[1]["role"] == "background"               # right above the base clips
    assert layers[1]["blocks"] == [{"bars": [1, 2], "file": "C:/v/a/final.mp4", "from": 3.5}]


def test_undo_and_redo(window):
    R, host, path = window
    lion = next(k for k, l in enumerate(R.A.lanes) if l.label == "lion")
    y = R._lane_y(lion) + 10
    spans = lambda: [(b.first, b.last) for b in R.A.lanes[lion].blocks]
    R.press(_x(R, 8), y)                                  # a new block
    R.release(_x(R, 10))
    R.press(_x(R, 9), y)                                  # moved
    R.release(_x(R, 11))
    R.press(_x(R, 11), y)                                 # a click changes nothing: no step
    R.release(_x(R, 11))
    assert spans() == [(3, 4), (10, 12)] and len(R.A.undo) == 2
    R.btn_undo()
    assert spans() == [(3, 4), (8, 10)]
    R.btn_undo()
    assert spans() == [(3, 4)]
    R.btn_undo()                                          # nothing more: stays
    assert spans() == [(3, 4)]
    R.btn_redo()
    R.btn_redo()
    assert spans() == [(3, 4), (10, 12)]
    R.btn_undo()
    R.press(_x(R, 14), y)                                 # a new edit forgets what was undone
    R.release(_x(R, 14))
    assert spans() == [(3, 4), (8, 10), (14, 14)] and R.A.redo == []


def test_undo_a_lane_and_across_a_save(window):
    R, host, path = window
    n = len(R.A.lanes)
    R.btn_add_background()
    assert len(R.A.lanes) == n + 1
    R.btn_undo()
    assert len(R.A.lanes) == n
    lion = next(k for k, l in enumerate(R.A.lanes) if l.label == "lion")
    y = R._lane_y(lion) + 10
    R.press(_x(R, 3), y)                                  # move it, save, then undo the move
    R.release(_x(R, 6))
    R.btn_save()
    assert "blocks" in yaml.safe_load(path.read_text(encoding="utf-8"))["video"]["layers"][1]
    R.btn_undo()
    R.btn_save()
    out = path.read_text(encoding="utf-8")
    assert "# the dancer" in out                          # still the same layer, comment and all
    layer = yaml.safe_load(out)["video"]["layers"][1]
    assert layer["bars"] == [[3, 4]] and "blocks" not in layer


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


def test_generators_and_fx_on_the_timeline(window):
    R, host, path = window
    assert [l.kind for l in R.A.lanes] == ["clip", "clips"]              # the top of the picture first
    R.insert_master("grain")
    lion = next(k for k, l in enumerate(R.A.lanes) if l.label == "lion")
    R.insert_above("shadertoy:nebulosa_arc", lion)
    assert [l.label for l in R.A.lanes] == ["grain", "nebulosa_arc", "lion", "clip montage"]
    assert R._is_master(0) and not R._is_master(1)
    gen = 1
    assert R.A.lanes[gen].always                                          # new: the whole song
    R.press(_x(R, 5), R._lane_y(gen) + 10)
    R.release(_x(R, 8))
    assert [(b.first, b.last) for b in R.A.lanes[gen].blocks] == [(5, 8)]
    R._toggle(0)                                                          # grain bypassed
    R.context_menu(_x(R, 3), R._lane_y(gen) + 10)                         # the menu builds
    assert dpg.does_item_exist("arr_menu")
    R.btn_save()
    text = path.read_text(encoding="utf-8")
    assert "bars: [[5, 8]]" in text
    layers = yaml.safe_load(text)["video"]["layers"]
    assert [l["source"] for l in layers] == ["clips", "footage", "shadertoy", "grain"]
    assert layers[2]["file"] == "shaders/shadertoy/nebulosa_arc.glsl" and layers[3]["enabled"] is False
    R.btn_undo()                                                          # the bypass
    R.btn_undo()                                                          # the bars
    R.btn_undo()                                                          # the generator
    assert [l.label for l in R.A.lanes] == ["grain", "lion", "clip montage"]
