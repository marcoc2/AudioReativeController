"""The Arrange window's model: lanes from footage layers, blocks placed by bars, new lanes."""
from pathlib import Path

import numpy as np

from core.arrangement import (Block, block_at, blocks_of, find_mask_track, grouped, list_clips,
                              members_of, new_foreground, pack, place, remove, role_of, with_blocks,
                              with_windows)

LAYERS = [
    {"source": "clips"},
    {"source": "pixel_sort", "amount": 0.5},
    {"source": "footage", "role": "background", "fit": "cover",
     "blocks": [{"bars": [1, 4], "file": "a.mp4", "from": 2.0}]},
    {"source": "footage", "file": "lion.mp4", "cutout": {"track": "x"}, "blocks": [{"bars": [5, 8]}]},
    {"source": "footage", "file": "bottles.mp4", "cues": {"reaper": "song.rpp"}},
    {"source": "rgb_split"},
]


def test_a_new_block_takes_the_bars_it_covers():
    blocks = [Block(1, 8, 10.0, 1.0, "a.mp4")]
    out = place(blocks, Block(3, 4, 0.0, 1.0, "b.mp4"), bar_seconds=2.0)
    assert out == [Block(1, 2, 10.0, 1.0, "a.mp4"), Block(3, 4, 0.0, 1.0, "b.mp4"),
                   Block(5, 8, 18.0, 1.0, "a.mp4")]                 # goes on 4 bars (8 s) later
    assert block_at(out, 4) == 1 and block_at(out, 9) is None
    assert remove(out, 1) == [out[0], out[2]]
    assert place([], Block(6, 2)) == [Block(2, 6)]                 # dragged backwards


def test_blocks_go_back_into_the_layer():
    spec = with_blocks(LAYERS[3], [Block(9, 12, 1.5), Block(5, 6)])
    assert spec["blocks"] == [{"bars": [5, 6]}, {"bars": [9, 12], "from": 1.5}]
    assert blocks_of(spec)[1] == Block(9, 12, 1.5)
    assert "blocks" not in with_blocks(LAYERS[3], [])
    assert LAYERS[3]["blocks"] == [{"bars": [5, 8]}]               # the original is left alone


def test_new_lanes_and_where_they_go(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seg = Path("render_output/footage/dancing/lion_segments")
    seg.mkdir(parents=True)
    (seg / "track.json").write_text('{"objects": [{"id": 1, "share": 0.7}, {"id": 2, "share": 0.2}]}')
    assert find_mask_track("C:/x/lion.mp4") == "render_output/footage/dancing/lion_segments"
    fg = new_foreground("C:/x/lion.mp4", [Block(2, 3)])
    assert fg["cutout"]["track"].endswith("lion_segments") and fg["blocks"] == [{"bars": [2, 3]}]
    assert fg["cutout"]["drop"] == [1]                             # the white behind him
    assert "cutout" not in new_foreground("C:/x/nobody.mp4")


def test_list_clips_filters_by_path(tmp_path):
    for p in ("a/final.mp4", "b/final.mp4", "b/other.mp4", "c/still.png", ".thumbs/x.png", "notes.txt"):
        f = tmp_path / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    names = [p.relative_to(tmp_path).as_posix() for p in list_clips(str(tmp_path), "FINAL")]
    assert names == ["a/final.mp4", "b/final.mp4"]
    assert len(list_clips(str(tmp_path))) == 4


def test_a_flat_scene_is_grouped_by_where_its_layers_sit():
    layers, converted = grouped(LAYERS)
    assert converted
    assert [(l["source"], l.get("name")) for l in layers] == [("clips", None), ("group", "BG"), ("group", "FG"),
                                                              ("rgb_split", None)]
    bg, fg = layers[1], layers[2]
    assert bg["canvas"] == "below" and bg["fx"] == [{"source": "pixel_sort", "amount": 0.5}]
    assert [m.label for m in members_of(bg)] == ["a"]                 # named by its clip
    assert [m.label for m in members_of(fg)] == ["lion", "bottles"]  # a clip above the characters: FG
    assert grouped(layers) == (layers, False)                      # grouped already: left so


def test_a_background_lane_of_several_clips_becomes_a_member_each():
    spec = {"source": "footage", "role": "background", "fit": "cover",
            "blocks": [{"bars": [1, 4], "file": "a.mp4", "from": 2.0}, {"bars": [5, 6], "file": "b.mp4"}]}
    bg = grouped([{"source": "clips"}, spec, {"source": "footage", "file": "lion.mp4", "cutout": {"track": "x"}}])[0][1]
    assert [(m.label, [(b.first, b.last, b.start) for b in m.blocks]) for m in members_of(bg)] == [
        ("a", [(1, 4, 2.0)]), ("b", [(5, 6, 0.0)])]


def test_members_at_the_same_time_stack_up():
    lane = {"layers": [{"source": "julia", "bars": [[1, 4]]}, {"source": "julia", "bars": [[5, 8]]},
                       {"source": "julia", "bars": [[3, 6]]}, {"source": "julia"}]}
    mems = members_of(lane)
    assert pack(mems, 16) == 3
    assert [m.row for m in mems] == [2, 2, 1, 0]                   # the topmost (last) on top
    assert mems[3].always and with_windows(mems[0].spec, [Block(2, 3)])["bars"] == [[2, 3]]


def test_bars_windows_show_as_blocks_and_become_blocks():
    spec = {"source": "footage", "file": "lion.mp4", "loop": True, "bars": [[5, 17], [36, 39]], "fade_out": 2}
    (mem,) = members_of({"layers": [spec]})
    assert mem.blocks == [Block(5, 17), Block(36, 39)] and not mem.always
    out = with_blocks(spec, place(mem.blocks, Block(20, 21)))
    assert "bars" not in out and "fade_out" not in out
    assert out["blocks"] == [{"bars": [5, 17]}, {"bars": [20, 21]}, {"bars": [36, 39]}]
    assert members_of({"layers": [{"source": "footage", "file": "x.mp4"}]})[0].always
