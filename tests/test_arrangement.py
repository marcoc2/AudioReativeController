"""The Arrange window's model: lanes from footage layers, blocks placed by bars, new lanes."""
from pathlib import Path

import numpy as np

from core.arrangement import (Block, block_at, blocks_of, find_mask_track, insert_at, lanes_of,
                              list_clips, master_at, new_background, new_foreground, place, remove,
                              role_of, with_blocks, with_windows)

LAYERS = [
    {"source": "clips"},
    {"source": "pixel_sort", "amount": 0.5},
    {"source": "footage", "role": "background", "fit": "cover",
     "blocks": [{"bars": [1, 4], "file": "a.mp4", "from": 2.0}]},
    {"source": "footage", "file": "lion.mp4", "cutout": {"track": "x"}, "blocks": [{"bars": [5, 8]}]},
    {"source": "footage", "file": "bottles.mp4", "cues": {"reaper": "song.rpp"}},
    {"source": "rgb_split"},
]


def test_lanes_are_the_layers():
    lanes = lanes_of(LAYERS)
    assert [(l.kind, l.label) for l in lanes] == [
        ("clips", "clip montage"), ("fx", "pixel_sort"), ("clip", "background"), ("clip", "lion"),
        ("clip", "bottles"), ("fx", "rgb_split")]
    lanes = [l for l in lanes if l.kind == "clip"]
    assert [(l.index, l.role, l.label, l.cued) for l in lanes] == [
        (2, "background", "background", False), (3, "foreground", "lion", False),
        (4, "foreground", "bottles", True)]
    assert lanes[0].blocks == [Block(1, 4, 2.0, 1.0, "a.mp4")]
    assert role_of({"source": "footage", "fit": "cover"}) == "background"


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
    assert new_background()["fit"] == "cover"
    assert insert_at(LAYERS, "background") == 1
    assert insert_at(LAYERS, "foreground") == 5                    # above the last character


def test_list_clips_filters_by_path(tmp_path):
    for p in ("a/final.mp4", "b/final.mp4", "b/other.mp4", "c/still.png", ".thumbs/x.png", "notes.txt"):
        f = tmp_path / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    names = [p.relative_to(tmp_path).as_posix() for p in list_clips(str(tmp_path), "FINAL")]
    assert names == ["a/final.mp4", "b/final.mp4"]
    assert len(list_clips(str(tmp_path))) == 4


def test_bars_windows_show_as_blocks_and_become_blocks():
    spec = {"source": "footage", "file": "lion.mp4", "loop": True, "bars": [[5, 17], [36, 39]], "fade_out": 2}
    lane = lanes_of([spec])[0]
    assert lane.blocks == [Block(5, 17), Block(36, 39)] and not lane.always
    out = with_blocks(spec, place(lane.blocks, Block(20, 21)))
    assert "bars" not in out and "fade_out" not in out
    assert out["blocks"] == [{"bars": [5, 17]}, {"bars": [20, 21]}, {"bars": [36, 39]}]
    assert lanes_of([{"source": "footage", "file": "x.mp4"}])[0].always


def test_generator_and_fx_lanes_are_shown_by_bars():
    layers = [{"source": "clips"}, {"source": "shadertoy", "file": "shaders/shadertoy/nebulosa_arc.glsl",
                                    "bars": [5, 17]},
              {"source": "grain", "enabled": False}, {"source": "meters"}]
    _, gen, grain, meters = lanes_of(layers)
    assert (gen.kind, gen.label, gen.blocks, gen.always) == ("generator", "nebulosa_arc", [Block(5, 17)], False)
    assert (grain.kind, grain.always, grain.enabled) == ("fx", True, False)
    spec = with_windows(gen.spec, place(gen.blocks, Block(30, 33)))
    assert spec["bars"] == [[5, 17], [30, 33]] and "blocks" not in spec
    assert "bars" not in with_windows(gen.spec, [])                  # none: the whole song
    assert master_at(layers) == 3                                    # under the meters read-out
    assert insert_at(layers, "generator") == 1                       # right above the montage
