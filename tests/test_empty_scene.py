"""A scene built bar by bar: what is not in a bar is black, the first layer obeys its bars
too, and without a clip montage the cuts are where the blocks begin."""
import numpy as np

from core.rhythm.grid import RhythmGrid
from core.video.layers import BlockCuts, build_compositor

W, H = 32, 18
GRID = RhythmGrid(bpm=120.0, time_signature=(4, 4), start_offset=0.0)      # bars of 2 s


def _red(**more):
    return {"source": "solid", "color": [255, 0, 0], **more}


def test_an_empty_scene_is_black():
    comp = build_compositor(None, {"layers": []}, [], W, H, grid=GRID)
    assert comp.names == ["BlackCanvas"]
    assert not comp.frame_at(1.0).any()


def test_the_first_layer_shows_only_in_its_bars():
    comp = build_compositor(None, {"layers": [_red(bars=[[1, 5]])]}, [], W, H, grid=GRID)
    comp.use_gpu = False
    assert tuple(comp.frame_at(9.0)[0, 0]) == (255, 0, 0)          # bar 5
    assert not comp.frame_at(10.5).any()                           # bar 6: nothing there


def test_a_first_layer_that_fills_every_frame_is_the_base_as_before():
    comp = build_compositor(None, {"layers": [_red()]}, [], W, H, grid=GRID)
    assert comp.names == ["solid"]


def test_the_black_under_the_stack_is_a_fresh_frame_each_time():
    comp = build_compositor(None, {"layers": []}, [], W, H, grid=GRID)
    comp.frame_at(0.0)[:] = 9
    assert not comp.frame_at(0.0).any()


def test_without_a_montage_a_cut_is_where_a_block_begins():
    layers = [_red(bars=[[1, 4]]), _red(bars=[[5, 8], [11, 12]]),
              {"source": "rgb_split", "bars": [[3, 3]]}]                    # an FX makes no cut
    assert list(BlockCuts(layers, GRID).cut_times(100.0)) == [8.0, 20.0]
    assert list(BlockCuts(layers, GRID).cut_times(10.0)) == [8.0]


def test_without_blocks_every_bar_is_a_cut():
    assert list(BlockCuts([_red()], GRID).cut_times(7.0)) == [2.0, 4.0, 6.0]
