"""time_map (each pixel from its own moment) and pixel_sort (bright runs sorted into streaks)."""
import numpy as np
import pytest

from core.rhythm.grid import RhythmGrid
from core.video.pixel_sort import PixelSort, sort_rows
from core.video.time_map import TimeMap

W, H, FPS = 40, 20, 10


def _frame(i):
    return np.full((H, W, 3), (i * 10) % 256, np.uint8)        # each frame its own grey


def test_luma_map_looks_back_where_it_is_dark():
    tm = TimeMap({"map": "luma", "depth": 0.5, "step": 0}, [], W, H, FPS)
    for i in range(8):
        f = _frame(i)
        f[:, : W // 2] = 255                                    # left half bright: now
        out = tm.process(f, i / FPS)
    assert (out[:, : W // 2] == 255).all()
    # the right half is dark (70 of 255): (1 - 70/255) * 0.5 s = 0.36 s, so 4 frames back
    assert (out[:, -1] == _frame(7 - 4)[0, 0]).all()


def test_rgb_split_in_time():
    grid = RhythmGrid(bpm=150.0)                                # a sixteenth is 0.1 s: one frame here
    tm = TimeMap({"rgb": True, "step": "1/16"}, [], W, H, FPS, grid=grid)
    for i in range(6):
        out = tm.process(_frame(i), i / FPS)
    assert out[0, 0, 0] == 50 and out[0, 0, 1] == 40 and out[0, 0, 2] == 30


def test_time_map_errors():
    with pytest.raises(ValueError):
        TimeMap({"map": "spiral"}, [], W, H, FPS)
    with pytest.raises(ValueError):
        TimeMap({"map": "figures"}, [], W, H, FPS)              # no mask track


def test_runs_sort_and_the_rest_stays():
    row = np.array([[10, 200, 120, 250, 5, 180, 90]], np.uint8)
    img = np.repeat(row[..., None], 3, axis=2)
    out = sort_rows(img, 100 / 255)[0, :, 0]
    assert list(out) == [10, 120, 200, 250, 5, 180, 90]      # the dark pixels keep their place


def test_pixel_sort_follows_its_amount_and_turns_at_cuts():
    f = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
    assert PixelSort({"amount": 0.0}, [], FPS).process(f, 0.0) is f
    assert not np.array_equal(PixelSort({"amount": 0.9}, [], FPS).process(f, 0.0), f)
    cuts = lambda until: np.array([1.0])
    ps = PixelSort({"amount": 0.9, "direction": ["horizontal", "vertical"]}, [], FPS, cuts=cuts)
    assert ps._direction(0.5) == "horizontal" and ps._direction(1.5) == "vertical"
    with pytest.raises(ValueError):
        PixelSort({"direction": ["horizontal", "vertical"]}, [], FPS)
