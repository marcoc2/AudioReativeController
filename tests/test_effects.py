import numpy as np
import pytest

from core.rhythm.grid import RhythmGrid


# ---------------------------------------------------------------------------
# CubeField / CubesLayer (GPU stage: reactive clips on a cube fly-through)

def _checker_tex(res=64):
    t = np.zeros((res, res, 3), dtype=np.uint8)
    t[: res // 2] = (200, 40, 40)          # top half red
    t[res // 2:] = (40, 40, 200)           # bottom half blue
    t[: res // 8, :] = (240, 240, 0)       # yellow strip marks the top
    return t


def test_cube_field_render_loop_and_determinism():
    try:
        from core.cubes import CubeField
        spec = {"n_cubes": 24, "faces_with_clips": 3, "face_resolution": 64,
                "seed": 7, "corridor_depth": 20.0}
        field = CubeField(96, 96, spec, seed=7)
    except Exception:
        pytest.skip("no GPU/moderngl context available")

    tex = _checker_tex()
    a = field.render(0.0, tex, [0, 0, 0])
    assert a.shape == (96, 96, 3) and a.dtype == np.uint8
    assert a.sum() > 0                                   # cubes are visible

    mid = field.render(0.5, tex, [0, 0, 0])
    assert not np.array_equal(a, mid)                    # camera flies forward

    end = field.render(0.999999, tex, [0, 0, 0])
    diff_loop = np.abs(a.astype(int) - end.astype(int)).mean()
    diff_half = np.abs(a.astype(int) - mid.astype(int)).mean()
    assert diff_loop < diff_half * 0.25                 # phase 0 == phase 1 (seamless)

    # texture is the reactive content: change it, frame must change
    alt = field.render(0.5, np.full_like(tex, 90), [0, 0, 0])
    assert not np.array_equal(mid, alt)

    # determinism: same seed -> identical field
    field2 = CubeField(96, 96, spec, seed=7)
    assert np.array_equal(field.render(0.3, tex, [0, 0, 0]),
                          field2.render(0.3, tex, [0, 0, 0]))


def test_cube_field_bg_color_fills_empty_space():
    try:
        from core.cubes import CubeField
        # zero cubes -> the whole frame is just the empty-space background
        field = CubeField(48, 48, {"n_cubes": 0, "face_resolution": 32}, seed=1)
    except Exception:
        pytest.skip("no GPU/moderngl context available")
    frame = field.render(0.0, np.zeros((32, 32, 3), np.uint8), [10, 20, 30])
    assert frame.shape == (48, 48, 3)
    assert np.allclose(frame.reshape(-1, 3).mean(0), [10, 20, 30], atol=2)


def _write_clip(path, color, dur=1.0, fps=8, size=64):
    import subprocess
    c = "0x%02x%02x%02x" % tuple(color)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c={c}:s={size}x{size}:r={fps}:d={dur}",
         "-pix_fmt", "yuv420p", str(path)], check=True)


def test_cubes_layer_self_contained(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    try:
        _write_clip(clips / "a.mp4", (200, 0, 0))
        _write_clip(clips / "b.mp4", (0, 0, 200))
    except Exception:
        pytest.skip("ffmpeg not available")

    from core.rhythm.midi_reader import MidiNote
    from core.video.layers import CubesLayer
    grid = RhythmGrid(bpm=120.0, fps=8)          # bar = 2s
    spec = {"source": "cubes", "clips_dir": str(clips), "face_resolution": 64,
            "n_cubes": 12, "faces_with_clips": 3, "loop_bars": 1, "seed": 3,
            "clip_reactivity": {"clip_per_bar": True}}
    try:
        layer = CubesLayer(spec, notes=[], width=96, height=96, fps=8, grid=grid)
    except Exception:
        pytest.skip("no GPU/moderngl context available")

    f0 = layer.frame_at(0.0)
    assert f0.shape == (96, 96, 3) and f0.dtype == np.uint8
    assert f0.sum() > 0
    # self-contained clip pool is live: a later time still renders a valid frame
    f1 = layer.frame_at(1.5)
    assert f1.shape == (96, 96, 3)


def _dominant_channels(frame, thresh=70):
    """Set of dominant color channels among clearly-colored pixels."""
    f = frame.astype(int)
    bright = f.max(-1) > thresh
    dom = f.argmax(-1)[bright]
    return {c for c in (0, 1, 2) if (dom == c).sum() > 40}


def test_clip_library_cover_crops_black_bars(tmp_path):
    import subprocess
    from core.video.clip_library import ClipLibrary
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "color=c=white:s=160x90:r=8:d=0.4",
                        "-pix_fmt", "yuv420p", str(tmp_path / "wide.mp4")], check=True)
    except Exception:
        pytest.skip("ffmpeg not available")

    contain = ClipLibrary(tmp_path, 64, 64, 8, fit="contain").get(0).frame(0)
    cover = ClipLibrary(tmp_path, 64, 64, 8, fit="cover").get(0).frame(0)
    # 16:9 white clip into a square face: contain letterboxes (black top row),
    # cover fills to the edge (no bars).
    assert int(contain[0].sum()) == 0        # top row is a black letterbox bar
    assert int(cover[0].sum()) > 0           # cover reaches the top edge
    assert int(cover.min()) > 200            # whole face is the (white) footage


def test_cubes_layer_one_clip_per_face(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    try:
        _write_clip(clips / "r.mp4", (220, 0, 0))
        _write_clip(clips / "g.mp4", (0, 220, 0))
        _write_clip(clips / "b.mp4", (0, 0, 220))
    except Exception:
        pytest.skip("ffmpeg not available")

    from core.video.layers import CubesLayer
    grid = RhythmGrid(bpm=120.0, fps=8)
    spec = {"source": "cubes", "clips_dir": str(clips), "face_resolution": 64,
            "n_cubes": 20, "faces_with_clips": 3, "one_clip_per_face": True,
            "loop_bars": 1, "seed": 5,
            "clip_reactivity": {"clip_per_bar": True, "clip_order": "shuffle"}}
    try:
        layer = CubesLayer(spec, notes=[], width=128, height=128, fps=8, grid=grid)
    except Exception:
        pytest.skip("no GPU/moderngl context available")

    assert len(layer.composers) == 3        # one screen per clip face-index
    # across a few frames the faces show different clips -> >= 2 distinct hues
    seen = set()
    for t in (0.0, 0.6, 1.2):
        seen |= _dominant_channels(layer.frame_at(t))
    assert len(seen) >= 2
