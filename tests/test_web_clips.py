"""The web Arrange's clips: put as long as they play (a long one asks first), files from the
Explorer, a frame of a clip, and the SAM2 cutout set from its panel."""
import subprocess

import pytest

from tests.test_web_session import _member, _session, project  # noqa: F401  (the fixture)


def _video(path, seconds):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=160x90:rate=10:duration={seconds}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def clips(tmp_path):
    return {"short": _video(tmp_path / "short.mp4", 4), "long": _video(tmp_path / "long.mp4", 40)}


def test_a_clip_takes_as_many_bars_as_it_plays_and_the_next_follows(project, clips):
    s = _session(project)
    st = s.place_clips("bg", 2, [{"path": str(clips["short"])}, {"path": str(clips["short"]), "from": 2.0}])
    blocks = sorted((b["first"], b["last"], b["from"]) for m in st["lanes"] if m["file"].endswith("short.mp4") for b in m["spans"])
    assert blocks == [(2, 3, 0.0), (4, 4, 2.0)]             # 4 s = 2 bars; from 2 s: 1 bar


def test_a_long_clip_asks_and_the_answer_puts_it(project, clips):
    s = _session(project)
    got = s.place_clips("fg", 1, [{"path": str(clips["long"])}, {"path": str(clips["short"])}])
    ask = got["asking"]
    n = len(s.bars()) - 1
    assert ask["bars"] == 20 and ask["whole"] == n and ask["to_end"] and ask["suggest"] == 8
    st = s.place_clips(ask["lane"], ask["first"], ask["todo"], 3, ask["row"])
    fg = [m for m in st["lanes"] if m["lane"] == "fg" and m["file"].endswith(("long.mp4", "short.mp4"))]
    assert sorted((m["spans"][0]["first"], m["spans"][0]["last"]) for m in fg) == [(1, 3), (4, 5)]
    st = s.place_clips("bg", 1, [{"path": str(clips["long"])}], "whole")
    assert any(m["spans"] and m["spans"][-1]["last"] == n for m in st["lanes"] if m["file"].endswith("long.mp4"))


def test_a_frame_of_a_clip(project, clips):
    s = _session(project)
    got = s.clip_frame(str(clips["short"]), 1.5)
    assert got["width"] == 160 and got["url"].endswith(".jpg")


def test_the_sam2_panel_writes_the_fx(project, clips):
    s = _session(project)
    s.place_clips("fg", 1, [{"path": str(clips["short"])}])
    got = s.sam_for_clip(str(clips["short"]))
    assert got["new"]
    fx = got["fx"]
    panel = s.sam_open(fx)
    assert panel["mode"] == "auto" and panel["max_objects"] == 4 and "not cut out yet" in panel["cache"]
    form = {**panel, "mode": "seeds", "at": 1.0, "objects": [{"box": [0.1, 0.1, 0.5, 0.9], "points": [[0.3, 0.5, 1]]}],
            "keep": "drop1"}
    s.sam_apply(fx, form)
    spec = s.get(fx)
    assert spec["seeds"] == {"at": 1.0, "objects": [{"box": [0.1, 0.1, 0.5, 0.9], "points": [[0.3, 0.5, 1]]}]}
    assert spec["drop"] == [1] and "keep" not in spec
    assert "error" in s.sam_apply(fx, {**form, "objects": [{"box": None, "points": []}]})
    assert s.sam_for_clip(str(clips["short"]))["new"] is False
    assert "error" in s.sam_for_clip(str(clips["long"]))
