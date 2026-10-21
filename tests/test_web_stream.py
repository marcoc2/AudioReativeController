"""The live preview's stream: the engine's frames made ahead of the page, in order, kept for
a loop played again, a new stream after an edit (a fake stack stands in for the engine)."""
import threading
import time

import numpy as np

from core.web import preview
from core.web.stream import EngineStream


class FakeSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.grid, self.doc, self.features_at, self.lib = object(), object(), (lambda t: {}), object()
        self.fps, self.dur = 10, 6.0

    def specs(self):
        return [{"source": "clips"}]

    def clips_dir(self):
        return "clips"


class Events:
    def __init__(self):
        self.got = []

    def emit(self, fn, data):
        self.got.append((fn, data))

    def log(self, kind, text):
        self.got.append(("log", text))


class Stack:
    def frame_at(self, t):
        return np.full((8, 8, 3), int(t * 10) % 255, dtype=np.uint8)


def _wait(cond, secs=5.0):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_frames_are_made_ahead_and_kept(monkeypatch):
    built = []
    monkeypatch.setattr(preview, "build_stack", lambda s, a, b, check, tell: built.append((a, b)) or (Stack(), 10, 8, 8))
    st = EngineStream(FakeSession(), Events())
    info = st.start(1.0)
    gen = info["gen"]
    assert _wait(lambda: st.made >= 20)                    # 2.5 s ahead of frame 0, at 10 fps
    got, state = st.frame(gen, 3)
    assert got[:2] == b"\xff\xd8" and state["made"] >= 3   # a JPEG
    assert st.start(1.3)["gen"] == gen                     # a second it has: the same stream
    assert st.start(1.0, force=True)["gen"] == gen + 1     # an edit: a new one
    assert st.frame(gen, 3) == (None, None)                # the old one is gone
    assert built[0][0] == 1.0


def test_it_waits_for_the_page_and_ends_at_the_song_end(monkeypatch):
    monkeypatch.setattr(preview, "build_stack", lambda s, a, b, check, tell: (Stack(), 10, 8, 8))
    st = EngineStream(FakeSession(), Events())
    gen = st.start(0.0)["gen"]
    assert _wait(lambda: st.made >= 25)
    time.sleep(0.3)
    assert st.made <= 26                                   # no further than 2.5 s past the page
    for fi in range(0, 60, 5):
        st.frame(gen, fi)
        time.sleep(0.05)
    assert _wait(lambda: st.ended)
    assert st.made == 59                                   # 6 s at 10 fps


def test_an_engine_error_is_told(monkeypatch):
    def broken(*a):
        raise ValueError("no such layer")
    monkeypatch.setattr(preview, "build_stack", broken)
    ev = Events()
    st = EngineStream(FakeSession(), ev)
    gen = st.start(0.0)["gen"]
    assert _wait(lambda: st.error is not None)
    assert "no such layer" in st.frame(gen, 0)[1]["error"]
    assert any(fn == "stream" for fn, _ in ev.got)
