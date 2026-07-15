"""Tests for core.video: ClipTransport ping-pong/reverse and ClipComposer triggers."""
import numpy as np
import pytest

from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import MidiNote
from core.video.composer import ClipComposer
from core.video.transport import ClipTransport


# ---------------------------------------------------------------------------
# stubs

class StubClip:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def frame(self, idx):
        idx = max(0, min(self.n - 1, idx))
        return np.full((1, 1, 3), idx, dtype=np.uint8)


class StubLibrary:
    def __init__(self, clips):
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def get(self, idx):
        return self.clips[idx % len(self.clips)]


def kick(t, vel=100):
    return MidiNote(time=t, pitch=36, velocity=vel, channel=9, duration=0.1)


def snare(t, vel=100):
    return MidiNote(time=t, pitch=38, velocity=vel, channel=9, duration=0.1)


def frame_val(composer, t):
    return int(composer.frame_at(t)[0, 0, 0])


# ---------------------------------------------------------------------------
# ClipTransport

def test_transport_reverse_toggles_direction():
    tp = ClipTransport(n_clips=2)
    assert tp.direction == 1
    tp.reverse()
    assert tp.direction == -1
    tp.reverse()
    assert tp.direction == 1


def test_transport_pingpong_at_end():
    tp = ClipTransport(n_clips=1)
    seq = []
    for _ in range(8):
        seq.append(tp.frame_index)
        tp.advance(clip_len=4)
    assert seq == [0, 1, 2, 3, 2, 1, 0, 1]
    assert tp.direction == 1  # bounced twice


def test_transport_pingpong_at_start_when_reversed():
    tp = ClipTransport(n_clips=1)
    tp.advance(clip_len=8)      # pos 1
    tp.reverse()
    tp.advance(clip_len=8)      # pos 0
    tp.advance(clip_len=8)      # bounce -> pos 1, forward
    assert tp.frame_index == 1
    assert tp.direction == 1


def test_transport_next_clip_wraps_and_resets():
    tp = ClipTransport(n_clips=3, start_clip=2)
    tp.advance(clip_len=10)
    tp.reverse()
    tp.next_clip()
    assert tp.clip_idx == 0
    assert tp.pos == 0.0
    assert tp.direction == 1


def test_transport_single_frame_clip():
    tp = ClipTransport(n_clips=1)
    for _ in range(5):
        tp.advance(clip_len=1)
    assert tp.frame_index == 0


# ---------------------------------------------------------------------------
# ClipComposer

def make_composer(notes, cfg, clip_len=8, n_clips=2, bpm=60.0):
    grid = RhythmGrid(bpm=bpm, time_signature=(4, 4), fps=4)  # bar = 4s
    lib = StubLibrary([StubClip(clip_len) for _ in range(n_clips)])
    return ClipComposer(lib, grid, notes, cfg)


def test_kick_reverses_playback():
    cfg = {"clip_per_bar": True,
           "triggers": {"kick": {"notes": [36], "actions": ["reverse"]}}}
    comp = make_composer([kick(1.25)], cfg)
    vals = [frame_val(comp, i * 0.25) for i in range(12)]
    # forward 0..4, kick at t=1.25 flips, then back down and bounce at 0
    assert vals == [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1]


def test_new_clip_each_bar():
    cfg = {"clip_per_bar": True, "triggers": {}}
    comp = make_composer([], cfg, n_clips=3)
    comp.frame_at(0.0)
    assert comp.transport.clip_idx == 0
    comp.frame_at(3.9)
    assert comp.transport.clip_idx == 0
    comp.frame_at(4.0)   # bar 1 starts
    assert comp.transport.clip_idx == 1
    comp.frame_at(8.1)   # bar 2
    assert comp.transport.clip_idx == 2


def test_snare_switches_clip():
    cfg = {"clip_per_bar": False,
           "triggers": {"snare": {"notes": [38, 40], "actions": ["next_clip"]}}}
    comp = make_composer([snare(0.6), snare(1.1)], cfg, n_clips=3)
    comp.frame_at(0.0)
    assert comp.transport.clip_idx == 0
    comp.frame_at(0.5)
    assert comp.transport.clip_idx == 0
    v = frame_val(comp, 0.75)          # snare consumed -> clip 1, restarts at 0
    assert comp.transport.clip_idx == 1
    assert v == 0
    comp.frame_at(1.25)
    assert comp.transport.clip_idx == 2


def test_clip_does_not_change_across_bars_when_disabled():
    cfg = {"clip_per_bar": False, "triggers": {}}
    comp = make_composer([], cfg, n_clips=3)
    comp.frame_at(0.0)
    comp.frame_at(4.5)
    comp.frame_at(9.0)
    assert comp.transport.clip_idx == 0


def test_min_velocity_filters_soft_hits():
    cfg = {"clip_per_bar": True,
           "triggers": {"kick": {"notes": [36], "actions": ["reverse"],
                                 "min_velocity": 64}}}
    comp = make_composer([kick(0.5, vel=30), kick(1.0, vel=100)], cfg)
    assert len(comp.events) == 1
    assert comp.events[0].time == 1.0


def test_seek_skips_past_events():
    cfg = {"clip_per_bar": True,
           "triggers": {"kick": {"notes": [36], "actions": ["reverse"]}}}
    comp = make_composer([kick(0.5), kick(2.0)], cfg)
    comp.seek(1.0)
    comp.frame_at(1.0)
    assert comp.transport.direction == 1   # 0.5 kick skipped
    comp.frame_at(2.0)
    assert comp.transport.direction == -1  # 2.0 kick applied


def test_unknown_action_raises():
    cfg = {"triggers": {"kick": {"notes": [36], "actions": ["explode"]}}}
    with pytest.raises(ValueError, match="explode"):
        make_composer([], cfg)
