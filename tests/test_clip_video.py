"""Tests for core.video: ClipTransport ping-pong/reverse and ClipComposer triggers."""
import numpy as np
import pytest

from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import MidiNote
from core.video.composer import ClipComposer, GravityWarp
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


# ---------------------------------------------------------------------------
# GravityWarp (speed warp around drum hits)

def test_gravity_peaks_at_hit_and_floors_far_away():
    g = GravityWarp(times=np.array([5.0]), peak=3.0, floor=0.3, radius=0.5)
    assert g.speed_at(5.0) == pytest.approx(3.0)
    assert g.speed_at(5.5) == pytest.approx(0.3)   # exactly at radius
    assert g.speed_at(0.0) == pytest.approx(0.3)   # far away
    assert g.speed_at(9.9) == pytest.approx(0.3)


def test_gravity_is_symmetric_and_monotonic():
    g = GravityWarp(times=np.array([5.0]), peak=3.0, floor=0.3, radius=0.5, curve=2.0)
    # symmetric: approaching and leaving at same distance -> same speed
    assert g.speed_at(4.8) == pytest.approx(g.speed_at(5.2))
    # monotonic: closer to the hit -> faster
    assert g.speed_at(4.9) > g.speed_at(4.7) > g.speed_at(4.55)


def test_gravity_nearest_hit_wins_between_two_hits():
    g = GravityWarp(times=np.array([2.0, 3.0]), peak=2.0, floor=0.5, radius=0.6)
    # midpoint is 0.5s from both hits -> same as being 0.5s from a single hit
    single = GravityWarp(times=np.array([2.0]), peak=2.0, floor=0.5, radius=0.6)
    assert g.speed_at(2.5) == pytest.approx(single.speed_at(2.5))
    assert g.speed_at(2.9) > g.speed_at(2.5)


def test_gravity_floor_never_stalls():
    g = GravityWarp(times=np.array([1.0]), peak=2.0, floor=0.0, radius=0.5)
    assert g.speed_at(0.0) >= 0.05


def test_composer_applies_gravity_speed_to_transport():
    cfg = {"clip_per_bar": True,
           "triggers": {"kick": {"notes": [36], "actions": [],
                                 "gravity": {"peak": 4.0, "floor": 1.0,
                                             "radius": 1.0, "curve": 1.0}}}}
    comp = make_composer([kick(1.0)], cfg, clip_len=64)
    comp.frame_at(1.0)                    # at the hit -> full speed
    assert comp.transport.speed == pytest.approx(4.0)
    p_fast = comp.transport.pos           # advanced 4 frames
    assert p_fast == pytest.approx(4.0)
    comp.frame_at(3.0)                    # 2s away, beyond radius -> floor
    assert comp.transport.speed == pytest.approx(1.0)


def test_composer_speed_is_one_without_gravity():
    cfg = {"clip_per_bar": True,
           "triggers": {"kick": {"notes": [36], "actions": ["reverse"]}}}
    comp = make_composer([kick(1.0)], cfg)
    assert comp.speed_at(1.0) == 1.0


# ---------------------------------------------------------------------------
# dry-run resolver

def test_resolver_segments_and_speed():
    from core.video.resolver import resolve_song
    grid = RhythmGrid(bpm=60.0, time_signature=(4, 4), fps=4)   # bar = 4s
    lib = StubLibrary([StubClip(8), StubClip(8)])
    cfg = {"clip_per_bar": True, "clip_order": "sequential",
           "triggers": {"kick": {"notes": [36], "actions": ["reverse"],
                                 "gravity": {"peak": 4.0, "floor": 1.0,
                                             "radius": 0.5, "curve": 1.0}}}}
    segs, times, speeds = resolve_song(lib, grid, [kick(1.25)], cfg,
                                       fps=4, start=0.0, end=8.0)
    assert segs[0].clip_idx == 0 and segs[0].direction == 1
    rev = [s for s in segs if s.direction == -1]
    assert rev and abs(rev[0].t0 - 1.25) < 1e-6          # reverse at the kick
    assert any(s.clip_idx == 1 for s in segs)            # bar switch to clip 1
    assert len(times) == 32
    # gravity: peak speed at the kick, floor far away
    import numpy as np
    assert speeds[np.argmin(np.abs(times - 1.25))] == pytest.approx(4.0)
    assert speeds[-1] == pytest.approx(1.0)
    # segments tile the range without gaps
    for a, b in zip(segs, segs[1:]):
        assert a.t1 == pytest.approx(b.t0)


# ---------------------------------------------------------------------------
# shuffle bag clip order

def shuffle_selections(comp, n_bars, bar_dur=4.0):
    picks = []
    for b in range(n_bars):
        comp.frame_at(b * bar_dur)
        picks.append(comp.transport.clip_idx)
    return picks


def test_shuffle_uses_every_clip_once_per_cycle():
    cfg = {"clip_per_bar": True, "clip_order": "shuffle", "seed": 123,
           "triggers": {}}
    comp = make_composer([], cfg, n_clips=5)
    picks = shuffle_selections(comp, 10)
    assert sorted(picks[:5]) == [0, 1, 2, 3, 4]   # first cycle: each exactly once
    assert sorted(picks[5:]) == [0, 1, 2, 3, 4]   # second cycle too


def test_shuffle_no_immediate_repeat_across_cycles():
    for seed in range(20):
        cfg = {"clip_per_bar": True, "clip_order": "shuffle", "seed": seed,
               "triggers": {}}
        comp = make_composer([], cfg, n_clips=4)
        picks = shuffle_selections(comp, 12)
        for a, b in zip(picks, picks[1:]):
            assert a != b


def test_until_hands_over_between_triggers():
    # kick fires until the first snare hit; snare fires from then on
    cfg = {"clip_per_bar": False,
           "triggers": {
               "kick":  {"notes": [36], "actions": ["next_clip"], "until": "snare"},
               "snare": {"notes": [38], "actions": ["next_clip"]},
           }}
    notes = [kick(0.5), kick(1.5), snare(2.0), kick(2.5), snare(3.0), kick(3.5)]
    comp = make_composer(notes, cfg, n_clips=5)
    fired = [(e.time, e.name) for e in comp.events]
    assert fired == [(0.5, "kick"), (1.5, "kick"), (2.0, "snare"), (3.0, "snare")]


def test_until_kick_at_exact_handover_time_is_dropped():
    cfg = {"clip_per_bar": False,
           "triggers": {
               "kick":  {"notes": [36], "actions": ["next_clip"], "until": "snare"},
               "snare": {"notes": [38], "actions": ["next_clip"]},
           }}
    comp = make_composer([kick(2.0), snare(2.0)], cfg)
    assert [(e.time, e.name) for e in comp.events] == [(2.0, "snare")]


def test_until_unknown_trigger_raises():
    cfg = {"triggers": {"kick": {"notes": [36], "actions": ["next_clip"],
                                 "until": "caixa"}}}
    with pytest.raises(ValueError, match="caixa"):
        make_composer([], cfg)


def test_until_applies_to_gravity_times():
    cfg = {"clip_per_bar": False,
           "triggers": {
               "kick":  {"notes": [36], "actions": [], "until": "snare",
                         "gravity": {"peak": 5.0, "floor": 1.0, "radius": 0.2}},
               "snare": {"notes": [38], "actions": ["next_clip"]},
           }}
    comp = make_composer([kick(1.0), snare(2.0), kick(3.0)], cfg)
    assert list(comp.gravity[0].times) == [1.0]   # kick at 3.0 cut off


def test_audio_trigger_uses_onset_loader():
    def fake_loader(spec):
        assert spec["audio"] == "stems/caixa.wav"
        return [MidiNote(time=t, pitch=-1, velocity=100, channel=0, duration=0.0)
                for t in (0.6, 1.2)]

    cfg = {"clip_per_bar": False,
           "triggers": {"snare": {"audio": "stems/caixa.wav",
                                  "actions": ["next_clip"]}}}
    grid = RhythmGrid(bpm=60.0, time_signature=(4, 4), fps=4)
    lib = StubLibrary([StubClip(8) for _ in range(3)])
    comp = ClipComposer(lib, grid, [], cfg, onset_loader=fake_loader)
    assert [e.time for e in comp.events] == [0.6, 1.2]
    comp.frame_at(0.0)
    assert comp.transport.clip_idx == 0
    comp.frame_at(0.75)   # first audio onset consumed
    assert comp.transport.clip_idx == 1


def test_exclude_drops_hits_near_other_trigger():
    # snare onsets at 1.0 and 2.0; kick at 1.02 -> the 1.0 onset is bleed
    def fake_loader(spec):
        return [MidiNote(time=t, pitch=-1, velocity=100, channel=0, duration=0.0)
                for t in (1.0, 2.0)]

    cfg = {"clip_per_bar": False,
           "triggers": {
               "kick":  {"notes": [36], "actions": []},
               "snare": {"audio": "s.wav", "actions": ["next_clip"],
                         "exclude": {"trigger": "kick", "window": 0.04}},
           }}
    grid = RhythmGrid(bpm=60.0, time_signature=(4, 4), fps=4)
    lib = StubLibrary([StubClip(8) for _ in range(3)])
    comp = ClipComposer(lib, grid, [kick(1.02)], cfg, onset_loader=fake_loader)
    assert [e.time for e in comp.events] == [2.0]


def test_exclude_then_until_handover_uses_cleaned_first_hit():
    # bleed onset at 1.0 must NOT count as the snare entrance; the real
    # first snare is 3.0, so kicks before 3.0 still fire
    def fake_loader(spec):
        return [MidiNote(time=t, pitch=-1, velocity=100, channel=0, duration=0.0)
                for t in (1.0, 3.0, 4.0)]

    cfg = {"clip_per_bar": False,
           "triggers": {
               "kick":  {"notes": [36], "actions": ["next_clip"], "until": "snare"},
               "snare": {"audio": "s.wav", "actions": ["next_clip"],
                         "exclude": {"trigger": "kick", "window": 0.04}},
           }}
    notes = [kick(1.0), kick(2.0), kick(3.5)]
    grid = RhythmGrid(bpm=60.0, time_signature=(4, 4), fps=4)
    lib = StubLibrary([StubClip(8) for _ in range(3)])
    comp = ClipComposer(lib, grid, notes, cfg, onset_loader=fake_loader)
    fired = [(e.time, e.name) for e in comp.events]
    assert fired == [(1.0, "kick"), (2.0, "kick"), (3.0, "snare"), (4.0, "snare")]


def test_exclude_unknown_trigger_raises():
    def fake_loader(spec):
        return []
    cfg = {"triggers": {"snare": {"audio": "s.wav", "actions": ["next_clip"],
                                  "exclude": {"trigger": "bumbo"}}}}
    grid = RhythmGrid(bpm=60.0, time_signature=(4, 4), fps=4)
    lib = StubLibrary([StubClip(8)])
    with pytest.raises(ValueError, match="bumbo"):
        ClipComposer(lib, grid, [], cfg, onset_loader=fake_loader)


def test_midi_until_audio_handover():
    def fake_loader(spec):
        return [MidiNote(time=2.0, pitch=-1, velocity=90, channel=0, duration=0.0)]

    cfg = {"clip_per_bar": False,
           "triggers": {
               "kick":  {"notes": [36], "actions": ["next_clip"], "until": "snare"},
               "snare": {"audio": "s.wav", "actions": ["next_clip"]},
           }}
    grid = RhythmGrid(bpm=60.0, time_signature=(4, 4), fps=4)
    lib = StubLibrary([StubClip(8) for _ in range(3)])
    comp = ClipComposer(lib, grid, [kick(1.0), kick(3.0)], cfg,
                        onset_loader=fake_loader)
    assert [(e.time, e.name) for e in comp.events] == [(1.0, "kick"), (2.0, "snare")]


def test_shuffle_reshuffles_instead_of_repeating_order():
    # with enough clips, at least one seed must produce differing cycle orders
    differing = False
    for seed in range(10):
        cfg = {"clip_per_bar": True, "clip_order": "shuffle", "seed": seed,
               "triggers": {}}
        comp = make_composer([], cfg, n_clips=6)
        picks = shuffle_selections(comp, 12)
        if picks[:6] != picks[6:]:
            differing = True
            break
    assert differing
