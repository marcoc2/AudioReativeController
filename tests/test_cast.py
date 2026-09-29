"""cast — instruments cast onto the objects of a mask track, by role."""
import numpy as np
import pytest

from core.rhythm.midi_reader import MidiNote
from core.segment.mask_track import write_mask_track
from core.video.cast import CastLayer
from core.video.layers import build_compositor
from tests.test_mask_track import BLUE, FPS, GREEN, H, RED, W, YELLOW, _ColourSegmenter, _frame

START = 13.0


@pytest.fixture(scope="module")
def track(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cast") / "seg"
    meta = write_mask_track("fake.mp4", lambda: (_frame(i) for i in range(20)), _ColourSegmenter(),
                            str(folder), FPS, start_time=START, reseed=0.3, log=lambda m: None)
    ids = {tuple(o["colour"]): o["id"] for o in meta["objects"]}
    return str(folder), ids


def _notes(*hits):
    return [MidiNote(time=t, pitch=36, velocity=100, channel=9, duration=0.1, track=name)
            for name, t in hits]


def _cast(folder, roles, notes=()):
    return CastLayer({"track": folder, "roles": roles}, list(notes), W, H, FPS)


def test_roles_are_filled_per_shot(track):
    folder, ids = track
    cast = _cast(folder, [{"who": "largest", "plays": {"notes": [36]}}])
    assert cast._cast("largest", 0) == [ids[RED]]
    assert cast._cast("second", 0) == [ids[BLUE]]
    assert cast._cast("third", 0) == [ids[YELLOW]]            # came in later, still in the shot
    assert cast._cast("fourth", 0) == []
    assert cast._cast("rightmost", 0) == [ids[BLUE]] and cast._cast("leftmost", 0) == [ids[RED]]
    assert cast._cast("largest", 12) == [ids[GREEN]]          # the next shot recasts
    assert cast._cast("background", 3) is None
    assert sorted(cast._cast("all", 3)) == sorted([ids[RED], ids[BLUE], ids[YELLOW]])


def _red_pixels(img):
    return int(np.all(np.abs(img.astype(int) - RED) < 12, axis=2).sum())


def test_punch_on_the_hit_only(track):
    folder, _ = track
    cast = _cast(folder, [{"who": "largest", "plays": {"track": "kick"}, "act": "punch", "amount": 0.3}],
                 _notes(("kick", START + 0.3)))
    before = cast.process(_frame(2), START + 0.2)
    assert np.array_equal(before, _frame(2))
    hit = cast.process(_frame(3), START + 0.3)
    assert _red_pixels(hit) > 1600 * 1.4                     # 1.3× wider and taller
    later = cast.process(_frame(9), START + 0.9)             # the envelope has died away
    assert np.array_equal(later, _frame(9))


def test_tint_the_background_leaves_the_objects(track):
    folder, _ = track
    cast = _cast(folder, [{"who": "background", "plays": {"track": "bass"}, "act": "tint",
                           "color": [0, 0, 255], "amount": 1.0}], _notes(("bass", START)))
    out = cast.process(_frame(0), START)
    assert (out[5, 5] != _frame(0)[5, 5]).any() and out[5, 5, 2] > out[5, 5, 0]
    assert (out[60, 30] == RED).all() and (out[20, 130] == BLUE).all()


def test_presence_until_the_instrument_plays(track):
    folder, _ = track
    cast = _cast(folder, [{"who": "largest", "plays": {"track": "voice"}, "act": "presence",
                           "hold": 0.4, "fade": 0.1}], _notes(("voice", START + 0.5)))
    dark = cast.process(_frame(2), START + 0.2)
    assert dark[60, 30].max() < 60 and (dark[20, 130] == BLUE).all()   # only the lead is a silhouette
    lit = cast.process(_frame(7), START + 0.7)
    assert (lit[60, 58] == RED).all()
    gone = cast.process(_frame(9), START + 1.2 - 0.2)                   # quiet for 0.5 s > hold + fade
    assert gone[60, 70].max() < 60


def test_shake_and_glow_change_the_picture_on_the_hit(track):
    folder, _ = track
    for act in ("shake", "glow"):
        cast = _cast(folder, [{"who": "largest", "plays": {"track": "snare"}, "act": act}],
                     _notes(("snare", START)))
        assert not np.array_equal(cast.process(_frame(0), START), _frame(0))


def test_in_a_scene_with_the_render_masks(track):
    folder, _ = track
    cfg = {"layers": [{"source": "solid", "color": [240, 240, 240]},
                      {"source": "cast", "roles": [{"who": "all", "plays": {"track": "kick"},
                                                    "act": "presence"}]}]}
    comp = build_compositor(None, cfg, _notes(("kick", START + 5)), W, H, fps=FPS, masks=folder)
    out = comp.frame_at(START)
    assert out[60, 30].max() < 60                                       # an object, still silent


def test_mistakes_are_named(track):
    folder, _ = track
    bad = [({"who": "tallest", "plays": {"track": "kick"}}, "who"),
           ({"who": "largest", "plays": {"track": "kick"}, "act": "dance"}, "act"),
           ({"who": "largest"}, "plays"),
           ({"who": "background", "plays": {"track": "kick"}, "act": "punch"}, "background"),
           ({"who": "largest", "plays": {"track": "kick"}, "act": "tint", "color": "chord"}, "harmony")]
    for role, word in bad:
        with pytest.raises(ValueError, match=word):
            _cast(folder, [role], _notes(("kick", START)))
    with pytest.raises(ValueError, match="mask track"):
        CastLayer({"roles": [{"plays": {"track": "kick"}}]}, [], W, H, FPS)


def _gpu(w, h):
    try:
        from core.video.gpu_compose import GpuComposer
        return GpuComposer(w, h)
    except Exception as exc:
        pytest.skip(f"no OpenGL: {exc}")


@pytest.mark.parametrize("act", ["punch", "shake", "tint", "glow", "presence"])
@pytest.mark.parametrize("size", [(W, H), (2 * W + 3, 2 * H - 1)])   # the track stretched, unevenly
def test_the_gpu_plays_what_numpy_plays(track, act, size):
    import cv2
    from core.video.gpu_compose import Frame
    folder, _ = track
    rw, rh = size
    g = _gpu(rw, rh)
    roles = [{"who": "largest", "plays": {"track": "snare"}, "act": act, "color": [40, 200, 255]},
             {"who": "background", "plays": {"track": "snare"}, "act": "tint", "amount": 0.3}]
    cast = CastLayer({"track": folder, "roles": roles}, _notes(("snare", START + 0.3)), rw, rh, FPS)
    noise = np.random.default_rng(7).integers(0, 40, (rh, rw, 3))
    frame = np.clip(cv2.resize(_frame(3), (rw, rh)).astype(int) - noise, 0, 255).astype(np.uint8)
    t = START + 0.35                        # just after the hit (for presence: still fading in)
    want = cast.process(frame, t)
    assert not np.array_equal(want, frame)
    g.begin()
    got = cast.process_gpu(Frame(g, cpu=frame), t, g).cpu()
    d = np.abs(got.astype(int) - want.astype(int))
    assert d.max() <= 2 and (d > 0).mean() < 0.01, (act, d.max(), (d > 0).mean())


def test_the_gpu_leaves_a_quiet_frame_alone(track):
    from core.video.gpu_compose import Frame
    folder, _ = track
    g = _gpu(W, H)
    cast = _cast(folder, [{"who": "largest", "plays": {"track": "kick"}, "act": "punch"}],
                 _notes(("kick", START + 0.9)))
    g.begin()
    f = Frame(g, cpu=_frame(2))
    assert cast.process_gpu(f, START + 0.2, g) is f and g.uploads == 0


def _held(name, t, dur, pitch=60):
    return MidiNote(time=t, pitch=pitch, velocity=127, channel=0, duration=dur, track=name)


def test_rings_ride_out_from_the_outline(track):
    folder, _ = track
    cast = _cast(folder, [{"who": "largest", "plays": {"track": "acid"}, "act": "rings", "width": 12,
                           "color": [0, 0, 255], "speed": 300}], [_held("acid", START, 0.2)])
    # radius at 0.2 s: 300 px/s at 720p -> 50 px/s here -> 10 px out of the red square,
    # whose right edge is at x 57 on that frame
    out = cast.process(_frame(2), START + 0.2).astype(int)
    blue = out[60, :, 2] - out[60, :, 0]
    assert blue[67] > 120 and blue[64] < blue[67] and blue[70] < blue[67]   # a ring, 10 px out
    assert blue[90] == 0                                                   # nothing beyond it
    assert (out[60, 40] == RED).all()                                      # the object is left alone
    assert np.array_equal(cast.process(_frame(9), START + 1.5), _frame(9))   # the ring has died


def test_echo_leaves_a_trail_while_the_note_is_held(track):
    folder, _ = track
    cast = _cast(folder, [{"who": "largest", "plays": {"track": "lead"}, "act": "echo", "amount": 1.0,
                           "every": 0.05}], [_held("lead", START, 0.6)])
    for i in range(6):                                           # the red square slides 4 px a frame
        out = cast.process(_frame(i), START + i / FPS)
    assert out[60, 15, 0] > out[60, 15, 2] + 40                  # where it was at frame 0: a red ghost
    assert (out[60, 50] == RED).all()                            # where it is now: itself
    assert (out[20, 130] == BLUE).all()                          # the others untouched
    short = _cast(folder, [{"who": "largest", "plays": {"track": "lead"}, "act": "echo"}],
                  [_held("lead", START, 0.05)])
    for i in range(6):
        out = short.process(_frame(i), START + i / FPS)
    assert np.array_equal(out, _frame(5))                        # staccato: its trail is already gone


@pytest.mark.parametrize("role", [
    {"act": "rings", "plays": {"track": "acid"}},
    {"act": "rings", "plays": {"track": "acid"}, "color": [255, 255, 0], "width": 6},
    {"act": "echo", "plays": {"track": "lead"}},
    {"act": "echo", "plays": {"track": "lead"}, "color": [80, 255, 120], "every": 0.1}])
@pytest.mark.parametrize("size", [(W, H), (2 * W + 3, 2 * H - 1)])
def test_the_gpu_keeps_step_with_numpy_over_time(track, role, size):
    import cv2
    from core.video.gpu_compose import Frame
    folder, _ = track
    rw, rh = size
    g = _gpu(rw, rh)
    notes = [_held("acid", START + 0.1, 0.2, 48), _held("acid", START + 0.3, 0.1, 67),
             _held("lead", START, 0.5), _held("lead", START + 0.7, 0.1)]
    spec = {"track": folder, "roles": [{"who": "largest", **role}]}
    cpu, gpu = (CastLayer(spec, notes, rw, rh, FPS) for _ in range(2))
    for i in range(14):                                          # across the cut at frame 10
        frame = cv2.resize(_frame(i), (rw, rh))
        t = START + i / FPS
        want = cpu.process(frame, t)
        g.begin()
        got = gpu.process_gpu(Frame(g, cpu=frame), t, g).cpu()
        d = np.abs(got.astype(int) - want.astype(int))
        assert d.max() <= 2 and (d > 0).mean() < 0.01, (role, i, d.max(), (d > 0).mean())


def test_the_background_has_no_outline_to_ring(track):
    folder, _ = track
    for act in ("rings", "echo"):
        with pytest.raises(ValueError, match="background"):
            _cast(folder, [{"who": "background", "plays": {"track": "acid"}, "act": act}])
