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
