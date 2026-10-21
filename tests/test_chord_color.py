"""chord_color: the chord sounding recolours the picture (numpy reference, GPU checked against it)."""
import numpy as np
import pytest

from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import MidiNote
from core.video.chord_color import ChordColor, chord_ramp

W, H = 48, 32


@pytest.fixture
def score(tmp_path):
    rows = [(0.0, 2.0, "E:min7"), (2.0, 4.0, "N"), (4.0, 6.0, "G:maj"), (6.0, 8.0, "B:dim")]
    (tmp_path / "chord.lab").write_text("\n".join("\t".join(map(str, r)) for r in rows) + "\n", encoding="utf-8")
    return {"score": str(tmp_path), "snap": "off"}


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    f = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    f[:, :8] = np.linspace(0, 255, H, dtype=np.uint8)[:, None, None]      # a grey ramp at the left
    return f


def test_the_ramp_runs_dark_to_light_and_minor_is_darker():
    major, minor = chord_ramp(7, "maj", (7, 11, 2)), chord_ramp(4, "min", (4, 7, 11))
    luma = lambda r: r.dot(np.array([0.299, 0.587, 0.114]))
    assert np.all(np.diff(luma(major)) > -1.0)                 # it climbs (a level of slack for hue turns)
    assert luma(major)[0] < 5 and luma(major)[-1] > 200
    assert luma(minor).mean() < luma(major).mean()
    sat = lambda r: (r.max(1) - r.min(1)) / np.maximum(r.max(1), 1)
    assert sat(minor)[64:192].mean() < sat(major)[64:192].mean()


def test_it_follows_the_chord_and_glides_in(score):
    layer = ChordColor({"harmony": score, "amount": 1.0, "glide": 0.5}, [], 30)
    em, g = layer.lut(1.0), layer.lut(5.0)
    assert np.array_equal(layer.lut(3.0), em)                  # no chord: the last one stays
    assert np.array_equal(layer.lut(-1.0), em)                 # before the first: the first
    mid = layer.lut(4.25)                                      # half way through the glide
    assert np.allclose(mid, (em + g) / 2, atol=1e-3)
    grey = np.full((H, W, 3), 128, np.uint8)
    out = layer.process(grey, 5.0)
    assert np.allclose(out[0, 0], np.floor(g[128] + 0.5), atol=1)


def test_amount_and_pulse(score):
    kick = [MidiNote(time=1.0, pitch=36, velocity=100, channel=0, duration=0.1, track="kick", track_index=0)]
    layer = ChordColor({"harmony": score, "amount": 0.0, "glide": 0,
                        "pulse": {"hits": {"track": "kick"}, "envelope": 0.5, "amount": 1.0}}, kick, 30)
    f = _frame()
    assert layer.process(f, 0.5) is f                          # nothing to do: the frame as it was
    assert layer.amount(1.0) == 1.0 and 0.0 < layer.amount(1.25) < 1.0
    assert not np.array_equal(layer.process(f, 1.0), f)
    with pytest.raises(ValueError):
        ChordColor({"amount": 1.0}, [], 30)                    # no harmony


def test_glide_as_a_note_value_needs_the_grid(score):
    grid = RhythmGrid(bpm=120.0)
    assert ChordColor({"harmony": score, "glide": "1/4"}, [], 30, grid=grid)._glide == 0.5
    assert ChordColor({"harmony": score}, [], 30, grid=grid)._glide == 0.5   # default: a beat


def _gpu(w, h):
    try:
        from core.video.gpu_compose import GpuComposer
        return GpuComposer(w, h)
    except Exception as e:
        if "context" in str(e).lower() or "moderngl" in str(e).lower():
            pytest.skip("no GPU/moderngl context available")
        raise


@pytest.mark.parametrize("amount", [0.35, 1.0])
def test_the_gpu_matches_numpy(score, amount):
    from core.video.gpu_compose import Frame
    g = _gpu(W, H)
    cpu, gpu = (ChordColor({"harmony": score, "amount": amount, "glide": 0.5}, [], 30) for _ in range(2))
    for i, t in enumerate((1.0, 4.1, 4.3, 7.0)):
        f = _frame(i)
        want = cpu.process(f, t)
        g.begin()
        got = gpu.process_gpu(Frame(g, cpu=f), t, g).cpu()
        d = np.abs(got.astype(int) - want.astype(int))
        assert d.max() <= 2 and (d > 0).mean() < 0.02, (t, d.max(), (d > 0).mean())
