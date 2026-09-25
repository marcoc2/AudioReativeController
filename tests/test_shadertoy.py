"""Shadertoy adapter (GPU): skips without a moderngl context."""
import numpy as np
import pytest

from core.rhythm import MidiNote
from core.shadertoy import load, wrap
from core.video.layers import ShadertoyLayer, build_compositor

W, H = 64, 36


def _toy(code, **kw):
    try:
        from core.shadertoy import Shadertoy
        return Shadertoy(code, W, H, **kw)
    except Exception as e:                      # only a missing GPU is a skip; bad GLSL must fail
        if "context" in str(e).lower() or "moderngl" in str(e).lower():
            pytest.skip("no GPU/moderngl context available")
        raise


def test_wrap_declares_what_the_shader_does_not():
    code = "uniform float u_a;\nvoid mainImage(out vec4 c, in vec2 f){ c = vec4(u_a, u_b, 0, 1); }"
    fs = wrap(code, extra=("u_a", "u_b"))
    assert fs.count("uniform float u_a;") == 1 and "uniform float u_b;" in fs
    assert fs.startswith("#version 330")
    with pytest.raises(ValueError):
        wrap("void main(){}")


def test_fragcoord_runs_up_as_on_shadertoy():
    toy = _toy("void mainImage(out vec4 c, in vec2 f){ vec2 uv = f / iResolution.xy; c = vec4(uv, 0.0, 1.0); }")
    out = toy.render(0.0)
    assert out[0, 0, 1] > 240 and out[-1, 0, 1] < 15          # green is uv.y: the top row is y = 1
    assert out[0, 0, 0] < 15 and out[0, -1, 0] > 240          # red is uv.x


def test_a_channel_comes_in_upright():
    img = np.zeros((H, W, 3), np.uint8)
    img[:H // 3, :, 0] = 255                                   # red on top
    img[:, :W // 4, 2] = 255                                   # blue on the left
    toy = _toy("void mainImage(out vec4 c, in vec2 f){ c = texture(iChannel0, f / iResolution.xy); }",
               channels={"image": {0: "frame"}})
    out = toy.render(0.0, frame=img)
    assert np.abs(out.astype(int) - img).max() <= 2


def test_time_and_own_uniforms_reach_the_shader():
    toy = _toy("void mainImage(out vec4 c, in vec2 f){ c = vec4(fract(iTime), u_k, float(iFrame) / 10.0, 1.0); }",
               extra=("u_k",))
    out = toy.render(2.5, u_k=0.25)
    out = toy.render(2.5, u_k=0.25)
    r, g, b = (int(v) for v in out[5, 5])
    assert abs(r - 128) <= 2 and abs(g - 64) <= 2 and abs(b - 26) <= 2


def test_the_example_file_and_its_header(tmp_path):
    shader = load("shaders/shadertoy/arc_lente_kick.glsl")
    assert shader["meta"]["license"] == "MIT" and "mainImage" in shader["image"]
    assert shader["channels"]["image"] == {0: "frame"}
    _toy(shader["image"], extra=("u_kick", "u_bass"), channels=shader["channels"])   # it compiles


def test_layer_is_a_post_op_on_frame_and_the_kick_rushes_its_clock(tmp_path):
    _toy("void mainImage(out vec4 c, in vec2 f){ c = vec4(0); }")
    shader = tmp_path / "clock.glsl"
    shader.write_text("// license: MIT\nvoid mainImage(out vec4 c, in vec2 f){"
                      " c = vec4(texture(iChannel0, f / iResolution.xy).rgb * 0.5, 1.0) + vec4(0, fract(iTime * 0.1), u_k, 0); }")
    notes = [MidiNote(time=1.0, pitch=36, velocity=100, channel=0, duration=0.1, track="kick", track_index=1)]
    spec = {"source": "shadertoy", "file": str(shader), "channel0": "frame",
            "rush": {"hits": {"track": "kick"}, "amount": 4.0, "envelope": 0.5},
            "uniforms": {"u_k": {"hits": {"track": "kick"}, "envelope": 0.5}}}
    layer = ShadertoyLayer(spec, notes, W, H, 30)
    assert layer.is_post
    comp = build_compositor(None, {"layers": [{"source": "solid", "color": [200, 100, 0]}, spec]},
                            notes, W, H, fps=30)
    for f in range(0, 30):
        out = comp.frame_at(f / 30)
    assert abs(int(out[5, 5, 0]) - 100) <= 2                  # the frame below went in (halved)
    clock0 = None
    for f in range(0, 60):
        layer._draw(np.zeros((H, W, 3), np.uint8), f / 30)
        if f == 29:
            clock0 = layer._clock
    assert layer._clock - clock0 > 1.0 + 0.5                   # a second of song, and the kick rushed it
    with pytest.raises(ValueError):
        ShadertoyLayer({"source": "shadertoy"}, notes, W, H, 30)
    with pytest.raises(ValueError):
        ShadertoyLayer({**spec, "uniforms": {"u_k": [1, 2]}}, notes, W, H, 30)


def test_a_buffer_keeps_floats_and_feeds_back_into_itself():
    # Buffer A counts frames in its own last value and keeps a negative number; Image shows both
    buf = ("void mainImage(out vec4 c, in vec2 f){"
           " vec4 last = texture(iChannel0, f / iResolution.xy);"
           " c = vec4(last.r + 1.0, -2.5, 0.0, 1.0); }")
    img = ("void mainImage(out vec4 c, in vec2 f){ vec4 b = texture(iChannel0, f / iResolution.xy);"
           " c = vec4(b.r / 10.0, -b.g / 10.0, 0.0, 1.0); }")
    toy = _toy(img, buffers={"a": buf}, channels={"a": {0: "buffer_a"}, "image": {0: "buffer_a"}})
    for _ in range(4):
        out = toy.render(0.0)
    r, g, _b = (int(v) for v in out[3, 3])
    assert abs(r - 102) <= 2                                   # 4 frames counted: 0.4
    assert abs(g - 64) <= 2                                    # -2.5 survived: 0.25
    with pytest.raises(ValueError):
        _toy(img, channels={"image": {0: "buffer_b"}})         # no such buffer


def test_the_two_pass_shadows_run_from_their_folder():
    shader = load("shaders/shadertoy/sombras_2d")
    assert shader["channels"]["image"] == {0: "buffer_a"} and "a" in shader
    from core.shadertoy import BUFFERS
    toy = _toy(shader["image"], buffers={b: shader.get(b) for b in BUFFERS}, channels=shader["channels"])
    out = toy.render(1.0)
    out = toy.render(1.0 + 1 / 30)
    assert out.mean() > 10 and out.max() > 150                 # lit, with bright spots near the lights


def _knob(how, notes, grid=None):
    from core.video.layers import _music_knob
    return _music_knob("u", how, notes, grid=grid)


def test_count_restarts_every_bar():
    from core.rhythm.grid import RhythmGrid
    grid = RhythmGrid.from_beats(np.arange(0, 8.0, 0.5), time_signature=(4, 4))   # bars at 0, 2, 4, 6 s
    kicks = [MidiNote(time=t, pitch=36, velocity=100, channel=0, duration=0.1, track="kick", track_index=3)
             for t in (0.0, 0.5, 1.0, 2.5, 2.9)]
    count = _knob({"count": {"track": "kick"}, "per": "bar"}, kicks, grid)
    assert [count(t) for t in (0.1, 0.6, 1.9, 2.1, 2.6, 3.0, 4.2)] == [1, 2, 3, 0, 1, 2, 0]
    with pytest.raises(ValueError):
        _knob({"count": {"track": "kick"}}, kicks, None)       # needs the bars


def test_pitch_glides_and_rate_keeps_turning():
    bass = [MidiNote(time=1.0, pitch=48, velocity=100, channel=0, duration=0.1, track="bass", track_index=6),
            MidiNote(time=2.0, pitch=53, velocity=100, channel=0, duration=0.1, track="bass", track_index=6)]
    pitch = _knob({"pitch": {"track": "bass"}, "low": 48, "high": 53, "glide": 0.1}, bass)
    vals = [pitch(t) for t in np.arange(1.0, 3.0, 1 / 30)]
    assert vals[0] == 0.0 and 0.0 < vals[31] < 1.0 and vals[-1] > 0.99   # low, sliding, then high
    turn = _knob({"rate": 2.0, "base": 0.5, "from": {"hits": {"track": "bass"}, "envelope": 0.5}}, bass)
    angles = [turn(t) for t in np.arange(0.0, 3.0, 1 / 30)]
    calm = angles[29] - angles[0]                              # before any note: base only
    busy = angles[59] - angles[30]                             # a note rang: faster
    assert abs(calm - 29 / 30 * 0.5) < 1e-6 and busy > calm + 0.3
    assert all(b >= a for a, b in zip(angles, angles[1:]))
    with pytest.raises(ValueError):
        _knob({"wobble": 1}, bass)


@pytest.mark.parametrize("folder, knobs", [
    ("sombras_2d_arc", ("u_kick", "u_snare", "u_turn", "u_orbit", "u_hat", "u_acid", "u_acid_y", "u_bass",
                        "u_bass_y", "u_spin")),
    ("sombras_2d_kicks", ("u_count", "u_flash"))])
def test_the_arc_copies_compile(folder, knobs):
    from core.shadertoy import BUFFERS
    shader = load(f"shaders/shadertoy/{folder}")
    toy = _toy(shader["image"], buffers={b: shader.get(b) for b in BUFFERS}, channels=shader["channels"],
               extra=knobs)
    out = toy.render(1.0, **{k: 0.5 for k in knobs})
    assert out.max() > 30


def test_octagrams_and_its_shapes():
    orig = _toy(load("shaders/shadertoy/octagrams.glsl")["image"])
    arc = _toy(load("shaders/shadertoy/octagrams_arc.glsl")["image"], extra=("u_shape",))
    a, square = orig.render(6.0), arc.render(6.0, u_shape=0.0)
    assert np.abs(a.astype(int) - square).mean() < 0.5           # shape 0 is the original's square
    circle, star = arc.render(6.0, u_shape=1.0), arc.render(6.0, u_shape=5.0)
    assert np.abs(circle.astype(int) - square).mean() > 5 and np.abs(star.astype(int) - circle).mean() > 5


def test_bar_steps_up_each_bar_with_a_morph():
    from core.rhythm.grid import RhythmGrid
    grid = RhythmGrid.from_beats(np.arange(0, 8.0, 0.5), time_signature=(4, 4))   # bars at 0, 2, 4, 6 s
    bar = _knob({"bar": {"step": 1, "morph": 0.5}}, [], grid)
    assert bar(0.0) == -1.0 and bar(0.6) == 0.0                 # bar 0: melts from -1 into 0
    assert 0.0 < bar(2.25) < 1.0 and bar(2.9) == 1.0 and bar(4.6) == 2.0


def test_harmony_glides_the_short_way_round():
    from core.video.layers import VeilsLayer
    chords = [MidiNote(time=0.0, pitch=5, velocity=100, channel=0, duration=2.0, track="chords", track_index=0),
              MidiNote(time=2.0, pitch=6, velocity=100, channel=0, duration=2.0, track="chords", track_index=0)]
    import core.video.layers as L
    orig = L._layer_events
    L._layer_events = lambda spec, notes, *a, **k: chords if "score" in (spec or {}) else orig(spec, notes, *a, **k)
    try:
        hue = _knob({"harmony": {"score": "x"}, "glide": 0.3}, [])
        vals = [hue(t) for t in np.arange(0.0, 4.0, 1 / 30)]
    finally:
        L._layer_events = orig
    f, fs = VeilsLayer.hue_of_root(5), VeilsLayer.hue_of_root(6)
    assert abs(vals[30] - f) < 1e-6 and abs(vals[-1] - fs) < 0.01
    gap = lambda a, b: abs((a - b + 0.5) % 1.0 - 0.5)
    assert all(gap(a, b) < 0.1 for a, b in zip(vals, vals[1:]))   # it slides, never jumps


def test_kaleidoscope_runs():
    out = _toy(load("shaders/shadertoy/kaleidoscope.glsl")["image"]).render(3.0)
    assert 20 < out.mean() < 250


def test_since_counts_from_each_of_the_latest_hits():
    kicks = [MidiNote(time=t, pitch=36, velocity=100, channel=0, duration=0.1, track="kick", track_index=3)
             for t in (1.0, 2.0)]
    last = _knob({"since": {"track": "kick"}}, kicks)
    before = _knob({"since": {"track": "kick"}, "nth": 1}, kicks)
    assert last(0.5) == 1000.0 and abs(last(1.25) - 0.25) < 1e-9 and abs(last(2.5) - 0.5) < 1e-9
    assert before(1.5) == 1000.0 and abs(before(2.5) - 1.5) < 1e-9


def test_kaleidoscope_arc_rings_and_paint():
    toy = _toy(load("shaders/shadertoy/kaleidoscope_arc.glsl")["image"], extra=("u_ring0", "u_hue"))
    orig = _toy(load("shaders/shadertoy/kaleidoscope.glsl")["image"])
    plain = toy.render(3.0)
    assert np.abs(plain.astype(int) - orig.render(3.0)).mean() < 0.5     # without the knobs: the original
    ring = toy.render(3.0, u_ring0=0.2, u_hue=0.0)
    calm = toy.render(3.0, u_ring0=100.0, u_hue=0.0)
    assert ring.mean() > calm.mean() + 2                                  # a fresh kick's ring lights it up
    r, g, b = (float(c) for c in calm.reshape(-1, 3).mean(0))
    assert r > g and r > b                                                # painted red (hue 0)


def test_sound_reaches_the_shader_as_on_shadertoy():
    from core.shadertoy import SOUND_TOP_HZ, sound_texture
    spectrum = np.linspace(0.0, 1.0, 1025)                     # 0 .. 22050 Hz, rising
    wave = np.full(512, 0.5)
    snd = sound_texture(spectrum, 22050.0, wave)
    assert snd.shape == (2, 512) and snd.dtype == np.uint8
    toy = _toy("void mainImage(out vec4 c, in vec2 f){ float x = f.x / iResolution.x;"
               " c = vec4(texture(iChannel0, vec2(x, 0.0)).r, texture(iChannel0, vec2(x, 1.0)).r, 0.0, 1.0); }",
               channels={"image": {0: "sound"}})
    out = toy.render(0.0, sound=snd)
    mid = out[H // 2, W // 2]                                  # x = 0.5: 5.5 kHz, a quarter of the way up
    assert abs(int(mid[0]) - int(255 * (SOUND_TOP_HZ / 2) / 22050)) <= 4    # row 0: the spectrum
    assert abs(int(mid[1]) - int(255 * 0.75)) <= 3                          # row 1: the waveform (0.5 -> 0.75)
    silent = toy.render(0.0)                                    # no sound given: silence, not garbage
    assert silent[..., 0].max() == 0


def test_starleidoscope_hears_the_music():
    from core.shadertoy import sound_texture
    shader = load("shaders/shadertoy/starleidoscope.glsl")
    assert shader["channels"]["image"] == {0: "sound"}
    toy = _toy(shader["image"], channels=shader["channels"])
    quiet = toy.render(5.0, sound=sound_texture(np.zeros(1025), 22050.0))
    loud = toy.render(5.0, sound=sound_texture(np.full(1025, 0.8), 22050.0))
    assert quiet.mean() > 3 and np.abs(quiet.astype(int) - loud).mean() > 1   # the music changes its colours


def test_a_buffer_read_with_mipmap_can_be_blurred():
    # Buffer A: a checkerboard of single pixels; read with the mipmap filter at a high level it is flat grey
    buf = "void mainImage(out vec4 c, in vec2 f){ c = vec4(mod(floor(f.x) + floor(f.y), 2.0)); }"
    img = "void mainImage(out vec4 c, in vec2 f){ c = textureLod(iChannel0, f / iResolution.xy, 4.0); }"
    sharp = _toy(img, buffers={"a": buf}, channels={"image": {0: "buffer_a"}}).render(0.0)
    soft = _toy(img, buffers={"a": buf}, channels={"image": {0: "buffer_a mipmap"}}).render(0.0)
    assert sharp.std() > 60                                     # no mipmaps: still the checkerboard
    assert soft.std() < 10 and abs(soft.mean() - 128) < 10     # mipmaps: its average


def test_diamond_dust_runs_with_its_credit():
    shader = load("shaders/shadertoy/diamond_dust")
    assert "Espeset" in shader["meta"]["author"] and shader["channels"]["image"] == {0: "buffer_c mipmap"}
    from core.shadertoy import BUFFERS
    toy = _toy(shader["image"], buffers={b: shader.get(b) for b in BUFFERS}, common=shader.get("common", ""),
               channels=shader["channels"])
    toy.render(2.0)
    out = toy.render(2.0 + 1 / 30)
    assert out.mean() > 15 and out.max() > 150                 # the golden dust, with bright sparkles


def test_feature_knob_scales_and_uniform_lists_are_seen():
    from core.video.layers import _music_knob
    knob = _music_knob("u", {"feature": "texture.noisiness", "gain": 0.3, "bias": -0.07}, [],
                       features_at=lambda t: {"texture": {"noisiness": 0.5}})
    assert abs(knob(0.0) - 0.08) < 1e-9
    fs = wrap("uniform float u_a, u_b;\nvoid mainImage(out vec4 c, in vec2 f){ c = vec4(u_a + u_b); }",
              extra=("u_a", "u_b", "u_c"))
    assert fs.count("u_b") == 2 and "uniform float u_c;" in fs   # u_b is in the list: not declared twice


def test_diamond_dust_arc_is_the_original_at_rest_and_moves_with_its_knobs():
    from core.shadertoy import BUFFERS
    knobs = ("u_warp", "u_rough", "u_flow", "u_breath", "u_ripple", "u_trem_phase", "u_glint", "u_spin")

    def make(folder, extra=()):
        sh = load(f"shaders/shadertoy/{folder}")
        return _toy(sh["image"], buffers={b: sh.get(b) for b in BUFFERS}, common=sh.get("common", ""),
                    channels=sh["channels"], extra=extra)
    orig, arc = make("diamond_dust"), make("diamond_dust_arc", knobs)
    orig.render(2.0), arc.render(2.0)
    a, b = orig.render(2.0 + 1 / 30), arc.render(2.0 + 1 / 30)
    assert np.abs(a.astype(int) - b).mean() < 0.5
    arc.render(2.0, u_warp=1.5, u_flow=0.2)
    moved = arc.render(2.0 + 1 / 30, u_warp=1.5, u_flow=0.2)
    assert np.abs(moved.astype(int) - b).mean() > 3


def test_prism_liquid_runs():
    from core.shadertoy import BUFFERS
    shader = load("shaders/shadertoy/prism_liquid")
    assert shader["channels"]["image"] == {0: "buffer_a"}
    toy = _toy(shader["image"], buffers={b: shader.get(b) for b in BUFFERS}, channels=shader["channels"])
    toy.render(4.0)
    out = toy.render(4.0 + 1 / 30)
    assert 20 < out.mean() < 220 and out.std() > 10


def test_prism_liquid_arc_is_the_original_at_rest_and_splashes_on_a_kick():
    from core.shadertoy import BUFFERS

    def make(folder, extra=()):
        sh = load(f"shaders/shadertoy/{folder}")
        return _toy(sh["image"], buffers={b: sh.get(b) for b in BUFFERS}, channels=sh["channels"], extra=extra)
    orig, arc = make("prism_liquid"), make("prism_liquid_arc", ("u_kick_age", "u_hue", "u_facets"))
    orig.render(4.0), arc.render(4.0)
    a, b = orig.render(4.0 + 1 / 30), arc.render(4.0 + 1 / 30)
    assert np.abs(a.astype(int) - b).mean() < 0.5
    arc.render(4.0, u_kick_age=0.1)
    splashed = arc.render(4.0 + 1 / 30, u_kick_age=0.1)
    assert np.abs(splashed.astype(int) - b).mean() > 1
    arc.render(4.0, u_hue=0.0)
    red = arc.render(4.0 + 1 / 30, u_hue=0.0).reshape(-1, 3).mean(0)
    assert red[0] > red[1] and red[0] > red[2]                 # the chord's colour (hue 0: red)
