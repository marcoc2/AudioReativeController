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
