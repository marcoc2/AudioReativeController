"""core/gpu.py — one shared GL context per thread."""
import threading

import numpy as np
import pytest

from core import gpu


def _ctx_or_skip():
    try:
        return gpu.context()
    except Exception as exc:                 # no GL on this machine
        pytest.skip(f"no OpenGL context: {exc}")


def test_layers_of_a_thread_share_one_context():
    from core.shader_pass import ShaderPass
    ctx = _ctx_or_skip()
    fs = "#version 330\nin vec2 v_uv; out vec4 f_color; uniform float u_v;\nvoid main(){ f_color = vec4(u_v); }"
    a, b = ShaderPass(fs, 8, 8, 1), ShaderPass(fs, 8, 8, 1)
    assert a.ctx is b.ctx is ctx
    assert a.draw(u_v=1.0).min() == 255 and b.draw(u_v=0.0).max() == 0     # they do not step on each other


def test_each_thread_its_own_and_release_frees_it():
    main = _ctx_or_skip()
    seen = {}

    def work():
        from core.shader_pass import ShaderPass
        fs = "#version 330\nin vec2 v_uv; out vec4 f_color;\nvoid main(){ f_color = vec4(0.5); }"
        p = ShaderPass(fs, 4, 4, 1)
        seen["ctx"], seen["img"] = p.ctx, p.draw()
        gpu.release()
        seen["after"] = getattr(gpu._local, "ctx", None)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert seen["ctx"] is not main and seen["after"] is None
    assert np.all(np.abs(seen["img"].astype(int) - 128) <= 1)
    assert gpu.context() is main                                            # this thread's is untouched
