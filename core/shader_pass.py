"""A full-screen fragment shader rendered on the GPU (moderngl), supersampled.

``ShaderPass(fs, w, h, supersample).draw(**uniforms)`` -> uint8 H x W x 3. The
shader gets ``v_uv`` (0..1) and writes ``f_color``; ``u_aspect`` is set for it
when declared, and uniforms it does not use are skipped (the compiler drops them).
"""
from __future__ import annotations

import numpy as np

VERTEX_SHADER = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

_DOWN_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2D u_tex; uniform int u_ss;
void main(){
    ivec2 base = ivec2(gl_FragCoord.xy) * u_ss; vec3 acc = vec3(0.0);
    for (int j = 0; j < u_ss; j++) for (int i = 0; i < u_ss; i++) acc += texelFetch(u_tex, base + ivec2(i, j), 0).rgb;
    f_color = vec4(acc / float(u_ss * u_ss), 1.0);
}
"""


class ShaderPass:
    def __init__(self, fragment_shader: str, width: int, height: int, supersample: int = 2):
        import moderngl
        self.W, self.H = int(width), int(height)
        self.ss = max(1, int(supersample))
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=fragment_shader)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._vbo = self.ctx.buffer(quad.tobytes())
        self._vao = self.ctx.vertex_array(self.prog, [(self._vbo, "2f", "in_pos")])
        # supersample into a texture, average it down on the GPU (a numpy mean-pool of the
        # big image costs ~120 ms at 720p; this costs ~1 ms)
        self._tex = self.ctx.texture((self.W * self.ss, self.H * self.ss), 3)
        self._fbo = self.ctx.framebuffer([self._tex])
        self._down = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=_DOWN_FS)
        self._down_vao = self.ctx.vertex_array(self._down, [(self._vbo, "2f", "in_pos")])
        self._out = self.ctx.simple_framebuffer((self.W, self.H), 3)
        self._mgl = moderngl

    def draw(self, textures=None, **uniforms) -> np.ndarray:
        """Render; ``textures`` maps texture units (1, 2, ...) to textures of this context."""
        # several GPU layers each own a context: make this one current, or its calls land in another's
        with self.ctx:
            u = self.prog
            uniforms.setdefault("u_aspect", self.W / self.H)
            for name, value in uniforms.items():
                if name in u:
                    u[name].value = value
            for unit, tex in (textures or {}).items():
                tex.use(unit)
            self._fbo.use()
            self._fbo.clear(0.0, 0.0, 0.0)
            self._vao.render(self._mgl.TRIANGLE_STRIP)
            self._out.use()
            self._tex.use(0)
            self._down["u_tex"].value = 0
            self._down["u_ss"].value = self.ss
            self._down_vao.render(self._mgl.TRIANGLE_STRIP)
            img = np.frombuffer(self._out.read(components=3), dtype=np.uint8).reshape(self.H, self.W, 3)[::-1]
        return img.copy()
