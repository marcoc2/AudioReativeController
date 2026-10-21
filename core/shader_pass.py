"""A full-screen fragment shader rendered on the GPU (moderngl), supersampled.

``ShaderPass(fs, w, h, supersample).draw(**uniforms)`` -> uint8 H x W x 3. The
shader gets ``v_uv`` (0..1) and writes ``f_color``; ``u_aspect`` is set for it
when declared, and uniforms it does not use are skipped (the compiler drops them).

``draw_into(texture, ...)`` renders the same picture into a texture of the compositor
(core/video/gpu_compose.py) instead of bringing it back to the CPU: rows in numpy's order
(top row first), which is upside down from GL's, so the averaging-down pass flips it.
``flip_copy(src, dst)`` turns such a texture back into GL's order (for a shader to read).
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

_DOWN_FLIP_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2D u_tex; uniform int u_ss;
void main(){
    ivec2 o = ivec2(gl_FragCoord.xy);
    int h = textureSize(u_tex, 0).y / u_ss;
    ivec2 base = ivec2(o.x, h - 1 - o.y) * u_ss; vec3 acc = vec3(0.0);
    for (int j = 0; j < u_ss; j++) for (int i = 0; i < u_ss; i++) acc += texelFetch(u_tex, base + ivec2(i, j), 0).rgb;
    f_color = vec4(acc / float(u_ss * u_ss), 1.0);
}
"""

_FLIP_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2D u_src;
void main(){
    ivec2 o = ivec2(gl_FragCoord.xy);
    f_color = vec4(texelFetch(u_src, ivec2(o.x, textureSize(u_src, 0).y - 1 - o.y), 0).rgb, 1.0);
}
"""


class ShaderPass:
    def __init__(self, fragment_shader: str, width: int, height: int, supersample: int = 2):
        import moderngl
        self.W, self.H = int(width), int(height)
        self.ss = max(1, int(supersample))
        from core.gpu import context
        self.ctx = context()                  # shared by every layer of the render (core/gpu.py)
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
        self._flip_down = self._flip = None           # made when first asked for
        self._fbos: dict = {}                         # framebuffers over textures drawn into

    def quad_program(self, fragment_shader: str):
        """Another full-screen program in this context, for passes of one's own: (program, vao)."""
        with self.ctx:
            prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=fragment_shader)
            return prog, self.ctx.vertex_array(prog, [(self._vbo, "2f", "in_pos")])

    def _render(self, textures, uniforms) -> None:
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

    def draw(self, textures=None, **uniforms) -> np.ndarray:
        """Render; ``textures`` maps texture units (1, 2, ...) to textures of this context."""
        # the shared context must be current in this thread for the calls below
        with self.ctx:
            self._render(textures, uniforms)
            self._out.use()
            self._tex.use(0)
            self._down["u_tex"].value = 0
            self._down["u_ss"].value = self.ss
            self._down_vao.render(self._mgl.TRIANGLE_STRIP)
            img = np.frombuffer(self._out.read(components=3), dtype=np.uint8).reshape(self.H, self.W, 3)[::-1]
        return img.copy()

    def _target(self, tex):
        fbo = self._fbos.get(id(tex))
        if fbo is None:
            fbo = self._fbos[id(tex)] = self.ctx.framebuffer([tex])
        return fbo

    def draw_into(self, target, textures=None, **uniforms):
        """Render into ``target`` (an RGB texture of W x H, rows in numpy's order); nothing
        comes back to the CPU. Returns ``target``."""
        if target.size != (self.W, self.H):
            raise ValueError(f"draw_into: target is {target.size}, the pass draws {(self.W, self.H)}")
        with self.ctx:
            if self._flip_down is None:
                self._flip_down = self.quad_program(_DOWN_FLIP_FS)
            self._render(textures, uniforms)
            prog, vao = self._flip_down
            self._target(target).use()
            self._tex.use(0)
            prog["u_tex"].value = 0
            prog["u_ss"].value = self.ss
            vao.render(self._mgl.TRIANGLE_STRIP)
        return target

    def flip_copy(self, src, dst):
        """Copy ``src`` into ``dst`` (same size) upside down: numpy's row order to GL's."""
        with self.ctx:
            if self._flip is None:
                self._flip = self.quad_program(_FLIP_FS)
            prog, vao = self._flip
            self._target(dst).use()
            src.use(0)
            prog["u_src"].value = 0
            vao.render(self._mgl.TRIANGLE_STRIP)
        return dst
