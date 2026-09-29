"""The compositor's GPU side — blending and mattes on the GPU, frames moved only when asked.

A ``Frame`` is the picture being built, held on the CPU (numpy), on the GPU (a texture)
or both: ``.cpu()`` / ``.tex()`` move it the first time the other side is asked for and
remember it. Layers that work in numpy keep doing so; the compositor blends and applies
mattes here, so a stack of layers pays one download at the end instead of a numpy blend
per layer.

The passes reproduce the numpy compositor (``blend_frames``, ``apply_matte``,
``LayerMatte``) — same maths, same rounding, OpenCV's own elliptic kernel and Gaussian
weights — so a scene looks the same either way (within a level or two).

Textures keep numpy's row order (row 0 written first, read back first) and every pass
addresses texels by ``gl_FragCoord``, so nothing is ever flipped.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import cv2
import numpy as np

from core.shader_pass import VERTEX_SHADER

MODES = {"normal": 0, "add": 1, "screen": 2, "multiply": 3}
MAX_TAPS = 4096                  # elliptic kernel offsets (grow up to ~35 px)
MAX_WEIGHTS = 128                # half a Gaussian kernel (feather sigma up to ~31 px)

_BLEND_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_below, u_top;
uniform int u_mode; uniform float u_op;
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    // back to the exact integers numpy has (255 reads as 254.99998, and floor() would keep 254)
    vec3 a = floor(texelFetch(u_below, p, 0).rgb * 255.0 + 0.5);
    vec3 t = floor(texelFetch(u_top, p, 0).rgb * 255.0 + 0.5);
    float k = min(1.0, u_op);
    vec3 b = t * k;
    vec3 o;
    if (u_mode == 1)      o = a + b;
    else if (u_mode == 2) o = 255.0 - (255.0 - a) * (255.0 - b) / 255.0;
    else if (u_mode == 3) o = a * ((1.0 - k) + k * t / 255.0);
    else                  o = a * (1.0 - k) + t * k;
    // numpy's astype(uint8); the GPU divides by multiplying with an inexact reciprocal, so a
    // result numpy gets as exactly 200 can land at 199.99998 here: nudge it over first
    f_color = vec4(floor(clamp(o, 0.0, 255.0) + 1e-3) / 255.0, 1.0);
}
"""

_APPLY_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_below, u_above, u_matte;
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    float m = texelFetch(u_matte, p, 0).r;
    vec3 a = floor(texelFetch(u_below, p, 0).rgb * 255.0 + 0.5);
    vec3 b = floor(texelFetch(u_above, p, 0).rgb * 255.0 + 0.5);
    vec3 o = a * (1.0 - m) + b * m;
    f_color = vec4(floor(o + 0.5) / 255.0, 1.0);
}
"""

# the selection (track size) stretched to the frame the way cv2.resize INTER_LINEAR does it
# (weights in float: the texture unit's own bilinear filter keeps only 8 bits of them), then
# grown (max) or shrunk (min) over OpenCV's elliptic kernel; out-of-frame taps are left out,
# as cv2.dilate/erode do
_GROW_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_sel;
uniform sampler2D u_offs;          // n x 1, the kernel's (dx, dy)
uniform vec2 u_size;
uniform int u_n; uniform int u_min;
float at(vec2 q){
    ivec2 n = textureSize(u_sel, 0);
    vec2 s = clamp((q + 0.5) * vec2(n) / u_size - 0.5, vec2(0.0), vec2(n - 1));
    ivec2 i0 = ivec2(floor(s)), i1 = min(i0 + 1, n - 1);
    vec2 f = s - vec2(i0);
    float a = texelFetch(u_sel, i0, 0).r, b = texelFetch(u_sel, ivec2(i1.x, i0.y), 0).r;
    float c = texelFetch(u_sel, ivec2(i0.x, i1.y), 0).r, d = texelFetch(u_sel, i1, 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
void main(){
    vec2 p = floor(gl_FragCoord.xy);
    float v = at(p);
    for (int i = 0; i < u_n; i++){
        vec2 q = p + texelFetch(u_offs, ivec2(i, 0), 0).rg;
        if (q.x < 0.0 || q.y < 0.0 || q.x >= u_size.x || q.y >= u_size.y) continue;
        float s = at(q);
        v = (u_min == 1) ? min(v, s) : max(v, s);
    }
    f_color = vec4(v, 0.0, 0.0, 1.0);
}
"""

# one direction of a separable Gaussian, BORDER_REFLECT_101 like cv2.GaussianBlur
_BLUR_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_src;
uniform ivec2 u_dir;
uniform int u_half;
uniform float u_w[%d];
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy), size = textureSize(u_src, 0);
    int n = (u_dir.x == 1) ? size.x : size.y;
    int c = (u_dir.x == 1) ? p.x : p.y;
    float acc = 0.0;
    for (int i = -u_half; i <= u_half; i++){
        int j = c + i;
        if (j < 0) j = -j;
        if (j >= n) j = 2 * n - 2 - j;
        ivec2 q = (u_dir.x == 1) ? ivec2(j, p.y) : ivec2(p.x, j);
        acc += texelFetch(u_src, q, 0).r * u_w[abs(i)];
    }
    f_color = vec4(acc, 0.0, 0.0, 1.0);
}
""" % MAX_WEIGHTS

_INVERT_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_src;
void main(){ f_color = vec4(1.0 - texelFetch(u_src, ivec2(gl_FragCoord.xy), 0).r, 0.0, 0.0, 1.0); }
"""


def ellipse_offsets(radius: int) -> np.ndarray:
    """The (dx, dy) of OpenCV's elliptic structuring element of that radius."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    ys, xs = np.nonzero(k)
    return np.stack([xs - radius, ys - radius], axis=1).astype(np.float32)


def gaussian_half(sigma: float) -> np.ndarray:
    """The centre-and-right half of the kernel cv2.GaussianBlur(float image, (0, 0), sigma) uses."""
    ksize = int(round(sigma * 8 + 1)) | 1
    k = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F).ravel()
    return k[ksize // 2:].astype(np.float32)


class GpuComposer:
    """Blend and matte passes at one frame size, in the thread's shared context."""

    def __init__(self, width: int, height: int):
        import moderngl
        from core.gpu import context
        self.mgl = moderngl
        self.ctx = ctx = context()
        self.W, self.H = int(width), int(height)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        with ctx:
            self._vbo = ctx.buffer(quad.tobytes())
            self._progs = {}
            for name, fs in (("blend", _BLEND_FS), ("apply", _APPLY_FS), ("grow", _GROW_FS),
                             ("blur", _BLUR_FS), ("invert", _INVERT_FS)):
                prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=fs)
                self._progs[name] = (prog, ctx.vertex_array(prog, [(self._vbo, "2f", "in_pos")]))
        self._pool: Dict[tuple, list] = {}
        self._used: Dict[tuple, int] = {}
        self._fbos: Dict[int, object] = {}
        self._sel: Dict[tuple, object] = {}
        self._kernels: Dict[tuple, np.ndarray] = {}
        self.uploads = self.downloads = 0
        self.moving = 0.0                    # seconds spent moving frames, for the profile

    # ---------------------------------------------------------------- textures
    def begin(self) -> None:
        """A new frame: the textures of the last one may be reused."""
        self._used = {}

    def _tex(self, comps: int = 3, dtype: str = "f1"):
        key = (comps, dtype)
        pool = self._pool.setdefault(key, [])
        n = self._used.get(key, 0)
        if n == len(pool):
            with self.ctx:
                pool.append(self.ctx.texture((self.W, self.H), comps, dtype=dtype))
        self._used[key] = n + 1
        return pool[n]

    def _fbo(self, tex):
        fbo = self._fbos.get(id(tex))
        if fbo is None:
            with self.ctx:
                fbo = self._fbos[id(tex)] = self.ctx.framebuffer([tex])
        return fbo

    def upload(self, img: np.ndarray):
        t0 = time.perf_counter()
        tex = self._tex()
        if img.shape[:2] != (self.H, self.W):          # numpy would broadcast it: so do we
            img = np.broadcast_to(img, (self.H, self.W, 3))
        with self.ctx:
            tex.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
        self.uploads += 1
        self.moving += time.perf_counter() - t0
        return tex

    def download(self, tex) -> np.ndarray:
        t0 = time.perf_counter()
        with self.ctx:
            buf = self._fbo(tex).read(components=3)
        self.downloads += 1
        self.moving += time.perf_counter() - t0
        # writable, like every frame the numpy layers have been given
        return np.frombuffer(bytearray(buf), np.uint8).reshape(self.H, self.W, 3)

    def _run(self, name: str, out, textures: dict, **uniforms):
        prog, vao = self._progs[name]
        with self.ctx:
            for unit, (uname, tex) in enumerate(textures.items()):
                tex.use(unit)
                prog[uname].value = unit
            for k, v in uniforms.items():
                if k in prog:
                    if isinstance(v, np.ndarray):
                        prog[k].write(v.tobytes())
                    else:
                        prog[k].value = v
            self._fbo(out).use()
            vao.render(self.mgl.TRIANGLE_STRIP)
        return out

    # ------------------------------------------------------------------ passes
    def blend(self, below, top, mode: str, opacity: float):
        """``blend_frames`` on the GPU: textures in, a texture out."""
        return self._run("blend", self._tex(), {"u_below": below, "u_top": top},
                         u_mode=MODES[mode], u_op=float(opacity))

    def apply(self, below, above, matte):
        """``apply_matte`` on the GPU (matte: an R32F texture of the frame size)."""
        return self._run("apply", self._tex(), {"u_below": below, "u_above": above, "u_matte": matte})

    def matte(self, selection: Optional[np.ndarray], grow: int, feather: float, outside: bool):
        """``LayerMatte`` on the GPU: the objects picked (uint8 0/1 at the track's size, or
        None: none) stretched, grown and feathered to the frame, inverted for ``outside``."""
        if selection is None:
            selection = np.zeros((1, 1), np.uint8)
        h, w = selection.shape
        sel = self._sel.get((w, h))
        with self.ctx:
            if sel is None:
                sel = self._sel[(w, h)] = self.ctx.texture((w, h), 1, dtype="f1")
            sel.write(np.ascontiguousarray(selection * 255 if selection.max() <= 1 else selection,
                                           dtype=np.uint8).tobytes())
        key = ("ellipse", abs(grow))
        if key not in self._kernels:
            offs = ellipse_offsets(abs(grow)) if grow else np.zeros((1, 2), np.float32)
            if len(offs) > MAX_TAPS:
                raise ValueError(f"mask: grow {grow} px is more than the GPU matte takes")
            with self.ctx:
                tex = self.ctx.texture((len(offs), 1), 2, data=offs.tobytes(), dtype="f4")
            self._kernels[key] = (tex, len(offs) if grow else 0)
        offs_tex, n = self._kernels[key]
        m = self._run("grow", self._tex(1, "f4"), {"u_sel": sel, "u_offs": offs_tex},
                      u_size=(float(self.W), float(self.H)), u_n=n, u_min=int(grow < 0))
        if feather > 0.25:
            key = ("gauss", round(float(feather), 6))
            if key not in self._kernels:
                self._kernels[key] = gaussian_half(feather)
            half = self._kernels[key]
            if len(half) > MAX_WEIGHTS:
                raise ValueError(f"mask: feather {feather:.1f} px is more than the GPU matte takes")
            w_arr = np.zeros(MAX_WEIGHTS, np.float32)
            w_arr[:len(half)] = half
            m = self._run("blur", self._tex(1, "f4"), {"u_src": m}, u_dir=(1, 0), u_half=len(half) - 1, u_w=w_arr)
            m = self._run("blur", self._tex(1, "f4"), {"u_src": m}, u_dir=(0, 1), u_half=len(half) - 1, u_w=w_arr)
        if outside:
            m = self._run("invert", self._tex(1, "f4"), {"u_src": m})
        return m


class Frame:
    """The picture being built: on the CPU, the GPU or both (each side made when asked)."""

    __slots__ = ("g", "_cpu", "_tex")

    def __init__(self, g: GpuComposer, cpu: Optional[np.ndarray] = None, tex=None):
        self.g, self._cpu, self._tex = g, cpu, tex

    def cpu(self) -> np.ndarray:
        if self._cpu is None:
            self._cpu = self.g.download(self._tex)
        return self._cpu

    def tex(self):
        if self._tex is None:
            self._tex = self.g.upload(self._cpu)
        return self._tex
