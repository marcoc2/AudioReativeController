"""Shadertoy — run a shader written for shadertoy.com on the GPU, as it is. GPU.

A Shadertoy shader is a ``mainImage(out vec4 fragColor, in vec2 fragCoord)`` that
reads Shadertoy's inputs; this wraps it in a full-screen pass (``core/shader_pass``)
and feeds them:

    iResolution     the size rendered (supersampled), z = 1
    iTime           the shader's clock: the layer can run it faster or slower with the
                    music (it need not be the song's time)
    iTimeDelta, iFrame, iFrameRate, iDate (a fixed date: same film every render),
    iSampleRate, iMouse (never pressed)
    iChannel0..3    2D textures, each one of:
                        "frame"      the picture given to ``render`` (the clip below)
                        "previous"   this shader's own last output
                        "buffer_a".."buffer_d"   a buffer pass (see below)
                        an image     (uint8 H x W x 3), fixed
                    uploaded bottom row first as Shadertoy does; iChannelResolution their sizes

plus any ``float`` uniforms of one's own (``extra``), declared for the shader when it
does not declare them itself, so music can drive knobs the shader exposes.

Passes: as on Shadertoy, ``Buffer A..D`` run first, in that order, each into a float
texture as big as the picture (RGBA 32-bit, all four channels kept, negative values
too), then ``Image``. A pass reading a buffer that has already run this frame gets
this frame's; reading itself, or one that runs later, it gets the last frame's (so a
buffer can feed back into itself). ``Common`` code is put above every pass. Cube maps,
sound and keyboard inputs are not supported.

Files live in ``shaders/shadertoy/`` (see its README): one ``.glsl`` for a one-pass
shader, or a folder with ``image.glsl``, ``buffer_a.glsl`` .. and ``common.glsl``. Each
file starts with ``// key: value`` lines (url, author, license, and ``iChannel0: ...``
for what its channels read); ``load(path)`` reads a file or a folder.
"""
from __future__ import annotations

import re
import time as _time
from pathlib import Path

import numpy as np

from core.shader_pass import ShaderPass

BUFFERS = ("a", "b", "c", "d")
CHANNELS = 4

_HEADER = """#version 330
in vec2 v_uv; out vec4 f_color;
uniform vec3 iResolution;
uniform float iTime, iTimeDelta, iFrameRate, iSampleRate;
uniform int iFrame;
uniform vec4 iMouse, iDate;
uniform vec3 iChannelResolution[4];
uniform float iChannelTime[4];
uniform sampler2D iChannel0, iChannel1, iChannel2, iChannel3;
"""

_IMAGE_MAIN = """
void main(){
    vec4 c = vec4(0.0, 0.0, 0.0, 1.0);
    mainImage(c, gl_FragCoord.xy);
    f_color = vec4(c.rgb, 1.0);
}
"""

_BUFFER_MAIN = """
void main(){
    vec4 c = vec4(0.0);
    mainImage(c, gl_FragCoord.xy);
    f_color = c;
}
"""

_META = re.compile(r"^\s*//\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.+?)\s*$")


def _header(text: str) -> dict:
    meta = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _META.match(line)
        if not m:
            break
        meta[m.group(1).lower()] = m.group(2)
    return meta


def _channels_of(meta: dict) -> dict:
    """{0: "buffer_a", ...} from ``// iChannel0: buffer_a`` header lines."""
    out = {}
    for i in range(CHANNELS):
        v = meta.get(f"ichannel{i}")
        if v:
            out[i] = v.split()[0].strip().lower()
    return out


def load(path) -> dict:
    """A shader file or folder: {"image": code, "a".."d": code, "common": code, "meta": header of
    the image, "channels": {pass: {i: source}} from the headers}."""
    path = Path(path)
    if path.is_dir():
        files = {"image": path / "image.glsl", "common": path / "common.glsl",
                 **{b: path / f"buffer_{b}.glsl" for b in BUFFERS}}
        if not files["image"].exists():
            raise ValueError(f"shadertoy: {path} has no image.glsl")
    else:
        files = {"image": path}
    out = {"channels": {}}
    for name, f in files.items():
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        out[name] = text
        if name != "common":
            out["channels"][name] = _channels_of(_header(text))
    out["meta"] = _header(out["image"])
    return out


def wrap(code: str, extra=(), common: str = "", buffer: bool = False) -> str:
    """The fragment shader for one pass: Shadertoy's inputs declared above it, a ``main`` below."""
    if "mainImage" not in code:
        raise ValueError("shadertoy: no mainImage() in the shader")
    strip = lambda s: "\n".join(line for line in s.splitlines() if not line.lstrip().startswith("#version"))
    body = strip(common) + "\n" + strip(code) if common else strip(code)
    own = "".join(f"uniform float {name};\n" for name in extra
                  if not re.search(rf"\buniform\s+\w+\s+{re.escape(name)}\b", body))
    return _HEADER + own + "#line 1\n" + body + (_BUFFER_MAIN if buffer else _IMAGE_MAIN)


def _set(prog, values: dict) -> None:
    for name, value in values.items():
        if name in prog:
            prog[name].value = value


class Shadertoy:
    CHANNELS = CHANNELS

    def __init__(self, image: str, width: int, height: int, supersample: int = 1, extra=(),
                 buffers=None, common: str = "", channels=None):
        """``image`` the Image pass; ``buffers`` {"a": code, ...}; ``channels`` {"image" | "a".. :
        {0..3: "frame" | "previous" | "buffer_a".. | uint8 image}}."""
        self.W, self.H = int(width), int(height)
        self.extra = tuple(extra)
        self.buffers = {b: code for b, code in (buffers or {}).items() if code}
        for b in self.buffers:
            if b not in BUFFERS:
                raise ValueError(f"shadertoy: unknown buffer {b!r} (use one of {BUFFERS})")
        self.channels = {k: dict(v) for k, v in (channels or {}).items()}
        for pass_, chans in self.channels.items():
            for i, src in chans.items():
                if isinstance(src, str) and src.startswith("buffer_") and src[7:] not in self.buffers:
                    raise ValueError(f"shadertoy: {pass_} reads {src} but there is no such buffer")
        self._pass = ShaderPass(wrap(image, self.extra, common), width, height, supersample=supersample)
        ss = self._pass.ss
        self.RW, self.RH = self.W * ss, self.H * ss
        ctx, mgl = self._pass.ctx, self._pass._mgl
        self._progs, self._tex_buf, self._fbo_buf, self._cur = {}, {}, {}, {}
        with ctx:
            for b, code in self.buffers.items():
                self._progs[b] = self._pass.quad_program(wrap(code, self.extra, common, buffer=True))
                zero = np.zeros((self.RH, self.RW, 4), np.float32).tobytes()
                texs = [ctx.texture((self.RW, self.RH), 4, data=zero, dtype="f4") for _ in range(2)]
                for t in texs:
                    t.filter = (mgl.LINEAR, mgl.LINEAR)
                    t.repeat_x = t.repeat_y = False
                self._tex_buf[b] = texs
                self._fbo_buf[b] = [ctx.framebuffer([t]) for t in texs]
                self._cur[b] = 0
        self._tex: dict = {}
        self._previous = np.zeros((self.H, self.W, 3), np.uint8)
        self._frame = 0

    def reads(self, source: str) -> bool:
        """Whether any pass reads ``source`` ("frame", "previous", ...)."""
        return any(isinstance(s, str) and s == source for chans in self.channels.values() for s in chans.values())

    def _upload(self, key, img: np.ndarray):
        img = np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[::-1])     # bottom row first
        h, w = img.shape[:2]
        comps = img.shape[2] if img.ndim == 3 else 1
        ctx, mgl = self._pass.ctx, self._pass._mgl
        tex = self._tex.get(key)
        if tex is None or tex.size != (w, h) or tex.components != comps:
            if tex is not None:
                tex.release()
            tex = ctx.texture((w, h), comps)
            tex.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
            tex.repeat_x = tex.repeat_y = True
            self._tex[key] = tex
        tex.write(img.tobytes())
        tex.build_mipmaps()
        return tex

    def _inputs(self, pass_: str, frame):
        """The textures a pass reads, bound to units 1..4, and their sizes."""
        textures, res = {}, [(0.0, 0.0, 1.0)] * CHANNELS
        for i, src in self.channels.get(pass_, {}).items():
            if isinstance(src, str) and src.startswith("buffer_"):
                b = src[7:]
                tex = self._tex_buf[b][self._cur[b]]     # this frame's if it has run, else the last one
                size = (self.RW, self.RH)
            else:
                img = frame if src == "frame" else self._previous if src == "previous" else src
                if img is None or isinstance(img, str):
                    continue
                tex = self._upload((pass_, i), img)
                size = (img.shape[1], img.shape[0])
            textures[i + 1] = tex
            res[i] = (float(size[0]), float(size[1]), 1.0)
        return textures, res

    def render(self, time: float, dt: float = 1.0 / 30, frame=None, **uniforms) -> np.ndarray:
        """Draw at shader time ``time``; ``frame`` the picture for "frame" channels; ``uniforms``
        the extra knobs."""
        now = _time.localtime(0)
        values = dict(iResolution=(float(self.RW), float(self.RH), 1.0), iTime=float(time),
                      iTimeDelta=float(dt), iFrameRate=1.0 / max(1e-6, float(dt)), iFrame=int(self._frame),
                      iSampleRate=44100.0, iMouse=(0.0, 0.0, 0.0, 0.0),
                      iDate=(float(now.tm_year), float(now.tm_mon - 1), float(now.tm_mday), float(time)),
                      iChannelTime=[float(time)] * CHANNELS,
                      **{f"iChannel{i}": i + 1 for i in range(CHANNELS)},
                      **{name: float(v) for name, v in uniforms.items()})
        with self._pass.ctx:
            for b in BUFFERS:
                if b not in self.buffers:
                    continue
                prog, vao = self._progs[b]
                textures, res = self._inputs(b, frame)
                _set(prog, {**values, "iChannelResolution": res})
                for unit, tex in textures.items():
                    tex.use(unit)
                nxt = 1 - self._cur[b]
                self._fbo_buf[b][nxt].use()
                vao.render(self._pass._mgl.TRIANGLE_STRIP)
                self._cur[b] = nxt
            textures, res = self._inputs("image", frame)
        out = self._pass.draw(textures=textures, **values, iChannelResolution=res)
        self._previous = out
        self._frame += 1
        return out
