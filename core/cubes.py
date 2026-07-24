"""CubeField — GPU-rendered lattice of textured cubes for a camera fly-through.

Role in the image: this is a *stage*, not a generated world. It takes an
external RGB frame (a reactive clip screen) and paints it onto the faces of
cubes floating in an otherwise empty, solid-color space, then flies a camera
straight down the middle of them. The cubes are the scenography; the clip is
the content shown on them.

Loop-safety (EFFECTS_GUIDELINES §6): the cubes live on a lattice that is
periodic in depth. Each cube's distance ahead of the camera is
``d = (z0 - phase) mod 1`` — a function of ``phase in [0,1)`` only — so the
whole configuration at ``phase=0`` is identical to ``phase=1``: the fly-through
loops seamlessly over its musical period. Cubes are placed in an annulus around
the flight axis (the center stays empty, the camera flies through the gap), so
a cube that reaches the camera has already left the frustum sideways — the
depth wrap happens off-screen, no pop.

Determinism: all cube placement/rotation comes from ``seed`` via a single RNG.

Renders offscreen at the output resolution and reads pixels back (an RTX-class
GPU does the draw in well under a millisecond; readback dominates).
"""
from __future__ import annotations

import numpy as np

_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
in vec2 in_uv;
in float in_isclip;
in float in_layer;
out vec2 v_uv;
out float v_isclip;
out float v_layer;
out float v_camz;
void main(){
    v_uv = in_uv;
    v_isclip = in_isclip;
    v_layer = in_layer;
    vec4 clip = u_mvp * vec4(in_pos, 1.0);
    v_camz = clip.w;            // = distance ahead of the camera (see _perspective)
    gl_Position = clip;
}
"""

_FS = """
#version 330
uniform sampler2DArray u_tex;    // one layer per reactive screen (per face-index)
uniform vec3 u_flatcol;
uniform vec3 u_bg;
uniform float u_fog0, u_fog1;
in vec2 v_uv;
in float v_isclip;
in float v_layer;
in float v_camz;
out vec4 f_color;
void main(){
    vec3 col = (v_isclip > 0.5)
        ? texture(u_tex, vec3(v_uv.x, 1.0 - v_uv.y, v_layer)).rgb  // this face's screen, upright
        : u_flatcol;
    float fog = clamp((v_camz - u_fog0) / max(1e-3, u_fog1 - u_fog0), 0.0, 1.0);
    col = mix(col, u_bg, fog);                             // far cubes dissolve into bg
    f_color = vec4(col, 1.0);
}
"""


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed perspective (camera looks down -z); clip.w == camera-space
    distance ahead, which the shaders reuse for fog."""
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _cube_mesh(faces_with_clips: int, n_screens: int) -> tuple:
    """Unit cube centered at origin as 36 vertices (6 faces x 2 tris).

    ``faces_with_clips`` (0..6) of the faces are flagged as clip faces
    (``in_isclip = 1``); the rest render flat. Each clip face i samples
    reactive screen ``i % n_screens`` (its texture-array layer). Returns
    interleaved (pos.xyz, uv.xy, isclip, layer) float32 for a single VBO.
    """
    # face basis: (origin corner, u edge, v edge) on the [-0.5,0.5]^3 cube
    faces = [
        ([-0.5, -0.5,  0.5], [1, 0, 0], [0, 1, 0]),   # +Z (front)
        ([ 0.5, -0.5, -0.5], [-1, 0, 0], [0, 1, 0]),  # -Z (back)
        ([ 0.5, -0.5,  0.5], [0, 0, -1], [0, 1, 0]),  # +X (right)
        ([-0.5, -0.5, -0.5], [0, 0, 1], [0, 1, 0]),   # -X (left)
        ([-0.5,  0.5,  0.5], [1, 0, 0], [0, 0, -1]),  # +Y (top)
        ([-0.5, -0.5, -0.5], [1, 0, 0], [0, 0, 1]),   # -Y (bottom)
    ]
    k = max(0, min(6, int(faces_with_clips)))
    ns = max(1, int(n_screens))
    verts = []
    for fi, (o, u, v) in enumerate(faces):
        o = np.asarray(o, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        isclip = 1.0 if fi < k else 0.0
        layer = float(fi % ns) if fi < k else 0.0
        corners = [(o, 0.0, 0.0), (o + u, 1.0, 0.0), (o + u + v, 1.0, 1.0),
                   (o, 0.0, 0.0), (o + u + v, 1.0, 1.0), (o + v, 0.0, 1.0)]
        for p, uu, vv in corners:
            verts.extend([p[0], p[1], p[2], uu, vv, isclip, layer])
    return np.asarray(verts, dtype=np.float32).reshape(-1, 7)


class CubeField:
    def __init__(self, width: int, height: int, spec: dict, seed=None, n_screens: int = 1):
        import moderngl  # lazy — matches mandelbulb; layer skips if unavailable
        self.W, self.H = int(width), int(height)
        self.n_screens = max(1, int(n_screens))
        self.ctx = moderngl.create_standalone_context()
        self._mgl = moderngl

        self.prog = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS)

        mesh = _cube_mesh(int(spec.get("faces_with_clips", 3)), self.n_screens)
        self.vbo = self.ctx.buffer(mesh.tobytes())
        self.vao = self.ctx.simple_vertex_array(
            self.prog, self.vbo, "in_pos", "in_uv", "in_isclip", "in_layer")

        face_res = int(spec.get("face_resolution", 256))
        self.tex = self.ctx.texture_array((face_res, face_res, self.n_screens), 3)
        self.tex.repeat_x = self.tex.repeat_y = False

        self.fbo = self.ctx.simple_framebuffer((self.W, self.H), components=3)

        # --- corridor geometry ---
        self.corridor = float(spec.get("corridor_depth", 22.0))
        self.near_pad = float(spec.get("near_pad", 0.6))
        r_min = float(spec.get("radius_min", 1.6))
        r_max = float(spec.get("radius_max", 5.5))
        self.fov = float(spec.get("fov", 62.0))
        self.flatcol = np.asarray(
            spec.get("face_color", [18, 18, 22]), dtype=np.float32) / 255.0

        n = int(spec.get("n_cubes", 40))
        rng = np.random.default_rng(seed)
        self.z0 = rng.random(n).astype(np.float32)                  # depth phase [0,1)
        ang = rng.random(n).astype(np.float32) * 2.0 * np.pi
        rad = np.sqrt(rng.random(n).astype(np.float32)) * (r_max - r_min) + r_min
        self.px = (np.cos(ang) * rad).astype(np.float32)
        self.py = (np.sin(ang) * rad).astype(np.float32)
        self.size = (rng.random(n).astype(np.float32) *
                     (float(spec.get("size_max", 1.5)) - float(spec.get("size_min", 0.7)))
                     + float(spec.get("size_min", 0.7)))
        self.spin = rng.integers(-2, 3, size=(n, 3)).astype(np.float32)  # int -> loop-safe
        self.phase0 = rng.random((n, 3)).astype(np.float32) * 2.0 * np.pi

        near = 0.05
        far = self.near_pad + self.corridor + (float(self.size.max()) if n else 2.0)
        self.proj = _perspective(self.fov, self.W / self.H, near, far)
        self.fog0 = self.corridor * float(spec.get("fog_start", 0.55))
        self.fog1 = self.corridor * float(spec.get("fog_end", 0.98))

        self.prog["u_tex"] = 0
        self.prog["u_flatcol"].value = tuple(float(c) for c in self.flatcol)
        self.prog["u_fog0"].value = self.fog0
        self.prog["u_fog1"].value = self.fog1

    @staticmethod
    def _model(x, y, depth, size, rx, ry, rz) -> np.ndarray:
        """Camera-space model matrix: place a cube of side ``size`` at
        (x, y, -depth) with an XYZ rotation. View is identity (camera at the
        origin looking down -z), so this matrix *is* the camera-space transform."""
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        R = (Rz @ Ry @ Rx) * size
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = R
        m[:3, 3] = (x, y, -depth)
        return m

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        cf = np.ascontiguousarray(frame, dtype=np.uint8)
        if cf.shape[:2] != (self.tex.height, self.tex.width):
            iy = (np.arange(self.tex.height) * cf.shape[0] // self.tex.height)
            ix = (np.arange(self.tex.width) * cf.shape[1] // self.tex.width)
            cf = np.ascontiguousarray(cf[iy][:, ix])
        return cf

    def render(self, phase: float, clip_frames, bg_color) -> np.ndarray:
        """Render the cube field at loop ``phase`` in [0,1). ``clip_frames`` is
        a single (H,W,3) screen or a list of ``n_screens`` screens (one per
        face-index); a single screen is broadcast to every clip face."""
        gl = self._mgl
        bg = np.asarray(bg_color, dtype=np.float32) / 255.0
        self.prog["u_bg"].value = (float(bg[0]), float(bg[1]), float(bg[2]))

        # upload each reactive screen into its own texture-array layer
        if isinstance(clip_frames, np.ndarray) and clip_frames.ndim == 3:
            clip_frames = [clip_frames]
        frames = list(clip_frames) or [np.zeros((self.tex.height, self.tex.width, 3),
                                                 dtype=np.uint8)]
        layers = np.empty((self.n_screens, self.tex.height, self.tex.width, 3),
                          dtype=np.uint8)
        for i in range(self.n_screens):
            layers[i] = self._fit(frames[i % len(frames)])
        self.tex.write(layers.tobytes())
        self.tex.use(0)

        self.fbo.use()
        self.ctx.clear(float(bg[0]), float(bg[1]), float(bg[2]))
        self.ctx.enable(gl.DEPTH_TEST)

        # depth ahead of each cube, periodic in phase -> seamless loop
        d = (self.z0 - float(phase)) % 1.0
        depth = self.near_pad + d * self.corridor
        rot = self.spin * float(phase) * 2.0 * np.pi + self.phase0  # int spin -> loop-safe

        order = np.argsort(-depth)  # far to near (painter-friendly; depth test does the rest)
        for i in order:
            model = self._model(self.px[i], self.py[i], depth[i], self.size[i],
                                 rot[i, 0], rot[i, 1], rot[i, 2])
            mvp = (self.proj @ model).T.astype("f4")   # transpose -> column-major for GL
            self.prog["u_mvp"].write(mvp.tobytes())
            self.vao.render(mode=gl.TRIANGLES, vertices=36)

        buf = self.fbo.read(components=3)
        # GL framebuffer is bottom-left origin; flip to the system's top-left
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.H, self.W, 3)[::-1].copy()

    def release(self) -> None:
        for obj in (self.vao, self.vbo, self.tex, self.fbo, self.prog):
            try:
                obj.release()
            except Exception:
                pass
        try:
            self.ctx.release()
        except Exception:
            pass
