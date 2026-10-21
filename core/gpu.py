"""The GPU context every layer shares — one OpenGL context per thread.

Layers used to open a context each, so a texture made by one could not be read by the
next and every frame went back to the CPU between them. Now ``context()`` hands every
layer of a render the same context, the ground the GPU compositor stands on.

One per *thread*: an OpenGL context can be current in one thread only, and the studio
builds and renders each preview in a thread of its own; ``release()`` frees the calling
thread's context (and everything made in it) when that thread is done.

Rules for code that draws in it, now that the context is shared:
  * set the state a pass needs (blend, depth test, point size) and put it back after;
  * bind the textures a pass reads (``tex.use(unit)``) every time, not once at start-up;
  * never ``release()`` the context itself — only what you made in it.
"""
from __future__ import annotations

import threading

_local = threading.local()


def context():
    """The calling thread's shared moderngl context (made on first use; GL 4.3 when there
    is one, for the compute shaders of ``flame``)."""
    ctx = getattr(_local, "ctx", None)
    if ctx is None:
        import moderngl
        try:
            ctx = moderngl.create_standalone_context(require=430)
        except Exception:
            ctx = moderngl.create_standalone_context()
        _local.ctx = ctx
    return ctx


def release() -> None:
    """Free the calling thread's context; the next ``context()`` makes a fresh one."""
    ctx = getattr(_local, "ctx", None)
    if ctx is not None:
        _local.ctx = None
        ctx.release()
