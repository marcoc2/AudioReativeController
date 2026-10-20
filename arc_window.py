"""The studio's window on Windows: files dropped from the Explorer, and the mouse cursor's
shape when the app wants one of its own.

dearpygui has neither: the viewport's window is told to accept files (DragAcceptFiles) and
its window procedure is wrapped to catch WM_DROPFILES and WM_SETCURSOR.

- A drop is only queued there (dearpygui is not touched from inside the window procedure);
  ``take()``, on the main loop, hands the drops over: (paths, (x, y)) with (x, y) where the
  mouse let go, in the viewport's client area (the space of ``dpg.get_mouse_pos(local=False)``).
- ``set_cursor("resize")`` shows the left-right arrows (a block's end that Shift+drag
  takes) until ``set_cursor(None)``; meanwhile WM_SETCURSOR is answered here, not by ImGui.
"""
import ctypes
import sys
from ctypes import wintypes

WM_DROPFILES = 0x0233
WM_SETCURSOR = 0x0020
HTCLIENT = 1
GWLP_WNDPROC = -4
CURSORS = {"resize": 32644, "arrow": 32512}      # IDC_SIZEWE, IDC_ARROW

_drops = []
_hooked = {}          # hwnd -> (the new window procedure, the old one): kept alive
_cursor = [None]      # the shape asked for (None: ImGui's own)


def take() -> list:
    """The drops since the last call: [(paths, (x, y))]."""
    out = _drops[:]
    del _drops[:]
    return out


def set_cursor(kind) -> None:
    """The cursor's shape over the window: "resize", or None (back to ImGui's)."""
    if kind == _cursor[0]:
        return
    _cursor[0] = kind
    if _hooked:                                          # now, not at the next mouse move
        _show(CURSORS.get(kind or "arrow", CURSORS["arrow"]))


def cursor():
    return _cursor[0]


def _show(idc: int) -> None:
    user32 = ctypes.windll.user32
    user32.LoadCursorW.restype = ctypes.c_void_p
    user32.LoadCursorW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    user32.SetCursor.argtypes = [ctypes.c_void_p]
    user32.SetCursor(user32.LoadCursorW(None, ctypes.c_void_p(idc)))


def install(title: str) -> bool:
    """The viewport titled ``title`` (made by this thread) takes files dropped on it and the
    cursor shapes asked for; False when it can't (not Windows, no such window)."""
    if sys.platform != "win32":
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = _shell32()
    hwnd = _viewport(user32, title)
    if not hwnd or hwnd in _hooked:
        return bool(hwnd)

    LRESULT = ctypes.c_ssize_t
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                                       wintypes.LPARAM]
    user32.CallWindowProcW.restype = LRESULT
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.DefWindowProcW.restype = LRESULT

    def proc(h, msg, wp, lp):
        if msg == WM_DROPFILES:
            try:
                _drops.append(_read(shell32, wp))
            except Exception:                            # a drop lost, never the window
                pass
            return 0
        if msg == WM_SETCURSOR and _cursor[0] in CURSORS and (lp or 0) & 0xFFFF == HTCLIENT:
            try:
                _show(CURSORS[_cursor[0]])
                return 1
            except Exception:                            # ImGui's cursor, then
                pass
        old = _hooked.get(h, (None, None))[1]
        if not old:                                      # (only while being hooked)
            return user32.DefWindowProcW(h, msg, wp, lp)
        return user32.CallWindowProcW(old, h, msg, wp, lp)

    new = WNDPROC(proc)
    _hooked[hwnd] = (new, None)
    old = user32.SetWindowLongPtrW(hwnd, GWLP_WNDPROC, ctypes.cast(new, ctypes.c_void_p))
    if not old:
        del _hooked[hwnd]
        return False
    _hooked[hwnd] = (new, old)
    shell32.DragAcceptFiles(hwnd, True)
    return True


def _shell32():
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    shell32.DragQueryFileW.restype = wintypes.UINT
    shell32.DragQueryPoint.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.POINT)]
    shell32.DragFinish.argtypes = [ctypes.c_void_p]
    shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
    return shell32


def _read(shell32, hdrop) -> tuple:
    """(paths, (x, y)) of a drop (HDROP), which is then let go."""
    n = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
    paths = []
    for i in range(n):
        size = shell32.DragQueryFileW(hdrop, i, None, 0) + 1
        buf = ctypes.create_unicode_buffer(size)
        shell32.DragQueryFileW(hdrop, i, buf, size)
        paths.append(buf.value)
    pt = wintypes.POINT()
    shell32.DragQueryPoint(hdrop, ctypes.byref(pt))
    shell32.DragFinish(hdrop)
    return paths, (float(pt.x), float(pt.y))


def _viewport(user32, title: str):
    """The visible top-level window of this thread titled ``title`` (0: none)."""
    found = []
    ENUM = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.EnumThreadWindows.argtypes = [wintypes.DWORD, ENUM, wintypes.LPARAM]

    def each(h, _):
        n = user32.GetWindowTextLengthW(h)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(h, buf, n + 1)
        if buf.value == title and user32.IsWindowVisible(h):
            found.append(h)
            return False
        return True

    user32.EnumThreadWindows(ctypes.windll.kernel32.GetCurrentThreadId(), ENUM(each), 0)
    return found[0] if found else 0
