"""The local server the page talks to: the page's own files, the media it may read (and
nothing else on the disk), the API over HTTP and what Python tells the page (events).

In the pywebview window the page calls Python through ``js_api`` and events are pushed
with ``evaluate_js``; the same page in a browser (``--browser``) calls ``POST /api/<name>``
and polls ``GET /events``, both with the page's token.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import secrets
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

from core.web import ROOT, UI

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("audio/mpeg", ".mp3")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/webm", ".webm")

EVENTS_KEPT = 500


class Media:
    """The files outside webui/ the page may read: each one allowed by being named here."""

    def __init__(self):
        self._files: dict = {}
        self._lock = threading.Lock()

    def url(self, path, fresh: bool = False) -> str:
        """``fresh``: a file made again under the same name gets a new URL (its time in it)."""
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        name = hashlib.sha1(str(p).lower().encode()).hexdigest()[:16] + p.suffix.lower()
        with self._lock:
            self._files[name] = p
        if fresh and p.is_file():
            return f"/media/{name}?v={p.stat().st_mtime_ns}"
        return f"/media/{name}"

    def get(self, name: str) -> Optional[Path]:
        with self._lock:
            return self._files.get(name)


class Events:
    """What Python tells the page: pushed into the web view when there is one, and kept
    (numbered) for a page in a browser to poll."""

    def __init__(self):
        self.window = None
        self._kept: list = []
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, fn: str, data) -> None:
        with self._lock:
            self._seq += 1
            self._kept.append({"seq": self._seq, "fn": fn, "data": data})
            del self._kept[:-EVENTS_KEPT]
        if self.window is not None:
            try:
                self.window.evaluate_js(f"window.arc && arc.event({json.dumps(fn)}, {json.dumps(data)})")
            except Exception:                    # the window is closing
                pass

    def since(self, seq: int) -> list:
        with self._lock:
            return [e for e in self._kept if e["seq"] > seq]

    def log(self, kind: str, text: str) -> None:
        """A line of the log: ``kind`` is in (sent to Python), out (asked), sys, warn, err, trace."""
        self.emit("log", {"kind": kind, "text": text})
        try:
            print(text if kind != "trace" else text.rstrip(), flush=True)
        except (OSError, UnicodeError):
            pass

    def gpu(self, what: str, on: bool) -> None:
        """Python's work on the GPU starts (``on``) or ends: while any runs, the page's live
        preview stops drawing (it would fight it for the card)."""
        with self._lock:
            jobs = getattr(self, "_gpu", set())
            (jobs.add if on else jobs.discard)(what)
            self._gpu = jobs
            busy = sorted(jobs)
        self.emit("gpu", {"busy": busy})


class Server:
    """The page's server on 127.0.0.1 (a free port); ``api`` answers ``/api/<name>``."""

    def __init__(self, api, media: Media, events: Events):
        self.api, self.media, self.events = api, media, events
        self.token = secrets.token_urlsafe(16)
        self.httpd: Optional[ThreadingHTTPServer] = None

    def start(self, port: int = 0) -> str:
        server = self

        class Handler(_Handler):
            owner = server
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/?t={self.token}"

    def shutdown(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()


def byte_range(header: str, size: int):
    """(start, end, status) for a ``Range`` header over a file of ``size`` bytes; status 416
    when the range starts past the end."""
    m = re.match(r"bytes=(\d*)-(\d*)$", (header or "").strip())
    if not m or not size or not (m.group(1) or m.group(2)):
        return 0, size - 1, 200
    if m.group(1):
        start = int(m.group(1))
        if start >= size:
            return 0, 0, 416
        end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        if end < start:
            return 0, 0, 416
        return start, end, 206
    return max(0, size - int(m.group(2))), size - 1, 206


class _Handler(BaseHTTPRequestHandler):
    server_version = "ARCStudio"
    owner: Server = None

    def log_message(self, *args):          # quiet
        pass

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _allowed(self) -> bool:
        return self.headers.get("X-ARC-Token") == self.owner.token

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if not self._allowed() or not path.startswith("/api/"):
            self.send_error(403)
            return
        name = path[5:]
        fn: Optional[Callable] = getattr(self.owner.api, name, None)
        if name.startswith("_") or not callable(fn):
            self.send_error(404)
            return
        try:
            args = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"[]")
        except ValueError:
            self.send_error(400)
            return
        self._json(fn(*args))

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path)
        if path == "/events":
            if not self._allowed():
                self.send_error(403)
                return
            since = int((urllib.parse.parse_qs(url.query).get("since") or ["0"])[0])
            self._json(self.owner.events.since(since))
            return
        f: Optional[Path] = None
        if path in ("/", "/index.html"):
            f = UI / "index.html"
        elif path.startswith("/ui/"):
            f = (UI / path[4:]).resolve()
            if UI.resolve() not in f.parents:
                f = None
        elif path.startswith("/media/"):
            f = self.owner.media.get(path[7:])
        if f is None or not f.is_file():
            self.send_error(404)
            return
        self._send(f)

    def _send(self, f: Path):
        size = f.stat().st_size
        start, end, status = byte_range(self.headers.get("Range"), size)
        if status == 416:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(f, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 16, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                left -= len(chunk)
