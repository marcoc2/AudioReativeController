#!/usr/bin/env python3
"""ARC Studio: one window, the Arrange, opened straight from a project (pywebview).

The window is HTML (webui/); everything it shows comes from core/ in this process
(core/web: the session, the load in steps, the engine's render). With no project it opens
the project picker; while a project loads, its steps show as they go.

Usage:
    python arc_studio_web.py --project projects/esfolado_elenco.yaml
    python arc_studio_web.py --audio input/song.wav --midi input/song.mid --scene scenes/song.yaml
    python arc_studio_web.py --browser        # the same page in a browser (for tests)
"""
from __future__ import annotations

import argparse
import sys
import threading


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="ARC Studio (the Arrange, in a web view)")
    ap.add_argument("project_file", nargs="?", default=None, help="a project YAML file (as --project)")
    ap.add_argument("--project", default=None, help="open a project YAML file")
    ap.add_argument("--audio", default=None, help="audio file (a project made on the spot)")
    ap.add_argument("--midi", default=None, help="MIDI file")
    ap.add_argument("--scene", default=None, help="scene YAML file")
    ap.add_argument("--stems", nargs="*", default=[], metavar="name=path", help="prebuilt stems, e.g. solo=input/solo.mp3")
    ap.add_argument("--skip-separation", action="store_true", help="no AI stem separation")
    ap.add_argument("--browser", action="store_true", help="serve the page and print its URL instead of opening a window")
    ap.add_argument("--port", type=int, default=0, help="the local server's port (0: a free one)")
    ap.add_argument("--token", default=None, help="with --browser: the page's key, fixed (for tests); else a random one")
    ap.add_argument("--debug", action="store_true", help="the web view's developer tools")
    args = ap.parse_args(argv)
    args.project = args.project or args.project_file
    return args


def main(argv=None) -> None:
    from core.web.api import Api, project_from_args
    from core.web.server import Server
    args = parse_args(argv)
    api = Api(project_from_args(args), browser=args.browser, skip_separation=args.skip_separation)
    server = Server(api, api._media, api._events)
    if args.browser and args.token:
        server.token = args.token
    url = server.start(args.port)
    if args.browser:
        print(f"ARC Studio: {url}", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    else:
        import webview
        window = webview.create_window("ARC Studio", url, js_api=api, width=1500, height=980, min_size=(1000, 680),
                                       background_color="#111318", maximized=True)
        api._attach(window)
        webview.start(debug=args.debug)
    api._close()
    server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
