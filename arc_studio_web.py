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


def own_popups() -> None:
    """A panel the page puts in a window of its own (``window.open("")``, about:blank) opens
    in a plain window of the app (a WinForms form, no address bar), whose WebView2 is handed
    to the page as that popup, so the page's script reaches and drives it; any other link
    still goes to the browser, as pywebview does."""
    from webview.platforms import edgechromium
    original = edgechromium.EdgeChrome.on_new_window_request

    def on_new_window_request(self, sender, args):
        if str(args.get_Uri()) not in ("about:blank", ""):
            original(self, sender, args)
            return
        WinForms = edgechromium.WinForms
        deferral = args.GetDeferral()
        form = WinForms.Form()
        form.Text = "ARC Studio"
        form.Icon = self.form.Icon
        form.BackColor = self.webview.DefaultBackgroundColor
        feats = args.WindowFeatures
        form.Width = int(feats.Width) if feats.HasSize else 560
        form.Height = int(feats.Height) if feats.HasSize else 800
        view = edgechromium.WebView2()
        view.DefaultBackgroundColor = self.webview.DefaultBackgroundColor
        view.Dock = WinForms.DockStyle.Fill
        form.Controls.Add(view)

        def ready(_sender, _args):
            core = view.CoreWebView2
            args.NewWindow = core                    # the page's popup is this window's view
            args.Handled = True
            deferral.Complete()
            core.DocumentTitleChanged += lambda s, e: setattr(form, "Text", core.DocumentTitle or "ARC Studio")
            core.WindowCloseRequested += lambda s, e: form.Close()
        view.CoreWebView2InitializationCompleted += ready
        form.Show()
        env = sender.Environment
        options = env.CreateCoreWebView2ControllerOptions()   # the same profile as the page's
        options.IsInPrivateModeEnabled = sender.Profile.IsInPrivateModeEnabled
        view.EnsureCoreWebView2Async(env, options)
    edgechromium.EdgeChrome.on_new_window_request = on_new_window_request


def main(argv=None) -> None:
    from core.web.api import Api, project_from_args, stems_from_args
    from core.web.server import Server
    args = parse_args(argv)
    api = Api(project_from_args(args), browser=args.browser, skip_separation=args.skip_separation,
              stems=stems_from_args(args))     # --stems alone: over the opened project's own
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
        own_popups()
        window = webview.create_window("ARC Studio", url, js_api=api, width=1500, height=980, min_size=(1000, 680),
                                       background_color="#111318", maximized=True)
        api._attach(window)
        webview.start(debug=args.debug)
    api._close()
    server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
