"""ARC Studio in a web view: the Arrange as an HTML page (webui/), its state in Python.

``server`` serves the page, the media it may read, the API over HTTP and the events;
``session`` keeps the open project and its arrangement (every edit, undo, save); ``load``
opens a project in steps the page shows as they go; ``render`` runs the engine; ``api`` is
what the page calls. Python is the only source of truth: the page draws what it is told.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]     # the repository: projects, scenes and media are relative to it
UI = ROOT / "webui"
WORK = ROOT / "render_output" / ".web"         # the scenes the engine renders from, its previews, covers
