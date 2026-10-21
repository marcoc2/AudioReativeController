"""The page itself, in a headless browser: it boots, opens a project, a drag edits a block.

Needs Playwright (``pip install playwright`` and ``playwright install chromium``); skipped
without it. One command: ``pytest tests/test_web_page.py``.
"""
import time

import pytest
import yaml

sync_api = pytest.importorskip("playwright.sync_api")

from core.web.api import Api                       # noqa: E402
from core.web.server import Server                 # noqa: E402
from tests.test_web_session import SCENE, _midi     # noqa: E402


@pytest.fixture
def page_url(tmp_path, monkeypatch):
    monkeypatch.setattr("core.web.api.PREFS", tmp_path / "prefs.json")
    scene = tmp_path / "scene.yaml"
    scene.write_text(SCENE, encoding="utf-8")
    proj = tmp_path / "proj.yaml"
    proj.write_text(yaml.safe_dump({"audio": str(tmp_path / "missing.wav"), "midi": str(_midi(tmp_path / "song.mid")),
                                    "scene": str(scene), "render": {"fps": 30, "resolution": "854x480"}}), encoding="utf-8")
    api = Api(str(proj), browser=True)
    server = Server(api, api._media, api._events)
    url = server.start()
    yield url, scene
    api._close()
    server.shutdown()


def test_the_page_boots_opens_and_a_drag_edits(page_url):
    url, scene = page_url
    with sync_api.sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        page.goto(url)
        page.wait_for_function("window.S && members.length > 0", timeout=20000)
        block = page.locator(".block.bg").first
        block.scroll_into_view_if_needed()
        box = block.bounding_box()
        before = page.evaluate("memberById(document.querySelector('.block.bg').dataset.id).spans[0].first")
        ppb = page.evaluate("ppb")
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2 + 2 * ppb, box["y"] + box["height"] / 2, steps=6)
        page.mouse.up()
        page.wait_for_function("dirty === true", timeout=5000)
        page.keyboard.press("Control+s")
        page.wait_for_function("dirty === false && savedAt !== ''", timeout=5000)
        browser.close()
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    firsts = [m["bars"][0][0] if isinstance(m["bars"][0], list) else m["bars"][0]
              for m in data["video"]["layers"][1]["layers"] if m.get("bars")]
    assert before + 2 in firsts
    time.sleep(0.1)
