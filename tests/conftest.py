import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sam2_on_the_gpu(monkeypatch):
    """No test runs SAM2 for real: without its weights, a cutout that would be made fails
    at once (a test that wants one made fakes the run)."""
    from core.video import sam_cutout
    monkeypatch.setattr(sam_cutout, "WEIGHTS", ())
    monkeypatch.delenv("ARC_SAM2_WEIGHTS", raising=False)
