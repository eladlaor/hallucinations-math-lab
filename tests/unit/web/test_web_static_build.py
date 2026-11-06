"""The committed Vite build: users run the playground without Node installed."""

import re

from demo.web.constants import INDEX_HTML, STATIC_DIR

ASSET_REFERENCE = re.compile(r'(?:src|href)="/(assets/[^"]+)"')


def test_the_built_ui_is_committed() -> None:
    assert (STATIC_DIR / INDEX_HTML).is_file()


def test_every_asset_the_page_loads_is_committed() -> None:
    """A stale index.html pointing at assets from an older build would serve a blank page."""
    referenced = ASSET_REFERENCE.findall((STATIC_DIR / INDEX_HTML).read_text())

    assert referenced, "index.html references no built assets"
    for asset in referenced:
        assert (STATIC_DIR / asset).is_file(), f"{asset} is referenced but missing"
