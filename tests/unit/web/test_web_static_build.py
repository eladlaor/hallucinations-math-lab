"""The committed Vite build: users run the playground without Node installed."""

import re
from pathlib import Path

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


README = Path(__file__).resolve().parents[3] / "README.md"
# The README's three ways in, cheapest first: the playground's tabs must read the same.
TAB_LABELS = ("Score answers", "Ask the model", "Measure the detectors")
RETIRED_TAB_LABEL = "Benchmark results"
QUOTE = "[`\"']"
TAB_LIST_GAP = ".{0,200}?"


def _built_script() -> str:
    referenced = ASSET_REFERENCE.findall((STATIC_DIR / INDEX_HTML).read_text())
    scripts = [asset for asset in referenced if asset.endswith(".js")]
    assert scripts, "index.html references no built script"
    return "".join((STATIC_DIR / asset).read_text() for asset in scripts)


def test_the_committed_build_names_the_tabs_as_the_readme_does() -> None:
    """A rebuilt bundle that lags a tab rename would show users a name the README never uses."""
    script = _built_script()

    # The labels also appear elsewhere (the Score button, the Ask tab's copy), so look for
    # the tab list itself: the three string literals in order, a few dozen characters apart.
    tab_list = TAB_LIST_GAP.join(f"{QUOTE}{re.escape(label)}{QUOTE}" for label in TAB_LABELS)
    assert re.search(tab_list, script), "the build has no tab list in the README's order"
    assert RETIRED_TAB_LABEL not in script


def test_the_readme_numbers_its_ways_in_in_tab_order() -> None:
    """The labels recur in README prose, so a bare `in` check would miss a reorder or rename."""
    readme = README.read_text()

    headings = [f"**{number}. {label}**" for number, label in enumerate(TAB_LABELS, start=1)]
    positions = [readme.find(heading) for heading in headings]
    assert all(position >= 0 for position in positions), f"README lacks one of {headings}"
    assert positions == sorted(positions), "README lists its ways in out of tab order"
