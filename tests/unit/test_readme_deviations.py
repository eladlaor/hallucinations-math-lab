"""The README's "How does this SEP differ from the paper?" table tracks SEPPaperDeviation."""

import re
from pathlib import Path

from hallucinations_math_lab.constants import SEPPaperDeviation

README = Path(__file__).resolve().parents[2] / "README.md"
DEVIATION_ROW = re.compile(r"^\| (D\d+) \|", re.MULTILINE)


def test_readme_deviation_table_lists_exactly_the_enum_ids() -> None:
    ids = DEVIATION_ROW.findall(README.read_text(encoding="utf-8"))
    assert ids == [str(d) for d in SEPPaperDeviation]
