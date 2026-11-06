"""The Score tab's presets: the SCENARIOS of examples/score_precollected_samples.py.

The example is imported, never copied, so the presets and the script cannot
drift. It is imported by name at runtime because mypy maps examples/ as
top-level modules (the examples import each other as scripts); the value is
then validated, so a change in its shape fails loudly here.
"""

import importlib
import logging

from pydantic import TypeAdapter, ValidationError

from demo.web.api_schemas import Preset
from demo.web.constants import PRESETS_ATTRIBUTE, PRESETS_MODULE
from hallucinations_math_lab import Sample
from hallucinations_math_lab.constants import LOG_EXTRA_KEY

logger = logging.getLogger(__name__)

_SCENARIOS_ADAPTER: TypeAdapter[dict[str, tuple[str, list[Sample]]]] = TypeAdapter(
    dict[str, tuple[str, list[Sample]]]
)


def load_presets() -> list[Preset]:
    """Every example scenario, in the example's order."""
    try:
        module = importlib.import_module(PRESETS_MODULE)
        scenarios = _SCENARIOS_ADAPTER.validate_python(getattr(module, PRESETS_ATTRIBUTE))
    except (ImportError, AttributeError, ValidationError) as e:
        logger.error(
            "Could not load the Score tab presets.",
            extra={LOG_EXTRA_KEY: {"module": PRESETS_MODULE, "error": str(e)}},
        )
        raise RuntimeError(
            f"Could not load {PRESETS_MODULE}.{PRESETS_ATTRIBUTE} for the presets; run the "
            f"playground from the repository root (`uv run python -m demo.web`): {e!s}"
        ) from e
    return [
        Preset(name=name, question=question, samples=samples)
        for name, (question, samples) in scenarios.items()
    ]
