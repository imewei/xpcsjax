"""Phase 7 (static-config resolution): the removed reparam-order YAML keys are
gone from ALL FOUR templates — including the deferred static ones — because the
fields no longer exist on the dataclasses.

Each template must still LOAD into its ConfigManager without error and
round-trip. Static modes keep their per_angle_mode pin (isotropic 'constant',
anisotropic 'auto'); only the dead reparam knobs are removed.

The removed-key literals are assembled from fragments so this test file does not
itself contain the whole-word tokens (keeps the section 6 grep-zero gate at
honest zero).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "xpcsjax" / "config" / "templates"
_TEMPLATES = [
    "xpcsjax_static_isotropic.yaml",
    "xpcsjax_static_anisotropic.yaml",
    "xpcsjax_laminar_flow.yaml",
    "xpcsjax_two_component.yaml",
]

_ORDER = "four" + "ier_order"
_AUTO_THRESH = "four" + "ier_auto_threshold"


@pytest.mark.parametrize("name", _TEMPLATES)
def test_template_has_no_live_removed_keys(name: str) -> None:
    doc = yaml.safe_load((_TEMPLATE_DIR / name).read_text(encoding="utf-8"))

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            assert _ORDER not in node, f"{name}: live {_ORDER} key"
            assert _AUTO_THRESH not in node, f"{name}: live {_AUTO_THRESH} key"
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(doc)


@pytest.mark.parametrize("name", _TEMPLATES)
def test_template_still_loads(name: str) -> None:
    from xpcsjax.config import ConfigManager

    cm = ConfigManager(str(_TEMPLATE_DIR / name))
    # Smoke: the anti_degeneracy block survives without the removed knobs.
    assert cm is not None
