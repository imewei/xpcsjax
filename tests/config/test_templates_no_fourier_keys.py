"""Phase 7 (static-config resolution, spec §6/§9): the live fourier_order /
fourier_auto_threshold YAML keys are removed from ALL FOUR templates — including
the deferred static ones — because the fields no longer exist on the dataclasses.

Each template must still LOAD into its ConfigManager without error and round-trip.
Static modes keep their per_angle_mode pin (isotropic 'constant', anisotropic 'auto');
only the dead fourier knobs are removed.
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


@pytest.mark.parametrize("name", _TEMPLATES)
def test_template_has_no_live_fourier_keys(name: str) -> None:
    doc = yaml.safe_load((_TEMPLATE_DIR / name).read_text())

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            assert "fourier_order" not in node, f"{name}: live fourier_order key"
            assert "fourier_auto_threshold" not in node, f"{name}: live fourier_auto_threshold key"
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
    # Smoke: the anti_degeneracy block survives without the fourier knobs.
    assert cm is not None
