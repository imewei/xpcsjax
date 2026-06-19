"""Data loading and validation pipeline for xpcsjax CLI.

Ported from heterodyne/cli/data_pipeline.py. Adapted to xpcsjax's
NLSQ-only surface, xpcsjax's analysis-mode taxonomy
(``static_anisotropic`` / ``static_isotropic`` / ``laminar_flow`` /
``two_component``), and the dict-shaped return of
:func:`xpcsjax.load_xpcs_data`.

Public API:
    * :func:`load_and_validate_data` -- load XPCS data, apply phi filtering.
    * :func:`resolve_phi_angles`     -- pick phi angles from CLI or config.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from xpcsjax.service.data import load_dataset
from xpcsjax.service.data import resolve_phi_angles as _service_resolve_phi_angles
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from xpcsjax.config.manager import ConfigManager

logger = get_logger(__name__)


def load_and_validate_data(
    args: argparse.Namespace,
    config_manager: ConfigManager,
) -> dict[str, Any]:
    """Load XPCS data + apply phi filtering (delegates to the service)."""
    return load_dataset(config_manager, phi_subset=getattr(args, "phi", None))


def resolve_phi_angles(
    args: argparse.Namespace,
    config_manager: ConfigManager,
) -> list[float] | None:
    """Pick phi angles from CLI args or config (delegates to the service)."""
    return _service_resolve_phi_angles(
        config_manager,
        cli_phi=getattr(args, "phi", None),
        phi_angles_str=getattr(args, "phi_angles", None),
    )


__all__ = ["load_and_validate_data", "resolve_phi_angles"]
