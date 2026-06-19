"""Headless config-loading and JAX-free validation service.

Provides config loading (``load_config``), template generation
(``get_template_config``), and in-process YAML validation
(``validate_config_dict``) without importing JAX or any xpcsjax engine module.
Uses ``ConfigManager``, ``ParameterRegistry``, ``AnalysisMode``, ``yaml``, and
``importlib.resources.files`` for packaged-template access.  Safe to call from
the GUI process in-process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import yaml

from xpcsjax.config.manager import ConfigManager
from xpcsjax.config.parameter_registry import ParameterRegistry
from xpcsjax.config.types import AnalysisMode
from xpcsjax.utils.logging import get_logger

# Mode -> packaged template filename. Read directly via importlib.resources so we
# NEVER import xpcsjax.cli.config_generator (which imports ConfigManager -> JAX).
_TEMPLATE_FILES = {
    "static_isotropic": "xpcsjax_static_isotropic.yaml",
    "static_anisotropic": "xpcsjax_static_anisotropic.yaml",
    "laminar_flow": "xpcsjax_laminar_flow.yaml",
    "two_component": "xpcsjax_two_component.yaml",
}

logger = get_logger(__name__)


def load_config(
    path: str | Path,
    *,
    mode: str | None = None,
    output_dir: str | Path | None = None,
) -> ConfigManager:
    """Load a YAML/JSON config and apply the mode + output-dir overrides.

    Parameters
    ----------
    path : str or pathlib.Path
        Config file path.
    mode : str, optional
        Overrides ``analysis_mode`` (re-normalized after the write).
    output_dir : str or pathlib.Path, optional
        Overrides the canonical ``output.directory`` schema key.

    Returns
    -------
    ConfigManager
        Manager holding the merged effective config.
    """
    logger.info("Loading configuration from %s", path)
    config_manager = ConfigManager(str(path))

    config = config_manager.config
    if not isinstance(config, dict):
        return config_manager

    if mode is not None:
        old_mode = config.get("analysis_mode")
        config["analysis_mode"] = mode
        if old_mode != mode:
            logger.info("Override: analysis_mode = %s (was %s)", mode, old_mode)
        try:
            config_manager._normalize_analysis_mode()
        except AttributeError:  # pragma: no cover
            pass

    if output_dir is not None:
        out = config.setdefault("output", {})
        if not isinstance(out, dict):  # pragma: no cover — defensive
            out = {}
            config["output"] = out
        old = out.get("directory")
        out["directory"] = str(output_dir)
        logger.info("Override: output.directory = %s (was %s)", output_dir, old)

    return config_manager


@dataclass(frozen=True)
class ValidationReport:
    """JAX-free outcome of validating a config dict.

    Attributes
    ----------
    ok : bool
        ``True`` when there are no errors.
    errors : list of str
        Hard validation failures (unknown mode, out-of-bounds value, length mismatch).
    warnings : list of str
        Soft issues (parameter name not used by the selected mode).
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def available_modes() -> list[str]:
    """Return the four known analysis-mode strings.

    Returns
    -------
    list of str
        The four mode strings: ``static_anisotropic``, ``static_isotropic``,
        ``laminar_flow``, ``two_component``.
    """
    return [m.value for m in AnalysisMode]


def template_dict(mode: str) -> dict:
    """Load the packaged YAML template for *mode* (JAX-free; raises on unknown mode).

    Parameters
    ----------
    mode : str
        One of the four known analysis modes.

    Returns
    -------
    dict
        Parsed YAML template content.

    Raises
    ------
    ValueError
        If *mode* is not one of the four known modes.
    """
    if mode not in _TEMPLATE_FILES:
        raise ValueError(f"unknown mode: {mode!r} (expected one of {available_modes()})")
    path = files("xpcsjax.config") / "templates" / _TEMPLATE_FILES[mode]
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def validate_config(config: dict) -> ValidationReport:
    """Validate a config dict JAX-free against the registry.

    Honors the template schema ``initial_parameters.{parameter_names, values}``.
    Bounds use the registry's per-name global bounds (``get_bounds``); config-level
    ``parameter_space.bounds`` overrides are the fit-time authority, so this is a
    lightweight editor sanity check, not the final bounds resolution.

    Parameters
    ----------
    config : dict
        Parsed config dict (e.g. from ``load_config`` or ``yaml.safe_load``).

    Returns
    -------
    ValidationReport
        Frozen report with ``ok``, ``errors``, and ``warnings``.
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        mode_enum = AnalysisMode(config.get("analysis_mode"))
    except ValueError:
        return ValidationReport(
            ok=False,
            errors=[
                f"unknown analysis_mode: {config.get('analysis_mode')!r} "
                f"(expected one of {available_modes()})"
            ],
        )

    registry = ParameterRegistry()
    expected = set(registry.get_all_param_names(mode_enum, include_scaling=False))

    ip = config.get("initial_parameters") or {}
    names = list(ip.get("parameter_names", []) or [])
    values = ip.get("values")

    for name in names:
        if name not in expected:  # mode-specific membership (e.g. v_beta vs beta)
            warnings.append(f"parameter {name!r} is not used by mode {mode_enum.value}")

    if isinstance(values, list):
        if len(values) != len(names):
            errors.append(
                f"initial_parameters.values has {len(values)} entries "
                f"but parameter_names has {len(names)}"
            )
        for name, value in zip(names, values, strict=False):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"{name}: value {value!r} is not numeric")
                continue
            if name in expected:
                lo, hi = registry.get_bounds(name)
                if not (lo <= numeric <= hi):
                    errors.append(f"{name}={numeric} is outside bounds ({lo}, {hi})")

    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)
