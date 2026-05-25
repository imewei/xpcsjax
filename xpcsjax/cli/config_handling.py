"""Configuration loading and CLI override merging for the xpcsjax CLI.

xpcsjax's :class:`~xpcsjax.config.manager.ConfigManager` differs from
heterodyne's in three ways that matter here:

* Constructor-based loading (``ConfigManager(str(path))``) rather than a
  ``from_yaml()`` classmethod.
* Direct dict access via ``cfg.config[...]`` rather than nested mutator
  helpers like heterodyne's ``update_optimization_config(group, key, value)``.
* ``update_config(key, value)`` for top-level updates only — nested
  optimization settings are written directly via ``cfg.config["optimization"][...]``.

This module wraps the resulting style into a single
``load_and_merge_config(yaml_path, args)`` call.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from xpcsjax.config.manager import ConfigManager
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_output_dir(
    args: argparse.Namespace | Any,
    cfg_manager: ConfigManager | None,
) -> Path | None:
    """Resolve the effective output directory (CLI > YAML > ``None``).

    This is the **single source of truth** for output-directory resolution,
    shared by result saving (``commands._resolve_output_dir``) and plot
    writing (``plot_dispatch.resolve_plots_dir``) so the two cannot drift
    apart — a normal run must not scatter JSON/NPZ results under the
    configured output tree while writing plots to the process cwd.

    Resolution order:
        1. ``args.output`` (CLI override)
        2. ``output.directory`` (canonical template schema), optionally under
           ``output.base_directory``
        3. ``output_settings.output_dir`` (legacy hand-written-config spelling)
        4. ``None`` (caller decides the fallback)
    """
    if getattr(args, "output", None) is not None:
        return Path(args.output)
    if cfg_manager is None:
        return None
    cfg = cfg_manager.config or {}

    # Preferred: output.directory (+ optional base_directory parent)
    output = cfg.get("output", {})
    if isinstance(output, dict):
        directory = output.get("directory")
        if directory:
            base = output.get("base_directory")
            if base:
                return Path(str(base)) / str(directory)
            return Path(str(directory))

    # Legacy fallback: output_settings.output_dir
    settings = cfg.get("output_settings", {})
    if isinstance(settings, dict):
        raw = settings.get("output_dir")
        if raw:
            return Path(str(raw))
    return None


def load_and_merge_config(
    yaml_path: Path | str,
    cli_args: argparse.Namespace,
) -> ConfigManager:
    """Load YAML config and apply CLI overrides.

    Args:
        yaml_path: Path to the YAML config file.
        cli_args: Parsed CLI namespace whose ``initial_*`` / mode / NLSQ
            attributes may override YAML values.

    Returns:
        ``ConfigManager`` with the merged effective config.
    """
    logger.info("Loading configuration from %s", yaml_path)
    config_manager = ConfigManager(str(yaml_path))
    apply_cli_overrides(config_manager, cli_args)
    return config_manager


def apply_cli_overrides(
    config_manager: ConfigManager,
    args: argparse.Namespace,
) -> None:
    """Mutate ``config_manager.config`` in place from CLI flags.

    Precedence: CLI args > YAML > parameter_registry defaults.
    """
    config = config_manager.config
    if config is None:  # pragma: no cover — load_config never returns None
        return

    # --- mode override ---
    cli_mode = getattr(args, "mode", None)
    if cli_mode is not None:
        old_mode = config.get("analysis_mode")
        config["analysis_mode"] = cli_mode
        if old_mode != cli_mode:
            logger.info("CLI override: analysis_mode = %s (was %s)", cli_mode, old_mode)
        # Re-normalize after mutation
        try:
            config_manager._normalize_analysis_mode()
        except AttributeError:  # pragma: no cover
            pass

    # --- output dir override ---
    # Canonical schema (per the shipped templates) is ``output.directory``.
    if getattr(args, "output", None) is not None:
        out = config.setdefault("output", {})
        if not isinstance(out, dict):  # pragma: no cover — defensive
            out = {}
            config["output"] = out
        old = out.get("directory")
        out["directory"] = str(args.output)
        logger.info("CLI override: output.directory = %s (was %s)", args.output, old)

    # NOTE: NLSQ runtime knobs (--multistart / --multistart-n /
    # --max-iterations / --tolerance / verbosity) are intentionally NOT
    # handled here. They are owned by
    # ``optimization_runner.apply_cli_overrides`` (the single authority for
    # the ``optimization.nlsq.*`` block), which writes the canonical keys
    # the NLSQ engine actually reads (multi_start.enable, ftol/xtol, ...).
    # Writing them in two places previously double-applied with divergent
    # keys and clobbered YAML settings.

    # --- parameter overrides ---
    _apply_parameter_overrides(config_manager, args)


# --- CLI flag attr -> canonical parameter name -----------------------
#
# Covers the union of all four modes. Names are intersected against the
# active parameter set for the resolved mode before being written, so a
# flag that doesn't apply to the active mode is safely ignored rather
# than written into the void or onto a colliding name.
#
# Per-mode canonical names (parameter_registry._MODE_PARAMS):
#   static_*    : D0, alpha, D_offset
#   laminar_flow: D0, alpha, D_offset, gamma_dot_t0, beta,
#                 gamma_dot_t_offset, phi0
#   two_component (heterodyne): D0_ref, alpha_ref, D_offset_ref,
#                 D0_sample, alpha_sample, D_offset_sample, v0, v_beta,
#                 v_offset, f0..f3, phi0_het
# Only entries with a corresponding ``--initial-*`` flag in args_parser.py
# appear here. two_component's reference/sample transport params
# (D0_ref, D0_sample, ...) and phi0_het have no CLI flags by design —
# with 14 parameters the two_component initial guess is set via YAML.
# The flags below still cover 7 of two_component's params (v0, v_beta,
# v_offset, f0..f3) since those names overlap.
_CLI_PARAM_MAP: dict[str, str] = {
    # Transport (static_* and laminar_flow)
    "initial_D0": "D0",
    "initial_alpha": "alpha",
    "initial_D_offset": "D_offset",
    # Shear / velocity exponent (laminar_flow)
    "initial_gamma_dot_t0": "gamma_dot_t0",
    "initial_gamma_dot_t_offset": "gamma_dot_t_offset",
    "initial_beta": "beta",
    # Velocity (two_component)
    "initial_v_beta": "v_beta",
    "initial_v0": "v0",
    "initial_v_offset": "v_offset",
    # Angle (laminar_flow)
    "initial_phi0": "phi0",
    # Fraction Fourier amplitudes (two_component)
    "initial_f0": "f0",
    "initial_f1": "f1",
    "initial_f2": "f2",
    "initial_f3": "f3",
}


def _apply_parameter_overrides(
    config_manager: ConfigManager,
    args: argparse.Namespace,
) -> None:
    """Write CLI ``--initial-*`` values into the canonical config block.

    ``ConfigManager.get_initial_parameters()`` reads from
    ``config["initial_parameters"]["values"]`` — a list positionally
    aligned with ``config["initial_parameters"]["parameter_names"]``.
    (It does NOT read ``config["parameters"]["initial_values"]``; writing
    there silently discarded every override.)

    Strategy: resolve the active parameter order, materialize the current
    initial values (filling registry mid-point defaults for nulls), apply
    the CLI overrides whose canonical name is in the active set, then write
    the canonical ``parameter_names`` + ``values`` lists back.
    """
    config = config_manager.config
    if config is None:  # pragma: no cover
        return

    # Collect CLI overrides keyed by canonical parameter name.
    overrides: dict[str, float] = {}
    for attr_name, param_name in _CLI_PARAM_MAP.items():
        value = getattr(args, attr_name, None)
        if value is not None:
            overrides[param_name] = float(value)
    if not overrides:
        return

    # Resolve the active parameter order and current (resolved) values.
    try:
        active_names = list(config_manager.get_active_parameters())
        current: dict[str, float] = dict(config_manager.get_initial_parameters())
    except Exception:
        logger.warning(
            "Could not resolve active parameters; CLI --initial-* overrides skipped"
        )
        return

    applied: dict[str, float] = {}
    for name, val in overrides.items():
        if name in active_names:
            current[name] = val
            applied[name] = val
        else:
            logger.debug(
                "CLI override --initial %s not in the active set for this mode; ignored",
                name,
            )
    if not applied:
        logger.warning(
            "No CLI --initial-* overrides matched the active parameter set "
            "(%s); none applied",
            ", ".join(active_names),
        )
        return

    # Write the canonical initial_parameters block that
    # get_initial_parameters() reads back.
    values = [float(current[name]) for name in active_names if name in current]
    names_written = [name for name in active_names if name in current]
    ip = config.setdefault("initial_parameters", {})
    if not isinstance(ip, dict):  # pragma: no cover — defensive
        ip = {}
        config["initial_parameters"] = ip
    ip["parameter_names"] = names_written
    ip["values"] = values

    for name, val in applied.items():
        logger.info("CLI override: initial %s = %.6g", name, val)


__all__ = [
    "load_and_merge_config",
    "apply_cli_overrides",
    "resolve_output_dir",
]
