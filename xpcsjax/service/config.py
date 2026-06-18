"""Headless config-loading service.

Argparse-free core of ``cli.config_handling.load_and_merge_config`` (mode +
output-directory overrides). JAX-free: imports only ``ConfigManager``; safe to
call from the GUI process in-process.
"""

from __future__ import annotations

from pathlib import Path

from xpcsjax.config.manager import ConfigManager
from xpcsjax.utils.logging import get_logger

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
