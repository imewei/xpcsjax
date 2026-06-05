"""Configuration file generator for xpcsjax NLSQ analysis.

Provides the ``xpcsjax-config`` console script for emitting populated
YAML configurations from xpcsjax's four mode-specific templates:

- ``static_anisotropic`` — 3-param diffusion with per-angle scaling
- ``static_isotropic``   — 3-param diffusion, single global scaling
- ``laminar_flow``       — 7-param diffusion + shear
- ``two_component``      — heterodyne (sample + reference) 14-param model

xpcsjax is NLSQ-only by design — Bayesian / CMC modes from the upstream
``heterodyne`` package are intentionally absent.
"""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from xpcsjax.config.manager import ConfigManager
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "generate_config",
    "get_template_path",
    "interactive_builder",
    "main",
    "show_template",
    "validate_config",
]


# -----------------------------------------------------------------------------
# Mode → template-filename map. These are the four production templates
# xpcsjax ships under ``xpcsjax/config/templates/``.
# -----------------------------------------------------------------------------
_MODE_TO_TEMPLATE: dict[str, str] = {
    "static_anisotropic": "xpcsjax_static_anisotropic.yaml",
    "static_isotropic": "xpcsjax_static_isotropic.yaml",
    "laminar_flow": "xpcsjax_laminar_flow.yaml",
    "two_component": "xpcsjax_two_component.yaml",
}

_VALID_MODES: tuple[str, ...] = tuple(_MODE_TO_TEMPLATE.keys())


def get_template_path(mode: str) -> Path:
    """Return the filesystem path to the YAML template for *mode*.

    Parameters
    ----------
    mode : str
        One of ``static_anisotropic``, ``static_isotropic``,
        ``laminar_flow``, or ``two_component``.

    Returns
    -------
    pathlib.Path
        Path to the mode-specific template YAML file shipped under
        ``xpcsjax/config/templates/``.

    Raises
    ------
    ValueError
        If *mode* is not one of the four supported modes.
    FileNotFoundError
        If the template file is missing from the installed package.
    """
    if mode not in _MODE_TO_TEMPLATE:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {', '.join(_VALID_MODES)}")

    filename = _MODE_TO_TEMPLATE[mode]
    # importlib.resources.files returns a Traversable; cast through Path for
    # downstream filesystem ops. Templates ship as real files inside the
    # installed package (no zip-import quirks expected for xpcsjax).
    template_path = Path(str(files("xpcsjax.config") / "templates" / filename))

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found for mode '{mode}': {template_path}")

    return template_path


def generate_config(
    mode: str,
    output_path: Path | str,
    overwrite: bool = False,
    data_path: str | None = None,
    q: float | None = None,
    dt: float | None = None,
    time_length: int | None = None,
) -> Path:
    """Generate a populated configuration file from a mode-specific template.

    Copies the template for *mode* and applies string-level substitutions for
    any provided data path / scattering / timing values, then writes the
    result to *output_path*.

    Parameters
    ----------
    mode : str
        Analysis mode -- one of ``static_anisotropic``, ``static_isotropic``,
        ``laminar_flow``, or ``two_component``.
    output_path : pathlib.Path or str
        Destination YAML path.
    overwrite : bool, optional
        If ``True``, replace an existing file at *output_path*.
    data_path : str or None, optional
        If provided, injected as the ``file_path`` entry.
    q : float or None, optional
        If provided, injected as ``scattering.wavevector_q``.
    dt : float or None, optional
        If provided, injected as ``analyzer_parameters.dt``.
    time_length : int or None, optional
        If provided, injected as ``end_frame`` (last frame, inclusive).

    Returns
    -------
    pathlib.Path
        Path to the generated config file.

    Raises
    ------
    ValueError
        For an unknown *mode*.
    FileExistsError
        If *output_path* exists and *overwrite* is ``False``.

    Notes
    -----
    Substitution is by exact placeholder match against the canonical template
    values; a missing placeholder is warned about (not raised) and skipped.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {', '.join(_VALID_MODES)}")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {output_path}. Use --overwrite to replace.")

    template_path = get_template_path(mode)
    with open(template_path, encoding="utf-8") as f:
        content = f.read()

    # ------------------------------------------------------------------
    # String-level substitutions against the canonical placeholder values
    # used in every xpcsjax template (verified across all 4 templates):
    #   file_path: null            wavevector_q: 0.0237
    #   dt: 0.1                    end_frame: 1000
    # ------------------------------------------------------------------
    import yaml

    substitutions: list[tuple[str, str]] = []

    if data_path is not None:
        safe = yaml.dump(data_path, default_flow_style=True).strip()
        substitutions.append(("file_path: null", f"file_path: {safe}"))

    if q is not None:
        safe = yaml.dump(q, default_flow_style=True).strip()
        substitutions.append(("wavevector_q: 0.0237", f"wavevector_q: {safe}"))

    if dt is not None:
        safe = yaml.dump(dt, default_flow_style=True).strip()
        substitutions.append(("dt: 0.1", f"dt: {safe}"))

    if time_length is not None:
        safe = yaml.dump(time_length, default_flow_style=True).strip()
        substitutions.append(("end_frame: 1000", f"end_frame: {safe}"))

    for placeholder, replacement in substitutions:
        if placeholder not in content:
            logger.warning(
                "Placeholder '%s' not found in template for mode '%s'",
                placeholder,
                mode,
            )
        content = content.replace(placeholder, replacement, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Generated xpcsjax configuration: %s (mode=%s)", output_path, mode)
    return output_path


def show_template(mode: str) -> None:
    """Print the contents of the template for *mode* to stdout.

    Parameters
    ----------
    mode : str
        One of the four supported analysis modes.

    Raises
    ------
    ValueError
        If *mode* is not supported.
    FileNotFoundError
        If the template file is missing from the package.
    """
    template_path = get_template_path(mode)
    with open(template_path, encoding="utf-8") as f:
        sys.stdout.write(f.read())


def validate_config(config_path: Path | str) -> bool:
    """Validate an existing YAML configuration file.

    Parses the YAML and attempts to construct a
    :class:`~xpcsjax.config.manager.ConfigManager` against it, printing
    progress and any failure reason to stdout.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Path to a YAML configuration file.

    Returns
    -------
    bool
        ``True`` if the file exists, parses as YAML, and constructs a valid
        ``ConfigManager``; ``False`` otherwise.
    """
    config_path = Path(config_path)
    print(f"Validating: {config_path}")

    if not config_path.exists():
        print(f"ERROR: File not found: {config_path}")
        return False

    # Parse YAML first to give a clean error for syntactic issues before
    # ConfigManager's heavier structural checks run.
    import yaml

    try:
        with open(config_path, encoding="utf-8") as f:
            yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(f"ERROR: Failed to parse YAML: {exc}")
        return False
    except OSError as exc:
        print(f"ERROR: Failed to read file: {exc}")
        return False

    try:
        ConfigManager(str(config_path))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        logger.error("Structural validation failed: %s", exc)
        print(f"Structural validation failed: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 — ConfigManager may raise custom types
        logger.error("Structural validation failed: %s", exc)
        print(f"Structural validation failed: {exc}")
        return False

    print("Result: VALID")
    return True


# -----------------------------------------------------------------------------
# Interactive builder (lightweight). Returns a config dict — the caller is
# responsible for serializing it. Kept simple; for production runs users
# should edit a generated template directly.
# -----------------------------------------------------------------------------
def _prompt(
    label: str,
    default: str,
    *,
    required: bool = False,
    cast: type | None = None,
) -> Any:
    """Prompt the user for a value, looping until a valid answer is given.

    Parameters
    ----------
    label : str
        Display label shown before the input.
    default : str
        Default value used when the answer is empty (and not required); shown
        in brackets.
    required : bool, optional
        If ``True``, an empty answer is rejected and re-prompted.
    cast : type or None, optional
        If given, the response is cast to this type; a failed cast re-prompts.

    Returns
    -------
    Any
        The user-supplied (or default) value, cast to *cast* when provided.
    """
    while True:
        suffix = f" [{default}]" if default and not required else ""
        raw = input(f"{label}{suffix}: ").strip()

        if not raw:
            if required:
                print("  This field is required.")
                continue
            raw = default

        if cast is not None:
            try:
                return cast(raw)
            except (ValueError, TypeError):
                print(f"  Invalid value. Expected {cast.__name__}.")
                continue

        return raw


def interactive_builder(mode: str) -> dict[str, Any]:
    """Build a minimal configuration dict via interactive prompts.

    Parameters
    ----------
    mode : str
        Target analysis mode, recorded under ``analysis_mode``.

    Returns
    -------
    dict
        Configuration dictionary suitable for :func:`yaml.safe_dump`.

    Raises
    ------
    ValueError
        If *mode* is not one of the four supported modes.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {', '.join(_VALID_MODES)}")

    print(f"=== xpcsjax Config Builder (mode={mode}) ===\n")

    data_path = _prompt("Data file path", "", required=True)
    q = _prompt("Wavevector q [Å⁻¹]", "0.0237", cast=float)
    dt = _prompt("Time step dt [seconds]", "0.1", cast=float)
    start_frame = _prompt("Starting frame (1-indexed)", "1", cast=int)
    end_frame = _prompt("Ending frame (inclusive)", "1000", cast=int)

    phi_raw = _prompt("Phi angles (comma-separated, degrees)", "0.0")
    try:
        phi_angles = [float(p.strip()) for p in phi_raw.split(",") if p.strip()]
    except ValueError:
        print("  Invalid phi angles, using default [0.0].")
        phi_angles = [0.0]

    output_dir = _prompt("Output directory", "./output")

    scattering: dict[str, Any] = {"wavevector_q": q}
    if phi_angles:
        scattering["phi_angles"] = phi_angles

    config: dict[str, Any] = {
        "metadata": {
            "config_version": "0.1.0",
            "description": f"xpcsjax {mode} — interactive build",
        },
        "analysis_mode": mode,
        "analyzer_parameters": {
            "dt": dt,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "scattering": scattering,
        },
        "experimental_data": {
            "file_path": data_path,
        },
        "optimization": {
            "method": "nlsq",
        },
        "output": {
            # Canonical key per the shipped templates is ``directory``
            # (read by the CLI's output resolver), not ``output_dir``.
            "directory": output_dir,
        },
    }

    logger.info("Interactive config built (mode=%s)", mode)
    print("\nConfiguration built successfully.")
    return config


# -----------------------------------------------------------------------------
# Argparse entry point
# -----------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``xpcsjax-config`` console script.

    Returns
    -------
    argparse.ArgumentParser
        Parser exposing ``--mode``, ``--output``, data/scattering/timing
        injection flags, and the ``--show-template`` / ``--validate`` /
        ``--interactive`` action flags.
    """
    parser = argparse.ArgumentParser(
        prog="xpcsjax-config",
        description=(
            "Generate xpcsjax configuration files from mode-specific templates. "
            "xpcsjax is NLSQ-only; Bayesian / CMC modes are out of scope."
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="static_anisotropic",
        choices=list(_VALID_MODES),
        help=("Analysis mode (default: static_anisotropic). Selects which template to populate."),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("xpcsjax_config.yaml"),
        help="Output path for configuration file (default: xpcsjax_config.yaml)",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="Path to experimental data file (injected as file_path)",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=None,
        help="Wavevector magnitude [Å⁻¹]",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step between frames [seconds]",
    )
    parser.add_argument(
        "--time-length",
        type=int,
        default=None,
        help="Number of frames (injected as end_frame, inclusive)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    parser.add_argument(
        "--show-template",
        action="store_true",
        help="Print template contents to stdout and exit (no file written)",
    )
    parser.add_argument(
        "--validate",
        "-V",
        action="store_true",
        help="Validate an existing config file (path taken from --output)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive config builder",
    )

    return parser


def main() -> None:
    """Run the ``xpcsjax-config`` console script.

    Parses arguments and dispatches to one of: print a template
    (``--show-template``), validate an existing config (``--validate``), run
    the interactive builder (``--interactive``), or generate a config from a
    template (the default). On user-facing errors, prints a message and exits
    with status 1 via :class:`SystemExit`.

    Examples
    --------
    Generate a laminar-flow config (typically invoked as the ``xpcsjax-config``
    console script):

    >>> from xpcsjax.cli.config_generator import main
    >>> main()  # doctest: +SKIP
    Created: xpcsjax_config.yaml
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.show_template:
        try:
            show_template(args.mode)
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}")
            raise SystemExit(1) from exc
        return

    if args.validate:
        is_valid = validate_config(args.output)
        raise SystemExit(0 if is_valid else 1)

    if args.interactive:
        try:
            config = interactive_builder(args.mode)
        except (ValueError, KeyboardInterrupt) as exc:
            print(f"\nAborted: {exc}")
            raise SystemExit(1) from exc

        output_path = Path(args.output)
        if output_path.exists() and not args.overwrite:
            print(f"Error: File exists: {output_path}. Use --overwrite to replace.")
            raise SystemExit(1)

        import yaml

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Created: {output_path}")
        return

    try:
        output = generate_config(
            mode=args.mode,
            output_path=args.output,
            overwrite=args.overwrite,
            data_path=args.data,
            q=args.q,
            dt=args.dt,
            time_length=args.time_length,
        )
        print(f"Created: {output}")
    except FileExistsError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
