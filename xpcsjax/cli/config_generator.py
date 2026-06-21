"""Configuration file generator for xpcsjax NLSQ analysis.

Provides the ``xpcsjax-config`` console script for emitting populated
YAML configurations from xpcsjax's four mode-specific templates:

- ``static_anisotropic`` — 3-param diffusion with per-angle scaling
- ``static_isotropic``   — 3-param diffusion, single global scaling
- ``laminar_flow``       — 7-param diffusion + shear
- ``two_component``      — heterodyne (sample + reference) 14-param model

xpcsjax is NLSQ-only by design — Bayesian / CMC modes from the upstream
``heterodyne`` package are intentionally absent.

Template/generation logic lives in :mod:`xpcsjax.cli.config_template`; this
module re-exports the public symbols and keeps the argument-parser entry point
so existing callers are unaffected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xpcsjax.cli.config_template import (
    _MODE_TO_TEMPLATE,  # noqa: F401 — re-exported for callers of this facade
    _VALID_MODES,
    generate_config,
    get_template_path,
    interactive_builder,
    show_template,
    validate_config,
)
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "build_parser",
    "generate_config",
    "get_template_path",
    "interactive_builder",
    "main",
    "show_template",
    "validate_config",
]


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


def build_parser() -> argparse.ArgumentParser:
    """Public factory alias for the config-generator parser.

    Returns
    -------
    argparse.ArgumentParser
        The argument parser for the ``xpcsjax-config`` console script.
    """
    return _build_parser()


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
