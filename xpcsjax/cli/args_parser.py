"""Argument parser for the xpcsjax CLI.

NLSQ-only by design (see project CLAUDE.md). All CMC / MCMC / Bayesian
flags from the upstream heterodyne CLI are intentionally absent.

Parameter override flags map to canonical names per
``parameter_registry._MODE_PARAMS``:

* ``static_anisotropic`` / ``static_isotropic`` — D0, alpha, D_offset
* ``laminar_flow`` — D0, alpha, D_offset, gamma_dot_t0, beta,
  gamma_dot_t_offset, phi0
* ``two_component`` — D0_ref/alpha_ref/D_offset_ref,
  D0_sample/alpha_sample/D_offset_sample, v0, v_beta, v_offset,
  f0..f3, phi0_het. The reference/sample transport params and phi0_het
  have no CLI flag (14 params is too many for the command line); set
  them in the YAML config. The v0/v_beta/v_offset/f0..f3 flags below DO
  apply to two_component.

Flags whose canonical name is not in the active mode's parameter set are
silently ignored by ``config_handling.apply_cli_overrides`` (it
intersects against ``ConfigManager.get_active_parameters()`` before
writing the canonical ``initial_parameters`` block).
"""

from __future__ import annotations

import argparse
from pathlib import Path

_VALID_MODES = ("static_anisotropic", "static_isotropic", "laminar_flow", "two_component")


def create_parser() -> argparse.ArgumentParser:
    """Build the xpcsjax CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="xpcsjax",
        description=(
            "xpcsjax — JAX-native NLSQ fitting for homodyne / heterodyne XPCS. "
            "Bayesian sampling is permanently out of scope for this package; "
            "use the upstream homodyne / heterodyne packages for that."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run NLSQ fit with a YAML config
  xpcsjax --config analysis.yaml

  # Override the output directory
  xpcsjax --config analysis.yaml --output ./results

  # Multistart NLSQ with 16 restarts
  xpcsjax --config analysis.yaml --multistart --multistart-n 16

  # Plot experimental data only (skip optimization)
  xpcsjax --config analysis.yaml --plot-experimental-data

  # Plot simulated C2 heatmaps from config parameters
  xpcsjax --config analysis.yaml --plot-simulated-data --phi-angles 0,45,90,135

Exit codes:
  0   Analysis completed and the optimizer converged (or no convergence
      check applies, e.g. plot-only runs).
  1   Unhandled exception during analysis. See log for traceback.
  2   Analysis ran but the optimizer did NOT converge. Output files are
      still written but the fit is not trustworthy.
  130 Interrupted by the user (Ctrl-C).
""",
    )

    # ------------------------------------------------------------------
    # Required: config path
    # ------------------------------------------------------------------
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to YAML configuration file (required).",
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory (overrides ``output.directory`` in YAML).",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "npz", "both"],
        default="both",
        help="Format for saved results (default: both).",
    )

    # ------------------------------------------------------------------
    # Mode / phi
    # ------------------------------------------------------------------
    parser.add_argument(
        "--mode",
        choices=_VALID_MODES,
        default=None,
        help=(
            "Force ``analysis_mode`` (overrides YAML). Must be one of "
            f"{_VALID_MODES}."
        ),
    )
    parser.add_argument(
        "--phi",
        type=float,
        nargs="+",
        default=None,
        help="Phi angles to analyze, in degrees (overrides config).",
    )

    # ------------------------------------------------------------------
    # NLSQ options
    # ------------------------------------------------------------------
    nlsq_group = parser.add_argument_group(
        "NLSQ options", "Solver and multistart controls."
    )
    # store_const with default=None (not store_true) so the override layer
    # can distinguish "user did not pass --multistart" (None → leave YAML
    # untouched) from "user passed it" (True). store_true's False default
    # would clobber a YAML ``multi_start.enable: true`` on every run.
    nlsq_group.add_argument(
        "--multistart",
        action="store_const",
        const=True,
        default=None,
        help="Enable LHS multistart for NLSQ (overrides YAML).",
    )
    nlsq_group.add_argument(
        "--no-multistart",
        dest="multistart",
        action="store_const",
        const=False,
        help="Explicitly disable multistart (overrides a YAML enable).",
    )
    nlsq_group.add_argument(
        "--multistart-n",
        type=int,
        default=None,
        help="Number of multistart restarts (default: from config, else 10).",
    )
    nlsq_group.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum trust-region iterations (overrides config).",
    )
    nlsq_group.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="NLSQ convergence tolerance (overrides config).",
    )

    # ------------------------------------------------------------------
    # Parameter overrides (highest precedence)
    # CLI > YAML > parameter_registry defaults.
    # Covers the union of all four modes; mode-incompatible flags are
    # silently ignored downstream.
    # ------------------------------------------------------------------
    param_group = parser.add_argument_group(
        "parameter overrides",
        "Override initial parameter values (highest precedence). "
        "Flags for parameters not used by the active mode are ignored.",
    )
    # Core transport (all modes)
    param_group.add_argument(
        "--initial-D0", type=float, default=None, metavar="VAL",
        help="Diffusion prefactor D0 [Å²/s^α].",
    )
    param_group.add_argument(
        "--initial-alpha", type=float, default=None, metavar="VAL",
        help="Transport exponent alpha.",
    )
    param_group.add_argument(
        "--initial-D-offset", type=float, default=None, metavar="VAL",
        help="Transport offset D_offset [Å²/s].",
    )
    # Laminar flow / two-component velocity
    param_group.add_argument(
        "--initial-gamma-dot-t0", type=float, default=None, metavar="VAL",
        help="Shear rate prefactor (laminar_flow).",
    )
    param_group.add_argument(
        "--initial-gamma-dot-t-offset", type=float, default=None, metavar="VAL",
        help="Shear rate offset (laminar_flow).",
    )
    param_group.add_argument(
        "--initial-beta", type=float, default=None, metavar="VAL",
        help="Velocity exponent beta (laminar_flow).",
    )
    param_group.add_argument(
        "--initial-v-beta", type=float, default=None, metavar="VAL",
        help="Velocity exponent v_beta (two_component).",
    )
    param_group.add_argument(
        "--initial-v0", type=float, default=None, metavar="VAL",
        help="Velocity prefactor v0 (two_component).",
    )
    param_group.add_argument(
        "--initial-v-offset", type=float, default=None, metavar="VAL",
        help="Velocity offset v_offset (two_component).",
    )
    # Angle parameters
    param_group.add_argument(
        "--initial-phi0", type=float, default=None, metavar="VAL",
        help="Flow angle offset phi0 [degrees].",
    )
    # Two-component Fourier amplitudes (per-angle fraction)
    param_group.add_argument(
        "--initial-f0", type=float, default=None, metavar="VAL",
        help="Sample fraction amplitude f0 (two_component).",
    )
    param_group.add_argument(
        "--initial-f1", type=float, default=None, metavar="VAL",
        help="Fourier coefficient f1 (two_component).",
    )
    param_group.add_argument(
        "--initial-f2", type=float, default=None, metavar="VAL",
        help="Fourier coefficient f2 (two_component).",
    )
    param_group.add_argument(
        "--initial-f3", type=float, default=None, metavar="VAL",
        help="Fourier coefficient f3 (two_component).",
    )
    # Note: per-angle scaling (contrast/offset) is not a single-value CLI
    # override — there is one pair per phi angle. Set scaling in the YAML
    # config's per-angle scaling block instead.

    # ------------------------------------------------------------------
    # Verbosity
    # ------------------------------------------------------------------
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors.",
    )

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU thread count for XLA (default: auto).",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable JIT compilation (for debugging only — much slower).",
    )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot", dest="plot", action="store_true", default=True,
        help="Generate plots after fitting (default).",
    )
    plot_group.add_argument(
        "--no-plot", dest="plot", action="store_false",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save fit-comparison plots to the output directory.",
    )
    parser.add_argument(
        "--plotting-backend",
        choices=["auto", "matplotlib", "datashader"],
        default="auto",
        help=(
            "Plotting backend: auto (Datashader if installed), "
            "matplotlib, or datashader (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--parallel-plots",
        action="store_true",
        help="Generate plots in parallel via multiprocessing (Datashader path).",
    )
    parser.add_argument(
        "--phi-angles",
        type=str,
        default=None,
        help=(
            "Comma-separated phi angles in degrees for simulated data "
            "(e.g. '0,45,90,135')."
        ),
    )

    # Standalone plot modes (skip optimization)
    parser.add_argument(
        "--plot-experimental-data",
        action="store_true",
        help="Plot experimental data for QC (skip optimization).",
    )
    parser.add_argument(
        "--plot-simulated-data",
        action="store_true",
        help="Plot simulated C2 heatmaps from config parameters (skip optimization).",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.3,
        help="Contrast for simulated data (default: %(default)s; requires --plot-simulated-data).",
    )
    parser.add_argument(
        "--offset-sim",
        type=float,
        default=1.0,
        help="Offset for simulated data (default: %(default)s; requires --plot-simulated-data).",
    )

    # ------------------------------------------------------------------
    # Version
    # ------------------------------------------------------------------
    _add_version_arg(parser)

    return parser


def _add_version_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--version`` with a best-effort version resolution."""
    try:
        import importlib.metadata as _md

        version = _md.version("xpcsjax")
    except Exception:  # pragma: no cover — uninstalled / dev tree
        try:
            from xpcsjax import __version__ as version
        except Exception:
            version = "unknown"
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version}",
    )


def validate_args(args: argparse.Namespace) -> list[str]:
    """Light validation pass. Returns non-fatal warning strings.

    Raises ``FileNotFoundError`` for unrecoverable issues (missing config).
    """
    warnings: list[str] = []

    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    if args.verbose > 0 and args.quiet:
        warnings.append("Both --verbose and --quiet specified; --quiet wins.")
        args.verbose = 0

    phi_angles_str: str | None = getattr(args, "phi_angles", None)
    if phi_angles_str is not None:
        try:
            [float(x.strip()) for x in phi_angles_str.split(",")]
        except ValueError:
            warnings.append(
                f"--phi-angles must be comma-separated numbers "
                f"(e.g. '0,45,90,135'); got: {phi_angles_str!r}"
            )

    if args.plot_experimental_data and args.plot_simulated_data:
        warnings.append(
            "Both --plot-experimental-data and --plot-simulated-data given; "
            "running both passes."
        )

    if args.multistart_n is not None and args.multistart_n <= 0:
        warnings.append(
            f"--multistart-n must be positive; got {args.multistart_n}. "
            "Falling back to config value."
        )
        args.multistart_n = None

    return warnings


__all__ = ["create_parser", "validate_args"]
