"""Main entry point for xpcsjax CLI.

NLSQ-only by design (see project CLAUDE.md). Bayesian / MCMC paths are
permanently out of scope.

The xpcsjax top-level package (``xpcsjax/__init__.py``) already configures
``JAX_ENABLE_X64``, ``XLA_FLAGS``, and ``NLSQ_SKIP_GPU_CHECK`` *before* any
JAX import. We do NOT duplicate that setup here, but we DO accept
``--threads`` / ``--no-jit`` and inject those into ``XLA_FLAGS`` before
the package-level JAX import fires — that's what ``_bootstrap_xla_env``
does.
"""

from __future__ import annotations

import os
import sys
import time


def _bootstrap_xla_env(argv: list[str] | None) -> None:
    """Seed XLA/JAX env vars from ``--threads`` / ``--no-jit`` before JAX loads.

    Pre-parses the thread-count and disable-JIT flags out of ``argv`` and
    writes the corresponding environment variables, which must happen *before*
    ``xpcsjax/__init__.py`` runs (it eagerly imports JAX).

    Parameters
    ----------
    argv : list of str or None
        Argument list to scan. ``None`` falls back to ``sys.argv[1:]``.

    Notes
    -----
    JAX reads ``XLA_FLAGS`` exactly once during backend initialization; any
    configuration written after the first ``import jax`` is silently ignored.
    The xpcsjax package init triggers that first import, so thread-count and
    disable-JIT flags must be set in ``os.environ`` before
    ``xpcsjax.cli.args_parser`` is imported.
    """
    raw = list(sys.argv[1:] if argv is None else argv)

    # CPU-only (per project CLAUDE.md: GPU support is out of scope)
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    threads: int | None = None
    no_jit = False
    i = 0
    while i < len(raw):
        token = raw[i]
        if token == "--threads" and i + 1 < len(raw):
            try:
                threads = int(raw[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if token.startswith("--threads="):
            try:
                threads = int(token.split("=", 1)[1])
            except ValueError:
                pass
            i += 1
            continue
        if token == "--no-jit":
            no_jit = True
        i += 1

    if threads is not None:
        existing = os.environ.get("XLA_FLAGS", "")
        tflags = f"--xla_cpu_multi_thread_eigen=true --intra_op_parallelism_threads={threads}"
        if tflags not in existing:
            os.environ["XLA_FLAGS"] = f"{existing} {tflags}".strip()
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    if no_jit:
        os.environ["JAX_DISABLE_JIT"] = "1"


def main(argv: list[str] | None = None) -> int:
    """Run the xpcsjax NLSQ analysis CLI end to end.

    Bootstraps XLA env vars, parses and validates arguments, configures
    logging, then dispatches the requested command (NLSQ fit or standalone
    plotting).

    Parameters
    ----------
    argv : list of str or None, optional
        Argument list; defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on an uncaught exception (or argument
        validation failure), 2 on NLSQ non-convergence, 130 on
        :class:`KeyboardInterrupt`.

    Examples
    --------
    Run a fit from a generated config (typically invoked via the ``xpcsjax``
    console script):

    >>> from xpcsjax.cli.main import main
    >>> main(["--config", "xpcsjax_config.yaml"])  # doctest: +SKIP
    0
    """
    _bootstrap_xla_env(argv)

    import logging as _logging

    # Match heterodyne: silence noisy backend-init messages
    _logging.getLogger("jax._src.xla_bridge").setLevel(_logging.ERROR)
    _logging.getLogger("jax._src.compiler").setLevel(_logging.ERROR)

    from xpcsjax.cli.args_parser import create_parser, validate_args

    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        warnings = validate_args(args)
        for warn in warnings:
            print(f"Warning: {warn}", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    from xpcsjax.cli.commands import dispatch_command
    from xpcsjax.utils.logging import configure_logging, get_logger, log_exception

    # Bootstrap console logging before the config file is parsed, so early
    # records (config loading) render in xpcsjax's format and stop propagating
    # to the root logger. A bare ``logging_config=None`` short-circuits in
    # configure_from_dict and would leave those records to nlsq's import-time
    # root StreamHandler (bare ``LEVEL:name:msg``). dispatch_command later
    # reconfigures from the config's ``logging:`` section (force=True).
    configure_logging(
        logging_config={"enabled": True, "console": {"enabled": True}},
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
    )
    logger = get_logger(__name__)

    start_time = time.perf_counter()

    try:
        exit_code = dispatch_command(args)
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        log_exception(logger, e, context={"command": "main"})
        return 1

    elapsed = time.perf_counter() - start_time
    if not args.quiet:
        logger.info("Analysis completed in %.1f seconds", elapsed)

    return exit_code


def main_xjexp() -> int:
    """Run the ``xjexp`` console script: plot experimental data only.

    Forwards to :func:`main` with ``--plot-experimental-data`` prepended,
    skipping optimization.

    Returns
    -------
    int
        Exit code from :func:`main`.
    """
    return main(["--plot-experimental-data", *sys.argv[1:]])


def main_xjsim() -> int:
    """Run the ``xjsim`` console script: plot simulated data only.

    Forwards to :func:`main` with ``--plot-simulated-data`` prepended,
    skipping optimization.

    Returns
    -------
    int
        Exit code from :func:`main`.
    """
    return main(["--plot-simulated-data", *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
