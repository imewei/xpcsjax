"""Backend and directory resolution helpers for the xpcsjax CLI plot pipeline.

Extracted from ``plot_dispatch`` so the shared call counter, run-id accessor,
output-directory resolver, and backend flag translator can be imported by
multiple plot modules without pulling in the full dispatch fan-out.

Public surface:
    _PLOT_DISPATCH_CALL_COUNTER  — monotonic per-dispatch token (itertools.count)
    _current_run_id()            — active run_id from log context, or None
    resolve_plots_dir(args, config_manager) -> Path
    should_use_datashader(backend) -> bool
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xpcsjax.utils.logging import _LOG_CONTEXT, get_logger

if TYPE_CHECKING:
    from xpcsjax.config import ConfigManager

logger = get_logger(__name__)


# Monotonic per-call token so the per-phi ``log_once`` keys never collapse
# across separate dispatch-function calls when ``run_id`` is None (no
# configured log context). Without it, two successive calls would share a
# static ``"None:..."`` key in
# the process-global dedup cache and the second call's per-angle warning would
# be silently suppressed. Keeping run_id in the key still scopes by run when
# one is set; the token scopes by dispatch-function call.
_PLOT_DISPATCH_CALL_COUNTER = itertools.count()


def _current_run_id() -> str | None:
    """Read the active ``run_id`` from the log-context registry, if any.

    Used to scope ``log_once`` rate-limit keys per analysis run so a per-angle
    render failure logged once in one fit does not stay silenced for later fits
    in the same long-lived process. Returns ``None`` when no run is in context.
    """
    ctx = _LOG_CONTEXT.get() or {}
    return ctx.get("run_id")


def resolve_plots_dir(args: Any, config_manager: ConfigManager | None) -> Path:
    """Resolve the directory where plots will be written.

    The output ROOT is resolved by the shared
    :func:`xpcsjax.cli.config_handling.resolve_output_dir` — the same resolver
    used by result saving — so plots land under the configured output tree
    (``output.directory`` / ``output.base_directory``, or the legacy
    ``output_settings.output_dir``) rather than the process cwd. Falls back to
    the current working directory only when nothing is configured.

    A ``plots/`` subdirectory is created beneath the resolved root.
    """
    # Local import keeps the matplotlib-free CLI import graph intact and
    # avoids a circular import (commands -> plot_dispatch).
    from xpcsjax.cli.config_handling import resolve_output_dir

    root = resolve_output_dir(args, config_manager)
    if root is None:
        root = Path(".")
    plots_dir = Path(root) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def should_use_datashader(backend: str | None) -> bool:
    """Translate ``--plotting-backend`` to the ``use_datashader`` boolean.

    "auto" lets the viz layer probe optional deps; we forward True and let
    ``xpcsjax.viz.nlsq_plots`` fall back to matplotlib if Datashader is
    unavailable.
    """
    if backend in (None, "auto", "datashader"):
        return True
    return False
