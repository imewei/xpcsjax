"""Headless post-fit plotting service.

Argparse-free core of ``cli.plot_dispatch._generate_post_fit_plots``. Worker-side
(imports ``xpcsjax.viz``); forces the Matplotlib ``Agg`` backend so rendering in a
child process never tries to grab a Qt backend. Do NOT import from the GUI process
and do NOT re-export from ``xpcsjax.service.__init__``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xpcsjax.utils.logging import get_logger, log_exception

if TYPE_CHECKING:
    from xpcsjax.config.manager import ConfigManager
    from xpcsjax.optimization.nlsq.results import OptimizationResult

logger = get_logger(__name__)


def _generate_nlsq_plots(**kwargs: Any) -> Any:
    """Indirection over ``xpcsjax.viz.generate_nlsq_plots`` (monkeypatch seam).

    Imported lazily so importing this module does not pull the viz stack until
    a plot is actually requested.
    """
    from xpcsjax.viz import generate_nlsq_plots

    return generate_nlsq_plots(**kwargs)


def generate_plots(
    result: OptimizationResult,
    data: dict[str, Any],
    config_manager: ConfigManager,
    plots_dir: Path,
    *,
    use_datashader: bool = False,
    parallel: bool = False,
) -> Path | None:
    """Render the full post-fit artifact set under ``plots_dir``.

    Parameters
    ----------
    result : OptimizationResult
        The completed NLSQ fit result to visualize.
    data : dict
        The XPCS data dict the fit was run against (``c2_exp`` / ``t1`` / ``t2``
        / ``phi_angles_list``).
    config_manager : ConfigManager
        Active config; supplies the physics model (``get_model``) and the merged
        config dict (``get_config``).
    plots_dir : pathlib.Path
        Directory the artifacts are written under.
    use_datashader : bool, default False
        Use the Datashader fast path for large two-time maps.
    parallel : bool, default False
        Render per-angle artifacts in parallel processes.

    Returns
    -------
    pathlib.Path or None
        ``plots_dir`` on success, or ``None`` if the model cannot be built or
        rendering raises. Failures are logged at WARNING and never re-raised
        (mirrors the legacy ``_generate_post_fit_plots`` helper).
    """
    import matplotlib

    matplotlib.use("Agg")

    try:
        model = config_manager.get_model()
    except Exception as exc:
        log_exception(
            logger, exc, context={"operation": "post_fit_plots_get_model"}, level=logging.WARNING
        )
        return None

    cfg = config_manager.get_config()

    try:
        _generate_nlsq_plots(
            model=model,
            result=result,
            data=data,
            config=cfg,
            output_dir=plots_dir,
            use_datashader=use_datashader,
            parallel=parallel,
        )
    except Exception as exc:
        log_exception(
            logger, exc, context={"operation": "generate_nlsq_plots"}, level=logging.WARNING
        )
        return None

    return plots_dir
