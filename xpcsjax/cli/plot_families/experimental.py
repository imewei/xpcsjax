"""Experimental-data plot family for the xpcsjax CLI.

Renders per-angle experimental C2 heatmaps (QC path — no fit required).
Heavy matplotlib imports are deferred to function bodies so this module does
not pull the plotting stack into the import graph for non-plotting CLI
invocations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from xpcsjax.utils.logging import get_logger, log_exception

logger = get_logger(__name__)


def _plot_experimental_data(data: dict[str, Any], plots_dir: Path) -> Path | None:
    """Render per-angle experimental C2 heatmaps for QC.

    Uses ``xpcsjax.viz.plot_nlsq_fit`` is unsuitable here (it requires a fit),
    so this routes through the single-panel ``plot_simulated_data`` entry
    point with the experimental array — the function plots any 2D C2 surface
    and annotates basic stats inline. One file per angle.
    """
    # Lazy: avoid pulling matplotlib into the import chain for non-plot commands.
    import matplotlib

    matplotlib.use("Agg")

    from xpcsjax.viz import plot_simulated_data

    c2_exp = np.asarray(data.get("c2_exp", data.get("c2")))
    phi_list = np.asarray(data.get("phi_angles_list", []), dtype=np.float64)
    t1 = data.get("t1")
    t2 = data.get("t2")

    if c2_exp.size == 0:
        logger.warning("No experimental c2 data to plot")
        return None

    if c2_exp.ndim == 2:
        c2_exp = c2_exp[np.newaxis, ...]

    for i in range(c2_exp.shape[0]):
        phi = float(phi_list[i]) if i < len(phi_list) else 0.0
        save_path = plots_dir / f"experimental_data_phi{int(round(phi))}.png"
        try:
            plot_simulated_data(
                c2_exp[i],
                t=np.asarray(t1) if t1 is not None else None,
                t2=np.asarray(t2) if t2 is not None else None,
                phi_deg=phi,
                save_path=save_path,
                title="Experimental C₂(t₁, t₂)",
            )
        except Exception as exc:
            log_exception(
                logger,
                exc,
                context={"operation": "plot_experimental_data", "phi": phi},
                level=logging.WARNING,
            )

    return plots_dir
