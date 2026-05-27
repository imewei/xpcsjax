"""Heterodyne joint multistart (Phase 1).

Wires the ``optimization.nlsq.multi_start`` config section into the heterodyne
(``two_component``) path. Mirrors the xpcsjax homodyne pattern
(``core.fit_nlsq_multistart``): each Latin-Hypercube start re-runs the whole
joint multi-phi fit, then a final authoritative fit from the winning start
produces the heterodyne ``OptimizationResult`` with full ``nlsq_diagnostics``.

Heterodyne multistart runs **sequentially**: the single-fit worker closes over a
JAX ``HeterodyneModel`` that is not process-picklable, so parallel workers
(``n_workers > 1``) are deferred to a follow-up.
"""
from __future__ import annotations

from typing import Any

from xpcsjax.optimization.nlsq.multistart import MultiStartConfig
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


def build_multistart_config(ms_dict: dict[str, Any]) -> MultiStartConfig:
    """Build a ``MultiStartConfig`` from the nested ``multi_start`` config dict.

    ``n_workers`` is clamped to 1 (sequential): the heterodyne single-fit worker
    closes over a JAX model that cannot be pickled for ``ProcessPoolExecutor``.
    """
    requested_workers = int(ms_dict.get("n_workers", 0) or 0)
    if requested_workers not in (0, 1):
        logger.warning(
            "Heterodyne multistart runs sequentially; ignoring n_workers=%d "
            "(parallel heterodyne multistart is a follow-up).",
            requested_workers,
        )
    return MultiStartConfig(
        enable=bool(ms_dict.get("enable", False)),
        n_starts=int(ms_dict.get("n_starts", 10)),
        seed=int(ms_dict.get("seed", 42)),
        sampling_strategy=str(ms_dict.get("sampling_strategy", "latin_hypercube")),
        n_workers=1,
        use_screening=bool(ms_dict.get("use_screening", True)),
        screen_keep_fraction=float(ms_dict.get("screen_keep_fraction", 0.5)),
        refine_top_k=int(ms_dict.get("refine_top_k", 3)),
        refinement_ftol=float(ms_dict.get("refinement_ftol", 1e-12)),
        degeneracy_threshold=float(ms_dict.get("degeneracy_threshold", 0.1)),
    )
