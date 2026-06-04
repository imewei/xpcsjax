"""Shared assembler for the anti-degeneracy layer-activation diagnostics block.

Both the homodyne (``laminar_flow``) and heterodyne (``two_component``) result
builders call :func:`assemble_anti_degeneracy_diagnostics` so the two modes emit
the SAME top-level ``nlsq_diagnostics`` key set for the L2/L3/L4/L5 layers.

Diagnostics-only: nothing here reads or writes solve state, so routing a mode
through it never changes a fit value.
"""

from __future__ import annotations

from typing import Any

#: Keys both modes are guaranteed to surface at the top level of nlsq_diagnostics.
#: ``gradient_monitor`` is included only when the L4 block is provided.
CORE_KEYS = (
    "hierarchical_active",
    "regularization_active",
    "shear_weighting",
    "gradient_monitor",
)


def assemble_anti_degeneracy_diagnostics(
    *,
    hierarchical_active: bool,
    regularization_active: bool,
    shear_weighting: Any,
    gradient_monitor: dict[str, Any] | None = None,
    **layer_detail: Any,
) -> dict[str, Any]:
    """Build the symmetric anti-degeneracy diagnostics block.

    Always emits ``hierarchical_active`` / ``regularization_active`` /
    ``shear_weighting``. Emits ``gradient_monitor`` only when provided (omitted ->
    key absent, never a ``None`` value). Mode-specific per-layer detail
    (``hierarchical_stages``, ``regularization_mode``, ...) is merged verbatim via
    ``**layer_detail``. Pure and total: never raises on these inputs; same inputs
    -> same output.

    Parameters
    ----------
    hierarchical_active, regularization_active : bool
        Whether L2 / L3 actually ran on this fit (coerced to ``bool``).
    shear_weighting : Any
        Mode-appropriate L5 value: the ``"not_applicable_heterodyne"`` sentinel
        for heterodyne, or laminar's real L5 state/diagnostics.
    gradient_monitor : dict, optional
        The shared L4 ``gradient_monitor`` block, if available.
    **layer_detail
        Extra per-layer keys to merge verbatim (path-dependent, optional).
    """
    block: dict[str, Any] = {
        "hierarchical_active": bool(hierarchical_active),
        "regularization_active": bool(regularization_active),
        "shear_weighting": shear_weighting,
    }
    if gradient_monitor is not None:
        block["gradient_monitor"] = gradient_monitor
    block.update(layer_detail)
    return block
