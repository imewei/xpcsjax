"""Model-agnostic point-evaluator adapter for the stratification engine.

Phase 1.1 extracts a thin seam between the homodyne stratification engine
(``StratifiedResidualFunctionJIT``) and the physics kernel it evaluates. The
engine should not hard-code ``compute_g2_scaled``; instead it calls
``evaluator.eval_points(...)`` on an injected :class:`PointEvaluator`.

The homodyne adapter, :class:`HomodynePointEvaluator`, is a byte-for-byte
pass-through to the real 9-arg
``compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt)`` kernel —
it merely re-maps the ``eval_points`` argument order ``(params, phi, t1, t2,
contrast, offset)`` onto the kernel's ``(params, t1, t2, phi, ...)`` order and
supplies the per-evaluator ``q``/``L``/``dt`` constants. Threading it through the
engine therefore preserves homodyne (``laminar_flow``) behavior exactly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from xpcsjax.core.physics_nlsq import compute_g2_scaled


@runtime_checkable
class PointEvaluator(Protocol):
    """Structural interface for a per-angle g2 point evaluator.

    Implementations map a physical-parameter vector plus per-angle scaling
    (contrast/offset) onto the scaled correlation surface over a ``(t1, t2)``
    grid for the given scattering angle(s). The stratification engine depends
    only on this surface, not on which physics kernel produced it.
    """

    def eval_points(
        self,
        params: Any,
        phi: Any,
        t1: Any,
        t2: Any,
        contrast: Any,
        offset: Any,
    ) -> Any:
        """Return the scaled g2 surface for ``phi`` over the ``(t1, t2)`` grid."""
        ...


class HomodynePointEvaluator:
    """Homodyne (``laminar_flow``) adapter over the 9-arg ``compute_g2_scaled``.

    Holds the per-fit constants (``q``, ``L``, ``dt``) so that the engine can
    call :meth:`eval_points` with only the parameters that vary per evaluation.
    This is a pure pass-through: the returned surface is bit-identical to calling
    ``compute_g2_scaled`` directly with the same arguments.
    """

    def __init__(self, *, analysis_mode: str, q: float, L: float, dt: float) -> None:
        self.analysis_mode = analysis_mode
        self.q = q
        self.L = L
        self.dt = dt

    def eval_points(
        self,
        params: Any,
        phi: Any,
        t1: Any,
        t2: Any,
        contrast: Any,
        offset: Any,
    ) -> Any:
        """Pass through to ``compute_g2_scaled`` with the kernel's arg order.

        Note the re-map: ``eval_points`` takes ``(phi, t1, t2, ...)`` but the
        kernel signature is ``(params, t1, t2, phi, q, L, contrast, offset, dt)``.
        """
        return compute_g2_scaled(
            params,
            t1,
            t2,
            phi,
            self.q,
            self.L,
            contrast,
            offset,
            self.dt,
        )
