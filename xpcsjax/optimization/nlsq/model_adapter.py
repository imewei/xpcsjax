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

:class:`HeterodynePointEvaluator` is the ``two_component`` counterpart. The
heterodyne kernel (``compute_c2_heterodyne_pointwise``) is *index*-based and uses
*per-angle* scaling, so the heterodyne evaluator carries the static grids plus
per-angle ``contrast_arr``/``offset_arr`` and converts the Protocol's *value*
arguments into kernel indices inside ``eval_points`` (the value->index bridge).
This adapter is created here in Phase 2.1 but is **not** yet wired into the
stratification engine — that is a later task.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp

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


class HeterodynePointEvaluator:
    """Heterodyne (``two_component``) adapter over the pointwise kernel.

    Bridges the *value*-based :class:`PointEvaluator` Protocol onto the
    *index*-based ``compute_c2_heterodyne_pointwise`` kernel. The kernel wants
    integer indices into the static ``phi_unique`` / ``t`` grids plus *per-angle*
    ``contrast``/``offset`` arrays of shape ``(n_phi,)`` (one per unique phi),
    whereas the Protocol hands ``eval_points`` physical ``(phi, t1, t2)`` values
    per scattered point and *per-point* scaling.

    The evaluator resolves this by holding the static grids and the per-angle
    scaling at construction, then deriving ``phi_idx``/``t1_idx``/``t2_idx``
    inside :meth:`eval_points` (value -> nearest-grid-index). Because heterodyne
    scaling is per-angle (not per-point), the per-point ``contrast``/``offset``
    arguments of the Protocol are **deliberately ignored**: the evaluator carries
    its own ``contrast_arr``/``offset_arr`` and lets the kernel gather them by
    ``phi_idx``. This is the deliberate value->index bridge.
    """

    def __init__(
        self,
        *,
        analysis_mode: str,
        t: Any,
        q: float,
        dt: float,
        phi_unique: Any,
        contrast_arr: Any,
        offset_arr: Any,
    ) -> None:
        self.analysis_mode = analysis_mode
        self.t = jnp.asarray(t)
        self.q = q
        self.dt = dt
        self.phi_unique = jnp.asarray(phi_unique)
        # Per-angle scaling, shape (n_phi,) — gathered by phi_idx in the kernel.
        self.contrast_arr = jnp.asarray(contrast_arr)
        self.offset_arr = jnp.asarray(offset_arr)

    def eval_points(
        self,
        params: Any,
        phi: Any,
        t1: Any,
        t2: Any,
        contrast: Any,
        offset: Any,
    ) -> Any:
        """Bridge value ``(phi, t1, t2)`` to the index-based pointwise kernel.

        ``contrast``/``offset`` (per-point) are ignored on purpose — heterodyne
        scaling is per-angle, so the evaluator uses its own ``contrast_arr`` /
        ``offset_arr`` gathered by ``phi_idx`` inside the kernel.
        """
        # Local import mirrors the kernel module's own lazy-import discipline and
        # keeps the homodyne default path free of the heterodyne backend.
        from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne_pointwise

        phi_idx = self._nearest_index(jnp.asarray(phi), self.phi_unique)
        t1_idx = self._nearest_index(jnp.asarray(t1), self.t)
        t2_idx = self._nearest_index(jnp.asarray(t2), self.t)

        return compute_c2_heterodyne_pointwise(
            params,
            self.t,
            self.q,
            self.dt,
            phi_unique=self.phi_unique,
            phi_idx=phi_idx,
            t1_idx=t1_idx,
            t2_idx=t2_idx,
            contrast=self.contrast_arr,
            offset=self.offset_arr,
        )

    @staticmethod
    def _nearest_index(values: Any, grid: Any) -> Any:
        """Map each value to the int32 index of its nearest grid entry.

        Detector angles and grid times are discrete, so an exact value lands on
        its grid index; nearest-match makes the bridge robust to float noise in
        the scattered values without changing the exact-membership result.
        """
        # (P, 1) vs (1, G) -> (P, G) distance; argmin over the grid axis.
        dist = jnp.abs(values[:, None] - grid[None, :])
        return jnp.argmin(dist, axis=1).astype(jnp.int32)
