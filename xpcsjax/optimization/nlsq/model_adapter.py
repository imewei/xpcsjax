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

:class:`HeterodynePointEvaluator` is the ``two_component`` counterpart and is
the exact structural parallel of the homodyne adapter. It is a thin pass-through
to the heterodyne **meshgrid** kernel
``compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset) -> (N, N)``,
re-mapping ``eval_points``'s ``(params, phi, t1, t2, contrast, offset)`` onto the
kernel's ``(params, t, q, dt, phi, contrast, offset)`` order. Exactly like
``HomodynePointEvaluator``, it returns the **full per-angle ``(n_t, n_t)``
meshgrid** for a single ``phi`` over the engine's time grid, and it **uses the
engine-supplied per-angle ``contrast``/``offset``** so the optimizer can drive
scaling (Phase 2.1 correction — the prior pointwise/index-based version returned
a diagonal and ignored the passed scaling, which is wrong for engine use).

Single-time-axis assumption: ``StratifiedResidualFunctionJIT`` evaluates a
two-time correlation over one time axis — it calls
``eval_points(..., self.t1_unique, self.t2_unique, ...)`` and then gathers from a
``(n_phi, n_t1, n_t2)`` grid. The heterodyne kernel takes a single time array
``t`` and returns ``(N, N)`` indexed ``[t1, t2]``; this adapter therefore uses
``t1`` as the kernel's time axis and returns ``(len(t1), len(t1))``. In this
engine ``t1_unique`` and ``t2_unique`` are the same time grid, so this is exact.
If a caller ever passed differing ``t1``/``t2`` grids, the heterodyne kernel has
no two-axis form to honour the distinction — ``t1`` is authoritative.
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


class HeterodynePointEvaluator:
    r"""Heterodyne (``two_component``) adapter over the meshgrid kernel.

    Exact structural parallel of :class:`HomodynePointEvaluator`: a thin
    pass-through to the heterodyne **meshgrid** kernel
    ``compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)`` which
    returns the full two-time ``(N, N)`` matrix indexed ``[t1, t2]``.

    The stratification engine (``StratifiedResidualFunctionJIT``) calls
    :meth:`eval_points` with a **scalar** ``phi``, the FULL ``t1``/``t2`` time
    grids, and a **scalar** per-angle ``contrast``/``offset`` sliced from the
    optimizer's parameter vector, then ``squeeze``\\s axis 0 and gathers from the
    resulting ``(n_phi, n_t, n_t)`` grid. So :meth:`eval_points` must (a) return
    the full per-angle ``(n_t, n_t)`` meshgrid and (b) **use** the supplied
    ``contrast``/``offset`` so the optimizer can drive scaling.

    Holds only the per-fit constants (``q``, ``dt``); ``analysis_mode`` is carried
    for parity with the homodyne adapter. There is no static ``contrast_arr`` /
    ``offset_arr`` — scaling is engine-supplied per call.
    """

    def __init__(self, *, analysis_mode: str, q: float, dt: float) -> None:
        self.analysis_mode = analysis_mode
        self.q = float(q)
        self.dt = float(dt)

    def eval_points(
        self,
        params: Any,
        phi: Any,
        t1: Any,
        t2: Any,
        contrast: Any,
        offset: Any,
    ) -> Any:
        """Pass through to ``compute_c2_heterodyne`` (meshgrid path).

        The engine supplies a scalar ``phi`` + full time grid (``t1``) + scalar
        per-angle ``contrast``/``offset``. The heterodyne two-time grid uses a
        single time axis, so ``t1`` IS the kernel's time array and the returned
        ``(n_t, n_t)`` matrix is indexed ``[t1, t2]`` — matching the engine's
        ``t1_idx * n_t2 + t2_idx`` gather. ``t2`` is unused (single-time-axis
        kernel; ``t1_unique == t2_unique`` in this engine).

        Shape contract: the homodyne kernel returns ``(1, n_t, n_t)`` for a
        scalar ``phi`` (a length-1 phi axis), and the engine peels it with
        ``squeeze(axis=0)``. The heterodyne meshgrid kernel returns a bare
        ``(n_t, n_t)`` matrix, so we prepend a length-1 phi axis here to keep the
        engine's ``squeeze(axis=0)`` contract identical across both adapters.
        """
        # Local import keeps the homodyne default path free of the heterodyne
        # backend, mirroring the kernel module's own lazy-import discipline.
        from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne

        del t2  # single time axis: t1 is authoritative (see class docstring)
        grid = compute_c2_heterodyne(
            params,
            t1,
            self.q,
            self.dt,
            phi,
            contrast,
            offset,
        )
        # Prepend the length-1 phi axis so the engine's squeeze(axis=0) yields
        # (n_t, n_t), matching HomodynePointEvaluator's (1, n_t, n_t) shape.
        return grid[None, :, :]


class HeterodynePointwiseEvaluator(HeterodynePointEvaluator):
    r"""Heterodyne adapter that ALSO evaluates only at scattered support points.

    Subclasses :class:`HeterodynePointEvaluator` so it still satisfies the grid
    :class:`PointEvaluator` Protocol (``eval_points`` returns the full per-angle
    ``(1, n_t, n_t)`` meshgrid, inherited unchanged). In addition it advertises
    ``supports_scattered = True`` and implements :meth:`eval_scattered`, which the
    stratification engine detects via ``getattr(evaluator, "supports_scattered",
    False)`` (a duck-typed capability — NOT a member of the ``PointEvaluator``
    Protocol, so homodyne's evaluator is unaffected).

    The scattered kernel ``compute_c2_heterodyne_pointwise`` gathers from the same
    per-t cumulative arrays as the meshgrid kernel, so for any point
    ``eval_scattered(...)[p]`` equals
    ``eval_points(..., phi_unique[phi_idx[p]], ...)[0][t1_idx[p], t2_idx[p]]`` to
    float precision (see ``heterodyne_jax_backend.py``). That is what makes the
    engine swap numerically equivalent rather than merely no-worse.
    """

    supports_scattered: bool = True

    def eval_scattered(
        self,
        params: Any,
        phi_unique: Any,
        t: Any,
        phi_idx: Any,
        t1_idx: Any,
        t2_idx: Any,
        contrast: Any,
        offset: Any,
    ) -> Any:
        """Return a flat ``(P,)`` theory vector at the scattered triples.

        ``contrast``/``offset`` are per-angle arrays of length ``n_phi`` (indexed
        internally by ``phi_idx``). ``t`` is the shared time grid (the engine's
        ``t1_unique``).
        """
        from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne_pointwise

        return compute_c2_heterodyne_pointwise(
            params,
            t,
            self.q,
            self.dt,
            phi_unique=phi_unique,
            phi_idx=phi_idx,
            t1_idx=t1_idx,
            t2_idx=t2_idx,
            contrast=contrast,
            offset=offset,
        )
