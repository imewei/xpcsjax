"""Single source of truth for the resolved per-angle scaling mode (spec §4 Seam 1).

Collapses the per-angle scaling contract to exactly three RESOLVED variants —
``constant`` (frozen), ``averaged`` (one optimized pair), ``individual`` (per-angle
optimized) — and provides the sole owner of the ``constant_scaling_threshold``
default. ``auto`` is input sugar resolving to ``averaged``/``individual``. The
removed legacy tokens are rejected by the generic
``else`` branch (no special-case arm).

Phase 0: pure unit; no call site imports this yet.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

import numpy as np

PerAngleMode = Literal["constant", "averaged", "individual"]

DEFAULT_CONSTANT_SCALING_THRESHOLD = 3
"""Sole owner of the auto->averaged/individual cutover default (spec §4 Seam 1)."""

_RESOLVED: frozenset[str] = frozenset(get_args(PerAngleMode))


def resolve_per_angle_mode(
    token: str,
    n_phi: int,
    constant_scaling_threshold: int = DEFAULT_CONSTANT_SCALING_THRESHOLD,
) -> PerAngleMode:
    """Resolve a user/config ``per_angle_mode`` token to a canonical variant.

    Parameters
    ----------
    token : str
        One of ``"constant"``, ``"averaged"``, ``"individual"``, or ``"auto"``.
    n_phi : int
        Number of unique phi angles (only consulted for ``"auto"``).
    constant_scaling_threshold : int, optional
        ``auto`` resolves to ``"averaged"`` when ``n_phi >= threshold`` else
        ``"individual"``. Defaults to :data:`DEFAULT_CONSTANT_SCALING_THRESHOLD`.

    Returns
    -------
    PerAngleMode
        The resolved variant: ``"constant"``, ``"averaged"``, or ``"individual"``.

    Raises
    ------
    ValueError
        For any token other than the four accepted strings — including the
        removed legacy tokens (e.g. the old reparameterization aliases).
    """
    if token in _RESOLVED:
        return token  # type: ignore[return-value]
    if token == "auto":
        threshold = max(int(constant_scaling_threshold), 1)
        return "averaged" if n_phi >= threshold else "individual"
    raise ValueError(
        f"unknown per_angle_mode {token!r}; valid: "
        "constant, averaged, individual, auto"
    )


def n_optimized(mode: PerAngleMode, n_phi: int) -> int:
    """Return the number of OPTIMIZED scaling parameters for a resolved mode.

    ``constant`` -> 0 (frozen), ``averaged`` -> 2, ``individual`` -> ``2 * n_phi``.

    Raises
    ------
    ValueError
        If ``mode`` is not already a resolved variant (e.g. ``"auto"`` or a
        removed token); callers must resolve first via
        :func:`resolve_per_angle_mode`.
    """
    if mode == "constant":
        return 0
    if mode == "averaged":
        return 2
    if mode == "individual":
        return 2 * int(n_phi)
    raise ValueError(
        f"unknown per_angle_mode {mode!r}; valid: constant, averaged, individual"
    )


class PerAngleScalingPlan:
    """Model-agnostic per-angle scaling bookkeeping (spec §4 Seam 3).

    Centralizes the seed/expand logic that the four execution paths duplicate
    today, owning the per-angle scaling transforms:
    ``individual`` = identity reshape, ``constant``/``averaged`` = mean-collapse /
    broadcast. The residual functions stay model-specific (homodyne grid vs
    heterodyne meshgrid); this object only owns layout + expansion.

    Parameters
    ----------
    mode : PerAngleMode
        Resolved variant (``constant`` / ``averaged`` / ``individual``).
    n_phi : int
        Number of unique phi angles.
    n_physics : int
        Number of physical parameters.
    quantile_scaling : tuple[np.ndarray, np.ndarray]
        ``(contrast_per_angle, offset_per_angle)``, each shape ``(n_phi,)``, as
        produced by
        :func:`xpcsjax.optimization.nlsq.parameter_utils.compute_quantile_per_angle_scaling`.
        For ``averaged`` the per-angle quantiles are averaged into the x0 seed;
        for ``constant`` they are the frozen scaling broadcast in the residual;
        for ``individual`` they are the per-angle x0 seed.

    Attributes
    ----------
    freeze : bool
        ``True`` iff ``mode == "constant"``.
    group_indices : list[tuple[int, int]]
        L3 regularization groups within the scaling head (from the canonical mapper).
    """

    def __init__(
        self,
        mode: PerAngleMode,
        n_phi: int,
        n_physics: int,
        quantile_scaling: tuple[np.ndarray, np.ndarray],
    ) -> None:
        if mode not in ("constant", "averaged", "individual"):
            raise ValueError(
                f"unknown per_angle_mode {mode!r}; valid: constant, averaged, individual"
            )
        self.mode: PerAngleMode = mode
        self.n_phi = int(n_phi)
        self.n_physics = int(n_physics)
        contrast, offset = quantile_scaling
        self._contrast = np.asarray(contrast, dtype=np.float64)
        self._offset = np.asarray(offset, dtype=np.float64)
        if self._contrast.shape != (self.n_phi,) or self._offset.shape != (self.n_phi,):
            raise ValueError(
                f"quantile_scaling arrays must be shape ({self.n_phi},), got "
                f"{self._contrast.shape} and {self._offset.shape}"
            )

    @property
    def freeze(self) -> bool:
        """Return ``True`` iff ``mode == "constant"``."""
        return self.mode == "constant"

    @property
    def n_optimized(self) -> int:
        """Return the number of optimized scaling parameters."""
        return n_optimized(self.mode, self.n_phi)

    # --- Reconciliation accessors (consumed by Phase 1+2 / Phase 3) ---
    @property
    def n_scaling(self) -> int:
        """Alias of :attr:`n_optimized` (the scaling-head length)."""
        return self.n_optimized

    @property
    def frozen_contrast(self) -> np.ndarray:
        """Frozen per-angle contrast from the quantile estimate (``constant`` mode)."""
        return self._contrast.copy()

    @property
    def frozen_offset(self) -> np.ndarray:
        """Frozen per-angle offset from the quantile estimate (``constant`` mode)."""
        return self._offset.copy()

    @property
    def group_indices(self) -> list[tuple[int, int]]:
        """Return L3 regularization groups within the scaling head."""
        # Delegate to the canonical mapper so there is one boundary authority.
        from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper

        return ParameterIndexMapper.canonical(
            mode=self.mode, n_phi=self.n_phi, n_physics=self.n_physics
        ).group_indices

    def seed_tail(self) -> np.ndarray:
        """x0 scaling head seed from the quantile estimate.

        ``constant`` -> empty (frozen); ``averaged`` -> ``[mean(c), mean(o)]``;
        ``individual`` -> ``[c_0..c_{nφ-1}, o_0..o_{nφ-1}]``.
        """
        if self.mode == "constant":
            return np.empty(0, dtype=np.float64)
        if self.mode == "averaged":
            return np.array(
                [self._contrast.mean(), self._offset.mean()], dtype=np.float64
            )
        return np.concatenate([self._contrast, self._offset])

    def seed_bounds(
        self,
        contrast_bounds: tuple[float, float] = (0.0, 1.0),
        offset_bounds: tuple[float, float] = (0.5, 1.5),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper bounds for the scaling head, matching ``seed_tail()`` length.

        ``constant`` -> empty arrays; ``averaged`` -> length-2 ``[c, o]``;
        ``individual`` -> length ``2*n_phi`` (contrast block then offset block).
        Call sites MUST pass the contrast/offset bounds from the parameter
        registry/config wherever optimizer bounds are built — do NOT rely on the
        defaults. The defaults here mirror the real quantile bounds in
        ``compute_quantile_per_angle_scaling`` (``parameter_utils.py:374``): contrast
        ``(0.0, 1.0)`` and offset ``(0.5, 1.5)`` — NOT the loose ``(0, 2)`` a draft
        assumed (that would mis-bound the offset by 3x on the low side).
        """
        clo, chi = contrast_bounds
        olo, ohi = offset_bounds
        if self.mode == "constant":
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        if self.mode == "averaged":
            return (
                np.array([clo, olo], dtype=np.float64),
                np.array([chi, ohi], dtype=np.float64),
            )
        lb = np.concatenate([np.full(self.n_phi, clo), np.full(self.n_phi, olo)])
        ub = np.concatenate([np.full(self.n_phi, chi), np.full(self.n_phi, ohi)])
        return lb, ub

    def expand_tail(self, theta_tail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Dense per-angle ``(contrast[n_phi], offset[n_phi])`` from a scaling tail.

        ``constant`` -> broadcast the frozen quantiles (tail ignored);
        ``averaged`` -> broadcast the two scalars; ``individual`` -> identity reshape.
        """
        theta_tail = np.asarray(theta_tail, dtype=np.float64)
        if self.mode == "constant":
            return self._contrast.copy(), self._offset.copy()
        if self.mode == "averaged":
            if theta_tail.shape != (2,):
                raise ValueError(
                    f"expected scaling tail of length 2 for averaged, got {theta_tail.shape}"
                )
            return (
                np.full(self.n_phi, theta_tail[0]),
                np.full(self.n_phi, theta_tail[1]),
            )
        # individual
        if theta_tail.shape != (2 * self.n_phi,):
            raise ValueError(
                f"expected scaling tail of length {2 * self.n_phi} for individual, "
                f"got {theta_tail.shape}"
            )
        return theta_tail[: self.n_phi].copy(), theta_tail[self.n_phi :].copy()

    def expand_tail_jax(self, theta_tail: Any) -> tuple[Any, Any]:
        """JIT-SAFE variant of :meth:`expand_tail` for JAX-traced residual closures.

        Identical semantics, but pure ``jnp`` — it never calls ``np.asarray`` on the
        argument, so it can be invoked inside an NLSQ-traced residual without raising
        ``TracerArrayConversionError`` (the exact failure mode the existing
        the per-angle expansion helper was introduced to avoid;
        ``heterodyne_core.py:2400-2427``). ``constant`` broadcasts the frozen quantiles
        (converted to ``jnp`` constants at construction-equivalent time, NOT from the
        traced ``theta_tail``); ``averaged`` broadcasts the two traced head scalars;
        ``individual`` is an identity reshape of the traced head. No shape assertions
        (they would force concretization); the caller slices the head to ``n_scaling``.
        """
        import jax.numpy as jnp

        if self.mode == "constant":
            c = jnp.asarray(self._contrast, dtype=jnp.float64)
            o = jnp.asarray(self._offset, dtype=jnp.float64)
            return c, o
        if self.mode == "averaged":
            return (
                jnp.full((self.n_phi,), theta_tail[0]),
                jnp.full((self.n_phi,), theta_tail[1]),
            )
        # individual: head is [c_0..c_{nφ-1}, o_0..o_{nφ-1}]
        return theta_tail[: self.n_phi], theta_tail[self.n_phi :]

    def expand_back(
        self, popt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split a scaling-first optimizer vector into dense per-angle scaling + physics.

        Returns ``(contrast[n_phi], offset[n_phi], physics[n_physics])``. For
        ``constant`` the optimizer vector is physics-only; the frozen quantiles
        supply the per-angle scaling.
        """
        popt = np.asarray(popt, dtype=np.float64)
        n_opt = self.n_optimized
        expected = n_opt + self.n_physics
        if popt.shape != (expected,):
            raise ValueError(
                f"expected popt of length {expected} for mode {self.mode!r}, "
                f"got {popt.shape}"
            )
        scaling_tail = popt[:n_opt]
        physics = popt[n_opt:]
        contrast, offset = self.expand_tail(scaling_tail)
        return contrast, offset, physics

    def expand_covariance(self, pcov: np.ndarray) -> np.ndarray:
        """Expand the OPTIMIZER covariance to the DENSE scaling-first layout.

        Companion to :meth:`expand_back` (which is params-only). The optimizer solves a
        compressed vector (``averaged`` -> ``2 + n_physics``; ``constant`` ->
        ``n_physics``); the dense result is ``2*n_phi + n_physics``. This builds the
        dense ``(D, D)`` covariance with ``D = 2*n_phi + n_physics``:

        - ``individual``: identity (already dense) — returned unchanged.
        - ``averaged``: the single contrast (resp. offset) variance is replicated onto
          every per-angle contrast (resp. offset) diagonal entry, and the shared
          scalar's covariance with physics is replicated across the ``n_phi`` rows/cols
          (off-angle entries among replicated params take the shared scalar variance —
          consistent with the constrained model where those angles share one parameter).
        - ``constant``: the per-angle scaling rows/cols are injected with ZERO variance
          (frozen, not estimated); the physics block is copied verbatim. ``pcov`` in is
          the physics-only ``(n_physics, n_physics)`` matrix.

        ``pcov`` may be ``None`` (failed/global-escape result) -> returns ``None``.
        """
        if pcov is None:
            return None
        pcov = np.asarray(pcov, dtype=np.float64)
        if self.mode == "individual":
            return pcov
        n_phi, n_phys = self.n_phi, self.n_physics
        dense_d = 2 * n_phi + n_phys
        dense = np.zeros((dense_d, dense_d), dtype=np.float64)
        # physics tail block (last n_phys rows/cols in both layouts)
        dense[2 * n_phi :, 2 * n_phi :] = pcov[-n_phys:, -n_phys:]
        if self.mode == "averaged":
            # optimizer order: [c_avg, o_avg, *physics]; replicate scalar blocks.
            for blk, src in ((slice(0, n_phi), 0), (slice(n_phi, 2 * n_phi), 1)):
                var = pcov[src, src]
                dense[blk, blk] = var  # full block = shared variance (replicated)
                # scalar<->physics cross terms replicated across the n_phi rows/cols
                cross = pcov[src, 2:]            # (n_phys,)
                dense[blk, 2 * n_phi :] = cross  # broadcast over the block rows
                dense[2 * n_phi :, blk] = cross[:, None]
        # constant: scaling rows/cols stay zero (frozen) — nothing else to fill.
        return dense
