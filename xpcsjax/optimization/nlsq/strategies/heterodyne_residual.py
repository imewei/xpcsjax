"""Heterodyne stratified residual function (Phase 2 hybrid-streaming).

Mirrors the structure of :class:`StratifiedResidualFunction` in ``residual.py``
but calls the heterodyne kernel :func:`compute_c2_heterodyne` instead of the
homodyne :func:`compute_g2_scaled`.

Key differences from the homodyne version:
- ``compute_c2_heterodyne(full_params, t, q, dt, phi_angle, contrast, offset)``
  takes the **full** 14-parameter vector and returns the complete ``(N, N)``
  two-time matrix; there is no separate t1/t2 grid vmap needed.
- Per-angle scaling (contrast, offset) comes from ``model.scaling.get_for_angle``
  rather than being prepended to the optimizer parameter vector.
- params vector contains ONLY the varying physics parameters (same layout as
  ``model.param_manager.get_initial_values()``).
- Diagonal entries (t1 == t2) are masked to zero residual, matching homodyne.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne
from xpcsjax.optimization.nlsq.strategies.stratified_ls import create_stratified_chunks
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel


class StratifiedHeterodyneResidualFunction:
    """Residual function for heterodyne XPCS that respects angle-stratified chunks.

    Mirrors :class:`StratifiedResidualFunction` but replaces the homodyne
    ``compute_g2_scaled`` kernel with ``compute_c2_heterodyne``.

    The optimizer ``params`` vector contains ONLY the varying physics parameters
    (not prepended with contrast/offset).  Per-angle scaling is read from
    ``model.scaling`` at construction time and held fixed during the residual
    evaluation.

    Args:
        stratified_data: :class:`HeterodyneStratifiedData` (or any object with the
            flat stratified layout fields accepted by ``create_stratified_chunks``).
        model:            Configured :class:`HeterodyneModel` providing ``t``, ``q``,
                          ``dt``, ``param_manager``, and ``scaling``.
        per_angle_scaling: If True, per-angle contrast/offset are read from
                          ``model.scaling.get_for_angle(angle_idx)`` for each
                          chunk.  If False, angle 0 scaling is used for all chunks.
        physical_param_names: Names of the *varying* physics parameters in ``params``
                          order (typically ``model.param_manager.varying_names``).
        logger:           Optional logger.

    Call signature:
        ``fn(params: np.ndarray) -> np.ndarray``

        ``params`` has shape ``(n_varying_physics,)`` — the same layout as
        ``model.param_manager.get_initial_values()``.

        Returns a 1-D float64 array of weighted residuals concatenated across
        all angle chunks.  Diagonal entries (``t1 == t2``) are zeroed out.
    """

    def __init__(
        self,
        stratified_data: Any,
        model: HeterodyneModel,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.per_angle_scaling = per_angle_scaling
        self.physical_param_names = physical_param_names
        self.logger = logger or get_logger(__name__)

        # Build chunks from the flat stratified layout (same chunker as homodyne)
        chunked = create_stratified_chunks(stratified_data, target_chunk_size=100_000)
        self.chunks = chunked.chunks

        if not self.chunks:
            raise ValueError("stratified_data produced no chunks")

        self.n_chunks = len(self.chunks)
        self.n_total_points = sum(len(chunk.g2) for chunk in self.chunks)

        # ------------------------------------------------------------------ #
        # Snapshot model state at construction time                           #
        # ------------------------------------------------------------------ #
        # Fixed full-parameter template: positions not in varying_indices hold
        # their fixed values; varying slots are overwritten on each __call__.
        self._fixed_full_jax = jnp.asarray(
            model.param_manager.get_full_values(), dtype=jnp.float64
        )
        self._varying_indices_jax = jnp.asarray(
            list(model.param_manager.varying_indices), dtype=jnp.int32
        )

        # Model geometry
        self._t = jnp.asarray(np.asarray(model.t, dtype=np.float64), dtype=jnp.float64)
        self._q = float(model.q)
        self._dt = float(model.dt)

        # Per-angle scaling snapshot (read once; held fixed during optimisation)
        n_phi_chunks = len(self.chunks)
        n_scaling_angles = model.scaling.n_angles  # may be < n_phi_chunks
        self._contrast: list[float] = []
        self._offset: list[float] = []
        for chunk_idx in range(n_phi_chunks):
            if per_angle_scaling:
                # Clamp to the number of scaling angles actually initialised in
                # the model.  For synthetic test datasets the model is built with
                # the default config (n_angles=1), but the data may have more
                # angles.  Clamping mirrors the homodyne convention of sharing
                # the baseline scaling across extra angles.
                safe_idx = min(chunk_idx, n_scaling_angles - 1)
                c, o = model.scaling.get_for_angle(safe_idx)
            else:
                c, o = model.scaling.get_for_angle(0)
            self._contrast.append(float(c))
            self._offset.append(float(o))

        # Sigma array (metadata, 3-D) from the original stratified_data object
        sigma_np = np.asarray(stratified_data.sigma, dtype=np.float64)
        self._sigma_jax = jnp.asarray(sigma_np)

        # Pre-compute per-chunk JAX arrays and diagonal masks
        self._chunks_jax = self._preconvert_chunks()

        self.logger.debug(
            "StratifiedHeterodyneResidualFunction: %d chunks, %d total points, "
            "n_varying_physics=%d",
            self.n_chunks,
            self.n_total_points,
            len(physical_param_names),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _preconvert_chunks(self) -> list[dict[str, Any]]:
        """Convert chunk attributes to JAX arrays once at construction time."""
        chunks_jax = []
        for chunk in self.chunks:
            phi_vals = np.asarray(chunk.phi, dtype=np.float64)
            t1_vals = np.asarray(chunk.t1, dtype=np.float64)
            t2_vals = np.asarray(chunk.t2, dtype=np.float64)

            # Build diagonal mask: exclude points where t1 == t2
            # We compare rounded float values to avoid fp equality issues
            # (the grid is built from the same t array so this is exact).
            t1_idx = np.searchsorted(np.unique(t1_vals), t1_vals)
            t2_idx = np.searchsorted(np.unique(t2_vals), t2_vals)
            off_diag_mask = jnp.asarray(t1_idx != t2_idx, dtype=jnp.bool_)

            # Unique phi value for this chunk (single angle per slab)
            unique_phi = float(np.unique(phi_vals)[0])

            cj: dict[str, Any] = {
                "phi": unique_phi,
                "t1": jnp.asarray(t1_vals),
                "t2": jnp.asarray(t2_vals),
                "g2": jnp.asarray(np.asarray(chunk.g2, dtype=np.float64)),
                "off_diag_mask": off_diag_mask,
                "q": float(chunk.q),
                "L": float(chunk.L),
                "dt": float(chunk.dt) if chunk.dt is not None else self._dt,
            }
            chunks_jax.append(cj)
        return chunks_jax

    def _full_params_jax(self, varying_params: jnp.ndarray) -> jnp.ndarray:
        """Scatter varying params into the full 14-parameter template."""
        return self._fixed_full_jax.at[self._varying_indices_jax].set(varying_params)

    def _compute_chunk_residual(
        self,
        chunk_jax: dict[str, Any],
        full: jnp.ndarray,
        contrast: float,
        offset: float,
    ) -> jnp.ndarray:
        """Compute the off-diagonal weighted residual for a single angle chunk.

        Args:
            chunk_jax:  Pre-converted JAX chunk dict.
            full:       Full 14-parameter JAX array.
            contrast:   Per-angle speckle contrast.
            offset:     Per-angle baseline offset.

        Returns:
            1-D residual array (off-diagonal entries only; diagonal → 0).
        """
        # Predict the full (n_t, n_t) C2 matrix at this phi angle
        c2_pred = compute_c2_heterodyne(
            full,
            self._t,
            self._q,
            self._dt,
            chunk_jax["phi"],
            contrast,
            offset,
        )  # shape: (n_t, n_t)

        # The chunk's g2 is the flattened observed matrix (n_t*n_t,)
        # c2_pred.ravel() aligns with the same row-major flattening
        c2_pred_flat = c2_pred.ravel()

        # Residual = (obs - pred); sigma handling below
        raw_res = chunk_jax["g2"] - c2_pred_flat

        # Apply off-diagonal mask: zero residuals on the main diagonal
        res_masked = jnp.where(chunk_jax["off_diag_mask"], raw_res, 0.0)

        return res_masked

    # ------------------------------------------------------------------ #
    # Public interface                                                    #
    # ------------------------------------------------------------------ #

    def __call__(self, params: np.ndarray) -> np.ndarray:
        """Evaluate the joint residual vector across all angle chunks.

        Args:
            params: Varying physics parameters, shape ``(n_varying_physics,)``.
                    Must match ``model.param_manager.varying_names`` in order.

        Returns:
            1-D float64 residual array, length = sum of off-diagonal entries
            across all angle chunks.
        """
        params_jax = jnp.asarray(params, dtype=jnp.float64)
        full = self._full_params_jax(params_jax)

        residual_parts: list[jnp.ndarray] = []
        for chunk_idx, chunk_jax in enumerate(self._chunks_jax):
            contrast = self._contrast[chunk_idx]
            offset = self._offset[chunk_idx]
            res = self._compute_chunk_residual(chunk_jax, full, contrast, offset)
            residual_parts.append(res)

        all_residuals = jnp.concatenate(residual_parts, axis=0)
        return np.asarray(all_residuals, dtype=np.float64)


def create_stratified_heterodyne_residual_function(
    stratified_data: Any,
    model: HeterodyneModel,
    per_angle_scaling: bool,
    physical_param_names: list[str],
    logger: logging.Logger | None = None,
) -> StratifiedHeterodyneResidualFunction:
    """Factory function for :class:`StratifiedHeterodyneResidualFunction`.

    Provides a consistent factory interface matching the homodyne pattern.

    Args:
        stratified_data:      :class:`HeterodyneStratifiedData` or compatible object.
        model:                Configured :class:`HeterodyneModel`.
        per_angle_scaling:    Whether to read per-angle contrast/offset from
                              ``model.scaling.get_for_angle``.
        physical_param_names: Ordered list of varying physics parameter names.
        logger:               Optional logger.

    Returns:
        Callable that maps ``params: np.ndarray → np.ndarray`` (1-D residuals).
    """
    return StratifiedHeterodyneResidualFunction(
        stratified_data=stratified_data,
        model=model,
        per_angle_scaling=per_angle_scaling,
        physical_param_names=physical_param_names,
        logger=logger,
    )
