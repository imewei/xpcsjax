"""
JAX JIT-compatible stratified residual function for NLSQ optimization.

This module provides a JIT-compatible version of StratifiedResidualFunction that uses
static shapes and vmap for vectorization, solving the JAX tracing incompatibility.

Key Improvements over original StratifiedResidualFunction:
- Uses jax.vmap for parallel chunk processing (no Python loops)
- Pads chunks to uniform size for static shapes (JIT-compatible)
- Fully JIT-compiled for maximum performance
- Maintains angle stratification guarantee

Author: Homodyne Development Team
Date: 2025-11-13
"""

from __future__ import annotations

import logging
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.model_adapter import (
    HomodynePointEvaluator,
    PointEvaluator,
)
from xpcsjax.utils.logging import get_logger, log_phase


class StratifiedResidualFunctionJIT:
    """
    JIT-compatible stratified residual function using padded vmap.

    This class solves the JAX JIT incompatibility by:

    1. Padding all chunks to uniform size (static shapes)
    2. Using ``jax.vmap`` for vectorized parallel processing
    3. Masking padded values in the final residuals

    The function maintains angle stratification (all chunks contain all angles)
    while being fully JIT-compilable. It is the model-agnostic stratification
    engine: it evaluates the physics surface through an injected
    :class:`~xpcsjax.optimization.nlsq.model_adapter.PointEvaluator` rather than
    hard-coding a kernel, so the same engine serves homodyne (``laminar_flow``)
    and heterodyne (``two_component``) fits.

    Attributes
    ----------
    phi_padded : jnp.ndarray
        Padded phi arrays, shape ``(n_chunks, max_chunk_size)``.
    t1_padded : jnp.ndarray
        Padded t1 arrays, shape ``(n_chunks, max_chunk_size)``.
    t2_padded : jnp.ndarray
        Padded t2 arrays, shape ``(n_chunks, max_chunk_size)``.
    g2_padded : jnp.ndarray
        Padded g2 observations, shape ``(n_chunks, max_chunk_size)``.
    mask : jnp.ndarray
        Boolean mask for real vs padded data, shape ``(n_chunks, max_chunk_size)``.
    n_chunks : int
        Number of stratified chunks.
    max_chunk_size : int
        Maximum points per chunk (the uniform padded size).
    n_real_points : int
        Total number of real (non-padded) data points.
    """

    def __init__(
        self,
        stratified_data: Any,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        logger: logging.Logger | None = None,
        fixed_contrast_per_angle: np.ndarray | None = None,
        fixed_offset_per_angle: np.ndarray | None = None,
        evaluator: PointEvaluator | None = None,
    ) -> None:
        """
        Initialize JIT-compatible stratified residual function.

        Parameters
        ----------
        stratified_data : Any
            Object with a ``.chunks`` attribute holding angle-stratified chunks
            and a ``.sigma`` array.
        per_angle_scaling : bool
            Whether per-angle scaling parameters are used.
        physical_param_names : list of str
            Physical parameter names.
        logger : logging.Logger, optional
            Logger for diagnostics; defaults to the module logger.
        fixed_contrast_per_angle : np.ndarray, optional
            Fixed per-angle contrast values (constant mode). When provided,
            contrast is NOT included in the parameter vector.
        fixed_offset_per_angle : np.ndarray, optional
            Fixed per-angle offset values (constant mode). When provided,
            offset is NOT included in the parameter vector.
        evaluator : PointEvaluator, optional
            Model-agnostic point evaluator. When ``None`` (the default), a
            :class:`~xpcsjax.optimization.nlsq.model_adapter.HomodynePointEvaluator`
            wrapping ``compute_g2_scaled`` is constructed from this class's own
            ``q``/``L``/``dt`` — preserving homodyne (``laminar_flow``) behavior
            exactly.
        """
        self.logger = logger or get_logger(__name__)
        self.chunks = stratified_data.chunks
        self.per_angle_scaling = per_angle_scaling
        self.physical_param_names = physical_param_names

        # Fixed per-angle scaling for constant mode
        # When both are provided, params contains ONLY physical parameters
        self.fixed_contrast_per_angle = None
        self.fixed_offset_per_angle = None
        self.use_fixed_scaling = False

        if fixed_contrast_per_angle is not None and fixed_offset_per_angle is not None:
            self.fixed_contrast_per_angle = jnp.asarray(fixed_contrast_per_angle)
            self.fixed_offset_per_angle = jnp.asarray(fixed_offset_per_angle)
            self.use_fixed_scaling = True
            self.logger.info(
                "Using fixed per-angle scaling from quantiles. "
                "Parameter vector contains ONLY physical parameters."
            )

        if not self.chunks:
            raise ValueError("stratified_data.chunks is empty")

        self.n_chunks = len(self.chunks)

        # Extract global metadata (same across all chunks)
        self.q, self.L, self.dt = self._extract_global_metadata()

        # Model-agnostic point evaluator. The engine evaluates the physics surface
        # through self._evaluator.eval_points(...) instead of hard-coding
        # compute_g2_scaled. When no evaluator is injected we build the homodyne
        # default from THIS object's q/L/dt. dt uses the SAME 0.001 fallback the
        # call sites applied (dt_value), so threading is bit-identical.
        dt_value = self.dt if self.dt is not None else 0.001
        self._evaluator: PointEvaluator = evaluator or HomodynePointEvaluator(
            analysis_mode="laminar_flow",
            q=self.q,
            L=self.L,
            dt=dt_value,
        )

        self.phi_unique, self.t1_unique, self.t2_unique = self._extract_unique_values()
        self.n_phi = len(self.phi_unique)

        # Prepare sigma array
        sigma_array = np.asarray(stratified_data.sigma, dtype=np.float64)
        self.sigma_jax = jnp.asarray(sigma_array)

        # Create padded arrays with static shapes
        self.logger.info(f"Creating padded arrays for {self.n_chunks} chunks...")
        (
            self.phi_padded,
            self.t1_padded,
            self.t2_padded,
            self.g2_padded,
            self.mask,
            self.max_chunk_size,
            self.n_real_points,
        ) = self._create_padded_arrays()

        self.logger.info(
            f"Padded arrays created: shape ({self.n_chunks}, {self.max_chunk_size}), "
            f"real points: {self.n_real_points:,}, "
            f"padding overhead: {(1 - self.n_real_points / (self.n_chunks * self.max_chunk_size)) * 100:.2f}%"
        )

        # JIT-compile the main residual computation
        # Note: Buffer donation (donate_argnums) is not used here because the
        # params array (small, e.g. 9 elements) never matches the output shape
        # (n_chunks * max_chunk_size), so JAX cannot reuse the buffer.
        self.logger.info("JIT-compiling residual function...")
        # T035: Add log_phase for JIT compilation timing with memory tracking
        with log_phase("jit_residual_compilation", logger=self.logger, track_memory=True) as phase:
            self._residual_fn_jit = jax.jit(self._compute_all_residuals)
        self.logger.info(f"JIT compilation setup complete in {phase.duration:.3f}s")

    def _extract_global_metadata(self) -> tuple[float, float, float | None]:
        """Extract q, L, dt from chunks (should be same for all chunks)."""
        q_values = [float(chunk.q) for chunk in self.chunks]
        L_values = [float(chunk.L) for chunk in self.chunks]
        dt_values = [float(chunk.dt) if chunk.dt is not None else None for chunk in self.chunks]

        # Validate consistency
        if not all(abs(q - q_values[0]) < 1e-9 for q in q_values):
            raise ValueError("Inconsistent q values across chunks")
        if not all(abs(L - L_values[0]) < 1e-6 for L in L_values):
            raise ValueError("Inconsistent L values across chunks")

        q = q_values[0]
        L = L_values[0]
        dt = dt_values[0] if dt_values[0] is not None else None

        self.logger.debug(f"Global metadata: q={q:.6f}, L={L:.1f}, dt={dt}")
        return q, L, dt

    def _extract_unique_values(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Extract unique phi, t1, t2 values from ALL chunks.

        CRITICAL: Must extract from all chunks, not just first chunk, because
        stratified chunking may distribute different subsets of t1/t2 values
        across different chunks.
        """
        # Concatenate values from all chunks to get complete set
        all_phi = np.concatenate([chunk.phi for chunk in self.chunks])
        all_t1 = np.concatenate([chunk.t1 for chunk in self.chunks])
        all_t2 = np.concatenate([chunk.t2 for chunk in self.chunks])

        # Extract unique values across all chunks.
        # Host-side (NumPy): this is a one-time __init__ grid/metadata
        # computation that never needs to be on-device or JIT-lowered. np.unique
        # returns sorted unique values, so it reproduces the prior
        # jnp.sort(jnp.unique(...)) exactly. The three GLOBAL unique arrays are
        # stored on self and consumed inside the JIT'd warm kernel
        # (_compute_single_chunk_residuals -> eval_points / searchsorted), so
        # they are converted back to JAX once below with jnp.asarray.
        phi_unique_np = np.unique(all_phi)
        t1_unique_np = np.unique(all_t1)
        t2_unique_np = np.unique(all_t2)

        phi_unique = jnp.asarray(phi_unique_np)
        t1_unique = jnp.asarray(t1_unique_np)
        t2_unique = jnp.asarray(t2_unique_np)

        self.logger.debug(
            f"Unique values (from all chunks): {len(phi_unique_np)} phi, {len(t1_unique_np)} t1, {len(t2_unique_np)} t2"
        )

        # Validation: check if we missed any values by comparing with first chunk.
        # Pure host-side comparison (only len() is consumed), so these stay NumPy.
        first_chunk = self.chunks[0]
        _phi_first = np.unique(first_chunk.phi)  # noqa: F841
        t1_first = np.unique(first_chunk.t1)
        t2_first = np.unique(first_chunk.t2)

        if len(t1_unique_np) != len(t1_first) or len(t2_unique_np) != len(t2_first):
            self.logger.debug(
                f"Stratified chunking: chunks have different time point subsets "
                f"(first chunk: {len(t1_first)} t1, all chunks: {len(t1_unique_np)} t1) - "
                f"using complete set from all chunks"
            )

        return phi_unique, t1_unique, t2_unique

    def _create_padded_arrays(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int]:
        """
        Create padded arrays with uniform size across all chunks.

        Returns
        -------
        tuple
            ``(phi_padded, t1_padded, t2_padded, g2_padded, mask,
            max_chunk_size, n_real_points)``.
        """
        # Determine max chunk size
        chunk_sizes = [len(chunk.phi) for chunk in self.chunks]
        max_chunk_size = max(chunk_sizes)
        n_real_points = sum(chunk_sizes)

        self.logger.debug(
            f"Max chunk size: {max_chunk_size:,}, total real points: {n_real_points:,}"
        )

        # Initialize padded arrays
        phi_padded = np.zeros((self.n_chunks, max_chunk_size), dtype=np.float64)
        t1_padded = np.zeros((self.n_chunks, max_chunk_size), dtype=np.float64)
        t2_padded = np.zeros((self.n_chunks, max_chunk_size), dtype=np.float64)
        g2_padded = np.zeros((self.n_chunks, max_chunk_size), dtype=np.float64)
        mask = np.zeros((self.n_chunks, max_chunk_size), dtype=bool)

        # Fill arrays with data and create mask
        for i, chunk in enumerate(self.chunks):
            n_points = len(chunk.phi)

            # Copy real data
            phi_padded[i, :n_points] = chunk.phi
            t1_padded[i, :n_points] = chunk.t1
            t2_padded[i, :n_points] = chunk.t2
            g2_padded[i, :n_points] = chunk.g2
            mask[i, :n_points] = True

            # Pad with last valid value (prevents out-of-bounds indexing)
            if n_points < max_chunk_size:
                phi_padded[i, n_points:] = chunk.phi[-1]
                t1_padded[i, n_points:] = chunk.t1[-1]
                t2_padded[i, n_points:] = chunk.t2[-1]
                g2_padded[i, n_points:] = chunk.g2[-1]
                # mask already False for padding

        # Convert to JAX arrays
        phi_padded_jax = jnp.asarray(phi_padded)
        t1_padded_jax = jnp.asarray(t1_padded)
        t2_padded_jax = jnp.asarray(t2_padded)
        g2_padded_jax = jnp.asarray(g2_padded)
        mask_jax = jnp.asarray(mask)

        return (
            phi_padded_jax,
            t1_padded_jax,
            t2_padded_jax,
            g2_padded_jax,
            mask_jax,
            max_chunk_size,
            n_real_points,
        )

    def _compute_single_chunk_residuals(
        self,
        phi_chunk: jnp.ndarray,
        t1_chunk: jnp.ndarray,
        t2_chunk: jnp.ndarray,
        g2_obs_chunk: jnp.ndarray,
        mask_chunk: jnp.ndarray,
        params_all: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute residuals for a single padded chunk.

        This function is designed to be ``vmap``-ped over the chunk dimension.

        Parameters
        ----------
        phi_chunk : jnp.ndarray
            Phi values for this chunk, shape ``(max_chunk_size,)``.
        t1_chunk : jnp.ndarray
            t1 values for this chunk, shape ``(max_chunk_size,)``.
        t2_chunk : jnp.ndarray
            t2 values for this chunk, shape ``(max_chunk_size,)``.
        g2_obs_chunk : jnp.ndarray
            Observed g2 for this chunk, shape ``(max_chunk_size,)``.
        mask_chunk : jnp.ndarray
            Mask for real vs padded data, shape ``(max_chunk_size,)``.
        params_all : jnp.ndarray
            All parameters ``[scaling_params, physical_params]``, or just
            ``[physical_params]`` when ``use_fixed_scaling=True``.

        Returns
        -------
        jnp.ndarray
            Masked residuals, shape ``(max_chunk_size,)``; padded values are zero.
        """
        # Extract scaling and physical parameters
        # Three modes:
        # 1. use_fixed_scaling=True: params_all = [physical_params only]
        #    contrast/offset come from self.fixed_contrast_per_angle/self.fixed_offset_per_angle
        # 2. per_angle_scaling=True: params_all = [contrast(n_phi), offset(n_phi), physical]
        # 3. per_angle_scaling=False: params_all = [contrast, offset, physical]

        if self.use_fixed_scaling:
            # CONSTANT MODE: Fixed per-angle scaling from quantiles
            # params_all contains ONLY physical parameters
            contrast = self.fixed_contrast_per_angle
            offset = self.fixed_offset_per_angle
            physical_params = params_all  # All params are physical
        elif self.per_angle_scaling:
            contrast = params_all[: self.n_phi]
            offset = params_all[self.n_phi : 2 * self.n_phi]
            physical_params = params_all[2 * self.n_phi :]
        else:
            contrast = params_all[0]
            offset = params_all[1]
            physical_params = params_all[2:]

        # Compute theoretical g2 using vectorized computation
        # NOTE: Warning for dt=None is emitted in __call__ (outside JIT trace)
        # dt is sourced via self._evaluator (built with the same 0.001 fallback).
        if self.use_fixed_scaling or self.per_angle_scaling:
            # Vectorize over phi with corresponding contrast/offset
            def compute_for_angle(
                phi_val: float, contrast_val: float, offset_val: float
            ) -> jnp.ndarray:
                return jnp.squeeze(
                    self._evaluator.eval_points(
                        physical_params,
                        jnp.asarray(phi_val),
                        self.t1_unique,
                        self.t2_unique,
                        contrast_val,
                        offset_val,
                    ),
                    axis=0,
                )

            compute_g2_vmap = jax.vmap(compute_for_angle, in_axes=(0, 0, 0))
            g2_theory_grid = compute_g2_vmap(self.phi_unique, contrast, offset)  # type: ignore[arg-type]
        else:
            # Legacy: single contrast/offset
            def compute_for_angle_scalar(phi_val: float) -> jnp.ndarray:
                # We use cast(float, ...) here to satisfy mypy, but at runtime these are JAX tracers
                # which compute_g2_scaled handles correctly despite the float type hint.
                from typing import cast  # noqa: F811 — intentional re-import in closure

                return jnp.squeeze(
                    self._evaluator.eval_points(
                        physical_params,
                        jnp.asarray(phi_val),
                        self.t1_unique,
                        self.t2_unique,
                        cast(float, contrast),
                        cast(float, offset),
                    ),
                    axis=0,
                )

            compute_g2_vmap_scalar = jax.vmap(compute_for_angle_scalar, in_axes=0)
            g2_theory_grid = compute_g2_vmap_scalar(self.phi_unique)  # type: ignore[arg-type]

        # NOTE: Diagonal correction is intentionally skipped here.
        # Residuals for t1==t2 points are masked out below via `non_diagonal`,
        # so theory grid diagonal values are never used in the optimization.
        # Skipping this call removes ~38% of residual computation time.

        # Flatten theory grid for indexing
        g2_theory_flat = g2_theory_grid.flatten()

        # Find indices of (phi, t1, t2) in the full grid
        # n_phi dimension used implicitly for grid shape: (n_phi, n_t1, n_t2)
        n_t1 = len(self.t1_unique)
        n_t2 = len(self.t2_unique)

        # Note: clip removed - stratified LS data comes from same chunks that build
        # unique arrays, so all values are guaranteed to be in range. The clip was
        # causing optimization to converge to wrong local minima (D0=91342 vs 19253).
        # Original clip added in ae4848c for streaming optimizer, but not needed here.
        # Cast to int64 BEFORE multiplication to prevent int32 overflow.
        # jnp.searchsorted returns int32; for large datasets (n_phi=100,
        # n_t1=5000, n_t2=5000) the product 99*25_000_000=2.475B exceeds
        # int32 max (2.147B), silently wrapping to a negative index.
        phi_indices = jnp.searchsorted(self.phi_unique, phi_chunk).astype(jnp.int64)
        t1_indices = jnp.searchsorted(self.t1_unique, t1_chunk).astype(jnp.int64)
        t2_indices = jnp.searchsorted(self.t2_unique, t2_chunk).astype(jnp.int64)

        # Compute flat indices
        flat_indices = phi_indices * (n_t1 * n_t2) + t1_indices * n_t2 + t2_indices

        # Extract theory values for chunk points
        g2_theory_chunk = g2_theory_flat[flat_indices]

        # Get sigma values for chunk points
        sigma_flat = self.sigma_jax.flatten()
        sigma_chunk = sigma_flat[flat_indices]

        # Compute weighted residuals — mask out zero-sigma points entirely
        EPS = 1e-10
        valid_sigma = sigma_chunk > EPS
        safe_sigma = jnp.where(valid_sigma, sigma_chunk, 1.0)
        residuals_raw = jnp.where(valid_sigma, (g2_obs_chunk - g2_theory_chunk) / safe_sigma, 0.0)

        # Mask out both padded values AND diagonal values (t1 == t2)
        # Diagonal points are autocorrelation artifacts, not physics
        # CRITICAL FIX (2026-01-15): Compare actual time VALUES, not indices.
        # t1_indices and t2_indices reference DIFFERENT arrays (t1_unique vs t2_unique),
        # so comparing indices is wrong. Must compare the actual t1_chunk and t2_chunk values.
        non_diagonal = jnp.abs(t1_chunk - t2_chunk) > 1e-15
        residuals_masked = jnp.where(mask_chunk & non_diagonal, residuals_raw, 0.0)

        return residuals_masked

    def _compute_all_residuals(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Compute residuals for all chunks using ``vmap`` (JIT-compiled).

        Parameters
        ----------
        params : jnp.ndarray
            All parameters (scaling + physical).

        Returns
        -------
        jnp.ndarray
            Flattened residuals INCLUDING padding, shape
            ``(n_chunks * max_chunk_size,)`` with zeros for padded values.
            Filtering happens in :meth:`__call__`.
        """
        # Cache vmap'd function to avoid JIT retrace on every call.
        # params is passed as an explicit unbatched argument (in_axes=None for the
        # last axis) instead of via closure capture. A new lambda (new Python object
        # identity) is created each call when params is captured by closure, forcing
        # JAX to retrace the vmap'd function on every optimizer iteration.
        if not hasattr(self, "_cached_chunk_vmap"):
            self._cached_chunk_vmap = jax.vmap(
                lambda phi, t1, t2, g2, mask, p: self._compute_single_chunk_residuals(
                    phi, t1, t2, g2, mask, p
                ),
                in_axes=(0, 0, 0, 0, 0, None),  # params (p) not batched
            )

        # Compute residuals for all chunks in parallel
        residuals_padded = self._cached_chunk_vmap(
            self.phi_padded,
            self.t1_padded,
            self.t2_padded,
            self.g2_padded,
            self.mask,
            params,
        )  # Shape: (n_chunks, max_chunk_size)

        # Flatten residuals (padding is already masked to zero in _compute_single_chunk_residuals)
        residuals_flat = residuals_padded.flatten()  # Shape: (n_chunks * max_chunk_size,)

        # Return full array (filtering happens in __call__ to avoid JIT boolean indexing)
        return residuals_flat

    def __call__(self, params: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        """
        Compute residuals (interface for NLSQ ``least_squares``).

        This method is JIT-traced by NLSQ, so it must use JAX operations only.
        Padded values are already masked to zero, so they do not contribute to
        the optimization objective (sum of squared residuals).

        Parameters
        ----------
        params : np.ndarray or jnp.ndarray
            Parameters.

        Returns
        -------
        jnp.ndarray
            Residuals, shape ``(n_chunks * max_chunk_size,)`` with zeros for
            padding. Padding zeros do not affect the optimization but increase
            the array size.
        """
        if self.dt is None:
            self.logger.warning(
                "StratifiedResidualFunctionJIT: dt is None; "
                "using dt=0.001 s as fallback. Physics factors may be incorrect."
            )
        params_jax = jnp.asarray(params, dtype=jnp.float64)
        residuals_jax = self._residual_fn_jit(params_jax)
        return cast(jnp.ndarray, residuals_jax)  # Keep as JAX array for JIT compatibility

    def validate_chunk_structure(self) -> bool:
        """
        Validate that all chunks contain all phi angles.

        Returns
        -------
        bool
            ``True`` if validation passes.

        Raises
        ------
        ValueError
            If any chunk's angle distribution does not match the expected set.
        """
        expected_angles = set(np.unique(np.round(np.asarray(self.phi_unique), decimals=6)))
        n_expected = len(expected_angles)

        self.logger.info(
            f"Validating chunk structure: {self.n_chunks} chunks, "
            f"{n_expected} expected angles per chunk"
        )

        for i, _chunk in enumerate(self.chunks):
            # Only check real data (not padding)
            n_real = int(np.sum(self.mask[i]))
            phi_real = self.phi_padded[i, :n_real]
            chunk_angles = set(np.unique(np.round(np.asarray(phi_real), decimals=6)))

            if chunk_angles != expected_angles:
                missing = expected_angles - chunk_angles
                extra = chunk_angles - expected_angles
                raise ValueError(
                    f"Chunk {i} has invalid angle distribution:\n"
                    f"  Missing angles: {sorted(missing)}\n"
                    f"  Extra angles: {sorted(extra)}\n"
                    f"  Expected {n_expected} angles, got {len(chunk_angles)}"
                )

        self.logger.info("Chunk structure validation passed: all chunks angle-complete")
        return True

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about the residual function."""
        return {
            "n_chunks": self.n_chunks,
            "max_chunk_size": self.max_chunk_size,
            "n_real_points": self.n_real_points,
            "padding_overhead_pct": (1 - self.n_real_points / (self.n_chunks * self.max_chunk_size))
            * 100,
            "n_phi": self.n_phi,
            "n_t1": len(self.t1_unique),
            "n_t2": len(self.t2_unique),
            "per_angle_scaling": self.per_angle_scaling,
            "jit_compiled": True,
        }

    def log_diagnostics(self) -> None:
        """Log diagnostic information about the residual function."""
        diag = self.get_diagnostics()
        self.logger.info("Stratified Residual Function Diagnostics:")
        self.logger.info(f"  Chunks: {diag['n_chunks']}")
        self.logger.info(f"  Max chunk size: {diag['max_chunk_size']:,}")
        self.logger.info(f"  Real points: {diag['n_real_points']:,}")
        self.logger.info(f"  Padding overhead: {diag['padding_overhead_pct']:.2f}%")
        self.logger.info(f"  Angles (phi): {diag['n_phi']}")
        self.logger.info(f"  Time points (t1): {diag['n_t1']}")
        self.logger.info(f"  Time points (t2): {diag['n_t2']}")
        self.logger.info(f"  Per-angle scaling: {diag['per_angle_scaling']}")
        self.logger.info(f"  JIT compiled: {diag['jit_compiled']}")
