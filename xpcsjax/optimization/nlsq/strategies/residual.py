from __future__ import annotations

import logging
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.core.physics_nlsq import compute_g2_scaled
from xpcsjax.utils.logging import get_logger


class StratifiedResidualFunction:
    """
    Residual function that respects angle-stratified chunk structure.

    This class wraps the model's residual computation to work with stratified chunks,
    ensuring that each chunk contains all phi angles. This is critical for per-angle
    scaling parameters to have non-zero gradients.

    The function is designed to work with NLSQ's least_squares() function, which calls
    the residual function at each optimization iteration.

    Attributes:
        chunks: List of angle-stratified data chunks
        model: TheoryEngine instance for computing residuals
        per_angle_scaling: Whether per-angle scaling is enabled
        logger: Logger instance for diagnostics
        n_chunks: Number of stratified chunks
        n_total_points: Total number of data points across all chunks
        compute_chunk_jit: JIT-compiled chunk residual computation
    """

    def __init__(
        self,
        stratified_data: Any,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the stratified residual function.

        Args:
            stratified_data: Object with .chunks attribute containing angle-stratified chunks.
                Each chunk must have: phi, t1, t2, g2, q, L, dt attributes.
                stratified_data.sigma contains the full 3D sigma array (metadata).
            per_angle_scaling: Whether per-angle scaling parameters are used.
            physical_param_names: List of physical parameter names (e.g., ['D0', 'alpha', 'D_offset'])
            logger: Optional logger for diagnostics.

        Raises:
            ValueError: If stratified_data.chunks is empty or invalid.
        """
        self.chunks = stratified_data.chunks
        sigma_array = np.asarray(stratified_data.sigma, dtype=np.float64)
        # M2: Only keep JAX array; numpy copy was labelled "for legacy paths"
        # but self.sigma is never referenced outside __init__.
        self._sigma_jax = jnp.asarray(sigma_array)
        del sigma_array  # Allow GC of the intermediate numpy copy
        self.per_angle_scaling = per_angle_scaling
        self.physical_param_names = physical_param_names
        self.logger = logger or get_logger(__name__)

        if not self.chunks:
            raise ValueError("stratified_data.chunks is empty")

        self.n_chunks = len(self.chunks)
        self.n_total_points = sum(len(chunk.g2) for chunk in self.chunks)

        # Determine number of unique angles from first chunk
        self.n_phi = len(np.unique(self.chunks[0].phi))

        # Determine expected parameter structure
        # Per-angle: [contrast_0, ..., contrast_{n-1}, offset_0, ..., offset_{n-1}, *physical]
        # Legacy: [contrast, offset, *physical]
        if per_angle_scaling:
            self.n_scaling_params = 2 * self.n_phi
        else:
            self.n_scaling_params = 2

        self.n_physical_params = len(physical_param_names)
        self.n_total_params = self.n_scaling_params + self.n_physical_params

        # Pre-compute unique values for each chunk (avoid jnp.unique in JIT)
        self._precompute_chunk_metadata()

        # Setup JIT-compiled functions
        self._setup_jax_functions()

        # Pre-convert chunk arrays to JAX (avoid jnp.asarray in loop)
        self._preconvert_chunk_arrays()

        self.logger.info(
            f"StratifiedResidualFunction initialized: "
            f"{self.n_chunks} chunks, {self.n_total_points:,} total points, "
            f"n_phi={self.n_phi}, per_angle_scaling={self.per_angle_scaling}, "
            f"n_scaling_params={self.n_scaling_params}, n_physical_params={self.n_physical_params}"
        )

    def _precompute_chunk_metadata(self) -> None:
        """
        Pre-compute GLOBAL unique values from ALL chunks to avoid jnp.unique() in JIT.

        This method extracts unique phi, t1, t2 values from ALL chunks combined
        and stores them as metadata. Each chunk gets the SAME global unique arrays
        to ensure correct flat indexing when accessing sigma_full array.

        This avoids ConcretizationTypeError when using jnp.unique() inside
        JIT-compiled functions.

        CRITICAL: Must use global unique values, not per-chunk subsets, because
        sigma_full dimensions are based on ALL data points across all chunks.

        Performance Optimization (Spec 006 - FR-001):
        Also pre-computes flat indices for each chunk to avoid jnp.searchsorted
        calls inside the JIT-compiled residual function. This provides ~15-20%
        per-iteration speedup.
        """
        # Extract GLOBAL unique values from ALL chunks combined
        # This ensures grid dimensions match sigma_full dimensions
        all_phi = np.concatenate([chunk.phi for chunk in self.chunks])
        all_t1 = np.concatenate([chunk.t1 for chunk in self.chunks])
        all_t2 = np.concatenate([chunk.t2 for chunk in self.chunks])

        global_phi_unique = jnp.sort(jnp.unique(jnp.asarray(all_phi)))
        global_t1_unique = jnp.sort(jnp.unique(jnp.asarray(all_t1)))
        global_t2_unique = jnp.sort(jnp.unique(jnp.asarray(all_t2)))

        # Store global dimensions for flat index computation
        self._n_t1_global = len(global_t1_unique)
        self._n_t2_global = len(global_t2_unique)

        self.logger.debug(
            f"Global unique values extracted from all chunks: "
            f"{len(global_phi_unique)} phi, "
            f"{self._n_t1_global} t1, "
            f"{self._n_t2_global} t2"
        )

        # A4: Store the global unique arrays directly. They are identical for
        # every chunk (grid dimensions must match sigma_full), so a per-chunk
        # metadata dict list was pure indirection — replaced by three attrs.
        self._phi_unique_global = global_phi_unique
        self._t1_unique_global = global_t1_unique
        self._t2_unique_global = global_t2_unique

        self._precomputed_flat_indices = []
        self._precomputed_t1_indices = []  # v2.14.2+: for diagonal masking
        self._precomputed_t2_indices = []  # v2.14.2+: for diagonal masking

        for chunk in self.chunks:
            # Pre-compute flat indices for this chunk (FR-001 optimization)
            # v2.14.2+: Also returns t1/t2 indices for diagonal masking
            flat_indices, t1_indices, t2_indices = self._compute_flat_indices(
                phi=chunk.phi,
                t1=chunk.t1,
                t2=chunk.t2,
                phi_unique=global_phi_unique,
                t1_unique=global_t1_unique,
                t2_unique=global_t2_unique,
            )
            self._precomputed_flat_indices.append(flat_indices)
            self._precomputed_t1_indices.append(t1_indices)
            self._precomputed_t2_indices.append(t2_indices)

        self.logger.debug(
            f"Pre-computed flat indices for {len(self._precomputed_flat_indices)} chunks"
        )

    def _compute_flat_indices(
        self,
        phi: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi_unique: jnp.ndarray,
        t1_unique: jnp.ndarray,
        t2_unique: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute flat indices for mapping chunk points to global grid positions.

        This helper method computes the 1D flat indices that map each point
        in a chunk to its position in the flattened 3D grid (phi × t1 × t2).

        Also returns t1_indices and t2_indices for diagonal masking (v2.14.2+).

        Performance Note (Spec 006 - FR-001):
        This method is called once during __init__ to pre-compute indices,
        avoiding expensive jnp.searchsorted calls during every optimization
        iteration. Expected speedup: 15-20% per iteration.

        Parameters
        ----------
        phi : np.ndarray
            Phi values for this chunk
        t1 : np.ndarray
            t1 values for this chunk
        t2 : np.ndarray
            t2 values for this chunk
        phi_unique : jnp.ndarray
            Global unique phi values (sorted)
        t1_unique : jnp.ndarray
            Global unique t1 values (sorted)
        t2_unique : jnp.ndarray
            Global unique t2 values (sorted)

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            - flat_indices: Flat indices for this chunk's points into the global grid
            - t1_indices: t1 indices for diagonal masking
            - t2_indices: t2 indices for diagonal masking
        """
        # Convert to JAX arrays for searchsorted
        phi_jax = jnp.asarray(phi)
        t1_jax = jnp.asarray(t1)
        t2_jax = jnp.asarray(t2)

        # Find indices in the sorted unique arrays.
        # Cast to int64 BEFORE multiplication to prevent int32 overflow.
        # jnp.searchsorted returns int32; for large datasets (n_phi=100,
        # n_t1=5000, n_t2=5000) the product 99*25_000_000=2.475B exceeds
        # int32 max (2.147B), silently wrapping to a negative index.
        phi_indices = jnp.searchsorted(phi_unique, phi_jax).astype(jnp.int64)
        t1_indices = jnp.searchsorted(t1_unique, t1_jax).astype(jnp.int64)
        t2_indices = jnp.searchsorted(t2_unique, t2_jax).astype(jnp.int64)

        # Convert to flat grid indices: phi * (n_t1 * n_t2) + t1 * n_t2 + t2
        n_t1 = len(t1_unique)
        n_t2 = len(t2_unique)
        flat_indices = phi_indices * (n_t1 * n_t2) + t1_indices * n_t2 + t2_indices

        return flat_indices, t1_indices, t2_indices

    def _setup_jax_functions(self) -> None:
        """
        Pre-compile JAX functions for performance.

        This method sets up JIT-compiled versions of the residual computation
        to maximize performance during optimization.
        """
        # Note: _compute_chunk_residuals_raw is no longer used — the hot path
        # is _call_jax_vectorized via _setup_vmap_functions.  Skip dead JIT
        # compilation to save ~0.2-0.5s at init.
        self.compute_chunk_jit = None

    def _preconvert_chunk_arrays(self) -> None:
        """
        Pre-convert chunk arrays to JAX arrays during initialization.

        This avoids repeated jnp.asarray() calls inside the optimization loop,
        providing ~10-15% speedup by eliminating array conversion overhead.

        Performance Optimization (Spec 006 - FR-004, FR-005):
        Also creates concatenated arrays (phi_all, t1_all, t2_all, g2_all) and
        chunk_boundaries for device-side iteration with jax.lax.scan.
        """
        self.chunks_jax = []
        for chunk in self.chunks:
            chunk_jax = {
                "phi": jnp.asarray(chunk.phi),
                "t1": jnp.asarray(chunk.t1),
                "t2": jnp.asarray(chunk.t2),
                "g2": jnp.asarray(chunk.g2),
                "q": float(chunk.q),
                "L": float(chunk.L),
                "dt": float(chunk.dt) if chunk.dt is not None else None,
            }
            self.chunks_jax.append(chunk_jax)
        self.logger.debug(f"Pre-converted {len(self.chunks_jax)} chunks to JAX arrays")

        # FR-004, FR-005: Create concatenated arrays for device-side iteration
        # This enables jax.lax.scan instead of Python loops
        self._concatenate_chunk_data()

    def _concatenate_chunk_data(self) -> None:
        """
        Concatenate all chunk data into single arrays for device-side iteration.

        Performance Optimization (Spec 006 - FR-004, FR-005):
        Instead of iterating over chunks in Python, we concatenate all data
        and use chunk_boundaries for index lookup. This enables jax.lax.scan
        for device-side iteration, reducing Python interpreter overhead.

        Attributes Created:
            g2_all: Concatenated g2 observations from all chunks
            flat_indices_all: Concatenated pre-computed flat indices
            t1_indices_all: Concatenated t1 indices for diagonal masking (v2.14.2+)
            t2_indices_all: Concatenated t2 indices for diagonal masking (v2.14.2+)
            chunk_boundaries: Array of boundary indices [0, len(chunk0), len(chunk0)+len(chunk1), ...]
            _chunk_q: q value (same for all chunks)
            _chunk_L: L value (same for all chunks)
            _chunk_dt: dt value (same for all chunks)
        """
        # Concatenate g2 observations
        g2_list = [cast(jnp.ndarray, chunk_jax["g2"]) for chunk_jax in self.chunks_jax]
        self.g2_all = jnp.concatenate(g2_list, axis=0)

        # Concatenate pre-computed flat indices
        self.flat_indices_all = jnp.concatenate(self._precomputed_flat_indices, axis=0)

        # v2.14.2+: Concatenate t1/t2 indices for diagonal masking
        self.t1_indices_all = jnp.concatenate(self._precomputed_t1_indices, axis=0)
        self.t2_indices_all = jnp.concatenate(self._precomputed_t2_indices, axis=0)

        # A2: Precompute the diagonal mask ONCE (was recomputed every iteration
        # inside the residual, polluting the jacfwd tape with two large float
        # temporaries). Diagonal points (t1 == t2) map to identical grid indices
        # by construction, so an integer-exact comparison is both correct and
        # avoids storing the ~370 MB float t1/t2 value arrays at 23M points.
        self._diag_mask = self.t1_indices_all != self.t2_indices_all

        # Compute chunk boundaries for index lookup
        chunk_sizes = [
            len(cast(jnp.ndarray, chunk_jax["g2"])) for chunk_jax in self.chunks_jax
        ]
        boundaries = [0]
        for size in chunk_sizes:
            boundaries.append(boundaries[-1] + size)
        # Use int64 to prevent overflow when cumulative point count exceeds
        # int32 max (2.147B) for large in-core datasets.
        self.chunk_boundaries = jnp.array(boundaries, dtype=jnp.int64)

        # Store common chunk parameters (assumed same for all chunks)
        self._chunk_q = cast(float, self.chunks_jax[0]["q"])
        self._chunk_L = cast(float, self.chunks_jax[0]["L"])
        self._chunk_dt = cast(float | None, self.chunks_jax[0]["dt"])

        # Store global unique arrays (A4: read directly from the precomputed
        # global attrs instead of a per-chunk metadata dict).
        self._phi_unique = self._phi_unique_global
        self._t1_unique = self._t1_unique_global
        self._t2_unique = self._t2_unique_global

        self.logger.debug(
            f"Concatenated chunk data: {len(self.g2_all):,} total points, "
            f"{len(self.chunk_boundaries) - 1} chunks, "
            f"boundaries={list(self.chunk_boundaries[:5])}..."
        )

        # Build stable vmap functions now that chunk metadata is available
        self._setup_vmap_functions()

        # M1: Free intermediate per-chunk data now that everything is
        # concatenated into device-side arrays (g2_all, flat_indices_all, etc.).
        # The hot path (_call_jax_vectorized) uses only the concatenated arrays.
        # Per-chunk lists were only needed by the dead _call_jax_chunked fallback.
        # For a 10M-point dataset this frees ~160+ MB of duplicate JAX arrays.
        del self._precomputed_flat_indices
        del self._precomputed_t1_indices
        del self._precomputed_t2_indices
        del self.chunks_jax

        # Cache diagnostics before freeing original numpy chunks (~320 MB for 10M pts).
        # validate_chunk_structure() and get_diagnostics() use these cached values
        # after chunks are freed; callers no longer need chunks to exist.
        self._cached_n_chunks = self.n_chunks
        self._cached_chunk_sizes = [len(c.g2) for c in self.chunks]
        self._cached_chunk_angle_counts = [len(np.unique(c.phi)) for c in self.chunks]
        self._cached_n_angles = self.n_phi

        # Run structural validation inline while chunks are still available.
        # This preserves the validation guarantee — after del self.chunks the
        # window is closed and validate_chunk_structure() returns the cached result.
        self._validate_chunk_structure_inline()

        del self.chunks

    def _validate_chunk_structure_inline(self) -> None:
        """Run chunk-structure validation while self.chunks is still available.

        Called from _concatenate_chunk_data() immediately before del self.chunks.
        Raises ValueError on failure so the constructor fails fast rather than
        producing a silently corrupt residual function.  On success, records the
        result in self._chunk_structure_valid so validate_chunk_structure() can
        return the cached outcome after chunks have been freed.
        """
        expected_angles = set(np.unique(np.round(self.chunks[0].phi, decimals=6)))
        n_expected = len(expected_angles)

        self.logger.debug(
            f"Inline chunk structure validation: {self.n_chunks} chunks, "
            f"{n_expected} expected angles per chunk"
        )

        for i, chunk in enumerate(self.chunks):
            chunk_angles = set(np.unique(np.round(chunk.phi, decimals=6)))

            if chunk_angles != expected_angles:
                missing = expected_angles - chunk_angles
                extra = chunk_angles - expected_angles
                error_msg = f"Chunk {i} has inconsistent angles:\n"
                if missing:
                    error_msg += f"  Missing: {missing}\n"
                if extra:
                    error_msg += f"  Extra: {extra}\n"
                raise ValueError(error_msg)

            if len(chunk.g2) == 0:
                raise ValueError(f"Chunk {i} has no data points")

            n_points = len(chunk.g2)
            if not (len(chunk.phi) == len(chunk.t1) == len(chunk.t2) == n_points):
                raise ValueError(
                    f"Chunk {i} has inconsistent array shapes: "
                    f"phi={len(chunk.phi)}, t1={len(chunk.t1)}, "
                    f"t2={len(chunk.t2)}, g2={len(chunk.g2)}"
                )

        self._chunk_structure_valid = True
        self.logger.debug("Inline chunk structure validation passed")

    def _setup_vmap_functions(self) -> None:
        """Create vmap-wrapped g2 computation functions once during init.

        Avoids re-creating closures on every NLSQ iteration (fixes #20-analog
        for residual.py). The closures capture stable values (t1_unique, q, L, dt)
        while physical_params is passed as an explicit argument.
        """
        if self._chunk_dt is None:
            self.logger.warning(
                "StratifiedResidualFunction: dt not set (chunk_dt is None); "
                "using dt=0.001 s as fallback. Physics factors may be incorrect."
            )
            dt_value = 0.001
        else:
            dt_value = self._chunk_dt

        # Per-angle scaling: physical_params, phi, contrast, offset all vary
        def _g2_per_angle(
            physical_params: jnp.ndarray,
            phi_val: float | jnp.ndarray,
            contrast_val: float | jnp.ndarray,
            offset_val: float | jnp.ndarray,
        ) -> jnp.ndarray:
            return jnp.squeeze(
                compute_g2_scaled(
                    params=physical_params,
                    t1=self._t1_unique,
                    t2=self._t2_unique,
                    phi=phi_val,
                    q=self._chunk_q,
                    L=self._chunk_L,
                    contrast=contrast_val,
                    offset=offset_val,
                    dt=dt_value,
                ),
                axis=0,
            )

        self._vmap_g2_per_angle = jax.vmap(_g2_per_angle, in_axes=(None, 0, 0, 0))

        # Scalar scaling: contrast/offset are scalars, only phi varies
        def _g2_scalar(
            physical_params: jnp.ndarray,
            contrast_val: float | jnp.ndarray,
            offset_val: float | jnp.ndarray,
            phi_val: float | jnp.ndarray,
        ) -> jnp.ndarray:
            return jnp.squeeze(
                compute_g2_scaled(
                    params=physical_params,
                    t1=self._t1_unique,
                    t2=self._t2_unique,
                    phi=phi_val,
                    q=self._chunk_q,
                    L=self._chunk_L,
                    contrast=contrast_val,
                    offset=offset_val,
                    dt=dt_value,
                ),
                axis=0,
            )

        self._vmap_g2_scalar = jax.vmap(_g2_scalar, in_axes=(None, None, None, 0))

    def _compute_chunk_residuals_raw(
        self,
        g2_obs: jnp.ndarray,
        sigma_full: jnp.ndarray,
        params_all: jnp.ndarray,
        phi_unique: jnp.ndarray,
        t1_unique: jnp.ndarray,
        t2_unique: jnp.ndarray,
        flat_indices: jnp.ndarray,
        t1_indices: jnp.ndarray,
        t2_indices: jnp.ndarray,
        q: float,
        L: float,
        dt: float | None,
    ) -> jnp.ndarray:
        """DEPRECATED: Dead code path -- not called from any live execution path.

        Retained for reference. The live path is _call_jax_vectorized.
        _call_jax routes exclusively to _call_jax_vectorized, and
        _call_jax_chunked raises RuntimeError immediately.
        """
        raise RuntimeError(
            "_compute_chunk_residuals_raw is deprecated. Use _call_jax_vectorized."
        )

    def _call_jax(self, params: jnp.ndarray) -> jnp.ndarray:
        """JAX-native residuals for use in JIT/Jacobian contexts.

        Performance Optimization (Spec 006 - FR-004, FR-005):
        Uses vectorized computation with concatenated arrays instead of Python
        loop over chunks. Computes theory grid ONCE and extracts all values
        using pre-computed flat indices, eliminating per-chunk overhead.

        This replaces the previous loop-based implementation:
        - Old: For each chunk, compute full g2 grid, extract chunk indices
        - New: Compute g2 grid once, extract ALL indices in single operation

        Expected speedup: 20-40% for chunked datasets.
        """
        return self._call_jax_vectorized(params)

    def _call_jax_vectorized(self, params: jnp.ndarray) -> jnp.ndarray:
        """Vectorized residual computation using concatenated arrays.

        Performance Optimization (Spec 006 - FR-004, FR-005):
        Instead of iterating over chunks in Python, computes theoretical g2
        grid ONCE and uses concatenated flat_indices_all to extract all
        values in a single vectorized operation.

        This eliminates:
        1. Python loop overhead
        2. Redundant g2 theory grid computation (was computed per-chunk)
        3. Multiple small kernel launches

        Args:
            params: Parameter array [scaling_params, physical_params]

        Returns:
            Weighted residuals for ALL data points
        """
        # A3: params already arrives as a jnp array on the JIT/Jacobian hot path
        # (the only entry points are jax_residual and __call__, the latter doing
        # its own jnp.asarray). Re-wrapping here was a redundant device op.
        params_jax = params
        sigma_full = self._sigma_jax

        # Extract scaling and physical parameters
        if self.per_angle_scaling:
            contrast = params_jax[: self.n_phi]
            offset = params_jax[self.n_phi : 2 * self.n_phi]
            physical_params = params_jax[2 * self.n_phi :]
        else:
            contrast = params_jax[0]
            offset = params_jax[1]
            physical_params = params_jax[2:]

        # Compute theoretical g2 grid ONCE for all data
        # (Previously computed redundantly per-chunk)
        # Uses pre-built vmap functions (created once in _setup_vmap_functions)
        # to avoid re-creating closures on every NLSQ iteration.
        if self.per_angle_scaling:
            g2_theory_grid = self._vmap_g2_per_angle(
                physical_params, self._phi_unique, contrast, offset
            )
        else:
            g2_theory_grid = self._vmap_g2_scalar(
                physical_params, contrast, offset, self._phi_unique
            )

        # Note: diagonal correction is not applied to the theory grid here.
        # Diagonal points (t1==t2) are masked to zero residuals below,
        # making any theory value at those points irrelevant to the fit.

        # Flatten and extract theory values for ALL points at once
        # (Single indexing operation instead of per-chunk)
        g2_theory_flat = g2_theory_grid.reshape(-1)
        g2_theory_all = g2_theory_flat[self.flat_indices_all]

        # Get sigma values for ALL points (single indexing operation)
        sigma_flat = sigma_full.reshape(-1)
        sigma_all = sigma_flat[self.flat_indices_all]

        # Compute ALL residuals — mask out zero-sigma points entirely
        EPS = 1e-10
        valid_sigma = sigma_all > EPS
        safe_sigma = jnp.where(valid_sigma, sigma_all, 1.0)
        residuals = jnp.where(
            valid_sigma, (self.g2_all - g2_theory_all) / safe_sigma, 0.0
        )

        # v2.14.2+ / A2: Mask diagonal points (t1 == t2) to zero. Diagonal
        # points are autocorrelation artifacts, not physics. The mask is
        # precomputed once in _concatenate_chunk_data (self._diag_mask) using
        # the integer grid indices — diagonal points have identical t1/t2
        # indices by construction, so the comparison is exact and needs no
        # float tolerance. This keeps the residual bitwise-equal to the prior
        # value-based test while removing two large float temporaries from the
        # jacfwd tape (and ~370 MB of stored value arrays at 23M points).
        residuals = jnp.where(self._diag_mask, residuals, 0.0)

        return residuals

    def _call_jax_chunked(self, params: jnp.ndarray) -> jnp.ndarray:
        """Original chunk-based residual computation — REMOVED.

        Per-chunk data was freed after concatenation (M1 memory optimization).
        The vectorized path (_call_jax_vectorized) is used exclusively.
        """
        raise RuntimeError(
            "_call_jax_chunked is unavailable: per-chunk data was freed "
            "after concatenation. Use _call_jax_vectorized instead."
        )

    def jax_residual(self, params: jnp.ndarray) -> jnp.ndarray:
        return self._call_jax(params)

    def __call__(self, params: np.ndarray) -> np.ndarray:
        params_jax = jnp.asarray(params)
        residuals_jax = self._call_jax(params_jax)
        return np.asarray(residuals_jax)

    def validate_chunk_structure(self) -> bool:
        """
        Validate that all chunks contain all phi angles.

        This is a critical validation to ensure per-angle parameter gradients
        will be non-zero. If any chunk is missing an angle, the gradient for
        that angle's parameters will be zero, causing optimization failure.

        Returns:
            True if validation passes

        Raises:
            ValueError: If any chunk is missing angles or has inconsistent structure
        """
        if not hasattr(self, "chunks"):
            # Chunks were freed by _concatenate_chunk_data() after inline validation
            # (_validate_chunk_structure_inline) already ran during __init__.
            # Return the cached result — True means construction succeeded.
            self.logger.info(
                "Chunk structure validation passed (cached -- validated during build)"
            )
            return getattr(self, "_chunk_structure_valid", True)

        # Chunks still live (unusual path, e.g. external test bypass): validate now.
        # Get expected angles from first chunk
        expected_angles = set(np.unique(np.round(self.chunks[0].phi, decimals=6)))
        n_expected = len(expected_angles)

        self.logger.info(
            f"Validating chunk structure: {self.n_chunks} chunks, "
            f"{n_expected} expected angles per chunk"
        )

        # Validate each chunk
        for i, chunk in enumerate(self.chunks):
            chunk_angles = set(np.unique(np.round(chunk.phi, decimals=6)))

            # Check angle completeness
            if chunk_angles != expected_angles:
                missing = expected_angles - chunk_angles
                extra = chunk_angles - expected_angles
                error_msg = f"Chunk {i} has inconsistent angles:\n"
                if missing:
                    error_msg += f"  Missing: {missing}\n"
                if extra:
                    error_msg += f"  Extra: {extra}\n"
                raise ValueError(error_msg)

            # Check for valid data
            if len(chunk.g2) == 0:
                raise ValueError(f"Chunk {i} has no data points")

            # Check array shapes match
            # Note: sigma is stored at parent level (self.sigma), not in chunks
            n_points = len(chunk.g2)
            if not (len(chunk.phi) == len(chunk.t1) == len(chunk.t2) == n_points):
                raise ValueError(
                    f"Chunk {i} has inconsistent array shapes: "
                    f"phi={len(chunk.phi)}, t1={len(chunk.t1)}, "
                    f"t2={len(chunk.t2)}, g2={len(chunk.g2)}"
                )

        self.logger.info("Chunk structure validation passed")
        return True

    def get_diagnostics(self) -> dict[str, Any]:
        """
        Get diagnostic information about the residual function.

        Returns:
            Dictionary containing:
                - n_chunks: Number of chunks
                - n_total_points: Total data points
                - n_angles: Number of unique phi angles
                - per_angle_scaling: Whether per-angle scaling is enabled
                - chunk_sizes: List of points per chunk
                - chunk_angle_counts: List of angles per chunk
                - min_chunk_size: Minimum chunk size
                - max_chunk_size: Maximum chunk size
                - mean_chunk_size: Mean chunk size
        """
        # Use cached arrays when chunks have been freed (normal post-init path).
        # _cached_chunk_sizes and _cached_chunk_angle_counts are set in
        # _concatenate_chunk_data() immediately before del self.chunks.
        if hasattr(self, "_cached_chunk_sizes"):
            chunk_sizes = self._cached_chunk_sizes
            chunk_angle_counts = self._cached_chunk_angle_counts
            n_angles = self._cached_n_angles
        else:
            # Chunks still live — compute directly (unusual path)
            chunk_sizes = [len(chunk.g2) for chunk in self.chunks]
            chunk_angle_counts = [len(np.unique(chunk.phi)) for chunk in self.chunks]
            n_angles = len(np.unique(self.chunks[0].phi))

        diagnostics = {
            "n_chunks": self.n_chunks,
            "n_total_points": self.n_total_points,
            "n_angles": n_angles,
            "per_angle_scaling": self.per_angle_scaling,
            "chunk_sizes": chunk_sizes,
            "chunk_angle_counts": chunk_angle_counts,
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "mean_chunk_size": np.mean(chunk_sizes),
        }

        return diagnostics

    def log_diagnostics(self) -> None:
        """Log diagnostic information for monitoring."""
        diag = self.get_diagnostics()
        self.logger.info(
            f"StratifiedResidualFunction diagnostics:\n"
            f"  Chunks: {diag['n_chunks']}\n"
            f"  Total points: {diag['n_total_points']:,}\n"
            f"  Angles: {diag['n_angles']}\n"
            f"  Per-angle scaling: {diag['per_angle_scaling']}\n"
            f"  Chunk sizes: min={diag['min_chunk_size']:,}, "
            f"max={diag['max_chunk_size']:,}, mean={diag['mean_chunk_size']:.0f}\n"
            f"  Angle counts per chunk: {set(diag['chunk_angle_counts'])}"
        )


def create_stratified_residual_function(
    stratified_data: Any,
    per_angle_scaling: bool,
    physical_param_names: list[str],
    logger: logging.Logger | None = None,
    validate: bool = True,
) -> StratifiedResidualFunction:
    """
    Factory function to create and validate a stratified residual function.

    This is a convenience function that creates a StratifiedResidualFunction,
    optionally validates its structure, and logs diagnostics.

    Args:
        stratified_data: Object with .chunks attribute containing angle-stratified chunks
        per_angle_scaling: Whether per-angle scaling parameters are used
        physical_param_names: List of physical parameter names (e.g., ['D0', 'alpha', 'D_offset'])
        logger: Optional logger for diagnostics
        validate: Whether to validate chunk structure (recommended)

    Returns:
        Validated StratifiedResidualFunction instance

    Raises:
        ValueError: If validation fails

    Example:
        >>> residual_fn = create_stratified_residual_function(
        ...     stratified_data=stratified_data,
        ...     per_angle_scaling=True,
        ...     physical_param_names=['D0', 'alpha', 'D_offset'],
        ...     validate=True
        ... )
        >>> residual_fn.log_diagnostics()
    """
    residual_fn = StratifiedResidualFunction(
        stratified_data=stratified_data,
        per_angle_scaling=per_angle_scaling,
        physical_param_names=physical_param_names,
        logger=logger,
    )

    if validate:
        residual_fn.validate_chunk_structure()

    residual_fn.log_diagnostics()

    return residual_fn
