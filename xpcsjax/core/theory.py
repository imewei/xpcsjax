"""Theory Computation Engine for Homodyne
==========================================

High-level interface to theoretical calculations for homodyne scattering analysis.
This module provides user-friendly wrappers around the JAX backend functions
with proper error handling, validation, and computational management.

The theory engine handles:
- Model selection and parameter management
- Efficient computation orchestration
- Memory management for large datasets
- Error handling and validation
- Performance monitoring and optimization hints
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from xpcsjax.utils.logging import get_logger, log_performance

# Import with fallback handling
try:
    from xpcsjax.core.jax_backend import (
        batch_chi_squared,
        jax_available,
        jnp,
    )
    from xpcsjax.core.physics_utils import safe_len
except ImportError:
    jax_available = False
    safe_len: Callable[..., int] = len  # type: ignore[no-redef]
    logger = get_logger(__name__)
    logger.error("Could not import JAX backend - theory computations disabled")

from xpcsjax.core.models import create_model
from xpcsjax.core.physics import PhysicsConstants

logger = get_logger(__name__)


class TheoryEngine:
    """High-level interface for theoretical homodyne calculations.

    Manages model selection, parameter validation, and efficient
    computation orchestration for homodyne scattering analysis.
    """

    def __init__(self, analysis_mode: str = "laminar_flow"):
        """Initialize theory engine with specified analysis mode.

        Args:
            analysis_mode: "static" or "laminar_flow"
        """
        self.analysis_mode = analysis_mode
        self.model = create_model(analysis_mode)
        self._validate_backend()

        logger.info(f"Theory engine initialized for {analysis_mode}")

    def _validate_backend(self) -> None:
        """Validate that computational backend is available."""
        if not jax_available:
            logger.warning("JAX backend not available - computations will be slower")

    @log_performance(threshold=0.01)
    def compute_g1(
        self,
        params: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        dt: float | None = None,
    ) -> Any:
        """Compute g1 correlation function.

        Args:
            params: Physical parameters
            t1, t2: Time grids
            phi: Angle grid
            q: Wave vector magnitude
            L: Sample-detector distance
            dt: Time step (if None, will be estimated from t1)

        Returns:
            g1 correlation function
        """
        # Validate inputs
        self._validate_computation_inputs(params, q, L)

        # Convert to JAX arrays if needed
        if jax_available:
            params_jax: Any = jnp.asarray(params, dtype=jnp.float64)
            t1_jax: Any = jnp.asarray(t1, dtype=jnp.float64)
            t2_jax: Any = jnp.asarray(t2, dtype=jnp.float64)
            phi_jax: Any = jnp.asarray(phi, dtype=jnp.float64)
        else:
            params_jax = params
            t1_jax = t1
            t2_jax = t2
            phi_jax = phi

        dt_arg: Any = dt
        return self.model.compute_g1(params_jax, t1_jax, t2_jax, phi_jax, q, L, dt_arg)

    @log_performance(threshold=0.01)
    def compute_g2(
        self,
        params: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        contrast: float,
        offset: float,
        dt: float | None = None,
    ) -> Any:
        """Compute g2 with scaled fitting: g₂ = offset + contrast × [g₁]²

        This is the core equation for homodyne analysis.

        Args:
            params: Physical parameters
            t1, t2: Time grids
            phi: Angle grid
            q: Wave vector magnitude
            L: Sample-detector distance
            contrast: Contrast parameter
            offset: Baseline offset
            dt: Time step in seconds. Required — unlike compute_g1, there is no
                dt estimation fallback for g2 because sinc_prefactor requires an
                exact dt. Pass dt explicitly from configuration.

        Returns:
            g2 correlation function

        Raises:
            ValueError: If dt is None.
        """
        # Fail fast at the API boundary with a clear message rather than letting
        # CombinedModel.compute_g2 raise a cryptic TypeError from an inner layer.
        if dt is None:
            raise ValueError(
                "TheoryEngine.compute_g2 requires an explicit dt (time step in seconds). "
                "Pass dt from configuration, e.g. dt=config.dt. "
                "Unlike compute_g1, there is no estimation fallback for compute_g2."
            )

        # Validate inputs
        self._validate_computation_inputs(params, q, L)
        self._validate_scaling_parameters(contrast, offset)

        # Convert to JAX arrays if needed
        if jax_available:
            params_jax: Any = jnp.asarray(params, dtype=jnp.float64)
            t1_jax: Any = jnp.asarray(t1, dtype=jnp.float64)
            t2_jax: Any = jnp.asarray(t2, dtype=jnp.float64)
            phi_jax: Any = jnp.asarray(phi, dtype=jnp.float64)
        else:
            params_jax = params
            t1_jax = t1
            t2_jax = t2
            phi_jax = phi

        dt_arg: Any = dt
        return self.model.compute_g2(
            params_jax, t1_jax, t2_jax, phi_jax, q, L, contrast, offset, dt_arg
        )

    @log_performance(threshold=0.05)
    def compute_chi_squared(
        self,
        params: np.ndarray,
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        contrast: float,
        offset: float,
    ) -> float:
        """Compute chi-squared goodness of fit.

        Args:
            params: Physical parameters
            data: Experimental correlation data
            sigma: Measurement uncertainties
            t1, t2: Time grids
            phi: Angle grid
            q: Wave vector magnitude
            L: Sample-detector distance
            contrast, offset: Scaling parameters

        Returns:
            Chi-squared value
        """
        # Validate inputs
        self._validate_computation_inputs(params, q, L)
        self._validate_scaling_parameters(contrast, offset)
        self._validate_data_inputs(data, sigma, t1, t2, phi)

        # Convert to JAX arrays if needed
        if jax_available:
            params_jax: Any = jnp.asarray(params, dtype=jnp.float64)
            data_jax: Any = jnp.asarray(data, dtype=jnp.float64)
            sigma_jax: Any = jnp.asarray(sigma, dtype=jnp.float64)
            t1_jax: Any = jnp.asarray(t1, dtype=jnp.float64)
            t2_jax: Any = jnp.asarray(t2, dtype=jnp.float64)
            phi_jax: Any = jnp.asarray(phi, dtype=jnp.float64)
        else:
            params_jax = params
            data_jax = data
            sigma_jax = sigma
            t1_jax = t1
            t2_jax = t2
            phi_jax = phi

        return self.model.compute_chi_squared(
            params_jax,
            data_jax,
            sigma_jax,
            t1_jax,
            t2_jax,
            phi_jax,
            q,
            L,
            contrast,
            offset,
        )

    @log_performance(threshold=0.1)
    def batch_computation(
        self,
        params_batch: np.ndarray,
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        contrast: float,
        offset: float,
    ) -> Any:
        """Compute chi-squared for multiple parameter sets efficiently.

        Leverages JAX vectorization for optimal performance.

        Args:
            params_batch: Array of parameter sets (n_sets, n_params)
            data: Experimental correlation data
            sigma: Measurement uncertainties
            t1, t2: Time grids
            phi: Angle grid
            q: Wave vector magnitude
            L: Sample-detector distance
            contrast, offset: Scaling parameters

        Returns:
            Chi-squared values for each parameter set
        """
        # Validate batch input
        if params_batch.ndim != 2:
            raise ValueError("params_batch must be 2D array (n_sets, n_params)")

        n_sets, n_params = params_batch.shape
        if n_params != self.model.n_params:
            raise ValueError(
                f"Expected {self.model.n_params} parameters, got {n_params}",
            )

        logger.debug(f"Batch computation for {n_sets} parameter sets")

        # Convert to JAX arrays if needed
        if jax_available:
            params_batch_jax: Any = jnp.asarray(params_batch, dtype=jnp.float64)
            data_jax: Any = jnp.asarray(data, dtype=jnp.float64)
            sigma_jax: Any = jnp.asarray(sigma, dtype=jnp.float64)
            t1_jax: Any = jnp.asarray(t1, dtype=jnp.float64)
            t2_jax: Any = jnp.asarray(t2, dtype=jnp.float64)
            phi_jax: Any = jnp.asarray(phi, dtype=jnp.float64)

            return batch_chi_squared(
                params_batch_jax,
                data_jax,
                sigma_jax,
                t1_jax,
                t2_jax,
                phi_jax,
                q,
                L,
                contrast,
                offset,
            )
        else:
            # Fallback: loop over parameter sets
            results = []
            for params in params_batch:
                chi2 = self.compute_chi_squared(
                    params,
                    data,
                    sigma,
                    t1,
                    t2,
                    phi,
                    q,
                    L,
                    contrast,
                    offset,
                )
                results.append(chi2)
            return np.array(results)

    def estimate_computation_cost(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
    ) -> dict[str, Any]:
        """Estimate computational cost for given data dimensions.

        Helps with performance planning and memory management.

        Args:
            t1, t2: Time grids
            phi: Angle grid

        Returns:
            Cost estimation dictionary
        """
        n_time_pairs = safe_len(t1) * safe_len(t2)
        phi_array = np.atleast_1d(phi)
        n_angles = safe_len(phi_array)
        n_total_points = n_time_pairs * n_angles

        # Rough performance estimates (operations per point)
        ops_per_point = {
            "static": 10,  # Diffusion only
            "static_isotropic": 10,  # Diffusion only
            "static_anisotropic": 15,  # Diffusion with anisotropy
            "laminar_flow": 50,  # Full model with shear
        }

        base_ops = ops_per_point.get(self.analysis_mode, 50)
        total_ops = n_total_points * base_ops

        # Memory estimates (bytes per point, rough)
        memory_per_point = 8 * 4  # ~4 float64 values per point
        total_memory_mb = (n_total_points * memory_per_point) / (1024**2)

        return {
            "n_time_pairs": n_time_pairs,
            "n_angles": n_angles,
            "n_total_points": n_total_points,
            "estimated_operations": total_ops,
            "estimated_memory_mb": total_memory_mb,
            "analysis_mode": self.analysis_mode,
            "backend": "JAX" if jax_available else "NumPy",
            "performance_tier": self._classify_performance_tier(total_ops),
        }

    def _classify_performance_tier(self, operations: int) -> str:
        """Classify computation as light, medium, or heavy."""
        if operations < 1e6:
            return "light"
        elif operations < 1e8:
            return "medium"
        else:
            return "heavy"

    def _validate_computation_inputs(
        self, params: np.ndarray, q: float, L: float
    ) -> None:
        """Validate core computation inputs."""
        # Skip parameter validation inside JIT compilation to avoid JAX tracer errors.
        # q and L are Python floats (not tracers), so we CAN validate them in JAX mode.
        if not jax_available:
            # Parameter validation only works with concrete (non-traced) values
            params_any: Any = params
            if not self.model.validate_parameters(params_any):
                logger.warning(
                    "Parameters outside recommended bounds - results may be unreliable",
                )

        # Experimental setup validation (q, L are Python floats, safe in all modes)
        if q <= 0:
            raise ValueError(f"Wave vector q must be positive, got {q}")
        if L <= 0:
            raise ValueError(f"Sample-detector distance L must be positive, got {L}")

        # Physical reasonableness checks
        if not (PhysicsConstants.Q_MIN_TYPICAL <= q <= PhysicsConstants.Q_MAX_TYPICAL):
            logger.warning(
                f"q = {q:.2e} outside typical range - check experimental setup",
            )
        # L is in Angstroms - check reasonable range
        # Typical range: 100,000 Å (10 μm) to 100,000,000 Å (10 mm)
        # Note: 1 Å = 1e-10 m, so 1e5 Å = 10 μm, 1e8 Å = 10 mm.
        if not (1e5 <= L <= 1e8):
            logger.warning(
                f"L = {L:.1f} AA outside typical range [1e5, 1e8] AA (10 um to 10 mm) - check experimental setup",
            )

    def _validate_scaling_parameters(self, contrast: float, offset: float) -> None:
        """Validate scaling parameters."""
        # Skip validation only for JAX tracers (inside @jit), not all JAX mode
        if jax_available:
            import jax.core as jax_core

            if isinstance(contrast, jax_core.Tracer) or isinstance(  # type: ignore[unreachable]
                offset,  # type: ignore[unreachable]
                jax_core.Tracer,
            ):
                return  # type: ignore[unreachable]

        if contrast <= 0:
            raise ValueError(f"Contrast must be positive, got {contrast}")
        if offset < 0:
            logger.warning(f"Negative offset {offset} - check baseline correction")

    def _validate_data_inputs(
        self,
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
    ) -> None:
        """Validate experimental data inputs."""
        # Skip validation only for JAX tracers (inside @jit), not all JAX mode
        if jax_available:
            import jax.core as jax_core

            if isinstance(data, jax_core.Tracer):  # type: ignore[unreachable]
                return  # type: ignore[unreachable]

        # Shape consistency
        phi_array = np.atleast_1d(phi)
        expected_shape = (safe_len(phi_array), safe_len(t1), safe_len(t2))
        if data.shape != expected_shape:
            raise ValueError(
                f"Data shape {data.shape} doesn't match expected {expected_shape}",
            )
        if sigma.shape != expected_shape:
            raise ValueError(
                f"Sigma shape {sigma.shape} doesn't match expected {expected_shape}",
            )

        # Data quality checks
        if np.any(sigma <= 0):
            raise ValueError("All uncertainties must be positive")
        if np.any(~np.isfinite(data)):
            raise ValueError("Data contains non-finite values")
        if np.any(~np.isfinite(sigma)):
            raise ValueError("Uncertainties contain non-finite values")

    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive model and engine information."""
        info = self.model.get_model_info()
        info.update(
            {
                "theory_engine_version": "2.0",
                "backend_available": jax_available,
                "supports_batch_computation": jax_available,
            },
        )
        return info

    def __repr__(self) -> str:
        backend = "JAX" if jax_available else "NumPy"
        return f"TheoryEngine(mode='{self.analysis_mode}', backend={backend})"


# Convenience functions for direct computation
def compute_g2_theory(
    params: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    analysis_mode: str = "laminar_flow",
    dt: float | None = None,
) -> Any:
    """Direct computation of g2 theory. Convenience wrapper for one-off calculations.

    Note: Creates a new TheoryEngine per call (includes model init overhead, logger
    I/O, and jnp.array construction). For repeated calls (e.g. parameter sweeps),
    create a single TheoryEngine instance and call engine.compute_g2() directly.

    Args:
        params: Physical parameters
        t1, t2: Time grids
        phi: Angle grid
        q: Wave vector magnitude
        L: Sample-detector distance
        contrast, offset: Scaling parameters
        analysis_mode: Analysis mode
        dt: Time step in seconds. Required for g2 (no estimation fallback).

    Returns:
        g2 correlation function
    """
    engine = TheoryEngine(analysis_mode)
    return engine.compute_g2(params, t1, t2, phi, q, L, contrast, offset, dt)


def compute_chi2_theory(
    params: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    analysis_mode: str = "laminar_flow",
) -> float:
    """Direct computation of chi-squared with minimal overhead.

    Args:
        params: Physical parameters
        data: Experimental data
        sigma: Uncertainties
        t1, t2: Time grids
        phi: Angle grid
        q: Wave vector magnitude
        L: Sample-detector distance
        contrast, offset: Scaling parameters
        analysis_mode: Analysis mode

    Returns:
        Chi-squared value
    """
    engine = TheoryEngine(analysis_mode)
    return engine.compute_chi_squared(
        params,
        data,
        sigma,
        t1,
        t2,
        phi,
        q,
        L,
        contrast,
        offset,
    )


# Export main classes and functions
__all__ = [
    "TheoryEngine",
    "compute_g2_theory",
    "compute_chi2_theory",
]
