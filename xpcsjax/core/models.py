"""Physical Models for XPCS Homodyne Analysis
=============================================

Object-oriented interface to the physical models implemented in the JAX backend.
Provides structured access to diffusion, shear, and combined models with
parameter validation and configuration management.

Homodyne Model
--------------------------
The measured intensity correlation uses per-angle scaling:

    c2(φ, t₁, t₂) = offset + contrast × [c1(φ, t₁, t₂)]²

with a separable field correlation function:

    c1(φ, t₁, t₂) = c1_diff(t₁, t₂) × c1_shear(φ, t₁, t₂)

Diffusion contribution:

    c1_diff(t₁, t₂) = exp[-(q² / 2) ∫|t₂ - t₁| D(t') dt']

Shear contribution:

    c1_shear(φ, t₁, t₂) = [sinc(Φ(φ, t₁, t₂))]²
    Φ(φ, t₁, t₂) = (1 / 2π) · q · L · cos(φ₀ - φ) · ∫|t₂ - t₁| γ̇(t') dt'

Time-dependent transport coefficients:

    D(t) = D₀ · t^α + D_offset
    γ̇(t) = γ̇₀ · t^β + γ̇_offset

Parameter sets:
- Static mode (3 params): D₀, α, D_offset (γ̇₀, β, γ̇_offset, φ₀ fixed/irrelevant)
- Laminar flow (7 params): D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀

Experimental parameters:
- q: scattering wavevector magnitude [Å⁻¹]
- L: gap/characteristic length [Å]
- φ: scattering angle [degrees]
- dt: frame time step [s]
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from xpcsjax.core.jax_backend import (
    compute_chi_squared,
    compute_g1_diffusion,
    compute_g1_shear,
    compute_g1_total,
    compute_g2_scaled,
    jax_available,
    jnp,
)
from xpcsjax.core.model_mixins import (
    BenchmarkingMixin,
    GradientCapabilityMixin,
    OptimizationRecommendationMixin,
)
from xpcsjax.core.physics import validate_parameters
from xpcsjax.core.physics_utils import safe_len
from xpcsjax.utils.logging import get_logger, log_calls

logger = get_logger(__name__)


class PhysicsModelBase(ABC):
    """Abstract base class for all physical models.

    Defines the interface that all models must implement and provides
    common functionality for parameter management and validation.
    """

    def __init__(self, name: str, parameter_names: list[str]):
        """Initialize base model.

        Args:
            name: Model name for identification
            parameter_names: List of parameter names in order
        """
        self.name = name
        self.parameter_names = parameter_names
        self.n_params = len(parameter_names)
        self._bounds = None
        self._default_values = None

    @abstractmethod
    def compute_g1(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        dt: float | None = None,
    ) -> jnp.ndarray:
        """Compute g1 correlation function for this model."""

    @abstractmethod
    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for optimization."""

    @abstractmethod
    def get_default_parameters(self) -> jnp.ndarray:
        """Get default parameter values."""

    def validate_parameters(self, params: jnp.ndarray) -> bool:
        """Validate parameter values against bounds and constraints."""
        return validate_parameters(params, self.get_parameter_bounds())  # type: ignore[arg-type]

    def get_parameter_dict(self, params: jnp.ndarray) -> dict[str, float]:
        """Convert parameter array to named dictionary."""
        # Ensure params is at least 1D to avoid 0D array indexing issues
        if jax_available and hasattr(params, "ndim"):
            # Convert JAX arrays to NumPy for safe indexing
            params_np = np.atleast_1d(np.asarray(params))
        else:
            params_np = np.atleast_1d(params)

        params_len = safe_len(params_np)
        if params_len != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {params_len}")

        # Convert to regular Python floats only when safe to do so
        try:
            # Try converting to float - will fail if in JIT context
            return {
                name: float(val)
                for name, val in zip(self.parameter_names, params_np, strict=False)
            }
        except (TypeError, ValueError, AttributeError):
            # In JIT context, keep as JAX arrays
            return dict(zip(self.parameter_names, params_np, strict=False))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', n_params={self.n_params})"
        )


class DiffusionModel(PhysicsModelBase):
    """Anomalous diffusion model: D(t) = D₀ t^α + D_offset

    Parameters:
    - D₀: Reference diffusion coefficient [Å²/s]
    - α: Diffusion time-dependence exponent [-]
    - D_offset: Baseline diffusion [Å²/s]

    Physical interpretation:
    - α = 0: Normal diffusion (Brownian motion)
    - α > 0: Super-diffusion (enhanced mobility)
    - α < 0: Sub-diffusion (restricted mobility)
    - D_offset: Residual diffusion at t=0
    """

    def __init__(self) -> None:
        super().__init__(
            name="anomalous_diffusion",
            parameter_names=["D0", "alpha", "D_offset"],
        )

    @log_calls(include_args=False)
    def compute_g1(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        dt: float | None = None,
    ) -> jnp.ndarray:
        """Compute diffusion contribution to g1.

        g₁_diff = exp[-q²/2 ∫|t₂-t₁| D(t')dt']
        """
        # Skip validation inside JIT to avoid JAX tracer boolean conversion errors
        # if not self.validate_parameters(params):
        #     logger.warning("Invalid diffusion parameters - results may be unreliable")

        # Pass q directly without conversion to avoid JAX tracing issues
        # The backend functions handle any necessary conversions

        return compute_g1_diffusion(params, t1, t2, q, dt)

    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Standard bounds for diffusion parameters."""
        return [
            (100.0, 1e5),  # D0: 100 to 1e5 Å²/s
            (-2.0, 2.0),  # alpha: -2 to 2
            (-1e5, 1e5),  # D_offset: -1e5 to 1e5 Å²/s
        ]

    def get_default_parameters(self) -> jnp.ndarray:
        """Default values for typical XPCS measurements."""
        return jnp.array([100.0, 0.0, 10.0])  # Normal diffusion with small offset


class ShearModel(PhysicsModelBase):
    """Time-dependent shear model: γ̇(t) = γ̇₀ t^β + γ̇_offset

    Parameters:
    - γ̇₀: Reference shear rate [s⁻¹]
    - β: Shear rate time-dependence exponent [-]
    - γ̇_offset: Baseline shear rate [s⁻¹]
    - φ₀: Angular offset parameter [degrees]

    Physical interpretation:
    - β = 0: Constant shear rate (steady shear)
    - β > 0: Increasing shear rate with time
    - β < 0: Decreasing shear rate with time
    - φ₀: Preferred flow direction angle
    """

    def __init__(self) -> None:
        super().__init__(
            name="time_dependent_shear",
            parameter_names=["gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"],
        )

    @log_calls(include_args=False)
    def compute_g1(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        dt: float | None = None,
    ) -> jnp.ndarray:
        """Compute shear contribution to g1.

        g₁_shear = [sinc(Φ)]² where Φ = (qL/2π) cos(φ₀-φ) ∫|t₂-t₁| γ̇(t') dt'
        """
        # Skip validation inside JIT to avoid JAX tracer boolean conversion errors
        # if not self.validate_parameters(params):
        #     logger.warning("Invalid shear parameters - results may be unreliable")

        # Pass q directly without conversion to avoid JAX tracing issues
        # The backend functions handle any necessary conversions

        # Create full parameter array with dummy diffusion parameters
        full_params = jnp.concatenate([jnp.array([100.0, 0.0, 10.0]), params])
        return compute_g1_shear(full_params, t1, t2, phi, q, L, dt)  # type: ignore[arg-type]

    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Standard bounds for shear parameters."""
        return [
            (1e-6, 0.5),  # gamma_dot_t0: 1e-6 to 0.5 s⁻¹
            (-2.0, 2.0),  # beta: -2 to 2
            (-0.1, 0.1),  # gamma_dot_t_offset: -0.1 to 0.1 s⁻¹
            (-10.0, 10.0),  # phi0: -10 to 10 degrees
        ]

    def get_default_parameters(self) -> jnp.ndarray:
        """Default values for typical shear flow."""
        return jnp.array([0.01, 0.0, 0.0, 0.0])  # Constant shear, zero offset


class CombinedModel(
    PhysicsModelBase,
    GradientCapabilityMixin,
    BenchmarkingMixin,
    OptimizationRecommendationMixin,
):
    """Combined diffusion + shear model for complete XPCS homodyne analysis.

    This is the full model used for laminar flow analysis with both
    anomalous diffusion and time-dependent shear.

    Parameters (7 total):
    - D₀, α, D_offset: Diffusion parameters
    - γ̇₀, β, γ̇_offset: Shear parameters
    - φ₀: Angular offset parameter

    For static analysis, only the first 3 diffusion parameters are used.

    Mixin capabilities:
    - GradientCapabilityMixin: gradient/Hessian access with backend selection
    - BenchmarkingMixin: performance benchmarking and accuracy validation
    - OptimizationRecommendationMixin: optimization guidance and model info
    """

    def __init__(self, analysis_mode: str = "laminar_flow"):
        """Initialize combined model.

        Args:
            analysis_mode: "static" or "laminar_flow"
        """
        self.analysis_mode = analysis_mode

        if analysis_mode in ("static", "static_isotropic", "static_anisotropic"):
            # Static mode: only diffusion parameters
            parameter_names = ["D0", "alpha", "D_offset"]
            name = "static_diffusion"
        else:
            # Laminar flow mode: all parameters
            parameter_names = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]
            name = "laminar_flow_complete"

        super().__init__(name=name, parameter_names=parameter_names)

        # Create component models
        self.diffusion_model = DiffusionModel()
        self.shear_model = ShearModel()

    @log_calls(include_args=False)
    def compute_g1(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        dt: float | None = None,
    ) -> jnp.ndarray:
        """Compute total g1 = g1_diffusion × g1_shear."""
        # Skip validation inside JIT to avoid JAX tracer boolean conversion errors
        # if not self.validate_parameters(params):
        #     logger.warning(
        #         "Invalid combined model parameters - results may be unreliable"
        #     )

        # Pass q directly without conversion to avoid JAX tracing issues
        # The backend functions handle any necessary conversions

        if self.analysis_mode.startswith("static"):
            # Static mode: only diffusion, no shear
            if logger.isEnabledFor(10):  # DEBUG
                logger.debug(
                    "CombinedModel.compute_g1: calling compute_g1_diffusion with params.shape=%s",
                    params.shape,
                )
            return compute_g1_diffusion(params, t1, t2, q, dt)
        else:
            # Laminar flow mode: full model
            if logger.isEnabledFor(10):  # DEBUG
                logger.debug(
                    "CombinedModel.compute_g1: calling compute_g1_total with params.shape=%s, t1.shape=%s, t2.shape=%s, phi.shape=%s, q=%s, L=%s, dt=%s",
                    params.shape,
                    t1.shape,
                    t2.shape,
                    phi.shape,
                    q,
                    L,
                    dt,
                )
            try:
                result = compute_g1_total(params, t1, t2, phi, q, L, dt)
                # Note: Skip debug logging of result values when traced by JAX
                # (jax.vmap/jit creates BatchTracer objects that can't be formatted)
                if logger.isEnabledFor(10):  # DEBUG level
                    try:
                        # Use nanmin/nanmax: g1 result may contain NaN from failed shards.
                        min_val = float(jnp.nanmin(result))
                        max_val = float(jnp.nanmax(result))
                        logger.debug(
                            f"CombinedModel.compute_g1: compute_g1_total completed, result.shape={result.shape}, min={min_val:.6e}, max={max_val:.6e}",
                        )
                    except (TypeError, ValueError):
                        # Likely a JAX tracer object during tracing
                        logger.debug(
                            f"CombinedModel.compute_g1: compute_g1_total completed, result.shape={result.shape}",
                        )
                return result
            # P2-R6-07: Narrow broad except — realistic failures from compute_g1_total
            # are ValueError (bad params), RuntimeError (XLA), or ArithmeticError.
            # Bare raise preserves the original traceback for all exception types.
            except (ValueError, RuntimeError, ArithmeticError) as e:
                logger.error(
                    f"CombinedModel.compute_g1: compute_g1_total failed with error: {e}",
                )
                logger.error("CombinedModel.compute_g1: traceback:", exc_info=True)
                raise

    def compute_g1_batch(
        self,
        params: jnp.ndarray,
        t1_batch: jnp.ndarray,
        t2_batch: jnp.ndarray,
        phi_batch: jnp.ndarray,
        q: float,
        L: float,
        dt: float | None = None,
    ) -> jnp.ndarray:
        """Compute g1 for a batch of points using vmap.

        Performance Optimization (Spec 001 - FR-006, T041): Vectorized computation
        using jax.vmap for batched point-wise g1 calculation, replacing Python loops.

        Parameters
        ----------
        params : jnp.ndarray
            Physical parameters array
        t1_batch : jnp.ndarray
            Batch of t1 values, shape (n_points,)
        t2_batch : jnp.ndarray
            Batch of t2 values, shape (n_points,)
        phi_batch : jnp.ndarray
            Batch of phi values, shape (n_points,)
        q : float
            Scattering wave vector magnitude [Å⁻¹]
        L : float
            Sample-detector distance (stator_rotor_gap) [Å]
        dt : float, optional
            Time step from configuration [s]

        Returns
        -------
        jnp.ndarray
            Batch of g1 values, shape (n_points,)
        """
        import jax

        # Cache the vmap'd function on first call to avoid JIT retrace overhead.
        # The closure captures `self` — same instance across calls preserves
        # function identity for JAX's trace cache.
        if not hasattr(self, "_cached_g1_vmap"):

            def _compute_g1_single(
                params_inner: Any,
                t1_val: Any,
                t2_val: Any,
                phi_val: Any,
                q_inner: Any,
                L_inner: Any,
                dt_inner: Any,
            ) -> Any:
                g1 = self.compute_g1(
                    params_inner,
                    jnp.array([t1_val]),
                    jnp.array([t2_val]),
                    jnp.array([phi_val]),
                    q_inner,
                    L_inner,
                    dt_inner,
                )
                return g1.flatten()[0]

            self._cached_g1_vmap = jax.vmap(
                _compute_g1_single,
                in_axes=(None, 0, 0, 0, None, None, None),
            )

        result: jnp.ndarray = self._cached_g1_vmap(
            params, t1_batch, t2_batch, phi_batch, q, L, dt
        )
        return result

    @log_calls(include_args=False)
    def compute_g2(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        contrast: float,
        offset: float,
        dt: float,
    ) -> jnp.ndarray:
        """Compute g2 with scaled fitting: g₂ = offset + contrast × [g₁]²

        Parameters
        ----------
        params : jnp.ndarray
            Physical parameters array
        t1, t2 : jnp.ndarray
            Time grids for correlation calculation
        phi : jnp.ndarray
            Scattering angles in degrees
        q : float
            Scattering wave vector magnitude [Å⁻¹]
        L : float
            Sample-detector distance (stator_rotor_gap) [Å]
        contrast : float
            Contrast parameter (β in literature)
        offset : float
            Baseline offset
        dt : float
            Time step from configuration [s] (REQUIRED).
            Fallback estimation has been removed for safety.

        Returns
        -------
        jnp.ndarray
            g2 correlation function

        Raises
        ------
        TypeError
            If dt is None (no longer accepts None)
        ValueError
            If dt <= 0 or not finite
        """
        # Validate dt before passing to backend
        if dt is None:
            raise TypeError(
                "dt parameter is required and cannot be None. "
                "Pass dt explicitly from configuration.",
            )

        # Pass to functional backend
        # The backend functions handle additional validation
        return compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt)

    @log_calls(include_args=False)
    def compute_chi_squared(
        self,
        params: jnp.ndarray,
        data: jnp.ndarray,
        sigma: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        contrast: float,
        offset: float,
    ) -> float:
        """Compute chi-squared goodness of fit."""
        result: float = compute_chi_squared(
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
        return result

    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Get bounds appropriate for analysis mode."""
        bounds = self.diffusion_model.get_parameter_bounds()

        if not self.analysis_mode.startswith("static"):
            # Add shear parameter bounds for laminar flow
            bounds.extend(self.shear_model.get_parameter_bounds())

        return bounds

    def get_default_parameters(self) -> jnp.ndarray:
        """Get default parameters appropriate for analysis mode."""
        defaults = self.diffusion_model.get_default_parameters()

        if not self.analysis_mode.startswith("static"):
            # Add shear parameter defaults for laminar flow
            shear_defaults = self.shear_model.get_default_parameters()
            defaults = jnp.concatenate([defaults, shear_defaults])

        return defaults

    # Mixin methods are inherited from:
    # - GradientCapabilityMixin: get_gradient_function, get_hessian_function,
    #   supports_gradients, get_best_gradient_method, get_gradient_capabilities
    # - BenchmarkingMixin: benchmark_gradient_performance, validate_gradient_accuracy
    # - OptimizationRecommendationMixin: get_optimization_recommendations, get_model_info


# Factory functions for easy model creation
def create_model(analysis_mode: str) -> CombinedModel:
    """Factory function to create appropriate model for analysis mode.

    Args:
        analysis_mode: "static" or "laminar_flow"

    Returns:
        Configured CombinedModel instance
    """
    valid_modes = ["static", "laminar_flow", "static_isotropic", "static_anisotropic"]
    if analysis_mode not in valid_modes:
        raise ValueError(
            f"Invalid analysis mode '{analysis_mode}'. Must be one of {valid_modes}",
        )

    logger.info(f"Creating model for analysis mode: {analysis_mode}")
    return CombinedModel(analysis_mode=analysis_mode)


def get_available_models() -> list[str]:
    """Get list of available analysis modes."""
    return ["static", "laminar_flow", "static_isotropic", "static_anisotropic"]


# Export main classes and functions
__all__ = [
    "PhysicsModelBase",
    "DiffusionModel",
    "ShearModel",
    "CombinedModel",
    "create_model",
    "get_available_models",
]
