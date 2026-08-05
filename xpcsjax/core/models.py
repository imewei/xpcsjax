"""Physical models for XPCS homodyne analysis.

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

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.core.jax_backend import (
    _compute_g1_diffusion_core,
    _compute_g1_total_core,
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
from xpcsjax.core.physics_utils import PI, safe_len
from xpcsjax.utils.logging import get_logger, log_calls

logger = get_logger(__name__)


class PhysicsModelBase(ABC):
    """Abstract base class for all physical models.

    Defines the interface that all models must implement and provides
    common functionality for parameter management and validation.
    """

    def __init__(self, name: str, parameter_names: list[str]):
        """Initialize the base model.

        Parameters
        ----------
        name : str
            Model name for identification.
        parameter_names : list of str
            Parameter names in optimization order.
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
        # Ensure params is at least 1D to avoid 0D array indexing issues.
        # np.asarray() raises on a jax.jit tracer, so that conversion must be
        # tried/caught here too, not just the later float() conversion below.
        params_arr: np.ndarray | jnp.ndarray
        try:
            if jax_available and hasattr(params, "ndim"):
                # Convert JAX arrays to NumPy for safe indexing
                params_arr = np.atleast_1d(np.asarray(params))
            else:
                params_arr = np.atleast_1d(params)
        except (TypeError, ValueError, AttributeError):
            # In JIT context: params is a tracer, keep it as a JAX array
            params_arr = jnp.atleast_1d(params)

        params_len = safe_len(params_arr)
        if params_len != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {params_len}")

        # Convert to regular Python floats only when safe to do so
        try:
            # Try converting to float - will fail if in JIT context
            return {
                name: float(val)
                for name, val in zip(self.parameter_names, params_arr, strict=False)
            }
        except (TypeError, ValueError, AttributeError):
            # In JIT context, keep as JAX arrays
            return dict(zip(self.parameter_names, params_arr, strict=False))

    def __repr__(self) -> str:
        """Return ``ClassName(name=..., n_params=...)``."""
        return f"{self.__class__.__name__}(name='{self.name}', n_params={self.n_params})"


class DiffusionModel(PhysicsModelBase):
    """Anomalous diffusion model with D(t) = D₀ t^α + D_offset.

    The three physical parameters (in order) are:

    - ``D0``: Reference diffusion coefficient [Å²/s].
    - ``alpha``: Diffusion time-dependence exponent [-].
    - ``D_offset``: Baseline diffusion [Å²/s].

    Physical interpretation of the exponent:

    - ``alpha = 0``: Normal diffusion (Brownian motion).
    - ``alpha > 0``: Super-diffusion (enhanced mobility).
    - ``alpha < 0``: Sub-diffusion (restricted mobility).
    - ``D_offset``: Residual diffusion at t=0.
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
        """Return the standard bounds for the diffusion parameters."""
        return [
            (100.0, 1e5),  # D0: 100 to 1e5 Å²/s
            (-2.0, 2.0),  # alpha: -2 to 2
            (-1e5, 1e5),  # D_offset: -1e5 to 1e5 Å²/s
        ]

    def get_default_parameters(self) -> jnp.ndarray:
        """Default values for typical XPCS measurements."""
        return jnp.array([100.0, 0.0, 10.0])  # Normal diffusion with small offset


class ShearModel(PhysicsModelBase):
    """Time-dependent shear model with γ̇(t) = γ̇₀ t^β + γ̇_offset.

    The four physical parameters (in order) are:

    - ``gamma_dot_t0``: Reference shear rate [s⁻¹].
    - ``beta``: Shear rate time-dependence exponent [-].
    - ``gamma_dot_t_offset``: Baseline shear rate [s⁻¹].
    - ``phi0``: Angular offset / flow direction [degrees].

    Physical interpretation of the exponent:

    - ``beta = 0``: Constant shear rate (steady shear).
    - ``beta > 0``: Increasing shear rate with time.
    - ``beta < 0``: Decreasing shear rate with time.
    - ``phi0``: Preferred flow direction angle.
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
        return compute_g1_shear(full_params, t1, t2, phi, q, L, dt)

    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Return the standard bounds for the shear parameters."""
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

    def __init__(self, analysis_mode: AnalysisMode = AnalysisMode.LAMINAR_FLOW):
        """Initialize the combined diffusion + shear model.

        Parameters
        ----------
        analysis_mode : AnalysisMode
            One of ``"static_anisotropic"``, ``"static_isotropic"`` (both use
            the 3 diffusion parameters), or ``"laminar_flow"`` (all 7
            parameters). Defaults to ``AnalysisMode.LAMINAR_FLOW``.
        """
        self.analysis_mode = analysis_mode

        if analysis_mode in ("static_isotropic", "static_anisotropic"):
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
        time_grid: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute g1 for a batch of (t1, t2, phi) points, element-wise.

        Calls the element-wise JAX core directly (mirrors the CMA-ES
        ``model_for_cmaes`` builder in ``optimization/nlsq/core.py``)
        instead of routing through ``compute_g1``/``compute_g1_total``.
        Those public wrappers call ``get_cached_meshgrid``, which
        meshgrids any 1D input of length <= 2000 into a 2D grid — fine
        for the matrix-mode public API, but wrong here: this method's
        contract is one distinct (t1, t2, phi) triple per batch element,
        and meshgridding collapses that pairing (previously done
        point-by-point via length-1 arrays, which always meshgridded to a
        degenerate 1x1 matrix and discarded t2, pinning g1 to its
        zero-lag value regardless of the true lag).

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
            Time step from configuration [s] (required; None raises).
        time_grid : jnp.ndarray, optional
            Full 1D unique-time grid covering the real data range, forwarded
            to the element-wise cores' cumulative-trapezoid integral. When
            omitted, the cores fall back to a fixed ``arange(10001)*dt``
            grid, which silently truncates (clamps via ``searchsorted``)
            datasets with more than 10001 unique time points — pass e.g.
            ``jnp.unique(t1_batch)`` for large datasets to avoid this.

        Returns
        -------
        jnp.ndarray
            Batch of g1 values, shape (n_points,)

        Raises
        ------
        ValueError
            If ``dt`` is None — the physics factors are dt-dependent and
            there is no safe default frame rate.
        """
        if dt is None:
            raise ValueError(
                "compute_g1_batch: dt must be provided explicitly (seconds). "
                "Physics factors are dt-dependent; there is no safe default frame rate."
            )
        wavevector_q_squared_half_dt = 0.5 * (q**2) * dt

        if self.analysis_mode.startswith("static"):
            result: jnp.ndarray = _compute_g1_diffusion_core(
                params, t1_batch, t2_batch, wavevector_q_squared_half_dt, dt, time_grid=time_grid
            )
        else:
            sinc_prefactor = 0.5 / PI * q * L * dt
            result = _compute_g1_total_core(
                params,
                t1_batch,
                t2_batch,
                phi_batch,
                wavevector_q_squared_half_dt,
                sinc_prefactor,
                dt,
                time_grid=time_grid,
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
        r"""Compute g2 with scaled fitting g₂ = offset + contrast × [g₁]².

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
        dt: float | None = None,
    ) -> float:
        """Compute chi-squared goodness of fit."""
        # The backend compute_chi_squared requires dt (physics factors are
        # dt-dependent); mirror the compute_g2 wrapper's None-guard + forward.
        if dt is None:
            raise TypeError(
                "dt parameter is required and cannot be None. "
                "Pass dt explicitly from configuration.",
            )
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
            dt,
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
def create_model(analysis_mode: AnalysisMode) -> CombinedModel:
    """Create the appropriate :class:`CombinedModel` for an analysis mode.

    Parameters
    ----------
    analysis_mode : AnalysisMode
        One of ``"static_anisotropic"``, ``"static_isotropic"``, or
        ``"laminar_flow"``.

    Returns
    -------
    CombinedModel
        Configured model instance for the requested mode.

    Raises
    ------
    ValueError
        If ``analysis_mode`` is not one of the supported homodyne modes.
    """
    valid_modes = ["static_anisotropic", "static_isotropic", "laminar_flow"]
    if analysis_mode not in valid_modes:
        raise ValueError(
            f"Invalid analysis mode '{analysis_mode}'. Must be one of {valid_modes}",
        )

    logger.info(f"Creating model for analysis mode: {analysis_mode}")
    return CombinedModel(analysis_mode=analysis_mode)


def get_available_models() -> list[str]:
    """Get list of available analysis modes."""
    return [
        "static_anisotropic",
        "static_isotropic",
        "laminar_flow",
        "two_component",
    ]


def make_model(config_or_manager: Any) -> PhysicsModelBase:
    """Construct the appropriate physics model from a config or ConfigManager.

    Dispatches based on the ``analysis_mode`` field:

    - ``"two_component"`` / ``"heterodyne"`` → :class:`HeterodyneModel`
      (xpcsjax.core.heterodyne_model)
    - ``"static_anisotropic"`` / ``"static_isotropic"`` /
      ``"laminar_flow"`` → :class:`CombinedModel` via :func:`create_model`

    Parameters
    ----------
    config_or_manager : ConfigManager or dict
        Either a :class:`~xpcsjax.config.ConfigManager` instance (with a
        ``.config`` dict attribute) or a raw config dict.

    Returns
    -------
    PhysicsModelBase
        A model instance whose ``analysis_mode`` matches the config.

    Raises
    ------
    ValueError
        If the resolved ``analysis_mode`` is not recognized.

    Examples
    --------
    >>> cfg = ConfigManager("config.yaml")  # analysis_mode: two_component
    >>> model = make_model(cfg)
    >>> isinstance(model, HeterodyneModel)
    True
    """
    # Accept both ConfigManager (has .config) and raw dict
    if hasattr(config_or_manager, "config") and config_or_manager.config is not None:
        cfg = config_or_manager.config
    elif isinstance(config_or_manager, dict):
        cfg = config_or_manager
    else:
        raise ValueError(
            f"make_model expects a ConfigManager or dict, got {type(config_or_manager).__name__}"
        )

    raw_mode = cfg.get("analysis_mode", "static_anisotropic")
    if not isinstance(raw_mode, str):
        raise ValueError(f"analysis_mode must be a string, got {type(raw_mode).__name__}")
    mode_lower = raw_mode.lower()

    # "static_ref" and "static_both" are reduced-parameter heterodyne modes
    # (validated in NLSQConfig.validate() against the module-level
    # _VALID_ANALYSIS_MODES frozenset in heterodyne_config.py). The reduced parameter
    # sets are only implemented by xpcsjax.core.heterodyne_models.ReducedModel,
    # which does not implement the PhysicsModelBase contract this factory
    # returns — HeterodyneModel always resolves the full 14-parameter
    # two_component set, so it cannot represent them. Raise rather than
    # silently substituting the wrong (full) model.
    if mode_lower in ("static_ref", "static_both"):
        raise NotImplementedError(
            f"analysis_mode={raw_mode!r} is not supported by make_model(): "
            "HeterodyneModel only implements the full two_component parameter "
            "set. Use xpcsjax.core.heterodyne_models.create_model() for the "
            "reduced static_ref/static_both models."
        )

    # Heterodyne / two-component dispatch.
    if "two_component" in mode_lower or "two-component" in mode_lower or "heterodyne" in mode_lower:
        # Local import to avoid circular dependency (heterodyne_model imports
        # PhysicsModelBase from this module).
        from xpcsjax.core.heterodyne_model import HeterodyneModel

        logger.info("make_model: dispatching to HeterodyneModel (mode=%s)", raw_mode)
        return HeterodyneModel()

    # Homodyne path — delegate to existing create_model factory
    logger.info("make_model: dispatching to CombinedModel (mode=%s)", raw_mode)
    return create_model(AnalysisMode(mode_lower))


# Export main classes and functions
__all__ = [
    "PhysicsModelBase",
    "DiffusionModel",
    "ShearModel",
    "CombinedModel",
    "create_model",
    "get_available_models",
    "make_model",
]
