"""HomodyneModel - Hybrid Architecture Wrapper
============================================

Hybrid architecture combining stateful robustness with functional JIT performance.

This module implements the long-term architectural recommendation from the
architectural comparison analysis, providing:

1. **Stateful Storage**: Configuration validated once, stored in instance
2. **Pre-computed Factors**: Physics factors computed once at initialization
3. **High-level API**: Simple methods using stored configuration
4. **JIT Performance**: Calls functional cores for optimal performance

Best of both worlds: Robustness + Performance

Usage Example
-------------
>>> from xpcsjax.core.homodyne_model import HomodyneModel
>>>
>>> # Create model from configuration
>>> config = load_config("config.yaml")
>>> model = HomodyneModel(config)
>>>
>>> # Compute C2 - NO dt parameter needed!
>>> params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
>>> phi_angles = np.array([0, 30, 45, 60, 90])
>>> c2 = model.compute_c2(params, phi_angles)
>>>
>>> # For plotting, use the viz module:
>>> from xpcsjax.viz import plot_simulated_data
>>> plot_simulated_data(c2, phi_angles, output_dir="./results")
"""

import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.core.jax_backend import compute_g2_scaled_with_factors, jnp
from xpcsjax.core.models import CombinedModel
from xpcsjax.core.physics_factors import create_physics_factors_from_config_dict
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


class HomodyneModel:
    """Hybrid architecture wrapper for homodyne XPCS analysis.

    This class combines the robustness of stateful object-oriented design
    with the performance of functional JAX programming. It:

    1. Stores configuration (dt, q, L) as instance state
    2. Pre-computes physics factors once at initialization
    3. Provides high-level methods that use stored state
    4. Calls JIT-compiled functional cores for performance

    Benefits
    --------
    - **Robustness**: Configuration validated once at initialization
    - **Performance**: Physics factors pre-computed, JIT-compiled cores
    - **Usability**: Simple API, no dt parameter passing needed
    - **Safety**: No dt estimation errors possible
    - **Efficiency**: Factors computed once, reused for all calculations

    Attributes
    ----------
    physics_factors : PhysicsFactors
        Pre-computed physics factors (q²dt/2, qLdt/2π)
    time_array : jnp.ndarray
        Time array for correlation calculations [s]
    t1_grid, t2_grid : jnp.ndarray
        2D time grids for correlation matrices
    model : xpcsjax.core.models.CombinedModel
        Underlying physics model (for backward compatibility)
    dt : float
        Time step [s]
    wavevector_q : float
        Scattering wave vector magnitude [Å⁻¹]
    stator_rotor_gap : float
        Sample-detector distance [Å]
    analysis_mode : str
        Analysis mode ("static_anisotropic", "static_isotropic", "laminar_flow")

    Examples
    --------
    Basic usage:

    >>> model = HomodyneModel(config)
    >>> c2 = model.compute_c2(params, phi_angles)

    Access configuration:

    >>> print(model.config_summary)
    >>> print(f"dt = {model.dt} s")
    >>> print(f"Pre-computed factors: {model.physics_factors}")
    """

    def __init__(self, config: dict):
        """Initialize HomodyneModel from configuration dictionary.

        Parameters
        ----------
        config : dict
            Homodyne configuration dictionary with structure::

                {
                    'analyzer_parameters': {
                        'temporal': {'dt': float, 'start_frame': int, 'end_frame': int},
                        'scattering': {'wavevector_q': float},
                        'geometry': {'stator_rotor_gap': float}
                    },
                    'analysis_settings': {...}  # Optional
                }

        Raises
        ------
        KeyError
            If required configuration keys are missing
        ValueError
            If configuration values are invalid
        """
        logger.info("Initializing HomodyneModel with hybrid architecture")

        # Extract and validate configuration
        self._extract_config(config)

        # Pre-compute physics factors ONCE
        self.physics_factors = create_physics_factors_from_config_dict(config)
        logger.info(f"Pre-computed physics factors: {self.physics_factors}")

        # Resolve end_frame sentinel (-1 means "use all frames")
        if self.end_frame < 0:
            raise ValueError(
                f"end_frame={self.end_frame} is a sentinel value and must be resolved "
                f"to a concrete frame index before constructing HomodyneModel. "
                f"Use XPCSDataLoader to resolve this value from the HDF5 file."
            )

        # Create time array
        n_time = self.end_frame - self.start_frame + 1
        self.time_array = jnp.linspace(
            0,
            self.dt * (n_time - 1),
            n_time,
            dtype=jnp.float64,
        )

        # Create time grids for correlation calculations
        self.t1_grid, self.t2_grid = jnp.meshgrid(
            self.time_array,
            self.time_array,
            indexing="ij",
        )

        logger.debug(
            f"Time array: n={n_time}, range=[0, {self.dt * (n_time - 1):.2f}] s",
        )

        # Create underlying model (for backward compatibility)
        self.model = CombinedModel(analysis_mode=AnalysisMode.parse(self.analysis_mode))

        logger.info("HomodyneModel initialized successfully")
        logger.info(f"  Analysis mode: {self.analysis_mode}")
        logger.info(f"  Time points: {n_time}")
        logger.info(f"  dt: {self.dt} s")

    def compute_c2(
        self,
        params: np.ndarray,
        phi_angles: np.ndarray,
        contrast: float = 0.5,
        offset: float = 1.0,
    ) -> np.ndarray:
        """Compute C2 correlation function using stored configuration.

        This high-level method:
        - Uses pre-computed time grids (self.t1_grid, self.t2_grid)
        - Uses pre-computed physics factors (self.physics_factors)
        - Calls JIT-compiled functional core for performance
        - Returns C2 for all phi angles

        NO dt parameter needed - uses stored configuration!

        Parameters
        ----------
        params : np.ndarray
            Physical parameters:
            - For laminar_flow (7 params): [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
            - For static (3 params): [D0, alpha, D_offset]
        phi_angles : np.ndarray
            Scattering angles [degrees], shape (n_phi,)
        contrast : float, default=0.5
            Contrast parameter (β in literature)
        offset : float, default=1.0
            Baseline offset

        Returns
        -------
        np.ndarray
            C2 correlation matrices, shape (n_phi, n_time, n_time)

        Examples
        --------
        >>> model = HomodyneModel(config)
        >>> params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        >>> phi_angles = np.array([0, 30, 45, 60, 90])
        >>> c2 = model.compute_c2(params, phi_angles)
        >>> print(c2.shape)  # (5, 100, 100) for 5 angles, 100 time points
        """
        # Convert to JAX arrays
        params_jax = jnp.array(params)
        phi_angles_jax = jnp.array(phi_angles)

        # Extract pre-computed factors
        q_factor, sinc_factor = self.physics_factors.to_tuple()

        # Single vectorized call: pass all phi angles at once.
        # _compute_g1_shear_core handles phi arrays in matrix mode via vmap,
        # returning shape (n_phi, n_times, n_times) — no Python loop needed.
        result = compute_g2_scaled_with_factors(
            params_jax,
            self.t1_grid,
            self.t2_grid,
            phi_angles_jax,
            q_factor,  # Pre-computed at init
            sinc_factor,  # Pre-computed at init
            contrast,
            offset,
            self.dt,  # Time step from experimental configuration
        )

        logger.debug(
            f"Computed C2 for {len(phi_angles)} angles, "
            f"shape: {result.shape}, "
            f"range: [{float(np.nanmin(result)):.4f}, {float(np.nanmax(result)):.4f}]",
        )

        return np.array(result)

    def compute_c2_single_angle(
        self,
        params: np.ndarray,
        phi: float,
        contrast: float = 0.5,
        offset: float = 1.0,
    ) -> np.ndarray:
        """Compute C2 correlation function for a single angle.

        Convenience method for single-angle calculations.

        Parameters
        ----------
        params : np.ndarray
            Physical parameters
        phi : float
            Scattering angle [degrees]
        contrast : float, default=0.5
            Contrast parameter
        offset : float, default=1.0
            Baseline offset

        Returns
        -------
        np.ndarray
            C2 correlation matrix, shape (n_time, n_time)
        """
        c2 = self.compute_c2(params, np.array([phi]), contrast, offset)
        result: np.ndarray = c2[0]
        return result

    def _extract_config(self, config: dict) -> None:
        """Extract and validate configuration parameters."""
        try:
            analyzer_params = config["analyzer_parameters"]

            # Temporal parameters
            self.dt = analyzer_params["temporal"]["dt"]
            self.start_frame = analyzer_params["temporal"]["start_frame"]
            self.end_frame = analyzer_params["temporal"]["end_frame"]

            # Physical parameters
            self.wavevector_q = analyzer_params["scattering"]["wavevector_q"]
            self.stator_rotor_gap = analyzer_params["geometry"]["stator_rotor_gap"]

            # Analysis mode
            self.analysis_mode = self._determine_analysis_mode(config)

        except KeyError as e:
            raise KeyError(
                f"Missing required configuration key: {e}. "
                f"Expected structure: config['analyzer_parameters'][...]",
            ) from e

    def _determine_analysis_mode(self, config: dict) -> str:
        """Determine analysis mode from configuration."""
        analysis_settings = config.get("analysis_settings", {})
        if analysis_settings:
            is_static = bool(analysis_settings.get("static_mode", False))
            is_isotropic = bool(analysis_settings.get("isotropic_mode", False))

            if is_static:
                return "static_isotropic" if is_isotropic else "static_anisotropic"
            return "laminar_flow"

        mode = config.get("analysis_mode")
        if mode:
            mode_lower = str(mode).lower()
            if "static" in mode_lower:
                return (
                    "static_isotropic"
                    if "isotropic" in mode_lower
                    else "static_anisotropic"
                )
            if mode_lower in {"laminar", "laminar_flow"}:
                return "laminar_flow"

        return "laminar_flow"

    @property
    def config_summary(self) -> dict:
        """Get configuration summary for logging/debugging.

        Returns
        -------
        dict
            Configuration summary with all key parameters
        """
        return {
            "dt": self.dt,
            "time_length": len(self.time_array),
            "time_range": [0, self.dt * (len(self.time_array) - 1)],
            "wavevector_q": self.wavevector_q,
            "stator_rotor_gap": self.stator_rotor_gap,
            "analysis_mode": self.analysis_mode,
            "physics_factors": self.physics_factors.to_dict(),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }

    def __repr__(self) -> str:
        """String representation of HomodyneModel."""
        return (
            f"HomodyneModel(\n"
            f"  analysis_mode='{self.analysis_mode}',\n"
            f"  dt={self.dt} s,\n"
            f"  time_points={len(self.time_array)},\n"
            f"  q={self.wavevector_q} AA^-1,\n"
            f"  L={self.stator_rotor_gap} AA\n"
            f")"
        )


__all__ = ["HomodyneModel"]
