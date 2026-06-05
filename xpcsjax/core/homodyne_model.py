"""Hybrid stateful/functional wrapper for homodyne XPCS analysis.

Combines stateful robustness with functional JIT performance. Configuration is
validated once and stored on the instance, physics factors are pre-computed at
initialization, and a high-level API delegates to JIT-compiled functional cores:

1. **Stateful storage**: configuration validated once, stored on the instance.
2. **Pre-computed factors**: physics factors computed once at initialization.
3. **High-level API**: simple methods that use the stored configuration.
4. **JIT performance**: calls functional cores for optimal performance.

See Also
--------
xpcsjax.core.heterodyne_model.HeterodyneModel : Two-component heterodyne analog.

Examples
--------
>>> from xpcsjax.core.homodyne_model import HomodyneModel
>>> import numpy as np
>>> config = load_config("config.yaml")
>>> model = HomodyneModel(config)
>>> # Compute C2 - no dt parameter needed; it comes from the stored config.
>>> params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
>>> phi_angles = np.array([0, 30, 45, 60, 90])
>>> c2 = model.compute_c2(params, phi_angles)
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
    """Hybrid stateful/functional wrapper for homodyne XPCS analysis.

    Combines the robustness of stateful object-oriented design with the
    performance of functional JAX programming:

    1. Stores configuration (``dt``, ``q``, ``L``) as instance state.
    2. Pre-computes physics factors once at initialization.
    3. Provides high-level methods that use the stored state.
    4. Calls JIT-compiled functional cores for performance.

    Because ``dt`` and the physics factors are taken from the stored
    configuration, callers never pass ``dt`` to :meth:`compute_c2`, which
    eliminates a class of ``dt``-estimation errors and lets the pre-computed
    factors be reused across every correlation calculation.

    Parameters
    ----------
    config : dict
        Homodyne configuration dictionary (see :meth:`__init__`).

    Attributes
    ----------
    physics_factors : PhysicsFactors
        Pre-computed physics factors (q²dt/2, qLdt/2π).
    time_array : jnp.ndarray
        Time array for correlation calculations [s].
    t1_grid : jnp.ndarray
        2D first-time grid for correlation matrices (``indexing="ij"``).
    t2_grid : jnp.ndarray
        2D second-time grid for correlation matrices (``indexing="ij"``).
    model : xpcsjax.core.models.CombinedModel
        Underlying physics model, retained for backward compatibility.
    dt : float
        Time step [s].
    wavevector_q : float
        Scattering wave vector magnitude [Å⁻¹].
    stator_rotor_gap : float
        Geometric gap / sample length L [Å].
    start_frame : int
        First frame index of the analyzed range.
    end_frame : int
        Last frame index of the analyzed range (already resolved; the ``-1``
        sentinel is rejected at construction).
    analysis_mode : str
        One of ``"static_anisotropic"``, ``"static_isotropic"``, or
        ``"laminar_flow"``.

    See Also
    --------
    compute_c2 : Primary entry point for computing correlation surfaces.
    xpcsjax.core.heterodyne_model.HeterodyneModel : Two-component heterodyne analog.

    Examples
    --------
    >>> model = HomodyneModel(config)
    >>> c2 = model.compute_c2(params, phi_angles)
    >>> print(model.config_summary)
    >>> print(f"dt = {model.dt} s")
    """

    def __init__(self, config: dict):
        """Initialize the model from a configuration dictionary.

        Extracts and validates the temporal/scattering/geometry configuration,
        pre-computes the physics factors, and builds the time array and the two
        2D time grids used by every correlation calculation.

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

            The analysis mode is derived from ``analysis_settings``
            (``static_mode`` / ``isotropic_mode``) when present, otherwise from
            a top-level ``analysis_mode`` string, defaulting to
            ``"laminar_flow"``.

        Raises
        ------
        KeyError
            If a required ``analyzer_parameters`` key is missing.
        ValueError
            If ``end_frame`` is still the ``-1`` sentinel (it must be resolved
            to a concrete frame index by :class:`XPCSDataLoader` before
            constructing the model).
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
        """Compute the C2 correlation surfaces for all phi angles.

        Uses the pre-computed time grids and physics factors stored on the
        instance and dispatches to the JIT-compiled functional core. All phi
        angles are handled in a single vectorized call (no Python loop), and
        ``dt`` is taken from the stored configuration rather than passed in.

        Parameters
        ----------
        params : np.ndarray
            Physical parameters in registry order. For ``laminar_flow`` (7
            params): ``[D0, alpha, D_offset, gamma_dot_t0, beta,
            gamma_dot_t_offset, phi0]``. For the static modes (3 params):
            ``[D0, alpha, D_offset]``.
        phi_angles : np.ndarray
            Scattering angles in degrees, shape ``(n_phi,)``.
        contrast : float, default=0.5
            Speckle contrast (β in the literature).
        offset : float, default=1.0
            Baseline offset.

        Returns
        -------
        np.ndarray
            C2 correlation matrices, shape ``(n_phi, n_time, n_time)``, where
            ``n_time = end_frame - start_frame + 1``. Returned as a NumPy array
            (the JAX result is materialized on the host).

        See Also
        --------
        compute_c2_single_angle : Convenience wrapper for a single phi angle.

        Examples
        --------
        >>> model = HomodyneModel(config)
        >>> params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        >>> phi_angles = np.array([0, 30, 45, 60, 90])
        >>> c2 = model.compute_c2(params, phi_angles)
        >>> c2.shape  # (5, 100, 100) for 5 angles, 100 time points
        (5, 100, 100)
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
        """Compute the C2 correlation surface for a single phi angle.

        Convenience wrapper around :meth:`compute_c2` that accepts a scalar
        angle and returns the single correlation matrix (the leading phi axis
        is dropped).

        Parameters
        ----------
        params : np.ndarray
            Physical parameters in registry order (see :meth:`compute_c2`).
        phi : float
            Scattering angle in degrees.
        contrast : float, default=0.5
            Speckle contrast (β in the literature).
        offset : float, default=1.0
            Baseline offset.

        Returns
        -------
        np.ndarray
            C2 correlation matrix, shape ``(n_time, n_time)``.

        See Also
        --------
        compute_c2 : Multi-angle entry point.
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
                return "static_isotropic" if "isotropic" in mode_lower else "static_anisotropic"
            if mode_lower in {"laminar", "laminar_flow"}:
                return "laminar_flow"

        return "laminar_flow"

    @property
    def config_summary(self) -> dict:
        """Summarize the stored configuration for logging/debugging.

        Returns
        -------
        dict
            Summary of the key configuration values: ``dt``, ``time_length``,
            ``time_range``, ``wavevector_q``, ``stator_rotor_gap``,
            ``analysis_mode``, ``physics_factors`` (as a dict), ``start_frame``,
            and ``end_frame``.
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
        """Return a human-readable summary of the model configuration."""
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
