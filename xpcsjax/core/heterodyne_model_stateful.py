"""Main heterodyne model wrapper class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.config.heterodyne_parameter_manager import ParameterManager
from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES
from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne, compute_residuals
from xpcsjax.core.heterodyne_models import TwoComponentModel
from xpcsjax.core.heterodyne_physics_factors import PhysicsFactors, create_physics_factors
from xpcsjax.core.heterodyne_scaling_utils import PerAngleScaling, ScalingConfig

if TYPE_CHECKING:
    pass


@dataclass
class HeterodyneModel:
    """Main heterodyne correlation model with stateful parameter management.

    This class provides a convenient interface for:

    - Managing model parameters through :class:`ParameterManager`
    - Computing correlation matrices
    - Computing residuals for fitting
    - Accessing pre-computed physics factors

    Notes
    -----
    This is the public ``HeterodyneModel`` lazy export. A
    ``result.parameters`` vector produced by the heterodyne fit is
    physics-first, ``[physics | contrast | offset]`` (homodyne is
    scaling-first), so do not assume the homodyne layout when reading it.

    Examples
    --------
    >>> model = HeterodyneModel.from_config(config)
    >>> c2 = model.compute_correlation(phi_angle=45.0)
    >>> residuals = model.compute_residuals(c2_data, phi_angle=45.0)
    """

    # Core model
    _model: TwoComponentModel = field(default_factory=TwoComponentModel)

    # Parameter management
    param_manager: ParameterManager = field(default_factory=ParameterManager)

    # Physics factors (pre-computed from config)
    _factors: PhysicsFactors | None = field(default=None)

    # Per-angle scaling (contrast/offset as fitted parameters)
    scaling: PerAngleScaling = field(default_factory=PerAngleScaling)

    # Cached time array
    _t: jnp.ndarray | None = field(default=None)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HeterodyneModel:
        """Create a model from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration with temporal, scattering, and parameters sections.

        Returns
        -------
        HeterodyneModel
            A configured model instance.
        """
        param_manager = ParameterManager.from_config(config)

        # Read from analyzer_parameters (canonical) with legacy fallback
        ap = config.get("analyzer_parameters", {})
        temporal = config.get("temporal", {})
        scattering = config.get("scattering", {})

        dt = float(ap.get("dt", temporal.get("dt", 1.0)))

        if "start_frame" in ap:
            start_frame = int(ap["start_frame"])
            end_frame = int(ap["end_frame"])
            n_times = end_frame - start_frame + 1
            t_start = dt  # relative time within window: first usable frame at 1×dt
        else:
            n_times = int(temporal.get("time_length", 1000))
            t_start = float(temporal.get("t_start", dt))

        ap_scat = ap.get("scattering", {})
        q = float(ap_scat.get("wavevector_q", scattering.get("wavevector_q", 0.01)))

        factors = create_physics_factors(
            n_times=n_times,
            dt=dt,
            q=q,
            phi_angle=0.0,
            t_start=float(t_start),
        )

        # Per-angle scaling config.
        # Priority: parameters.scaling values (via ParameterSpace) > scaling
        # section > registry defaults.  This ensures YAML parameter overrides
        # propagate to the PerAngleScaling object (homodyne parity).
        scaling_cfg = config.get("scaling", {})
        space_scaling = param_manager.space.scaling_values
        scaling = PerAngleScaling.from_config(
            ScalingConfig(
                n_angles=int(scaling_cfg.get("n_angles", 1)),
                mode=str(scaling_cfg.get("mode", "constant")),
                initial_contrast=float(
                    scaling_cfg.get("initial_contrast", space_scaling["contrast"])
                ),
                initial_offset=float(scaling_cfg.get("initial_offset", space_scaling["offset"])),
            )
        )

        return cls(
            _model=TwoComponentModel(),
            param_manager=param_manager,
            _factors=factors,
            scaling=scaling,
            _t=factors.t,
        )

    @property
    def n_params(self) -> int:
        """Total number of model parameters (14)."""
        return 14

    @property
    def n_varying(self) -> int:
        """Number of varying parameters."""
        return self.param_manager.n_varying

    @property
    def param_names(self) -> tuple[str, ...]:
        """All parameter names in canonical order."""
        return ALL_PARAM_NAMES

    @property
    def varying_names(self) -> list[str]:
        """Names of varying parameters."""
        return self.param_manager.varying_names

    @property
    def q(self) -> float:
        """Scattering wavevector magnitude."""
        if self._factors is None:
            raise ValueError("Physics factors not initialized")
        return self._factors.q

    @property
    def dt(self) -> float:
        """Time step."""
        if self._factors is None:
            raise ValueError("Physics factors not initialized")
        return self._factors.dt

    @property
    def t(self) -> jnp.ndarray:
        """Time array."""
        if self._t is None:
            raise ValueError("Time array not initialized")
        return self._t

    @property
    def n_times(self) -> int:
        """Number of time points.

        Reads from the live ``_t`` array rather than the (potentially stale)
        ``_factors.n_times`` so that the count stays correct after
        ``sync_time_axis`` modifies ``_t``.
        """
        if self._t is None:
            if self._factors is None:
                raise ValueError("Physics factors not initialized")
            return self._factors.n_times
        return int(len(self._t))

    def sync_time_axis(self, t: np.ndarray) -> None:
        """Trim model time axis to match post-exclusion data length.

        The data pipeline may remove leading time points (e.g. t=0 singularity
        exclusion), shrinking the data array.  This method trims the same
        number of leading points from the model's own seconds-based time axis
        (computed from start_frame and dt) so shapes align without discarding
        the correct absolute-time values.
        """
        n_data = len(t)
        if self._t is None:
            raise ValueError("Model time axis not initialized; call from_config first")
        n_model = len(self._t)
        if n_data < n_model:
            self._t = self._t[n_model - n_data :]
        elif n_data > n_model:
            # More data points than model — expand using the model's dt spacing
            extra = jnp.arange(1, n_data - n_model + 1, dtype=jnp.float64) * self.dt
            self._t = jnp.concatenate([self._t, self._t[-1:] + extra])

    def get_params(self) -> np.ndarray:
        """Return the current full parameter array.

        Returns
        -------
        np.ndarray
            Parameter array of shape ``(14,)``.
        """
        return self.param_manager.get_full_values()

    def get_params_dict(self) -> dict[str, float]:
        """Return the current parameters as a name-keyed dictionary."""
        return self.param_manager.get_parameter_dict()

    def set_params(self, params: np.ndarray | dict[str, float]) -> None:
        """Set parameter values.

        Parameters
        ----------
        params : np.ndarray or dict of str to float
            Either an array of shape ``(14,)`` or a dict keyed by parameter
            name.
        """
        self.param_manager.update_values(params)

    def compute_correlation(
        self,
        phi_angle: float = 0.0,
        params: np.ndarray | None = None,
        contrast: float | None = None,
        offset: float | None = None,
        angle_idx: int = 0,
    ) -> jnp.ndarray:
        """Compute the two-time correlation matrix.

        Parameters
        ----------
        phi_angle : float, optional
            Detector phi angle in degrees, default ``0.0``.
        params : np.ndarray, optional
            Parameter array; uses the stored values when ``None``.
        contrast : float, optional
            Speckle contrast override; uses per-angle scaling when ``None``.
        offset : float, optional
            Baseline offset override; uses per-angle scaling when ``None``.
        angle_idx : int, optional
            Angle index for the per-angle scaling lookup (0-based).

        Returns
        -------
        jnp.ndarray
            Correlation matrix ``c2(t1, t2)``, shape ``(N, N)``.
        """
        if params is None:
            params = self.get_params()

        # Use per-angle scaling unless explicitly overridden
        if contrast is None or offset is None:
            sc_contrast, sc_offset = self.scaling.get_for_angle(angle_idx)
            if contrast is None:
                contrast = sc_contrast
            if offset is None:
                offset = sc_offset

        return compute_c2_heterodyne(  # type: ignore[no-any-return]
            jnp.asarray(params),
            self.t,
            self.q,
            self.dt,
            phi_angle,
            contrast,
            offset,
        )

    def compute_residuals(
        self,
        c2_data: np.ndarray | jnp.ndarray,
        phi_angle: float = 0.0,
        params: np.ndarray | None = None,
        weights: np.ndarray | jnp.ndarray | None = None,
        contrast: float | None = None,
        offset: float | None = None,
        angle_idx: int = 0,
    ) -> jnp.ndarray:
        """Compute residuals between model and data.

        Parameters
        ----------
        c2_data : np.ndarray or jnp.ndarray
            Experimental correlation data.
        phi_angle : float, optional
            Detector phi angle in degrees, default ``0.0``.
        params : np.ndarray, optional
            Parameter array; uses the stored values when ``None``.
        weights : np.ndarray or jnp.ndarray, optional
            Weights (``1 / sigma**2``); defaults to ones when ``None``.
        contrast : float, optional
            Speckle contrast override; uses per-angle scaling when ``None``.
        offset : float, optional
            Baseline offset override; uses per-angle scaling when ``None``.
        angle_idx : int, optional
            Angle index for the per-angle scaling lookup (0-based).

        Returns
        -------
        jnp.ndarray
            Flattened residual array.
        """
        if params is None:
            params = self.get_params()

        if contrast is None or offset is None:
            sc_contrast, sc_offset = self.scaling.get_for_angle(angle_idx)
            if contrast is None:
                contrast = sc_contrast
            if offset is None:
                offset = sc_offset

        return compute_residuals(
            jnp.asarray(params),
            self.t,
            self.q,
            self.dt,
            phi_angle,
            jnp.asarray(c2_data),
            jnp.asarray(weights) if weights is not None else None,
            contrast,
            offset,
        )

    def compute_g1_reference(self, params: np.ndarray | None = None) -> jnp.ndarray:
        """Compute the reference g1 correlation.

        Parameters
        ----------
        params : np.ndarray, optional
            Parameter array; uses the stored values when ``None``.

        Returns
        -------
        jnp.ndarray
            Reference ``g1`` array, shape ``(N,)``.
        """
        if params is None:
            params = self.get_params()
        return self._model.compute_g1_reference(params, self.t, self.q)

    def compute_g1_sample(self, params: np.ndarray | None = None) -> jnp.ndarray:
        """Compute the sample g1 correlation.

        Parameters
        ----------
        params : np.ndarray, optional
            Parameter array; uses the stored values when ``None``.

        Returns
        -------
        jnp.ndarray
            Sample ``g1`` array, shape ``(N,)``.
        """
        if params is None:
            params = self.get_params()
        return self._model.compute_g1_sample(params, self.t, self.q)

    def compute_fraction(self, params: np.ndarray | None = None) -> jnp.ndarray:
        """Compute the sample fraction evolution.

        Parameters
        ----------
        params : np.ndarray, optional
            Parameter array; uses the stored values when ``None``.

        Returns
        -------
        jnp.ndarray
            Sample fraction array, shape ``(N,)``.
        """
        if params is None:
            params = self.get_params()
        return self._model.compute_fraction(params, self.t)

    def create_residual_function(
        self,
        c2_data: np.ndarray | jnp.ndarray,
        phi_angle: float,
        weights: np.ndarray | jnp.ndarray | None = None,
        angle_idx: int = 0,
    ) -> Any:
        """Create a residual function for optimization.

        Returns a function that takes the varying parameters and returns
        residuals (the fixed parameters are baked in).

        Parameters
        ----------
        c2_data : np.ndarray or jnp.ndarray
            Experimental correlation data.
        phi_angle : float
            Detector phi angle in degrees.
        weights : np.ndarray or jnp.ndarray, optional
            Weights; defaults to ones when ``None``.
        angle_idx : int, optional
            Index into per-angle scaling for the contrast/offset lookup.

        Returns
        -------
        callable
            A JIT-compiled function mapping varying params to residuals.
        """
        c2_jax = jnp.asarray(c2_data)
        weights_jax = jnp.asarray(weights) if weights is not None else jnp.ones_like(c2_jax)
        t = self.t
        q = self.q
        dt = self.dt

        contrast_val, offset_val = self.scaling.get_for_angle(angle_idx)

        varying_idx_jax = jnp.array(self.param_manager.varying_indices)
        fixed_values_jax = jnp.array(self.param_manager.get_full_values())

        @jax.jit
        def residual_fn(varying_params: jnp.ndarray) -> jnp.ndarray:
            # Reconstruct full params
            full_params = fixed_values_jax.at[varying_idx_jax].set(varying_params)

            return compute_residuals(
                full_params,
                t,
                q,
                dt,
                phi_angle,
                c2_jax,
                weights_jax,
                contrast_val,
                offset_val,
            )

        return residual_fn

    def summary(self) -> str:
        """Return a summary of the model configuration.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "HeterodyneModel Summary",
            "=" * 40,
            f"Time points: {self.n_times}",
            f"Time step: {self.dt}",
            f"Wavevector q: {self.q}",
            f"Total params: {self.n_params}",
            f"Varying params: {self.n_varying}",
            "",
            "Current Parameters:",
            "-" * 40,
        ]

        params = self.get_params_dict()
        for name in ALL_PARAM_NAMES:
            vary = "vary" if name in self.varying_names else "fixed"
            lines.append(f"  {name:18s}: {params[name]:12.4e} ({vary})")

        return "\n".join(lines)
