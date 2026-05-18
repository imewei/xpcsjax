"""Per-angle scaling utilities for heterodyne XPCS analysis.

Provides functions for expanding scalar contrast/offset parameters
into per-angle arrays, and applying per-angle scaling when computing
correlation matrices at multiple detector angles.

The per-angle scaling system transforms the scalar contrast and offset
(previously hardcoded function arguments) into fitted parameters that
can vary independently for each scattering angle.

The physics basis for quantile estimation:
    C2 = offset + contrast × [correlation terms] / normalization
    - At large time lags, correlation → 0, so C2 → offset (the "floor")
    - At small time lags, correlation ≈ 1, so C2 ≈ contrast + offset (the "ceiling")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from xpcsjax.config.parameter_registry import get_registry
from xpcsjax.utils.logging import get_logger


def _scaling_param_info(name: str):
    """Look up a scaling parameter's ParameterInfo via the xpcsjax registry.

    Wraps :func:`xpcsjax.config.parameter_registry.get_registry` to provide
    the legacy ``SCALING_PARAMS["contrast"|"offset"]`` access pattern with
    xpcsjax bound-attribute naming (``lower_bound`` / ``upper_bound``).
    """
    return get_registry().get_param_info(name)


class _ScalingParamProxy:
    """Mapping-shaped access to scaling parameter info.

    Mirrors the upstream ``SCALING_PARAMS`` Mapping interface while
    delegating bound lookups (``min_bound`` / ``max_bound``) to xpcsjax's
    ``lower_bound`` / ``upper_bound`` attributes.
    """

    class _BoundsAdapter:
        def __init__(self, info):
            self._info = info

        @property
        def min_bound(self):
            return self._info.lower_bound

        @property
        def max_bound(self):
            return self._info.upper_bound

        def __getattr__(self, name):
            return getattr(self._info, name)

    def __getitem__(self, name: str):
        return self._BoundsAdapter(_scaling_param_info(name))


SCALING_PARAMS = _ScalingParamProxy()

logger = get_logger(__name__)


@dataclass
class ScalingConfig:
    """Configuration for per-angle scaling behavior.

    Attributes:
        n_angles: Number of detector angles.
        mode: Scaling mode — one of:
            - "constant": Same contrast/offset for all angles (default).
            - "individual": Independent contrast/offset per angle.
        initial_contrast: Starting contrast value(s).
        initial_offset: Starting offset value(s).
    """

    n_angles: int = 1
    mode: str = "constant"
    initial_contrast: float = 0.5
    initial_offset: float = 1.0


@dataclass
class PerAngleScaling:
    """Per-angle contrast and offset parameter manager.

    Manages arrays of contrast_i and offset_i values (one per angle),
    tracks which are varying in optimization, and provides
    expand/compress operations for the optimizer interface.
    """

    n_angles: int = 1
    contrast: np.ndarray = field(default_factory=lambda: np.array([0.5]))
    offset: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    vary_contrast: np.ndarray = field(default_factory=lambda: np.array([True]))
    vary_offset: np.ndarray = field(default_factory=lambda: np.array([True]))

    @classmethod
    def from_config(cls, config: ScalingConfig) -> PerAngleScaling:
        """Create from ScalingConfig."""
        n = config.n_angles
        contrast_info = SCALING_PARAMS["contrast"]
        offset_info = SCALING_PARAMS["offset"]

        contrast = np.full(n, config.initial_contrast)
        offset_arr = np.full(n, config.initial_offset)

        if config.mode == "constant":
            # Only first angle varies; rest are locked to the same value
            vary_c = np.zeros(n, dtype=bool)
            vary_o = np.zeros(n, dtype=bool)
            vary_c[0] = True
            vary_o[0] = True
        elif config.mode == "individual":
            vary_c = np.ones(n, dtype=bool)
            vary_o = np.ones(n, dtype=bool)
        elif config.mode == "auto":
            # Auto mode: all angles vary, but initial values will be set
            # via estimate_per_angle_scaling() after data is available
            vary_c = np.ones(n, dtype=bool)
            vary_o = np.ones(n, dtype=bool)
        elif config.mode == "constant_averaged":
            # Like constant, but initial values come from averaged per-angle
            # estimates via compute_averaged_scaling()
            vary_c = np.zeros(n, dtype=bool)
            vary_o = np.zeros(n, dtype=bool)
            vary_c[0] = True
            vary_o[0] = True
        else:
            raise ValueError(
                f"Unknown scaling mode: {config.mode!r}. "
                f"Valid modes: 'constant', 'individual', 'auto', "
                f"'constant_averaged'"
            )

        # Clip to bounds
        contrast = np.clip(contrast, contrast_info.min_bound, contrast_info.max_bound)
        offset_arr = np.clip(offset_arr, offset_info.min_bound, offset_info.max_bound)

        return cls(
            n_angles=n,
            contrast=contrast,
            offset=offset_arr,
            vary_contrast=vary_c,
            vary_offset=vary_o,
        )

    @property
    def n_scaling_params(self) -> int:
        """Total number of per-angle scaling parameters (2 * n_angles)."""
        return 2 * self.n_angles

    @property
    def n_varying_scaling(self) -> int:
        """Number of varying scaling parameters."""
        return int(np.sum(self.vary_contrast) + np.sum(self.vary_offset))

    @property
    def varying_indices(self) -> np.ndarray:
        """Indices of varying scaling parameters in the scaling array."""
        # Scaling array layout: [contrast_0, ..., contrast_n, offset_0, ..., offset_n]
        c_indices = np.where(self.vary_contrast)[0]
        o_indices = np.where(self.vary_offset)[0] + self.n_angles
        return np.concatenate([c_indices, o_indices])

    def get_scaling_array(self) -> np.ndarray:
        """Get full scaling parameter array.

        Returns:
            Array of shape (2 * n_angles,): [contrast_0...n, offset_0...n]
        """
        return np.concatenate([self.contrast, self.offset])

    def get_varying_values(self) -> np.ndarray:
        """Get only the varying scaling parameter values.

        Returns:
            Array of shape (n_varying_scaling,)
        """
        full = self.get_scaling_array()
        return full[self.varying_indices]  # type: ignore[no-any-return]

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds for varying scaling parameters.

        Returns:
            (lower, upper) each of shape (n_varying_scaling,)
        """
        contrast_info = SCALING_PARAMS["contrast"]
        offset_info = SCALING_PARAMS["offset"]

        lower_full = np.concatenate(
            [
                np.full(self.n_angles, contrast_info.min_bound),
                np.full(self.n_angles, offset_info.min_bound),
            ]
        )
        upper_full = np.concatenate(
            [
                np.full(self.n_angles, contrast_info.max_bound),
                np.full(self.n_angles, offset_info.max_bound),
            ]
        )

        idx = self.varying_indices
        return lower_full[idx], upper_full[idx]

    def update_from_varying(self, varying_values: np.ndarray) -> None:
        """Update scaling parameters from optimizer output.

        Args:
            varying_values: Array of shape (n_varying_scaling,)
        """
        full = self.get_scaling_array()
        full[self.varying_indices] = varying_values

        self.contrast = full[: self.n_angles].copy()
        self.offset = full[self.n_angles :].copy()

        # Propagate constant mode: copy first angle to all
        if np.sum(self.vary_contrast) == 1 and self.vary_contrast[0]:
            self.contrast[:] = self.contrast[0]
        if np.sum(self.vary_offset) == 1 and self.vary_offset[0]:
            self.offset[:] = self.offset[0]

    def get_for_angle(self, angle_idx: int) -> tuple[float, float]:
        """Get contrast and offset for a specific angle.

        Args:
            angle_idx: Angle index (0-based)

        Returns:
            (contrast, offset) for that angle
        """
        return float(self.contrast[angle_idx]), float(self.offset[angle_idx])

    def initialize_from_data(
        self,
        c2_data: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi_indices: np.ndarray,
    ) -> None:
        """Initialize scaling values from data using quantile estimation.

        Only meaningful for 'auto' and 'constant_averaged' modes.
        For 'auto', sets per-angle initial values.
        For 'constant_averaged', sets averaged initial value for first angle.

        Args:
            c2_data: Pooled C2 correlation values.
            t1: Pooled first time coordinates.
            t2: Pooled second time coordinates.
            phi_indices: Index mapping each point to its angle (0-based).
        """
        contrast_bounds = (
            SCALING_PARAMS["contrast"].min_bound,
            SCALING_PARAMS["contrast"].max_bound,
        )
        offset_bounds = (
            SCALING_PARAMS["offset"].min_bound,
            SCALING_PARAMS["offset"].max_bound,
        )

        estimates = estimate_per_angle_scaling(
            c2_data=c2_data,
            t1=t1,
            t2=t2,
            phi_indices=phi_indices,
            n_phi=self.n_angles,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
        )

        for i in range(self.n_angles):
            self.contrast[i] = estimates[f"contrast_{i}"]
            self.offset[i] = estimates[f"offset_{i}"]

        # For constant_averaged, collapse to single averaged value
        if np.sum(self.vary_contrast) == 1 and self.vary_contrast[0]:
            avg_c = float(np.nanmean(self.contrast))
            avg_o = float(np.nanmean(self.offset))
            self.contrast[:] = avg_c
            self.offset[:] = avg_o


def estimate_contrast_offset_from_quantiles(
    c2_data: np.ndarray,
    delta_t: np.ndarray,
    contrast_bounds: tuple[float, float] = (0.0, 1.0),
    offset_bounds: tuple[float, float] = (0.5, 1.5),
    lag_floor_quantile: float = 0.80,
    lag_ceiling_quantile: float = 0.20,
    value_quantile_low: float = 0.10,
    value_quantile_high: float = 0.90,
) -> tuple[float, float]:
    """Estimate contrast and offset from C2 data using quantile analysis.

    Uses the correlation decay structure:
        C2 = offset + contrast × [correlation] / normalization

    At large time lags, correlation → 0, so C2 → offset (the "floor").
    At small time lags, correlation ≈ 1, so C2 ≈ contrast + offset (the "ceiling").

    Args:
        c2_data: C2 correlation values (1D).
        delta_t: Time lag values |t1 - t2| (same shape as c2_data).
        contrast_bounds: Valid bounds for contrast.
        offset_bounds: Valid bounds for offset.
        lag_floor_quantile: Quantile threshold for "large lag" region (top 20%).
        lag_ceiling_quantile: Quantile threshold for "small lag" region (bottom 20%).
        value_quantile_low: Quantile for robust floor estimation.
        value_quantile_high: Quantile for robust ceiling estimation.

    Returns:
        (contrast_est, offset_est) clipped to bounds.
    """
    c2 = np.asarray(c2_data)
    dt = np.asarray(delta_t)

    # Filter non-finite values
    finite_mask = np.isfinite(c2)
    if not np.all(finite_mask):
        c2 = c2[finite_mask]
        dt = dt[finite_mask]

    # Not enough data for robust estimation — return midpoints
    if len(c2) < 100:
        contrast_mid = (contrast_bounds[0] + contrast_bounds[1]) / 2.0
        offset_mid = (offset_bounds[0] + offset_bounds[1]) / 2.0
        return contrast_mid, offset_mid

    # Find lag thresholds
    lag_threshold_high = np.percentile(dt, lag_floor_quantile * 100)
    lag_threshold_low = np.percentile(dt, lag_ceiling_quantile * 100)

    # OFFSET: from large-lag region where correlation → 0
    large_lag_mask = dt >= lag_threshold_high
    if np.sum(large_lag_mask) >= 10:
        c2_floor_region = c2[large_lag_mask]
        offset_est = np.percentile(c2_floor_region, value_quantile_low * 100)
    else:
        offset_est = np.percentile(c2, value_quantile_low * 100)

    offset_est = float(np.clip(offset_est, offset_bounds[0], offset_bounds[1]))

    # CONTRAST: from small-lag region where correlation ≈ 1
    small_lag_mask = dt <= lag_threshold_low
    if np.sum(small_lag_mask) >= 10:
        c2_ceiling_region = c2[small_lag_mask]
        c2_ceiling = np.percentile(c2_ceiling_region, value_quantile_high * 100)
    else:
        c2_ceiling = np.percentile(c2, value_quantile_high * 100)

    contrast_est = c2_ceiling - offset_est
    contrast_est = float(np.clip(contrast_est, contrast_bounds[0], contrast_bounds[1]))

    return contrast_est, offset_est


def estimate_per_angle_scaling(
    c2_data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi_indices: np.ndarray,
    n_phi: int,
    contrast_bounds: tuple[float, float],
    offset_bounds: tuple[float, float],
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> dict[str, float]:
    """Estimate contrast and offset initial values for each phi angle.

    Uses vectorized grouped operations with searchsorted for efficient
    per-angle estimation.

    Args:
        c2_data: Pooled C2 correlation values.
        t1: Pooled first time coordinates.
        t2: Pooled second time coordinates.
        phi_indices: Index mapping each data point to its phi angle (0 to n_phi-1).
        n_phi: Number of unique phi angles.
        contrast_bounds: Valid bounds for contrast.
        offset_bounds: Valid bounds for offset.
        log: Logger for diagnostic messages.

    Returns:
        Dictionary with keys 'contrast_0', 'offset_0', 'contrast_1', etc.
    """
    if log is None:
        log = logger

    c2 = np.asarray(c2_data)
    t1_arr = np.asarray(t1)
    t2_arr = np.asarray(t2)
    phi_idx = np.asarray(phi_indices)

    delta_t = np.abs(t1_arr - t2_arr)

    contrast_mid = (contrast_bounds[0] + contrast_bounds[1]) / 2.0
    offset_mid = (offset_bounds[0] + offset_bounds[1]) / 2.0

    contrast_results = np.full(n_phi, contrast_mid)
    offset_results = np.full(n_phi, offset_mid)

    points_per_angle = np.bincount(phi_idx, minlength=n_phi)
    sufficient_mask = points_per_angle >= 100

    if not np.any(sufficient_mask):
        log.info(
            "All %d angles have insufficient data, using midpoint defaults",
            n_phi,
        )
        return {
            **{f"contrast_{i}": contrast_mid for i in range(n_phi)},
            **{f"offset_{i}": offset_mid for i in range(n_phi)},
        }

    # Sort by phi index for efficient grouped operations
    sort_idx = np.argsort(phi_idx)
    c2_sorted = c2[sort_idx]
    delta_t_sorted = delta_t[sort_idx]
    phi_sorted = phi_idx[sort_idx]

    group_starts = np.searchsorted(phi_sorted, np.arange(n_phi))
    group_ends = np.searchsorted(phi_sorted, np.arange(n_phi), side="right")

    for i in range(n_phi):
        if not sufficient_mask[i]:
            continue

        start, end = group_starts[i], group_ends[i]
        c2_angle = c2_sorted[start:end]
        delta_t_angle = delta_t_sorted[start:end]

        contrast_est, offset_est = estimate_contrast_offset_from_quantiles(
            c2_angle,
            delta_t_angle,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
        )

        contrast_results[i] = contrast_est
        offset_results[i] = offset_est

        log.debug(
            "Angle %d: estimated contrast=%.4f, offset=%.4f from %d data points",
            i,
            contrast_est,
            offset_est,
            end - start,
        )

    estimates: dict[str, float] = {}
    for i in range(n_phi):
        estimates[f"contrast_{i}"] = float(contrast_results[i])
        estimates[f"offset_{i}"] = float(offset_results[i])

    return estimates


def compute_averaged_scaling(
    c2_data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi_indices: np.ndarray,
    n_phi: int,
    contrast_bounds: tuple[float, float],
    offset_bounds: tuple[float, float],
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute averaged contrast and offset for constant mode.

    Estimates per-angle values via quantile analysis, then averages
    to produce single values for constant mode optimization.

    Args:
        c2_data: Pooled C2 correlation values.
        t1: Pooled first time coordinates.
        t2: Pooled second time coordinates.
        phi_indices: Index mapping each data point to its phi angle.
        n_phi: Number of unique phi angles.
        contrast_bounds: Valid bounds for contrast.
        offset_bounds: Valid bounds for offset.
        log: Logger for diagnostic messages.

    Returns:
        (contrast_avg, offset_avg, contrast_per_angle, offset_per_angle)
    """
    if log is None:
        log = logger

    estimates = estimate_per_angle_scaling(
        c2_data=c2_data,
        t1=t1,
        t2=t2,
        phi_indices=phi_indices,
        n_phi=n_phi,
        contrast_bounds=contrast_bounds,
        offset_bounds=offset_bounds,
        log=log,
    )

    contrast_per_angle = np.array([estimates[f"contrast_{i}"] for i in range(n_phi)])
    offset_per_angle = np.array([estimates[f"offset_{i}"] for i in range(n_phi)])

    contrast_avg = float(np.nanmean(contrast_per_angle))
    offset_avg = float(np.nanmean(offset_per_angle))

    log.info(
        "Averaged scaling for constant mode: "
        "contrast=%.4f [%.4f, %.4f], offset=%.4f [%.4f, %.4f]",
        contrast_avg,
        float(np.nanmin(contrast_per_angle)),
        float(np.nanmax(contrast_per_angle)),
        offset_avg,
        float(np.nanmin(offset_per_angle)),
        float(np.nanmax(offset_per_angle)),
    )

    return contrast_avg, offset_avg, contrast_per_angle, offset_per_angle
