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

    Attributes
    ----------
    n_angles : int
        Number of detector angles.
    mode : str
        Scaling mode — one of ``"constant"`` (same contrast/offset for all
        angles, the default), ``"individual"`` (independent per angle),
        ``"auto"``, or ``"constant_averaged"``.
    initial_contrast : float
        Starting contrast value.
    initial_offset : float
        Starting offset value.
    """

    n_angles: int = 1
    mode: str = "constant"
    initial_contrast: float = 0.5
    initial_offset: float = 1.0


@dataclass
class PerAngleScaling:
    """Per-angle contrast and offset parameter manager.

    Manages arrays of ``contrast_i`` and ``offset_i`` values (one per angle),
    tracks which are varying in optimization, and provides expand/compress
    operations for the optimizer interface.

    Attributes
    ----------
    n_angles : int
        Number of detector angles.
    contrast : np.ndarray
        Per-angle contrast values, shape ``(n_angles,)``.
    offset : np.ndarray
        Per-angle offset values, shape ``(n_angles,)``.
    vary_contrast : np.ndarray
        Boolean mask of which contrast entries vary, shape ``(n_angles,)``.
    vary_offset : np.ndarray
        Boolean mask of which offset entries vary, shape ``(n_angles,)``.
    """

    n_angles: int = 1
    contrast: np.ndarray = field(default_factory=lambda: np.array([0.5]))
    offset: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    vary_contrast: np.ndarray = field(default_factory=lambda: np.array([True]))
    vary_offset: np.ndarray = field(default_factory=lambda: np.array([True]))

    @classmethod
    def from_config(cls, config: ScalingConfig) -> PerAngleScaling:
        """Create a manager from a :class:`ScalingConfig`.

        Parameters
        ----------
        config : ScalingConfig
            Per-angle scaling configuration.

        Returns
        -------
        PerAngleScaling
            A manager with contrast/offset arrays and vary-masks set per the
            requested mode and clipped to registry bounds.

        Raises
        ------
        ValueError
            If ``config.mode`` is not one of ``"constant"``, ``"individual"``,
            ``"auto"``, or ``"constant_averaged"``.
        """
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
        """Return the full scaling parameter array.

        Returns
        -------
        np.ndarray
            Array of shape ``(2 * n_angles,)`` laid out as
            ``[contrast_0..n, offset_0..n]``.
        """
        return np.concatenate([self.contrast, self.offset])

    def get_varying_values(self) -> np.ndarray:
        """Return only the varying scaling parameter values.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_varying_scaling,)``.
        """
        full = self.get_scaling_array()
        return full[self.varying_indices]  # type: ignore[no-any-return]

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return bounds for the varying scaling parameters.

        Returns
        -------
        tuple of np.ndarray
            ``(lower, upper)``, each of shape ``(n_varying_scaling,)``.
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

        Parameters
        ----------
        varying_values : np.ndarray
            Array of shape ``(n_varying_scaling,)``.
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
        """Return the contrast and offset for a specific angle.

        Parameters
        ----------
        angle_idx : int
            Angle index (0-based).

        Returns
        -------
        tuple of float
            ``(contrast, offset)`` for that angle.
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

        Only meaningful for ``"auto"`` and ``"constant_averaged"`` modes. For
        ``"auto"`` it sets per-angle initial values; for
        ``"constant_averaged"`` it collapses to a single averaged value.

        Parameters
        ----------
        c2_data : np.ndarray
            Pooled C2 correlation values.
        t1 : np.ndarray
            Pooled first time coordinates.
        t2 : np.ndarray
            Pooled second time coordinates.
        phi_indices : np.ndarray
            Index mapping each point to its angle (0-based).
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

    Uses the correlation decay structure
    ``C2 = offset + contrast * [correlation] / normalization``. At large time
    lags the correlation tends to 0, so ``C2`` tends to ``offset`` (the
    "floor"); at small time lags the correlation is near 1, so ``C2`` is near
    ``contrast + offset`` (the "ceiling").

    Parameters
    ----------
    c2_data : np.ndarray
        C2 correlation values (1D).
    delta_t : np.ndarray
        Time-lag values ``|t1 - t2|``, same shape as ``c2_data``.
    contrast_bounds : tuple of float, optional
        Valid bounds for contrast.
    offset_bounds : tuple of float, optional
        Valid bounds for offset.
    lag_floor_quantile : float, optional
        Quantile threshold for the "large lag" region (top 20%).
    lag_ceiling_quantile : float, optional
        Quantile threshold for the "small lag" region (bottom 20%).
    value_quantile_low : float, optional
        Quantile for robust floor estimation.
    value_quantile_high : float, optional
        Quantile for robust ceiling estimation.

    Returns
    -------
    tuple of float
        ``(contrast_est, offset_est)`` clipped to the supplied bounds.
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

    Uses vectorized grouped operations with ``searchsorted`` for efficient
    per-angle estimation.

    Parameters
    ----------
    c2_data : np.ndarray
        Pooled C2 correlation values.
    t1 : np.ndarray
        Pooled first time coordinates.
    t2 : np.ndarray
        Pooled second time coordinates.
    phi_indices : np.ndarray
        Index mapping each data point to its phi angle (``0`` to ``n_phi-1``).
    n_phi : int
        Number of unique phi angles.
    contrast_bounds : tuple of float
        Valid bounds for contrast.
    offset_bounds : tuple of float
        Valid bounds for offset.
    log : logging.Logger or logging.LoggerAdapter, optional
        Logger for diagnostic messages; falls back to the module logger.

    Returns
    -------
    dict of str to float
        Dictionary keyed by ``"contrast_0"``, ``"offset_0"``,
        ``"contrast_1"``, and so on.
    """
    if log is None:
        log = logger

    c2 = np.asarray(c2_data)
    t1_arr = np.asarray(t1)
    t2_arr = np.asarray(t2)
    phi_idx = np.asarray(phi_indices).astype(np.intp)

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

    Estimates per-angle values via quantile analysis, then averages them to
    produce single values for constant-mode optimization.

    Parameters
    ----------
    c2_data : np.ndarray
        Pooled C2 correlation values.
    t1 : np.ndarray
        Pooled first time coordinates.
    t2 : np.ndarray
        Pooled second time coordinates.
    phi_indices : np.ndarray
        Index mapping each data point to its phi angle.
    n_phi : int
        Number of unique phi angles.
    contrast_bounds : tuple of float
        Valid bounds for contrast.
    offset_bounds : tuple of float
        Valid bounds for offset.
    log : logging.Logger or logging.LoggerAdapter, optional
        Logger for diagnostic messages; falls back to the module logger.

    Returns
    -------
    tuple
        ``(contrast_avg, offset_avg, contrast_per_angle, offset_per_angle)``.
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
        "Averaged scaling for constant mode: contrast=%.4f [%.4f, %.4f], offset=%.4f [%.4f, %.4f]",
        contrast_avg,
        float(np.nanmin(contrast_per_angle)),
        float(np.nanmax(contrast_per_angle)),
        offset_avg,
        float(np.nanmin(offset_per_angle)),
        float(np.nanmax(offset_per_angle)),
    )

    return contrast_avg, offset_avg, contrast_per_angle, offset_per_angle


def estimate_per_angle_scaling_from_quantile(
    c2_data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi_indices: np.ndarray,
    n_phi: int,
    quantile: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-angle contrast β(φ_k) and offset ō(φ_k) from the data.

    Mirrors homodyne's anti-degeneracy estimator using a dual lag-region
    quantile approach. A naive diagonal-only formula

        β̂(φ_k) = q_{0.95}[c2(φ_k, t_i, t_i)] - 1
        ō̂(φ_k) = q_{0.05}[c2(φ_k, t_i, t_i)]

    cannot recover the offset because the diagonal samples all satisfy
    ``c2(t, t) = offset + contrast`` (decay = 1), so both quantiles
    collapse to ``offset + contrast``. We therefore generalize to:

        small-lag (|t1 - t2| in bottom ~20%): high-quantile gives
            ceiling ≈ offset + contrast
        large-lag (|t1 - t2| in top    ~20%): low-quantile  gives
            floor   ≈ offset (after decay)

    yielding ``offset = q_low(large-lag)`` and
    ``contrast = q_high(small-lag) - offset``. This delegates to the
    existing :func:`estimate_contrast_offset_from_quantiles` for each
    phi index.

    Used by the heterodyne ``constant`` per-angle mode to freeze β(φ_k)
    and ō(φ_k) before the NLSQ trust-region solve runs (Task B1 of the
    Phase 6 heterodyne ↔ homodyne mode parity plan).

    Parameters
    ----------
    c2_data : np.ndarray
        Correlation stack, shape ``(n_phi, n_t, n_t)`` or flattened of equal
        length to ``t1``, ``t2``, ``phi_indices``.
    t1, t2 : np.ndarray
        Two-time grids, same shape as ``c2_data``.
    phi_indices : np.ndarray
        Integer phi index for each entry, same shape as ``c2_data``.
    n_phi : int
        Expected number of phi angles. Validated against ``phi_indices`` so
        callers cannot silently get a shorter output array when one or more
        angles are missing from the pooled data.
    quantile : float, default 0.95
        Upper value-quantile for ceiling estimation; the lower value-quantile
        ``1 - quantile`` is used for the floor. The lag-region thresholds
        (bottom 20% / top 20%) are fixed at the homodyne defaults.

    Returns
    -------
    contrast_hat : np.ndarray
        Shape ``(n_phi,)``, estimate of β(φ_k), dtype float64.
    offset_hat : np.ndarray
        Shape ``(n_phi,)``, estimate of ō(φ_k), dtype float64.

    Raises
    ------
    ValueError
        - If ``n_phi <= 0``.
        - If ``phi_indices`` contains a value ``>= n_phi`` (out-of-range).
        - If any phi index in ``[0, n_phi)`` has no samples in
          ``phi_indices`` (an angle is missing from the pooled data).
        - If any phi index has fewer than 100 finite samples — below the
          threshold used by the underlying dual-region helper for reliable
          quantile estimation (see
          :func:`estimate_contrast_offset_from_quantiles`).
    """
    if n_phi <= 0:
        raise ValueError(f"n_phi must be positive, got {n_phi}")

    c2_flat = np.asarray(c2_data).ravel()
    t1_flat = np.asarray(t1).ravel()
    t2_flat = np.asarray(t2).ravel()
    phi_flat = np.asarray(phi_indices).ravel().astype(np.intp)
    if phi_flat.size == 0:
        raise ValueError("phi_indices is empty; expected at least one phi index")
    if int(phi_flat.max()) >= n_phi:
        raise ValueError(
            f"phi_indices contains values >= n_phi={n_phi}; max index = {int(phi_flat.max())}"
        )
    delta_t_flat = np.abs(t1_flat - t2_flat)

    # Wide bounds: the wrapper exposes a bounds-free signature per the
    # Task B1 spec. Downstream callers that need clipping to physical
    # bounds (e.g. the constant-mode fit assembler) clip the returned
    # estimates themselves.
    wide_contrast_bounds = (-np.inf, np.inf)
    wide_offset_bounds = (-np.inf, np.inf)

    contrast_hat = np.empty(n_phi, dtype=np.float64)
    offset_hat = np.empty(n_phi, dtype=np.float64)

    for k in range(n_phi):
        cell = phi_flat == k
        if not cell.any():
            raise ValueError(
                f"no samples for phi index {k} — phi_indices does not "
                f"cover the full [0, n_phi={n_phi}) range"
            )
        c2_angle = c2_flat[cell]
        delta_t_angle = delta_t_flat[cell]
        # Guard against the underlying helper's ``len(c2) < 100`` fallback:
        # with our wide ±inf bounds it would otherwise return NaN
        # (midpoint of unbounded interval) rather than failing loudly.
        n_finite = int(np.isfinite(c2_angle).sum())
        if n_finite < 100:
            raise ValueError(
                f"phi index {k}: only {n_finite} finite samples (need "
                f">=100 for reliable quantile estimation); check the "
                f"input grid or use the averaged-scaling path instead"
            )
        c_est, o_est = estimate_contrast_offset_from_quantiles(
            c2_angle,
            delta_t_angle,
            contrast_bounds=wide_contrast_bounds,
            offset_bounds=wide_offset_bounds,
            value_quantile_low=1.0 - quantile,
            value_quantile_high=quantile,
        )
        contrast_hat[k] = c_est
        offset_hat[k] = o_est

    return contrast_hat, offset_hat
