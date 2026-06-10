"""Gradient Collapse Monitor for Anti-Degeneracy Defense.

This module provides runtime detection of gradient collapse (physical params
losing gradient signal) with automatic response actions.

Part of the Anti-Degeneracy Defense System.

Detection Mechanism::

    Monitor the ratio:
        ratio = norm(grad_physical) / norm(grad_per_angle)

    If ratio < threshold for N consecutive iterations:
        - Gradient collapse detected
        - Physical params are losing signal to per-angle params

Response Actions
----------------
- "warn": Log warning only
- "hierarchical": Switch to hierarchical optimization mode
- "reset": Reset per-angle params to mean values
- "abort": Abort optimization and return best params so far
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from xpcsjax.optimization.nlsq.config import safe_float, safe_int
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


_DEBUG_CURVEFIT_CALLBACK = None


def _set_debug_curvefit_callback(cb):
    """Inject a debug callback into the solve paths (test seam).

    Used by tests to verify that the solver forwards a per-iteration callback
    to NLSQ. Not used in production wiring -- the real callback is passed
    explicitly there.

    Parameters
    ----------
    cb : callable
        Callback to install as the global debug hook.
    """
    global _DEBUG_CURVEFIT_CALLBACK
    _DEBUG_CURVEFIT_CALLBACK = cb


def _get_debug_curvefit_callback():
    """Return the currently installed debug curve-fit callback, or ``None``."""
    return _DEBUG_CURVEFIT_CALLBACK


@dataclass
class GradientMonitorConfig:
    """Configuration for gradient collapse detection.

    Attributes
    ----------
    enable : bool
        Whether to enable gradient monitoring. Default True.
    ratio_threshold : float
        Ratio of norm(grad_physical) / norm(grad_per_angle) below this triggers detection.
        Default 0.01 (physical gradient is 1% of per-angle gradient).
    consecutive_triggers : int
        Must trigger N consecutive times to confirm collapse. Default 5.
    response_mode : str
        Response action on collapse detection:
        - "warn": Log warning only
        - "hierarchical": Switch to hierarchical optimization
        - "reset": Reset per-angle params to mean
        - "abort": Abort and return best params
    reset_per_angle_to_mean : bool
        When resetting, reset per-angle to mean values. Default True.
    lambda_multiplier_on_collapse : float
        Multiply regularization λ by this on collapse. Default 10.0.
    check_interval : int
        Check every N iterations. Default 1 (every iteration).
    """

    enable: bool = True
    ratio_threshold: float = 0.01
    consecutive_triggers: int = 5
    response_mode: Literal["warn", "hierarchical", "reset", "abort"] = "hierarchical"
    reset_per_angle_to_mean: bool = True
    lambda_multiplier_on_collapse: float = 10.0
    check_interval: int = 1
    # NEW (Dec 2025): Watch specific parameter indices for gradient collapse
    # For laminar_flow: index 2*n_phi + 3 is gamma_dot_t0
    watch_parameters: list[int] | None = None
    watch_threshold: float = 1e-8  # Gradient magnitude below this triggers warning
    watch_consecutive_triggers: int = 3  # Must trigger N consecutive times (like ratio-based)
    watch_min_iteration: int = 5  # Skip checks before this iteration (warmup grace period)

    @classmethod
    def from_dict(cls, config_dict: dict) -> GradientMonitorConfig:
        """Create config from dictionary with safe type conversion."""
        # Parse watch_parameters list
        watch_params_raw = config_dict.get("watch_parameters")
        watch_parameters = None
        if watch_params_raw is not None:
            if isinstance(watch_params_raw, list):
                watch_parameters = [int(x) for x in watch_params_raw]
            elif isinstance(watch_params_raw, int):
                watch_parameters = [watch_params_raw]

        return cls(
            enable=bool(config_dict.get("enable", True)),
            ratio_threshold=safe_float(config_dict.get("ratio_threshold"), 0.01),
            consecutive_triggers=safe_int(config_dict.get("consecutive_triggers"), 5),
            response_mode=cast(
                Literal["warn", "hierarchical", "reset", "abort"],
                config_dict.get("response", "hierarchical"),
            ),
            reset_per_angle_to_mean=bool(config_dict.get("reset_per_angle_to_mean", True)),
            lambda_multiplier_on_collapse=safe_float(
                config_dict.get("lambda_multiplier_on_collapse"), 10.0
            ),
            check_interval=safe_int(config_dict.get("check_interval"), 1),
            watch_parameters=watch_parameters,
            watch_threshold=safe_float(config_dict.get("watch_threshold"), 1e-8),
            watch_consecutive_triggers=safe_int(config_dict.get("watch_consecutive_triggers"), 3),
            watch_min_iteration=safe_int(config_dict.get("watch_min_iteration"), 5),
        )


@dataclass
class CollapseEvent:
    """Record of a gradient collapse event.

    Attributes
    ----------
    iteration : int
        Iteration when collapse was detected.
    ratio : float
        Gradient ratio at detection.
    physical_grad_norm : float
        Physical parameter gradient norm.
    per_angle_grad_norm : float
        Per-angle parameter gradient norm.
    response_mode : str
        Response action taken.
    """

    iteration: int
    ratio: float
    physical_grad_norm: float
    per_angle_grad_norm: float
    response_mode: str


class GradientCollapseMonitor:
    """Monitor for detecting and responding to gradient collapse.

    This monitor tracks the ratio of physical to per-angle gradient norms
    during optimization. When the ratio drops below a threshold for
    consecutive iterations, it indicates that physical parameters are
    losing gradient signal (being absorbed by per-angle parameters).

    Parameters
    ----------
    config : GradientMonitorConfig
        Monitor configuration.
    physical_indices : list of int
        Indices of physical parameters in the full parameter vector.
    per_angle_indices : list of int
        Indices of per-angle parameters in the full parameter vector.

    Attributes
    ----------
    collapse_detected : bool
        Whether gradient collapse has been detected.
    consecutive_count : int
        Current count of consecutive low-ratio iterations.

    Notes
    -----
    History is capped at MAX_HISTORY_SIZE to prevent memory leaks during
    long-running optimizations. Older entries are discarded when the limit
    is reached.

    Examples
    --------
    >>> config = GradientMonitorConfig(ratio_threshold=0.01, consecutive_triggers=5)
    >>> monitor = GradientCollapseMonitor(config, physical_indices=[6,7,8,9,10,11,12],
    ...                                    per_angle_indices=list(range(6)))
    >>> for iter in range(100):
    ...     gradients = compute_gradients(params)
    ...     status = monitor.check(gradients, iter)
    ...     if status == "COLLAPSE_DETECTED":
    ...         response = monitor.get_response()
    ...         # Take action based on response
    """

    # Maximum history entries to prevent memory leaks
    # At ~100 bytes per entry, 1000 entries = ~100 KB max
    MAX_HISTORY_SIZE: int = 1000

    def __init__(
        self,
        config: GradientMonitorConfig,
        physical_indices: Sequence[int] | np.ndarray,
        per_angle_indices: Sequence[int] | np.ndarray,
    ):
        """Initialize gradient collapse monitor.

        Parameters
        ----------
        config : GradientMonitorConfig
            Monitor configuration.
        physical_indices : Sequence[int] or np.ndarray
            Indices of physical parameters. Converted to numpy array internally
            to support both NumPy and JAX array indexing.
        per_angle_indices : Sequence[int] or np.ndarray
            Indices of per-angle parameters (or Fourier coefficients when
            Fourier reparameterization is active). Converted to numpy array
            internally.

        Notes
        -----
        When Fourier reparameterization is active, per_angle_indices should
        correspond to Fourier coefficient indices (typically 10 for order=2),
        not independent per-angle indices (2 * n_phi).
        """
        self.config = config
        # Use numpy arrays for indices to support both NumPy and JAX array indexing
        # JAX arrays don't support Python list indexing (non-tuple sequence error)
        self.physical_indices: np.ndarray = np.asarray(physical_indices, dtype=np.intp)
        self.per_angle_indices: np.ndarray = np.asarray(per_angle_indices, dtype=np.intp)

        # Use a deque with bounded maxlen so that appending automatically
        # drops the oldest entry — O(1) on both ends vs O(n) list.pop(0).
        self.history: collections.deque[dict] = collections.deque(maxlen=self.MAX_HISTORY_SIZE)
        self.consecutive_count: int = 0
        self.collapse_detected: bool = False
        self.collapse_events: list[CollapseEvent] = []

        # Track best params for recovery
        self.best_params: np.ndarray | None = None
        self.best_loss: float = float("inf")

        # Track consecutive triggers for watched parameters
        self._watch_consecutive_counts: dict[int, int] = {}
        self._watch_collapse_detected: dict[int, bool] = {}
        if config.watch_parameters:
            for param_idx in config.watch_parameters:
                self._watch_consecutive_counts[param_idx] = 0
                self._watch_collapse_detected[param_idx] = False

    def check(
        self,
        gradients: np.ndarray,
        iteration: int,
        params: np.ndarray | None = None,
        loss: float | None = None,
    ) -> str:
        """Check for gradient collapse.

        Parameters
        ----------
        gradients : np.ndarray
            Full gradient vector.
        iteration : int
            Current iteration number.
        params : np.ndarray, optional
            Current parameters (for response actions and tracking).
        loss : float, optional
            Current loss value (for tracking best params).

        Returns
        -------
        str
            Status: "OK", "WARNING", "COLLAPSE_DETECTED"
        """
        if not self.config.enable:
            return "OK"

        # Skip if not on check interval
        if iteration % self.config.check_interval != 0:
            return "OK"

        # Track best params
        if params is not None and loss is not None:
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = params.copy()

        # Compute gradient norms
        physical_grad_norm = np.linalg.norm(gradients[self.physical_indices])
        per_angle_grad_norm = np.linalg.norm(gradients[self.per_angle_indices])

        # Compute ratio. A zero / non-finite per-angle(scaling) denominator
        # means the scaling block itself collapsed (the opposite degeneracy
        # end) -> ratio is inf, which the dual-ended trigger below treats as a
        # collapse rather than masking it with a tiny epsilon.
        denom = per_angle_grad_norm
        if denom > 0 and np.isfinite(denom):
            ratio = float(physical_grad_norm / denom)
        else:
            ratio = float("inf")

        # Record history.  deque(maxlen=MAX_HISTORY_SIZE) drops the oldest
        # entry automatically on append — no manual pop loop needed.
        self.history.append(
            {
                "iteration": iteration,
                "physical_grad_norm": float(physical_grad_norm),
                "per_angle_grad_norm": float(per_angle_grad_norm),
                "ratio": float(ratio),
            }
        )

        # Check for collapse at either degeneracy end: a low ratio (physical
        # block collapsing) OR a non-finite ratio (scaling block collapsed,
        # denom was zero/non-finite). Since `inf < threshold` is False, the
        # latter would be silently missed by a single-ended test.
        triggered = (ratio < self.config.ratio_threshold) or (not np.isfinite(ratio))
        if triggered:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0
            # Re-arm detection after recovery so future collapses are tracked
            self.collapse_detected = False

        # Trigger collapse detection (re-arms after recovery)
        if self.consecutive_count >= self.config.consecutive_triggers:
            if not self.collapse_detected:
                self.collapse_detected = True
                event = CollapseEvent(
                    iteration=iteration,
                    ratio=float(ratio),
                    physical_grad_norm=float(physical_grad_norm),
                    per_angle_grad_norm=float(per_angle_grad_norm),
                    response_mode=self.config.response_mode,
                )
                self.collapse_events.append(event)

                logger.warning(
                    f"GRADIENT COLLAPSE DETECTED at iteration {iteration}! "
                    f"ratio={ratio:.6f} (threshold={self.config.ratio_threshold})"
                )
                logger.warning(f"  Physical gradient norm: {physical_grad_norm:.6e}")
                logger.warning(f"  Per-angle gradient norm: {per_angle_grad_norm:.6e}")
                logger.warning(f"  Response mode: {self.config.response_mode}")

            return "COLLAPSE_DETECTED"

        # NEW (Dec 2025): Check watched parameters for gradient collapse
        # This specifically monitors parameters like gamma_dot_t0 that can
        # collapse to zero during L-BFGS warmup when data is angle-sequential
        # Uses consecutive trigger mechanism to avoid false positives during warmup
        if self.config.watch_parameters is not None:
            # Skip checks before minimum iteration (warmup grace period)
            if iteration >= self.config.watch_min_iteration:
                for param_idx in self.config.watch_parameters:
                    if param_idx < len(gradients):
                        grad_mag = abs(float(gradients[param_idx]))
                        # Store in history for diagnostics
                        self.history[-1][f"watched_param_{param_idx}_grad"] = grad_mag

                        if grad_mag < self.config.watch_threshold:
                            self._watch_consecutive_counts[param_idx] += 1
                        else:
                            # Reset consecutive count when gradient recovers
                            self._watch_consecutive_counts[param_idx] = 0
                            self._watch_collapse_detected[param_idx] = False

                        # Check for collapse (consecutive triggers threshold)
                        if (
                            self._watch_consecutive_counts[param_idx]
                            >= self.config.watch_consecutive_triggers
                            and not self._watch_collapse_detected[param_idx]
                        ):
                            self._watch_collapse_detected[param_idx] = True
                            logger.warning(
                                f"WATCHED PARAMETER GRADIENT COLLAPSE CONFIRMED at iteration {iteration}! "
                                f"param[{param_idx}] gradient={grad_mag:.2e} < "
                                f"threshold={self.config.watch_threshold:.2e} "
                                f"for {self._watch_consecutive_counts[param_idx]} consecutive iterations"
                            )
                        elif (
                            grad_mag < self.config.watch_threshold
                            and self._watch_consecutive_counts[param_idx] == 1
                        ):
                            # Log debug info on first trigger (not yet confirmed)
                            logger.debug(
                                f"Watched parameter gradient low at iteration {iteration}: "
                                f"param[{param_idx}] gradient={grad_mag:.2e}"
                            )

        if self.consecutive_count > 0:
            return "WARNING"

        return "OK"

    def get_response(self) -> dict | None:
        """Get response action after collapse detection.

        Returns
        -------
        dict or None
            Response action dictionary, or None if no collapse.
        """
        if not self.collapse_detected:
            return None

        return {
            "mode": self.config.response_mode,
            "reset_per_angle": self.config.reset_per_angle_to_mean,
            "lambda_multiplier": self.config.lambda_multiplier_on_collapse,
            "best_params": self.best_params,
            "best_loss": self.best_loss,
            "history": list(self.history)[-10:],  # Last 10 entries
            "collapse_events": self.collapse_events,
        }

    def compute_reset_params(self, params: np.ndarray, n_phi: int) -> np.ndarray:
        """Compute parameters with per-angle values reset to mean.

        Parameters
        ----------
        params : np.ndarray
            Current parameter vector.
        n_phi : int
            Number of phi angles.

        Returns
        -------
        np.ndarray
            Parameters with per-angle values reset.
        """
        reset_params = params.copy()

        # Assuming per-angle layout: [contrast_0..n_phi, offset_0..n_phi, physical...]
        if len(self.per_angle_indices) >= 2:
            # Reset contrast to mean
            contrast_indices = self.per_angle_indices[:n_phi]
            contrast_mean = np.nanmean(params[contrast_indices])
            reset_params[contrast_indices] = contrast_mean

            # Reset offset to mean
            offset_indices = self.per_angle_indices[n_phi : 2 * n_phi]
            offset_mean = np.nanmean(params[offset_indices])
            reset_params[offset_indices] = offset_mean

            logger.info(
                f"Reset per-angle params: contrast={contrast_mean:.4f}, offset={offset_mean:.4f}"
            )

        return reset_params

    def reset(self) -> None:
        """Reset monitor state for new optimization run."""
        self.history = collections.deque(maxlen=self.MAX_HISTORY_SIZE)
        self.consecutive_count = 0
        self.collapse_detected = False
        self.collapse_events = []
        self.best_params = None
        self.best_loss = float("inf")
        # Reset watched parameter tracking
        if self.config.watch_parameters:
            for param_idx in self.config.watch_parameters:
                self._watch_consecutive_counts[param_idx] = 0
                self._watch_collapse_detected[param_idx] = False

    def get_diagnostics(self) -> dict:
        """Get monitoring diagnostics for logging.

        Returns
        -------
        dict
            Diagnostic information.
        """
        if not self.history:
            return {
                "enabled": self.config.enable,
                "n_checks": 0,
            }

        ratios = [h["ratio"] for h in self.history]
        physical_norms = [h["physical_grad_norm"] for h in self.history]

        diag = {
            "enabled": self.config.enable,
            "n_checks": len(self.history),
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "mean_ratio": float(np.nanmean(ratios)),
            "final_ratio": ratios[-1] if ratios else None,
            "min_physical_grad": min(physical_norms),
            "max_physical_grad": max(physical_norms),
            "mean_physical_grad": float(np.nanmean(physical_norms)),
            "collapse_detected": self.collapse_detected,
            "consecutive_triggers": self.consecutive_count,
            "n_collapse_events": len(self.collapse_events),
            "response_mode": self.config.response_mode,
            "threshold": self.config.ratio_threshold,
        }
        # Add watched parameter diagnostics
        if self.config.watch_parameters:
            diag["watch_parameters"] = self.config.watch_parameters
            diag["watch_consecutive_counts"] = dict(self._watch_consecutive_counts)
            diag["watch_collapse_detected"] = dict(self._watch_collapse_detected)
        return diag

    def log_summary(self) -> None:
        """Log monitoring summary."""
        diag = self.get_diagnostics()

        if not diag["enabled"]:
            logger.info("Gradient monitoring: DISABLED")
            return

        if diag["n_checks"] == 0:
            logger.info("Gradient monitoring: No checks performed")
            return

        logger.info("Gradient Collapse Monitor Summary:")
        logger.info(f"  Checks performed: {diag['n_checks']}")
        logger.info(
            f"  Gradient ratio: min={diag['min_ratio']:.6f}, "
            f"max={diag['max_ratio']:.6f}, mean={diag['mean_ratio']:.6f}"
        )
        logger.info(f"  Threshold: {diag['threshold']}")

        if diag["collapse_detected"]:
            logger.warning(f"  COLLAPSE DETECTED: {diag['n_collapse_events']} events")
            logger.warning(f"  Response mode: {diag['response_mode']}")
        else:
            logger.info("  Status: No collapse detected")


def create_gradient_function_with_monitoring(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    monitor: GradientCollapseMonitor,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap gradient function to include monitoring.

    Parameters
    ----------
    grad_fn : Callable[[np.ndarray], np.ndarray]
        Original gradient function.
    monitor : GradientCollapseMonitor
        Monitor instance.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Wrapped gradient function that records to monitor.
    """
    iteration_counter = [0]  # Mutable counter

    def monitored_grad_fn(params: np.ndarray) -> np.ndarray:
        """Compute the gradient, record it to the monitor, and return it unchanged."""
        gradients = grad_fn(params)
        monitor.check(gradients, iteration_counter[0], params=params)
        iteration_counter[0] += 1
        return gradients

    return monitored_grad_fn


def build_gradient_collapse_callback(monitor, grad_fn, *, update_frequency=None):
    """Return an NLSQ ``curve_fit`` callback that feeds ``monitor`` each iteration.

    Strictly observational: computes ``grad_fn(params)`` and calls
    ``monitor.check(...)``; returns ``None`` and never mutates solve state.
    ``grad_fn`` errors are swallowed (best-effort, debug-logged) so the monitor
    can never abort a fit.

    Parameters
    ----------
    monitor : GradientCollapseMonitor
        Pre-constructed with the mode's physical / per-angle (scaling) indices.
    grad_fn : callable
        ``grad_fn(params: np.ndarray) -> np.ndarray`` — full gradient of the
        scalar loss (typically ``jax.grad(lambda p: 0.5*sum(residual(p)**2))``).
    update_frequency : int or None, optional
        Throttle for the extra per-iteration ``jax.grad`` evaluation. When
        ``None`` (default) the effective frequency falls back to the monitor's
        ``check_interval`` (or 1). With ``freq == 1`` the callback fires every
        iteration (current behavior); with ``freq == N`` it computes the
        gradient / runs ``monitor.check`` only on iterations where
        ``iteration % N == 0``. Strictly diagnostic — never mutates solve state.
    """
    freq = int(
        update_frequency
        if update_frequency is not None
        else (getattr(monitor.config, "check_interval", 1) or 1)
    )

    def callback(iteration, cost, params, info=None, **kwargs):
        """Feed the current iterate to the monitor (NLSQ per-iteration callback).

        Best-effort and strictly diagnostic: throttled by ``freq``, swallows any
        ``grad_fn`` error, and always returns ``None`` so it cannot abort a fit.
        """
        if freq > 1 and int(iteration) % freq != 0:
            return None
        try:
            p = np.asarray(params, dtype=np.float64)
            g = np.asarray(grad_fn(p), dtype=np.float64)
            monitor.check(g, int(iteration), params=p, loss=float(cost))
        except Exception:  # pragma: no cover - monitor must never break a fit
            logger.debug("gradient-collapse callback skipped (non-fatal)", exc_info=True)
        return None

    return callback


def gradient_monitor_diagnostics(monitor, *, mechanism="per_iteration_gradient_ratio"):
    """Build the canonical L4 ``gradient_monitor`` diagnostics block from a monitor.

    When the monitor recorded zero observations (callback never fired) the block
    is tagged ``mechanism="post_solve_fallback"`` so callers run the post-solve
    covariance-condition check instead.
    """
    ratios = [float(h["ratio"]) for h in monitor.history]
    n_obs = len(ratios)
    return {
        "collapse_detected": bool(monitor.collapse_detected),
        "trigger_count": int(len(monitor.collapse_events)),
        "min_gradient_ratio": float(min(ratios)) if ratios else float("nan"),
        "max_gradient_ratio": float(max(ratios)) if ratios else float("nan"),
        "n_observations": int(n_obs),
        "ratio_threshold": float(monitor.config.ratio_threshold),
        "consecutive_triggers": int(monitor.config.consecutive_triggers),
        "mechanism": mechanism if n_obs > 0 else "post_solve_fallback",
    }
