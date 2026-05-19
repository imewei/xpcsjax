"""Progress bar and logging callbacks for NLSQ optimization.

This module provides progress tracking for NLSQ fitting operations,
integrating with the NLSQ package's callback system.

Features:
- tqdm progress bar for fitting operations
- Iteration logging with configurable interval
- Multi-start progress tracking
- Streaming optimization progress

Part of homodyne v2.7.0 architecture.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    import logging

    from nlsq.callbacks import CallbackBase

    from xpcsjax.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)


@dataclass
class ProgressConfig:
    """Configuration for progress tracking.

    Attributes
    ----------
    enable_progress_bar : bool
        Whether to show tqdm progress bar.
    verbose : int
        Verbosity level: 0=quiet, 1=normal, 2=detailed.
    log_interval : int
        Log every N iterations when verbose >= 2.
    max_nfev : int
        Maximum function evaluations (for progress bar total).
    description : str
        Description for progress bar.
    """

    enable_progress_bar: bool = True
    verbose: int = 1
    log_interval: int = 10
    max_nfev: int = 1000
    description: str = "NLSQ Fitting"

    @classmethod
    def from_nlsq_config(
        cls,
        nlsq_config: NLSQConfig,
        max_nfev: int | None = None,
        description: str = "NLSQ Fitting",
    ) -> ProgressConfig:
        """Create ProgressConfig from NLSQConfig.

        Parameters
        ----------
        nlsq_config : NLSQConfig
            NLSQ configuration object.
        max_nfev : int, optional
            Max function evaluations. Uses nlsq_config.max_iterations if None.
        description : str
            Description for progress bar.

        Returns
        -------
        ProgressConfig
            Progress configuration.
        """
        return cls(
            enable_progress_bar=nlsq_config.enable_progress_bar,
            verbose=nlsq_config.verbose,
            log_interval=nlsq_config.log_iteration_interval,
            max_nfev=max_nfev or nlsq_config.max_iterations,
            description=description,
        )


class HomodyneIterationLogger:
    """Iteration logger that integrates with homodyne's logging system.

    Logs optimization progress at configurable intervals using the
    homodyne logging infrastructure.

    Parameters
    ----------
    verbose : int
        Verbosity level: 0=quiet, 1=normal (milestones), 2=detailed.
    log_interval : int
        Log every N iterations when verbose >= 2.
    logger_instance : logging.Logger, optional
        Logger to use. Defaults to module logger.
    """

    def __init__(
        self,
        verbose: int = 1,
        log_interval: int = 10,
        logger_instance: logging.Logger | None = None,
    ):
        self.verbose = verbose
        self.log_interval = log_interval
        self._logger = logger_instance or logger
        self._start_time: float | None = None
        self._best_cost: float = float("inf")
        self._last_logged_iter: int = -1
        self._milestone_costs = [1e-2, 1e-4, 1e-6, 1e-8]
        self._passed_milestones: set[float] = set()

    def __call__(
        self,
        iteration: int,
        cost: float,
        params: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        """Log iteration information based on verbosity settings."""
        if self._start_time is None:
            self._start_time = time.perf_counter()
            if self.verbose >= 1:
                self._logger.info(
                    f"NLSQ optimization started | initial cost: {cost:.6e}"
                )

        # Update best cost
        if cost < self._best_cost:
            self._best_cost = cost

        elapsed = time.perf_counter() - self._start_time

        # Verbose = 2: Log at regular intervals
        if (
            self.verbose >= 2
            and iteration - self._last_logged_iter >= self.log_interval
        ):
            grad_norm = info.get("gradient_norm", float("nan"))
            nfev = info.get("nfev", iteration + 1)
            self._logger.info(
                f"Iter {iteration:4d} | "
                f"cost: {cost:.6e} | "
                f"grad: {grad_norm:.3e} | "
                f"nfev: {nfev:4d} | "
                f"time: {elapsed:.2f}s"
            )
            self._last_logged_iter = iteration

        # Verbose = 1: Log cost milestones
        elif self.verbose == 1:
            for milestone in self._milestone_costs:
                if cost <= milestone and milestone not in self._passed_milestones:
                    self._passed_milestones.add(milestone)
                    self._logger.info(
                        f"Milestone: cost reached {milestone:.0e} at iter {iteration} | "
                        f"time: {elapsed:.2f}s"
                    )
                    break

    def close(self) -> None:
        """Log final summary."""
        if self._start_time is not None and self.verbose >= 1:
            elapsed = time.perf_counter() - self._start_time
            self._logger.info(
                f"NLSQ optimization complete | "
                f"best cost: {self._best_cost:.6e} | "
                f"total time: {elapsed:.2f}s"
            )


def create_progress_callback(
    config: ProgressConfig | None = None,
    enable_progress_bar: bool = True,
    verbose: int = 1,
    log_interval: int = 10,
    max_nfev: int = 1000,
    description: str = "NLSQ Fitting",
) -> tuple[CallbackBase | None, HomodyneIterationLogger | None]:
    """Create progress callback chain for NLSQ optimization.

    Creates a callback chain with optional progress bar and iteration logger.

    Parameters
    ----------
    config : ProgressConfig, optional
        Progress configuration. If provided, overrides other parameters.
    enable_progress_bar : bool
        Whether to show tqdm progress bar.
    verbose : int
        Verbosity level: 0=quiet, 1=normal, 2=detailed.
    log_interval : int
        Log every N iterations when verbose >= 2.
    max_nfev : int
        Maximum function evaluations for progress bar.
    description : str
        Description for progress bar.

    Returns
    -------
    tuple[CallbackBase | None, HomodyneIterationLogger | None]
        (callback, iteration_logger) - callback for NLSQ, logger for manual close.
        Returns (None, None) if no callbacks are needed.
    """
    if config is not None:
        enable_progress_bar = config.enable_progress_bar
        verbose = config.verbose
        log_interval = config.log_interval
        max_nfev = config.max_nfev
        description = config.description

    # Heterogeneous callback list: holds ProgressBar (when nlsq is available)
    # and HomodyneIterationLogger (when verbose>=1). They share the NLSQ
    # callback duck-type — annotate as ``list[Any]`` rather than introducing
    # a Protocol that exists only to satisfy mypy.
    callbacks: list[Any] = []
    iteration_logger = None

    # Create progress bar callback
    if enable_progress_bar:
        try:
            from nlsq.callbacks import ProgressBar

            progress_bar = ProgressBar(max_nfev=max_nfev, desc=description)
            callbacks.append(progress_bar)
        except ImportError:
            logger.debug("NLSQ ProgressBar not available")

    # Create iteration logger
    if verbose >= 1:
        iteration_logger = HomodyneIterationLogger(
            verbose=verbose,
            log_interval=log_interval,
        )
        callbacks.append(iteration_logger)

    # Return appropriate callback
    if not callbacks:
        return None, None
    elif len(callbacks) == 1:
        return callbacks[0], iteration_logger
    else:
        try:
            from nlsq.callbacks import CallbackChain

            return CallbackChain(*callbacks), iteration_logger
        except ImportError:
            # Fallback to just the first callback
            return callbacks[0], iteration_logger


class MultiStartProgressTracker:
    """Progress tracker for multi-start optimization.

    Provides a progress bar and logging for multi-start optimization,
    tracking the progress of multiple starting points.

    Parameters
    ----------
    n_starts : int
        Total number of starting points.
    enable_progress_bar : bool
        Whether to show tqdm progress bar.
    verbose : int
        Verbosity level.
    description : str
        Description for progress bar.
    """

    def __init__(
        self,
        n_starts: int,
        enable_progress_bar: bool = True,
        verbose: int = 1,
        description: str = "Multi-start NLSQ",
    ):
        self.n_starts = n_starts
        self.enable_progress_bar = enable_progress_bar
        self.verbose = verbose
        self.description = description

        self._pbar = None
        self._start_time = time.perf_counter()
        self._completed = 0
        self._successful = 0
        self._failed = 0
        self._best_chi_squared = float("inf")
        self._best_start_idx: int | None = None
        self._tqdm_available = False

        # Initialize progress bar
        if enable_progress_bar:
            try:
                from tqdm.auto import tqdm  # type: ignore[import-untyped]

                self._pbar = tqdm(
                    total=n_starts,
                    desc=description,
                    unit="start",
                    dynamic_ncols=True,
                    leave=True,
                )
                self._tqdm_available = True
                logger.debug(f"Progress bar initialized: {n_starts} starts")
            except ImportError:
                logger.warning(
                    "tqdm not available for progress bar display. "
                    "Install with: pip install tqdm"
                )
            except (AttributeError, RuntimeError, ValueError) as e:
                logger.warning(f"Failed to initialize progress bar: {e}")

        if verbose >= 1:
            logger.info(
                f"Multi-start optimization: {n_starts} starting points, "
                f"progress_bar={'enabled' if self._tqdm_available else 'disabled'}"
            )

    def update(
        self,
        start_idx: int,
        success: bool,
        chi_squared: float,
        message: str = "",
        wall_time: float | None = None,
    ) -> None:
        """Update progress after a single start completes.

        Parameters
        ----------
        start_idx : int
            Index of the completed starting point.
        success : bool
            Whether optimization was successful.
        chi_squared : float
            Final chi-squared value.
        message : str, optional
            Status message.
        wall_time : float, optional
            Time taken for this optimization in seconds.
        """
        self._completed += 1
        is_new_best = False

        if success:
            self._successful += 1
            if chi_squared < self._best_chi_squared:
                self._best_chi_squared = chi_squared
                self._best_start_idx = start_idx
                is_new_best = True
        else:
            self._failed += 1

        # Calculate elapsed time and ETA
        elapsed = time.perf_counter() - self._start_time
        remaining = self.n_starts - self._completed
        avg_time_per_start = elapsed / self._completed if self._completed > 0 else 0
        eta = avg_time_per_start * remaining

        # Update progress bar
        if self._pbar is not None:
            postfix = {
                "ok": f"{self._successful}/{self._completed}",
                "best": f"{self._best_chi_squared:.4e}",
            }
            if eta > 0:
                postfix["ETA"] = f"{eta:.0f}s"
            self._pbar.set_postfix(postfix)
            self._pbar.update(1)

        # Log detailed progress for verbose >= 2
        if self.verbose >= 2:
            status = "OK" if success else "FAILED"
            time_str = f", time={wall_time:.1f}s" if wall_time is not None else ""
            new_best_str = " [NEW BEST]" if is_new_best else ""
            logger.info(
                f"Start {start_idx + 1:3d}/{self.n_starts} [{status:6s}] | "
                f"chi2={chi_squared:.4e} | best={self._best_chi_squared:.4e}"
                f"{new_best_str}{time_str}"
            )
        elif self.verbose >= 1 and is_new_best:
            # Log new best even at verbose=1
            logger.info(
                f"New best at start {start_idx + 1}/{self.n_starts}: "
                f"chi2={chi_squared:.4e}"
            )

        # Log failures at verbose >= 1
        if not success and self.verbose >= 1 and message:
            logger.warning(f"Start {start_idx + 1}/{self.n_starts} failed: {message}")

    def close(self) -> None:
        """Close progress bar and log summary."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        elapsed = time.perf_counter() - self._start_time
        avg_time = elapsed / self._completed if self._completed > 0 else 0

        if self.verbose >= 1:
            success_rate = (
                self._successful / self._completed * 100 if self._completed > 0 else 0
            )
            logger.info(
                f"Multi-start summary: {self._successful}/{self._completed} successful "
                f"({success_rate:.0f}%), {self._failed} failed"
            )
            logger.info(
                f"Best result: chi2={self._best_chi_squared:.4e} at start {self._best_start_idx}"
            )
            logger.info(f"Timing: total={elapsed:.1f}s, avg={avg_time:.1f}s/start")

    def __enter__(self) -> MultiStartProgressTracker:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> Literal[False]:
        """Context manager exit.

        Returns ``Literal[False]`` so any exception raised inside the ``with``
        block propagates. mypy's ``[exit-return]`` check requires this exact
        type when the body always returns False.
        """
        self.close()
        return False


def create_streaming_progress_callback(
    n_total_points: int,
    batch_size: int,
    max_epochs: int,
    enable_progress_bar: bool = True,
    verbose: int = 1,
) -> Callable[[int, np.ndarray, float], bool] | None:
    """Create a progress callback for streaming optimization.

    Parameters
    ----------
    n_total_points : int
        Total number of data points.
    batch_size : int
        Batch size for streaming.
    max_epochs : int
        Maximum number of epochs.
    enable_progress_bar : bool
        Whether to show progress bar.
    verbose : int
        Verbosity level.

    Returns
    -------
    Callable or None
        Callback function for streaming optimizer, or None if not needed.
    """
    if not enable_progress_bar and verbose == 0:
        return None

    batches_per_epoch = (n_total_points + batch_size - 1) // batch_size
    total_iterations = max_epochs * batches_per_epoch

    # Initialize progress bar
    pbar = None
    if enable_progress_bar:
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(
                total=total_iterations,
                desc="Streaming NLSQ",
                unit="batch",
            )
        except ImportError:
            pass

    # State tracking. Annotated ``dict[str, Any]`` because the closure mixes
    # float (best_loss, start_time), int (last_epoch), and list[float]
    # (epoch_losses) — no single TypedDict captures the heterogeneous shape
    # without forcing every read site through a cast.
    state: dict[str, Any] = {
        "start_time": time.perf_counter(),
        "last_epoch": -1,
        "best_loss": float("inf"),
        "epoch_losses": [],
    }

    def callback(iteration: int, params: np.ndarray, loss: float) -> bool:
        """Progress callback for streaming optimizer."""
        current_epoch = iteration // batches_per_epoch

        # Update best loss
        if loss < state["best_loss"]:
            state["best_loss"] = loss

        state["epoch_losses"].append(loss)

        # Update progress bar
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "epoch": current_epoch,
                    "loss": f"{loss:.6e}",
                    "best": f"{state['best_loss']:.6e}",
                }
            )

        # Log at epoch boundaries
        if current_epoch > state["last_epoch"] and verbose >= 1:
            elapsed = time.perf_counter() - state["start_time"]
            avg_loss = np.mean(state["epoch_losses"]) if state["epoch_losses"] else loss

            logger.info(
                f"Epoch {current_epoch:3d}/{max_epochs} | "
                f"avg loss: {avg_loss:.6e} | "
                f"best: {state['best_loss']:.6e} | "
                f"time: {elapsed:.2f}s"
            )

            state["last_epoch"] = current_epoch
            state["epoch_losses"] = []

        return False  # Don't stop early

    # Add cleanup
    callback._pbar = pbar  # type: ignore[attr-defined]

    return callback
