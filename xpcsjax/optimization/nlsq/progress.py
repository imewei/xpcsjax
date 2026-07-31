"""Progress bar and logging callbacks for NLSQ optimization.

Features:
- Multi-start progress tracking

Part of the homodyne architecture.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


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
                    "tqdm not available for progress bar display. Install with: pip install tqdm"
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
                f"New best at start {start_idx + 1}/{self.n_starts}: chi2={chi_squared:.4e}"
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
            success_rate = self._successful / self._completed * 100 if self._completed > 0 else 0
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
