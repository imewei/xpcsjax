"""Batch-level statistics tracking for streaming optimization.

This module provides a circular buffer for tracking batch-level optimization
statistics, success rates, and error distributions during streaming optimization.
"""

from collections import defaultdict, deque
from typing import Any

# Type alias for batch record
BatchRecord = dict[str, Any]


class BatchStatistics:
    """Circular buffer for tracking batch-level statistics.

    Maintains statistics for the most recent N batches (default 100) to
    provide running averages and trends without unbounded memory growth.

    Attributes
    ----------
    buffer : deque
        Circular buffer storing batch records (max_size most recent)
    total_batches : int
        Total number of batches processed (all time)
    total_successes : int
        Total number of successful batches (all time)
    total_failures : int
        Total number of failed batches (all time)
    error_counts : dict
        Count of each error type encountered (all time)

    Examples
    --------
    >>> stats = BatchStatistics(max_size=100)
    >>> stats.record_batch(
    ...     batch_idx=0,
    ...     success=True,
    ...     loss=0.123,
    ...     iterations=50,
    ...     recovery_actions=[]
    ... )
    >>> stats.get_success_rate()
    1.0
    """

    def __init__(self, max_size: int = 100):
        """Initialize batch statistics tracker.

        Parameters
        ----------
        max_size : int, optional
            Maximum number of batches to keep in circular buffer, by default 100
        """
        self.buffer: deque[BatchRecord] = deque(maxlen=max_size)
        self.total_batches = 0
        self.total_successes = 0
        self.total_failures = 0
        self.error_counts: defaultdict[str, int] = defaultdict(int)

    def record_batch(
        self,
        batch_idx: int,
        success: bool,
        loss: float,
        iterations: int,
        recovery_actions: list[str],
        error_type: str | None = None,
    ) -> None:
        """Record statistics for a single batch.

        Parameters
        ----------
        batch_idx : int
            Batch index (0-indexed)
        success : bool
            Whether batch optimization succeeded
        loss : float
            Final loss value for this batch
        iterations : int
            Number of iterations performed
        recovery_actions : list of str
            List of recovery actions applied (if any)
        error_type : str, optional
            Type of error encountered (if failed), by default None
        """
        batch_record = {
            "batch_idx": batch_idx,
            "success": success,
            "loss": loss,
            "iterations": iterations,
            "recovery_actions": recovery_actions,
            "error_type": error_type,
        }

        self.buffer.append(batch_record)
        self.total_batches += 1

        if success:
            self.total_successes += 1
        else:
            self.total_failures += 1
            if error_type:
                self.error_counts[error_type] += 1

    def get_success_rate(self) -> float:
        """Calculate success rate from recent batches in buffer.

        Returns
        -------
        float
            Success rate (0.0 to 1.0) from recent batches. Returns 1.0 when
            no batches have been recorded yet (optimistic prior) so that quality
            gates do not falsely reject the first batch. Callers that need to
            distinguish "no data yet" should check BatchStatistics.total_batches.
        """
        if not self.buffer:
            return 1.0

        successes = sum(1 for batch in self.buffer if batch["success"])
        return successes / len(self.buffer)

    def get_average_loss(self) -> float:
        """Calculate average loss from recent successful batches.

        Returns
        -------
        float
            Average loss from successful batches in buffer
        """
        successful_batches = [b for b in self.buffer if b["success"]]
        if not successful_batches:
            return float("inf")

        total_loss: float = sum(float(b["loss"]) for b in successful_batches)
        return total_loss / len(successful_batches)

    def get_average_iterations(self) -> float:
        """Calculate average iterations from recent batches.

        Returns
        -------
        float
            Average number of iterations per batch
        """
        if not self.buffer:
            return 0.0

        total_iterations: int = sum(int(b["iterations"]) for b in self.buffer)
        return float(total_iterations) / len(self.buffer)

    def get_statistics(self) -> dict[str, Any]:
        """Return comprehensive statistics dictionary.

        Returns
        -------
        dict
            Dictionary containing:
            - total_batches: Total batches processed (all time)
            - total_successes: Total successful batches (all time)
            - total_failures: Total failed batches (all time)
            - success_rate: Success rate from recent batches
            - average_loss: Average loss from recent successful batches
            - average_iterations: Average iterations per batch
            - error_distribution: Dictionary of error type counts
            - recent_batches: List of recent batch records
        """
        return {
            "total_batches": self.total_batches,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": self.get_success_rate(),
            "average_loss": self.get_average_loss(),
            "average_iterations": self.get_average_iterations(),
            "error_distribution": dict(self.error_counts),
            "recent_batches": list(self.buffer),
        }

    def __repr__(self) -> str:
        """Return string representation of statistics."""
        return (
            f"BatchStatistics(total={self.total_batches}, "
            f"successes={self.total_successes}, "
            f"failures={self.total_failures}, "
            f"success_rate={self.get_success_rate():.2%})"
        )
