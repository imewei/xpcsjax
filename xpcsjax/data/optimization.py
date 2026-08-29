"""Dataset size-aware optimization for the NLSQ pipeline.

Memory-efficient data processing strategies for different dataset sizes.
Implements chunked processing, progressive loading, and batch optimization
for NLSQ optimization.

Key features
------------
- Size-aware processing strategies (<1M, 1-10M, >20M points)
- Memory-efficient chunked processing for large datasets
- Progressive loading with intelligent caching
- JAX-optimized batch processing
- Integration with NLSQ pipelines

Notes
-----
xpcsjax is NLSQ-only by design. The ``method`` argument threaded through
this module exists so the public boundary can *reject* non-NLSQ methods
(Bayesian sampling: CMC / MCMC) with a clear :class:`ValueError`; those
pathways are permanently out of scope (see ``CLAUDE.md``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from xpcsjax.data.types import DatasetInfo, ProcessingStrategy
from xpcsjax.utils.logging import get_logger, log_performance

# JAX imports with fallback
try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np  # type: ignore[misc]


logger = get_logger(__name__)


class DatasetOptimizer:
    """Dataset size-aware optimization for NLSQ.

    Provides memory-efficient processing strategies based on dataset size:
    - Small (<1M): In-memory processing with full JAX acceleration
    - Medium (1-10M): Efficient batching with partial memory optimization
    - Large (>20M): Distributed chunked processing with streaming
    """

    def __init__(
        self,
        memory_limit_mb: float = 4096.0,
        enable_compression: bool = True,
        max_workers: int | None = None,
    ):
        """Initialize the dataset optimizer.

        Parameters
        ----------
        memory_limit_mb : float, optional
            Maximum memory usage in MB.
        enable_compression : bool, optional
            Enable data compression for large datasets.
        max_workers : int or None, optional
            Maximum parallel workers; ``None`` auto-detects via
            :meth:`_detect_optimal_workers`.
        """
        self.memory_limit_mb = memory_limit_mb
        self.enable_compression = enable_compression
        self.max_workers = max_workers or self._detect_optimal_workers()

        # Strategy cache for repeated operations
        self._strategy_cache: dict[int, ProcessingStrategy] = {}

        logger.info("Dataset optimizer initialized:")
        logger.info(f"  Memory limit: {memory_limit_mb:.1f} MB")
        logger.info(f"  Compression: {enable_compression}")
        logger.info(f"  Workers: {self.max_workers}")

    def analyze_dataset(
        self,
        data: np.ndarray,
        sigma: np.ndarray | None = None,
    ) -> DatasetInfo:
        """Analyze dataset characteristics and recommend a processing strategy.

        Categorizes the dataset by point count (``small`` < 1M, ``medium`` <
        10M, ``large`` otherwise), derives chunk and batch sizes, then scales
        them down with a 20% safety margin if the estimated memory usage
        exceeds :attr:`memory_limit_mb`.

        Parameters
        ----------
        data : numpy.ndarray
            Primary data array.
        sigma : numpy.ndarray or None, optional
            Optional uncertainty array.

        Returns
        -------
        DatasetInfo
            Analysis results and processing-size recommendations.
        """
        # Calculate memory usage
        memory_usage = self._calculate_memory_usage(data, sigma)

        # Categorize dataset size
        size = data.size
        if size < 1_000_000:
            category = "small"
            chunk_size = size  # Process everything at once
            batch_size = min(1000, size // 10)
            progressive_loading = False
        elif size < 10_000_000:
            category = "medium"
            chunk_size = min(500_000, size // 4)
            batch_size = min(500, size // 100)
            progressive_loading = True
        else:
            category = "large"
            chunk_size = min(100_000, size // 20)
            batch_size = min(100, size // 1000)
            progressive_loading = True

        # Adjust for memory constraints. Floor at 1: an aggressive scale_factor
        # (e.g. a very small/zero memory_limit_mb) would otherwise round down
        # to a 0 chunk_size/batch_size, which downstream chunked-iterator code
        # divides by, raising ZeroDivisionError instead of a clear guard here.
        if memory_usage > self.memory_limit_mb:
            scale_factor = self.memory_limit_mb / memory_usage
            chunk_size = max(1, int(chunk_size * scale_factor * 0.8))  # 20% safety margin
            batch_size = max(1, int(batch_size * scale_factor * 0.8))

        dataset_info = DatasetInfo(
            size=size,
            category=category,
            memory_usage_mb=memory_usage,
            recommended_chunk_size=chunk_size,
            recommended_batch_size=batch_size,
            use_progressive_loading=progressive_loading,
        )

        logger.info("Dataset analysis complete:")
        logger.info(f"  Size: {size:,} points ({category})")
        logger.info(f"  Memory: {memory_usage:.1f} MB")
        logger.info(f"  Chunk size: {chunk_size:,}")
        logger.info(f"  Batch size: {batch_size}")

        return dataset_info

    def get_processing_strategy(
        self,
        dataset_info: DatasetInfo,
        method: str = "nlsq",
    ) -> ProcessingStrategy:
        """Build an optimized processing strategy for the NLSQ method.

        Results are memoized in :attr:`_strategy_cache` keyed by dataset size,
        method, and memory limit.

        Parameters
        ----------
        dataset_info : DatasetInfo
            Dataset analysis results from :meth:`analyze_dataset`.
        method : str, optional
            ``"nlsq"`` (xpcsjax is NLSQ-only). Used only as part of the
            cache key and in log messages here.

        Returns
        -------
        ProcessingStrategy
            Strategy (chunk/batch sizes, worker count, JAX config) optimized
            for the dataset.
        """
        cache_key = hash(
            (
                dataset_info.size,
                dataset_info.category,
                dataset_info.recommended_chunk_size,
                dataset_info.recommended_batch_size,
                dataset_info.use_progressive_loading,
                method,
                self.memory_limit_mb,
            )
        )

        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]

        # Base strategy from dataset analysis
        chunk_size = dataset_info.recommended_chunk_size
        batch_size = dataset_info.recommended_batch_size

        # NLSQ-only: larger batches than the conservative default.
        batch_size = min(batch_size * 2, chunk_size)
        jax_config = {
            "xla_python_client_mem_fraction": "0.8",
            "jax_enable_x64": "true",  # Float64 mandatory (params span 6+ orders)
            "jax_platforms": "cpu",
        }

        # Determine parallel workers based on dataset size
        if dataset_info.category == "small":
            workers = 1  # No need for parallelization
        elif dataset_info.category == "medium":
            workers = min(self.max_workers, 4)
        else:
            workers = self.max_workers

        strategy = ProcessingStrategy(
            chunk_size=chunk_size,
            batch_size=batch_size,
            memory_limit_mb=self.memory_limit_mb,
            use_caching=dataset_info.use_progressive_loading,
            use_compression=self.enable_compression and dataset_info.category == "large",
            parallel_workers=workers,
            jax_config=jax_config,
        )

        self._strategy_cache[cache_key] = strategy

        logger.info(f"Processing strategy for {method.upper()}:")
        logger.info(f"  Chunk size: {chunk_size:,}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Workers: {workers}")
        logger.info(f"  Caching: {strategy.use_caching}")
        logger.info(f"  Compression: {strategy.use_compression}")

        return strategy

    def create_chunked_iterator(
        self,
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        chunk_size: int,
    ) -> Iterator[tuple[np.ndarray, ...]]:
        """Create a memory-efficient chunked iterator for large datasets.

        The ``data``, ``sigma``, and ``phi`` arrays are sliced along the data
        axis; ``t1`` and ``t2`` are 2D time meshgrids and are emitted whole on
        every chunk. Chunks are converted to JAX arrays when JAX is available.

        Parameters
        ----------
        data : numpy.ndarray
            Primary data array, sliced per chunk.
        sigma : numpy.ndarray
            Uncertainty array, sliced per chunk (may be ``None``).
        t1 : numpy.ndarray
            First time meshgrid, emitted whole on each chunk.
        t2 : numpy.ndarray
            Second time meshgrid, emitted whole on each chunk.
        phi : numpy.ndarray
            Phi-angle array, sliced per chunk when it has more than one entry.
        chunk_size : int
            Number of data points per chunk.

        Yields
        ------
        tuple of numpy.ndarray
            ``(data_chunk, sigma_chunk, t1, t2, phi_chunk)`` for each chunk.
        """
        n_data = len(data)
        n_chunks = (n_data + chunk_size - 1) // chunk_size

        logger.info(
            f"Creating chunked iterator: {n_chunks} chunks of {chunk_size:,} points",
        )

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_data)

            # Extract chunk with proper indexing
            data_chunk = data[start_idx:end_idx]
            sigma_chunk = sigma[start_idx:end_idx] if sigma is not None else None

            # Time arrays are 2D meshgrids - don't chunk them by data indices
            # They should remain constant for all chunks as they represent the time grid
            t1_chunk = t1  # Keep full 2D meshgrid
            t2_chunk = t2  # Keep full 2D meshgrid
            if len(phi) > 1:
                if len(phi) != n_data:
                    raise ValueError(
                        f"phi length ({len(phi)}) must match data length ({n_data}) "
                        "when phi is a per-data-point array"
                    )
                phi_chunk = phi[start_idx:end_idx]
            else:
                phi_chunk = phi

            # Convert to JAX arrays if available
            if JAX_AVAILABLE:
                data_chunk = jnp.array(data_chunk)  # type: ignore[assignment]
                if sigma_chunk is not None:
                    sigma_chunk = jnp.array(sigma_chunk)  # type: ignore[assignment]
                t1_chunk = jnp.array(t1_chunk)  # type: ignore[assignment]
                t2_chunk = jnp.array(t2_chunk)  # type: ignore[assignment]
                phi_chunk = jnp.array(phi_chunk)  # type: ignore[assignment]

            yield data_chunk, sigma_chunk, t1_chunk, t2_chunk, phi_chunk

    @log_performance()
    def optimize_for_nlsq(
        self,
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Optimize data processing specifically for NLSQ.

        Analyzes the dataset, resolves the NLSQ strategy, applies the JAX
        environment configuration, and (for ``large`` datasets only) attaches a
        chunked iterator.

        Parameters
        ----------
        data : numpy.ndarray
            Primary data array.
        sigma : numpy.ndarray
            Uncertainty array.
        t1 : numpy.ndarray
            First time meshgrid.
        t2 : numpy.ndarray
            Second time meshgrid.
        phi : numpy.ndarray
            Phi-angle array.
        **kwargs : Any
            Additional optimization parameters (forwarded for API symmetry).

        Returns
        -------
        dict
            Optimized processing configuration with keys ``dataset_info``,
            ``strategy``, ``chunked_iterator``, and ``preprocessing_time``.

        Notes
        -----
        The JAX environment variables are only honored before JAX's first
        import; if JAX is already imported they are effectively ignored.
        """
        dataset_info = self.analyze_dataset(data, sigma)
        strategy = self.get_processing_strategy(dataset_info, "nlsq")

        # Apply JAX configuration
        if JAX_AVAILABLE:
            for key, value in strategy.jax_config.items():
                import os

                # NOTE: These env vars are only effective before JAX's first import.
                # If JAX is already imported, use jax.config.update() instead.
                os.environ[key.upper()] = value
                logger.debug(
                    "Set JAX env var %s=%s (may be ignored if JAX already imported)",
                    key.upper(),
                    value,
                )

        optimization_config = {
            "dataset_info": dataset_info,
            "strategy": strategy,
            "chunked_iterator": None,
            "preprocessing_time": 0.0,
        }

        # Setup chunked processing for large datasets
        if dataset_info.category == "large":
            # create_chunked_iterator returns a lazy generator: no chunk work runs
            # until it is consumed downstream, so there is no preprocessing cost to
            # time here. preprocessing_time stays 0.0 by design rather than recording
            # the (meaningless) generator-object construction time.
            optimization_config["chunked_iterator"] = self.create_chunked_iterator(
                data,
                sigma,
                t1,
                t2,
                phi,
                strategy.chunk_size,
            )

        return optimization_config

    @log_performance()
    def estimate_processing_time(
        self,
        dataset_info: DatasetInfo,
        method: str = "nlsq",
    ) -> dict[str, float]:
        """Estimate processing time from empirical throughput rates.

        Parameters
        ----------
        dataset_info : DatasetInfo
            Dataset analysis results.
        method : str, optional
            Processing method; ``"nlsq"`` uses the fast JAX-accelerated rate.

        Returns
        -------
        dict
            Time estimates with keys ``estimated_seconds``,
            ``estimated_minutes``, ``effective_rate``, and ``efficiency``.
        """
        # Base processing rates (points per second) based on empirical measurements
        if method.lower() == "nlsq":
            base_rate = 100000 if JAX_AVAILABLE else 20000  # NLSQ fast with JAX
        else:
            base_rate = 1000

        # Adjust for dataset size effects
        if dataset_info.category == "small":
            efficiency = 1.0  # Full efficiency
        elif dataset_info.category == "medium":
            efficiency = 0.8  # Some overhead from chunking
        else:
            efficiency = 0.6  # More overhead from distributed processing

        effective_rate = base_rate * efficiency
        estimated_time = dataset_info.size / effective_rate

        return {
            "estimated_seconds": estimated_time,
            "estimated_minutes": estimated_time / 60,
            "effective_rate": effective_rate,
            "efficiency": efficiency,
        }

    def _calculate_memory_usage(
        self,
        data: np.ndarray,
        sigma: np.ndarray | None = None,
    ) -> float:
        """Estimate working-set memory usage in MB.

        Sums the byte sizes of ``data`` (and ``sigma`` when given) and applies a
        factor of 4 to account for intermediate computations.

        Parameters
        ----------
        data : numpy.ndarray
            Primary data array.
        sigma : numpy.ndarray or None, optional
            Optional uncertainty array.

        Returns
        -------
        float
            Estimated memory usage in megabytes.
        """
        memory_bytes = data.nbytes
        if sigma is not None:
            memory_bytes += sigma.nbytes

        # Add overhead for intermediate computations (factor of 3-4)
        memory_bytes *= 4

        return memory_bytes / (1024 * 1024)  # Convert to MB

    def _detect_optimal_workers(self) -> int:
        """Detect the optimal number of parallel workers.

        Returns
        -------
        int
            CPU count capped at 8, or 4 if the count cannot be determined.
        """
        try:
            import os

            return min(os.cpu_count() or 1, 8)  # Cap at 8 workers
        except (OSError, AttributeError):
            return 4  # Safe default


# Convenience functions for integration with existing codebase
def create_dataset_optimizer(**kwargs: Any) -> DatasetOptimizer:
    """Create a :class:`DatasetOptimizer` with sensible defaults.

    Parameters
    ----------
    **kwargs : Any
        Forwarded to :class:`DatasetOptimizer`; only ``memory_limit_mb``,
        ``enable_compression``, and ``max_workers`` are retained, other keys
        are silently dropped.

    Returns
    -------
    DatasetOptimizer
        A configured optimizer instance.
    """
    # Filter kwargs to only include valid parameters for DatasetOptimizer
    valid_params = {"memory_limit_mb", "enable_compression", "max_workers"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return DatasetOptimizer(**filtered_kwargs)


def optimize_for_method(
    data: np.ndarray,
    sigma: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    method: str = "nlsq",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one-shot optimization for a given method.

    Parameters
    ----------
    data : numpy.ndarray
        Primary data array.
    sigma : numpy.ndarray
        Uncertainty array.
    t1 : numpy.ndarray
        First time meshgrid.
    t2 : numpy.ndarray
        Second time meshgrid.
    phi : numpy.ndarray
        Phi-angle array.
    method : str, optional
        Must be ``"nlsq"`` (xpcsjax is NLSQ-only; see ``CLAUDE.md``).
    **kwargs : Any
        Additional optimization parameters forwarded to
        :func:`create_dataset_optimizer`.

    Returns
    -------
    dict
        Optimization configuration dictionary.

    Raises
    ------
    ValueError
        If ``method`` is not ``"nlsq"``. This is an intentional defensive
        guard: Bayesian sampling methods (CMC / MCMC) are permanently out of
        scope for xpcsjax and are rejected at this boundary rather than
        silently routed.
    """
    optimizer = create_dataset_optimizer(**kwargs)

    if method.lower() != "nlsq":
        raise ValueError(
            f"Unknown method: {method}. xpcsjax is NLSQ-only; "
            "Bayesian sampling methods (CMC/MCMC) are permanently out of scope."
        )
    return optimizer.optimize_for_nlsq(data, sigma, t1, t2, phi)


# Export main classes and functions
__all__ = [
    "DatasetInfo",
    "ProcessingStrategy",
    "DatasetOptimizer",
    "create_dataset_optimizer",
    "optimize_for_method",
]
