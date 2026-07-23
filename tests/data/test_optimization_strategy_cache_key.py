"""Regression: strategy cache key must reflect memory-scaled recommendations.

``get_processing_strategy`` memoized on ``hash((size, method, memory_limit_mb))``.
Two ``analyze_dataset`` calls with equal ``.size`` but different ``sigma`` can
cross the ``memory_limit_mb`` threshold differently, producing different
``recommended_chunk_size``/``batch_size`` — yet the old key could not tell them
apart, so the second call served the first's stale strategy. The fix widens the
key to include the size/category/chunk/batch/progressive fields the strategy
actually depends on.
"""

import numpy as np

from xpcsjax.data.optimization import DatasetOptimizer


def test_equal_size_different_sigma_get_distinct_cached_strategies():
    """Equal .size, sigma crossing the memory limit -> distinct strategies.

    With float64 data of 100k points (0.76 MB), the working-set estimate is
    ~3.05 MB without sigma and ~6.1 MB with an equal-shaped sigma. A 4.0 MB
    limit leaves the no-sigma case unscaled (chunk_size == size) but scales the
    with-sigma case down — so the two recommended chunk sizes differ.
    """
    opt = DatasetOptimizer(memory_limit_mb=4.0)
    data = np.zeros(100_000, dtype=np.float64)
    sigma = np.zeros(100_000, dtype=np.float64)

    info_no_sigma = opt.analyze_dataset(data)
    info_with_sigma = opt.analyze_dataset(data, sigma)

    # Sanity: equal size, but the memory scaling made the recommendations differ.
    assert info_no_sigma.size == info_with_sigma.size
    assert info_no_sigma.recommended_chunk_size != info_with_sigma.recommended_chunk_size

    strat_no_sigma = opt.get_processing_strategy(info_no_sigma)
    strat_with_sigma = opt.get_processing_strategy(info_with_sigma)

    # Under the old size-only key the second call collides and returns the first
    # (stale) strategy. The widened key keeps them distinct.
    assert strat_no_sigma.chunk_size == info_no_sigma.recommended_chunk_size
    assert strat_with_sigma.chunk_size == info_with_sigma.recommended_chunk_size
    assert strat_no_sigma.chunk_size != strat_with_sigma.chunk_size
