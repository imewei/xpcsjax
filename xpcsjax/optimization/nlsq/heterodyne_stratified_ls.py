"""Heterodyne stratified least-squares solver.

Mirrors the homodyne stratified-LS path (strategies/stratified_ls.py) for the
heterodyne two_component model. Reuses the model-agnostic chunking helpers and
adds a heterodyne-specific joint pointwise residual whose per-angle scaling is
expanded from the varying parameter vector each iteration.

Parameter packing is physics-first ([physics | scaling]) to match the rest of
the heterodyne result handling. The objective equals the in-memory joint fit's
objective; the only intended behavioral change is the seed-42 pre-shuffle.
"""
from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.strategies.chunking import (
    create_angle_stratified_indices,
)

_SHUFFLE_SEED = 42


def reorder_for_stratification(
    phi_flat: np.ndarray,
    target_chunk_size: int = 100_000,
    *,
    shuffle: bool = True,
    seed: int = _SHUFFLE_SEED,
) -> tuple[np.ndarray, list[int]]:
    """Return a permutation that angle-stratifies (and optionally shuffles) points.

    Parameters
    ----------
    phi_flat : np.ndarray
        Per-point angle labels, shape ``(N,)``.
    target_chunk_size : int
        Interleaved-stratification chunk target (model-agnostic, from chunking.py).
    shuffle : bool
        If True, apply a fixed-seed permutation after stratification to break
        angle-sequential ordering (homodyne local-minimum-avoidance parity).
    seed : int
        Shuffle seed (fixed at 42 for reproducibility; matches homodyne).

    Returns
    -------
    (perm, chunk_sizes) : tuple[np.ndarray, list[int]]
        ``perm`` reorders any per-point array; ``chunk_sizes`` are the
        interleaved chunk sizes from stratification.
    """
    perm, chunk_sizes = create_angle_stratified_indices(phi_flat, target_chunk_size)
    perm = np.asarray(perm, dtype=np.int64)
    if shuffle:
        rng = np.random.RandomState(seed)
        perm = perm[rng.permutation(len(perm))]
    return perm, list(chunk_sizes)
