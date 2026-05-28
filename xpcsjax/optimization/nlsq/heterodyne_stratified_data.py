"""Heterodyne stratified data adapter for hybrid-streaming Phase 2.

Converts ``(n_phi, n_t, n_t)`` two-time correlation matrices + phi angles
into the flat stratified layout that :func:`create_stratified_chunks` consumes.

Field names mirror the homodyne ``StratifiedData`` contract exactly so that
the same chunker (``create_stratified_chunks``) works unchanged with the
heterodyne residual function.

The heterodyne two-time matrix ``C2[i, j]`` is the off-diagonal correlation
at times ``(t[i], t[j])``.  Diagonal entries (``i == j``) are autocorrelation
artefacts and are excluded from the fit, matching the homodyne convention in
``StratifiedResidualFunction._diag_mask``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel


@dataclass
class HeterodyneStratifiedData:
    """Flat stratified data layout for heterodyne XPCS.

    Field names are ground-truth from ``create_stratified_chunks``:
    ``phi_flat``, ``t1_flat``, ``t2_flat``, ``g2_flat``, ``sigma``,
    ``q``, ``L``, ``dt``, ``chunk_sizes``.

    Attributes:
        phi_flat: Flattened phi labels for every (t1, t2) pair, shape (N_total,).
        t1_flat:  Flattened t1 values, shape (N_total,).
        t2_flat:  Flattened t2 values, shape (N_total,).
        g2_flat:  Flattened observed C2 values, shape (N_total,).
        sigma:    Uncertainty array — stored as a 3-D (n_phi, n_t, n_t) array
                  for compatibility with the ``StratifiedResidualFunction``
                  interface which expects a 3-D sigma. Uniform (all ones) when
                  ``weights=None``.
        q:        Scattering wavevector magnitude (float).
        L:        Path-length placeholder (float, 0.0 — not used by heterodyne
                  kernel, but required by ``create_stratified_chunks``).
        dt:       Time step (float).
        chunk_sizes: List giving the number of flat points per angle slab.
                     Each slab is one angle × n_t × n_t off-diagonal points.
        n_phi:    Number of phi angles.
        n_t:      Number of time points.
    """

    phi_flat: np.ndarray
    t1_flat: np.ndarray
    t2_flat: np.ndarray
    g2_flat: np.ndarray
    sigma: np.ndarray  # shape (n_phi, n_t, n_t), all ones if unweighted
    q: float
    L: float
    dt: float
    chunk_sizes: list[int]
    n_phi: int
    n_t: int

    # per-angle mapping: chunk index → angle index (same as phi index here)
    angle_indices: list[int] = field(default_factory=list)


def build_heterodyne_stratified_data(
    model: HeterodyneModel,
    c2: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray | None = None,
) -> HeterodyneStratifiedData:
    """Build a :class:`HeterodyneStratifiedData` from model + raw C2 data.

    Args:
        model:   Configured :class:`HeterodyneModel` — provides ``t``, ``q``,
                 ``dt``.  ``model.t`` must already be synced to the data time
                 axis (call ``model.sync_time_axis(t)`` before calling here if
                 necessary).
        c2:      Observed two-time correlation matrices.  Accepted shapes:

                 * ``(n_phi, n_t, n_t)`` — multi-angle (standard)
                 * ``(n_t, n_t)``        — single angle; a leading axis is added

        phi:     Phi angles in degrees, shape ``(n_phi,)``.
        weights: Optional inverse-variance weights, same shape as ``c2``
                 (broadcastable).  ``None`` → uniform sigma = 1.

    Returns:
        :class:`HeterodyneStratifiedData` ready for ``create_stratified_chunks``.

    Raises:
        ValueError: On shape mismatch between ``c2``, ``phi``, and ``model.t``.
    """

    # ------------------------------------------------------------------ #
    # 1. Normalise c2 to (n_phi, n_t, n_t)                               #
    # ------------------------------------------------------------------ #
    c2_arr = np.asarray(c2, dtype=np.float64)
    if c2_arr.ndim == 2:
        c2_arr = c2_arr[np.newaxis, ...]  # (1, n_t, n_t)

    if c2_arr.ndim != 3:
        raise ValueError(f"c2 must be 2-D or 3-D, got shape {c2.shape}")

    n_phi_data, n_t_data, n_t2 = c2_arr.shape
    if n_t_data != n_t2:
        raise ValueError(
            f"c2 must be square in the time dimensions, got ({n_t_data}, {n_t2})"
        )

    phi_arr = np.asarray(phi, dtype=np.float64).ravel()
    if phi_arr.shape[0] != n_phi_data:
        raise ValueError(
            f"phi length ({phi_arr.shape[0]}) must match c2 first dim ({n_phi_data})"
        )

    # ------------------------------------------------------------------ #
    # 2. Sync model time axis if necessary                                #
    # ------------------------------------------------------------------ #
    t_model = np.asarray(model.t, dtype=np.float64)
    if len(t_model) != n_t_data:
        model.sync_time_axis(
            np.arange(1, n_t_data + 1, dtype=np.float64) * model.dt
        )
        t_model = np.asarray(model.t, dtype=np.float64)

    n_phi = n_phi_data
    n_t = n_t_data

    # ------------------------------------------------------------------ #
    # 3. Build sigma (uncertainty) array                                  #
    # ------------------------------------------------------------------ #
    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.ndim == 2:
            w_arr = w_arr[np.newaxis, ...]
        # Broadcast to (n_phi, n_t, n_t)
        w_arr = np.broadcast_to(w_arr, (n_phi, n_t, n_t)).copy()
        # sigma = 1/sqrt(weight), guard against zeros
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.where(w_arr > 0, 1.0 / np.sqrt(w_arr), 1.0)
    else:
        sigma = np.ones((n_phi, n_t, n_t), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 4. Flatten angle by angle into per-slab arrays                      #
    # ------------------------------------------------------------------ #
    # For each angle we flatten the full n_t × n_t grid (including diagonal).
    # The diagonal is handled by the residual function via _diag_mask, which
    # zeros residuals where t1_index == t2_index.  We follow the same
    # convention as homodyne: include ALL pairs in the flat arrays.
    chunk_sizes: list[int] = []
    angle_indices: list[int] = []

    phi_slabs: list[np.ndarray] = []
    t1_slabs: list[np.ndarray] = []
    t2_slabs: list[np.ndarray] = []
    g2_slabs: list[np.ndarray] = []

    # Pre-build the t1/t2 grid (same for every angle)
    t1_grid, t2_grid = np.meshgrid(t_model, t_model, indexing="ij")  # (n_t, n_t)
    t1_flat_angle = t1_grid.ravel()
    t2_flat_angle = t2_grid.ravel()
    slab_size = n_t * n_t

    for angle_idx in range(n_phi):
        phi_val = phi_arr[angle_idx]
        g2_slab = c2_arr[angle_idx].ravel()  # (n_t*n_t,)
        phi_slab = np.full(slab_size, phi_val, dtype=np.float64)

        phi_slabs.append(phi_slab)
        t1_slabs.append(t1_flat_angle.copy())
        t2_slabs.append(t2_flat_angle.copy())
        g2_slabs.append(g2_slab)

        chunk_sizes.append(slab_size)
        angle_indices.append(angle_idx)

    phi_flat = np.concatenate(phi_slabs)
    t1_flat = np.concatenate(t1_slabs)
    t2_flat = np.concatenate(t2_slabs)
    g2_flat = np.concatenate(g2_slabs)

    return HeterodyneStratifiedData(
        phi_flat=phi_flat,
        t1_flat=t1_flat,
        t2_flat=t2_flat,
        g2_flat=g2_flat,
        sigma=sigma,
        q=float(model.q),
        L=0.0,  # not used by heterodyne kernel; required by create_stratified_chunks
        dt=float(model.dt),
        chunk_sizes=chunk_sizes,
        n_phi=n_phi,
        n_t=n_t,
        angle_indices=angle_indices,
    )
