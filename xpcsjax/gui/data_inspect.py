"""JAX-free HDF5 inspection for the Data tab (h5py only — no xpcsjax.data/JAX)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from xpcsjax.gui.views.raster import rasterize


@dataclass(frozen=True)
class DatasetInfo:
    """Shape/dtype metadata for one HDF5 dataset (no array data loaded)."""

    name: str
    shape: tuple
    dtype: str


def read_hdf5_metadata(path: str | Path) -> list[DatasetInfo]:
    """List every dataset in the file with shape + dtype, loading no array data."""
    infos: list[DatasetInfo] = []

    def _visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            infos.append(DatasetInfo(name=name, shape=tuple(obj.shape), dtype=str(obj.dtype)))

    with h5py.File(path, "r") as f:
        f.visititems(_visit)
    return infos


# Shared C₂ layout descriptor (spec §6) — keyed by data_type. The GUI preview reads
# HDF5 directly (JAX-free, bypassing xpcsjax.data), so it must NOT re-derive the
# layout ad hoc: this table is the single source of truth and MUST mirror the loader.
# On disk BOTH real formats store C₂ as an HDF5 **group of per-angle 2-D
# half-matrices** (NOT a 3-D dataset); the full matrix is reconstructed as
# ``c2_half + c2_half.T`` with the diagonal halved, mirroring
# ``xpcsjax/data/xpcs_loader.py::_reconstruct_full_matrix`` (verified at
# xpcs_loader.py:1283/1356-1357, :1553/1578-1580, :1695-1708). ``key_order`` matches
# the loader: APS-old iterates keys in HDF5 creation order, APS-U sorts them.
_C2_PREVIEW_LAYOUTS: dict[str, dict] = {
    "aps_old": {"group": "exchange/C2T_all", "key_order": "creation"},          # integer keys, creation order
    "aps_u": {"group": "xpcs/twotime/correlation_map", "key_order": "sorted"},  # zero-padded c2_* keys
}


def _reconstruct_c2(half: np.ndarray) -> np.ndarray:
    """Full two-time matrix from a stored half: ``c2_half + c2_half.T``, diagonal halved.

    Mirrors ``xpcsjax/data/xpcs_loader.py::_reconstruct_full_matrix``. It commutes
    with equal-stride decimation — ``_reconstruct_c2(half[::s, ::s])`` equals
    ``_reconstruct_c2(half)[::s, ::s]`` — so a bounded strided read stays correct.
    """
    full = half + half.T
    full[np.diag_indices(full.shape[0])] /= 2
    return full


def read_c2_preview(
    path: str | Path,
    dataset: str,
    *,
    data_type: str | None = None,
    phi_index: int = 0,
    max_dim: int = 512,
) -> np.ndarray | None:
    """Read one per-angle two-time C₂ matrix and block-mean downsample it for display.

    For a known ``data_type`` the angle's matrix lives in the HDF5 group named by
    ``_C2_PREVIEW_LAYOUTS`` as a 2-D **half**-matrix; we pick the ``phi_index``-th
    key (creation order for APS-old, sorted for APS-U — mirroring the loader), read it
    with a bounded strided hyperslab, and reconstruct it via :func:`_reconstruct_c2`.
    Returns ``None`` ("preview unavailable") when the C₂ group is absent or empty — it
    never renders a guessed layout. When ``data_type`` is ``None`` a best-effort
    heuristic reads ``dataset`` directly (raw 2-D matrix, or one slice of a 3-D array)
    without reconstruction. ``dataset`` is ignored for a known format.
    """
    with h5py.File(path, "r") as f:
        layout = _C2_PREVIEW_LAYOUTS.get(data_type) if data_type else None
        if layout is not None:
            grp = f.get(layout["group"])
            if not isinstance(grp, h5py.Group) or len(grp) == 0:
                return None  # C₂ group absent/empty → preview unavailable
            keys = sorted(grp.keys()) if layout["key_order"] == "sorted" else list(grp.keys())
            half_dset = grp[keys[max(0, min(int(phi_index), len(keys) - 1))]]
            if half_dset.ndim != 2 or half_dset.shape[0] != half_dset.shape[1]:
                return None
            step = max(1, int(np.ceil(max(half_dset.shape) / max_dim)))
            arr = _reconstruct_c2(np.asarray(half_dset[::step, ::step]))  # bounded + symmetric
        elif dataset not in f:
            return None
        else:  # data_type unknown: best-effort raw read, no reconstruction
            dset = f[dataset]
            if dset.ndim == 3:  # heuristic: assume (n_phi, t, t)
                idx = max(0, min(int(phi_index), dset.shape[0] - 1))
                step = max(1, int(np.ceil(max(dset.shape[1:]) / max_dim)))
                arr = np.asarray(dset[idx, ::step, ::step])
            elif dset.ndim == 2:
                step = max(1, int(np.ceil(max(dset.shape) / max_dim)))
                arr = np.asarray(dset[::step, ::step])
            else:
                return None
    return rasterize(arr, max_dim=max_dim)
