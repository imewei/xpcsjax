"""APS old / APS-U HDF5 format detection and readers.

Extracted from :mod:`xpcsjax.data.xpcs_loader` (2026-09-15 review, finding
D6) as a thin move: ``_load_aps_old_format``/``_load_aps_u_format`` were
instance methods reading ``self.config`` and calling a handful of other
loader methods; they are now free functions taking the loader instance
explicitly as their first argument, and the original methods on
``XPCSDataLoader`` delegate to them unchanged. ``_detect_format`` never used
``self`` at all and is now a plain function of ``hdf_path``. No behaviour
change.

Low-level shared guards (``XPCSDataFormatError``, ``_check_square_matrix``,
``_check_frame_count``, ``_check_allocation_budget``) stay defined in
``xpcs_loader.py`` and are imported here lazily (inside the functions that
need them) to avoid a circular import, since ``xpcs_loader.py`` imports this
module at module level to build its delegating methods — the same
in-function-import pattern already used elsewhere in this codebase for
cycle avoidance (see ``optimization/`` per the root CLAUDE.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from xpcsjax.utils.logging import get_logger, log_performance

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from xpcsjax.data.xpcs_loader import XPCSDataLoader

logger = get_logger(__name__)

# HDF5 chunk-cache tuning for the two production format loaders.
# APS correlation matrices are (n_t, n_t) float64.  At worst-case n_t=1000
# each matrix is 8 MB; we size the cache to hold ~12 matrices comfortably.
# rdcc_nslots must be a prime roughly 100× the number of cached chunks.
_HDF5_RDCC_N_MATRICES: int = 12
_HDF5_RDCC_MATRIX_BYTES: int = 1000 * 1000 * 8  # float64, n_t=1000 worst-case
_HDF5_RDCC_NBYTES: int = _HDF5_RDCC_N_MATRICES * _HDF5_RDCC_MATRIX_BYTES  # 96 MB
_HDF5_RDCC_NSLOTS: int = 6257  # prime; ≥ 100 × _HDF5_RDCC_N_MATRICES
_HDF5_RDCC_W0: float = 0.75  # prefer evicting chunks not likely to be re-read


def guard_aps_u_intermediate_allocation(
    corr_group: Any, c2_keys: list[str], valid_bin_indices: Any, *, source: str
) -> None:
    """Bound an intermediate correlation-matrix list BEFORE it is accumulated.

    Both the APS-U loader and the APS-old quality-filtering branch reconstruct
    and append every candidate correlation matrix to a Python list
    (``c2_matrices_for_filtering`` / ``candidate_matrices``) prior to the
    post-selection allocation guard on the final stacked buffer. A crafted file
    with many large bins could therefore exhaust RAM during that accumulation,
    defeating :data:`MAX_CORRELATION_ALLOC_BYTES`. Probe the first valid
    matrix's shape via h5py metadata (``.shape``/``.dtype`` only — no full
    array read) and apply the same square/frame/budget guards used on the
    final buffer, scaled by the number of matrices that will be loaded
    (SEC-2 parity). A no-op when no valid bin index is in range. Despite the
    name (kept for git-blame continuity), this helper is format-agnostic —
    reused as-is for both loader paths rather than duplicated.
    """
    from xpcsjax.data.xpcs_loader import (
        _check_allocation_budget,
        _check_frame_count,
        _check_square_matrix,
    )

    in_range = [bi for bi in valid_bin_indices if bi < len(c2_keys)]
    if not in_range:
        return
    probe = corr_group[c2_keys[in_range[0]]]
    probe_shape = tuple(int(d) for d in probe.shape)
    itemsize = np.dtype(probe.dtype).itemsize
    # Reconstructed matrix is square (c2_half + c2_half.T), so the half-matrix is
    # square (n_t, n_t) and bounds the per-matrix reconstructed size.
    _check_square_matrix(probe_shape, source=source)
    _check_frame_count(probe_shape[-1], source=source)
    _check_allocation_budget(len(in_range), probe_shape[-1], itemsize, source=source)


@log_performance(threshold=0.1)
def detect_format(hdf_path: str) -> str:
    """Detect whether an HDF5 file is APS old or APS-U new format.

    Returns
    -------
    str
        ``"aps_u"`` for APS-U format, ``"aps_old"`` for APS old format, or
        ``"unknown"`` for unrecognized or empty files.
    """
    with h5py.File(hdf_path, "r") as f:
        # Check for APS-U format keys
        if (
            "xpcs" in f
            and "qmap" in f["xpcs"]
            and "dynamic_v_list_dim0" in f["xpcs/qmap"]
            and "twotime" in f["xpcs"]
            and "correlation_map" in f["xpcs/twotime"]
        ):
            return "aps_u"

        # Check for APS old format keys
        if (
            "xpcs" in f
            and "dqlist" in f["xpcs"]
            and "dphilist" in f["xpcs"]
            and "exchange" in f
            and "C2T_all" in f["exchange"]
        ):
            return "aps_old"

        # Log the top-level keys for debugging unrecognized formats
        top_keys = list(f.keys())
        logger.warning(
            f"Unrecognized HDF5 format: top-level keys={top_keys}. "
            "Expected APS-U (xpcs/twotime/correlation_map) or "
            "APS old (xpcs/dqlist + exchange/C2T_all)."
        )
        return "unknown"


@log_performance(threshold=0.8)
def load_aps_old_format(loader: XPCSDataLoader, hdf_path: str) -> dict[str, Any]:
    """Load data from APS old format HDF5 file.

    Optimization: Uses selective HDF5 reads when quality filtering
    is disabled. Instead of loading all matrices upfront, we:
    1. First determine which indices are needed based on q-selection
    2. Only load those specific matrices from HDF5

    This reduces I/O by up to 98% for typical datasets where only ~23 of
    ~1150 matrices are actually used.
    """
    from xpcsjax.data.xpcs_loader import (
        _check_allocation_budget,
        _check_frame_count,
        _check_square_matrix,
    )

    with h5py.File(
        hdf_path,
        "r",
        rdcc_nbytes=_HDF5_RDCC_NBYTES,
        rdcc_nslots=_HDF5_RDCC_NSLOTS,
        rdcc_w0=_HDF5_RDCC_W0,
    ) as f:
        # Load q and phi lists (small metadata - always needed)
        dqlist = f["xpcs/dqlist"][0, :]  # Shape (1, N) -> (N,)
        dphilist = f["xpcs/dphilist"][0, :]  # Shape (1, N) -> (N,)

        # Load correlation data from exchange/C2T_all
        c2t_group = f["exchange/C2T_all"]
        # APS old format: keys are in HDF5 creation order, which IS the correct
        # positional order matching dqlist/dphilist indices. Do NOT sort — integer
        # keys like "1","2","10" sort lexicographically wrong. The APS-U path uses
        # sorted() because it has zero-padded keys (c2_00001, c2_00002, …).
        c2_keys = list(c2t_group.keys())
        if not c2_keys:
            raise ValueError(
                f"APS old-format HDF5 file contains no correlation matrices "
                f"in 'exchange/C2T_all': {hdf_path}"
            )
        # Pre-filter baseline: all candidate matrices before (q,phi) selection.
        loader._n_prefilter_matrices = len(c2_keys)

        # Check if quality-based filtering is enabled (requires loading all matrices)
        filtering_config = loader.config.get("data_filtering", {})
        quality_filtering_enabled = filtering_config.get("enabled", False) and filtering_config.get(
            "quality_filtering", {}
        ).get("enabled", False)

        # Select optimal q-vector first (doesn't require matrices)
        logger.debug("Selecting optimal q-vector for caching")
        selected_q_idx = loader._select_optimal_wavevector(dqlist)
        selected_q = dqlist[selected_q_idx]

        # Calculate q-vector tolerance as fraction of selected q-vector
        q_tolerance_fraction = loader.config.get("q_tolerance_fraction", 0.1)
        q_tolerance = selected_q * q_tolerance_fraction
        q_matching_indices = np.where(np.abs(dqlist - selected_q) <= q_tolerance)[0]

        # If we still get too few phi angles, expand the search
        if len(q_matching_indices) < 5:
            # Sort by distance from selected q and take closest N entries
            q_distances = np.abs(dqlist - selected_q)
            closest_indices = np.argsort(q_distances)
            # Take up to 10 closest q-vectors to ensure good phi angle coverage
            n_desired = min(10, len(closest_indices))
            q_matching_indices_list = [int(i) for i in closest_indices[:n_desired]]
            q_matching_indices = np.array(q_matching_indices_list, dtype=int)
            logger.debug(
                f"Expanded selection to {len(q_matching_indices)} closest q-vectors for better phi coverage",
            )

        logger.debug(
            f"Selected {len(q_matching_indices)} (q,phi) pairs with q-range: "
            f"{dqlist[q_matching_indices].min():.6f} - {dqlist[q_matching_indices].max():.6f} AA^-1",
        )

        if quality_filtering_enabled:
            # Two-pass optimization: metadata filter first, then load + quality filter
            # Pass 1: phi/q filtering without loading matrices (metadata only)
            logger.debug("Quality filtering enabled - running metadata-only pre-filter")
            metadata_indices = loader._get_selected_indices(
                dqlist,
                dphilist,
                None,  # No matrices needed for phi-only filtering
            )

            # Narrow to candidates via q + phi intersection
            if metadata_indices is not None:
                candidate_indices = np.intersect1d(q_matching_indices, metadata_indices)
            else:
                candidate_indices = q_matching_indices

            logger.debug(
                f"Pre-filter: {len(c2_keys)} total -> {len(candidate_indices)} candidates "
                f"({len(candidate_indices) / len(c2_keys) * 100:.1f}% I/O reduction)"
            )

            # Pass 2: load only candidate matrices from HDF5
            # SEC-2 (parity with APS-U): bound the intermediate accumulation
            # up front so a crafted file with many large candidate bins cannot
            # exhaust RAM before the post-selection guard on the final buffer.
            guard_aps_u_intermediate_allocation(
                c2t_group,
                c2_keys,
                candidate_indices,
                source="HDF5 correlation dataset (APS-old quality-filter intermediate)",
            )
            candidate_matrices = []
            for idx in candidate_indices:
                key = c2_keys[int(idx)]
                c2_half = c2t_group[key][()]
                c2_full = loader._reconstruct_full_matrix(c2_half)
                candidate_matrices.append(c2_full)

            # Apply quality filtering on the loaded subset
            quality_indices = loader._get_selected_indices(
                dqlist[candidate_indices],
                dphilist[candidate_indices],
                candidate_matrices,
            )

            # Map quality filter results back to original indices
            if quality_indices is not None:
                final_indices = candidate_indices[quality_indices]
                selected_c2_matrices = [candidate_matrices[i] for i in quality_indices]
                logger.debug(
                    f"After quality filtering: {len(candidate_indices)} -> {len(final_indices)} matrices",
                )
            else:
                final_indices = candidate_indices
                selected_c2_matrices = candidate_matrices
        else:
            # OPTIMIZATION: No quality filtering - selective HDF5 reads
            # Only load the matrices we actually need (up to 98% I/O reduction)
            logger.debug("Applying phi-only filtering (no quality filtering)")
            selected_indices = loader._get_selected_indices(
                dqlist,
                dphilist,
                None,  # Don't pass matrices - not needed for phi-only filtering
            )

            # Apply additional phi filtering if enabled
            if selected_indices is not None:
                final_indices = np.intersect1d(q_matching_indices, selected_indices)
                logger.debug(
                    f"After phi filtering: {len(q_matching_indices)} -> {len(final_indices)} matrices",
                )
            else:
                final_indices = q_matching_indices
                logger.debug(
                    f"No phi filtering - using all {len(final_indices)} (q,phi) pairs",
                )

            # Selective load: only read the matrices we need.
            # C1 perf: pre-allocate a single C-order output buffer and write
            # each reconstructed matrix directly, eliminating the Python-list
            # accumulation + np.array() re-stack copy (~30-50% peak-RSS saving).
            logger.info(
                f"Selective HDF5 read: loading {len(final_indices)} of {len(c2_keys)} matrices "
                f"({len(final_indices) / len(c2_keys) * 100:.1f}% I/O)"
            )
            n_sel = len(final_indices)
            if n_sel == 0:
                raise ValueError(
                    "Phi/q filtering selected zero (q,phi) pairs to load; check "
                    "the angle/q-range filters in the config against the dataset "
                    "(no matrices match the requested ranges)."
                )
            # Read first matrix to get the time-axis dimension without storing it.
            # Preserve the source dtype (do NOT force float64): _reconstruct_full_matrix
            # did the c2_half + c2_half.T arithmetic in the stored dtype, and parity is
            # bit-exact — upcasting here would change the reconstructed bits.
            _probe_half = c2t_group[c2_keys[int(final_indices[0])]][()]
            _check_square_matrix(_probe_half.shape, source="HDF5 correlation dataset")
            _n_t = _probe_half.shape[0]
            _check_frame_count(int(_n_t), source="HDF5 correlation dataset")
            _check_allocation_budget(
                int(n_sel),
                int(_n_t),
                _probe_half.dtype.itemsize,
                source="HDF5 correlation dataset",
            )
            c2_matrices_array = np.empty((n_sel, _n_t, _n_t), dtype=_probe_half.dtype, order="C")
            # Write the already-read probe matrix into slot 0 (exact same arithmetic
            # as _reconstruct_full_matrix: c2_half + c2_half.T, diagonal /= 2).
            c2_matrices_array[0] = _probe_half + _probe_half.T
            _diag_idx = np.diag_indices(_n_t)
            c2_matrices_array[0][_diag_idx] /= 2
            del _probe_half
            # Load remaining matrices directly into pre-allocated slots.
            for _out_i, idx in enumerate(final_indices[1:], start=1):
                key = c2_keys[int(idx)]
                _c2_half = c2t_group[key][()]
                c2_matrices_array[_out_i] = _c2_half + _c2_half.T
                c2_matrices_array[_out_i][_diag_idx] /= 2

        # Shared zero-selection guard. BOTH the quality-filtering and the
        # phi-only branches resolve ``final_indices`` above. The phi-only
        # branch additionally fast-fails earlier (before its pre-allocated
        # probe read of ``final_indices[0]``), but the quality-filtering
        # branch has no such early guard: an empty candidate set or a quality
        # pass that rejects every row leaves ``final_indices`` empty and
        # ``selected_c2_matrices`` empty, which would otherwise become an
        # ``np.array([])`` and flow downstream as a malformed empty c2 stack.
        # Fail loudly here for every APS-old selection path instead.
        if len(final_indices) == 0:
            raise ValueError(
                "Phi/q/quality filtering selected zero (q,phi) pairs to load; "
                "check the angle/q-range/quality filters in the config against "
                "the dataset (no matrices match the requested ranges)."
            )

        # Extract metadata for final indices
        filtered_dqlist = dqlist[final_indices]
        filtered_dphilist = dphilist[final_indices]
        # c2_matrices_array already built (pre-allocated in no-quality-filter path;
        # stacked from candidate_matrices list in the quality-filter path below)
        if quality_filtering_enabled:
            c2_matrices_array = np.array(selected_c2_matrices)

        # Apply frame slicing to selected q-vector data
        logger.debug(
            f"Applying frame slicing to selected q-vector data: shape {c2_matrices_array.shape}",
        )
        c2_exp = loader._apply_frame_slicing_to_selected_q(c2_matrices_array)

        # Calculate 1D time array (meshgrids generated by NLSQ as needed)
        time_1d = loader._calculate_time_arrays(c2_exp.shape[-1])

        return {
            "wavevector_q_list": filtered_dqlist,  # Selected q-vectors (may be multiple for APS old)
            "phi_angles_list": filtered_dphilist,  # Corresponding phi angles
            "t1": time_1d,  # 1D time array starting from 0: [0, dt, 2*dt, ...]
            "t2": time_1d.copy(),  # Independent copy (prevent aliasing mutation)
            "c2_exp": c2_exp,  # Shape: (n_selected_pairs, sliced_frames, sliced_frames)
        }


@log_performance(threshold=0.8)
def load_aps_u_format(loader: XPCSDataLoader, hdf_path: str) -> dict[str, Any]:
    """Load data from APS-U new format HDF5 file using processed_bins mapping.

    D7 (2026-09-15 review): (q,phi) selection is resolved from metadata
    alone before any correlation matrix is read, mirroring
    :func:`load_aps_old_format`'s two-pass structure. Only the matrices
    needed for the final selection (or, when quality filtering is
    enabled, the metadata-narrowed candidate set) are ever read from
    HDF5 -- the previous version unconditionally read every valid bin
    into RAM before running (q,phi) selection at all.
    """
    from xpcsjax.data.xpcs_loader import (
        XPCSDataFormatError,
        _check_allocation_budget,
        _check_frame_count,
        _check_square_matrix,
    )

    with h5py.File(
        hdf_path,
        "r",
        rdcc_nbytes=_HDF5_RDCC_NBYTES,
        rdcc_nslots=_HDF5_RDCC_NSLOTS,
        rdcc_w0=_HDF5_RDCC_W0,
    ) as f:
        # Load the processed_bins mapping - this tells us which (q,phi) pairs have correlation data
        processed_bins = f["xpcs/twotime/processed_bins"][()]

        # Load the q and phi lists
        q_values = f["xpcs/qmap/dynamic_v_list_dim0"][()]  # All q values
        phi_values = f["xpcs/qmap/dynamic_v_list_dim1"][()]  # All phi values available

        n_q = len(q_values)
        n_phi = len(phi_values)

        logger.debug(f"APS-U format: {n_q} q-values, {n_phi} phi-values")
        logger.debug(f"Q range: {q_values.min():.6f} to {q_values.max():.6f} A^-1")
        logger.debug(f"Phi values: {phi_values}")
        logger.debug(
            f"Processed bins: {len(processed_bins)} correlation matrices available",
        )
        # Pre-filter baseline: all available bins before (q,phi) validity selection.
        loader._n_prefilter_matrices = len(processed_bins)

        # The processed_bins represent which (q,phi) combinations have correlation data
        # We need to map these to actual (q,phi) pairs using the grid structure
        # For APS-U format: bin_idx = processed_bin - 1; q_idx = bin_idx // n_phi; phi_idx = bin_idx % n_phi
        qphi_pairs = []
        valid_bin_indices = []

        for i, processed_bin in enumerate(processed_bins):
            bin_idx = processed_bin - 1  # Convert to 0-based
            q_idx = bin_idx // n_phi
            phi_idx = bin_idx % n_phi

            # Check if indices are valid
            if 0 <= q_idx < n_q and 0 <= phi_idx < n_phi:
                q_val = q_values[q_idx]
                phi_val = phi_values[phi_idx]
                qphi_pairs.append((q_val, phi_val))
                valid_bin_indices.append(
                    i,
                )  # Track which correlation matrix this corresponds to
            else:
                logger.warning(
                    f"Invalid bin mapping: processed_bin={processed_bin}, q_idx={q_idx}, phi_idx={phi_idx}",
                )

        if len(qphi_pairs) == 0:
            raise XPCSDataFormatError(
                "No valid (q,phi) pairs found from processed_bins mapping",
            )

        # Convert to arrays for processing
        qphi_array = np.array(qphi_pairs)
        filtered_dqlist = qphi_array[:, 0]  # q values for valid pairs
        filtered_dphilist = qphi_array[:, 1]  # phi values for valid pairs
        valid_bin_indices_arr = np.array(valid_bin_indices, dtype=int)

        logger.debug(
            f"Extracted {len(valid_bin_indices)} valid (q,phi) pairs from processed_bins",
        )

        corr_group = f["xpcs/twotime/correlation_map"]
        c2_keys = sorted(
            corr_group.keys(),
        )  # Sort alphabetically (which works for c2_00001 format)

        def _read_matrices(qphi_positions: NDArray) -> list[NDArray]:
            """Reconstruct matrices for the given qphi-space positions.

            Bounds the intermediate accumulation up front (SEC-2, parity
            with APS-old): probes the first matrix and rejects an
            oversized/over-budget file before the list is built, so a
            crafted APS-U file cannot exhaust RAM during accumulation.
            bin_idx out of range raises rather than skipping the matrix
            without skipping its (q,phi) pair (A1: that used to silently
            mislabel every later matrix by one slot).
            """
            bin_indices = [int(valid_bin_indices_arr[int(p)]) for p in qphi_positions]
            guard_aps_u_intermediate_allocation(
                corr_group,
                c2_keys,
                bin_indices,
                source="HDF5 correlation dataset (APS-U intermediate)",
            )
            matrices = []
            for bin_idx in bin_indices:
                if bin_idx >= len(c2_keys):
                    raise XPCSDataFormatError(
                        f"Matrix index {bin_idx} exceeds available matrices "
                        f"({len(c2_keys)}) - APS-U file is inconsistent between "
                        "processed_bins mapping and the correlation_map group",
                    )
                c2_half = corr_group[c2_keys[bin_idx]][()]
                matrices.append(loader._reconstruct_full_matrix(c2_half))
            return matrices

        # Select optimal q-vector (closest match) from metadata alone -
        # no correlation matrix needs to be read for this decision.
        selected_q_idx = loader._select_optimal_wavevector(filtered_dqlist)
        selected_q = filtered_dqlist[selected_q_idx]
        logger.debug(
            f"Selected optimal q-vector: {selected_q:.6f} AA^-1 (index {selected_q_idx})",
        )
        q_matching_indices = np.where(np.abs(filtered_dqlist - selected_q) < 1e-10)[0]
        logger.debug(
            f"Found {len(q_matching_indices)} (q,phi) pairs matching selected q-vector",
        )

        # Quality filtering needs real matrix data, so (mirroring
        # load_aps_old_format) it alone requires an intermediate load -
        # narrowed to the q/phi metadata candidates first, never the full
        # valid-bin set. The far more common phi/q-only path never reads
        # a matrix it won't end up using.
        filtering_config = loader.config.get("data_filtering", {})
        quality_filtering_enabled = filtering_config.get("enabled", False) and filtering_config.get(
            "quality_filtering", {}
        ).get("enabled", False)

        selected_c2_matrices: list[NDArray] | None
        if quality_filtering_enabled:
            # Pass 1: phi/q metadata pre-filter (no HDF5 reads). Same
            # data_filtering config as pass 2 below, so a fallback here
            # would otherwise double-record the DATA-1 degradation signal
            # for one logical fallback -- only pass 2 (the final decision)
            # records it.
            metadata_indices = loader._get_selected_indices(
                filtered_dqlist,
                filtered_dphilist,
                None,  # No matrices needed for phi-only filtering
                record_degradation=False,
            )
            candidate_indices = (
                np.intersect1d(q_matching_indices, metadata_indices)
                if metadata_indices is not None
                else q_matching_indices
            )
            logger.debug(
                f"Pre-filter: {len(valid_bin_indices)} total -> "
                f"{len(candidate_indices)} candidates "
                f"({len(candidate_indices) / max(len(valid_bin_indices), 1) * 100:.1f}% I/O)"
            )

            # Pass 2: load only the candidates, then quality-filter them.
            candidate_matrices = _read_matrices(candidate_indices)
            quality_indices = loader._get_selected_indices(
                filtered_dqlist[candidate_indices],
                filtered_dphilist[candidate_indices],
                candidate_matrices,
            )
            if quality_indices is not None:
                final_indices = candidate_indices[quality_indices]
                selected_c2_matrices = [candidate_matrices[i] for i in quality_indices]
                logger.debug(
                    f"After quality filtering: {len(candidate_indices)} -> "
                    f"{len(final_indices)} matrices",
                )
            else:
                final_indices = candidate_indices
                selected_c2_matrices = candidate_matrices
        else:
            # OPTIMIZATION: phi-only filtering - no matrices needed for the
            # decision, so none are read until final_indices is known.
            selected_indices = loader._get_selected_indices(
                filtered_dqlist,
                filtered_dphilist,
                None,  # Don't pass matrices - not needed for phi-only filtering
            )
            if selected_indices is not None:
                final_indices = np.intersect1d(q_matching_indices, selected_indices)
                logger.debug(
                    f"After intersecting with phi filtering: {len(final_indices)} pairs remain",
                )
            else:
                final_indices = q_matching_indices
                logger.debug(
                    f"No phi filtering applied - using all {len(final_indices)} pairs for selected q-vector",
                )
            selected_c2_matrices = None  # deferred: read selectively below

        # Extract data for selected indices. An empty selection must abort,
        # not silently fall back to (q,phi) index 0 (which would fit an
        # unrelated q-vector) — audit C9.
        final_indices = loader._require_nonempty_selection(final_indices, selected_q=selected_q)

        # Use final indices for both (q,phi) pairs and correlation matrices
        final_dqlist = filtered_dqlist[final_indices]
        final_dphilist = filtered_dphilist[final_indices]

        n_sel = len(final_indices)
        logger.debug(f"Final selection: {n_sel} correlation matrices")
        logger.info(
            f"Selective HDF5 read: loading {n_sel} of {len(valid_bin_indices)} "
            f"valid-bin matrices "
            f"({n_sel / max(len(valid_bin_indices), 1) * 100:.1f}% I/O)"
        )

        if selected_c2_matrices is not None:
            # Quality-filtering path: matrices for final_indices were
            # already read above as part of candidate_matrices - stack
            # them directly instead of re-reading from HDF5 (mirrors
            # load_aps_old_format's quality-filtering branch, which
            # likewise just np.array()s its already-loaded matrices
            # rather than using the preallocated-buffer path below).
            c2_matrices_array = np.array(selected_c2_matrices)
        else:
            # C1 perf: pre-allocate a single C-order output buffer and write
            # each selected matrix directly, eliminating the Python-list +
            # np.array() re-stack copy (~30-50% peak-RSS saving at
            # 23M-point scale). Only the matrices in final_indices are
            # ever read from HDF5.
            bin_indices = [int(valid_bin_indices_arr[int(i)]) for i in final_indices]
            guard_aps_u_intermediate_allocation(
                corr_group,
                c2_keys,
                bin_indices,
                source="HDF5 correlation dataset (APS-U intermediate)",
            )
            if bin_indices[0] >= len(c2_keys):
                raise XPCSDataFormatError(
                    f"Matrix index {bin_indices[0]} exceeds available matrices "
                    f"({len(c2_keys)}) - APS-U file is inconsistent between "
                    "processed_bins mapping and the correlation_map group",
                )
            # Read first matrix to get the time-axis dimension without
            # storing it, and preserve the source dtype (do NOT force
            # float64): the c2_half + c2_half.T arithmetic below runs in
            # the stored dtype, and parity is bit-exact -- upcasting here
            # would change the reconstructed bits.
            _first_half = corr_group[c2_keys[bin_indices[0]]][()]
            _check_square_matrix(_first_half.shape, source="HDF5 correlation dataset")
            _n_t_u = _first_half.shape[0]
            _check_frame_count(int(_n_t_u), source="HDF5 correlation dataset")
            _check_allocation_budget(
                len(bin_indices),
                int(_n_t_u),
                _first_half.dtype.itemsize,
                source="HDF5 correlation dataset",
            )
            c2_matrices_array = np.empty(
                (len(bin_indices), _n_t_u, _n_t_u), dtype=_first_half.dtype, order="C"
            )
            # Same arithmetic as _reconstruct_full_matrix: c2_half + c2_half.T,
            # diagonal /= 2.
            c2_matrices_array[0] = _first_half + _first_half.T
            _diag_idx_u = np.diag_indices(_n_t_u)
            c2_matrices_array[0][_diag_idx_u] /= 2
            del _first_half
            for _out_j, bin_idx in enumerate(bin_indices[1:], start=1):
                if bin_idx >= len(c2_keys):
                    raise XPCSDataFormatError(
                        f"Matrix index {bin_idx} exceeds available matrices "
                        f"({len(c2_keys)}) - APS-U file is inconsistent between "
                        "processed_bins mapping and the correlation_map group",
                    )
                _c2_half = corr_group[c2_keys[bin_idx]][()]
                c2_matrices_array[_out_j] = _c2_half + _c2_half.T
                c2_matrices_array[_out_j][_diag_idx_u] /= 2

        # Apply frame slicing to the selected q-vector data
        c2_exp = loader._apply_frame_slicing_to_selected_q(c2_matrices_array)

        # Calculate 1D time array (meshgrids generated by NLSQ as needed)
        time_1d = loader._calculate_time_arrays(c2_exp.shape[-1])

        return {
            "wavevector_q_list": final_dqlist,
            "phi_angles_list": final_dphilist,
            "t1": time_1d,  # 1D time array starting from 0: [0, dt, 2*dt, ...]
            "t2": time_1d.copy(),  # Independent copy (prevent aliasing mutation)
            "c2_exp": c2_exp,
        }
