"""NPZ cache load/save/validate for :class:`xpcsjax.data.xpcs_loader.XPCSDataLoader`.

Extracted from ``xpcs_loader.py`` (2026-09-15 review, finding D6) as a thin
move: these were instance methods reading/writing ``self.config`` /
``self.analyzer_config`` / ``self.exp_config`` and calling a handful of other
loader methods; they are now free functions taking the loader instance
explicitly as their first argument, and the original methods on
``XPCSDataLoader`` delegate to them unchanged. No behaviour change.

Low-level shared guards (``XPCSDataFormatError``, ``_check_square_matrix``,
``_check_frame_count``, ``_check_allocation_budget``, ``_validate_loaded_arrays``)
stay defined in ``xpcs_loader.py`` and are imported here lazily (inside the
functions that need them) to avoid a circular import, since ``xpcs_loader.py``
imports this module at module level to build its delegating methods — the
same in-function-import pattern already used elsewhere in this codebase for
cycle avoidance (see ``optimization/`` per the root CLAUDE.md).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.utils.logging import get_logger, log_performance

if TYPE_CHECKING:
    from xpcsjax.data.xpcs_loader import XPCSDataLoader

logger = get_logger(__name__)

# Regex to detect old str.format()-style placeholders: {var} or {var:.4f}.
# Negative lookbehind excludes ${var}, which is already valid Template syntax.
_OLD_FORMAT_RE = re.compile(r"(?<!\$)\{(\w+)(?::[^}]*)?\}")


def migrate_cache_template(template: str) -> str:
    """Auto-convert old {var} format templates to ${var} syntax.

    Returns the template unchanged if it has no bare {var} placeholders
    (including templates that already use $ syntax throughout).
    Logs a warning on first migration. Handles templates that mix ${var}
    and {var} syntax by migrating only the bare {var} placeholders.
    """
    if _OLD_FORMAT_RE.search(template):
        migrated = _OLD_FORMAT_RE.sub(r"${\1}", template)
        logger.warning(
            "Cache template uses deprecated {var} format; auto-migrated to ${var}. "
            "Update your YAML config: %r -> %r",
            template,
            migrated,
        )
        return migrated
    return template


def hash_filter_config(filter_config: dict[str, Any]) -> str:
    """Stable short fingerprint of a filter-settings dict for cache validation."""
    canonical = json.dumps(filter_config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def peek_npz_array_header(npz: Any, key: str) -> tuple[tuple[int, ...], np.dtype]:
    """Read an ``.npz`` member's shape/dtype from its ``.npy`` header.

    Does not materialize the array.

    ``np.load(..., mmap_mode="r")`` is silently a no-op for zip-backed ``.npz``
    files: ``NpzFile.__getitem__`` always fully decompresses into RAM
    regardless of ``mmap_mode`` (verified on numpy 2.4.6 — the returned object
    is a plain ``ndarray``, never a ``memmap``). An allocation-budget guard
    that reads ``npz[key]`` before checking its shape has therefore already
    caused the OOM it exists to prevent. Reading only the member's header
    bytes here lets the caller guard BEFORE any full-array read.
    """
    from numpy.lib import format as npy_format

    with npz.zip.open(f"{key}.npy") as f:
        version = npy_format.read_magic(f)
        if version == (1, 0):
            shape, _fortran_order, dtype = npy_format.read_array_header_1_0(f)
        else:
            shape, _fortran_order, dtype = npy_format.read_array_header_2_0(f)
    return shape, dtype


def source_hdf_stat(loader: XPCSDataLoader) -> tuple[str, int, int] | None:
    """Return the configured source HDF5 file's identity for cache keying.

    Returns ``(basename, size, mtime_ns)``, or ``None`` if it doesn't
    resolve to an existing HDF5 source file (e.g. a direct NPZ override
    with no underlying HDF5 -- including reusing an xpcsjax-written cache
    as the data file, whose own ``source_file`` metadata refers to a
    different, now-irrelevant HDF5 name).
    """
    exp_config = getattr(loader, "exp_config", {})
    data_folder = exp_config.get("data_folder_path", "./")
    data_file = exp_config.get("data_file_name", "")
    if not data_file or data_file.endswith(".npz"):
        return None
    hdf_path = os.path.join(data_folder, data_file)
    try:
        st = os.stat(hdf_path)
    except OSError:
        return None
    return os.path.basename(hdf_path), st.st_size, st.st_mtime_ns


@log_performance(threshold=0.2)
def load_from_cache(loader: XPCSDataLoader, cache_path: str) -> dict[str, Any]:
    """Load data from NPZ cache file with q-vector validation.

    Returns 1D time arrays for NLSQ (meshgrids generated on demand).
    Only supports new 1D array cache format. Old 2D caches must be regenerated.

    Cache files live in config-controlled paths, so this loader treats them
    as untrusted input: ``allow_pickle=False`` blocks object deserialization,
    metadata is read from a JSON-encoded scalar (``cache_metadata_json``),
    and legacy object-array ``cache_metadata`` is refused.
    """
    from xpcsjax.data.xpcs_loader import (
        _check_allocation_budget,
        _check_frame_count,
        _check_square_matrix,
        _validate_loaded_arrays,
    )

    with np.load(cache_path, allow_pickle=False, mmap_mode="r") as data:
        if "cache_metadata_json" in data:
            metadata_text = str(np.asarray(data["cache_metadata_json"]).item())
            try:
                metadata = json.loads(metadata_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cache {cache_path} has malformed cache_metadata_json (not valid JSON): {exc}"
                ) from exc
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Cache {cache_path}: cache_metadata_json must encode a "
                    f"JSON object, got {type(metadata).__name__}"
                )
            validate_cache_q_vector(loader, metadata)
            logger.debug(f"Cache metadata validation passed: {metadata}")
        elif "cache_metadata" in data.files:
            # Legacy object-serialized metadata is a trust-boundary problem
            # (arbitrary code via deserialization from a config-controlled
            # path). Refuse it; the user must regenerate the cache.
            raise ValueError(
                f"Cache {cache_path} uses the legacy 'cache_metadata' "
                "object-array format, which xpcsjax refuses to deserialize "
                "for safety. Delete the cache file and regenerate; the new "
                "format stores metadata as JSON under "
                "'cache_metadata_json'."
            )
        # No metadata key at all: this is a plain source NPZ (e.g. a
        # user-provided data file via ``data_file_name``), NOT a q-selective
        # cache. Its q lives in ``wavevector_q_list`` inside the file, so there
        # is no cross-q reuse to guard against — load it directly. Only
        # SELECTIVE caches (always carrying ``cache_metadata_json``) need the
        # q-vector validation above.

        # Extract correlation data — np.array() copies from mmap before
        # the context manager closes the file (prevents dangling mmap views).
        # allow_pickle=False causes object-dtype arrays to raise here; we
        # surface that as a clearer error rather than letting numpy's
        # internal message leak.
        try:
            # threat-03: validate the cached correlation shape BEFORE copying
            # it into RAM. mmap_mode="r" is a no-op for zip-backed .npz
            # (see peek_npz_array_header) — reading data["c2_exp"] first to
            # inspect its shape would materialize the full array before the
            # guard runs, which is exactly the OOM this exists to prevent.
            # Peek the .npy header instead (no allocation), and read the
            # array itself exactly once below. (Inside the try so an
            # object-dtype member still surfaces the friendly allow_pickle
            # message below.)
            c2_shape, c2_dtype = peek_npz_array_header(data, "c2_exp")
            if len(c2_shape) == 3:
                _check_square_matrix(
                    (int(c2_shape[-2]), int(c2_shape[-1])),
                    source=cache_path,
                )
                _check_frame_count(int(c2_shape[-1]), source=cache_path)
                _check_allocation_budget(
                    int(c2_shape[0]),
                    int(c2_shape[-1]),
                    c2_dtype.itemsize,
                    source=cache_path,
                )

            c2_exp = np.array(data["c2_exp"])
            cached_t1 = np.array(data["t1"])
            cached_t2 = np.array(data["t2"])
            wavevector_q_list = np.array(data["wavevector_q_list"])
            phi_angles_list = np.array(data["phi_angles_list"])
        except ValueError as exc:
            raise ValueError(
                f"Cache {cache_path} contains an object-dtype array under "
                "a data key, which is not allowed (allow_pickle=False). "
                "Delete the cache file and regenerate."
            ) from exc

        # Reject old 2D meshgrid cache format
        if cached_t1.ndim == 2 or cached_t2.ndim == 2:
            raise ValueError(
                f"Old 2D meshgrid cache format detected in {cache_path}. "
                "Please delete the cache file and regenerate with current code. "
                "New cache format uses 1D time arrays."
            )

        # t1/t2 are fully derived from dt + frame count ([0, dt, 2*dt, ...],
        # see _calculate_time_arrays) — never independently meaningful data.
        # Recompute from the CURRENT config's dt instead of trusting whatever
        # is stored in the cache: a cache built by an older xpcsjax version,
        # or by an external conversion script (e.g. a one-off HDF5->NPZ
        # converter), can encode a different time-origin convention (e.g.
        # 1-indexed frames, t[0] = dt instead of 0). ``validate_cache_q_vector``'s
        # dt fingerprint only fires when the cache carries a ``dt`` metadata
        # key, so a foreign/legacy cache without one would otherwise pass
        # silently and desync frame-0 alignment (t1==0/t2==0 boundary) between
        # this run's plots/fit and every other cache-free or in-repo run. This
        # guarantees ALL analysis modes and both cached and freshly-loaded data
        # share one time-axis convention.
        t1 = loader._calculate_time_arrays(c2_exp.shape[-1])
        t2 = t1
        if cached_t1.shape == t1.shape and not np.allclose(cached_t1, t1, atol=1e-9):
            logger.warning(
                f"Cache {cache_path}: stored t1/t2 time axis does not match "
                f"the current dt={loader.analyzer_config.get('dt')}s convention "
                "(expected [0, dt, 2*dt, ...]). Ignoring the cached axis and "
                "recomputing it — delete and regenerate this cache to silence "
                "this warning.",
            )

        result = {
            "wavevector_q_list": wavevector_q_list,
            "phi_angles_list": phi_angles_list,
            "t1": t1,  # 1D array: [0, dt, 2*dt, ...] — always recomputed, never cached
            "t2": t2,  # 1D array: [0, dt, 2*dt, ...] — always recomputed, never cached
            "c2_exp": c2_exp,
        }
        _validate_loaded_arrays(result, source=cache_path)
        return result


@log_performance(threshold=0.3)
def save_to_cache(loader: XPCSDataLoader, data: dict[str, Any], cache_path: str) -> None:
    """Save processed data to NPZ cache file with q-vector metadata.

    Also records the source HDF5 file's basename/size/mtime, so a later
    load can detect that the underlying file was replaced (A2: the cache
    was previously keyed only on frame window + q, so pointing
    ``data_file`` at a different file in the same folder with the same
    frame window would silently serve the stale cache).
    """
    # Ensure cache directory exists
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Convert JAX arrays back to numpy for caching
    cache_data: dict[str, Any] = {}
    for key, value in data.items():
        if hasattr(value, "device"):  # JAX array
            cache_data[key] = np.array(value)
        else:
            cache_data[key] = value

    # Add cache metadata for q-vector validation
    scattering_config = loader.analyzer_config.get("scattering", {})
    config_q = scattering_config.get("wavevector_q", 0.0054)

    # Calculate actual q-vector stats from cached data
    q_values = cache_data["wavevector_q_list"]
    # Use nan-safe variants: q_values from HDF5 may contain NaN for bad pixels.
    actual_q = float(np.nanmean(q_values)) if len(q_values) > 0 else config_q
    q_variance = float(np.nanstd(q_values)) if len(q_values) > 1 else 0.0

    cache_metadata = {
        "config_wavevector_q": float(config_q),
        "actual_wavevector_q": actual_q,
        "q_variance": q_variance,
        "q_count": len(q_values),
        # dt drives the cached t1/t2 time axes (_calculate_time_arrays); a
        # config edit to dt with an otherwise-matching q/frame window must
        # not silently reuse a cache built for the old time axis.
        "dt": float(loader.analyzer_config.get("dt", 1.0)),
        # Report the actually-applied (clamped) frame window recorded by
        # _apply_frame_slicing_to_selected_q. Fall back to the raw config /
        # sliced width only if slicing was never run on this instance.
        "start_frame": getattr(
            loader, "_applied_start_frame", loader.analyzer_config.get("start_frame", 1)
        ),
        "end_frame": getattr(
            loader,
            "_applied_end_frame",
            cache_data["c2_exp"].shape[-1] + loader.analyzer_config.get("start_frame", 1) - 1,
        ),
        "phi_count": len(cache_data["phi_angles_list"]),
        "cache_version": "2.0",
        "selective_q_caching": True,
        # q_tolerance_fraction drives the q-band width used to select which
        # (q, phi) rows get cached (_load_aps_old_format); like dt, it isn't
        # covered by filter_config_hash, so a tolerance-only config edit with
        # an otherwise-matching q/frame window must not silently reuse a
        # cache built under the old (wider/narrower) band.
        "q_tolerance_fraction": float(loader.config.get("q_tolerance_fraction", 0.1)),
        # Fingerprint of the filter settings (phi range, quality/data
        # filtering) that shaped the cached (q, phi) selection, so a config
        # change with the same start/end frame + q is detected instead of
        # silently reusing a stale cache.
        "filter_config_hash": hash_filter_config(loader.config.get("data_filtering", {})),
        # The LEGACY phi filter (_integrate_with_phi_filtering -> PhiAngleFilter)
        # narrows the cached (q, phi) selection further, but reads a DIFFERENT
        # config subtree than XPCSDataFilter, so filter_config_hash above does
        # not cover it. It runs whenever data_filtering is enabled without a
        # `phi_range` block (that key is what short-circuits the legacy branch),
        # so a target_ranges edit with an otherwise-matching q/frame window must
        # not silently reuse a cache built for the old angle set.
        "angle_filtering_hash": hash_filter_config(
            loader.config.get("optimization_config", {}).get("angle_filtering", {})
        ),
    }

    # A2: key the cache to its source HDF5 file so replacing/re-reducing the
    # source, or pointing data_file at a same-named/sibling file with a
    # matching frame window, is detected instead of silently reused.
    stat = source_hdf_stat(loader)
    if stat is not None:
        cache_metadata["source_file"] = stat[0]
        cache_metadata["source_size"] = stat[1]
        cache_metadata["source_mtime_ns"] = stat[2]

    # Metadata is stored as a JSON-encoded scalar (not a Python dict via
    # object pickling) so the loader can read it with allow_pickle=False.
    cache_data["cache_metadata_json"] = np.asarray(json.dumps(cache_metadata))

    # Save with compression if specified. Write to a uniquely-named temp
    # file in the same directory, then atomically rename into place, so
    # concurrent loaders (e.g. parallel pytest-xdist workers, concurrent
    # fit processes sharing a cache dir) never observe a partially-written
    # NPZ at cache_path.
    # Suffix must stay ".npz" — np.savez appends it to any filename that
    # doesn't already end in ".npz", which would break the later rename.
    tmp_path = f"{cache_path}.tmp{os.getpid()}_{uuid.uuid4().hex[:8]}.npz"
    try:
        if loader.exp_config.get("cache_compression", True):
            np.savez_compressed(tmp_path, **cache_data)
        else:
            np.savez(tmp_path, **cache_data)
        os.replace(tmp_path, cache_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise

    # Log cache statistics
    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    logger.info(f"Cache saved: {Path(cache_path).name}")
    logger.debug(f"Cache full path: {cache_path}")
    logger.info(
        f"Cache size: {file_size_mb:.2f} MB, Q-vectors: {cache_metadata['q_count']}, Phi angles: {cache_metadata['phi_count']}",
    )
    logger.debug(f"Q-vector: {actual_q:.6f} +/- {q_variance:.6f} A^-1")


def validate_cache_q_vector(loader: XPCSDataLoader, cache_metadata: dict[str, Any]) -> None:
    """Validate that cached q-vector is compatible with current configuration."""
    from xpcsjax.data.xpcs_loader import CacheStaleError, XPCSDataFormatError

    # Source identity FIRST: a replaced / re-reduced HDF5 is a cache MISS
    # (CacheStaleError -> the loader re-reads the HDF5 and rewrites the
    # cache). It must short-circuit ahead of the config-mismatch checks
    # below, which raise hard XPCSDataFormatError -- a re-reduced dataset
    # usually arrives with a re-configured q/phi filter too, and that case
    # should reload, not tell the user to delete the cache by hand.
    # Check the cache's source-file fingerprint (A2). Older keys alone
    # (frame window + q) can't distinguish the HDF5 being replaced, or
    # data_file pointing at a same-frame-window sibling file, from an
    # untouched source - warn-only when the key predates this check,
    # matching the other legacy-metadata fallbacks below.
    stat = source_hdf_stat(loader)
    cached_source_file = cache_metadata.get("source_file")
    if cached_source_file is None:
        logger.warning(
            "Cache metadata predates source-file fingerprinting; cannot "
            "verify the cache matches the current source HDF5 file.",
        )
    elif stat is not None:
        current_name, current_size, current_mtime_ns = stat
        # Name/size changed: a genuine replacement/re-reduction of the
        # source. This is a cache MISS, not a hard failure -- the caller
        # (load_experimental_data) catches CacheStaleError and reloads from
        # the HDF5, overwriting the stale cache.
        if cached_source_file != current_name or cache_metadata.get("source_size") != current_size:
            raise CacheStaleError(
                f"Cache source-file mismatch: cache was built from "
                f"'{cached_source_file}' but the configured source is now "
                f"'{current_name}' (or its size changed). The source HDF5 "
                f"file was replaced or re-reduced.",
            )
        # mtime alone changed: a content-preserving touch (cp without -p,
        # rsync, backup restore, network-FS mtime shift) does not invalidate
        # the cache -- warn and keep serving it.
        if cache_metadata.get("source_mtime_ns") != current_mtime_ns:
            logger.warning(
                "Cache source HDF5 mtime changed but name and size still "
                "match; serving the cache as-is (delete and regenerate if "
                "the content actually changed).",
            )

    scattering_config = loader.analyzer_config.get("scattering", {})
    current_config_q = scattering_config.get("wavevector_q", 0.0054)
    cached_config_q = cache_metadata.get("config_wavevector_q", current_config_q)

    # Check if configuration q-vectors match (within floating point precision).
    # The cache is q-keyed (selective_q_caching stores q-selected c2_exp /
    # wavevector_q_list), so reusing it for a different configured q would
    # return another q's correlation data. Refuse it — mirrors the existing
    # legacy-cache refusal; the user must regenerate or point at a q-specific
    # cache (audit C1).
    if abs(current_config_q - cached_config_q) > 1e-8:
        raise XPCSDataFormatError(
            f"Cache q-vector mismatch: configured wavevector_q="
            f"{current_config_q:.6f} AA^-1 but cache was built for "
            f"{cached_config_q:.6f} AA^-1. The cache is q-specific; delete it "
            f"and regenerate, or use a q-keyed cache_filename_template "
            f"(e.g. include ${{wavevector_q}}).",
        )

    # Check if the phi/quality/data-filtering settings that shaped the
    # cached (q, phi) selection still match. start/end frame + q alone
    # aren't enough to key the cache: a phi_range or quality_filtering
    # change with the same frames/q would otherwise silently reuse a
    # cache built under the old filter settings.
    current_filter_hash = hash_filter_config(loader.config.get("data_filtering", {}))
    cached_filter_hash = cache_metadata.get("filter_config_hash")
    if cached_filter_hash is not None and cached_filter_hash != current_filter_hash:
        raise XPCSDataFormatError(
            "Cache filter-config mismatch: the phi_range/data_filtering/"
            "quality_filtering settings have changed since this cache was "
            "built. The cache is filter-specific; delete it and regenerate."
        )
    if cached_filter_hash is None:
        logger.warning(
            "Cache metadata predates filter-config fingerprinting; cannot "
            "verify phi/quality-filtering settings match the current config.",
        )

    # Check the LEGACY phi-filter settings (optimization_config.angle_filtering).
    # These live outside the data_filtering subtree fingerprinted above but
    # still shape the cached (q, phi) selection via _integrate_with_phi_filtering,
    # so they need their own key -- same shape as the dt / q_tolerance_fraction
    # checks below.
    current_angle_hash = hash_filter_config(
        loader.config.get("optimization_config", {}).get("angle_filtering", {})
    )
    cached_angle_hash = cache_metadata.get("angle_filtering_hash")
    if cached_angle_hash is None:
        logger.warning(
            "Cache metadata predates angle-filtering fingerprinting; cannot "
            "verify the legacy optimization_config.angle_filtering settings "
            "match the current config.",
        )
    elif cached_angle_hash != current_angle_hash:
        raise XPCSDataFormatError(
            "Cache angle-filtering mismatch: the "
            "optimization_config.angle_filtering settings (enabled / "
            "target_ranges / fallback_to_all_angles) have changed since this "
            "cache was built. The cache's (q, phi) selection is "
            "angle-filter-specific; delete it and regenerate."
        )

    # Check if the time step (t1/t2 axis generator) matches. dt is not
    # folded into filter_config_hash, so a dt-only edit with an otherwise
    # matching q/frame window would silently reuse the old time axis.
    current_dt = float(loader.analyzer_config.get("dt", 1.0))
    cached_dt = cache_metadata.get("dt")
    if cached_dt is None:
        logger.warning(
            "Cache metadata predates dt fingerprinting; cannot verify the "
            "cached time axis matches the current dt.",
        )
    elif abs(current_dt - float(cached_dt)) > 1e-12:
        raise XPCSDataFormatError(
            f"Cache dt mismatch: configured dt={current_dt:.6g}s but cache "
            f"was built for dt={float(cached_dt):.6g}s. The cache's t1/t2 "
            f"time axes are dt-specific; delete it and regenerate.",
        )

    # Check if the q-band tolerance used to select cached (q, phi) rows
    # matches. Like dt, this is not folded into filter_config_hash, so a
    # tolerance-only edit with an otherwise matching q/frame window would
    # otherwise silently reuse a cache built for a different (q, phi) band.
    current_q_tolerance = float(loader.config.get("q_tolerance_fraction", 0.1))
    cached_q_tolerance = cache_metadata.get("q_tolerance_fraction")
    if cached_q_tolerance is None:
        logger.warning(
            "Cache metadata predates q_tolerance_fraction fingerprinting; "
            "cannot verify the cached (q, phi) selection matches the "
            "current tolerance.",
        )
    elif abs(current_q_tolerance - float(cached_q_tolerance)) > 1e-12:
        raise XPCSDataFormatError(
            f"Cache q_tolerance_fraction mismatch: configured "
            f"q_tolerance_fraction={current_q_tolerance:.6g} but cache was "
            f"built for {float(cached_q_tolerance):.6g}. The cache's (q, phi) "
            f"selection is tolerance-specific; delete it and regenerate."
        )

    # Check if cache uses selective q-caching
    is_selective = cache_metadata.get("selective_q_caching", False)
    if not is_selective:
        logger.warning(
            "Loading legacy cache without selective q-vector optimization",
        )
    else:
        actual_q = cache_metadata.get("actual_wavevector_q", cached_config_q)
        q_variance = cache_metadata.get("q_variance", 0.0)
        logger.debug(
            f"Validated selective cache: q={actual_q:.6f} +/- {q_variance:.6f} AA^-1",
        )
