"""NLSQ result saving functions for xpcsjax XPCS analysis.

This module provides functions for saving NLSQ optimization results to disk,
including JSON parameter files and NPZ data files.
Extracted from cli/commands.py for better modularity.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from xpcsjax.io.json_utils import json_safe, json_serializer
from xpcsjax.utils.logging import get_logger
from xpcsjax.utils.path_validation import get_safe_output_dir

logger = get_logger(__name__)


def save_nlsq_json_files(
    param_dict: dict[str, Any],
    analysis_dict: dict[str, Any],
    convergence_dict: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save 3 JSON files: parameters, analysis results, convergence metrics.

    Parameters
    ----------
    param_dict : dict[str, Any]
        Parameter dictionary with {name: {value, uncertainty}}
    analysis_dict : dict[str, Any]
        Analysis results with method, fit_quality, dataset_info, etc.
    convergence_dict : dict[str, Any]
        Convergence diagnostics with status, iterations, recovery_actions
    output_dir : Path
        Output directory for JSON files

    Returns
    -------
    None
        Files saved to disk

    Notes
    -----
    Creates 3 JSON files:
    - parameters.json: Complete parameter values and uncertainties
    - analysis_results_nlsq.json: Analysis summary and fit quality
    - convergence_metrics.json: Convergence diagnostics and device info
    """
    output_dir = get_safe_output_dir(output_dir)
    param_file = output_dir / "parameters.json"
    analysis_file = output_dir / "analysis_results_nlsq.json"
    convergence_file = output_dir / "convergence_metrics.json"

    # Pre-sanitize all dicts before passing to json.dump.  Python's json encoder
    # handles plain float natively (emitting the invalid JSON tokens NaN/Infinity)
    # and therefore never calls the `default` hook for those values.  Calling
    # json_safe() up-front ensures NaN → null and Inf → "Infinity" regardless of
    # where they appear in nested structures.
    safe_param = json_safe(param_dict)
    safe_analysis = json_safe(analysis_dict)
    safe_convergence = json_safe(convergence_dict)

    try:
        # Save parameters.json
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(safe_param, f, indent=2, default=json_serializer)
        # T056: Log file path and write completion
        logger.debug(f"Saved parameters to {param_file}")

        # Save analysis_results_nlsq.json
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(safe_analysis, f, indent=2, default=json_serializer)
        logger.debug(f"Saved analysis results to {analysis_file}")

        # Save convergence_metrics.json
        with open(convergence_file, "w", encoding="utf-8") as f:
            json.dump(safe_convergence, f, indent=2, default=json_serializer)
        logger.debug(f"Saved convergence metrics to {convergence_file}")

        # T058a: Log file sizes after all writes succeed (inside try to catch stat errors)
        total_size_kb = (
            param_file.stat().st_size
            + analysis_file.stat().st_size
            + convergence_file.stat().st_size
        ) / 1024
        logger.info(f"Saved 3 JSON files to {output_dir} (total: {total_size_kb:.1f} KB)")
    except OSError as e:
        raise OSError(f"Failed to write NLSQ JSON files to {output_dir}: {e}") from e


def save_nlsq_npz_file(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    c2_raw: np.ndarray,
    c2_scaled: np.ndarray,
    c2_solver: np.ndarray | None,
    per_angle_scaling: np.ndarray,
    per_angle_scaling_solver: np.ndarray,
    residuals: np.ndarray,
    residuals_norm: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    q: float,
    output_dir: Path,
) -> None:
    """Save NPZ file with experimental/theoretical data and metadata.

    Parameters
    ----------
    phi_angles : np.ndarray
        Scattering angles (n_angles,)
    c2_exp : np.ndarray
        Experimental correlation data (n_angles, n_t1, n_t2)
    c2_raw : np.ndarray
        Raw theoretical fits before scaling (n_angles, n_t1, n_t2)
    c2_scaled : np.ndarray
        Scaled theoretical fits (n_angles, n_t1, n_t2)
    c2_solver : np.ndarray | None
        Solver-evaluated theoretical fits (optional, n_angles, n_t1, n_t2)
    per_angle_scaling : np.ndarray
        Per-angle scaling parameters (n_angles, 2) [contrast, offset]
    per_angle_scaling_solver : np.ndarray
        Original per-angle scaling parameters from the solver (n_angles, 2)
    residuals : np.ndarray
        Residuals: exp - scaled (n_angles, n_t1, n_t2)
    residuals_norm : np.ndarray
        Normalized residuals (n_angles, n_t1, n_t2)
    t1 : np.ndarray
        Time array 1 (n_t1,)
    t2 : np.ndarray
        Time array 2 (n_t2,)
    q : float
        Wavevector magnitude [1/Å]
    output_dir : Path
        Output directory

    Returns
    -------
    None
        NPZ file saved to disk
    """
    output_dir = get_safe_output_dir(output_dir)
    npz_file = output_dir / "fitted_data.npz"

    # Coerce all array inputs to NumPy — JAX arrays trigger implicit host-device
    # transfer inside np.savez_compressed, which can stall the compute graph.
    # Explicit np.asarray() makes the transfer predictable and allows overlap.
    phi_angles = np.asarray(phi_angles, dtype=np.float64)
    c2_exp = np.asarray(c2_exp, dtype=np.float64)
    c2_raw = np.asarray(c2_raw, dtype=np.float64)
    c2_scaled = np.asarray(c2_scaled, dtype=np.float64)
    per_angle_scaling = np.asarray(per_angle_scaling, dtype=np.float64)
    per_angle_scaling_solver = np.asarray(per_angle_scaling_solver, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    residuals_norm = np.asarray(residuals_norm, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    if c2_solver is not None:
        c2_solver = np.asarray(c2_solver, dtype=np.float64)

    # I/O boundary validation: shapes must be internally consistent.
    n_angles = phi_angles.shape[0]
    n_t1 = t1.shape[0]
    n_t2 = t2.shape[0]
    expected_c2_shape = (n_angles, n_t1, n_t2)
    for name, arr in [
        ("c2_exp", c2_exp),
        ("c2_raw", c2_raw),
        ("c2_scaled", c2_scaled),
        ("residuals", residuals),
        ("residuals_norm", residuals_norm),
    ]:
        if arr.shape != expected_c2_shape:
            raise ValueError(
                f"{name}.shape {arr.shape} does not match expected "
                f"(n_angles={n_angles}, n_t1={n_t1}, n_t2={n_t2})"
            )
    if per_angle_scaling.shape[0] != n_angles:
        raise ValueError(
            f"per_angle_scaling.shape[0]={per_angle_scaling.shape[0]} != n_angles={n_angles}"
        )
    if per_angle_scaling_solver.shape[0] != n_angles:
        raise ValueError(
            f"per_angle_scaling_solver.shape[0]={per_angle_scaling_solver.shape[0]} "
            f"!= n_angles={n_angles}"
        )
    if c2_solver is not None and c2_solver.shape != expected_c2_shape:
        raise ValueError(
            f"c2_solver.shape {c2_solver.shape} does not match expected "
            f"(n_angles={n_angles}, n_t1={n_t1}, n_t2={n_t2})"
        )

    # Build kwargs dict to handle optional c2_solver
    save_dict: dict[str, Any] = {
        # Experimental data (2 arrays)
        "phi_angles": phi_angles,
        "c2_exp": c2_exp,
        # Theoretical fits (4 arrays)
        "c2_theoretical_raw": c2_raw,
        "c2_theoretical_scaled": c2_scaled,
        "per_angle_scaling": per_angle_scaling,
        "per_angle_scaling_solver": per_angle_scaling_solver,
        # Residuals (2 arrays)
        "residuals": residuals,
        "residuals_normalized": residuals_norm,
        # Coordinate arrays (3 arrays)
        "t1": t1,
        "t2": t2,
        "q": np.array([q]),  # Wrap scalar in array
    }

    # Add c2_solver only if it's not None
    if c2_solver is not None:
        save_dict["c2_solver_scaled"] = c2_solver

    try:
        np.savez_compressed(npz_file, **save_dict)
    except OSError as e:
        raise OSError(f"Failed to write NPZ file to {npz_file}: {e}") from e

    # T058a: Log file path and file size after write completion
    # save_dict always has 11 base arrays; c2_solver_scaled is optional (+1).
    n_arrays = 11 + (1 if c2_solver is not None else 0)
    try:
        file_size_mb = npz_file.stat().st_size / (1024 * 1024)
        size_str = f"{file_size_mb:.2f} MB"
    except OSError:
        size_str = "size unknown"
    logger.info(f"Saved NPZ file with {n_arrays} arrays to {npz_file} ({size_str})")
