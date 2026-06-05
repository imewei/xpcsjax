"""JAX-First Optimization for xpcsjax.

JAX-native NLSQ-only optimization system for robust parameter estimation
in XPCS analysis.

This module implements the streamlined optimization philosophy:
1. NLSQ as the sole method (fast, reliable parameter estimation)
2. Unified physics model: c2_fitted = c2_theory * contrast + offset

Key Features:
- NLSQ trust-region optimization (Levenberg-Marquardt) as foundation
- CPU-primary architecture (GPU optional via JAX)
- Dataset size-aware optimization strategies

Note: xpcsjax v0.1 ships JAX-native NLSQ only.
"""

from __future__ import annotations

import logging
from typing import Any

# Import submodules as attributes for hasattr() checks
# These imports expose the submodule packages even if their contents fail to import
from xpcsjax.optimization import nlsq

_logger = logging.getLogger(__name__)

# Handle NLSQ imports with intelligent fallback
try:
    from xpcsjax.optimization.nlsq import (  # Chunking; Residual; Sequential
        MultiStartConfig,
        MultiStartResult,
        NLSQResult,
        NLSQWrapper,
        OptimizationResult,
        StratificationDiagnostics,
        StratifiedResidualFunction,
        StratifiedResidualFunctionJIT,
        create_angle_stratified_data,
        create_angle_stratified_indices,
        create_stratified_residual_function,
        fit_nlsq,
        fit_nlsq_jax,
        fit_nlsq_multistart,
        optimize_per_angle_sequential,
        should_use_stratification,
    )

    NLSQ_AVAILABLE = True
except ImportError as e:
    _logger.warning("Could not import NLSQ optimization: %s", e)
    fit_nlsq = None  # type: ignore[assignment]
    fit_nlsq_jax = None  # type: ignore[assignment]
    fit_nlsq_multistart = None  # type: ignore[assignment]
    MultiStartConfig = None  # type: ignore[assignment,misc]
    MultiStartResult = None  # type: ignore[assignment,misc]
    NLSQResult = None  # type: ignore[assignment,misc]
    NLSQWrapper = None  # type: ignore[assignment,misc]
    OptimizationResult = None  # type: ignore[assignment,misc]
    StratificationDiagnostics = None  # type: ignore[assignment,misc]
    create_angle_stratified_data = None  # type: ignore[assignment]
    create_angle_stratified_indices = None  # type: ignore[assignment]
    should_use_stratification = None  # type: ignore[assignment]
    StratifiedResidualFunction = None  # type: ignore[assignment,misc]
    StratifiedResidualFunctionJIT = None  # type: ignore[assignment,misc]
    create_stratified_residual_function = None  # type: ignore[assignment]
    optimize_per_angle_sequential = None  # type: ignore[assignment]
    NLSQ_AVAILABLE = False

# Module status
OPTIMIZATION_STATUS = {
    "nlsq_available": NLSQ_AVAILABLE,
}

# Primary API functions
__all__ = [
    # Primary optimization methods
    "fit_nlsq",  # Single-entry NLSQ wrapper (v0.1)
    "fit_nlsq_jax",  # NLSQ trust-region (PRIMARY)
    "fit_nlsq_multistart",  # Multi-start NLSQ (v2.6.0)
    # Result classes
    "NLSQResult",
    "MultiStartConfig",
    "MultiStartResult",
    # NLSQ components
    "NLSQWrapper",
    "OptimizationResult",
    "StratificationDiagnostics",
    "create_angle_stratified_data",
    "create_angle_stratified_indices",
    "should_use_stratification",
    "StratifiedResidualFunction",
    "StratifiedResidualFunctionJIT",
    "create_stratified_residual_function",
    "optimize_per_angle_sequential",
    # Status information
    "OPTIMIZATION_STATUS",
    "NLSQ_AVAILABLE",
    # Submodules
    "nlsq",
]


def get_optimization_info() -> dict[str, Any]:
    """Get information about available optimization methods.

    Returns
    -------
    dict
        Dictionary with availability status and recommendations
    """
    info: dict[str, Any] = {
        "status": OPTIMIZATION_STATUS.copy(),
        "primary_method": "nlsq" if NLSQ_AVAILABLE else None,
        "recommendations": [],
    }

    if NLSQ_AVAILABLE:
        info["recommendations"].append(
            "Use fit_nlsq() for fast, reliable parameter estimation",
        )

    if not NLSQ_AVAILABLE:
        info["recommendations"].append(
            "Install NLSQ for optimization capabilities",
        )

    return info
