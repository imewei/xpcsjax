"""NLSQ Optimization Strategies Subpackage.

This subpackage contains strategy implementations for NLSQ optimization:
- chunking.py: Angle-stratified chunking for large datasets
- residual.py: Stratified residual function for per-angle optimization
- residual_jit.py: JIT-compiled version of stratified residual
- sequential.py: Sequential per-angle optimization
- executors.py: Strategy pattern executors for optimization algorithms
"""

from xpcsjax.optimization.nlsq.strategies.chunking import (
    StratificationDiagnostics,
    analyze_angle_distribution,
    compute_stratification_diagnostics,
    create_angle_stratified_data,
    create_angle_stratified_indices,
    estimate_stratification_memory,
    format_diagnostics_report,
    should_use_stratification,
)
from xpcsjax.optimization.nlsq.strategies.executors import (
    ExecutionResult,
    LargeDatasetExecutor,
    OptimizationExecutor,
    StandardExecutor,
    StreamingExecutor,
    get_executor,
)
from xpcsjax.optimization.nlsq.strategies.residual import (
    StratifiedResidualFunction,
    create_stratified_residual_function,
)
from xpcsjax.optimization.nlsq.strategies.residual_jit import (
    StratifiedResidualFunctionJIT,
)
from xpcsjax.optimization.nlsq.strategies.sequential import (
    JAC_SAMPLE_SIZE,
    optimize_per_angle_sequential,
)

__all__ = [
    # Chunking
    "StratificationDiagnostics",
    "analyze_angle_distribution",
    "compute_stratification_diagnostics",
    "create_angle_stratified_data",
    "create_angle_stratified_indices",
    "estimate_stratification_memory",
    "format_diagnostics_report",
    "should_use_stratification",
    # Residual
    "StratifiedResidualFunction",
    "StratifiedResidualFunctionJIT",
    "create_stratified_residual_function",
    # Sequential
    "JAC_SAMPLE_SIZE",
    "optimize_per_angle_sequential",
    # Executors (Strategy pattern)
    "ExecutionResult",
    "OptimizationExecutor",
    "StandardExecutor",
    "LargeDatasetExecutor",
    "StreamingExecutor",
    "get_executor",
]
