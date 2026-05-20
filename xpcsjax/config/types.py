"""Type Definitions for Homodyne Configuration System
==================================================

TypedDict definitions for configuration structures and parameter management.
Provides type safety and IDE autocomplete for configuration dictionaries.
"""

from typing import Any, Literal, TypedDict

from xpcsjax.config.parameter_registry import AnalysisMode

__all__ = ["AnalysisMode"]  # re-export for back-compat


class BoundDict(TypedDict, total=False):
    """Parameter bound specification.

    Attributes
    ----------
    name : str
        Parameter name
    min : float
        Minimum value
    max : float
        Maximum value
    type : str
        Bound type (e.g., "Normal", "LogNormal")
    """

    name: str
    min: float
    max: float
    type: str


class InitialParametersConfig(TypedDict, total=False):
    """Initial parameters section of configuration.

    Attributes
    ----------
    parameter_names : list[str]
        List of parameter names to optimize
    values : list[float]
        Initial values for parameters
    active_parameters : list[str], optional
        Subset of parameters to actively optimize
    fixed_parameters : dict[str, float], optional
        Parameters held fixed during optimization
    """

    parameter_names: list[str]
    values: list[float]
    active_parameters: list[str]
    fixed_parameters: dict[str, float]


class ParameterSpaceConfig(TypedDict, total=False):
    """Parameter space section of configuration.

    Attributes
    ----------
    bounds : list[BoundDict]
        Parameter bounds specifications
    priors : list[dict], optional
        Prior distributions for Bayesian methods
    """

    bounds: list[BoundDict]
    priors: list[dict[str, Any]]


class ExperimentalDataConfig(TypedDict, total=False):
    """Experimental data section of configuration.

    Attributes
    ----------
    file_path : str
        Path to HDF5 data file
    data_folder_path : str, optional
        Legacy: folder containing data
    data_file_name : str, optional
        Legacy: data file name
    phi_angles_file : str, optional
        Path to phi angles file
    """

    file_path: str
    data_folder_path: str
    data_file_name: str
    phi_angles_file: str


class StreamingConfig(TypedDict, total=False):
    """Streaming optimization configuration.

    Configuration for NLSQ AdaptiveHybridStreamingOptimizer with checkpoint
    management and fault tolerance for unlimited dataset sizes.

    Attributes
    ----------
    enable_checkpoints : bool
        Enable checkpoint save/resume functionality
    checkpoint_dir : str
        Directory for checkpoint files
    checkpoint_frequency : int
        Save checkpoint every N batches
    resume_from_checkpoint : bool
        Auto-detect and resume from latest checkpoint
    keep_last_checkpoints : int
        Number of recent checkpoints to keep (older ones deleted)
    enable_fault_tolerance : bool
        Enable numerical validation and error recovery
    max_retries_per_batch : int
        Maximum retry attempts per failed batch
    min_success_rate : float
        Minimum batch success rate (0.0-1.0) before failing optimization
    """

    enable_checkpoints: bool
    checkpoint_dir: str
    checkpoint_frequency: int
    resume_from_checkpoint: bool
    keep_last_checkpoints: int
    enable_fault_tolerance: bool
    max_retries_per_batch: int
    min_success_rate: float


class StratificationConfig(TypedDict, total=False):
    """Angle-stratified chunking configuration (v2.2+).

    Configuration for angle-stratified data reorganization to fix per-angle
    parameter incompatibility with NLSQ chunking on large datasets.

    Root Cause Fixed:
    -----------------
    NLSQ's arbitrary chunking can create chunks without certain phi angles,
    resulting in zero gradients for per-angle parameters (contrast[i], offset[i])
    and silent optimization failures (0 iterations, unchanged parameters).

    Solution:
    ---------
    Reorganize data BEFORE optimization to ensure every chunk contains all phi
    angles, making gradients always well-defined.

    Attributes
    ----------
    enabled : bool | str
        Enable stratification: true (force on), false (force off), "auto" (default)
        "auto" applies stratification when: per_angle_scaling=True AND n_points>=100k
    target_chunk_size : int
        Target size for stratified chunks (default: 100_000)
        Should match NLSQ's internal chunk size for optimal results
    max_imbalance_ratio : float
        Maximum angle imbalance ratio before falling back to sequential optimization
        Default: 5.0 (use sequential if max_count/min_count > 5.0)
    force_sequential_fallback : bool
        Force sequential per-angle optimization instead of stratification
        Useful for highly imbalanced datasets (default: false)
    check_memory_safety : bool
        Check available memory before stratification (default: true)
        Warns if peak memory > 70% of available
    use_index_based : bool
        Use index-based (zero-copy) stratification for very large datasets
        Reduces memory overhead from 2x to ~1% (default: false)
    collect_diagnostics : bool
        Collect detailed stratification diagnostics (performance, chunk balance, angle coverage)
        Minimal overhead ~0.01s (default: false)
    log_diagnostics : bool
        Log diagnostic report to console (requires collect_diagnostics=true)
        Useful for troubleshooting and validation (default: false)

    Examples
    --------
    Default (automatic):
        stratification:
          enabled: "auto"  # Auto-activates for large datasets with per-angle scaling

    Force enabled for all datasets:
        stratification:
          enabled: true
          target_chunk_size: 100000

    Disable (use original data):
        stratification:
          enabled: false

    Highly imbalanced angles:
        stratification:
          max_imbalance_ratio: 10.0  # More lenient
          force_sequential_fallback: false

    Memory-constrained systems:
        stratification:
          check_memory_safety: true
          use_index_based: true  # Minimal memory overhead

    References
    ----------
    Ultra-Think Analysis: ultra-think-20251106-012247
    Performance: <1% overhead (0.15s for 3M points)
    Memory: 2x peak (temporary) or ~1% (index-based)
    """

    enabled: bool | str  # true, false, or "auto"
    target_chunk_size: int
    max_imbalance_ratio: float
    force_sequential_fallback: bool
    check_memory_safety: bool
    use_index_based: bool
    collect_diagnostics: bool
    log_diagnostics: bool


class SequentialConfig(TypedDict, total=False):
    """Sequential per-angle optimization configuration (v2.2+).

    Configuration for sequential optimization fallback when stratification
    cannot be applied (e.g., extreme angle imbalance >5.0 ratio).

    Strategy:
    ---------
    1. Split data by phi angle
    2. Optimize each angle independently via nlsq.CurveFit (JAX-native trust-region)
    3. Combine results using weighted averaging (inverse variance weighting)

    Use Cases:
    ----------
    - Extreme angle imbalance (ratio > 5.0)
    - Stratification explicitly disabled
    - Memory-constrained environments
    - Debugging and validation

    Attributes
    ----------
    min_success_rate : float
        Minimum fraction of angles that must converge (default: 0.5)
        Optimization fails if success_rate < min_success_rate
        Range: 0.0-1.0
    weighting : str
        Method for combining per-angle results (default: "inverse_variance")
        Options:

        - "inverse_variance": Optimal statistical weighting (w_i = 1/σ²_i)
        - "uniform": Equal weights for all angles
        - "n_points": Weight by number of data points per angle

    Examples
    --------
    Default configuration (optimal statistical combination):
        sequential:
          min_success_rate: 0.5
          weighting: "inverse_variance"

    Require higher convergence rate:
        sequential:
          min_success_rate: 0.8  # 80% of angles must converge

    Equal weighting (not recommended):
        sequential:
          weighting: "uniform"

    References
    ----------
    Module: xpcsjax.optimization.sequential_angle
    Inverse Variance Weighting: https://en.wikipedia.org/wiki/Inverse-variance_weighting
    """

    min_success_rate: float
    weighting: str


class NLSQValidationConfig(TypedDict, total=False):
    """NLSQ fit quality validation configuration.

    Configuration for validating NLSQ optimization results.
    Used by xpcsjax.optimization.nlsq.validation to classify fit quality.

    Attributes
    ----------
    chi2_good_threshold : float
        Reduced chi-squared below which fit is "good" (default: 2.0)
    chi2_acceptable_threshold : float
        Reduced chi-squared below which fit is "acceptable" (default: 5.0)
    min_parameter_significance : float
        Minimum parameter/uncertainty ratio for significance (default: 2.0)
    max_condition_number : float
        Maximum covariance matrix condition number (default: 1e12)
    """

    chi2_good_threshold: float
    chi2_acceptable_threshold: float
    min_parameter_significance: float
    max_condition_number: float


class OptimizationConfig(TypedDict, total=False):
    """Optimization section of configuration.

    Attributes
    ----------
    method : str
        Optimization method ("nlsq" or "auto")
    lsq : dict, optional
        NLSQ-specific settings
    angle_filtering : dict, optional
        Angle filtering settings
    streaming : StreamingConfig, optional
        Streaming optimization settings (checkpoint management, fault tolerance)
    stratification : StratificationConfig, optional
        Angle-stratified chunking settings (v2.2+, fixes per-angle parameter compatibility)
    sequential : SequentialConfig, optional
        Sequential per-angle optimization settings (v2.2+, fallback for extreme imbalance)
    """

    method: Literal["nlsq", "auto"]
    lsq: dict[str, Any]
    angle_filtering: dict[str, Any]
    streaming: StreamingConfig
    stratification: StratificationConfig
    sequential: SequentialConfig
    nlsq_validation: NLSQValidationConfig


class HomodyneConfig(TypedDict, total=False):
    """Complete xpcsjax configuration structure.

    Attributes
    ----------
    config_version : str
        Configuration file version
    analysis_mode : str
        Analysis mode ("static", "laminar_flow")
    experimental_data : ExperimentalDataConfig
        Experimental data specification
    parameter_space : ParameterSpaceConfig
        Parameter bounds and priors
    initial_parameters : InitialParametersConfig
        Initial parameter values
    optimization : OptimizationConfig
        Optimization settings
    output : dict, optional
        Output settings
    """

    config_version: str
    analysis_mode: AnalysisMode
    experimental_data: ExperimentalDataConfig
    parameter_space: ParameterSpaceConfig
    initial_parameters: InitialParametersConfig
    optimization: OptimizationConfig
    output: dict[str, Any]



# Parameter names for different modes
STATIC_PARAM_NAMES: list[str] = ["D0", "alpha", "D_offset"]
LAMINAR_FLOW_PARAM_NAMES: list[str] = [
    "D0",
    "alpha",
    "D_offset",
    "gamma_dot_t0",
    "beta",
    "gamma_dot_t_offset",
    "phi0",
]
SCALING_PARAM_NAMES: list[str] = ["contrast", "offset"]


# Parameter name mapping
PARAMETER_NAME_MAPPING: dict[str, str] = {
    "gamma_dot_0": "gamma_dot_t0",
    "gamma_dot_offset": "gamma_dot_t_offset",
    "phi_0": "phi0",
}
