"""Data Loading and Management for Homodyne
=============================================

Comprehensive data loading infrastructure supporting XPCS experimental data
from multiple synchrotron sources with YAML-first configuration, intelligent
caching, and JAX integration.

Key Features:
- YAML-first configuration with JSON support
- Support for APS old format and APS-U new format HDF5 files
- Intelligent NPZ caching system
- JAX array output with numpy fallback
- Physics-based data validation
- Integration with modern core architecture

Primary Components:
- XPCSDataLoader: Main class for loading XPCS data
- load_xpcs_data: Convenience function for simple data loading
- PhiAngleFilter: Intelligent angle filtering for optimization performance
- Configuration system with YAML/JSON support
- Data validation and quality checks

Example Usage:
    >>> from xpcsjax.data import XPCSDataLoader, load_xpcs_data, filter_phi_angles
    >>>
    >>> # Using YAML configuration
    >>> data = load_xpcs_data("xpcs_config.yaml")
    >>>
    >>> # Using loader class
    >>> loader = XPCSDataLoader(config_path="config.yaml")
    >>> data = loader.load_experimental_data()
    >>>
    >>> # Apply phi angle filtering for performance
    >>> angles = data['phi_angles_list']
    >>> indices, filtered_angles = filter_phi_angles(angles)
    >>>
    >>> # Check data structure
    >>> print(data.keys())
    >>> dict_keys(['wavevector_q_list', 'phi_angles_list', 't1', 't2', 'c2_exp'])
"""

# Handle imports with graceful fallback for missing dependencies
try:
    from xpcsjax.data.xpcs_loader import (
        XPCSConfigurationError,
        XPCSDataFormatError,
        XPCSDataLoader,
        XPCSDependencyError,
        load_xpcs_config,
        load_xpcs_data,
    )

    HAS_XPCS_LOADER = True
    _loader_error = None
except ImportError as e:
    HAS_XPCS_LOADER = False
    _loader_error = str(e)

    # Create placeholder classes for graceful degradation
    class XPCSDataLoader:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"XPCS loader not available: {_loader_error}")

    class XPCSDataFormatError(Exception):
        pass

    class XPCSDependencyError(Exception):
        pass

    class XPCSConfigurationError(Exception):
        pass

    def load_xpcs_data(*args, **kwargs):
        raise ImportError(f"XPCS loader not available: {_loader_error}")

    def load_xpcs_config(*args, **kwargs):
        raise ImportError(f"XPCS loader not available: {_loader_error}")


# Import additional components when available (re-exported for public API)
try:
    from xpcsjax.data.validation import (  # noqa: F401
        DataQualityReport,
        validate_xpcs_data,
    )

    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False

try:
    from xpcsjax.data.phi_filtering import (  # noqa: F401
        PhiAngleFilter,
        create_anisotropic_ranges,
        create_isotropic_ranges,
        filter_phi_angles,
        filter_phi_angles_jax,
    )

    HAS_PHI_FILTERING = True
except ImportError:
    HAS_PHI_FILTERING = False

try:
    from xpcsjax.data.angle_filtering import (  # noqa: F401
        angle_in_range,
        apply_angle_filtering,
        apply_angle_filtering_for_optimization,
        apply_angle_filtering_for_plot,
        normalize_angle_to_symmetric_range,
    )

    HAS_ANGLE_FILTERING = True
except ImportError:
    HAS_ANGLE_FILTERING = False

try:
    from xpcsjax.data.preprocessing import (  # noqa: F401
        NoiseReductionMethod,
        NormalizationMethod,
        PreprocessingConfigurationError,
        PreprocessingError,
        PreprocessingPipeline,
        PreprocessingProvenance,
        PreprocessingResult,
        PreprocessingStage,
        create_default_preprocessing_config,
        preprocess_xpcs_data,
    )

    HAS_PREPROCESSING = True
except ImportError:
    HAS_PREPROCESSING = False

try:
    from xpcsjax.data.optimization import (  # noqa: F401
        DatasetOptimizer,
        create_dataset_optimizer,
        optimize_for_method,
    )
    from xpcsjax.data.types import (  # noqa: F401
        DatasetInfo,
        ProcessingStrategy,
    )

    HAS_OPTIMIZATION = True
except ImportError:
    HAS_OPTIMIZATION = False

# Version and feature information
__version__ = "2.23.1"
__features__ = {
    "xpcs_loader": HAS_XPCS_LOADER,
    "validation": HAS_VALIDATION,
    "phi_filtering": HAS_PHI_FILTERING,
    "angle_filtering": HAS_ANGLE_FILTERING,
    "preprocessing": HAS_PREPROCESSING,
    "optimization": HAS_OPTIMIZATION,
    "validators": True,  # Validators module (v2.7+)
    "yaml_config": True,  # Always available through fallbacks
    "json_support": True,  # Always available
}


def get_data_module_info() -> dict:
    """Get information about data module capabilities.

    Returns:
        Dictionary with feature availability and version info
    """
    info = {
        "version": __version__,
        "features": __features__.copy(),
        "xpcs_formats_supported": ["APS_old", "APS-U"] if HAS_XPCS_LOADER else [],
        "config_formats_supported": ["YAML", "JSON"],
    }

    if not HAS_XPCS_LOADER:
        info["loader_error"] = _loader_error

    return info


# Main exports
__all__ = [
    # Core loader
    "XPCSDataLoader",
    "load_xpcs_data",
    "load_xpcs_config",
    # Exceptions
    "XPCSDataFormatError",
    "XPCSDependencyError",
    "XPCSConfigurationError",
    # Utility functions
    "get_data_module_info",
]

# Conditional exports
if HAS_VALIDATION:
    __all__.extend(["validate_xpcs_data", "DataQualityReport"])

if HAS_PHI_FILTERING:
    __all__.extend(
        [
            "PhiAngleFilter",
            "filter_phi_angles",
            "create_anisotropic_ranges",
            "create_isotropic_ranges",
        ],
    )

if HAS_ANGLE_FILTERING:
    __all__.extend(
        [
            "normalize_angle_to_symmetric_range",
            "angle_in_range",
            "apply_angle_filtering",
            "apply_angle_filtering_for_optimization",
            "apply_angle_filtering_for_plot",
        ],
    )

if HAS_PREPROCESSING:
    __all__.extend(
        [
            "PreprocessingPipeline",
            "PreprocessingResult",
            "PreprocessingProvenance",
            "PreprocessingStage",
            "NormalizationMethod",
            "NoiseReductionMethod",
            "PreprocessingError",
            "PreprocessingConfigurationError",
            "create_default_preprocessing_config",
            "preprocess_xpcs_data",
        ],
    )

if HAS_OPTIMIZATION:
    __all__.extend(
        [
            "DatasetOptimizer",
            "optimize_for_method",
            "DatasetInfo",
            "ProcessingStrategy",
            "create_dataset_optimizer",
        ],
    )

# Validators (v2.7+)
try:
    from xpcsjax.data.validators import (  # noqa: F401
        VALIDATION_RULES,
        validate_by_rules,
        validate_enum_value,
        validate_file_path,
        validate_frame_range,
        validate_numeric_range,
        validate_positive_value,
    )

    HAS_VALIDATORS = True
    __all__.extend(
        [
            "VALIDATION_RULES",
            "validate_by_rules",
            "validate_enum_value",
            "validate_file_path",
            "validate_frame_range",
            "validate_numeric_range",
            "validate_positive_value",
        ],
    )
except ImportError:
    HAS_VALIDATORS = False
