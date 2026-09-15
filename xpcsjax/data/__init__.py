"""Data loading and management for the homodyne data layer.

Comprehensive data loading infrastructure supporting XPCS experimental data
from multiple synchrotron sources with YAML-first configuration, intelligent
caching, and JAX integration.

Key Features
------------
- YAML-first configuration with JSON support.
- Support for APS old format (``"aps_old"``) and APS-U unified format
  (``"aps_u"``) HDF5 files.
- Intelligent NPZ caching system.
- JAX array output with numpy fallback.
- Physics-based data validation.
- Integration with the modern core architecture.

Primary Components
------------------
- :class:`~xpcsjax.data.xpcs_loader.XPCSDataLoader`: Main class for loading
  XPCS data.
- :func:`~xpcsjax.data.xpcs_loader.load_xpcs_data`: Convenience function for
  simple data loading.
- :class:`~xpcsjax.data.phi_filtering.PhiAngleFilter`: Intelligent angle
  filtering for optimization performance.
- Configuration system with YAML/JSON support.
- Data validation and quality checks.

Examples
--------
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

from typing import Any

# Every submodule below ships in-package (no optional extras) and always
# imports successfully (2026-09-15 review, finding B3) — the try/except
# ImportError "maybe-present" shims and their HAS_* flags this block used to
# carry were dead code; a broken import here is a real bug, not a
# degraded-but-working install, so it should raise loudly instead of
# silently flipping a feature off (see tests/data/test_data_package_features.py,
# which pins this contract after PR #68 let exactly that happen unnoticed).
from xpcsjax.data.angle_filtering import (
    angle_in_range,
    apply_angle_filtering,
    apply_angle_filtering_for_optimization,
    apply_angle_filtering_for_plot,
    normalize_angle_to_symmetric_range,
)

# Typed dataset container (numpy-only; safe to import unconditionally).
from xpcsjax.data.dataset import XpcsDataset
from xpcsjax.data.optimization import (
    DatasetOptimizer,
    create_dataset_optimizer,
    optimize_for_method,
)
from xpcsjax.data.phi_filtering import (  # noqa: F401
    PhiAngleFilter,
    create_anisotropic_ranges,
    create_isotropic_ranges,
    filter_phi_angles,
    filter_phi_angles_jax,
)
from xpcsjax.data.preprocessing import (
    NoiseReductionMethod,
    NormalizationMethod,
    PreprocessingConfigurationError,
    PreprocessingError,
    PreprocessingPipeline,
    PreprocessingProvenance,
    PreprocessingResult,
    PreprocessingStage,
)
from xpcsjax.data.types import (
    DatasetInfo,
    ProcessingStrategy,
)
from xpcsjax.data.validation import (
    DataQualityReport,
    validate_xpcs_data,
)
from xpcsjax.data.validators import (
    VALIDATION_RULES,
    validate_by_rules,
    validate_enum_value,
    validate_file_path,
    validate_frame_range,
    validate_numeric_range,
    validate_positive_value,
)
from xpcsjax.data.xpcs_loader import (
    XPCSConfigurationError,
    XPCSDataFormatError,
    XPCSDataLoader,
    XPCSDependencyError,
    load_xpcs_config,
    load_xpcs_data,
)

# Data-layer version, surfaced by get_data_module_info() (documented public
# surface, see docs/source/development/releasing.rst).
__version__ = "2.23.1"


def get_data_module_info() -> dict:
    """Return information about data module capabilities.

    Returns
    -------
    dict
        Mapping with the data-layer version and the supported XPCS /
        configuration formats.
    """
    # Annotated ``dict[str, Any]`` because the value types intentionally mix
    # ``str`` (version) and ``list[str]`` (formats). Narrower hints rot when
    # the dict grows.
    info: dict[str, Any] = {
        "version": __version__,
        "xpcs_formats_supported": ["APS_old", "APS-U"],
        "config_formats_supported": ["YAML", "JSON"],
    }

    return info


# Main exports. Every name below is unconditionally imported above (all are
# hard/in-tree dependencies) - a single literal list, not built from HAS_*
# flags gated .extend() calls, per this module's __all__ convention.
__all__ = [
    # Core loader
    "XPCSDataLoader",
    "XpcsDataset",
    "load_xpcs_data",
    "load_xpcs_config",
    # Exceptions
    "XPCSDataFormatError",
    "XPCSDependencyError",
    "XPCSConfigurationError",
    # Utility functions
    "get_data_module_info",
    # Validation
    "validate_xpcs_data",
    "DataQualityReport",
    # Phi filtering
    "PhiAngleFilter",
    "filter_phi_angles",
    "create_anisotropic_ranges",
    "create_isotropic_ranges",
    # Angle filtering
    "normalize_angle_to_symmetric_range",
    "angle_in_range",
    "apply_angle_filtering",
    "apply_angle_filtering_for_optimization",
    "apply_angle_filtering_for_plot",
    # Preprocessing
    "PreprocessingPipeline",
    "PreprocessingResult",
    "PreprocessingProvenance",
    "PreprocessingStage",
    "NormalizationMethod",
    "NoiseReductionMethod",
    "PreprocessingError",
    "PreprocessingConfigurationError",
    # Optimization
    "DatasetOptimizer",
    "optimize_for_method",
    "DatasetInfo",
    "ProcessingStrategy",
    "create_dataset_optimizer",
    # Validators
    "VALIDATION_RULES",
    "validate_by_rules",
    "validate_enum_value",
    "validate_file_path",
    "validate_frame_range",
    "validate_numeric_range",
    "validate_positive_value",
]
