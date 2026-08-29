"""Configuration system for XPCS data loading.

YAML-first configuration system with JSON support for XPCS data loading.
Provides configuration validation, schema definitions, and format conversion
utilities.

This module supports:

- YAML configuration loading and validation.
- JSON configuration support.
- Configuration schema validation.
- Migration utilities from JSON to YAML.
- Integration with modern configuration management.

Notes
-----
Configuration structure:

- ``experimental_data``: File paths and data parameters.
- ``analyzer_parameters``: Analysis settings (time, frames).
- ``enhanced_features``: Enhanced features and optimizations.
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Handle YAML dependency
try:
    from types import ModuleType

    import yaml

    HAS_YAML = True
    yaml_module: ModuleType | None = yaml
except ImportError:
    HAS_YAML = False
    yaml_module = None

# xpcsjax.data.validators is a sibling module shipped unconditionally in the
# same package — never an optional/extra dependency — so the ImportError
# fallback this try/except once guarded (a duplicate ~100-line inline
# validator plus stub logger) was unreachable dead code. Removed.
from xpcsjax.data.validators import (
    validate_enum_value,
    validate_file_path,
    validate_frame_range,
    validate_numeric_range,
    validate_positive_value,
)
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    missing_optional: list[str]


class XPCSConfigurationError(Exception):
    """Raised when XPCS configuration is invalid."""


# Configuration schema definitions
XPCS_CONFIG_SCHEMA = {
    "experimental_data": {
        "required": {
            "data_folder_path": str,
            "data_file_name": str,
        },
        "optional": {
            "phi_angles_path": str,
            "cache_file_path": str,
            "cache_filename_template": str,
            "cache_compression": bool,
            "apply_diagonal_correction": bool,
        },
        "defaults": {
            "phi_angles_path": "./output/",
            "cache_file_path": None,  # Will use data_folder_path if None
            "cache_filename_template": "cached_c2_frames_${start_frame}_${end_frame}.npz",
            "cache_compression": True,
            "apply_diagonal_correction": True,
        },
    },
    "data_filtering": {
        "required": {},
        "optional": {
            "enabled": bool,
            "q_range": dict,
            "phi_range": dict,
            "quality_threshold": (int, float),
            "frame_filtering": dict,
            "combine_criteria": str,
            "fallback_on_empty": bool,
            "validation_level": str,
        },
        "defaults": {
            "enabled": False,
            "q_range": {},
            "phi_range": {},
            "quality_threshold": None,
            "frame_filtering": {},
            "combine_criteria": "AND",  # "AND", "OR"
            "fallback_on_empty": True,
            "validation_level": "basic",  # "basic", "strict"
        },
    },
    "analyzer_parameters": {
        "required": {
            "dt": (int, float),
            "start_frame": int,
            "end_frame": int,
        },
        "optional": {
            "frame_step": int,
            "time_unit": str,
        },
        "defaults": {
            "frame_step": 1,
            "time_unit": "seconds",
        },
    },
    "v2_features": {
        "required": {},
        "optional": {
            "output_format": str,
            "validation_level": str,
            "performance_optimization": bool,
            "physics_validation": bool,
            "cache_strategy": str,
            "parallel_processing": bool,
            "gpu_acceleration": bool,
        },
        "defaults": {
            "output_format": "auto",  # "numpy", "jax", "auto"
            "validation_level": "basic",  # "none", "basic", "full"
            "performance_optimization": True,
            "physics_validation": False,
            "cache_strategy": "intelligent",  # "none", "simple", "intelligent"
            "parallel_processing": False,
            "gpu_acceleration": False,
        },
    },
}


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    XPCSConfigurationError
        If PyYAML is unavailable, the file is missing, the file is empty or
        unparseable, or the root node is not a mapping.
    """
    if not HAS_YAML or yaml_module is None:
        raise XPCSConfigurationError("PyYAML required for YAML configuration files")

    config_path = Path(config_path)

    if not config_path.exists():
        raise XPCSConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = yaml_module.safe_load(f)

        if config_data is None:
            raise XPCSConfigurationError(f"Empty or invalid YAML file: {config_path}")

        logger.debug(f"Loaded YAML configuration from: {config_path}")
        # Validate root is a dict — assert is stripped with -O flag and raises
        # AssertionError (wrong type). Use a proper domain exception instead.
        if not isinstance(config_data, dict):
            raise XPCSConfigurationError(
                f"YAML file must contain a mapping at root level, got "
                f"{type(config_data).__name__}: {config_path}"
            )
        return config_data

    except yaml_module.YAMLError as e:
        raise XPCSConfigurationError(
            f"Failed to parse YAML configuration {config_path}: {e}",
        ) from e


def load_json_config(config_path: str | Path) -> dict[str, Any]:
    """Load a JSON configuration file with automatic YAML conversion.

    Parameters
    ----------
    config_path
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    XPCSConfigurationError
        If the file is missing, the file is unparseable, or the root node is
        not an object.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise XPCSConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)

        logger.debug(f"Loaded JSON configuration from: {config_path}")
        logger.info("Consider migrating to YAML format for improved readability")
        # Validate root is a dict — assert is stripped with -O flag and raises
        # AssertionError (wrong type). Use a proper domain exception instead.
        if not isinstance(config_data, dict):
            raise XPCSConfigurationError(
                f"JSON file must contain an object at root level, got "
                f"{type(config_data).__name__}: {config_path}"
            )
        return config_data

    except json.JSONDecodeError as e:
        raise XPCSConfigurationError(
            f"Failed to parse JSON configuration {config_path}: {e}",
        ) from e


def validate_config_schema(
    config: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> ConfigValidationResult:
    """Validate a configuration against a schema.

    Parameters
    ----------
    config
        Configuration dictionary to validate.
    schema
        Schema to validate against; defaults to ``XPCS_CONFIG_SCHEMA``.

    Returns
    -------
    ConfigValidationResult
        Validation result holding errors and warnings.
    """
    if schema is None:
        schema = XPCS_CONFIG_SCHEMA

    # A YAML section written with no nested value (e.g. `experimental_data:`)
    # parses to None; coerce to {} up front so every section-dict access
    # below -- including inside _validate_parameter_values(config) -- sees a
    # dict instead of crashing on `x not in None` / `None.get(...)`. A
    # non-None, non-dict section (e.g. a mis-indented YAML list) is a real
    # structural error rather than an absent section, so it is flagged
    # explicitly below instead of being silently absorbed into {}.
    malformed_sections = {
        key: type(value).__name__
        for key, value in config.items()
        if value is not None and not isinstance(value, dict)
    }
    config = {key: (value if isinstance(value, dict) else {}) for key, value in config.items()}

    errors = [
        f"Configuration section '{key}' must be a mapping, got {type_name}"
        for key, type_name in malformed_sections.items()
    ]
    warnings = []
    missing_optional = []

    for section_name, section_schema in schema.items():
        if not isinstance(section_schema, dict):
            continue

        if section_name not in config:
            if section_schema.get("required"):
                errors.append(f"Missing required configuration section: {section_name}")
            else:
                warnings.append(
                    f"Missing optional configuration section: {section_name}",
                )
            continue

        section_config = config[section_name]

        # Check required parameters
        required_params = section_schema.get("required", {})
        if not isinstance(required_params, dict):
            continue

        for param_name, param_type in required_params.items():
            if param_name not in section_config:
                errors.append(
                    f"Missing required parameter: {section_name}.{param_name}",
                )
            else:
                value = section_config[param_name]
                if isinstance(param_type, tuple):
                    # Multiple allowed types
                    if not any(isinstance(value, t) for t in param_type):
                        errors.append(
                            f"Parameter {section_name}.{param_name} has wrong type: "
                            f"expected {param_type}, got {type(value)}",
                        )
                else:
                    if not isinstance(value, param_type):
                        errors.append(
                            f"Parameter {section_name}.{param_name} has wrong type: "
                            f"expected {param_type}, got {type(value)}",
                        )

        # Check optional parameters
        optional_params = section_schema.get("optional", {})
        if not isinstance(optional_params, dict):
            continue

        for param_name, param_type in optional_params.items():
            if param_name not in section_config:
                missing_optional.append(f"{section_name}.{param_name}")
            else:
                value = section_config[param_name]
                if isinstance(param_type, tuple):
                    if not any(isinstance(value, t) for t in param_type):
                        warnings.append(
                            f"Parameter {section_name}.{param_name} has unexpected type: "
                            f"expected {param_type}, got {type(value)}",
                        )
                else:
                    if not isinstance(value, param_type):
                        warnings.append(
                            f"Parameter {section_name}.{param_name} has unexpected type: "
                            f"expected {param_type}, got {type(value)}",
                        )

    # Validate specific parameter values
    errors.extend(_validate_parameter_values(config))
    warnings.extend(_validate_parameter_warnings(config))

    return ConfigValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        missing_optional=missing_optional,
    )


def _validate_parameter_values(config: dict[str, Any]) -> list[str]:
    """Validate specific parameter value constraints.

    Delegates to focused validator functions from validators module for
    reduced cyclomatic complexity and improved testability.
    """
    errors: list[str] = []

    # Extract config sections
    exp_data = config.get("experimental_data", {})
    analyzer = config.get("analyzer_parameters", {})
    data_filtering = config.get("data_filtering", {})
    v2_features = config.get("v2_features", {})

    errors.extend(
        validate_file_path(
            exp_data.get("data_folder_path"),
            exp_data.get("data_file_name"),
        )
    )

    errors.extend(
        validate_frame_range(
            analyzer.get("start_frame"),
            analyzer.get("end_frame"),
        )
    )

    errors.extend(validate_positive_value(analyzer.get("dt"), "dt"))

    # Data filtering validation (only when enabled)
    if data_filtering.get("enabled", False):
        errors.extend(
            validate_numeric_range(
                data_filtering.get("q_range"),
                "q_range",
                require_positive=True,
            )
        )
        errors.extend(
            validate_numeric_range(
                data_filtering.get("phi_range"),
                "phi_range",
                value_bounds=(-360, 360),
                allow_wrapped=True,
            )
        )
        errors.extend(
            validate_positive_value(
                data_filtering.get("quality_threshold"),
                "quality_threshold",
            )
        )
        errors.extend(
            validate_enum_value(
                data_filtering.get("combine_criteria"),
                "combine_criteria",
                ["AND", "OR"],
                default="AND",
            )
        )
        errors.extend(
            validate_enum_value(
                data_filtering.get("validation_level"),
                "data_filtering.validation_level",
                ["basic", "strict"],
                default="basic",
            )
        )

    # v2_features validation
    errors.extend(
        validate_enum_value(
            v2_features.get("output_format"),
            "output_format",
            ["numpy", "jax", "auto"],
            default="auto",
        )
    )
    errors.extend(
        validate_enum_value(
            v2_features.get("validation_level"),
            "validation_level",
            ["none", "basic", "full"],
            default="basic",
        )
    )
    errors.extend(
        validate_enum_value(
            v2_features.get("cache_strategy"),
            "cache_strategy",
            ["none", "simple", "intelligent"],
            default="intelligent",
        )
    )

    return errors


def _validate_parameter_warnings(config: dict[str, Any]) -> list[str]:
    """Generate warnings for parameter values that may cause issues."""
    warnings = []

    analyzer = config.get("analyzer_parameters", {})

    # Warn about very large frame ranges
    start_frame = analyzer.get("start_frame", 1)
    end_frame = analyzer.get("end_frame", 1000)
    if end_frame != -1:
        frame_count = end_frame - start_frame + 1
        if frame_count > 10000:
            warnings.append(
                f"Large frame range ({frame_count} frames) may result in long processing time",
            )

    # Warn about very small dt values
    dt = analyzer.get("dt")
    if dt is not None and dt < 1e-6:
        warnings.append(f"Very small dt value ({dt}) - check time units")

    return warnings


def apply_config_defaults(
    config: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply default values to a configuration.

    Parameters
    ----------
    config
        Configuration dictionary.
    schema
        Schema carrying default values; defaults to ``XPCS_CONFIG_SCHEMA``.

    Returns
    -------
    dict
        Configuration with defaults applied.
    """
    if schema is None:
        schema = XPCS_CONFIG_SCHEMA

    # Deep-copy so defaults are applied only to the returned config; the
    # previous shallow copy() left nested section dicts shared with the
    # caller's input and mutated them in place (contract violation).
    config_with_defaults = copy.deepcopy(config)

    for section_name, section_schema in schema.items():
        if not isinstance(section_schema, dict):
            continue

        if section_name not in config_with_defaults:
            config_with_defaults[section_name] = {}

        section_config = config_with_defaults[section_name]
        if not isinstance(section_config, dict):
            # Same null-section case as validate_config_schema: coerce to {}
            # so defaults apply instead of raising on `param_name not in None`.
            section_config = {}
            config_with_defaults[section_name] = section_config
        defaults = section_schema.get("defaults", {})
        if not isinstance(defaults, dict):
            continue

        for param_name, default_value in defaults.items():
            if param_name not in section_config:
                # Special handling for cache_file_path default
                if param_name == "cache_file_path" and default_value is None:
                    data_folder = config_with_defaults.get("experimental_data", {}).get(
                        "data_folder_path",
                    )
                    if data_folder:
                        section_config[param_name] = data_folder
                else:
                    section_config[param_name] = default_value
                logger.debug(
                    f"Applied default for {section_name}.{param_name}: {default_value}",
                )

    return config_with_defaults


# Export main functions
__all__ = [
    "load_yaml_config",
    "load_json_config",
    "validate_config_schema",
    "apply_config_defaults",
    "ConfigValidationResult",
    "XPCSConfigurationError",
    "XPCS_CONFIG_SCHEMA",
]
