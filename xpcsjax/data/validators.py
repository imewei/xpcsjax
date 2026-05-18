"""Configuration Validators for XPCS Data Loading
================================================

Focused validator functions for configuration parameter validation.
Extracted from config.py to reduce cyclomatic complexity and improve testability.

Each validator function:
- Takes specific parameters to validate
- Returns a list of error messages (empty if valid)
- Has single responsibility
"""

import os
from typing import Any


def validate_file_path(
    folder: str | None,
    filename: str | None,
    *,
    check_folder: bool = True,
    check_file: bool = True,
) -> list[str]:
    """Validate file path existence.

    Args:
        folder: Directory path
        filename: File name within the directory
        check_folder: Whether to validate folder existence
        check_file: Whether to validate file existence

    Returns:
        List of error messages (empty if valid)
    """
    errors: list[str] = []

    if check_folder and folder and not os.path.exists(folder):
        errors.append(f"Data folder does not exist: {folder}")

    if check_file and folder and filename:
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            errors.append(f"Data file does not exist: {full_path}")

    return errors


def validate_frame_range(
    start_frame: int | None,
    end_frame: int | None,
    *,
    min_frame: int = 1,
) -> list[str]:
    """Validate frame range parameters.

    Args:
        start_frame: Starting frame index
        end_frame: Ending frame index
        min_frame: Minimum allowed frame value

    Returns:
        List of error messages (empty if valid)
    """
    errors: list[str] = []

    if start_frame is not None and end_frame is not None:
        if end_frame != -1 and start_frame >= end_frame:
            errors.append(
                f"start_frame ({start_frame}) must be less than end_frame ({end_frame})"
            )
        if start_frame < min_frame:
            errors.append(f"start_frame ({start_frame}) must be >= {min_frame}")

    return errors


def validate_positive_value(
    value: float | int | None,
    field_name: str,
    *,
    allow_zero: bool = False,
) -> list[str]:
    """Validate that a value is positive.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        allow_zero: Whether zero is allowed

    Returns:
        List of error messages (empty if valid)
    """
    errors: list[str] = []

    if value is not None:
        if allow_zero:
            if value < 0:
                errors.append(f"{field_name} ({value}) must be non-negative")
        else:
            if value <= 0:
                errors.append(f"{field_name} ({value}) must be positive")

    return errors


def validate_numeric_range(
    range_dict: dict[str, Any] | None,
    field_name: str,
    *,
    require_positive: bool = False,
    value_bounds: tuple[float, float] | None = None,
    allow_wrapped: bool = False,
) -> list[str]:
    """Validate a min/max range dictionary.

    Args:
        range_dict: Dictionary with 'min' and 'max' keys
        field_name: Name of the field for error messages
        require_positive: Whether values must be positive
        value_bounds: Optional (min, max) bounds for allowed values
        allow_wrapped: Whether to allow min >= max (for wrapped ranges like phi)

    Returns:
        List of error messages (empty if valid)
    """
    errors: list[str] = []

    if not range_dict:
        return errors

    min_val = range_dict.get("min")
    max_val = range_dict.get("max")

    if require_positive:
        if min_val is not None and min_val <= 0:
            errors.append(f"{field_name}.min ({min_val}) must be positive")
        if max_val is not None and max_val <= 0:
            errors.append(f"{field_name}.max ({max_val}) must be positive")

    if (
        min_val is not None
        and max_val is not None
        and min_val >= max_val
        and not allow_wrapped
    ):
        errors.append(
            f"{field_name}.min ({min_val}) must be less than {field_name}.max ({max_val})"
        )

    if value_bounds is not None:
        lower, upper = value_bounds
        if min_val is not None and not (lower <= min_val <= upper):
            errors.append(
                f"{field_name}.min ({min_val}) should be in range [{lower}, {upper}]"
            )
        if max_val is not None and not (lower <= max_val <= upper):
            errors.append(
                f"{field_name}.max ({max_val}) should be in range [{lower}, {upper}]"
            )

    return errors


def validate_enum_value(
    value: str | None,
    field_name: str,
    allowed_values: list[str],
    *,
    default: str | None = None,
) -> list[str]:
    """Validate that a value is one of the allowed enum values.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        allowed_values: List of allowed values
        default: Default value (used if value is None)

    Returns:
        List of error messages (empty if valid)
    """
    errors: list[str] = []

    actual_value = value if value is not None else default

    if actual_value is not None and actual_value not in allowed_values:
        allowed_str = ", ".join(allowed_values)
        errors.append(
            f"{field_name} must be one of: {allowed_str} (got: {actual_value})"
        )

    return errors


# Validation rule definitions for schema-driven validation
VALIDATION_RULES: dict[str, dict[str, Any]] = {
    "combine_criteria": {
        "type": "enum",
        "allowed": ["AND", "OR"],
        "default": "AND",
    },
    "data_filtering.validation_level": {
        "type": "enum",
        "allowed": ["basic", "strict"],
        "default": "basic",
    },
    "v2_features.output_format": {
        "type": "enum",
        "allowed": ["numpy", "jax", "auto"],
        "default": "auto",
    },
    "v2_features.validation_level": {
        "type": "enum",
        "allowed": ["none", "basic", "full"],
        "default": "basic",
    },
    "v2_features.cache_strategy": {
        "type": "enum",
        "allowed": ["none", "simple", "intelligent"],
        "default": "intelligent",
    },
    "q_range": {
        "type": "range",
        "require_positive": True,
    },
    "phi_range": {
        "type": "range",
        "value_bounds": (-360, 360),
    },
}


def validate_by_rules(
    config: dict[str, Any],
    section: str,
    rules: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Validate configuration section using predefined rules.

    Args:
        config: Full configuration dictionary
        section: Section to validate (e.g., 'data_filtering')
        rules: Optional custom rules (defaults to VALIDATION_RULES)

    Returns:
        List of error messages (empty if valid)
    """
    errors: list[str] = []
    rules = rules or VALIDATION_RULES
    section_config = config.get(section, {})

    for field_path, rule in rules.items():
        # Check if this rule applies to the current section
        if field_path.startswith(f"{section}."):
            field_name = field_path.split(".", 1)[1]
            value = section_config.get(field_name)
        elif "." not in field_path and section == "data_filtering":
            # Top-level rules for data_filtering
            value = section_config.get(field_path)
            field_name = field_path
        else:
            continue

        rule_type = rule.get("type")

        if rule_type == "enum":
            errors.extend(
                validate_enum_value(
                    value,
                    f"{section}.{field_name}" if section else field_name,
                    rule["allowed"],
                    default=rule.get("default"),
                )
            )
        elif rule_type == "range":
            errors.extend(
                validate_numeric_range(
                    value if isinstance(value, dict) else None,
                    field_name,
                    require_positive=rule.get("require_positive", False),
                    value_bounds=rule.get("value_bounds"),
                )
            )

    return errors


__all__ = [
    "validate_file_path",
    "validate_frame_range",
    "validate_positive_value",
    "validate_numeric_range",
    "validate_enum_value",
    "validate_by_rules",
    "VALIDATION_RULES",
]
