"""Configuration system for the xpcsjax package.

Provides configuration management, parameter handling, and physics validation
for the NLSQ-only XPCS analysis pipeline.

The central entry point is :class:`~xpcsjax.config.manager.ConfigManager`, which
loads and normalizes YAML/JSON configs and serves typed accessors.
:mod:`~xpcsjax.config.parameter_registry` is the single source of truth for
parameter names, bounds, and the
:class:`~xpcsjax.config.parameter_registry.AnalysisMode` enum; all other modules
read from it.

Modules
-------
manager
    :class:`~xpcsjax.config.manager.ConfigManager` for loading YAML/JSON configs.
parameter_registry
    Parameter registration, defaults, bounds, and the ``AnalysisMode`` enum
    (single source of truth).
parameter_space
    Parameter-space and bounds handling.
parameter_manager
    Parameter validation and active-parameter/bounds resolution per mode.
physics_validators
    Physics-based constraint validation.
types
    ``TypedDict`` definitions (including the ``data_type`` vocabulary) for type
    safety.
parameter_names
    Parameter-name constants.
"""

from xpcsjax.config.manager import ConfigManager, load_xpcs_config
from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import (
    ParameterInfo,
    ParameterRegistry,
    get_all_param_names,
    get_bounds,
    get_defaults,
    get_param_names,
    get_registry,
)
from xpcsjax.config.parameter_space import ParameterSpace

# Physics validators
from xpcsjax.config.physics_validators import (
    PHYSICS_CONSTRAINTS,
    ConstraintRule,
    ConstraintSeverity,
    PhysicsViolation,
    validate_all_parameters,
    validate_cross_parameter_constraints,
    validate_single_parameter,
)

# Type definitions
from xpcsjax.config.types import (
    BoundDict,
    ExperimentalDataConfig,
    InitialParametersConfig,
    ParameterSpaceConfig,
)

__all__ = [
    # Configuration management
    "ConfigManager",
    "load_xpcs_config",
    # Parameter manager
    "ParameterManager",
    # Parameter registry
    "ParameterInfo",
    "ParameterRegistry",
    "get_registry",
    "get_param_names",
    "get_all_param_names",
    "get_bounds",
    "get_defaults",
    # Parameter space
    "ParameterSpace",
    # Physics validators
    "ConstraintRule",
    "ConstraintSeverity",
    "PhysicsViolation",
    "PHYSICS_CONSTRAINTS",
    "validate_all_parameters",
    "validate_cross_parameter_constraints",
    "validate_single_parameter",
    # Type definitions
    "BoundDict",
    "ExperimentalDataConfig",
    "InitialParametersConfig",
    "ParameterSpaceConfig",
]
