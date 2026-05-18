"""Configuration system for the xpcsjax package.

Provides configuration management, parameter handling, and physics validation.

Modules:
- manager.py: Main ConfigManager class for loading YAML/JSON configs
- parameter_registry.py: Parameter registration and defaults
- parameter_space.py: Parameter space and prior distribution handling
- parameter_manager.py: Parameter validation and management
- physics_validators.py: Physics-based constraint validation (v2.7+)
- types.py: TypedDict definitions for type safety
- parameter_names.py: Parameter name constants
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
from xpcsjax.config.parameter_space import ParameterSpace, PriorDistribution

# Physics validators (v2.7+)
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
    # Parameter manager (v2.4+)
    "ParameterManager",
    # Parameter registry (v2.4.1+)
    "ParameterInfo",
    "ParameterRegistry",
    "get_registry",
    "get_param_names",
    "get_all_param_names",
    "get_bounds",
    "get_defaults",
    # Parameter space
    "ParameterSpace",
    "PriorDistribution",
    # Physics validators (v2.7+)
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
