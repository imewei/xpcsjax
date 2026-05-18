"""Minimal Configuration Management for Homodyne
===================================================

Simplified configuration system with preserved API compatibility.
Provides essential YAML/JSON loading with the same interface as the original
ConfigManager while removing complex features not needed for core functionality.

Note: GPU support removed in v2.3.0 - CPU-only execution.
"""

import json
from pathlib import Path
from typing import Any

# Handle YAML dependency
try:
    from types import ModuleType

    import yaml

    HAS_YAML = True
    yaml_module: ModuleType | None = yaml
    _YAMLError: type[BaseException] = yaml.YAMLError
except ImportError:
    HAS_YAML = False
    yaml_module = None
    _YAMLError = Exception

# Import minimal logging
try:
    from xpcsjax.utils.logging import get_logger

    HAS_LOGGING = True
except ImportError:
    import logging
    from typing import Any as _Any

    HAS_LOGGING = False

    def get_logger(name: str, **kwargs: _Any) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class ConfigManager:
    """Minimal configuration manager for xpcsjax v2 scattering analysis.

    Provides simplified configuration loading with preserved API compatibility.

    Key Features:
    - YAML/JSON configuration file loading
    - Compatible .config attribute access
    - Preserved constructor signature
    - Graceful fallback to defaults
    - CPU-only execution (GPU support removed in v2.3.0)

    Usage:
        config_manager = ConfigManager('my_config.yaml')
        data = config_manager.config
    """

    def __init__(
        self,
        config_file: str = "xpcsjax_config.yaml",
        config_override: dict[str, Any] | None = None,
    ):
        """Initialize configuration manager.

        Parameters
        ----------
        config_file : str
            Path to YAML/JSON configuration file
        config_override : dict, optional
            Override configuration data instead of loading from file
        """
        self.config_file = config_file
        self.config: dict[str, Any] | None = None

        # Cache for ParameterManager to avoid repeated instantiation
        self._cached_param_manager: Any | None = None

        if config_override is not None:
            self.config = config_override.copy()
            logger.info("Configuration loaded from override data")
        else:
            self.load_config()

        # Normalize schema for backward compatibility
        self._normalize_schema()

    def load_config(self) -> None:
        """Load and parse YAML/JSON configuration file.

        Supports both YAML and JSON formats with graceful fallback
        to default configuration if loading fails.
        """
        try:
            if self.config_file is None:
                raise ValueError("Configuration file path cannot be None")

            config_path = Path(self.config_file)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_file}",
                )

            # Determine file format and load accordingly
            file_extension = config_path.suffix.lower()

            # Use 8KB buffering for improved I/O performance on large config files
            with open(config_path, buffering=8192, encoding="utf-8") as f:
                if file_extension in [".yaml", ".yml"] and HAS_YAML and yaml_module:
                    self.config = yaml_module.safe_load(f)
                elif file_extension == ".json":
                    self.config = json.load(f)
                elif HAS_YAML and yaml_module:
                    # Try YAML first for unknown extensions
                    content = f.read()
                    try:
                        self.config = yaml_module.safe_load(content)
                    except yaml_module.YAMLError:
                        # Fallback to JSON
                        self.config = json.loads(content)
                else:
                    # Only JSON available
                    self.config = json.load(f)

            logger.info(f"Configuration loaded from: {self.config_file}")

            # Display version information if available
            if self.config is None:
                logger.warning(
                    "Configuration file '%s' is empty or null; using defaults",
                    self.config_file,
                )
                self.config = self._get_default_config()
                return

            if isinstance(self.config, dict) and "metadata" in self.config:
                version = self.config["metadata"].get("config_version", "Unknown")
                logger.info(f"Configuration version: {version}")

            # Optional validation (can be disabled via environment variable)
            import os

            if os.environ.get("HOMODYNE_VALIDATE_CONFIG", "true").lower() == "true":
                self._validate_config()

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.info("Using default configuration...")
            self.config = self._get_default_config()
        except FileNotFoundError:
            # Re-raise immediately: wrong config path must be reported, not silenced.
            # Proceeding with stub defaults would produce confusing downstream errors.
            raise
        except (
            OSError,
            ValueError,
            UnicodeDecodeError,
            TypeError,
            KeyError,
            _YAMLError,
        ) as e:
            logger.error(f"Configuration parsing error: {e}")
            logger.info("Using default configuration...")
            self.config = self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration structure.

        T052: Logs default value application at DEBUG level.

        Returns minimal configuration that supports basic analysis modes.
        CPU-only execution (GPU support removed in v2.3.0).
        """
        # T052: Log default value application
        logger.debug("Applying default configuration values (fallback)")
        return {
            "metadata": {
                "config_version": "2.18.0",
                "description": "Default minimal configuration (CPU-only)",
            },
            "analysis_mode": "static",
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": -1,
            },
            "experimental_data": {
                "file_path": None,
                "cache_directory": "./cache",
                "use_caching": True,
            },
            "optimization": {
                "method": "nlsq",
                "lsq": {
                    "max_iterations": 10000,
                    "tolerance": 1e-8,
                    "method": "trf",
                },
                "mcmc": {
                    "n_samples": 1000,
                    "n_warmup": 1000,
                    "n_chains": 4,
                    "target_accept_prob": 0.8,
                },
            },
            "output": {
                "formats": ["yaml", "npz"],
                "include_diagnostics": True,
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "console": {"enabled": True},
                "file": {"enabled": False},
            },
        }

    def get_config(self) -> dict[str, Any]:
        """Get the current configuration dictionary.

        Returns
        -------
        Dict[str, Any]
            Current configuration dictionary
        """
        if self.config is None:
            return {}
        return self.config

    def update_config(self, key: str, value: Any) -> None:
        """Update a configuration value using dot notation.

        Parameters
        ----------
        key : str
            Configuration key (supports dot notation like 'optimization.method')
        value : Any
            New value to set
        """
        if self.config is None:
            self.config = {}

        keys = key.split(".")
        config_ref = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]

        # Set the value
        config_ref[keys[-1]] = value

    def is_static_mode_enabled(self) -> bool:
        """Check if static analysis mode is enabled."""
        if not self.config:
            return True
        analysis_mode = self.config.get("analysis_mode", "static_isotropic")
        return "static" in analysis_mode.lower()

    def get_target_angle_ranges(self) -> dict[str, Any]:
        """Get angle filtering ranges."""
        if not self.config:
            return {"enabled": False}

        optimization = self.config.get("optimization", {})
        angle_filtering = optimization.get("angle_filtering", {})
        if not isinstance(angle_filtering, dict):
            logger.warning(
                "optimization.angle_filtering must be a dict, ignoring (got %s)",
                type(angle_filtering).__name__,
            )
            return {"enabled": False}
        return angle_filtering

    def _get_parameter_manager(self) -> Any:
        """Get or create cached ParameterManager.

        This avoids creating a new ParameterManager on every config access,
        providing ~14x speedup for repeated parameter queries.

        Returns
        -------
        ParameterManager
            Cached ParameterManager instance
        """
        if self._cached_param_manager is None:
            from xpcsjax.config.parameter_manager import ParameterManager

            # Determine analysis mode
            analysis_mode = "laminar_flow"
            if self.is_static_mode_enabled():
                analysis_mode = "static"

            # Create and cache ParameterManager
            self._cached_param_manager = ParameterManager(self.config, analysis_mode)
            logger.debug(f"Created cached ParameterManager for mode: {analysis_mode}")

        return self._cached_param_manager

    def get_parameter_bounds(
        self,
        parameter_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get parameter bounds from configuration (cached).

        Uses cached ParameterManager internally for improved performance.

        Parameters
        ----------
        parameter_names : list of str, optional
            List of parameter names to get bounds for. If None, returns bounds
            for all parameters in the current analysis mode.

        Returns
        -------
        list of dict
            List of bound dictionaries with keys: 'name', 'min', 'max', 'type'

        Examples
        --------
        >>> config_mgr = ConfigManager("config.yaml")
        >>> bounds = config_mgr.get_parameter_bounds(["D0", "alpha"])
        >>> bounds[0]
        {'min': 1.0, 'max': 1000000.0, 'name': 'D0', 'type': 'Normal'}

        Notes
        -----
        This method uses a cached ParameterManager for ~14x speedup on repeated calls.
        """
        bounds = self._get_parameter_manager().get_parameter_bounds(parameter_names)
        if not isinstance(bounds, list):
            raise TypeError(
                f"ParameterManager.get_parameter_bounds returned {type(bounds).__name__}, expected list"
            )
        return bounds

    def get_active_parameters(self) -> list[str]:
        """Get list of active (physical) parameters from configuration (cached).

        Uses cached ParameterManager internally for improved performance.

        Returns
        -------
        list of str
            List of parameter names to be optimized. Falls back to mode-appropriate
            parameters if not specified in config.

        Examples
        --------
        >>> config_mgr = ConfigManager("config.yaml")
        >>> config_mgr.get_active_parameters()
        ['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta', 'gamma_dot_t_offset', 'phi0']

        Notes
        -----
        This method uses a cached ParameterManager for ~14x speedup on repeated calls.
        """
        params = self._get_parameter_manager().get_active_parameters()
        if not isinstance(params, list):
            raise TypeError(
                f"ParameterManager.get_active_parameters returned {type(params).__name__}, expected list"
            )
        return params

    def get_initial_parameters(
        self,
        use_midpoint_defaults: bool = True,
    ) -> dict[str, float]:
        """Get initial parameter values from configuration.

        Loads initial parameter values from the `initial_parameters.values` section
        of the configuration. If values are null or missing, calculates mid-point
        defaults from parameter bounds.

        Parameters
        ----------
        use_midpoint_defaults : bool
            If True (default), calculate mid-point defaults when values are null.
            If False, raise an error when values are missing.

        Returns
        -------
        dict[str, float]
            Dictionary mapping parameter names (canonical) to initial values.
            Only includes active parameters (excludes fixed parameters).

        Raises
        ------
        ValueError
            If values are null and use_midpoint_defaults is False.
            If number of values doesn't match number of parameter names.

        Examples
        --------
        >>> # With explicit values in config
        >>> config = {
        ...     'initial_parameters': {
        ...         'parameter_names': ['D0', 'alpha', 'D_offset'],
        ...         'values': [1000.0, 0.5, 10.0]
        ...     }
        ... }
        >>> config_mgr = ConfigManager(config_override=config)
        >>> config_mgr.get_initial_parameters()
        {'D0': 1000.0, 'alpha': 0.5, 'D_offset': 10.0}

        >>> # With null values (mid-point defaults)
        >>> config = {
        ...     'initial_parameters': {
        ...         'parameter_names': ['D0', 'alpha'],
        ...         'values': null
        ...     }
        ... }
        >>> config_mgr = ConfigManager(config_override=config)
        >>> params = config_mgr.get_initial_parameters()
        >>> # params['D0'] will be mid-point of bounds: (min + max) / 2

        Notes
        -----
        - Uses ParameterManager for name mapping (gamma_dot_0 → gamma_dot_t0)
        - Respects active_parameters and fixed_parameters from config
        - Logs when using mid-point defaults
        - Returns only active parameters (fixed parameters excluded)
        """
        if not self.config:
            logger.warning("No configuration loaded, using empty initial parameters")
            return {}

        # Get initial_parameters section
        initial_params = self.config.get("initial_parameters", {})
        if not initial_params:
            logger.info(
                "No initial_parameters section in config, using mid-point defaults"
            )
            return self._calculate_midpoint_defaults()

        # Get parameter names from config
        param_names_config = initial_params.get("parameter_names")
        if not param_names_config or not isinstance(param_names_config, list):
            logger.info(
                "No parameter_names in initial_parameters, using active parameters from mode"
            )
            return self._calculate_midpoint_defaults()

        # Get parameter values from config
        param_values = initial_params.get("values")

        # Handle null/missing values
        if param_values is None:
            if use_midpoint_defaults:
                logger.info(
                    f"initial_parameters.values is null, calculating mid-point defaults for {len(param_names_config)} parameters"
                )
                return self._calculate_midpoint_defaults()
            else:
                raise ValueError(
                    "initial_parameters.values is null and use_midpoint_defaults is False"
                )

        # Validate that values is a list
        if not isinstance(param_values, list):
            raise ValueError(
                f"initial_parameters.values must be a list, got {type(param_values)}"
            )

        # Validate length match
        if len(param_values) != len(param_names_config):
            raise ValueError(
                f"Number of values ({len(param_values)}) does not match "
                f"number of parameter_names ({len(param_names_config)})"
            )

        # Get ParameterManager for name mapping (used for validation)
        _param_manager = self._get_parameter_manager()  # noqa: F841

        # Import name mapping once at the top of this section
        from xpcsjax.config.types import PARAMETER_NAME_MAPPING

        # Build initial parameters dict with name mapping
        initial_params_dict: dict[str, float] = {}
        for param_name, value in zip(param_names_config, param_values, strict=False):
            # Apply name mapping (e.g., gamma_dot_0 → gamma_dot_t0)
            canonical_name = PARAMETER_NAME_MAPPING.get(param_name, param_name)
            initial_params_dict[canonical_name] = float(value)

        # Filter by active_parameters if specified
        active_params_config = initial_params.get("active_parameters")
        if active_params_config and isinstance(active_params_config, list):
            # Map active parameter names to canonical names
            active_canonical = set()
            for name in active_params_config:
                canonical = PARAMETER_NAME_MAPPING.get(name, name)
                active_canonical.add(canonical)

            # Filter to only active parameters
            initial_params_dict = {
                k: v for k, v in initial_params_dict.items() if k in active_canonical
            }
            logger.info(
                f"Filtered to {len(initial_params_dict)} active parameters: {list(initial_params_dict.keys())}"
            )

        # Exclude fixed_parameters
        fixed_params = initial_params.get("fixed_parameters")
        if fixed_params and isinstance(fixed_params, dict):
            # Map fixed parameter names to canonical names
            fixed_canonical = set()
            for name in fixed_params.keys():
                canonical = PARAMETER_NAME_MAPPING.get(name, name)
                fixed_canonical.add(canonical)

            # Remove fixed parameters from initial_params_dict
            initial_params_dict = {
                k: v for k, v in initial_params_dict.items() if k not in fixed_canonical
            }
            logger.info(
                f"Excluded {len(fixed_canonical)} fixed parameters, "
                f"{len(initial_params_dict)} remaining"
            )

        # Load per-angle scaling parameters (contrast, offset) if present
        per_angle_scaling = initial_params.get("per_angle_scaling")
        if per_angle_scaling and isinstance(per_angle_scaling, dict):
            # Extract contrast and offset arrays
            contrast_values = per_angle_scaling.get("contrast")
            offset_values = per_angle_scaling.get("offset")

            if contrast_values is not None and isinstance(contrast_values, list):
                if len(contrast_values) == 1:
                    # Single-angle: use scalar contrast
                    initial_params_dict["contrast"] = float(contrast_values[0])
                    logger.info(
                        f"Loaded scalar contrast from per_angle_scaling: {contrast_values[0]}"
                    )
                else:
                    # Multi-angle: use per-angle contrast_0, contrast_1, ...
                    for idx, val in enumerate(contrast_values):
                        initial_params_dict[f"contrast_{idx}"] = float(val)
                    logger.info(
                        f"Loaded {len(contrast_values)} per-angle contrast values"
                    )

            if offset_values is not None and isinstance(offset_values, list):
                if len(offset_values) == 1:
                    # Single-angle: use scalar offset
                    initial_params_dict["offset"] = float(offset_values[0])
                    logger.info(
                        f"Loaded scalar offset from per_angle_scaling: {offset_values[0]}"
                    )
                else:
                    # Multi-angle: use per-angle offset_0, offset_1, ...
                    for idx, val in enumerate(offset_values):
                        initial_params_dict[f"offset_{idx}"] = float(val)
                    logger.info(f"Loaded {len(offset_values)} per-angle offset values")

        logger.info(
            f"Loaded initial parameters from config: {list(initial_params_dict.keys())}"
        )

        return initial_params_dict

    def _calculate_midpoint_defaults(self) -> dict[str, float]:
        """Calculate mid-point default values from parameter bounds.

        Returns
        -------
        dict[str, float]
            Dictionary mapping parameter names to mid-point values: (min + max) / 2

        Notes
        -----
        - Uses ParameterManager to get bounds
        - Only includes active parameters (excludes fixed)
        - Logs calculation for transparency
        """
        param_manager = self._get_parameter_manager()

        # Get active parameter names (already excludes fixed parameters)
        active_params = param_manager.get_active_parameters()

        # Get bounds for active parameters
        bounds_list = param_manager.get_parameter_bounds(active_params)

        # Calculate mid-points
        midpoint_dict: dict[str, float] = {}
        for bound_dict in bounds_list:
            param_name = bound_dict["name"]
            min_val = bound_dict["min"]
            max_val = bound_dict["max"]
            midpoint = (min_val + max_val) / 2.0
            midpoint_dict[param_name] = midpoint

        logger.info(
            f"Calculated mid-point defaults for {len(midpoint_dict)} parameters"
        )
        logger.debug(f"Mid-point values: {midpoint_dict}")

        return midpoint_dict

    def validate_per_angle_scaling(self, n_phi: int) -> list[str]:
        """Validate per-angle scaling array lengths against number of phi angles.

        This method should be called after loading phi angles from data to verify
        that the per_angle_scaling arrays in the config match the actual number
        of angles in the data.

        Parameters
        ----------
        n_phi : int
            Number of phi angles in the loaded data.

        Returns
        -------
        list[str]
            List of validation warnings (empty if all valid).

        Raises
        ------
        ValueError
            If per-angle scaling arrays have incorrect length and cannot be used.

        Examples
        --------
        >>> config_mgr = ConfigManager("config.yaml")
        >>> warnings = config_mgr.validate_per_angle_scaling(n_phi=5)
        >>> if warnings:
        ...     for w in warnings:
        ...         logger.warning(w)
        """
        warnings: list[str] = []

        if not self.config:
            return warnings

        initial_params = self.config.get("initial_parameters", {})
        per_angle_scaling = initial_params.get("per_angle_scaling")

        if not per_angle_scaling or not isinstance(per_angle_scaling, dict):
            return warnings

        contrast_values = per_angle_scaling.get("contrast")
        offset_values = per_angle_scaling.get("offset")

        # Validate contrast array length
        if contrast_values is not None and isinstance(contrast_values, list):
            n_contrast = len(contrast_values)
            if n_contrast != n_phi and n_contrast != 1:
                raise ValueError(
                    f"per_angle_scaling.contrast has {n_contrast} values but data has "
                    f"{n_phi} phi angles. Must have either 1 (scalar) or {n_phi} values."
                )
            if n_contrast == 1 and n_phi > 1:
                warnings.append(
                    f"per_angle_scaling.contrast has 1 value but data has {n_phi} angles. "
                    f"Using scalar contrast for all angles."
                )

        # Validate offset array length
        if offset_values is not None and isinstance(offset_values, list):
            n_offset = len(offset_values)
            if n_offset != n_phi and n_offset != 1:
                raise ValueError(
                    f"per_angle_scaling.offset has {n_offset} values but data has "
                    f"{n_phi} phi angles. Must have either 1 (scalar) or {n_phi} values."
                )
            if n_offset == 1 and n_phi > 1:
                warnings.append(
                    f"per_angle_scaling.offset has 1 value but data has {n_phi} angles. "
                    f"Using scalar offset for all angles."
                )

        # Cross-check contrast and offset array lengths
        if (
            contrast_values is not None
            and offset_values is not None
            and isinstance(contrast_values, list)
            and isinstance(offset_values, list)
        ):
            n_contrast = len(contrast_values)
            n_offset = len(offset_values)
            if n_contrast != n_offset and n_contrast > 1 and n_offset > 1:
                warnings.append(
                    f"per_angle_scaling arrays have different lengths: "
                    f"contrast={n_contrast}, offset={n_offset}. This may cause issues."
                )

        if warnings:
            for w in warnings:
                logger.warning(w)

        return warnings

    def get_cmc_config(self) -> dict[str, Any]:
        """Get CMC (Consensus Monte Carlo) configuration with validation and defaults.

        Extracts and validates the CMC configuration section from the optimization
        settings. Applies default values for missing fields and validates ranges
        and backend compatibility.

        Returns
        -------
        dict
            CMC configuration dictionary with validated settings including:
            - enable: bool or "auto"
            - min_points_for_cmc: int
            - sharding: dict with strategy, num_shards, max_points_per_shard
            - backend: dict with name, checkpoint settings
            - combination: dict with method, validation settings
            - per_shard_mcmc: dict with num_warmup, num_samples, etc.
            - validation: dict with convergence criteria

        Raises
        ------
        ValueError
            If required CMC fields are invalid or incompatible with hardware

        Examples
        --------
        >>> config_mgr = ConfigManager("cmc_config.yaml")
        >>> cmc_config = config_mgr.get_cmc_config()
        >>> print(cmc_config["sharding"]["strategy"])
        'stratified'

        Notes
        -----
        - Automatically applies sensible defaults for missing fields
        - Validates value ranges (e.g., num_shards > 0)
        - Checks backend compatibility with detected hardware
        - Logs migration warnings for deprecated settings
        """
        if not self.config:
            return self._get_default_cmc_config()

        optimization = self.config.get("optimization", {})
        cmc_raw = optimization.get("cmc", {})

        # If no CMC config, return defaults
        if not cmc_raw:
            logger.debug("No CMC configuration found, using defaults")
            return self._get_default_cmc_config()

        # Start with defaults and override with user settings
        cmc_config = self._get_default_cmc_config()
        self._merge_cmc_config(cmc_config, cmc_raw)

        # Validate the configuration
        self._validate_cmc_config(cmc_config)

        # Check for deprecated settings
        self._check_cmc_deprecated_settings(optimization)

        return cmc_config

    def _get_default_cmc_config(self) -> dict[str, Any]:
        """Get default CMC configuration.

        T052: Logs default value application at DEBUG level.

        Returns
        -------
        dict
            Default CMC configuration with sensible defaults
        """
        # T052: Log default value application
        logger.debug("Applying default CMC configuration values")
        return {
            "enable": "auto",
            "min_points_for_cmc": 100000,
            "sharding": {
                "strategy": "random",
                "num_shards": "auto",
                "max_points_per_shard": "auto",
            },
            "backend": {
                "name": "auto",
                "enable_checkpoints": True,
                "checkpoint_frequency": 10,
                "checkpoint_dir": "./checkpoints/cmc",
                "keep_last_checkpoints": 3,
                "resume_from_checkpoint": True,
            },
            "combination": {
                "method": "robust_consensus_mc",
                "validate_results": True,
                "min_success_rate": 0.90,
                "min_success_rate_warning": 0.80,
            },
            # Per-shard NUTS defaults are tuned to keep
            # laminar_flow CMC workloads below the 2 hour
            # per-shard timeout on typical CPU nodes.
            # These values are intentionally lighter than
            # early prototypes (fewer chains / samples).
            "per_shard_mcmc": {
                "num_warmup": 500,
                "num_samples": 1500,
                "num_chains": 4,
                "target_accept_prob": 0.85,
                "subsample_size": "auto",
            },
            "validation": {
                "strict_mode": True,
                "min_per_shard_ess": 100.0,
                "max_per_shard_rhat": 1.1,
                "max_between_shard_kl": 2.0,
                "min_success_rate": 0.90,
                "max_divergence_rate": 0.10,
                "require_nlsq_warmstart": False,
                "use_nlsq_informed_priors": True,
                "nlsq_prior_width_factor": 2.0,
                "max_parameter_cv": 1.0,
                "heterogeneity_abort": True,
            },
        }

    def _merge_cmc_config(self, defaults: dict[str, Any], user: dict[str, Any]) -> None:
        """Merge user CMC configuration into defaults (recursive).

        Parameters
        ----------
        defaults : dict
            Default configuration dictionary (modified in place)
        user : dict
            User-provided configuration to merge
        """
        for key, value in user.items():
            if (
                key in defaults
                and isinstance(defaults[key], dict)
                and isinstance(value, dict)
            ):
                # Recursive merge for nested dictionaries
                self._merge_cmc_config(defaults[key], value)
            else:
                # Direct override for non-dict values
                defaults[key] = value

    def _validate_cmc_config(self, cmc_config: dict[str, Any]) -> None:
        """Validate CMC configuration values.

        Parameters
        ----------
        cmc_config : dict
            CMC configuration to validate

        Raises
        ------
        ValueError
            If configuration values are invalid
        """
        # Validate enable field
        enable = cmc_config.get("enable")
        if enable not in [True, False, "auto"]:
            raise ValueError(
                f"CMC enable must be True, False, or 'auto', got: {enable}"
            )

        # Validate min_points_for_cmc
        min_points = cmc_config.get("min_points_for_cmc", 0)
        if not isinstance(min_points, int) or min_points < 1:
            raise ValueError(
                f"min_points_for_cmc must be a positive integer (>= 1), got: {min_points}"
            )

        # Validate sharding
        sharding = cmc_config.get("sharding", {})
        strategy = sharding.get("strategy", "stratified")
        if strategy not in ["stratified", "random", "contiguous"]:
            raise ValueError(
                f"Sharding strategy must be 'stratified', 'random', or 'contiguous', got: {strategy}"
            )

        num_shards = sharding.get("num_shards", "auto")
        if num_shards != "auto" and (
            not isinstance(num_shards, int) or num_shards <= 0
        ):
            raise ValueError(
                f"num_shards must be 'auto' or positive integer, got: {num_shards}"
            )

        # Note: initialization config section is deprecated in v2.1.0
        # CMC now uses identity mass matrix by default (no SVI initialization)

        # Validate backend (handle both old dict schema and new string schema)
        backend = cmc_config.get("backend", {})

        # Handle new schema: backend is a string ("jax" or "numpy") for computational backend
        # vs old schema: backend is a dict with name key for parallel execution backend
        if isinstance(backend, str):
            # New schema: computational backend as string
            valid_computational_backends = ["jax", "numpy"]
            if backend not in valid_computational_backends:
                raise ValueError(
                    f"Computational backend must be one of {valid_computational_backends}, got: {backend}"
                )

            # Check for new backend_config field (parallel execution)
            backend_config = cmc_config.get("backend_config", {})
            if backend_config:
                backend_name = backend_config.get("name", "auto")
                valid_parallel_backends = [
                    "auto",
                    "pjit",
                    "multiprocessing",
                    "pbs",
                    "slurm",
                    "jax",  # legacy alias, mapped to pjit downstream
                ]
                if backend_name not in valid_parallel_backends:
                    raise ValueError(
                        f"Parallel execution backend must be one of {valid_parallel_backends}, got: {backend_name}"
                    )
        else:
            # Old schema: backend is dict with name for parallel execution
            backend_name = backend.get("name", "auto")
            valid_backends = [
                "auto",
                "pjit",
                "multiprocessing",
                "pbs",
                "slurm",
                "jax",  # legacy alias, mapped to pjit downstream
            ]
            if backend_name not in valid_backends:
                raise ValueError(
                    f"Backend name must be one of {valid_backends}, got: {backend_name}"
                )

        # Validate combination
        combination = cmc_config.get("combination", {})
        comb_method = combination.get("method", "robust_consensus_mc")
        valid_methods = [
            "consensus_mc",
            "robust_consensus_mc",
            "weighted_gaussian",
            "simple_average",
            "auto",
        ]
        if comb_method not in valid_methods:
            raise ValueError(
                f"Combination method must be one of {valid_methods}, got: {comb_method}"
            )

        min_success = combination.get("min_success_rate", 0.9)
        if not isinstance(min_success, (int, float)) or not 0.0 <= min_success <= 1.0:
            raise ValueError(
                f"min_success_rate must be between 0.0 and 1.0, got: {min_success}"
            )

        # Validate per_shard_mcmc
        per_shard = cmc_config.get("per_shard_mcmc", {})
        for key in ["num_warmup", "num_samples", "num_chains"]:
            value = per_shard.get(key, 1)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"per_shard_mcmc.{key} must be a positive integer, got: {value}"
                )

        # Validate validation settings
        validation = cmc_config.get("validation", {})
        ess = validation.get("min_per_shard_ess", 100)
        if not isinstance(ess, (int, float)) or ess < 0:
            raise ValueError(f"min_per_shard_ess must be non-negative, got: {ess}")

        rhat = validation.get("max_per_shard_rhat", 1.1)
        if not isinstance(rhat, (int, float)) or rhat < 1.0:
            raise ValueError(f"max_per_shard_rhat must be >= 1.0, got: {rhat}")

        logger.debug("CMC configuration validation passed")

    def _check_cmc_deprecated_settings(self, optimization: dict[str, Any]) -> None:
        """Check for deprecated CMC settings and log warnings.

        Parameters
        ----------
        optimization : dict
            Optimization section of configuration
        """
        # Check for old CMC keys that might have been used in early prototypes
        deprecated_keys = {
            "consensus_monte_carlo": "Use 'cmc' instead of 'consensus_monte_carlo'",
            "parallel_mcmc": "Parallel MCMC is now configured via 'cmc.backend'",
        }

        for old_key, message in deprecated_keys.items():
            if old_key in optimization:
                logger.warning(
                    f"Deprecated CMC configuration key '{old_key}' detected. {message}"
                )

        # Check for deprecated sharding keys
        cmc = optimization.get("cmc", {})
        sharding = cmc.get("sharding", {})
        if "optimal_shard_size" in sharding:
            logger.warning(
                "Deprecated sharding key 'optimal_shard_size' detected. "
                "Use 'max_points_per_shard' instead."
            )

    def _validate_config(self) -> None:
        """Lightweight configuration validation.

        Checks for required sections and valid values.
        Can be disabled by setting HOMODYNE_VALIDATE_CONFIG=false environment variable.

        T051: Logs key configuration values at INFO level.
        T052: Logs default value applications at DEBUG level.
        T053: Logs unusual settings as warnings.
        """
        _KNOWN_TOP_LEVEL_KEYS = {
            "metadata",
            "analysis_mode",
            "analyzer_parameters",
            "analysis_settings",
            "experimental_data",
            "phi_filtering",
            "initial_parameters",
            "parameter_space",
            "optimization",
            "noise_estimation",
            "performance",
            "logging",
            "quality_control",
            "plotting",
            "output",
            "validation",
            "config_version",
        }

        if not self.config:
            logger.warning("Configuration is empty")
            return

        # Warn about unknown top-level keys (possible typos)
        unknown_keys = set(self.config.keys()) - _KNOWN_TOP_LEVEL_KEYS
        if unknown_keys:
            logger.warning(
                "Unknown top-level config keys (possible typo): %s", unknown_keys
            )

        # Check for required sections
        required_sections = ["analysis_mode"]
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing recommended section: {section}")

        # Validate analysis_mode value
        valid_modes = ["static", "laminar_flow"]
        mode = self.config.get("analysis_mode", "")
        if mode and mode not in valid_modes:
            logger.warning(
                f"Unknown analysis_mode: '{mode}'. Valid modes: {valid_modes}",
            )

        # T051: Log key configuration values at INFO level
        self._log_key_config_values()

        # T053: Log unusual but valid settings with warnings
        self._log_unusual_settings()

        logger.debug("Configuration validation completed")

    def _log_key_config_values(self) -> None:
        """T051: Log key configuration values at INFO level.

        Logs analysis mode, dataset info, and optimizer selection.
        """
        if not self.config:
            return

        # Analysis mode
        mode = self.config.get("analysis_mode", "unknown")
        logger.info(f"Analysis mode: {mode}")

        # Dataset info
        exp_data = self.config.get("experimental_data", {})
        file_path = exp_data.get("file_path")
        if file_path:
            logger.info(f"Data file: {file_path}")

        # Optimizer selection
        optimization = self.config.get("optimization", {})
        method = optimization.get("method", "nlsq")
        logger.info(f"Optimizer: {method}")

        # Log dataset size estimate if available
        nlsq_config = optimization.get("nlsq", {})
        memory_fraction = nlsq_config.get("memory_fraction")
        if memory_fraction:
            logger.debug(f"Memory fraction: {memory_fraction}")
            if not (0 < memory_fraction < 1):
                logger.warning(
                    "memory_fraction=%s outside valid range (0, 1); should be between 0 and 1",
                    memory_fraction,
                )

    def _log_unusual_settings(self) -> None:
        """T053: Log unusual but valid settings with impact warnings.

        Warns about settings that may have unexpected effects.
        """
        if not self.config:
            return

        optimization = self.config.get("optimization", {})

        # Warn about very high iteration limits
        nlsq_config = optimization.get("nlsq", {}) or optimization.get("lsq", {})
        max_iter = nlsq_config.get("max_iterations", 10000)
        if max_iter > 50000:
            logger.warning(
                f"High max_iterations ({max_iter}) may cause long runtimes. "
                f"Consider 10000-20000 for most analyses."
            )

        # Warn about very loose tolerance
        tolerance = nlsq_config.get("tolerance", 1e-8)
        if tolerance > 1e-4:
            logger.warning(
                f"Loose tolerance ({tolerance}) may produce imprecise results. "
                f"Consider 1e-8 or tighter for production."
            )

        # Warn about very tight tolerance
        if tolerance < 1e-14:
            logger.warning(
                f"Very tight tolerance ({tolerance}) may cause convergence issues. "
                f"Machine precision limits apply."
            )

        # Warn about force_stratified_ls with large datasets
        force_stratified = nlsq_config.get("force_stratified_ls", False)
        if force_stratified:
            logger.warning(
                "force_stratified_ls=True enabled. "
                "This uses full Jacobian (high memory) - ensure sufficient RAM."
            )

        # Warn about disabled anti-degeneracy for laminar_flow
        mode = self.config.get("analysis_mode", "static")
        anti_deg = nlsq_config.get("anti_degeneracy", {})
        if mode == "laminar_flow":
            hierarchical = anti_deg.get("hierarchical", {})
            if hierarchical.get("enable") is False:
                logger.warning(
                    "hierarchical.enable=False for laminar_flow may cause "
                    "gradient cancellation issues with many phi angles."
                )

    def _normalize_schema(self) -> None:
        """Normalize configuration schema for backward compatibility.

        Handles multiple configuration format versions by converting
        legacy formats to modern standardized formats transparently.
        """
        if not self.config:
            return

        self._normalize_analysis_mode()
        self._normalize_experimental_data()
        self._validate_config_version()

    def _normalize_analysis_mode(self) -> None:
        """Normalize analysis_mode to canonical lowercase form.

        Handles case-insensitive input and legacy mode names:
        - "STATIC", "Static" → "static"
        - "LAMINAR_FLOW", "Laminar_Flow" → "laminar_flow"
        - "static_isotropic" → "static" (legacy alias)
        - "static_anisotropic" → "static" (legacy alias)
        """
        if self.config is None or "analysis_mode" not in self.config:
            return

        mode = self.config["analysis_mode"]
        if not isinstance(mode, str):
            return

        original_mode = mode
        normalized_mode = mode.lower()

        # Handle legacy aliases
        if normalized_mode in ("static_isotropic", "static_anisotropic"):
            normalized_mode = "static"

        if normalized_mode != original_mode:
            self.config["analysis_mode"] = normalized_mode
            logger.debug(
                f"Normalized analysis_mode: '{original_mode}' -> '{normalized_mode}'"
            )

    def _validate_config_version(self) -> None:
        """Validate config_version against package version.

        Warns if config version doesn't match package version, which may
        indicate incompatible configuration schema.
        """
        if self.config is None or "metadata" not in self.config:
            return

        config_version = self.config["metadata"].get("config_version")
        if not config_version:
            return

        # Get package version
        try:
            from xpcsjax import __version__ as package_version

            # Extract major.minor for comparison (ignore patch)
            def get_major_minor(version: str) -> str:
                parts = version.split(".")
                if len(parts) >= 2:
                    return f"{parts[0]}.{parts[1]}"
                return version

            config_mm = get_major_minor(str(config_version))
            package_mm = get_major_minor(str(package_version))

            if config_mm != package_mm:
                logger.warning(
                    f"Config version mismatch: config={config_version}, "
                    f"package={package_version}. Configuration schema may be incompatible."
                )
        except ImportError:
            # Package version not available, skip validation
            pass

    def _normalize_experimental_data(self) -> None:
        """Normalize experimental_data section.

        Supports two formats:
        1. Template/Legacy: data_folder_path + data_file_name
        2. Modern: file_path

        The normalization adds the missing format while preserving
        the original fields for backward compatibility.
        """
        if self.config is None or "experimental_data" not in self.config:
            return

        from pathlib import Path

        exp_data = self.config["experimental_data"]

        # Handle legacy composite format (data_folder_path + data_file_name)
        if "data_folder_path" in exp_data and "data_file_name" in exp_data:
            folder_path = exp_data["data_folder_path"]
            filename = exp_data["data_file_name"]

            # Skip normalization if either value is None
            if folder_path is None or filename is None:
                logger.debug(
                    "Skipping normalization: data_folder_path or data_file_name is None",
                )
                return

            folder = Path(folder_path)

            # Resolve relative paths for consistency
            # Note: Keep as-is if already absolute to preserve user intent
            file_path = folder / filename

            # Add modern format while preserving legacy fields
            exp_data["file_path"] = str(file_path)
            logger.info(
                f"Normalized legacy config format:\n"
                f"   {folder} + {filename}\n"
                f"   -> file_path: {file_path}",
            )

        # Handle phi angles similarly
        if "phi_angles_path" in exp_data and "phi_angles_file" in exp_data:
            phi_folder = Path(exp_data["phi_angles_path"])
            phi_file = exp_data["phi_angles_file"]
            phi_path = phi_folder / phi_file

            # Add combined path for convenience
            exp_data["phi_angles_full_path"] = str(phi_path)
            logger.debug(f"Normalized phi angles path: {phi_path}")


def load_xpcs_config(config_path: str) -> dict[str, Any]:
    """Load XPCS configuration from file.

    Convenience function for loading configuration files.

    Parameters
    ----------
    config_path : str
        Path to configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    manager = ConfigManager(config_path)
    return manager.config if manager.config is not None else {}
