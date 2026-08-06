"""Configuration management for the xpcsjax NLSQ analysis package.

Loads and normalizes YAML/JSON configuration files and exposes a stable
interface for parameter management, analysis-mode dispatch, and bounds
configuration. The package is NLSQ-only; no Bayesian/MCMC configuration block
is read or honored here.

The public surface is :class:`ConfigManager` (lazy-exported from the top-level
:mod:`xpcsjax` package) plus the :func:`load_xpcs_config` convenience loader.
Parameter names, bounds, and the canonical set of analysis modes are owned by
:mod:`xpcsjax.config.parameter_registry` — this module reads from that registry
rather than defining parameters itself.

See Also
--------
xpcsjax.config.parameter_registry : Single source of truth for parameter
    names, bounds, and the :class:`~xpcsjax.config.parameter_registry.AnalysisMode`
    enum.
xpcsjax.config.parameter_manager.ParameterManager : Resolves active parameters
    and bounds for a given mode; constructed and cached by :class:`ConfigManager`.
"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xpcsjax.config.parameter_registry import AnalysisMode

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
        """Return a stdlib logger (fallback when xpcsjax logging is unavailable)."""
        return logging.getLogger(name)


logger = get_logger(__name__)


class ConfigManager:
    """Load, normalize, and serve an xpcsjax analysis configuration.

    Reads a YAML or JSON configuration file (or accepts an in-memory override
    dict), normalizes legacy schema variants, optionally validates the result,
    and exposes the parsed mapping through the :attr:`config` attribute and a
    set of typed accessors. CPU-only execution; GPU support is out of scope.

    The four canonical analysis modes are ``static_anisotropic``,
    ``static_isotropic``, ``laminar_flow``, and ``two_component`` (the
    members of :class:`~xpcsjax.config.parameter_registry.AnalysisMode`).
    Mode synonyms (e.g. ``heterodyne`` / ``two-component`` → ``two_component``,
    bare ``static`` → ``static_anisotropic``) are canonicalized at construction
    by :meth:`_normalize_analysis_mode`.

    Construction is intentionally lenient: an unknown or malformed
    ``analysis_mode`` is *not* rejected in :meth:`__init__` — it is lowercased,
    a warning is logged, and the invalid value is deferred to the point of use.
    Strict mode validation happens when the value is coerced to the enum, i.e.
    on first access of the :attr:`analysis_mode` property (which calls
    ``AnalysisMode(value)`` and raises :class:`ValueError` for a non-canonical
    string) and inside :class:`~xpcsjax.config.parameter_manager.ParameterManager`
    when bounds or active parameters are requested.

    Parameters
    ----------
    config_file : str, optional
        Path to a YAML (``.yaml`` / ``.yml``) or JSON (``.json``) configuration
        file. Ignored when ``config_override`` is provided. Defaults to
        ``"xpcsjax_config.yaml"``.
    config_override : dict, optional
        Pre-built configuration mapping. When given, the file at
        ``config_file`` is never read; a shallow copy of this dict becomes
        :attr:`config`.

    Attributes
    ----------
    config : dict
        The parsed, normalized configuration mapping. Guaranteed to be a dict
        after construction (never ``None``) — a missing or non-mapping file
        falls back to :meth:`_get_default_config`.
    config_file : str
        The path passed at construction (retained even when an override was
        used).

    Raises
    ------
    FileNotFoundError
        From :meth:`__init__` via :meth:`load_config` when ``config_override``
        is ``None`` and ``config_file`` does not exist. Other parse/IO errors
        fall back to defaults rather than propagating.

    See Also
    --------
    load_xpcs_config : Convenience function returning just the ``config`` dict.
    xpcsjax.config.parameter_registry.AnalysisMode : Canonical analysis-mode enum.
    xpcsjax.config.parameter_manager.ParameterManager : Backs the bounds and
        active-parameter accessors.

    Notes
    -----
    The ``data_type`` field (closed vocabulary ``"aps_old"`` / ``"aps_u"`` in
    :mod:`xpcsjax.config.types`) is *not* consulted or validated here; the data
    loader auto-detects the format from the HDF5 structure.

    Examples
    --------
    >>> cfg = ConfigManager(config_override={"analysis_mode": "laminar_flow"})
    >>> cfg.config["analysis_mode"]
    'laminar_flow'
    >>> cfg.analysis_mode
    <AnalysisMode.LAMINAR_FLOW: 'laminar_flow'>
    >>> cfg.update_config("optimization.method", "nlsq")
    >>> cfg.config["optimization"]["method"]
    'nlsq'
    """

    def __init__(
        self,
        config_file: str = "xpcsjax_config.yaml",
        config_override: dict[str, Any] | None = None,
    ):
        """Initialize the configuration manager.

        Loads from ``config_file`` (via :meth:`load_config`) unless
        ``config_override`` is given, then normalizes the schema and, unless
        disabled by ``XPCSJAX_VALIDATE_CONFIG=false``, runs lightweight
        validation.

        Parameters
        ----------
        config_file : str, optional
            Path to a YAML/JSON configuration file. Ignored when
            ``config_override`` is provided.
        config_override : dict, optional
            In-memory configuration mapping used instead of loading from file.
            A shallow copy is stored as :attr:`config`.

        Raises
        ------
        FileNotFoundError
            When ``config_override`` is ``None`` and ``config_file`` does not
            exist (re-raised from :meth:`load_config`).
        """
        self.config_file = config_file
        # M-1: config is non-optional — always a dict after __init__ (load_config
        # and the override path both guarantee it, and load_config coerces any
        # non-mapping load to defaults). The vestigial ``| None`` and the
        # ``if self.config is None`` guards it forced are gone. Annotated
        # ``dict[str, Any]`` (not the closed ``XpcsConfig`` TypedDict) because
        # update_config assigns dynamic dot-notation keys; ``XpcsConfig`` in
        # config/types.py documents the schema for typed consumers.
        self.config: dict[str, Any] = {}

        # Cache for ParameterManager to avoid repeated instantiation
        self._cached_param_manager: Any | None = None

        # Set by load_config() when the parsed config fails post-load physics
        # validation. The parsed config is still kept as self.config (see
        # load_config's docstring), so callers that care whether validation
        # actually passed before launching a fit must check this flag —
        # otherwise the only signal is a logger.error line.
        self.config_validation_error: Exception | None = None

        if config_override is not None:
            self.config = config_override.copy()
            logger.info("Configuration loaded from override data")
        else:
            self.load_config()

        # Normalize schema for backward compatibility
        self._normalize_schema()

        # Validate config for config_override path (load_config() path validates
        # inside load_config; override path skipped it, so validate here).
        if config_override is not None:
            import os

            if os.environ.get("XPCSJAX_VALIDATE_CONFIG", "true").lower() == "true":
                self._validate_config()

    @property
    def analysis_mode(self) -> "AnalysisMode":
        """Return the validated analysis mode as a typed enum.

        Centralizes the scattered ``config.get("analysis_mode", ...)`` string
        lookups behind one typed accessor. Because
        :class:`~xpcsjax.config.parameter_registry.AnalysisMode` is a
        ``StrEnum``, existing string comparisons keep working.

        Returns
        -------
        AnalysisMode
            The mode from ``config["analysis_mode"]`` coerced to the enum,
            defaulting to ``AnalysisMode.STATIC_ISOTROPIC`` when the key is
            absent.

        Raises
        ------
        ValueError
            If ``config["analysis_mode"]`` holds a non-canonical string (this
            is where an invalid mode deferred from construction is finally
            rejected).
        """
        from xpcsjax.config.parameter_registry import AnalysisMode

        return AnalysisMode(self.config.get("analysis_mode", "static_isotropic"))

    def load_config(self) -> None:
        """Load and parse the YAML/JSON configuration file into :attr:`config`.

        Dispatches on the file extension (``.yaml`` / ``.yml`` → YAML,
        ``.json`` → JSON, anything else → YAML-then-JSON best effort). On a
        successful load the result is normalized to a mapping, and validation
        runs unless ``XPCSJAX_VALIDATE_CONFIG=false``. Parse and IO failures
        (other than a missing file) fall back to :meth:`_get_default_config`.

        Raises
        ------
        FileNotFoundError
            If :attr:`config_file` does not exist. This is re-raised rather
            than silenced so a wrong path is reported instead of producing
            confusing downstream errors from stub defaults.
        ImportError
            If a ``.yaml`` / ``.yml`` file is requested but PyYAML is not
            installed.

        Notes
        -----
        An empty/null file or a non-mapping document (scalar or list) is
        treated as a load failure and replaced with the default config, which
        is what keeps :attr:`config` a non-optional dict.
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

            # A .yaml/.yml file requires PyYAML — fail clearly here rather than
            # falling through to json.load and surfacing a confusing
            # JSONDecodeError on valid YAML (M-5).
            if file_extension in [".yaml", ".yml"] and not (HAS_YAML and yaml_module):
                raise ImportError(
                    f"PyYAML is required to load '{config_path}' (a {file_extension} "
                    "file). Install it with `uv pip install pyyaml`."
                )

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

            # M-1: guarantee config is always a mapping. An empty/null file
            # parses to None; a scalar/list YAML parses to a non-dict. Both fall
            # back to defaults so self.config is never None and never a
            # non-mapping — the invariant that lets it be typed non-optional.
            if not isinstance(self.config, dict):
                logger.warning(
                    "Configuration file '%s' did not parse to a mapping; using defaults",
                    self.config_file,
                )
                self.config = self._get_default_config()
                return

            metadata = self.config.get("metadata")
            if isinstance(metadata, dict):
                version = metadata.get("config_version", "Unknown")
                logger.info(f"Configuration version: {version}")

            # Anchor relative data paths to the config file's directory so the
            # same config loads identically from the CLI (run inside the data
            # folder) and the GUI worker subprocess (run elsewhere).
            self._resolve_data_paths()

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.info("Using default configuration...")
            self.config = self._get_default_config()
            return
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
            return

        # Optional validation (can be disabled via environment variable). Kept
        # OUTSIDE the parse/IO try-block above: a validation-stage error must
        # not fall into the same handler that discards a successfully-parsed
        # user config in favor of stub defaults — only a genuine parse/IO
        # failure should do that.
        self.config_validation_error = None
        if os.environ.get("XPCSJAX_VALIDATE_CONFIG", "true").lower() == "true":
            try:
                self._validate_config()
            except (ValueError, TypeError, KeyError) as e:
                logger.error(
                    f"Configuration validation error: {e}; keeping the parsed "
                    "configuration as-is (validation failure, not a parse failure)"
                )
                self.config_validation_error = e

    # Path keys under ``experimental_data`` that name a file or directory. These
    # are the keys the loader resolves against ``data_folder_path`` / reads
    # directly (see ``xpcsjax/data/xpcs_loader.py``).
    _DATA_PATH_KEYS: tuple[str, ...] = (
        "data_folder_path",
        "cache_file_path",
        "cache_directory",
        "phi_angles_path",
        "file_path",
    )

    def _resolve_data_paths(self) -> None:
        """Anchor relative ``experimental_data`` paths to the config's directory.

        Paths in a config file are interpreted relative to *that file*, not the
        process working directory — so a config that uses ``data_folder_path:
        "./"`` loads the same data whether launched from the CLI (run inside the
        data folder) or from the GUI worker subprocess (a separate process whose
        working directory is the launch dir, not the config dir). Without this,
        ``./`` resolved against the wrong CWD and the loader raised
        ``FileNotFoundError`` even though the data sat beside the config.

        Resolution order per key: expand ``${VARS}`` and a leading ``~``; if the
        result is still a plain relative path, join it onto the config file's
        directory. No-ops (so the rtol=1e-10 parity baselines are untouched):

        - absolute paths (``expandvars``/``expanduser`` are no-ops on them);
        - ``${VARS}`` that stay unresolved — an unset env var is left literal so
          the failure is honest, never silently re-anchored to the config dir;
        - the ``config_override`` path, which has no backing file to anchor to.
        """
        if not self.config_file or not isinstance(self.config, dict):
            return
        exp = self.config.get("experimental_data")
        if not isinstance(exp, dict):
            return
        try:
            base = str(Path(self.config_file).expanduser().resolve().parent)
        except (OSError, ValueError):  # pragma: no cover — defensive
            return

        for key in self._DATA_PATH_KEYS:
            value = exp.get(key)
            if not isinstance(value, str) or not value:
                continue
            expanded = os.path.expanduser(os.path.expandvars(value))
            if "$" in expanded:
                # Unresolved ${VAR}: leave it literal rather than mis-anchor it.
                continue
            if ".." in expanded.replace("\\", "/").split("/"):
                # A ``..`` component is left literal. ``normpath(join(base, ...))``
                # would collapse the ``..`` into an absolute path with no ``..``
                # left, silently defeating the loader's literal-``..`` traversal
                # guard and ``validate_save_path``. These paths were rejected
                # downstream before this anchoring existed, so leaving them
                # untouched preserves that posture (an honest downstream error).
                continue
            if not os.path.isabs(expanded):
                expanded = os.path.normpath(os.path.join(base, expanded))
            exp[key] = expanded

    def _get_default_config(self) -> dict[str, Any]:
        """Build the minimal fallback configuration mapping.

        Returns a minimal configuration that supports the basic analysis modes,
        used whenever loading fails or the file is missing/non-mapping. Logs the
        fallback application at DEBUG level. CPU-only.

        Returns
        -------
        dict
            Default configuration mapping with ``analysis_mode``,
            ``optimization``, ``output``, and ``logging`` sections.
        """
        # T052: Log default value application
        logger.debug("Applying default configuration values (fallback)")
        return {
            "metadata": {
                "config_version": "0.1.2",
                "description": "Default minimal configuration",
            },
            "analysis_mode": "static_anisotropic",
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
        """Return the current configuration mapping.

        Returns
        -------
        dict
            The live :attr:`config` mapping (not a copy).
        """
        return self.config

    def update_config(self, key: str, value: Any) -> None:
        """Set a configuration value addressed by a dot-notation key.

        Intermediate mappings are created as needed, then the cached
        :class:`~xpcsjax.config.parameter_manager.ParameterManager` is
        invalidated because the mutation may change ``analysis_mode``, bounds,
        or active parameters.

        Parameters
        ----------
        key : str
            Configuration key in dot notation, e.g. ``"optimization.method"``.
        value : Any
            New value to assign at ``key``.
        """
        keys = key.split(".")
        config_ref = self.config

        # Navigate to the parent of the target key, creating intermediate
        # mappings as needed. A ``None`` intermediate (an explicit null YAML
        # section such as ``optimization:`` left blank) is treated as absent and
        # replaced with a mapping; a non-dict scalar is a genuine shape conflict
        # and raises a clear error instead of a cryptic TypeError downstream.
        for k in keys[:-1]:
            existing = config_ref.get(k)
            if existing is None:
                config_ref[k] = {}
            elif not isinstance(existing, dict):
                raise TypeError(
                    f"Cannot set {key!r}: intermediate key {k!r} holds a "
                    f"{type(existing).__name__}, not a mapping"
                )
            config_ref = config_ref[k]

        # Set the value
        config_ref[keys[-1]] = value

        # Invalidate cached ParameterManager — config mutations may change
        # analysis_mode, parameter_space.bounds, or active_parameters.
        self._cached_param_manager = None

    def is_static_mode_enabled(self) -> bool:
        """Report whether the configured analysis mode is a static mode.

        Returns
        -------
        bool
            ``True`` if ``config["analysis_mode"]`` contains ``"static"``
            (case-insensitive), or if no configuration is loaded (the static
            default). ``False`` for ``laminar_flow`` and ``two_component``.
        """
        if not self.config:
            return True
        # Delegate to the typed accessor so a non-str / non-canonical analysis_mode
        # raises the same ValueError as everywhere else (the single deferred-
        # validation point) instead of an AttributeError from str.lower().
        return "static" in self.analysis_mode.value.lower()

    def get_model(self) -> Any:
        """Construct the physics model class for this config's analysis mode.

        Thin wrapper over :func:`xpcsjax.core.models.make_model` so that engine
        and test code can build the appropriate model directly from a
        ``ConfigManager`` instance. Routing matches ``make_model``:
        ``two_component`` / ``heterodyne`` → ``HeterodyneModel``; the homodyne
        modes → ``CombinedModel``.

        Returns
        -------
        Any
            The model instance produced by ``make_model`` (a ``HeterodyneModel``
            or ``CombinedModel``).

        See Also
        --------
        xpcsjax.core.models.make_model : The underlying factory.

        Notes
        -----
        The import is lazy because :mod:`xpcsjax.core.models` pulls in JAX and
        the model class hierarchy, which is overkill for callers that only need
        parameter bounds or registry lookups.
        """
        from xpcsjax.core.models import make_model

        return make_model(self)

    def get_target_angle_ranges(self) -> dict[str, Any]:
        """Return the angle-filtering configuration block.

        Returns
        -------
        dict
            The ``optimization.angle_filtering`` mapping, or ``{"enabled":
            False}`` when no config is loaded, the block is absent, or it is not
            a mapping.
        """
        if not self.config:
            return {"enabled": False}

        optimization = self.config.get("optimization", {})
        if not isinstance(optimization, dict):
            return {"enabled": False}
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
            from xpcsjax.config.parameter_registry import AnalysisMode

            # Determine analysis mode (Task 28: include two_component branch).
            # Order: explicit two_component → static (via is_static_mode_enabled)
            # → laminar_flow fallback. Use AnalysisMode members (not raw strings)
            # so the value typechecks at the ParameterManager boundary.
            raw_mode = ""
            if self.config:
                cfg_mode = self.config.get("analysis_mode", "")
                if isinstance(cfg_mode, str):
                    raw_mode = cfg_mode.lower()

            analysis_mode: AnalysisMode
            if (
                "two_component" in raw_mode
                or "two-component" in raw_mode
                or "heterodyne" in raw_mode
            ):
                analysis_mode = AnalysisMode.TWO_COMPONENT
            elif self.is_static_mode_enabled():
                # Preserve an explicit isotropic/anisotropic choice.
                if "anisotropic" in raw_mode:
                    analysis_mode = AnalysisMode.STATIC_ANISOTROPIC
                elif "isotropic" in raw_mode:
                    analysis_mode = AnalysisMode.STATIC_ISOTROPIC
                elif raw_mode == "":
                    # Absent analysis_mode: match the .analysis_mode property's
                    # canonical default of config.get("analysis_mode",
                    # "static_isotropic") so the two resolution paths agree.
                    analysis_mode = AnalysisMode.STATIC_ISOTROPIC
                else:
                    # A bare "static" (and other static-containing strings)
                    # normalizes to the angular-resolved variant, mirroring
                    # AnalysisMode's own normalization.
                    analysis_mode = AnalysisMode.STATIC_ANISOTROPIC
            else:
                analysis_mode = AnalysisMode.LAMINAR_FLOW

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
            One bound dictionary per parameter, with keys ``'name'``, ``'min'``,
            ``'max'``, and ``'type'``.

        Raises
        ------
        TypeError
            If the underlying ``ParameterManager.get_parameter_bounds`` does not
            return a list.

        Examples
        --------
        >>> config_mgr = ConfigManager("config.yaml")
        >>> bounds = config_mgr.get_parameter_bounds(["D0", "alpha"])
        >>> bounds[0]
        {'min': 1.0, 'max': 1000000.0, 'name': 'D0', 'type': 'Normal'}

        See Also
        --------
        xpcsjax.config.parameter_registry : Source of the bound values.

        Notes
        -----
        Uses a cached ParameterManager for ~14x speedup on repeated calls.
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
            Parameter names to be optimized. Falls back to mode-appropriate
            parameters if not specified in config.

        Raises
        ------
        TypeError
            If the underlying ``ParameterManager.get_active_parameters`` does
            not return a list.

        Examples
        --------
        >>> config_mgr = ConfigManager("config.yaml")
        >>> config_mgr.get_active_parameters()
        ['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta', 'gamma_dot_t_offset', 'phi0']

        Notes
        -----
        Uses a cached ParameterManager for ~14x speedup on repeated calls.
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
            if not use_midpoint_defaults:
                raise ValueError(
                    "No initial_parameters section in config and use_midpoint_defaults is False"
                )
            logger.info("No initial_parameters section in config, using mid-point defaults")
            return self._calculate_midpoint_defaults()

        # Get parameter names from config
        param_names_config = initial_params.get("parameter_names")
        if not param_names_config or not isinstance(param_names_config, list):
            if not use_midpoint_defaults:
                raise ValueError(
                    "No parameter_names in initial_parameters and use_midpoint_defaults is False"
                )
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
            raise ValueError(f"initial_parameters.values must be a list, got {type(param_values)}")

        # Validate length match
        if len(param_values) != len(param_names_config):
            raise ValueError(
                f"Number of values ({len(param_values)}) does not match "
                f"number of parameter_names ({len(param_names_config)})"
            )

        # Get ParameterManager for name mapping (used for validation)
        _param_manager = self._get_parameter_manager()  # noqa: F841

        # Import name mapping once at the top of this section
        from xpcsjax.config.types import PARAMETER_NAME_MAPPING, coerce_finite_float

        # Build initial parameters dict with name mapping
        initial_params_dict: dict[str, float] = {}
        for param_name, value in zip(param_names_config, param_values, strict=False):
            # Apply name mapping (e.g., gamma_dot_0 → gamma_dot_t0)
            canonical_name = PARAMETER_NAME_MAPPING.get(param_name, param_name)
            # Reject non-finite (NaN/±inf) — this value becomes the optimizer x0.
            initial_params_dict[canonical_name] = coerce_finite_float(
                value, context=f"initial_parameters.values[{canonical_name!r}]"
            )

        # Load per-angle scaling parameters (contrast, offset) if present.
        # Injected BEFORE the fixed_parameters filter below so contrast/offset
        # keys are excluded the same as physics parameters (otherwise they'd
        # leak past it, contradicting the fixed-parameter exclusion this method
        # promises). They are deliberately EXEMPT from the active_parameters
        # filter: active_parameters is a physics-oriented whitelist that never
        # names contrast/offset (or their per-angle contrast_N/offset_N forms),
        # so subjecting these injected keys to it would silently drop every
        # explicitly-configured per-angle scaling value whenever
        # active_parameters is set.
        per_angle_scaling_keys: set[str] = set()
        per_angle_scaling = initial_params.get("per_angle_scaling")
        if per_angle_scaling and isinstance(per_angle_scaling, dict):
            # Extract contrast and offset arrays
            contrast_values = per_angle_scaling.get("contrast")
            offset_values = per_angle_scaling.get("offset")

            if contrast_values is not None and isinstance(contrast_values, list):
                if len(contrast_values) == 1:
                    # Single-angle: use scalar contrast
                    initial_params_dict["contrast"] = coerce_finite_float(
                        contrast_values[0],
                        context="initial_parameters.per_angle_scaling.contrast[0]",
                    )
                    per_angle_scaling_keys.add("contrast")
                    logger.info(
                        f"Loaded scalar contrast from per_angle_scaling: {contrast_values[0]}"
                    )
                else:
                    # Multi-angle: use per-angle contrast_0, contrast_1, ...
                    for idx, val in enumerate(contrast_values):
                        key = f"contrast_{idx}"
                        initial_params_dict[key] = coerce_finite_float(
                            val,
                            context=f"initial_parameters.per_angle_scaling.contrast[{idx}]",
                        )
                        per_angle_scaling_keys.add(key)
                    logger.info(f"Loaded {len(contrast_values)} per-angle contrast values")

            if offset_values is not None and isinstance(offset_values, list):
                if len(offset_values) == 1:
                    # Single-angle: use scalar offset
                    initial_params_dict["offset"] = coerce_finite_float(
                        offset_values[0],
                        context="initial_parameters.per_angle_scaling.offset[0]",
                    )
                    per_angle_scaling_keys.add("offset")
                    logger.info(f"Loaded scalar offset from per_angle_scaling: {offset_values[0]}")
                else:
                    # Multi-angle: use per-angle offset_0, offset_1, ...
                    for idx, val in enumerate(offset_values):
                        key = f"offset_{idx}"
                        initial_params_dict[key] = coerce_finite_float(
                            val,
                            context=f"initial_parameters.per_angle_scaling.offset[{idx}]",
                        )
                        per_angle_scaling_keys.add(key)
                    logger.info(f"Loaded {len(offset_values)} per-angle offset values")

        # Filter by active_parameters if specified (per-angle-scaling keys are
        # exempt — see the comment at their injection above).
        active_params_config = initial_params.get("active_parameters")
        if active_params_config and isinstance(active_params_config, list):
            # Map active parameter names to canonical names
            active_canonical = set()
            for name in active_params_config:
                canonical = PARAMETER_NAME_MAPPING.get(name, name)
                active_canonical.add(canonical)

            # Filter to only active parameters (plus exempt per-angle-scaling keys)
            initial_params_dict = {
                k: v
                for k, v in initial_params_dict.items()
                if k in active_canonical or k in per_angle_scaling_keys
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

        logger.info(f"Loaded initial parameters from config: {list(initial_params_dict.keys())}")

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

        logger.info(f"Calculated mid-point defaults for {len(midpoint_dict)} parameters")
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

        initial_params = self.config.get("initial_parameters") or {}
        if not isinstance(initial_params, dict):
            return warnings
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
            # Heterodyne grouped config format (two_component)
            "parameters",
            "temporal",
            "scattering",
            "scaling",
        }

        if not self.config:
            logger.warning("Configuration is empty")
            return

        # Warn about unknown top-level keys (possible typos)
        unknown_keys = set(self.config.keys()) - _KNOWN_TOP_LEVEL_KEYS
        if unknown_keys:
            logger.warning("Unknown top-level config keys (possible typo): %s", unknown_keys)

        # Check for required sections
        required_sections = ["analysis_mode"]
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing recommended section: {section}")

        # Validate analysis_mode value against post-normalization canonical set.
        # _normalize_analysis_mode() runs before this in the config_override path
        # and after in the load_config() path, so accept both raw and canonical
        # strings here to avoid false warnings in either path.
        valid_modes = [
            "static_anisotropic",
            "static_isotropic",
            "laminar_flow",
            "two_component",
            "heterodyne",  # accepted raw; _normalize rewrites to "two_component"
            "two-component",  # accepted raw; _normalize rewrites to "two_component"
            "static",  # accepted raw (deprecated); _normalize rewrites to "static_anisotropic"
        ]
        mode = self.config.get("analysis_mode", "")
        mode_normalized = mode.lower() if isinstance(mode, str) else mode
        if mode_normalized and mode_normalized not in valid_modes:
            logger.warning(
                "Unknown analysis_mode: '%s'. Valid modes: %s",
                mode,
                ["static_anisotropic", "static_isotropic", "laminar_flow", "two_component"],
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
        if not isinstance(exp_data, dict):
            exp_data = {}
        file_path = exp_data.get("file_path")
        if file_path:
            logger.info(f"Data file: {file_path}")

        # Optimizer selection
        optimization = self.config.get("optimization", {})
        if not isinstance(optimization, dict):
            logger.warning(
                "optimization must be a dict, ignoring (got %s)", type(optimization).__name__
            )
            optimization = {}
        method = optimization.get("method", "nlsq")
        logger.info(f"Optimizer: {method}")

        # Log dataset size estimate if available
        nlsq_config = optimization.get("nlsq", {})
        if not isinstance(nlsq_config, dict):
            logger.warning(
                "optimization.nlsq must be a dict, ignoring (got %s)", type(nlsq_config).__name__
            )
            nlsq_config = {}
        memory_fraction = nlsq_config.get("memory_fraction")
        if memory_fraction:
            logger.debug(f"Memory fraction: {memory_fraction}")
            # Guard the ordered comparison: a non-numeric value (e.g. a quoted
            # YAML scalar parsed as str) would raise TypeError on `0 < value`.
            if not isinstance(memory_fraction, (int, float)) or isinstance(memory_fraction, bool):
                logger.warning(
                    "memory_fraction=%s is not numeric; should be a float between 0 and 1",
                    memory_fraction,
                )
            elif not (0 < memory_fraction < 1):
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
        if not isinstance(optimization, dict):
            logger.warning(
                "optimization must be a dict, ignoring (got %s)", type(optimization).__name__
            )
            optimization = {}

        # Warn about very high iteration limits
        nlsq_config = optimization.get("nlsq", {}) or optimization.get("lsq", {})
        if not isinstance(nlsq_config, dict):
            logger.warning(
                "optimization.nlsq/lsq must be a dict, ignoring (got %s)",
                type(nlsq_config).__name__,
            )
            nlsq_config = {}
        # Guard the ordered comparisons: a non-numeric value (e.g. a quoted
        # YAML scalar parsed as str) would raise TypeError on `max_iter > ...`,
        # mirroring the memory_fraction guard above (a TypeError here
        # propagates out of _validate_config; on the config_file= path
        # load_config() now catches it separately from parse/IO failures and
        # keeps the parsed config instead of discarding it for stub defaults).
        max_iter = nlsq_config.get("max_iterations", 10000)
        if isinstance(max_iter, (int, float)) and not isinstance(max_iter, bool):
            if max_iter > 50000:
                logger.warning(
                    f"High max_iterations ({max_iter}) may cause long runtimes. "
                    f"Consider 10000-20000 for most analyses."
                )
        else:
            logger.warning("max_iterations=%s is not numeric; ignoring", max_iter)

        # Warn about very loose/tight tolerance
        tolerance = nlsq_config.get("tolerance", 1e-8)
        if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool):
            if tolerance > 1e-4:
                logger.warning(
                    f"Loose tolerance ({tolerance}) may produce imprecise results. "
                    f"Consider 1e-8 or tighter for production."
                )

            if tolerance < 1e-14:
                logger.warning(
                    f"Very tight tolerance ({tolerance}) may cause convergence issues. "
                    f"Machine precision limits apply."
                )
        else:
            logger.warning("tolerance=%s is not numeric; ignoring", tolerance)

        # Warn about force_stratified_ls with large datasets
        force_stratified = nlsq_config.get("force_stratified_ls", False)
        if force_stratified:
            logger.warning(
                "force_stratified_ls=True enabled. "
                "This uses full Jacobian (high memory) - ensure sufficient RAM."
            )

        # Warn about disabled anti-degeneracy for laminar_flow. Normalize the
        # raw analysis_mode string the same way _validate_config's valid_modes
        # check tolerates both orderings (this method may run before or after
        # _normalize_analysis_mode() depending on the load path) — otherwise a
        # case-variant or synonym mode string (e.g. "LAMINAR_FLOW") silently
        # skips this warning even when hierarchical is explicitly disabled.
        mode_raw = self.config.get("analysis_mode", "static_anisotropic")
        if isinstance(mode_raw, str):
            from xpcsjax.config.parameter_registry import AnalysisMode

            try:
                mode = AnalysisMode.parse(mode_raw, allow_bare_static=True).value
            except ValueError:
                mode = mode_raw
        else:
            mode = mode_raw
        anti_deg = nlsq_config.get("anti_degeneracy", {})
        if not isinstance(anti_deg, dict):
            anti_deg = {}
        if mode == "laminar_flow":
            hierarchical = anti_deg.get("hierarchical", {})
            if not isinstance(hierarchical, dict):
                hierarchical = {}
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

        Handles case-insensitive input and synonyms. Consistent with
        :func:`xpcsjax.config.parameter_registry.ParameterRegistry._normalize_mode`:

        - "STATIC_ANISOTROPIC", "Static_Anisotropic" → "static_anisotropic"
        - "STATIC_ISOTROPIC", "Static_Isotropic" → "static_isotropic"
        - "LAMINAR_FLOW", "Laminar_Flow" → "laminar_flow"
        - "HETERODYNE", "Heterodyne", "two-component" → "two_component"

        Bare "static" is accepted as a deprecated alias for
        ``static_anisotropic`` — the angle-resolved drop-in for legacy
        ``"static"`` configs, matching the canonical mapping in
        :meth:`AnalysisMode.parse` (``allow_bare_static=True``). A deprecation
        warning nudges users to choose explicitly; truly unknown modes are
        left as-is and deferred to construction-time validation.
        """
        if "analysis_mode" not in self.config:
            return

        mode = self.config["analysis_mode"]
        if not isinstance(mode, str):
            return

        from xpcsjax.config.parameter_registry import AnalysisMode

        original_mode = mode

        # Bare legacy "static" is a deprecated alias for "static_anisotropic".
        # Warn (don't hard-fail): homodyne configs and the characterization
        # parity oracle legitimately use bare "static", and the registry's
        # single normalization authority already canonicalizes it that way.
        if mode.lower() == "static":
            logger.warning(
                "analysis_mode='static' is deprecated; mapping to "
                "'static_anisotropic' (angle-resolved drop-in). Set "
                "'static_anisotropic' or 'static_isotropic' explicitly to "
                "silence this warning."
            )

        # Canonicalize synonyms (e.g. 'heterodyne' -> 'two_component', bare
        # 'static' -> 'static_anisotropic') via the single normalization
        # authority (M-8). Unknown values are lowercased and deferred to
        # construction-time mode validation rather than raised here.
        try:
            normalized_mode = AnalysisMode.parse(mode, allow_bare_static=True).value
        except ValueError:
            normalized_mode = mode.lower()

        if normalized_mode != original_mode:
            self.config["analysis_mode"] = normalized_mode
            logger.debug(f"Normalized analysis_mode: '{original_mode}' -> '{normalized_mode}'")

    def _validate_config_version(self) -> None:
        """Validate config_version against package version.

        Warns if config version doesn't match package version, which may
        indicate incompatible configuration schema.
        """
        metadata = self.config.get("metadata")
        if not isinstance(metadata, dict):
            return

        config_version = metadata.get("config_version")
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
        if "experimental_data" not in self.config:
            return

        from pathlib import Path

        exp_data = self.config["experimental_data"]
        if not isinstance(exp_data, dict):
            return

        # Handle legacy composite format (data_folder_path + data_file_name)
        if "data_folder_path" in exp_data and "data_file_name" in exp_data:
            folder_path = exp_data["data_folder_path"]
            filename = exp_data["data_file_name"]

            # Compose only when both values are present. Skip just this block on
            # None (do not early-return — phi normalization below is separate
            # and must still run).
            if folder_path is None or filename is None:
                logger.debug(
                    "Skipping normalization: data_folder_path or data_file_name is None",
                )
            else:
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
            phi_path_val = exp_data["phi_angles_path"]
            phi_file_val = exp_data["phi_angles_file"]

            # Mirror the data-folder None guard: a present-but-null path/file
            # would otherwise raise TypeError inside Path()/__truediv__.
            if phi_path_val is None or phi_file_val is None:
                logger.debug(
                    "Skipping phi-angle normalization: phi_angles_path or phi_angles_file is None",
                )
            else:
                phi_path = Path(phi_path_val) / phi_file_val

                # Add combined path for convenience
                exp_data["phi_angles_full_path"] = str(phi_path)
                logger.debug(f"Normalized phi angles path: {phi_path}")


def load_xpcs_config(config_path: str) -> dict[str, Any]:
    """Load an XPCS configuration file and return the parsed mapping.

    Convenience wrapper that constructs a :class:`ConfigManager` and returns its
    :attr:`~ConfigManager.config` dict.

    Parameters
    ----------
    config_path : str
        Path to a YAML/JSON configuration file.

    Returns
    -------
    dict
        The parsed, normalized configuration mapping.

    See Also
    --------
    ConfigManager : The full configuration manager, for typed accessors and
        validation.
    """
    manager = ConfigManager(config_path)
    return manager.config if manager.config is not None else {}
