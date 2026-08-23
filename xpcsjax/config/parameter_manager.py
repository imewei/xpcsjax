"""Parameter manager for xpcsjax.

Centralized parameter management system for handling parameter bounds,
active parameters, and validation for the xpcsjax NLSQ analysis pipeline.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode, get_registry
from xpcsjax.config.types import (
    PARAMETER_NAME_MAPPING,
    SCALING_PARAM_NAMES,
    BoundDict,
    HomodyneConfig,
    coerce_finite_float,
)

if TYPE_CHECKING:
    # Type-only import: xpcsjax.core.physics pulls in JAX, so it must not load at
    # import time. Under `from __future__ import annotations` the `-> ValidationResult`
    # return annotations are strings, so this TYPE_CHECKING import satisfies type
    # checkers without loading JAX. `validate_parameters_detailed` is NOT imported
    # here — it is used only at runtime, via the lazy local import added in Step 6
    # (importing it here too would be an unused-import F401).
    from xpcsjax.core.physics import ValidationResult

from xpcsjax.utils.logging import get_logger

# Import physics validators for constraint checking
try:
    from xpcsjax.config.physics_validators import (
        ConstraintSeverity,
        validate_all_parameters,
    )

    HAS_PHYSICS_VALIDATORS = True
except ImportError:
    HAS_PHYSICS_VALIDATORS = False

    # Fallback severity class if module not available. The real class is
    # imported from ``xpcsjax.config.physics_validators`` in the try branch
    # above; mypy correctly flags the conditional redefinition as a name
    # collision. The shapes are intentionally identical (both expose ``ERROR``)
    # so callers don't need to branch — the type: ignore acknowledges the
    # well-defined fallback contract.
    class ConstraintSeverity:  # type: ignore[no-redef]
        """Severity levels for physics constraint violations."""

        ERROR = "error"
        WARNING = "warning"
        INFO = "info"


logger = get_logger(__name__)


class ParameterManager:
    """Centralized parameter management system.

    Handles:
    - Parameter bounds (with config override support)
    - Active parameters selection
    - Parameter validation
    - Default parameter values
    - Parameter name mapping

    Parameters
    ----------
    config_dict : dict, optional
        Configuration dictionary. If None, uses hardcoded defaults.
    analysis_mode : str, optional
        Analysis mode ('static_anisotropic', 'static_isotropic', 'laminar_flow'). Auto-detected from config if not provided.

    Examples
    --------
    >>> pm = ParameterManager(config_dict, analysis_mode="laminar_flow")
    >>> bounds = pm.get_parameter_bounds(["D0", "alpha", "D_offset"])
    >>> active = pm.get_active_parameters()
    """

    def __init__(
        self,
        config_dict: HomodyneConfig | dict[str, Any] | None = None,
        analysis_mode: AnalysisMode = AnalysisMode.LAMINAR_FLOW,
    ):
        """Initialize ParameterManager."""
        # ``HomodyneConfig`` is a TypedDict, which mypy doesn't auto-collapse
        # to ``dict[str, Any]`` even though the runtime shape is identical.
        # Cast explicitly so the rest of this class sees a uniform dict.
        self.config_dict: dict[str, Any] = dict(config_dict) if config_dict else {}
        self.analysis_mode = analysis_mode

        # Performance caching for repeated queries.
        # ``cache_key = tuple(parameter_names)`` — order-preserving (NOT
        # sorted): the bounds list is order-sensitive, so ["D0","alpha"] and
        # ["alpha","D0"] must cache separately.
        self._bounds_cache: dict[tuple[str, ...], list[BoundDict]] = {}
        self._active_params_cache: list[str] | None = None
        self._cache_enabled: bool = True

        # Default bounds for all known parameters.
        # Sourced from the ParameterRegistry (single source of truth) — never
        # redeclare numeric bounds here. To change a bound, edit
        # ``xpcsjax/config/parameter_registry.py``.
        self._default_bounds: dict[str, BoundDict] = self._build_default_bounds()

        # Parameter name aliases/mappings (use constant from types)
        self._param_name_mapping = PARAMETER_NAME_MAPPING

        # Load config bounds if available
        self._load_config_bounds()

        # tied_parameters is a real, fully-implemented feature ONLY for
        # two_component (see xpcsjax/config/heterodyne_parameter_space.py --
        # ParameterSpace here has no tied/DOF-reduction mechanism at all).
        # ConfigManager._validate_config only checks unknown TOP-LEVEL keys,
        # so a tied_parameters entry under initial_parameters would otherwise
        # be silently accepted and silently ignored -- the parameter stays
        # independently free instead of being constrained, a scientifically
        # different (and unintended) fit with zero warning. Reject loudly
        # instead of a config bypass. Checked on ANY truthy value (not just a
        # well-formed dict): a malformed tied_parameters (e.g. a YAML list
        # typo) must not silently bypass this guard either -- shape
        # validation is heterodyne_parameter_space.py's job for the mode
        # that actually supports the key, not a reason to skip this reject.
        initial_params = self.config_dict.get("initial_parameters")
        if isinstance(initial_params, dict):
            tied_raw = initial_params.get("tied_parameters")
            if tied_raw and not self._mode_is_two_component(self.analysis_mode):
                raise ValueError(
                    f"initial_parameters.tied_parameters is set ({tied_raw!r}) "
                    f"but analysis_mode={self.analysis_mode!r} does not support "
                    "parameter tying -- tied_parameters is implemented only for "
                    "two_component. Remove tied_parameters, or use "
                    "fixed_parameters/active_parameters to constrain this mode's "
                    "parameters instead."
                )

    @staticmethod
    def _build_default_bounds() -> dict[str, BoundDict]:
        """Materialize default bounds from the ParameterRegistry.

        The ParameterRegistry is the single source of truth for parameter
        ranges. This helper translates every registered parameter into the
        ``BoundDict`` shape that ParameterManager exposes downstream, so the
        manager never carries its own (potentially divergent) numeric
        constants. To change a bound, edit ``parameter_registry.py``.

        Returns
        -------
        dict[str, BoundDict]
            Mapping from parameter name to ``{min, max, name}``.
        """
        registry = get_registry()
        result: dict[str, BoundDict] = {}
        for name, info in registry._PARAMETERS.items():
            result[name] = {
                "min": float(info.lower_bound),
                "max": float(info.upper_bound),
                "name": name,
            }
        return result

    def _load_config_bounds(self) -> None:
        """Load parameter bounds from configuration and merge with defaults."""
        if not self.config_dict:
            return

        param_space = self.config_dict.get("parameter_space") or {}
        if not isinstance(param_space, dict):
            return
        if "bounds" not in param_space:
            return

        config_bounds = param_space["bounds"]
        if not isinstance(config_bounds, list):
            logger.warning("parameter_space.bounds must be a list, ignoring")
            return

        # Merge config bounds with defaults
        for bound_dict in config_bounds:
            if not isinstance(bound_dict, dict):
                continue

            # Work on a shallow copy so we neither mutate the caller's config
            # dicts (the float() coercion below) nor alias the same dict object
            # into _default_bounds (which would couple later internal bound
            # edits back into the user's config and vice versa).
            bound_dict = dict(bound_dict)

            raw_name = bound_dict.get("name")
            if not raw_name or not isinstance(raw_name, str):
                continue

            # Apply name mapping. ``.get`` returns Any from a dict[str, Any]
            # mapping; narrow back to str so the BoundDict indexing below
            # type-checks cleanly.
            param_name: str = self._param_name_mapping.get(raw_name, raw_name)

            # Convert min/max to floats (handles YAML string parsing like
            # "1e5"). A null bound ("min:" left blank in YAML) is treated as
            # unspecified and dropped so the registry default is preserved,
            # instead of raising TypeError from float(None). A non-finite bound
            # (NaN/±inf) is rejected at the coercion boundary — it would
            # otherwise poison the (lower, upper) arrays handed to the solver.
            for _bound_key in ("min", "max"):
                if _bound_key in bound_dict:
                    if bound_dict[_bound_key] is None:
                        del bound_dict[_bound_key]
                    else:
                        bound_dict[_bound_key] = coerce_finite_float(
                            bound_dict[_bound_key],
                            context=f"parameter_space.bounds[{param_name!r}].{_bound_key}",
                        )

            # Update default bounds with config values. ``bound_dict`` is
            # typed ``dict[Any, Any]`` because it came from a YAML parse;
            # cast to the BoundDict TypedDict shape so mypy can match the
            # _default_bounds entries.
            bound_typed = cast(BoundDict, bound_dict)
            if param_name in self._default_bounds:
                self._default_bounds[param_name].update(bound_typed)
                # Ensure name is canonical after mapping
                self._default_bounds[param_name]["name"] = param_name
            else:
                # New parameter not in defaults: require min/max so the registry
                # doesn't "know" an incomplete parameter and defer a confusing raw
                # KeyError('min') to a much later get_bounds_as_arrays() call.
                missing = [k for k in ("min", "max") if k not in bound_typed]
                if missing:
                    raise ValueError(
                        f"parameter_space.bounds entry for unregistered parameter "
                        f"{param_name!r} is missing required key(s): {missing}. "
                        f"New parameters must specify both 'min' and 'max'."
                    )
                self._default_bounds[param_name] = bound_typed

            # Validate after the merge, not on the raw entry: a one-sided
            # override (only 'min' given) can invert against the registry
            # default 'max'. Mirrors ParameterInfo.__post_init__ — equal
            # bounds are allowed (a fixed parameter), min > max is not.
            merged = self._default_bounds[param_name]
            if merged["min"] > merged["max"]:
                raise ValueError(
                    f"parameter_space.bounds entry for {raw_name!r}: "
                    f"min {merged['min']} exceeds max {merged['max']}."
                )

        logger.debug(f"Loaded bounds from config for {len(config_bounds)} parameters")

    def _extract_base_param_name(self, name: str) -> str | None:
        """Extract base parameter name from indexed parameter names.

        Handles per-angle parameter names in both the bracket form
        ('contrast[0]', 'offset[15]', used by ``parameter_space.py``) and the
        underscore form ('contrast_0', 'offset_15', produced by
        ``ConfigManager.get_initial_parameters()`` and used throughout the
        rest of the codebase).

        Parameters
        ----------
        name : str
            Parameter name, possibly indexed like 'contrast[0]' or 'contrast_0'.

        Returns
        -------
        str or None
            Base parameter name ('contrast', 'offset') or None if not a pattern match.
        """
        # Match contrast[N]/offset[N] or contrast_N/offset_N, N a non-negative integer.
        match = re.match(r"^(contrast|offset)(?:\[\d+\]|_\d+)$", name)
        if match:
            return match.group(1)
        return None

    def validate_physical_constraints(
        self,
        params: dict[str, float],
        severity_level: str = "warning",
    ) -> ValidationResult:
        """Validate physics-based parameter constraints beyond simple bounds.

        Checks for physically impossible or unusual parameter values based on
        theoretical understanding of XPCS and soft matter dynamics.

        Uses registry-driven validation from physics_validators module for
        reduced cyclomatic complexity and improved maintainability.

        Parameters
        ----------
        params : dict[str, float]
            Parameter dictionary with parameter_name: value pairs
        severity_level : str
            Minimum severity to report: "error", "warning", or "info"
            - "error": Only physically impossible values
            - "warning": Unusual but possible values (default)
            - "info": All noteworthy observations

        Returns
        -------
        ValidationResult
            Validation result with severity-categorized violations

        Examples
        --------
        >>> pm = ParameterManager()
        >>> params = {"D0": 1000.0, "alpha": 1.5, "gamma_dot_t0": -0.001}
        >>> result = pm.validate_physical_constraints(params)
        >>> if not result.valid:
        ...     print(result.violations)
        ['alpha = 1.50: strongly superdiffusive (α > 1 is rare, check if intended)',
         'gamma_dot_t0 = -0.001: negative shear rate (physically impossible)']

        References
        ----------
        - Subdiffusion (α < 0): Höfling & Franosch, Rep. Prog. Phys. 76, 046602 (2013)
        - XPCS theory: He et al., PNAS 121, e2401162121 (2024)
        """
        from xpcsjax.core.physics import ValidationResult

        if HAS_PHYSICS_VALIDATORS:
            # Use registry-driven validation (reduced complexity)
            physics_violations = validate_all_parameters(params, ConstraintSeverity(severity_level))
            violations = [v.format() for v in physics_violations]
        else:
            # Fallback to inline validation
            violations = self._validate_physical_constraints_fallback(params, severity_level)

        # Create validation result
        is_valid = len(violations) == 0
        if is_valid:
            message = (
                f"Physics constraints validated successfully ({len(params)} parameters checked)"
            )
        else:
            message = f"Physics validation found {len(violations)} issue(s)"

        return ValidationResult(
            valid=is_valid,
            violations=violations,
            parameters_checked=len(params),
            message=message,
        )

    def _validate_physical_constraints_fallback(
        self,
        params: dict[str, float],
        severity_level: str = "warning",
    ) -> list[str]:
        """Fallback validation when physics_validators module not available."""
        violations: list[str] = []
        severity_priority = {"error": 3, "warning": 2, "info": 1}
        min_priority = severity_priority.get(severity_level, 2)

        def add_violation(param: str, value: float, message: str, severity: str) -> None:
            if severity_priority.get(severity, 0) >= min_priority:
                violations.append(f"{param} = {value:.3e}: {message} [{severity}]")

        # Diffusion parameters
        if "D0" in params:
            D0 = params["D0"]
            if D0 <= 0:
                add_violation("D0", D0, "non-positive diffusion coefficient", "error")
            elif D0 > 1e7:
                add_violation("D0", D0, "extremely large diffusion coefficient", "warning")

        if "alpha" in params:
            alpha = params["alpha"]
            if alpha < -1.5:
                add_violation("alpha", alpha, "very strongly subdiffusive", "warning")
            elif alpha > 1.0:
                add_violation("alpha", alpha, "strongly superdiffusive", "warning")
            elif -0.1 < alpha < 0.1:
                add_violation("alpha", alpha, "near-normal diffusion", "info")

        if "D_offset" in params and params["D_offset"] < 0:
            add_violation("D_offset", params["D_offset"], "negative offset", "warning")

        # Shear flow parameters
        if "gamma_dot_t0" in params:
            gamma_dot = params["gamma_dot_t0"]
            if gamma_dot < 0:
                add_violation("gamma_dot_t0", gamma_dot, "negative shear rate", "error")
            elif gamma_dot > 0.5:
                add_violation("gamma_dot_t0", gamma_dot, "very high shear rate", "warning")
            elif 0 < gamma_dot < 1e-6:
                add_violation("gamma_dot_t0", gamma_dot, "very low shear rate", "info")

        if "beta" in params and (params["beta"] < -2.0 or params["beta"] > 2.0):
            add_violation("beta", params["beta"], "time exponent outside range", "warning")

        if "phi0" in params and abs(params["phi0"]) > 10.0:
            add_violation("phi0", params["phi0"], "flow angle outside [-10, 10] deg", "info")

        # Scaling parameters
        if "contrast" in params:
            c = params["contrast"]
            if c <= 0 or c > 1.0:
                add_violation("contrast", c, "contrast outside (0, 1]", "error")
            elif c < 0.1:
                add_violation("contrast", c, "very low contrast", "warning")

        if "offset" in params and params["offset"] <= 0:
            add_violation("offset", params["offset"], "non-positive baseline", "error")

        # Cross-parameter constraints
        if all(k in params for k in ["D0", "alpha", "D_offset"]):
            if params["D0"] > 0 and params["D_offset"] > 0.5 * params["D0"]:
                ratio = params["D_offset"] / params["D0"]
                add_violation(
                    "D_offset",
                    params["D_offset"],
                    f"offset is {ratio:.1%} of D0",
                    "info",
                )

        return violations

    def get_parameter_bounds(
        self,
        parameter_names: list[str] | None = None,
    ) -> list[BoundDict]:
        """Get parameter bounds configuration (with caching for performance).

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
        >>> pm = ParameterManager(config_dict)
        >>> bounds = pm.get_parameter_bounds(["D0", "alpha"])
        >>> bounds[0]["name"]
        'D0'

        Notes
        -----
        Results are cached for performance. Repeated calls with the same
        parameter_names will return cached results instantly.
        """
        if parameter_names is None:
            # Get all parameters for current mode
            parameter_names = self.get_all_parameter_names()

        # Create cache key — preserve order: bounds list is order-sensitive, so
        # (["D0", "alpha"]) and (["alpha", "D0"]) must cache separately.
        cache_key = tuple(parameter_names)

        # Check cache first (if caching enabled)
        if self._cache_enabled and cache_key in self._bounds_cache:
            logger.debug(
                f"Returning cached bounds for {len(parameter_names)} parameters",
            )
            # Deep-copy each BoundDict entry, not just the list container, so a
            # caller mutating a returned bound cannot corrupt the cache for
            # subsequent callers (mirrors the cold path at line 424).
            return [b.copy() for b in self._bounds_cache[cache_key]]

        # Apply name mapping
        mapped_names = [self._param_name_mapping.get(name, name) for name in parameter_names]

        # Get bounds for each parameter
        bounds_list = []
        for name in mapped_names:
            # Check direct match first
            if name in self._default_bounds:
                bounds_list.append(self._default_bounds[name].copy())
            else:
                # Handle per-angle parameter names like contrast[0], offset[1], etc.
                base_name = self._extract_base_param_name(name)
                if base_name and base_name in self._default_bounds:
                    bound_copy = self._default_bounds[base_name].copy()
                    bound_copy["name"] = name  # Keep the indexed name
                    bounds_list.append(bound_copy)
                else:
                    # Registry-as-source-of-truth: parameters not registered in
                    # ParameterRegistry are a programming error rather than a
                    # silent default. Surface the problem instead of inventing
                    # bounds that downstream solvers will silently consume.
                    raise KeyError(
                        f"Unknown parameter '{name}': not in ParameterRegistry "
                        f"and not a recognized per-angle scaling name "
                        f"(e.g. 'contrast[0]', 'offset[3]'). Register the "
                        f"parameter in xpcsjax/config/parameter_registry.py.",
                    )

        # Cache the result
        if self._cache_enabled:
            self._bounds_cache[cache_key] = [b.copy() for b in bounds_list]

        return bounds_list

    def get_active_parameters(self) -> list[str]:
        """Get list of active (physical) parameters from configuration (cached).

        This returns only the physical parameters (excludes scaling parameters
        like contrast and offset).

        Returns
        -------
        list of str
            List of parameter names to be optimized. Falls back to mode-appropriate
            parameters if not specified in config.

        Examples
        --------
        >>> pm = ParameterManager(config_dict, "laminar_flow")
        >>> pm.get_active_parameters()
        ['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta', 'gamma_dot_t_offset', 'phi0']

        Notes
        -----
        Results are cached after first call for performance.
        """
        # Check cache first
        if self._cache_enabled and self._active_params_cache is not None:
            logger.debug("Returning cached active parameters")
            return self._active_params_cache.copy()

        # Compute active parameters
        if not self.config_dict:
            active_params = self._get_default_active_parameters()
        else:
            # Try to get from initial_parameters section
            initial_params = self.config_dict.get("initial_parameters") or {}
            if not isinstance(initial_params, dict):
                initial_params = {}

            # Check for explicit active_parameters list
            active_params_config = initial_params.get("active_parameters")
            if active_params_config is not None and isinstance(active_params_config, list):
                # Apply name mapping. ``.get(name, name)`` returns the mapped
                # name when present, the original ``name`` otherwise — never
                # None. mypy can't see the fallback guarantee through
                # ``dict.get``'s typing, so coerce to str explicitly.
                # Strip scaling names (contrast/offset, incl. per-angle
                # contrast_N/offset_N): this method's contract is
                # physics-only (see docstring, and service/persist.py's
                # "physics-only by design" note) — a caller listing
                # contrast/offset here to mean "vary in the fit" must not
                # leak them past this filter, or callers building a params
                # array from these names against a physics-only initial
                # value dict (e.g. simulated-plot rendering) KeyError.
                active_params = [
                    str(self._param_name_mapping.get(name, name))
                    for name in active_params_config
                    if name not in SCALING_PARAM_NAMES
                    and self._extract_base_param_name(name) is None
                ]
            else:
                # Fall back to parameter_names from initial_parameters
                param_names = initial_params.get("parameter_names")
                if param_names and isinstance(param_names, list):
                    # Apply name mapping (see comment above on the str() coerce).
                    active_params = [
                        str(self._param_name_mapping.get(name, name)) for name in param_names
                    ]
                    # Guard against a config whose analysis_mode was changed
                    # (e.g. --mode two_component on the CLI) without updating
                    # initial_parameters.parameter_names to match: silently
                    # handing the stale, wrong-mode names downstream produces
                    # a wrong-length parameter vector. Fall back to mode
                    # defaults on any name outside the current mode's set.
                    try:
                        valid_names = set(
                            get_registry().get_param_names(AnalysisMode(self.analysis_mode))
                        )
                    except ValueError:
                        valid_names = None
                    if valid_names is not None and not set(active_params) <= valid_names:
                        logger.warning(
                            "initial_parameters.parameter_names %s does not match "
                            "analysis_mode %r's parameter set; falling back to mode defaults",
                            active_params,
                            self.analysis_mode,
                        )
                        active_params = self._get_default_active_parameters()
                else:
                    # Ultimate fallback to mode defaults
                    active_params = self._get_default_active_parameters()

        # Cache the result
        if self._cache_enabled:
            self._active_params_cache = active_params.copy()

        return active_params

    def _get_default_active_parameters(self) -> list[str]:
        """Get default active parameters based on analysis mode."""
        # The registry is the single source of truth for every mode's parameter
        # set. Substring dispatch (not a direct registry lookup on
        # ``self.analysis_mode``) because ConfigManager defers mode validation:
        # this fallback must survive a mode string the registry would reject.
        mode = self.analysis_mode.lower()
        if "static" in mode:
            return get_registry().get_param_names(AnalysisMode.STATIC_ANISOTROPIC)
        if self._mode_is_two_component(mode):
            return get_registry().get_param_names(AnalysisMode.TWO_COMPONENT)
        return get_registry().get_param_names(AnalysisMode.LAMINAR_FLOW)

    @staticmethod
    def _mode_is_two_component(mode: str) -> bool:
        """Substring-match ``mode`` against two_component's known synonyms.

        Deliberately NOT ``AnalysisMode.parse(mode) == AnalysisMode.TWO_COMPONENT``
        or a strict ``==`` -- ConfigManager defers ``analysis_mode`` validation
        (see class docstring / :meth:`_get_default_active_parameters`), so this
        must tolerate a mode string the registry would reject rather than
        raising. Single source of truth for the synonym set within this class;
        shared by :meth:`_get_default_active_parameters` and the
        ``tied_parameters`` cross-mode guard in ``__init__``.
        """
        mode_lower = str(mode).lower()
        return (
            "two_component" in mode_lower
            or "two-component" in mode_lower
            or "heterodyne" in mode_lower
        )

    def get_all_parameter_names(self) -> list[str]:
        """Get all parameter names including scaling parameters.

        Returns
        -------
        list of str
            Complete list of parameter names (scaling + physical)

        Examples
        --------
        >>> pm = ParameterManager(config_dict, "laminar_flow")
        >>> pm.get_all_parameter_names()
        ['contrast', 'offset', 'D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta', ...]
        """
        # Scaling parameters first (canonical order), then physical parameters.
        # De-duplicate: if user config lists scaling params in active_parameters,
        # don't emit them twice.
        scaling_set = set(SCALING_PARAM_NAMES)
        all_params = SCALING_PARAM_NAMES.copy()
        for p in self.get_active_parameters():
            if p not in scaling_set:
                all_params.append(p)
        return all_params

    def get_effective_parameter_count(self) -> int:
        """Get the effective number of physical parameters (excludes scaling).

        Returns
        -------
        int
            Number of physical parameters used in the analysis:

            - Static mode: 3 (D0, alpha, D_offset)
            - Laminar flow mode: 7 (D0, alpha, D_offset, gamma_dot_t0, beta,
              gamma_dot_t_offset, phi0)

        Examples
        --------
        >>> pm = ParameterManager(config_dict, "static_anisotropic")
        >>> pm.get_effective_parameter_count()
        3
        """
        return len(self.get_active_parameters())

    def get_total_parameter_count(self) -> int:
        """Get total number of parameters including scaling parameters.

        Returns
        -------
        int
            Total parameter count (scaling + physical)

        Examples
        --------
        >>> pm = ParameterManager(config_dict, "laminar_flow")
        >>> pm.get_total_parameter_count()
        9
        """
        return len(self.get_all_parameter_names())

    def validate_parameters(
        self,
        params: np.ndarray,
        param_names: list[str] | None = None,
        tolerance: float = 1e-10,
    ) -> ValidationResult:
        """Validate parameter values against bounds.

        Parameters
        ----------
        params : np.ndarray
            Parameter array to validate
        param_names : list of str, optional
            Parameter names. If None, uses all parameter names for current mode.
        tolerance : float
            Tolerance for bounds checking (default: 1e-10)

        Returns
        -------
        ValidationResult
            Detailed validation result

        Examples
        --------
        >>> pm = ParameterManager(config_dict)
        >>> params = np.array([0.5, 1.0, 1000.0, 0.5, 10.0])
        >>> result = pm.validate_parameters(params, ["contrast", "offset", "D0", "alpha", "D_offset"])
        >>> if not result.valid:
        ...     print(result.violations)
        """
        from xpcsjax.core.physics import validate_parameters_detailed

        if param_names is None:
            param_names = self.get_all_parameter_names()

        # Get bounds for these parameters
        bounds_list_dict = self.get_parameter_bounds(param_names)

        # Convert to tuple format for validation
        bounds_tuples = [(bound_dict["min"], bound_dict["max"]) for bound_dict in bounds_list_dict]

        # Use the detailed validation from physics module
        result = validate_parameters_detailed(
            params,
            bounds_tuples,
            param_names=param_names,
            tolerance=tolerance,
        )

        return result

    def get_bounds_as_tuples(
        self,
        parameter_names: list[str] | None = None,
    ) -> list[tuple[float, float]]:
        """Get parameter bounds as list of (min, max) tuples.

        Convenience method for compatibility with optimization code.

        Parameters
        ----------
        parameter_names : list of str, optional
            Parameter names. If None, uses all parameter names for current mode.

        Returns
        -------
        list of tuple
            List of (min, max) tuples

        Examples
        --------
        >>> pm = ParameterManager(config_dict)
        >>> pm.get_bounds_as_tuples(["D0", "alpha"])
        [(1.0, 1000000.0), (-2.0, 2.0)]
        """
        bounds_dicts = self.get_parameter_bounds(parameter_names)
        return [(b["min"], b["max"]) for b in bounds_dicts]

    def get_bounds_as_arrays(
        self,
        parameter_names: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds as separate lower and upper arrays.

        Convenience method for compatibility with optimization code.

        Parameters
        ----------
        parameter_names : list of str, optional
            Parameter names. If None, uses all parameter names for current mode.

        Returns
        -------
        lower_bounds : np.ndarray
            Array of lower bounds
        upper_bounds : np.ndarray
            Array of upper bounds

        Examples
        --------
        >>> pm = ParameterManager(config_dict)
        >>> lower, upper = pm.get_bounds_as_arrays(["D0", "alpha"])
        >>> lower
        array([1.e+00, -2.e+00])
        >>> upper
        array([1.e+06, 2.e+00])
        """
        bounds_dicts = self.get_parameter_bounds(parameter_names)
        lower_bounds = np.array([b["min"] for b in bounds_dicts])
        upper_bounds = np.array([b["max"] for b in bounds_dicts])
        return lower_bounds, upper_bounds

    def get_fixed_parameters(self) -> dict[str, float]:
        """Get parameters that should be held fixed during optimization.

        Returns
        -------
        dict[str, float]
            Dictionary of parameter_name: fixed_value pairs

        Examples
        --------
        >>> config = {
        ...     "initial_parameters": {
        ...         "fixed_parameters": {"D_offset": 10.0}
        ...     }
        ... }
        >>> pm = ParameterManager(config)
        >>> pm.get_fixed_parameters()
        {'D_offset': 10.0}
        """
        if not self.config_dict:
            return {}

        initial_params = self.config_dict.get("initial_parameters") or {}
        if not isinstance(initial_params, dict):
            initial_params = {}
        fixed_params = initial_params.get("fixed_parameters", {})

        if not isinstance(fixed_params, dict):
            logger.warning("fixed_parameters must be a dict, ignoring")
            return {}

        return fixed_params

    def is_parameter_active(self, param_name: str) -> bool:
        """Check if a parameter is active (being optimized).

        Parameters
        ----------
        param_name : str
            Parameter name to check

        Returns
        -------
        bool
            True if parameter is active, False if fixed

        Examples
        --------
        >>> pm = ParameterManager(config)
        >>> pm.is_parameter_active("D0")
        True
        >>> pm.is_parameter_active("contrast")  # if fixed
        False
        """
        active_params = self.get_active_parameters()
        fixed_params = self.get_fixed_parameters()

        # Apply name mapping to input parameter
        canonical_name = self._param_name_mapping.get(param_name, param_name)

        # Check if parameter is in fixed list (need to check both config and canonical names)
        is_fixed = False
        for fixed_name in fixed_params.keys():
            fixed_canonical = self._param_name_mapping.get(fixed_name, fixed_name)
            if canonical_name == fixed_canonical or canonical_name == fixed_name:
                is_fixed = True
                break

        # Parameter is active if it's in active list and not fixed
        return canonical_name in active_params and not is_fixed

    def get_optimizable_parameters(self) -> list[str]:
        """Get list of parameters that should be optimized (active - fixed).

        Returns
        -------
        list[str]
            List of parameter names to optimize (excludes fixed parameters)

        Examples
        --------
        >>> config = {
        ...     "initial_parameters": {
        ...         "parameter_names": ["D0", "alpha", "D_offset"],
        ...         "fixed_parameters": {"D_offset": 10.0}
        ...     }
        ... }
        >>> pm = ParameterManager(config)
        >>> pm.get_optimizable_parameters()
        ['D0', 'alpha']
        """
        active_params = self.get_active_parameters()
        fixed_params = self.get_fixed_parameters()

        # Map fixed parameter names to canonical names
        fixed_canonical = set()
        for fixed_name in fixed_params.keys():
            canonical = self._param_name_mapping.get(fixed_name, fixed_name)
            fixed_canonical.add(canonical)

        # Return active parameters that are not fixed
        return [p for p in active_params if p not in fixed_canonical]

    def __repr__(self) -> str:
        """Return a concise string representation of manager state."""
        active_params = self.get_active_parameters()
        fixed_params = self.get_fixed_parameters()
        # Use the canonical set-difference (get_optimizable_parameters) rather
        # than a cardinality subtraction: fixed_parameters may contain scaling
        # or foreign names not present in active_params, which would otherwise
        # yield a wrong or negative count.
        optimizable = len(self.get_optimizable_parameters())

        return (
            f"ParameterManager(mode={self.analysis_mode}, "
            f"active_params={len(active_params)}, "
            f"fixed_params={len(fixed_params)}, "
            f"optimizable={optimizable}, "
            f"total_params={self.get_total_parameter_count()})"
        )
