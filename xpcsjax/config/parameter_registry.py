"""Parameter Registry for xpcsjax Analysis

Centralized parameter registry that eliminates 8x duplication of parameter
definitions across the codebase. Provides:
- Parameter metadata (names, types, bounds, defaults, descriptions)
- Per-angle parameter expansion
- Validation utilities

This module consolidates parameter information that was previously duplicated in:
- backends/multiprocessing.py (worker validation)
- data_prep.py (data preprocessing)
- several test fixtures

Created as part of code quality remediation (Dec 2025).
Addresses code review finding of 8x parameter name duplication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    pass

# T055: Module-level logger for parameter registry
logger = get_logger(__name__)


class AnalysisMode(StrEnum):
    """Canonical XPCS analysis modes.

    ``StrEnum`` (a ``str`` subclass): each member compares equal to and hashes
    identically to its string value, so existing string comparisons, dict
    lookups (e.g. ``_MODE_PARAMS``), and YAML/JSON round-trips keep working
    unchanged — while static checkers now treat the set of modes as closed.
    """

    STATIC_ANISOTROPIC = "static_anisotropic"
    STATIC_ISOTROPIC = "static_isotropic"
    LAMINAR_FLOW = "laminar_flow"
    TWO_COMPONENT = "two_component"

    @classmethod
    def parse(cls, raw: str, *, allow_bare_static: bool = False) -> AnalysisMode:
        """Normalize an arbitrary mode string to a canonical ``AnalysisMode``.

        M-8: single source of truth for synonym handling, so ``ConfigManager``
        and ``ParameterRegistry`` cannot drift apart on what a mode string means.
        Recognizes (case-insensitively): ``heterodyne`` / ``two-component`` →
        ``TWO_COMPONENT``; ``laminar*`` → ``LAMINAR_FLOW``; ``static*isotropic`` →
        ``STATIC_ISOTROPIC``; ``static*anisotropic`` → ``STATIC_ANISOTROPIC``.

        Bare ``"static"`` is ambiguous. It is rejected unless
        ``allow_bare_static=True`` (registry-internal callers), in which case it
        maps to ``STATIC_ANISOTROPIC`` — the angle-resolved drop-in for legacy
        ``"static"`` configs.
        """
        m = raw.lower()
        if "laminar" in m:
            return cls.LAMINAR_FLOW
        if "two_component" in m or "two-component" in m or "heterodyne" in m:
            return cls.TWO_COMPONENT
        if "static" in m and "anisotropic" in m:
            return cls.STATIC_ANISOTROPIC
        if "static" in m and "isotropic" in m:
            return cls.STATIC_ISOTROPIC
        if "static" in m:
            if allow_bare_static:
                return cls.STATIC_ANISOTROPIC
            raise ValueError(
                f"analysis_mode={raw!r} is ambiguous and no longer accepted. "
                "Use 'static_anisotropic' (angle-resolved; drop-in for legacy "
                "'static') or 'static_isotropic' (angle-collapsed) explicitly."
            )
        raise ValueError(
            f"Unknown analysis mode: {raw!r}. Expected one of {[e.value for e in cls]}."
        )


@dataclass(frozen=True)
class ParameterInfo:
    """Metadata for a single parameter.

    Attributes
    ----------
    name : str
        Canonical parameter name
    description : str
        Human-readable description
    dtype : type
        Python/numpy type (float, int, etc.)
    default : float
        Default value for initialization
    lower_bound : float
        Lower bound for optimization
    upper_bound : float
        Upper bound for optimization
    prior_mean : float | None
        Prior mean for Bayesian inference (None for uniform prior)
    prior_std : float | None
        Prior standard deviation (None for uniform prior)
    units : str
        Physical units (e.g., 'Å²/s', 'radians')
    is_scaling : bool
        True if this is a per-angle scaling parameter
    is_physical : bool
        True if this is a physical model parameter
    is_flow : bool
        True if this is a flow-specific parameter
    log_space : bool
        True if parameter should be sampled in log space (e.g., D0)
    """

    name: str
    description: str
    dtype: type = float
    default: float = 1.0
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    prior_mean: float | None = None
    prior_std: float | None = None
    units: str = ""
    is_scaling: bool = False
    is_physical: bool = False
    is_flow: bool = False
    log_space: bool = False
    # Upstream-heterodyne compatibility fields (set by the registry data tables
    # for parameters that came from the heterodyne ParameterInfo schema). The
    # defaults match upstream behavior so xpcsjax-native parameters still work.
    vary_default: bool = True
    group: str = ""

    def __post_init__(self) -> None:
        """Reject inverted bounds at construction (frozen-dataclass safe).

        ``lower_bound > upper_bound`` defines an empty feasible interval and is
        always a definition error; catching it here makes it unrepresentable
        rather than surfacing as a confusing zero-volume bounds failure deep in
        the optimizer.
        """
        if self.lower_bound > self.upper_bound:
            raise ValueError(
                f"ParameterInfo({self.name!r}): lower_bound {self.lower_bound} "
                f"exceeds upper_bound {self.upper_bound}."
            )

    # ------------------------------------------------------------------
    # Upstream-heterodyne alias surface — the ported heterodyne config
    # modules (parameter_space.py, parameter_manager.py) read ``min_bound``,
    # ``max_bound`` and ``unit`` on ParameterInfo. Expose them as read-only
    # aliases so we don't have to dual-name every entry in the registry.
    # ------------------------------------------------------------------
    @property
    def min_bound(self) -> float:
        """Alias for ``lower_bound`` (upstream heterodyne API)."""
        return self.lower_bound

    @property
    def max_bound(self) -> float:
        """Alias for ``upper_bound`` (upstream heterodyne API)."""
        return self.upper_bound

    @property
    def unit(self) -> str:
        """Alias for ``units`` (upstream heterodyne API uses singular)."""
        return self.units

    def validate_value(self, value: float) -> bool:
        """Check if value is within bounds (upstream heterodyne API)."""
        return self.lower_bound <= value <= self.upper_bound

    def clip_value(self, value: float) -> float:
        """Clip value to bounds (upstream heterodyne API)."""
        return min(max(value, self.lower_bound), self.upper_bound)


class ParameterRegistry:
    """Centralized registry of all parameter definitions.

    This class provides a single source of truth for parameter metadata,
    eliminating duplication across the codebase.

    Examples
    --------
    >>> registry = ParameterRegistry()
    >>> registry.get_param_names("static_anisotropic")
    ['D0', 'alpha', 'D_offset']

    >>> registry.get_all_param_names("static_anisotropic", n_angles=3, include_scaling=True)
    ['contrast_0', 'contrast_1', 'contrast_2',
     'offset_0', 'offset_1', 'offset_2',
     'D0', 'alpha', 'D_offset']

    >>> registry.get_bounds("D0")
    (100.0, 100000.0)
    """

    # Singleton instance
    _instance: ParameterRegistry | None = None

    # Parameter definitions
    _PARAMETERS: dict[str, ParameterInfo] = {
        # Scaling parameters (per-angle)
        "contrast": ParameterInfo(
            name="contrast",
            description="Contrast factor for c2 = contrast × c1² + offset",
            default=0.5,
            lower_bound=0.0,
            upper_bound=1.0,
            prior_mean=0.5,
            prior_std=0.25,
            units="",
            is_scaling=True,
        ),
        "offset": ParameterInfo(
            name="offset",
            description="Baseline offset for c2 = contrast × c1² + offset",
            default=1.0,
            lower_bound=0.5,
            upper_bound=1.5,
            prior_mean=1.0,
            prior_std=0.25,
            units="",
            is_scaling=True,
        ),
        # Physical diffusion parameters
        "D0": ParameterInfo(
            name="D0",
            description="Diffusion coefficient amplitude",
            default=1000.0,
            lower_bound=100.0,
            upper_bound=100000.0,
            prior_mean=1000.0,
            prior_std=1000.0,
            units="Å²/s",
            is_physical=True,
            log_space=True,
        ),
        "alpha": ParameterInfo(
            name="alpha",
            description="Anomalous diffusion exponent (α < 0: sub, α > 0: super)",
            default=0.5,
            lower_bound=-2.0,
            upper_bound=2.0,
            prior_mean=0.5,
            prior_std=0.5,
            units="",
            is_physical=True,
        ),
        "D_offset": ParameterInfo(
            name="D_offset",
            description="Baseline diffusion offset",
            default=10.0,
            lower_bound=-1e5,
            upper_bound=1e5,
            prior_mean=10.0,
            prior_std=200.0,
            units="Å²/s",
            is_physical=True,
            log_space=True,
        ),
        # Flow parameters (laminar flow mode only)
        "gamma_dot_t0": ParameterInfo(
            name="gamma_dot_t0",
            description="Shear rate amplitude at t=0",
            default=0.01,
            lower_bound=1e-6,
            upper_bound=0.5,
            prior_mean=0.01,
            prior_std=0.1,
            units="s⁻¹",
            is_physical=True,
            is_flow=True,
            log_space=True,
        ),
        "beta": ParameterInfo(
            name="beta",
            description="Shear rate time exponent (γ̇(t) = γ̇₀ × t^β)",
            default=0.5,
            lower_bound=-2.0,
            upper_bound=2.0,
            prior_mean=0.0,
            prior_std=0.5,
            units="",
            is_physical=True,
            is_flow=True,
        ),
        "gamma_dot_t_offset": ParameterInfo(
            name="gamma_dot_t_offset",
            description="Baseline shear rate offset",
            default=0.0,
            lower_bound=-0.1,
            upper_bound=0.1,
            prior_mean=0.0,
            prior_std=0.02,
            units="s⁻¹",
            is_physical=True,
            is_flow=True,
            log_space=False,
        ),
        "phi0": ParameterInfo(
            name="phi0",
            description="Flow direction angle",
            default=0.0,
            lower_bound=-10.0,
            upper_bound=10.0,
            prior_mean=0.0,
            prior_std=5.0,
            units="degrees",
            is_physical=True,
            is_flow=True,
        ),
        # Heterodyne (two-component) parameters
        # Reference component diffusion
        "D0_ref": ParameterInfo(
            name="D0_ref",
            description="Reference diffusion prefactor",
            default=1e4,
            lower_bound=0.0,
            upper_bound=1e6,
            units="Å²/s",
            is_physical=True,
            log_space=True,
        ),
        "alpha_ref": ParameterInfo(
            name="alpha_ref",
            description="Reference transport exponent",
            default=0.0,
            lower_bound=-2.0,
            upper_bound=2.0,
            units="",
            is_physical=True,
        ),
        "D_offset_ref": ParameterInfo(
            name="D_offset_ref",
            description="Reference diffusion offset",
            default=0.0,
            lower_bound=-1e4,
            upper_bound=1e4,
            units="Å²/s",
            is_physical=True,
        ),
        # Sample component diffusion
        "D0_sample": ParameterInfo(
            name="D0_sample",
            description="Sample diffusion prefactor",
            default=1e4,
            lower_bound=0.0,
            upper_bound=1e6,
            units="Å²/s",
            is_physical=True,
            log_space=True,
        ),
        "alpha_sample": ParameterInfo(
            name="alpha_sample",
            description="Sample transport exponent",
            default=0.0,
            lower_bound=-2.0,
            upper_bound=2.0,
            units="",
            is_physical=True,
        ),
        "D_offset_sample": ParameterInfo(
            name="D_offset_sample",
            description="Sample diffusion offset",
            default=0.0,
            lower_bound=-1e4,
            upper_bound=1e4,
            units="Å²/s",
            is_physical=True,
        ),
        # Velocity family (heterodyne flow)
        "v0": ParameterInfo(
            name="v0",
            description="Velocity amplitude",
            default=1e3,
            lower_bound=0.0,
            upper_bound=1e6,
            units="Å/s",
            is_physical=True,
            is_flow=True,
            log_space=True,
        ),
        "v_beta": ParameterInfo(
            name="v_beta",
            description="Velocity time exponent (renamed from heterodyne docs' `beta`)",
            default=1.0,
            lower_bound=0.0,
            upper_bound=2.0,
            units="",
            is_physical=True,
            is_flow=True,
        ),
        "v_offset": ParameterInfo(
            name="v_offset",
            description="Velocity offset",
            default=0.0,
            lower_bound=-100.0,
            upper_bound=100.0,
            units="Å/s",
            is_physical=True,
            is_flow=True,
        ),
        # Sample fraction polynomial coefficients
        "f0": ParameterInfo(
            name="f0",
            description="Sample fraction coefficient 0",
            default=0.5,
            lower_bound=0.0,
            upper_bound=1.0,
            units="",
            is_physical=True,
        ),
        "f1": ParameterInfo(
            name="f1",
            description="Sample fraction coefficient 1",
            default=0.0,
            lower_bound=-1.0,
            upper_bound=1.0,
            units="",
            is_physical=True,
        ),
        "f2": ParameterInfo(
            name="f2",
            description="Sample fraction coefficient 2",
            default=0.0,
            lower_bound=-1.0,
            upper_bound=1.0,
            units="",
            is_physical=True,
        ),
        "f3": ParameterInfo(
            name="f3",
            description="Sample fraction coefficient 3",
            default=0.0,
            lower_bound=-1.0,
            upper_bound=1.0,
            units="",
            is_physical=True,
        ),
        # Heterodyne flow angle (degrees; renamed from heterodyne docs' `phi0`
        # to avoid name collision with homodyne's phi0). Both use degrees:
        # the kernel computes deg2rad(phi_angle + phi0_het), so phi0_het is a
        # degree-valued offset on the detector angle, matching upstream [-10, 10].
        "phi0_het": ParameterInfo(
            name="phi0_het",
            description="Flow angle (heterodyne; degrees)",
            default=0.0,
            lower_bound=-10.0,
            upper_bound=10.0,
            units="degrees",
            is_physical=True,
        ),
    }

    # Analysis mode definitions
    _MODE_PARAMS: dict[str, list[str]] = {
        "static_anisotropic": ["D0", "alpha", "D_offset"],
        "static_isotropic": ["D0", "alpha", "D_offset"],
        "laminar_flow": [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ],
        "two_component": [
            "D0_ref",
            "alpha_ref",
            "D_offset_ref",
            "D0_sample",
            "alpha_sample",
            "D_offset_sample",
            "v0",
            "v_beta",
            "v_offset",
            "f0",
            "f1",
            "f2",
            "f3",
            "phi0_het",
        ],
    }

    def __new__(cls) -> ParameterRegistry:
        """Singleton pattern - return existing instance if available."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def scaling_names(self) -> tuple[str, ...]:
        """Base names of all scaling parameters (derived from ``is_scaling`` flag).

        Returns a tuple in registration order (contrast, offset) so that
        downstream consumers produce deterministic parameter orderings.
        Cached after first access since ``_PARAMETERS`` is immutable.

        Returns
        -------
        tuple[str, ...]
            e.g. ``("contrast", "offset")``
        """
        try:
            return self._scaling_names_cache
        except AttributeError:
            result = tuple(name for name, info in self._PARAMETERS.items() if info.is_scaling)
            self._scaling_names_cache: tuple[str, ...] = result
            return result

    def get_param_info(self, name: str) -> ParameterInfo:
        """Get parameter metadata.

        Parameters
        ----------
        name : str
            Parameter name (e.g., 'D0', 'contrast')

        Returns
        -------
        ParameterInfo
            Parameter metadata

        Raises
        ------
        KeyError
            If parameter name is unknown
        """
        # Try exact name first
        if name in self._PARAMETERS:
            return self._PARAMETERS[name]

        # Strip per-angle numeric suffix (e.g., contrast_0 -> contrast, offset_12 -> offset)
        base_name = re.sub(r"_\d+$", "", name)

        if base_name not in self._PARAMETERS:
            raise KeyError(
                f"Unknown parameter: {name}. Valid parameters: {list(self._PARAMETERS.keys())}"
            )
        return self._PARAMETERS[base_name]

    # ------------------------------------------------------------------
    # Mapping protocol — compatibility with upstream heterodyne config
    # modules ported into xpcsjax (parameter_space, parameter_manager).
    # ------------------------------------------------------------------
    def __getitem__(self, name: str) -> ParameterInfo:
        """Alias for :meth:`get_param_info` so the registry behaves like a mapping."""
        return self.get_param_info(name)

    def __iter__(self):
        """Iterate over registered parameter names (canonical order)."""
        return iter(self._PARAMETERS)

    def __len__(self) -> int:
        """Number of registered parameters."""
        return len(self._PARAMETERS)

    def __contains__(self, name: str) -> bool:
        """Membership test by parameter name (exact match only)."""
        return name in self._PARAMETERS

    def get_param_names(
        self,
        analysis_mode: AnalysisMode,
    ) -> list[str]:
        """Get physical parameter names for analysis mode.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode: 'static_anisotropic', 'static_isotropic', or 'laminar_flow'

        Returns
        -------
        list[str]
            Physical parameter names (without per-angle scaling)
        """
        mode = self._normalize_mode(analysis_mode)
        return self._MODE_PARAMS[mode].copy()

    def get_all_param_names(
        self,
        analysis_mode: AnalysisMode,
        n_angles: int = 1,
        include_scaling: bool = True,
    ) -> list[str]:
        """Get all parameter names including per-angle scaling.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode
        n_angles : int
            Number of angles for per-angle scaling parameters
        include_scaling : bool
            If True, include contrast_i and offset_i parameters

        Returns
        -------
        list[str]
            Complete parameter names in Bayesian sampling order:
            [contrast_0..n, offset_0..n, D0, alpha, ...]

        Notes
        -----
        The Bayesian sampler requires parameters in EXACT order as model.sample().
        This method returns parameters in the correct order for init_to_value().
        """
        names: list[str] = []

        if include_scaling:
            # Per-angle scaling names derived from is_scaling flag
            for sname in self.scaling_names:
                for i in range(n_angles):
                    names.append(f"{sname}_{i}")

        # Physical parameters LAST
        names.extend(self.get_param_names(analysis_mode))

        return names

    def get_bounds(
        self,
        name: str,
    ) -> tuple[float, float]:
        """Get parameter bounds.

        Parameters
        ----------
        name : str
            Parameter name

        Returns
        -------
        tuple[float, float]
            (lower_bound, upper_bound)
        """
        info = self.get_param_info(name)
        return (info.lower_bound, info.upper_bound)

    def get_all_bounds(
        self,
        analysis_mode: AnalysisMode,
        n_angles: int = 1,
        include_scaling: bool = True,
    ) -> tuple[list[float], list[float]]:
        """Get bounds for all parameters.

        T055: Logs parameter bounds at DEBUG level.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode
        n_angles : int
            Number of angles
        include_scaling : bool
            Include per-angle scaling parameters

        Returns
        -------
        tuple[list[float], list[float]]
            (lower_bounds, upper_bounds) in parameter order
        """
        names = self.get_all_param_names(analysis_mode, n_angles, include_scaling)
        lower = []
        upper = []

        for name in names:
            lb, ub = self.get_bounds(name)
            lower.append(lb)
            upper.append(ub)

        # T055: Log parameter bounds at DEBUG level
        logger.debug(f"Parameter bounds for {analysis_mode} mode ({len(names)} params):")
        # Log physical parameters (not per-angle scaling) for clarity
        physical_params = self.get_param_names(analysis_mode)
        for name in physical_params:
            lb, ub = self.get_bounds(name)
            logger.debug(f"  {name}: [{lb:.4g}, {ub:.4g}]")

        return lower, upper

    def get_defaults(
        self,
        analysis_mode: AnalysisMode,
        n_angles: int = 1,
        include_scaling: bool = True,
    ) -> list[float]:
        """Get default values for all parameters.

        T055: Logs parameter initial values at DEBUG level.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode
        n_angles : int
            Number of angles
        include_scaling : bool
            Include per-angle scaling parameters

        Returns
        -------
        list[float]
            Default values in parameter order
        """
        names = self.get_all_param_names(analysis_mode, n_angles, include_scaling)
        defaults = [self.get_param_info(name).default for name in names]

        # T055: Log initial values at DEBUG level
        logger.debug(f"Default initial values for {analysis_mode} mode ({len(names)} params):")
        physical_params = self.get_param_names(analysis_mode)
        for name in physical_params:
            info = self.get_param_info(name)
            logger.debug(f"  {name}: {info.default:.4g}")

        return defaults

    def get_num_params(
        self,
        analysis_mode: AnalysisMode,
        n_angles: int = 1,
        include_scaling: bool = True,
    ) -> int:
        """Get total number of parameters.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode
        n_angles : int
            Number of angles
        include_scaling : bool
            Include per-angle scaling parameters

        Returns
        -------
        int
            Total parameter count
        """
        return len(self.get_all_param_names(analysis_mode, n_angles, include_scaling))

    def validate_param_values(
        self,
        values: dict[str, float] | list[float],
        analysis_mode: AnalysisMode,
        n_angles: int = 1,
        include_scaling: bool = True,
    ) -> None:
        """Validate parameter values against bounds.

        Parameters
        ----------
        values : dict or list
            Parameter values (dict of name->value or list in order)
        analysis_mode : str
            Analysis mode
        n_angles : int
            Number of angles
        include_scaling : bool
            Include per-angle scaling parameters

        Raises
        ------
        ValueError
            If any value is out of bounds
        """
        names = self.get_all_param_names(analysis_mode, n_angles, include_scaling)

        if isinstance(values, dict):
            for name, value in values.items():
                if name not in names:
                    continue  # Skip unknown parameters
                lb, ub = self.get_bounds(name)
                if value < lb or value > ub:
                    raise ValueError(f"Parameter {name}={value} out of bounds [{lb}, {ub}]")
        else:
            if len(values) != len(names):
                raise ValueError(f"Expected {len(names)} values, got {len(values)}")
            for name, value in zip(names, values, strict=True):
                lb, ub = self.get_bounds(name)
                if value < lb or value > ub:
                    raise ValueError(f"Parameter {name}={value} out of bounds [{lb}, {ub}]")

    def expand_initial_values(
        self,
        initial_values: dict[str, float],
        n_angles: int,
    ) -> dict[str, float]:
        """Expand scalar scaling values to per-angle parameters.

        Parameters
        ----------
        initial_values : dict
            Initial parameter values (may have 'contrast'/'offset' scalars)
        n_angles : int
            Number of angles

        Returns
        -------
        dict[str, float]
            Expanded values with contrast_i and offset_i

        Examples
        --------
        >>> registry = ParameterRegistry()
        >>> registry.expand_initial_values({'contrast': 0.5, 'offset': 1.0, 'D0': 1000}, n_angles=3)
        {'contrast_0': 0.5, 'contrast_1': 0.5, 'contrast_2': 0.5,
         'offset_0': 1.0, 'offset_1': 1.0, 'offset_2': 1.0,
         'D0': 1000}
        """
        result: dict[str, float] = {}

        # Expand scaling parameters (derived from is_scaling flag)
        scaling = self.scaling_names
        for sname in scaling:
            val = initial_values.get(sname, self._PARAMETERS[sname].default)
            for i in range(n_angles):
                result[f"{sname}_{i}"] = val

        # Copy non-scaling parameters as-is
        for key, value in initial_values.items():
            if key not in scaling:
                result[key] = value

        return result

    def _normalize_mode(self, mode: str) -> str:
        """Normalize analysis mode string (registry-internal, lenient).

        Delegates to :meth:`AnalysisMode.parse` — the single normalization
        authority (M-8) — with ``allow_bare_static=True`` so a bare ``"static"``
        still maps to ``static_anisotropic`` for registry-internal callers.
        Returns the canonical ``str`` value (``AnalysisMode`` is a ``str``
        subclass, so the return type and comparisons are unchanged).
        """
        return AnalysisMode.parse(mode, allow_bare_static=True).value


def get_registry() -> ParameterRegistry:
    """Get the global ParameterRegistry instance.

    Returns
    -------
    ParameterRegistry
        Singleton registry instance (guaranteed by ParameterRegistry.__new__)
    """
    return ParameterRegistry()


# Module-level singleton alias for compatibility with upstream heterodyne
# config modules ported into xpcsjax (parameter_space, parameter_manager).
# ``ParameterRegistry.__new__`` makes this idempotent — every call returns the
# same instance.
DEFAULT_REGISTRY: ParameterRegistry = ParameterRegistry()


# Re-export the SCALING_PARAMS proxy from ``heterodyne_scaling_utils`` so the
# ported heterodyne orchestrator (``heterodyne_core``) can import it from the
# same module path the upstream code expects. Lazy import to avoid a circular
# import at module load.
def __getattr__(name: str):  # pragma: no cover - thin import shim
    if name == "SCALING_PARAMS":
        from xpcsjax.core.heterodyne_scaling_utils import SCALING_PARAMS

        return SCALING_PARAMS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Convenience functions that delegate to the singleton


def get_param_names(analysis_mode: AnalysisMode) -> list[str]:
    """Get physical parameter names for analysis mode."""
    return get_registry().get_param_names(analysis_mode)


def get_all_param_names(
    analysis_mode: AnalysisMode,
    n_angles: int = 1,
    include_scaling: bool = True,
) -> list[str]:
    """Get all parameter names including per-angle scaling."""
    return get_registry().get_all_param_names(analysis_mode, n_angles, include_scaling)


def get_bounds(name: str) -> tuple[float, float]:
    """Get parameter bounds."""
    return get_registry().get_bounds(name)


def get_defaults(
    analysis_mode: AnalysisMode,
    n_angles: int = 1,
    include_scaling: bool = True,
) -> list[float]:
    """Get default values for all parameters."""
    return get_registry().get_defaults(analysis_mode, n_angles, include_scaling)


__all__ = [
    "ParameterInfo",
    "ParameterRegistry",
    "get_registry",
    "get_param_names",
    "get_all_param_names",
    "get_bounds",
    "get_defaults",
]
