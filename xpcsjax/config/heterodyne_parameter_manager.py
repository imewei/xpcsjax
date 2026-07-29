"""Parameter manager for heterodyne model optimization."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.config.heterodyne_parameter_names import (
    ALL_PARAM_NAMES,
    ALL_PARAM_NAMES_WITH_SCALING,
    PARAM_GROUPS,
    SCALING_PARAMS,
)
from xpcsjax.config.heterodyne_parameter_space import ParameterSpace, registry_info
from xpcsjax.config.heterodyne_physics_validators import ValidationResult, validate_parameters
from xpcsjax.config.types import BoundDict
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    import jax.numpy as jnp

logger = get_logger(__name__)


@dataclass
class ParameterManager:
    """Manage heterodyne parameter values, constraints, and transformations.

    Provides the bridge between configuration and optimization by:

    - Managing which parameters vary versus are fixed.
    - Handling parameter transformations (e.g. bounded to unbounded).
    - Constructing full parameter arrays from varying subsets.
    - Validating parameter values against physics constraints.

    Bounds and defaults are sourced from the central
    :data:`~xpcsjax.config.parameter_registry.DEFAULT_REGISTRY`; this manager
    never declares its own numeric bounds. Performance caching is enabled by
    default for repeated bound and active-parameter queries.
    """

    space: ParameterSpace = field(default_factory=ParameterSpace)

    # Performance caching — populated lazily via __post_init__
    _bounds_cache: dict[tuple[str, ...], list[BoundDict]] = field(
        default_factory=dict, init=False, repr=False
    )
    _active_params_cache: list[str] | None = field(default=None, init=False, repr=False)
    _cache_enabled: bool = field(default=True, init=False, repr=False)

    # B006: cached index lists (invalidated by set_vary)
    _varying_indices_cache: list[int] | None = field(default=None, init=False, repr=False)
    _fixed_indices_cache: list[int] | None = field(default=None, init=False, repr=False)
    _varying_names_cache: list[str] | None = field(default=None, init=False, repr=False)
    _tied_idx_pairs_cache: list[tuple[int, int]] | None = field(
        default=None, init=False, repr=False
    )

    # B007: cached full-values array (invalidated by update_values)
    _full_values_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    # Frozen snapshot of config-specified initial values — set at construction and
    # never mutated by update_values/set_params (re-seedable only via the explicit
    # reseed_initial_values()). get_initial_values() reads from here so that each
    # phi-angle optimization starts from config values regardless of what a
    # previous fit stored in space.values.
    _initial_values_snapshot: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Build default bounds lookup from the registry, then merge config overrides."""
        self._default_bounds: dict[str, BoundDict] = {}
        for name in ALL_PARAM_NAMES_WITH_SCALING:
            # registry_info resolves kernel names (beta -> v_beta, phi0 -> phi0_het)
            # so the velocity exponent / flow angle source heterodyne bounds.
            info = registry_info(name)
            self._default_bounds[name] = BoundDict(
                name=name,
                min=info.min_bound,
                max=info.max_bound,
                type="TruncatedNormal",
            )

        # Sync _default_bounds with config-overridden bounds from ParameterSpace
        # so that both sources agree (homodyne parity with _load_config_bounds).
        self._sync_bounds_from_space()

        # Freeze initial values so get_initial_values() always returns the config
        # starting point regardless of model.set_params() calls between phi-angle fits.
        self._initial_values_snapshot = copy.deepcopy(self.space.values)

    def _sync_bounds_from_space(self) -> None:
        """Merge config-overridden bounds from ParameterSpace into _default_bounds.

        Homodyne parity: equivalent to ``_load_config_bounds()`` which reads
        ``parameter_space.bounds`` from the config dict and calls ``.update()``
        on the in-memory bounds.  Here the bounds already live in
        ``self.space.bounds`` (populated by ``ParameterSpace.from_config``),
        so we just copy them over.
        """
        for name in ALL_PARAM_NAMES_WITH_SCALING:
            # registry_info applies the heterodyne kernel-name aliases (see above).
            reg = registry_info(name)
            lo, hi = self.space.bounds.get(name, (reg.min_bound, reg.max_bound))
            if name in self._default_bounds:
                if lo != reg.min_bound or hi != reg.max_bound:
                    logger.debug(
                        "Config overrides bounds for %s: [%.4g, %.4g] -> [%.4g, %.4g]",
                        name,
                        reg.min_bound,
                        reg.max_bound,
                        lo,
                        hi,
                    )
                self._default_bounds[name]["min"] = lo
                self._default_bounds[name]["max"] = hi

    # ------------------------------------------------------------------
    # Core existing API
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Total number of physics model parameters (14)."""
        return len(ALL_PARAM_NAMES)

    @property
    def n_varying(self) -> int:
        """Number of physics parameters that vary in optimization."""
        return len(self.varying_names)

    @property
    def varying_names(self) -> list[str]:
        """Names of varying physics parameters (excludes scaling)."""
        if self._varying_names_cache is None:
            self._varying_names_cache = self.space.varying_physics_names
        return list(self._varying_names_cache)

    @property
    def varying_indices(self) -> list[int]:
        """Indices of varying parameters in the 14-element physics array."""
        if self._varying_indices_cache is None:
            self._varying_indices_cache = [
                i for i, name in enumerate(ALL_PARAM_NAMES) if self.space.vary.get(name, False)
            ]
        return list(self._varying_indices_cache)

    @property
    def fixed_indices(self) -> list[int]:
        """Indices of fixed parameters in the 14-element physics array."""
        if self._fixed_indices_cache is None:
            self._fixed_indices_cache = [
                i for i, name in enumerate(ALL_PARAM_NAMES) if not self.space.vary.get(name, False)
            ]
        return list(self._fixed_indices_cache)

    @property
    def tied_idx_pairs(self) -> list[tuple[int, int]]:
        """``(child_index, parent_index)`` pairs in the 14-element physics array.

        Empty when no ``tied_parameters`` are configured. Both indices are
        static Python ints (not JAX tracers) -- safe to close over inside a
        JIT-traced residual closure, exactly like ``varying_indices``.
        """
        if self._tied_idx_pairs_cache is None:
            name_to_idx = {name: i for i, name in enumerate(ALL_PARAM_NAMES)}
            self._tied_idx_pairs_cache = [
                (name_to_idx[child], name_to_idx[parent])
                for child, parent in self.space.tied.items()
            ]
        return list(self._tied_idx_pairs_cache)

    def get_initial_values(self) -> np.ndarray:
        """Get initial parameter values for optimization.

        Returns the config-specified starting point, not the current fitted state.
        Reads from the frozen snapshot (set at construction, re-seedable only via
        ``reseed_initial_values()``) so that repeated calls (e.g. across
        multi-angle loops) always return the same config values even after
        model.set_params() has mutated space.values.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_varying,)`` with initial values for the varying
            parameters.
        """
        full = np.array(
            [
                self._initial_values_snapshot.get(name, self.space.values.get(name, 0.0))
                for name in ALL_PARAM_NAMES
            ]
        )
        return full[self.varying_indices]

    def get_full_values(self) -> np.ndarray:
        """Get all 14 parameter values.

        Returns a read-only cached array (``writeable=False``).
        Use ``.copy()`` if mutation is required.

        Returns
        -------
        numpy.ndarray
            Read-only array of shape ``(14,)``.
        """
        if self._full_values_cache is None:
            arr = self.space.get_initial_array()
            arr.flags.writeable = False
            self._full_values_cache = arr
        return self._full_values_cache

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds for the varying physics parameters.

        Returns
        -------
        tuple of numpy.ndarray
            ``(lower, upper)``, each of shape ``(n_varying,)``.
        """
        lower_full, upper_full = self.space.get_bounds_arrays()
        idx = self.varying_indices
        return lower_full[idx], upper_full[idx]

    def expand_varying_to_full(
        self,
        varying_params: np.ndarray | jnp.ndarray,
    ) -> np.ndarray:
        """Expand varying parameters to full 14-parameter array.

        Fixed parameters are filled from stored values.

        Parameters
        ----------
        varying_params : numpy.ndarray or jax.numpy.ndarray
            Array of shape ``(n_varying,)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(14,)``.

        Raises
        ------
        ValueError
            If ``varying_params`` does not have exactly ``len(self.varying_indices)``
            entries.
        """
        if len(varying_params) != len(self.varying_indices):
            raise ValueError(
                f"Expected {len(self.varying_indices)} varying values, got {len(varying_params)}"
            )
        full = self.get_full_values().copy()
        for i, idx in enumerate(self.varying_indices):
            full[idx] = float(varying_params[i])
        for child_idx, parent_idx in self.tied_idx_pairs:
            full[child_idx] = full[parent_idx]
        return full

    def expand_reduced_result(
        self,
        parameters_reduced: np.ndarray,
        covariance_reduced: np.ndarray | None,
        uncertainties_reduced: np.ndarray | None,
        *,
        n_scaling: int,
        scaling_first: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand a reduced joint optimizer result to the full-14-physics layout.

        The reduced vector is ``[physics_varying | scaling]`` (physics-first,
        e.g. the in-memory joint-fit paths) or ``[scaling | physics_varying]``
        (scaling-first, e.g. the streaming/stratified-LS paths), length
        ``n_varying + n_scaling``; ``covariance_reduced`` is square of that
        size. The scaling block (and all its cross-covariance with physics)
        passes through unchanged; the physics block expands from
        ``n_varying`` to 14, filling any ordinary fixed (non-tied) physics
        row/column with NaN (no covariance was computed for a constant) and
        mirroring each tied child's row/column onto its parent's (same free
        variable, not independently estimated).

        Parameters
        ----------
        parameters_reduced : numpy.ndarray
            Shape ``(n_varying + n_scaling,)``.
        covariance_reduced : numpy.ndarray or None
            Shape ``(n_varying + n_scaling, n_varying + n_scaling)``, or
            ``None`` (e.g. a global-escape result with no covariance solve)
            -> the physics/scaling covariance block of the output is all-NaN.
        uncertainties_reduced : numpy.ndarray or None
            Shape ``(n_varying + n_scaling,)``, or ``None`` -> all-NaN output.
        n_scaling : int
            Number of scaling DOF in the reduced vector (0 for ``constant``,
            2 for ``averaged``, ``2 * n_phi`` for ``individual``).
        scaling_first : bool
            ``True`` for ``[scaling | physics]`` layout, ``False`` for
            ``[physics | scaling]``.

        Returns
        -------
        tuple of numpy.ndarray
            ``(parameters_full, covariance_full, uncertainties_full)`` with
            the physics block expanded to 14 and the scaling block
            unchanged, in the same overall ordering as the input.
        """
        n_varying = len(self.varying_indices)
        reduced = np.asarray(parameters_reduced, dtype=np.float64)
        expected_len = n_varying + n_scaling
        if reduced.shape[0] != expected_len:
            raise ValueError(
                f"expand_reduced_result: parameters_reduced has length "
                f"{reduced.shape[0]}, expected n_varying + n_scaling = "
                f"{n_varying} + {n_scaling} = {expected_len}"
            )
        if covariance_reduced is not None:
            cov_check = np.asarray(covariance_reduced, dtype=np.float64)
            if cov_check.shape != (expected_len, expected_len):
                raise ValueError(
                    f"expand_reduced_result: covariance_reduced has shape "
                    f"{cov_check.shape}, expected ({expected_len}, {expected_len})"
                )
        if uncertainties_reduced is not None:
            unc_check = np.asarray(uncertainties_reduced, dtype=np.float64)
            if unc_check.shape[0] != expected_len:
                raise ValueError(
                    f"expand_reduced_result: uncertainties_reduced has length "
                    f"{unc_check.shape[0]}, expected {expected_len}"
                )

        if scaling_first:
            scaling_vals = reduced[:n_scaling]
            physics_varying = reduced[n_scaling:]
            scaling_pos = list(range(n_scaling))
            physics_pos = list(range(n_scaling, n_scaling + n_varying))
        else:
            physics_varying = reduced[:n_varying]
            scaling_vals = reduced[n_varying:]
            physics_pos = list(range(n_varying))
            scaling_pos = list(range(n_varying, n_varying + n_scaling))

        physics_full = self.expand_varying_to_full(physics_varying)

        if scaling_first:
            parameters_full = np.concatenate([scaling_vals, physics_full])
            full_scaling_pos = list(range(n_scaling))
            full_physics_pos = list(range(n_scaling, n_scaling + 14))
        else:
            parameters_full = np.concatenate([physics_full, scaling_vals])
            full_physics_pos = list(range(14))
            full_scaling_pos = list(range(14, 14 + n_scaling))

        n_full = 14 + n_scaling
        cov_full = np.full((n_full, n_full), np.nan, dtype=np.float64)
        unc_full = np.full(n_full, np.nan, dtype=np.float64)

        if uncertainties_reduced is not None:
            unc_r = np.asarray(uncertainties_reduced, dtype=np.float64)
            for i, _ in enumerate(scaling_pos):
                unc_full[full_scaling_pos[i]] = unc_r[scaling_pos[i]]
            for a, idx in enumerate(self.varying_indices):
                unc_full[full_physics_pos[idx]] = unc_r[physics_pos[a]]

        if covariance_reduced is not None:
            cov_r = np.asarray(covariance_reduced, dtype=np.float64)
            for i in range(n_scaling):
                for j in range(n_scaling):
                    cov_full[full_scaling_pos[i], full_scaling_pos[j]] = cov_r[
                        scaling_pos[i], scaling_pos[j]
                    ]
            for a, ia in enumerate(self.varying_indices):
                fa = full_physics_pos[ia]
                for b, ib in enumerate(self.varying_indices):
                    fb = full_physics_pos[ib]
                    cov_full[fa, fb] = cov_r[physics_pos[a], physics_pos[b]]
                for i in range(n_scaling):
                    fi = full_scaling_pos[i]
                    cov_full[fa, fi] = cov_r[physics_pos[a], scaling_pos[i]]
                    cov_full[fi, fa] = cov_r[scaling_pos[i], physics_pos[a]]

        for child_idx, parent_idx in self.tied_idx_pairs:
            fc = full_physics_pos[child_idx]
            fp = full_physics_pos[parent_idx]
            unc_full[fc] = unc_full[fp]
            cov_full[fc, :] = cov_full[fp, :]
            cov_full[:, fc] = cov_full[:, fp]

        return parameters_full, cov_full, unc_full

    def extract_varying(self, full_params: np.ndarray | jnp.ndarray) -> np.ndarray:
        """Extract the varying parameters from a full array.

        Parameters
        ----------
        full_params : numpy.ndarray or jax.numpy.ndarray
            Array of shape ``(14,)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_varying,)``.

        Raises
        ------
        ValueError
            If ``full_params`` does not have exactly 14 entries.
        """
        if len(full_params) != len(ALL_PARAM_NAMES):
            raise ValueError(f"Expected {len(ALL_PARAM_NAMES)} values, got {len(full_params)}")
        return np.array([full_params[i] for i in self.varying_indices])

    def update_values(self, params: np.ndarray | dict[str, float]) -> None:
        """Update the stored parameter values.

        Does NOT move the frozen snapshot ``get_initial_values()`` reads from —
        use this to record fitted/in-progress state. See also
        :meth:`reseed_initial_values` to also move that snapshot when seeding a
        genuinely new starting point (e.g. one multistart candidate).

        Parameters
        ----------
        params : numpy.ndarray or dict
            Either an array of shape ``(14,)`` or a dict keyed by parameter
            name.
        """
        if isinstance(params, dict):
            self.space.update_from_dict(params)
        else:
            params_dict = self.space.array_to_dict(np.asarray(params))
            self.space.update_from_dict(params_dict)
        # Invalidate full-values cache — values have changed
        self._full_values_cache = None

    def reseed_initial_values(self, params: np.ndarray | dict[str, float]) -> None:
        """Re-seed the starting point that ``get_initial_values()`` returns.

        ``update_values`` deliberately does NOT move the frozen snapshot
        ``get_initial_values()`` reads from (see :meth:`get_initial_values`'s
        docstring) — it exists to record fitted/in-progress state without
        perturbing the config-specified starting point read repeatedly during
        a single fit.
        Use this method instead when the intent is a genuinely NEW starting
        point for a fresh solve (e.g. one multistart candidate); it updates
        both the live values and the frozen snapshot together.

        Parameters
        ----------
        params : numpy.ndarray or dict
            Either an array of shape ``(14,)`` or a dict keyed by parameter
            name.
        """
        self.update_values(params)
        self._initial_values_snapshot = copy.deepcopy(self.space.values)

    def get_parameter_dict(self) -> dict[str, float]:
        """Get current parameter values as dictionary."""
        return dict(self.space.values)

    def set_vary(self, name: str, vary: bool) -> None:
        """Set whether a parameter varies in optimization.

        Invalidates the relevant caches.

        Parameters
        ----------
        name : str
            Parameter name (physics or scaling).
        vary : bool
            Whether to vary this parameter during optimization.

        Raises
        ------
        ValueError
            If ``name`` is not a known parameter.
        """
        if name not in ALL_PARAM_NAMES_WITH_SCALING:
            raise ValueError(f"Unknown parameter: {name}")
        self.space.vary[name] = vary
        # Varying status change affects active/fixed and index caches
        self._active_params_cache = None
        self._varying_names_cache = None
        self._varying_indices_cache = None
        self._fixed_indices_cache = None

    def set_bounds(self, name: str, lower: float, upper: float) -> None:
        """Set bounds for a parameter.

        Invalidates the bounds cache for any query that includes this parameter.

        Parameters
        ----------
        name : str
            Parameter name (physics or scaling).
        lower : float
            Lower bound.
        upper : float
            Upper bound.

        Raises
        ------
        ValueError
            If ``name`` is not a known parameter, or if ``lower > upper``.
        """
        if name not in ALL_PARAM_NAMES_WITH_SCALING:
            raise ValueError(f"Unknown parameter: {name}")
        if lower > upper:
            raise ValueError(f"Inverted bound for {name!r}: lower={lower} > upper={upper}")
        self.space.bounds[name] = (lower, upper)
        # Update the local default_bounds mirror and flush cache
        if name in self._default_bounds:
            self._default_bounds[name]["min"] = lower
            self._default_bounds[name]["max"] = upper
        self._bounds_cache.clear()

    def validate_physics(self, params: np.ndarray | None = None) -> list[str]:
        """Validate parameters against physics constraints.

        Parameters
        ----------
        params : numpy.ndarray, optional
            Full parameter array of shape ``(14,)``, or ``None`` to use the
            stored values.

        Returns
        -------
        list of str
            Violation messages (errors and warnings); empty if valid.
        """
        if params is None:
            params = self.get_full_values()

        result = validate_parameters(params)
        return result.errors + result.warnings

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ParameterManager:
        """Create a ParameterManager from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Full configuration dict.

        Returns
        -------
        ParameterManager
            Configured manager wrapping a ParameterSpace built from ``config``.
        """
        space = ParameterSpace.from_config(config)
        return cls(space=space)

    def get_group_values(self, group: str) -> dict[str, float]:
        """Get parameter values for a specific group.

        Parameters
        ----------
        group : str
            Group name, one of ``'reference'``, ``'sample'``, ``'velocity'``,
            ``'fraction'``, ``'angle'``, or ``'scaling'``.

        Returns
        -------
        dict
            Mapping from parameter name to value for the group.

        Raises
        ------
        ValueError
            If ``group`` is not a recognized group name.
        """
        if group not in PARAM_GROUPS:
            raise ValueError(f"Unknown group: {group}")
        return {name: self.space.values[name] for name in PARAM_GROUPS[group]}

    # ------------------------------------------------------------------
    # New API: bounds queries
    # ------------------------------------------------------------------

    def get_parameter_bounds(
        self,
        parameter_names: list[str] | None = None,
    ) -> list[BoundDict]:
        """Get the parameter bounds configuration, with caching.

        Parameters
        ----------
        parameter_names : list of str, optional
            Names of parameters to retrieve bounds for. If ``None``, returns
            bounds for all 16 parameters (14 physics + 2 scaling) in canonical
            order.

        Returns
        -------
        list of BoundDict
            One entry per requested parameter, with keys ``'name'``, ``'min'``,
            ``'max'``, ``'type'``.

        Raises
        ------
        KeyError
            If a requested name is not a recognized heterodyne parameter.

        Notes
        -----
        Results are cached per unique (order-sensitive) parameter set. The
        cache is invalidated automatically by :meth:`set_bounds`.
        """
        if parameter_names is None:
            parameter_names = list(ALL_PARAM_NAMES_WITH_SCALING)

        # Preserve order: bounds list is order-sensitive.
        cache_key = tuple(parameter_names)

        if self._cache_enabled and cache_key in self._bounds_cache:
            logger.debug("Returning cached bounds for %d parameters", len(parameter_names))
            return [b.copy() for b in self._bounds_cache[cache_key]]  # type: ignore[return-value]

        bounds_list: list[BoundDict] = []
        for name in parameter_names:
            if name in self._default_bounds:
                # Always reflect live space.bounds (may differ from registry defaults
                # if set_bounds() was called)
                lo, hi = self.space.bounds.get(
                    name,
                    (
                        self._default_bounds[name]["min"],
                        self._default_bounds[name]["max"],
                    ),
                )
                bounds_list.append(BoundDict(name=name, min=lo, max=hi, type="TruncatedNormal"))
            else:
                raise KeyError(
                    f"Unknown parameter '{name}': not in ParameterRegistry "
                    f"and not a recognized heterodyne parameter. "
                    f"Valid names: {list(ALL_PARAM_NAMES_WITH_SCALING)}"
                )

        if self._cache_enabled:
            self._bounds_cache[cache_key] = [b.copy() for b in bounds_list]  # type: ignore[misc]

        return bounds_list

    def get_bounds_as_tuples(
        self,
        parameter_names: list[str] | None = None,
    ) -> list[tuple[float, float]]:
        """Get parameter bounds as a list of (min, max) tuples.

        Convenience method for compatibility with optimization code that
        expects the scipy-style bounds format.

        Parameters
        ----------
        parameter_names : list of str, optional
            Parameter names. If ``None``, uses all 16 parameters.

        Returns
        -------
        list of tuple of float
            ``(min, max)`` tuples, one per parameter.
        """
        return [(b["min"], b["max"]) for b in self.get_parameter_bounds(parameter_names)]

    def get_bounds_as_arrays(
        self,
        parameter_names: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds as separate lower and upper numpy arrays.

        Convenience method for NLSQ and JAX optimizers that consume separate
        lower/upper bound arrays.

        Parameters
        ----------
        parameter_names : list of str, optional
            Parameter names. If ``None``, uses all 16 parameters.

        Returns
        -------
        tuple of numpy.ndarray
            ``(lower_bounds, upper_bounds)``, each of shape ``(n_params,)``.
        """
        bd = self.get_parameter_bounds(parameter_names)
        lower = np.array([b["min"] for b in bd])
        upper = np.array([b["max"] for b in bd])
        return lower, upper

    # ------------------------------------------------------------------
    # New API: active / fixed / optimizable parameter queries
    # ------------------------------------------------------------------

    def get_active_parameters(self) -> list[str]:
        """Get physics parameter names that are marked as varying.

        Returns the 14-element physics parameters (excludes scaling) whose
        ``vary`` flag is True in the current ParameterSpace. Every registry
        ``vary_default`` is ``True``, so an empty result reflects an explicit
        all-fixed configuration, not an unconfigured manager — it is returned
        as-is rather than papered over with a full-parameter fallback.

        Results are cached; call :meth:`set_vary` to invalidate automatically.

        Returns
        -------
        list of str
            Varying physics parameter names in canonical order.
        """
        if self._cache_enabled and self._active_params_cache is not None:
            logger.debug("Returning cached active parameters")
            return list(self._active_params_cache)

        active = self.space.varying_physics_names

        if self._cache_enabled:
            self._active_params_cache = list(active)

        return active

    def get_all_parameter_names(self) -> list[str]:
        """Get all parameter names: scaling parameters first, then physics.

        Returns
        -------
        list of str
            The 16 names (``contrast``, ``offset``, then the 14 physics
            parameters) in canonical order.
        """
        return list(SCALING_PARAMS) + list(ALL_PARAM_NAMES)

    def get_effective_parameter_count(self) -> int:
        """Count active (varying) physics parameters, excluding scaling.

        Returns
        -------
        int
            Number of physics parameters whose ``vary`` flag is ``True``.
        """
        return len(self.get_active_parameters())

    def get_total_parameter_count(self) -> int:
        """Get the total parameter count, including scaling and physics.

        Returns
        -------
        int
            Always 16 for the heterodyne model (14 physics + 2 scaling).
        """
        return len(ALL_PARAM_NAMES_WITH_SCALING)

    def get_fixed_parameters(self) -> dict[str, float]:
        """Return physics parameters that are held fixed during optimization.

        A parameter is considered fixed when its ``vary`` flag is False in the
        ParameterSpace.  Scaling parameters (contrast, offset) are excluded
        from this result — use :meth:`get_parameter_dict` to access their values.

        Returns
        -------
        dict
            Mapping from each fixed physics parameter name to its current value.
        """
        return {
            name: self.space.values[name]
            for name in ALL_PARAM_NAMES
            if not self.space.vary.get(name, False)
        }

    def is_parameter_active(self, param_name: str) -> bool:
        """Check whether a physics parameter is active (``vary=True``).

        Parameters
        ----------
        param_name : str
            Physics parameter name to check. Must be one of the 14 physics
            parameters; scaling names always return ``False``.

        Returns
        -------
        bool
            ``True`` if the parameter's ``vary`` flag is ``True``, else
            ``False``.
        """
        if param_name not in ALL_PARAM_NAMES:
            return False
        return bool(self.space.vary.get(param_name, False))

    def get_optimizable_parameters(self) -> list[str]:
        """Return physics parameters that should be optimized.

        Equivalent to active parameters (vary=True). Scaling parameters are
        handled separately and are not included.

        Returns
        -------
        list of str
            Physics parameter names with ``vary=True``, in canonical order.
        """
        return self.get_active_parameters()

    # ------------------------------------------------------------------
    # New API: physics constraint validation with severity
    # ------------------------------------------------------------------

    def validate_physical_constraints(
        self,
        params: dict[str, float] | np.ndarray | None = None,
        severity_level: str = "warning",
    ) -> ValidationResult:
        """Validate physics-based constraints beyond simple bound checking.

        Checks for physically impossible or unusual parameter combinations
        based on the heterodyne two-component scattering model.

        Parameters
        ----------
        params : dict or numpy.ndarray, optional
            Parameter dict, array of shape ``(14,)``, or ``None`` to use the
            stored values. Dict keys must be physics parameter names.
        severity_level : str, default "warning"
            Minimum severity to include in the result, one of ``"error"``
            (physically impossible values only), ``"warning"`` (unusual but
            possible values), or ``"info"`` (all noteworthy observations).
            Currently the heterodyne validator does not distinguish severity
            internally; this argument is accepted for API parity with homodyne
            and is reserved for future use.

        Returns
        -------
        ValidationResult
            Result carrying ``is_valid``, ``errors``, and ``warnings``.
        """
        if params is None:
            arr = self.get_full_values()
        elif isinstance(params, dict):
            arr = self.get_full_values().copy()
            param_dict_full = self.space.array_to_dict(arr)
            param_dict_full.update({k: v for k, v in params.items() if k in param_dict_full})
            arr = np.array([param_dict_full[name] for name in ALL_PARAM_NAMES])
        else:
            arr = np.asarray(params, dtype=float)

        result = validate_parameters(arr)

        if severity_level == "error":
            # Suppress warnings, keep only errors
            return ValidationResult(
                is_valid=len(result.errors) == 0,
                errors=result.errors,
                warnings=[],
            )

        return result

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Concise string representation of manager state."""
        n_active = len(self.get_active_parameters())
        n_fixed = len(self.get_fixed_parameters())
        n_varying_scaling = sum(1 for name in SCALING_PARAMS if self.space.vary.get(name, False))
        return (
            f"ParameterManager("
            f"n_physics={self.n_params}, "
            f"active={n_active}, "
            f"fixed={n_fixed}, "
            f"scaling_varying={n_varying_scaling}, "
            f"total={self.get_total_parameter_count()})"
        )
