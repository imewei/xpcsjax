"""HeterodyneModel — adapter exposing two-component heterodyne XPCS physics
through the xpcsjax :class:`PhysicsModelBase` contract.

This module is the heterodyne analog of :mod:`xpcsjax.core.homodyne_model`,
adapted to the abstract :class:`~xpcsjax.core.models.PhysicsModelBase` so the
NLSQ engine can drive homodyne and heterodyne fits through a single interface.

The 14 physics parameters are sourced from the shared parameter registry
(see :mod:`xpcsjax.config.parameter_registry`, Tasks 24–25). Internally the
residual / correlation evaluation delegates to the ported heterodyne kernels
(:func:`xpcsjax.core.heterodyne_jax_backend.compute_c2_heterodyne`,
Task 26).

Notes
-----
- NLSQ-only: no NumPyro / Blackjax / ArviZ imports.
- Complete-mirror: kernels are local under ``xpcsjax.core``; no imports from
  the upstream ``homodyne`` / ``heterodyne`` packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from xpcsjax.config.parameter_registry import (
    AnalysisMode,
    get_param_names,
    get_registry,
)
from xpcsjax.core.models import PhysicsModelBase

if TYPE_CHECKING:  # pragma: no cover - typing only
    from xpcsjax.config.parameter_registry import ParameterInfo


_TWO_COMPONENT_MODE: AnalysisMode = "two_component"


class HeterodyneModel(PhysicsModelBase):
    """Two-component reference + sample heterodyne XPCS model (14 physics params).

    Wraps the heterodyne physics into the xpcsjax :class:`PhysicsModelBase`
    contract so the NLSQ engine can drive it identically to
    :class:`~xpcsjax.core.models.DiffusionModel` /
    :class:`~xpcsjax.core.models.CombinedModel`.

    The wrapped kernel,
    :func:`~xpcsjax.core.heterodyne_jax_backend.compute_c2_heterodyne`,
    takes a 1-D time array ``t`` and a scalar phi angle and returns the
    two-time correlation matrix ``c2`` of shape ``(N, N)``. Because the
    PhysicsModelBase contract advertises ``compute_g1(... t1, t2, phi, q, L)``,
    this wrapper offers a thin :meth:`compute_g1` that builds ``t`` from the
    diagonal of the supplied ``t1`` grid and ``vmap``-s the heterodyne kernel
    over the phi axis.

    Parameters
    ----------
    None
        The constructor takes no arguments; parameter ordering and bounds come
        from the shared registry. A future :meth:`from_config` classmethod can
        be added when a HeterodyneConfig schema lands.

    Attributes
    ----------
    analysis_mode : str
        Always ``"two_component"``.
    parameter_names : list of str
        14 parameter names in canonical order (from the registry).
    """

    analysis_mode: AnalysisMode = _TWO_COMPONENT_MODE

    def __init__(self) -> None:
        registry = get_registry()
        names = list(get_param_names(self.analysis_mode))
        super().__init__(name="heterodyne_two_component", parameter_names=names)
        self._info: dict[str, ParameterInfo] = {n: registry.get_param_info(n) for n in names}

    # ------------------------------------------------------------------
    # Required PhysicsModelBase abstract methods
    # ------------------------------------------------------------------
    def compute_g1(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,  # noqa: ARG002 — kept for interface uniformity (heterodyne is L-free here)
        dt: float | None = None,
    ) -> jnp.ndarray:
        """Compute the heterodyne two-time correlation surface(s).

        Strictly speaking this returns ``c2`` (not ``g1``) because the
        heterodyne kernel exposes the full ``c2 = offset + contrast * g1²``
        form. Per the :class:`PhysicsModelBase` contract we still surface it
        as ``compute_g1`` so the wrapper is interchangeable with
        :class:`DiffusionModel` / :class:`CombinedModel` at the NLSQ call
        site; downstream consumers that need a pure ``g1`` can subtract the
        baseline.

        Parameters
        ----------
        params : jnp.ndarray
            14-element parameter vector in registry order.
        t1, t2 : jnp.ndarray
            Time grids. Must be consistent; the kernel uses the diagonal
            ``t = t1[:, 0]`` (or ``t1[0, :]``) to construct the 1-D time
            array internally.
        phi : jnp.ndarray
            Phi angles (scalar or 1-D array). Vectorized via ``vmap``.
        q : float
            Scattering wavevector magnitude.
        L : float
            Geometric length (unused by the heterodyne kernel — kept for
            interface compatibility).
        dt : float | None
            Time step. If ``None``, inferred from ``t1`` spacing.

        Returns
        -------
        jnp.ndarray
            For scalar ``phi``: shape ``(N, N)``.
            For 1-D ``phi`` of length ``n_phi``: shape ``(n_phi, N, N)``.
        """
        from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne

        # Build 1-D time array from the t1 grid diagonal.
        t1_arr = jnp.asarray(t1)
        if t1_arr.ndim == 2:
            # Assume meshgrid with indexing="ij": rows vary in t1, cols in t2.
            t = t1_arr[:, 0]
        else:
            t = t1_arr.reshape(-1)

        if dt is None:
            # t is a concrete array here (compute_g1 is not JIT-compiled).
            # Use plain Python/numpy arithmetic to get a Python float — avoids
            # creating a JAX 0-d array that forces JIT retracing on every new t.
            dt_float = float(t[1] - t[0]) if len(t) > 1 else 1.0
        else:
            dt_float = float(dt)

        phi_arr = jnp.asarray(phi)
        if phi_arr.ndim == 0:
            return compute_c2_heterodyne(
                params=params,
                t=t,
                q=q,
                dt=dt_float,
                phi_angle=phi_arr,
                contrast=1.0,
                offset=0.0,
            )

        # Vectorize across phi angles.
        def _per_phi(phi_scalar: jnp.ndarray) -> jnp.ndarray:
            return compute_c2_heterodyne(
                params=params,
                t=t,
                q=q,
                dt=dt_float,
                phi_angle=phi_scalar,
                contrast=1.0,
                offset=0.0,
            )

        return jax.vmap(_per_phi)(phi_arr)

    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for the 14 heterodyne params (registry order)."""
        return [
            (self._info[n].lower_bound, self._info[n].upper_bound) for n in self.parameter_names
        ]

    def get_default_parameters(self) -> jnp.ndarray:
        """Get default parameter values for the 14 heterodyne params."""
        return jnp.asarray([self._info[n].default for n in self.parameter_names])

    # ------------------------------------------------------------------
    # Convenience aliases (Phase-3 / Task-13 callers used ``param_names``)
    # ------------------------------------------------------------------
    @property
    def param_names(self) -> list[str]:
        """Alias for :attr:`parameter_names` (Task 13 compatibility)."""
        return self.parameter_names

    @property
    def parameter_bounds(self) -> list[tuple[float, float]]:
        """Alias surface for callers that read bounds as a property."""
        return self.get_parameter_bounds()

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Alias for :attr:`parameter_bounds` (Task 13 compatibility)."""
        return self.get_parameter_bounds()

    # ------------------------------------------------------------------
    # Residual hook used by the NLSQ engine wrapper layer
    # ------------------------------------------------------------------
    def compute_residual(
        self,
        params: jnp.ndarray,
        data: dict[str, Any],
        ctx: dict[str, Any] | None = None,  # noqa: ARG002 — reserved for future ctx
    ) -> jnp.ndarray:
        """Compute the flat residual vector ``model - data`` for NLSQ.

        Parameters
        ----------
        params : jnp.ndarray
            14-element parameter vector.
        data : dict
            Required keys:
              - ``c2_exp``: experimental ``c2`` matrix or stack
                ``(n_phi, N, N)``.
              - ``t``: 1-D time array of length ``N``.
              - ``q``: scattering wavevector (float).
              - ``phi_angles_list`` or ``phi_angle``: angle(s) in degrees.
            Optional keys:
              - ``dt``: time step (default 1.0).
              - ``contrast``: speckle contrast (default 1.0).
              - ``offset``: baseline offset (default 0.0).
        ctx : dict, optional
            Reserved for future context (unused).
        """
        from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne

        t = jnp.asarray(data["t"])
        q = float(data["q"])
        dt = float(data.get("dt", 1.0))
        contrast = float(data.get("contrast", 1.0))
        offset = float(data.get("offset", 0.0))

        phi = jnp.asarray(data.get("phi_angles_list", data.get("phi_angle")))

        def _per_phi(phi_scalar: jnp.ndarray) -> jnp.ndarray:
            return compute_c2_heterodyne(
                params=params,
                t=t,
                q=q,
                dt=dt,
                phi_angle=phi_scalar,
                contrast=contrast,
                offset=offset,
            )

        if phi.ndim == 0:
            c2_model = _per_phi(phi)
        else:
            c2_model = jax.vmap(_per_phi)(phi)

        c2_exp = jnp.asarray(data["c2_exp"])
        return (c2_model - c2_exp).reshape(-1)


__all__ = ["HeterodyneModel"]
