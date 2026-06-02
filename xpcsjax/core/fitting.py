"""Homodyne NLSQ parameter space.
=================================

Defines :class:`ParameterSpace`, the bounds container consumed by the NLSQ
optimizer (see :mod:`xpcsjax.optimization.nlsq`) for homodyne analysis. It maps
an :class:`~xpcsjax.config.parameter_registry.AnalysisMode` to the ordered list
of physical-parameter ``(min, max)`` bounds, with optional override from a
configuration manager via :class:`~xpcsjax.config.parameter_manager.ParameterManager`.

The actual least-squares solve lives in the upstream ``nlsq`` library
(``CurveFit``) and the per-angle scaling helpers in
:mod:`xpcsjax.optimization.nlsq.parameter_utils`; this module only owns the
bounds definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParameterSpace:
    """Parameter space definition with bounds for NLSQ optimization.

    Implements specified parameter ranges for both scaling and physical
    parameters. Supports configuration-based bound override when
    config_manager is provided.
    """

    # Scaling parameters (always present)
    # FIXED (Nov 11, 2025): Updated bounds to match homodyne physics g₂ = 1 + β×g₁²
    # - contrast (β): Physical range [0, 1] where 0=no signal, 1=perfect contrast
    # - offset: Deviation from baseline=1.0, range [0.5, 1.5] allows ±50% variation
    contrast_bounds: tuple[float, float] = (0.0, 1.0)  # Physical contrast range
    offset_bounds: tuple[float, float] = (0.5, 1.5)

    # Physical parameter bounds (mode-dependent)
    D0_bounds: tuple[float, float] = (100.0, 100000.0)
    alpha_bounds: tuple[float, float] = (-2.0, 2.0)
    D_offset_bounds: tuple[float, float] = (-100000.0, 100000.0)

    # Laminar flow parameters (only for laminar_flow mode)
    gamma_dot_t0_bounds: tuple[float, float] = (1e-6, 0.5)
    beta_bounds: tuple[float, float] = (-2.0, 2.0)
    gamma_dot_t_offset_bounds: tuple[float, float] = (-0.1, 0.1)
    phi0_bounds: tuple[float, float] = (-10.0, 10.0)  # degrees

    # Data ranges
    fitted_range: tuple[float, float] = (0.0, 2.0)
    theory_range: tuple[float, float] = (0.0, 1.0)

    # Optional configuration manager for bound override
    config_manager: Any | None = None

    def get_param_bounds(self, analysis_mode: AnalysisMode) -> list[tuple[float, float]]:
        """Get parameter bounds based on analysis mode with configuration override support.

        Uses ParameterManager for consistent parameter handling and name mapping.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode: "static_anisotropic", "static_isotropic", or "laminar_flow"

        Returns
        -------
        list of tuple
            List of (min, max) bounds tuples for each parameter
        """
        # Strategy 1: Use ParameterManager for full integration (Phase 4.2+)
        if self.config_manager:
            try:
                from xpcsjax.config.parameter_manager import ParameterManager

                # Get config dict from manager
                config_dict = None
                if hasattr(self.config_manager, "config"):
                    config_dict = self.config_manager.config
                elif isinstance(self.config_manager, dict):
                    config_dict = self.config_manager

                # Create ParameterManager
                param_manager = ParameterManager(config_dict, analysis_mode)

                # Get active parameters (physical only, excludes scaling)
                active_params = param_manager.get_active_parameters()

                # Get bounds as tuples
                bounds = param_manager.get_bounds_as_tuples(active_params)

                logger.info(
                    f"Loaded {len(bounds)} parameter bounds from ParameterManager for {analysis_mode} mode",
                )
                return bounds

            except (TypeError, KeyError, AttributeError, ValueError) as e:
                logger.warning(
                    f"Failed to use ParameterManager: {e}, falling back to defaults",
                )

        # Fallback to hardcoded defaults
        logger.debug(f"Using default hardcoded bounds for {analysis_mode} mode")
        bounds = [
            self.D0_bounds,
            self.alpha_bounds,
            self.D_offset_bounds,
        ]

        if analysis_mode == "laminar_flow":
            bounds.extend(
                [
                    self.gamma_dot_t0_bounds,
                    self.beta_bounds,
                    self.gamma_dot_t_offset_bounds,
                    self.phi0_bounds,
                ],
            )

        return bounds


__all__ = ["ParameterSpace"]
