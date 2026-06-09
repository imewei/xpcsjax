"""Anti-Degeneracy Controller - Orchestrator for 5-Layer Defense System.

This module provides a clean interface for initializing and coordinating
the 5-layer anti-degeneracy defense system for NLSQ optimization.

The controller encapsulates:
- Layer 1: Fourier/Constant Reparameterization
- Layer 2: Hierarchical Optimization
- Layer 3: Adaptive CV-based Regularization
- Layer 4: Gradient Collapse Monitoring
- Layer 5: Shear-Sensitivity Weighting

Usage::

    controller = AntiDegeneracyController.from_config(
        config_dict, n_phi, phi_angles, n_physical
    )
    if controller.is_enabled:
        # Use controller.fourier, controller.hierarchical, etc.
        transformed_params = controller.transform_params_to_fourier(initial_params)
        model_fn = controller.wrap_model_fn(base_model_fn)

Version: 2.9.0
Author: Claude Code
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.adaptive_regularization import (
    AdaptiveRegularizationConfig,
    AdaptiveRegularizer,
)
from xpcsjax.optimization.nlsq.fourier_reparam import (
    FourierReparamConfig,
    FourierReparameterizer,
)
from xpcsjax.optimization.nlsq.gradient_monitor import (
    GradientCollapseMonitor,
    GradientMonitorConfig,
)
from xpcsjax.optimization.nlsq.hierarchical import (
    HierarchicalConfig,
    HierarchicalOptimizer,
)
from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper
from xpcsjax.optimization.nlsq.shear_weighting import (
    ShearSensitivityWeighting,
    ShearWeightingConfig,
)
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


# Task 29: Layer-gating by analysis_mode (model lineage).
# Map: layer class-name -> set of analysis_modes where it is ACTIVE.
# A layer NOT listed here is active for all modes (default-active).
# ShearSensitivityWeighting (L5) is laminar_flow-specific: it up-weights data
# near the flow direction phi0 to exploit laminar_flow's shear-sensitivity peak
# (d g1_shear / d gamma_dot ~ cos(phi0 - phi)). This weighting scheme is tied to
# laminar_flow's shear-rate (gamma_dot) kernel. The static modes
# (static_anisotropic / static_isotropic) have no flow term at all. Heterodyne
# (two_component) DOES have a velocity/flow term (v0, v_offset, phi0_het), but it
# is structurally different from laminar_flow's shear rate, so the same weighting
# does not transfer — its angular information is already well distributed across
# phi. L5 is therefore active for laminar_flow ONLY.
_LAYER_GATES: dict[str, frozenset[str]] = {
    "ShearSensitivityWeighting": frozenset({"laminar_flow"}),
}


@dataclass
class AntiDegeneracyConfig:
    """Configuration for the Anti-Degeneracy Defense System.

    Attributes
    ----------
    enable : bool
        Master switch for all anti-degeneracy defenses.
    per_angle_mode : str
        Mode for per-angle parameters: "individual", "constant", "fourier", or "auto".
    fourier_order : int
        Order of Fourier series (order=2 -> 5 coefficients per group).
    fourier_auto_threshold : int
        n_phi threshold for auto mode to switch to Fourier.
    constant_scaling_threshold : int
        n_phi threshold for auto mode to use constant scaling (n_phi >= threshold).
    hierarchical_enable : bool
        Enable hierarchical two-stage optimization.
    hierarchical_max_outer_iterations : int
        Maximum outer iterations for hierarchical optimization.
    hierarchical_outer_tolerance : float
        Convergence tolerance on physical parameter change.
    regularization_mode : str
        Regularization mode: "absolute", "relative", or "auto".
    regularization_lambda : float
        Base regularization strength.
    regularization_target_cv : float
        Target coefficient of variation (0-1).
    regularization_target_contribution : float
        Target regularization contribution to loss (0-1).
    gradient_monitoring_enable : bool
        Enable gradient collapse monitoring.
    gradient_ratio_threshold : float
        Collapse threshold for norm(grad_physical)/norm(grad_per_angle).
    gradient_consecutive_triggers : int
        Number of consecutive triggers to confirm collapse.
    gradient_response_mode : str
        Response action: "warn", "hierarchical", "reset", "abort".
    execute_layers : bool
        Opt-in gate for the L2 hierarchical + L3 regularization anti-degeneracy
        ESCAPE on the >=1M stratified-LS path. Default ``False`` runs the single
        baseline solve (layers configured + diagnosed but not executed —
        byte-identical to the pre-escape path). ``True`` runs the escape after the
        baseline and keeps it only under the keep-better guard (never worse than
        the baseline). The escape is EXPENSIVE (~3-5x the baseline fit wall-time),
        so it is an opt-in for genuinely-stuck / degenerate fits, not a default.
    """

    enable: bool = True
    per_angle_mode: str = "auto"
    fourier_order: int = 2
    fourier_auto_threshold: int = 6
    constant_scaling_threshold: int = 3
    hierarchical_enable: bool = True
    hierarchical_max_outer_iterations: int = 5
    hierarchical_outer_tolerance: float = 1e-6
    hierarchical_physical_max_iterations: int = 100
    hierarchical_per_angle_max_iterations: int = 50
    regularization_mode: str = "relative"
    regularization_lambda: float = 1.0
    regularization_target_cv: float = 0.10
    regularization_target_contribution: float = 0.10
    regularization_max_cv: float = 0.20
    gradient_monitoring_enable: bool = True
    gradient_ratio_threshold: float = 0.01
    gradient_consecutive_triggers: int = 5
    gradient_response_mode: str = "hierarchical"
    # Layer 5: Shear-Sensitivity Weighting
    shear_weighting_enable: bool = True
    shear_weighting_min_weight: float = 0.3
    shear_weighting_alpha: float = 1.0
    shear_weighting_update_frequency: int = 1
    shear_weighting_normalize: bool = True
    # Future gate: numeric L2/L3 execution on stratified-LS path (currently inert)
    execute_layers: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> AntiDegeneracyConfig:
        """Create config from nested dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary with structure::

                {
                    "enable": bool,
                    "per_angle_mode": str,
                    "fourier_order": int,
                    "fourier_auto_threshold": int,
                    "hierarchical": {...},
                    "regularization": {...},
                    "gradient_monitoring": {...}
                }

        Returns
        -------
        AntiDegeneracyConfig
            Validated configuration object.
        """
        hierarchical = config_dict.get("hierarchical", {})
        regularization = config_dict.get("regularization", {})
        gradient_monitoring = config_dict.get("gradient_monitoring", {})
        shear_weighting = config_dict.get("shear_weighting", {})

        return cls(
            enable=config_dict.get("enable", True),
            per_angle_mode=config_dict.get("per_angle_mode", "auto"),
            fourier_order=config_dict.get("fourier_order", 2),
            fourier_auto_threshold=config_dict.get("fourier_auto_threshold", 6),
            constant_scaling_threshold=config_dict.get("constant_scaling_threshold", 3),
            # Hierarchical
            hierarchical_enable=hierarchical.get("enable", True),
            hierarchical_max_outer_iterations=hierarchical.get("max_outer_iterations", 5),
            hierarchical_outer_tolerance=float(hierarchical.get("outer_tolerance", 1e-6)),
            hierarchical_physical_max_iterations=hierarchical.get("physical_max_iterations", 100),
            hierarchical_per_angle_max_iterations=hierarchical.get("per_angle_max_iterations", 50),
            # Regularization
            regularization_mode=regularization.get("mode", "relative"),
            regularization_lambda=float(regularization.get("lambda", 1.0)),
            regularization_target_cv=float(regularization.get("target_cv", 0.10)),
            regularization_target_contribution=float(
                regularization.get("target_contribution", 0.10)
            ),
            regularization_max_cv=float(regularization.get("max_cv", 0.20)),
            # Gradient monitoring
            gradient_monitoring_enable=gradient_monitoring.get("enable", True),
            gradient_ratio_threshold=float(gradient_monitoring.get("ratio_threshold", 0.01)),
            gradient_consecutive_triggers=gradient_monitoring.get("consecutive_triggers", 5),
            gradient_response_mode=gradient_monitoring.get("response", "hierarchical"),
            # Shear weighting
            shear_weighting_enable=shear_weighting.get("enable", True),
            shear_weighting_min_weight=float(shear_weighting.get("min_weight", 0.3)),
            shear_weighting_alpha=float(shear_weighting.get("alpha", 1.0)),
            shear_weighting_update_frequency=int(shear_weighting.get("update_frequency", 1)),
            shear_weighting_normalize=shear_weighting.get("normalize", True),
            # Future gate: numeric L2/L3 execution on stratified-LS path (currently inert)
            execute_layers=bool(config_dict.get("execute_layers", False)),
        )


@dataclass
class AntiDegeneracyController:
    """Orchestrator for the 5-Layer Anti-Degeneracy Defense System.

    Owns and coordinates the five defense layers as a single object: it resolves
    the per-angle scaling mode, builds the layer components that the resolved mode
    enables, and exposes the parameter transforms, callbacks, and diagnostics the
    NLSQ solver paths consume.

    The five layers, in order, are L1 Fourier/constant reparameterization
    (:class:`~xpcsjax.optimization.nlsq.fourier_reparam.FourierReparameterizer`),
    L2 hierarchical optimization
    (:class:`~xpcsjax.optimization.nlsq.hierarchical.HierarchicalOptimizer`),
    L3 adaptive CV-based regularization
    (:class:`~xpcsjax.optimization.nlsq.adaptive_regularization.AdaptiveRegularizer`),
    L4 gradient-collapse monitoring
    (:class:`~xpcsjax.optimization.nlsq.gradient_monitor.GradientCollapseMonitor`),
    and L5 shear-sensitivity weighting
    (:class:`~xpcsjax.optimization.nlsq.shear_weighting.ShearSensitivityWeighting`).

    Attributes
    ----------
    config : AntiDegeneracyConfig
        Configuration for the defense system.
    n_phi : int
        Number of phi angles.
    n_physical : int
        Number of physical parameters.
    phi_angles : np.ndarray
        Array of phi angles in radians.
    fourier : FourierReparameterizer or None
        Layer 1: Fourier reparameterization component.
    hierarchical : HierarchicalOptimizer or None
        Layer 2: Hierarchical optimization component.
    regularizer : AdaptiveRegularizer or None
        Layer 3: Adaptive regularization component.
    monitor : GradientCollapseMonitor or None
        Layer 4: Gradient-collapse monitoring component (strictly diagnostic).
    shear_weighter : ShearSensitivityWeighting or None
        Layer 5: Shear-sensitivity weighting component (``laminar_flow`` only).
    per_angle_mode_actual : str
        Actual mode used ("constant", "fourier", or "independent").

    Notes
    -----
    L5 is gated to ``laminar_flow`` via ``_LAYER_GATES``; every other layer is
    default-active for all modes (see :meth:`is_layer_active`). When
    ``analysis_mode`` is ``None`` (the homodyne characterization gate path),
    :meth:`is_layer_active` returns ``True`` for every layer. L4 is strictly
    diagnostic -- enabling the monitor is bit-identical to disabling it.

    Examples
    --------
    >>> controller = AntiDegeneracyController.from_config(
    ...     config_dict, n_phi, phi_angles, n_physical
    ... )
    >>> if controller.is_enabled:
    ...     transformed = controller.transform_params_to_fourier(initial_params)
    """

    config: AntiDegeneracyConfig
    n_phi: int
    n_physical: int
    phi_angles: np.ndarray
    fourier: FourierReparameterizer | None = None
    hierarchical: HierarchicalOptimizer | None = None
    regularizer: AdaptiveRegularizer | None = None
    monitor: GradientCollapseMonitor | None = None
    shear_weighter: ShearSensitivityWeighting | None = None  # Layer 5
    mapper: ParameterIndexMapper | None = None  # T018: Centralized index mapping
    per_angle_mode_actual: str = "disabled"
    # Task 29: model-lineage gating for anti-degeneracy layers.
    # None = backward-compatible "all layers active" behavior used by the
    # homodyne characterization gate at rtol=1e-10. When set (e.g. by
    # HeterodyneModel passing "two_component"), Layer 5 is short-circuited.
    analysis_mode: AnalysisMode | None = None
    # Fixed per-angle quantile estimates for constant mode (v2.17.0+)
    _fixed_contrast_per_angle: np.ndarray | None = field(default=None, repr=False)
    _fixed_offset_per_angle: np.ndarray | None = field(default=None, repr=False)
    _is_initialized: bool = field(default=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config_dict: dict[str, Any],
        n_phi: int,
        phi_angles: np.ndarray,
        n_physical: int,
        per_angle_scaling: bool = True,
        is_laminar_flow: bool = True,
        analysis_mode: AnalysisMode | None = None,
    ) -> AntiDegeneracyController:
        """Create controller from configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Anti-degeneracy configuration dictionary.
        n_phi : int
            Number of phi angles.
        phi_angles : np.ndarray
            Array of phi angles in radians.
        n_physical : int
            Number of physical parameters (7 for laminar_flow, 14 for two_component).
        per_angle_scaling : bool
            Whether per-angle scaling is enabled.
        is_laminar_flow : bool
            Whether this is laminar_flow mode.
        analysis_mode : str | None
            Model lineage ("static_anisotropic", "static_isotropic", "laminar_flow",
            "two_component"). Threaded into the controller so Task 29's
            ``_LAYER_GATES`` can short-circuit homodyne-only layers
            (currently Layer 5) for heterodyne fits. ``None`` preserves
            backward-compatible "all layers active" behavior used by the
            homodyne characterization gate.
        """
        config = AntiDegeneracyConfig.from_dict(config_dict)

        controller = cls(
            config=config,
            n_phi=n_phi,
            n_physical=n_physical,
            phi_angles=phi_angles,
            analysis_mode=analysis_mode,
        )

        # Init gate. ``is_laminar_flow`` initializes the full controller for the
        # laminar path; ``two_component`` (heterodyne) also initializes so its
        # ≥1M stratified-LS path gets banner/diagnostic-surface parity with
        # laminar. L5 is independently gated off for two_component by
        # ``_LAYER_GATES`` (``is_layer_active``), so this does NOT enable shear
        # weighting. Static homodyne modes (is_laminar_flow=False,
        # analysis_mode != "two_component") still skip init exactly as before.
        # Both existing call sites (core.py / strategies/stratified_ls.py) pass
        # is_laminar_flow correctly — note core.py also serves static homodyne
        # with is_laminar_flow=False — so the new two_component disjunct is False
        # for every existing caller and the rtol=1e-10 homodyne baselines stay
        # green.
        _is_two_component = (
            analysis_mode is not None
            and cls._normalize_mode(str(analysis_mode)) == "two_component"
        )
        if config.enable and per_angle_scaling and (is_laminar_flow or _is_two_component):
            controller._initialize_components()

        return controller

    def _initialize_components(self) -> None:
        """Initialize all 4 layers of the defense system."""
        config = self.config

        # T018-T020: Determine actual per-angle mode with auto-selection logic
        # v2.18.0: Distinct semantics for auto vs explicit constant:
        #   - auto (n_phi >= threshold): "auto_averaged" → 9 params, OPTIMIZED averaged scaling
        #   - constant (explicit): "fixed_constant" → 7 params, FIXED per-angle scaling
        #   - individual: per-angle scaling OPTIMIZED
        if config.per_angle_mode == "auto":
            if self.n_phi >= config.constant_scaling_threshold:
                # AUTO mode with large n_phi: optimize averaged scaling (9 params)
                # Computes N quantile estimates, averages to 1 contrast + 1 offset
                # These 2 averaged values ARE OPTIMIZED along with 7 physical params
                self.per_angle_mode_actual = "auto_averaged"
                logger.info("=" * 60)
                logger.info("ANTI-DEGENERACY: Auto-selected 'auto_averaged' mode")
                logger.info(
                    f"  Reason: n_phi ({self.n_phi}) >= "
                    f"constant_scaling_threshold ({config.constant_scaling_threshold})"
                )
                logger.info("  Behavior: Quantile estimates -> AVERAGED -> OPTIMIZED")
                logger.info(
                    f"  Parameters: {self.n_physical} physical + 2 averaged scaling "
                    f"= {self.n_physical + 2} total"
                )
                logger.info("=" * 60)
            else:
                # Use individual per-angle parameters for few angles (N < 3)
                self.per_angle_mode_actual = "individual"
                logger.info("=" * 60)
                logger.info("ANTI-DEGENERACY: Auto-selected 'individual' mode")
                logger.info(
                    f"  Reason: n_phi ({self.n_phi}) < "
                    f"constant_scaling_threshold ({config.constant_scaling_threshold})"
                )
                logger.info(
                    f"  Parameters: {self.n_physical} physical + {2 * self.n_phi} per-angle "
                    f"= {self.n_physical + 2 * self.n_phi} total"
                )
                logger.info("=" * 60)
        elif config.per_angle_mode == "constant":
            # EXPLICIT constant mode: FIXED per-angle scaling (7 params)
            # Computes N quantile estimates, uses per-angle values DIRECTLY (NOT averaged)
            # Only 7 physical params are optimized; scaling is FIXED
            self.per_angle_mode_actual = "fixed_constant"
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Using explicit 'constant' mode -> fixed_constant")
            logger.info(f"  n_phi: {self.n_phi}")
            logger.info("  Behavior: Quantile estimates -> per-angle values FIXED (NOT optimized)")
            logger.info(f"  Parameters: {self.n_physical} physical only (scaling FIXED from quantiles)")
            logger.info("=" * 60)
        else:
            # Other explicit modes (fourier or individual)
            self.per_angle_mode_actual = config.per_angle_mode
            logger.debug(
                f"ANTI-DEGENERACY: Using explicit per_angle_mode: {self.per_angle_mode_actual}"
            )

        # T021: Determine use_constant flag for mapper
        # Both auto_averaged and fixed_constant use constant-style mapping
        use_constant = self.per_angle_mode_actual in ("auto_averaged", "fixed_constant")

        # Layer 1: Fourier Reparameterization (only if fourier mode)
        if self.per_angle_mode_actual == "fourier":
            fourier_config = FourierReparamConfig(
                mode="fourier",
                fourier_order=config.fourier_order,
                auto_threshold=config.fourier_auto_threshold,
            )
            self.fourier = FourierReparameterizer(self.phi_angles, fourier_config)
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Layer 1 - Fourier Reparameterization")
            logger.info(f"  Mode: {self.per_angle_mode_actual}")
            logger.info(f"  n_phi: {self.n_phi}, Fourier order: {config.fourier_order}")
            logger.info(f"  Parameter reduction: {2 * self.n_phi} -> {self.fourier.n_coeffs}")
            logger.info("=" * 60)
        # Note: auto_averaged and fixed_constant logging already done in mode selection above

        # T022: Create ParameterIndexMapper with correct use_constant flag
        # This provides centralized, consistent index mapping for all subsequent layers
        self.mapper = ParameterIndexMapper(
            n_phi=self.n_phi,
            n_physical=self.n_physical,
            fourier=self.fourier,
            use_constant=use_constant,
        )
        logger.debug(
            f"ANTI-DEGENERACY: ParameterIndexMapper created: {self.mapper.get_diagnostics()}"
        )

        # Layer 2: Hierarchical Optimization
        if config.hierarchical_enable:
            hier_config = HierarchicalConfig(
                enable=True,
                max_outer_iterations=config.hierarchical_max_outer_iterations,
                outer_tolerance=config.hierarchical_outer_tolerance,
                physical_max_iterations=config.hierarchical_physical_max_iterations,
                per_angle_max_iterations=config.hierarchical_per_angle_max_iterations,
            )
            self.hierarchical = HierarchicalOptimizer(
                config=hier_config,
                n_phi=self.n_phi,
                n_physical=self.n_physical,
                fourier_reparameterizer=self.fourier,
            )
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Layer 2 - Hierarchical Optimization")
            logger.info("  Enabled: True")
            logger.info(f"  Max outer iterations: {config.hierarchical_max_outer_iterations}")
            logger.info(f"  Outer tolerance: {config.hierarchical_outer_tolerance}")
            logger.info("=" * 60)

        # Layer 3: Adaptive Regularization
        # T020: Use mapper.get_group_indices() instead of n_phi-based calculation
        # This fixes the dimension mismatch when Fourier reparameterization is active
        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            mode=cast(Literal["absolute", "relative", "auto"], config.regularization_mode),
            lambda_base=config.regularization_lambda,
            target_cv=config.regularization_target_cv,
            target_contribution=config.regularization_target_contribution,
            max_cv=config.regularization_max_cv,
            group_indices=self.mapper.get_group_indices(),  # T020: Use mapper indices
        )
        self.regularizer = AdaptiveRegularizer(
            reg_config,
            self.mapper.n_per_group,  # T020: Use Fourier-aware n_per_group
        )
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY: Layer 3 - Adaptive Regularization")
        logger.info(f"  Mode: {config.regularization_mode}")
        logger.info(f"  Auto-tuned lambda: {self.regularizer.lambda_value:.2f}")
        logger.info(f"  Target CV: {config.regularization_target_cv}")
        logger.info("=" * 60)

        # Layer 4: Gradient Collapse Monitor
        # Use mapper for consistent index calculation
        if config.gradient_monitoring_enable:
            per_angle_indices = self.mapper.get_per_angle_indices()
            physical_indices = self.mapper.get_physical_indices()

            monitor_config = GradientMonitorConfig(
                enable=True,
                ratio_threshold=config.gradient_ratio_threshold,
                consecutive_triggers=config.gradient_consecutive_triggers,
                response_mode=cast(
                    Literal["warn", "hierarchical", "reset", "abort"],
                    config.gradient_response_mode,
                ),
            )
            self.monitor = GradientCollapseMonitor(
                config=monitor_config,
                physical_indices=physical_indices,
                per_angle_indices=per_angle_indices,
            )
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Layer 4 - Gradient Collapse Monitor")
            logger.info("  Enabled: True")
            logger.info(f"  Ratio threshold: {config.gradient_ratio_threshold}")
            logger.info(f"  Response mode: {config.gradient_response_mode}")
            logger.info("=" * 60)

        # Layer 5: Shear-Sensitivity Weighting
        # Task 29: gate by model lineage. Heterodyne (two_component) has no
        # shear rate to weight, so this layer short-circuits to a no-op
        # (shear_weighter stays None and the existing `if self.shear_weighter
        # is None` guards in get_shear_weights / update_shear_phi0 take over).
        if (
            config.shear_weighting_enable
            and self.n_phi >= 3
            and self.is_layer_active("ShearSensitivityWeighting")
        ):
            sw_config = ShearWeightingConfig(
                enable=True,
                min_weight=config.shear_weighting_min_weight,
                alpha=config.shear_weighting_alpha,
                update_frequency=config.shear_weighting_update_frequency,
                initial_phi0=0.0,  # Will be updated from initial params
                normalize=config.shear_weighting_normalize,
            )
            self.shear_weighter = ShearSensitivityWeighting(
                phi_angles=self.phi_angles,
                n_physical=self.n_physical,
                phi0_index=6,  # phi0 is last of 7 physical params
                config=sw_config,
            )
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Layer 5 - Shear-Sensitivity Weighting")
            logger.info("  Enabled: True")
            logger.info(f"  n_phi: {self.n_phi}")
            logger.info(f"  min_weight: {config.shear_weighting_min_weight:.2f}")
            logger.info(f"  alpha: {config.shear_weighting_alpha:.1f}")
            logger.info("=" * 60)

        self._is_initialized = True

    @property
    def is_enabled(self) -> bool:
        """Check if the defense system is enabled and initialized."""
        return self._is_initialized and self.config.enable

    @property
    def use_fourier(self) -> bool:
        """Check if Fourier reparameterization is active."""
        return self.fourier is not None

    @property
    def use_constant(self) -> bool:
        """Check if constant scaling mode is active (either auto_averaged or fixed_constant).

        Both modes use constant-style parameter mapping (9 params for auto_averaged,
        7 params for fixed_constant), as opposed to individual mode (7 + 2*n_phi params).
        """
        return self.per_angle_mode_actual in ("auto_averaged", "fixed_constant")

    @property
    def use_fixed_scaling(self) -> bool:
        """Check if using FIXED per-angle scaling (7 params, not optimized).

        Returns True only for explicit constant mode ("fixed_constant"), where:
        - Per-angle contrast/offset are FIXED from quantile estimation
        - Only 7 physical parameters are optimized
        - Scaling is NOT part of the optimization

        This is DIFFERENT from auto_averaged mode, where:
        - Averaged contrast/offset ARE optimized (9 params total)
        """
        return self.per_angle_mode_actual == "fixed_constant"

    @property
    def use_averaged_scaling(self) -> bool:
        """Check if using OPTIMIZED averaged scaling (9 params).

        Returns True only for auto mode with n_phi >= threshold ("auto_averaged"), where:
        - N per-angle quantile estimates are averaged to 1 contrast + 1 offset
        - These 2 averaged values ARE OPTIMIZED along with 7 physical params
        - Total: 9 parameters
        """
        return self.per_angle_mode_actual == "auto_averaged"

    @property
    def use_hierarchical(self) -> bool:
        """Check if hierarchical optimization is active."""
        return self.hierarchical is not None

    @property
    def use_shear_weighting(self) -> bool:
        """Check if shear-sensitivity weighting is active."""
        return self.shear_weighter is not None

    @property
    def execute_layers(self) -> bool:
        """Return the stratified-LS L2/L3 escape gate flag.

        Read by both stratified-LS solvers (``stratified_ls.py`` /
        ``heterodyne_stratified_ls.py``) to gate the keep-better-guarded L2/L3
        escape. Default ``False`` runs the byte-identical single baseline solve;
        ``True`` runs the (expensive, opt-in) escape.

        Returns
        -------
        bool
            ``True`` when the L2/L3 escape is requested; ``False`` (default) runs
            the single baseline solve.
        """
        return self.config.execute_layers

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        """Normalize analysis-mode synonyms consistent with parameter_registry.

        - ``heterodyne`` is a synonym for ``two_component``.
        - hyphens are converted to underscores; case is lowered.
        """
        m = mode.lower().replace("-", "_")
        if m == "heterodyne":
            return "two_component"
        return m

    def is_layer_active(self, layer_name: str) -> bool:
        """Report whether a named layer is active for this ``analysis_mode``.

        Task 29 -- model-lineage gating of the 5-layer defense system.

        Backward-compatible: if ``analysis_mode`` was not provided at
        construction, all layers are active (preserves the homodyne
        characterization gate's rtol=1e-10 behavior).

        Parameters
        ----------
        layer_name : str
            Class name of the layer to query, e.g.
            ``"ShearSensitivityWeighting"``, ``"FourierReparameterizer"``,
            ``"HierarchicalOptimizer"``, ``"AdaptiveRegularizer"``,
            ``"GradientCollapseMonitor"``.

        Returns
        -------
        bool
            ``True`` if the layer is active for the current ``analysis_mode``.
        """
        if self.analysis_mode is None:
            return True
        mode = self._normalize_mode(self.analysis_mode)
        gates = _LAYER_GATES.get(layer_name)
        if gates is None:
            # Not in gate dict -> default active for every mode.
            return True
        return mode in gates

    @property
    def n_per_angle_params(self) -> int:
        """Get the number of per-angle parameters (optimized scaling params).

        Returns
        -------
        int
            The count of scaling parameters that participate in the optimization,
            which depends on the resolved per-angle scaling mode:

            - ``fixed_constant``: ``0`` (scaling is frozen, not optimized)
            - ``auto_averaged``: ``2`` (one contrast, one offset)
            - ``fourier``: ``n_coeffs`` (Fourier coefficients)
            - ``individual``: ``2 * n_phi`` (per-angle contrast + offset)
        """
        if self.use_fixed_scaling:
            return 0  # Scaling is FIXED, not part of optimization
        if self.use_averaged_scaling:
            return 2  # One contrast, one offset (optimized)
        if self.fourier:
            return self.fourier.n_coeffs
        return 2 * self.n_phi

    def transform_params_to_fourier(
        self, params: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
        """Transform per-angle parameters to Fourier coefficients.

        Parameters
        ----------
        params : np.ndarray
            Full parameter array: [contrast(n_phi), offset(n_phi), physical].

        Returns
        -------
        tuple
            (fourier_params, original_bounds_if_transformed)
            fourier_params: [contrast_coeffs, offset_coeffs, physical]
            bounds: (lower, upper) in Fourier space if transformation applied
        """
        if not self.use_fourier:
            return params, None

        # Transform to Fourier - fourier must be initialized if use_fourier is True
        assert self.fourier is not None, "Fourier reparameterizer must be initialized"

        # Split parameters
        contrast = params[: self.n_phi]
        offset = params[self.n_phi : 2 * self.n_phi]
        physical = params[2 * self.n_phi :]

        # Transform to Fourier
        contrast_coeffs = self.fourier.to_fourier(contrast)
        offset_coeffs = self.fourier.to_fourier(offset)

        return np.concatenate([contrast_coeffs, offset_coeffs, physical]), None

    def transform_params_from_fourier(self, fourier_params: np.ndarray) -> np.ndarray:
        """Transform Fourier coefficients back to per-angle parameters.

        Parameters
        ----------
        fourier_params : np.ndarray
            Fourier parameter array: [contrast_coeffs, offset_coeffs, physical].

        Returns
        -------
        np.ndarray
            Per-angle parameter array: [contrast(n_phi), offset(n_phi), physical].
        """
        if not self.use_fourier:
            return fourier_params

        # Access fourier attributes - fourier must be initialized if use_fourier is True
        assert self.fourier is not None, "Fourier reparameterizer must be initialized"

        n_coeffs = self.fourier.n_coeffs_per_param

        # Extract Fourier coefficients
        contrast_coeffs = fourier_params[:n_coeffs]
        offset_coeffs = fourier_params[n_coeffs : 2 * n_coeffs]
        physical = fourier_params[2 * n_coeffs :]

        # Transform back to per-angle
        contrast = self.fourier.from_fourier(contrast_coeffs)
        offset = self.fourier.from_fourier(offset_coeffs)

        return np.concatenate([contrast, offset, physical])

    def transform_params_to_constant(self, params: np.ndarray) -> np.ndarray:
        """Transform per-angle parameters to constant mode.

        Computes mean contrast and offset across all angles.

        Parameters
        ----------
        params : np.ndarray
            Full parameter array: [contrast(n_phi), offset(n_phi), physical].

        Returns
        -------
        np.ndarray
            Constant mode parameters: [contrast_mean, offset_mean, physical].
        """
        if not self.use_constant:
            return params

        # Split parameters
        contrast = params[: self.n_phi]
        offset = params[self.n_phi : 2 * self.n_phi]
        physical = params[2 * self.n_phi :]

        # Compute mean values (NaN-safe: degenerate optimizations can produce NaN params)
        contrast_mean = np.nanmean(contrast)
        offset_mean = np.nanmean(offset)

        return np.concatenate([[contrast_mean], [offset_mean], physical])

    def transform_params_from_constant(self, constant_params: np.ndarray) -> np.ndarray:
        """Transform constant mode parameters to per-angle form.

        Expands single contrast/offset values to all angles.

        Parameters
        ----------
        constant_params : np.ndarray
            Constant mode parameters: [contrast, offset, physical].

        Returns
        -------
        np.ndarray
            Per-angle parameters: [contrast(n_phi), offset(n_phi), physical].
        """
        if not self.use_constant:
            return constant_params

        # Extract constant values and physical parameters
        contrast_const = constant_params[0]
        offset_const = constant_params[1]
        physical = constant_params[2:]

        # Expand to per-angle arrays
        contrast = np.full(self.n_phi, contrast_const)
        offset = np.full(self.n_phi, offset_const)

        return np.concatenate([contrast, offset, physical])

    def get_group_variance_indices(self) -> list[tuple[int, int]] | None:
        """Get group variance indices for NLSQ regularization.

        T024: Delegates to ParameterIndexMapper for consistent index calculation
        regardless of Fourier mode.

        Returns
        -------
        list[tuple[int, int]] | None
            List of (start, end) tuples for each parameter group.
        """
        if not self.is_enabled:
            return None

        # T024: Delegate to mapper for consistent Fourier-aware indices
        if self.mapper is not None:
            return self.mapper.get_group_indices()

        # Fallback for backward compatibility (should not reach here in normal use)
        if self.fourier is None:
            raise ValueError(
                "get_group_variance_indices called but neither mapper nor fourier is initialized. "
                "This can occur with per_angle_mode='constant' where group variance is not applicable."
            )
        n_per_group = self.fourier.n_coeffs_per_param if self.use_fourier else self.n_phi
        return [(0, n_per_group), (n_per_group, 2 * n_per_group)]

    def get_diagnostics(self) -> dict[str, Any]:
        """Get comprehensive diagnostics from all components.

        Returns
        -------
        dict
            Nested diagnostics from all 5 layers.
        """
        diag: dict[str, Any] = {
            "version": "2.18.0",
            "enabled": self.is_enabled,
            "execute_layers": self.execute_layers,
            "per_angle_mode": self.config.per_angle_mode,  # Config value
            "per_angle_mode_actual": self.per_angle_mode_actual,  # Resolved actual mode
            "use_constant": self.use_constant,
            "use_fixed_scaling": self.use_fixed_scaling,
            "use_averaged_scaling": self.use_averaged_scaling,
            "use_fourier": self.use_fourier,
            "use_shear_weighting": self.use_shear_weighting,
            "n_phi": self.n_phi,
            "n_physical": self.n_physical,
            "n_per_angle_params": self.n_per_angle_params,
            "n_total_params": self.n_physical + self.n_per_angle_params,
            "has_fixed_per_angle_scaling": self.has_fixed_per_angle_scaling(),
        }

        # Add fixed per-angle scaling info if available
        if self.has_fixed_per_angle_scaling():
            # Ensure arrays are not None before computing statistics
            assert self._fixed_contrast_per_angle is not None, "Fixed contrast must be set"
            assert self._fixed_offset_per_angle is not None, "Fixed offset must be set"
            diag["fixed_per_angle_scaling"] = {
                "contrast_mean": float(np.nanmean(self._fixed_contrast_per_angle)),
                "contrast_std": float(np.nanstd(self._fixed_contrast_per_angle)),
                "offset_mean": float(np.nanmean(self._fixed_offset_per_angle)),
                "offset_std": float(np.nanstd(self._fixed_offset_per_angle)),
            }

        # Add mapper diagnostics
        if self.mapper:
            diag["mapper"] = self.mapper.get_diagnostics()

        if self.fourier:
            diag["fourier"] = self.fourier.get_diagnostics()

        if self.hierarchical:
            diag["hierarchical"] = self.hierarchical.get_diagnostics()

        if self.regularizer:
            diag["regularization"] = self.regularizer.get_diagnostics()

        if self.monitor:
            diag["gradient_monitor"] = self.monitor.get_diagnostics()

        if self.shear_weighter:
            diag["shear_weighting"] = self.shear_weighter.get_diagnostics()

        return diag

    def reset_monitor(self) -> None:
        """Reset the gradient collapse monitor state."""
        if self.monitor:
            self.monitor.reset()

    def get_shear_weights(self) -> np.ndarray | None:
        """Get shear-sensitivity weights for residuals.

        Returns
        -------
        np.ndarray | None
            Array of weights (one per phi angle), or None if not enabled.
        """
        if self.shear_weighter is None:
            return None
        return self.shear_weighter.get_weights()

    def update_shear_phi0(self, params: np.ndarray, iteration: int = 0) -> None:
        """Update the phi0 value in shear weighter.

        Parameters
        ----------
        params : np.ndarray
            Current parameter vector.
        iteration : int
            Current iteration number.
        """
        if self.shear_weighter is not None:
            self.shear_weighter.update_phi0(params, iteration)

    def compute_fixed_per_angle_scaling(
        self,
        stratified_data: Any,
        contrast_bounds: tuple[float, float] = (0.0, 1.0),
        offset_bounds: tuple[float, float] = (0.5, 1.5),
    ) -> None:
        """Compute and store fixed per-angle contrast/offset from quantiles.

        This method uses physics-informed quantile analysis to estimate
        contrast and offset for each phi angle independently.

        In "constant" mode (v2.17.0+):
        1. Computes N contrast + N offset values from quantile estimation
        2. These are averaged to 1 contrast + 1 offset for optimization
        3. Optimizer works with 9 parameters: 7 physical + 2 averaged scaling
        4. The individual per-angle estimates are stored for diagnostics

        Parameters
        ----------
        stratified_data : StratifiedData
            Data containing per-angle g2_flat, phi_flat, t1_flat, t2_flat arrays.
        contrast_bounds : tuple[float, float]
            Valid bounds for contrast parameter.
        offset_bounds : tuple[float, float]
            Valid bounds for offset parameter.

        Notes
        -----
        This method should be called before optimization when using
        per_angle_mode="constant". The estimates can be retrieved using
        get_fixed_per_angle_scaling().

        Unlike least-squares estimation, this approach:
        1. Does not require a model (purely data-driven)
        2. Uses physics-informed quantile analysis
        3. Is robust to outliers
        """
        from xpcsjax.optimization.nlsq.parameter_utils import (
            compute_quantile_per_angle_scaling,
        )

        if not self.use_constant:
            logger.warning(
                "compute_fixed_per_angle_scaling called but not in constant mode; "
                "estimates will be stored but may not be used"
            )

        logger.info("=" * 60)
        logger.info("Estimating per-angle scaling from quantiles")
        logger.info("=" * 60)

        contrast_per_angle, offset_per_angle = compute_quantile_per_angle_scaling(
            stratified_data=stratified_data,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
            logger=logger,
        )

        self._fixed_contrast_per_angle = contrast_per_angle
        self._fixed_offset_per_angle = offset_per_angle

        logger.info(
            f"Fixed per-angle scaling stored:\n"
            f"  n_phi: {self.n_phi}\n"
            f"  Contrast: mean={np.nanmean(contrast_per_angle):.4f}, "
            f"std={np.nanstd(contrast_per_angle):.4f}\n"
            f"  Offset: mean={np.nanmean(offset_per_angle):.4f}, "
            f"std={np.nanstd(offset_per_angle):.4f}"
        )

    def get_fixed_per_angle_scaling(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get the fixed per-angle contrast/offset estimates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            (contrast_per_angle, offset_per_angle) if computed, None otherwise.
        """
        if self._fixed_contrast_per_angle is None or self._fixed_offset_per_angle is None:
            return None
        return self._fixed_contrast_per_angle, self._fixed_offset_per_angle

    def has_fixed_per_angle_scaling(self) -> bool:
        """Check if fixed per-angle scaling has been computed.

        Returns
        -------
        bool
            True if fixed scaling is available.
        """
        return (
            self._fixed_contrast_per_angle is not None and self._fixed_offset_per_angle is not None
        )

    def create_nlsq_callbacks(self) -> dict[str, Any]:
        """Create callbacks for NLSQ's CurveFit integration.

        This method creates callbacks that can be passed to NLSQ's CurveFit
        or AdaptiveHybridStreamingOptimizer to enable anti-degeneracy defenses.

        Returns
        -------
        dict
            Dictionary of callbacks compatible with NLSQ:
            - 'loss_augmentation': Callable for regularization loss
            - 'iteration_callback': Callable for gradient monitoring
            - 'group_variance_indices': Indices for group variance regularization

        Notes
        -----
        For NLSQ v0.4+, callbacks can be passed to CurveFit.curve_fit() or
        injected into HybridStreamingConfig.

        Example
        -------
        >>> controller = AntiDegeneracyController.from_config(config, n_phi, phi_angles, n_physical)
        >>> callbacks = controller.create_nlsq_callbacks()
        >>> result = fitter.curve_fit(f, xdata, ydata, **callbacks)
        """
        if not self.is_enabled:
            return {}

        callbacks: dict[str, Any] = {}

        # Group variance indices for NLSQ's internal regularization
        group_indices = self.get_group_variance_indices()
        if group_indices:
            callbacks["group_variance_indices"] = group_indices

        # Regularization lambda value from adaptive regularizer
        if self.regularizer:
            callbacks["group_variance_lambda"] = self.regularizer.lambda_value

        # Loss augmentation callback for Layer 3 (Adaptive Regularization)
        if self.regularizer:

            def loss_augmentation(params: np.ndarray, residuals: np.ndarray) -> float:
                """Add regularization penalty to loss."""
                # Compute MSE from residuals
                mse = float(np.nanmean(residuals**2))
                n_points = len(residuals)
                assert self.regularizer is not None
                return float(self.regularizer.compute_regularization(params, mse, n_points))

            callbacks["loss_augmentation"] = loss_augmentation

        # Iteration callback for Layer 4 (Gradient Monitoring)
        if self.monitor:

            def iteration_callback(
                iteration: int,
                params: np.ndarray,
                cost: float,
                gradient: np.ndarray | None = None,
            ) -> None:
                """Monitor gradients for collapse detection."""
                if gradient is not None:
                    assert self.monitor is not None
                    self.monitor.check(gradient, iteration, params, cost)

            callbacks["iteration_callback"] = iteration_callback

        logger.debug(f"Created NLSQ callbacks: {list(callbacks.keys())}")
        return callbacks

    def create_hybrid_streaming_config_kwargs(self) -> dict[str, Any]:
        """Create kwargs for NLSQ's HybridStreamingConfig.

        Returns kwargs that can be used to configure NLSQ's
        AdaptiveHybridStreamingOptimizer with anti-degeneracy features.

        Returns
        -------
        dict
            Configuration kwargs for HybridStreamingConfig:
            - 'enable_group_variance_regularization': bool
            - 'group_variance_lambda': float
            - 'group_variance_indices': list[tuple[int, int]]

        Notes
        -----
        For NLSQ v0.4+, pass these to HybridStreamingConfig constructor.

        Example
        -------
        >>> controller = AntiDegeneracyController.from_config(...)
        >>> kwargs = controller.create_hybrid_streaming_config_kwargs()
        >>> config = HybridStreamingConfig(**kwargs)
        """
        if not self.is_enabled:
            return {}

        kwargs: dict[str, Any] = {}

        # Group variance regularization
        group_indices = self.get_group_variance_indices()
        if group_indices and self.regularizer:
            kwargs["enable_group_variance_regularization"] = True
            kwargs["group_variance_lambda"] = self.regularizer.lambda_value
            kwargs["group_variance_indices"] = group_indices

        # Shear-sensitivity weighting
        if self.shear_weighter is not None:
            kwargs["enable_residual_weighting"] = True
            kwargs["residual_weights"] = self.shear_weighter.get_weights().tolist()

        logger.debug(f"Created HybridStreamingConfig kwargs: {list(kwargs.keys())}")
        return kwargs
