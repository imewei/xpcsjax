"""Configuration for NLSQ optimization in the heterodyne analysis pipeline.

This module defines the full configuration hierarchy for non-linear least squares
fitting of heterodyne XPCS correlation functions:

- ``HybridRecoveryConfig``  — progressive retry / fallback parameters
- ``NLSQValidationConfig``  — post-fit validation thresholds
- ``NLSQConfig``             — master configuration dataclass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

# Per-angle scaling vocabularies. ``PerAngleMode`` is what a user/config may set
# (``"independent"`` is a deprecated alias for ``"individual"``);
# ``ResolvedPerAngleMode`` is the canonical token ``_resolve_effective_mode``
# dispatches on after applying the auto/threshold rules.
PerAngleMode = Literal["individual", "fourier", "auto", "constant", "independent"]
ResolvedPerAngleMode = Literal["constant", "averaged", "fourier", "individual"]

# ---------------------------------------------------------------------------
# Safe type-conversion utilities
# ---------------------------------------------------------------------------

_SENTINEL = object()


def safe_float(value: Any, default: float) -> float:
    """Convert *value* to float, returning *default* on failure.

    Parameters
    ----------
    value
        Arbitrary input that should be numeric.
    default
        Fallback value used when conversion fails.

    Returns
    -------
    float
        ``float(value)`` on success, *default* otherwise (a failed conversion
        is logged as a warning).
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(
            "safe_float: could not convert %r to float, using default %s",
            value,
            default,
        )
        return default


def safe_int(value: Any, default: int) -> int:
    """Convert *value* to int, returning *default* on failure.

    Parameters
    ----------
    value
        Arbitrary input that should be integral.
    default
        Fallback value used when conversion fails.

    Returns
    -------
    int
        ``int(value)`` on success, *default* otherwise (a failed conversion is
        logged as a warning).
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(
            "safe_int: could not convert %r to int, using default %s",
            value,
            default,
        )
        return default


# ---------------------------------------------------------------------------
# HybridRecoveryConfig
# ---------------------------------------------------------------------------

_VALID_WORKFLOWS: frozenset[str] = frozenset({"auto", "auto_global", "hpc"})
_VALID_GOALS: frozenset[str] = frozenset({"fast", "robust", "quality", "memory_efficient"})
_VALID_ANALYSIS_MODES: frozenset[str] = frozenset({"static_ref", "static_both", "two_component"})
_VALID_NLSQ_STABILITY: frozenset[str] = frozenset({"auto", "check", "off"})


@dataclass
class HybridRecoveryConfig:
    """Progressive retry / fallback parameters for NLSQ recovery.

    When a fit fails to converge, the optimizer retries with progressively
    more aggressive regularisation and a smaller trust region.  Each attempt
    *k* (0-based) applies the following scaling to the baseline settings:

    - learning rate  : ``lr_decay  ** k``
    - regularisation : ``lambda_growth ** k``
    - trust radius   : ``trust_decay ** k``

    Attributes
    ----------
    max_retries : int
        Maximum number of recovery attempts before giving up.
    lr_decay : float
        Multiplicative factor applied to the learning rate per retry
        (``< 1`` shrinks the effective step size).
    lambda_growth : float
        Multiplicative factor applied to the regularisation strength per retry
        (``> 1`` increases damping).
    trust_decay : float
        Multiplicative factor applied to the trust-region radius per retry
        (``< 1`` tightens the constraint).
    perturb_scale : float
        Standard deviation of the Gaussian perturbation added to the starting
        parameters before each retry, expressed as a fraction of the parameter
        range.
    """

    max_retries: int = 3
    lr_decay: float = 0.5
    lambda_growth: float = 10.0
    trust_decay: float = 0.5
    perturb_scale: float = 0.1

    def get_retry_settings(self, attempt: int) -> dict[str, float]:
        """Return scaled optimiser settings for a given retry attempt.

        Parameters
        ----------
        attempt
            0-based retry index. ``attempt=0`` returns the unscaled baseline
            (scale factor 1).

        Returns
        -------
        dict of str to float
            Mapping with keys ``"lr_scale"``, ``"lambda_scale"``, and
            ``"trust_radius_scale"``, each a multiplicative factor to apply to
            the corresponding optimiser hyperparameter.

        Raises
        ------
        ValueError
            If *attempt* is negative.
        """
        if attempt < 0:
            raise ValueError(f"attempt must be >= 0, got {attempt}")
        return {
            "lr_scale": self.lr_decay**attempt,
            "lambda_scale": self.lambda_growth**attempt,
            "trust_radius_scale": self.trust_decay**attempt,
        }


# ---------------------------------------------------------------------------
# NLSQValidationConfig
# ---------------------------------------------------------------------------


@dataclass
class NLSQValidationConfig:
    """Thresholds used when validating post-fit quality metrics.

    Attributes
    ----------
    chi2_warn_low : float
        Reduced chi-squared below this value triggers a warning (possible
        over-fitting or under-estimated errors).
    chi2_warn_high : float
        Reduced chi-squared above this value triggers a warning (possible
        under-fitting or an under-estimated model).
    chi2_fail_high : float
        Reduced chi-squared above this value is treated as a hard failure.
    max_relative_uncertainty : float
        Maximum acceptable relative uncertainty (``sigma / |param|``) for any
        fitted parameter; ``1.0`` means 100%.
    correlation_warn : float
        Off-diagonal correlation-coefficient magnitude above this threshold
        triggers a collinearity warning.
    """

    # Reduced chi-squared thresholds
    chi2_warn_low: float = 0.5
    chi2_warn_high: float = 2.0
    chi2_fail_high: float = 10.0

    # Uncertainty validation
    max_relative_uncertainty: float = 1.0  # 100 %

    # Correlation threshold for parameters
    correlation_warn: float = 0.95


# ---------------------------------------------------------------------------
# StratificationConfig
# ---------------------------------------------------------------------------


@dataclass
class StratificationConfig:
    """Angle-stratified chunking settings for the heterodyne NLSQ path.

    Mirrors the upstream homodyne wrapper
    (``homodyne/optimization/nlsq/wrapper.py::_apply_stratification_if_needed``),
    where the block lives at ``config.config["optimization"]["stratification"]``
    -- a SIBLING of ``optimization.nlsq``, not nested inside it.  Field names and
    defaults match homodyne 1:1.

    Attributes
    ----------
    enabled : bool or str
        ``False`` disables stratification entirely. ``"auto"`` (default) and
        ``True`` are equivalent for heterodyne: both defer to
        ``should_use_stratification`` plus the ``>=1M`` solver gate. There is no
        separate force-on path -- the ``>=1M`` gate is the real control
        (stratification only swaps the solver, which engages at ``>=1M``
        regardless of this flag).
    target_chunk_size : int
        Target scalar count per stratified chunk, forwarded to
        ``fit_heterodyne_stratified_least_squares(target_chunk_size=...)``.
    max_imbalance_ratio : float
        Maximum tolerated angle-count imbalance before the stratified path is
        skipped. This is the SOLE imbalance gate in ``_fit_nlsq_heterodyne``:
        ``should_use_stratification`` is called with ``imbalance_ratio=0.0`` so
        its internal hard-coded 5.0 cutoff cannot pre-empt this value, which
        therefore moves the cutoff in either direction (tighten below 5.0 or
        loosen above it).
    force_sequential_fallback : bool
        Accepted for homodyne-config compatibility but **inert for
        heterodyne.** With ``individual`` mode scoped out of stratified-LS (it
        already uses the sequential per-angle path in ``heterodyne_core``),
        there is no heterodyne behavior this knob maps to. Parsed so a shared
        homodyne/heterodyne config does not trip the "unrecognised key"
        warning; it has no effect on the heterodyne path.
    check_memory_safety : bool
        When ``True``, the heterodyne stratified-LS driver consults the memory
        estimate's ``is_safe`` flag and logs a (non-fatal) warning if the
        projected peak exceeds the safe RAM fraction. When ``False``, the
        warning is suppressed.
    use_index_based : bool
        Threaded into the stratified-LS driver's diagnostics and memory
        estimate. Heterodyne is structurally index-based, so the value is
        informational, but the recorded diagnostic reflects this config setting
        rather than a hard-coded literal.
    """

    enabled: bool | str = "auto"
    target_chunk_size: int = 100_000
    max_imbalance_ratio: float = 5.0
    force_sequential_fallback: bool = False
    check_memory_safety: bool = True
    use_index_based: bool = False

    @classmethod
    def from_optimization_block(cls, opt_block: dict[str, Any] | None) -> StratificationConfig:
        """Parse the ``optimization`` block's ``stratification`` sub-dict.

        Accepts the full ``optimization`` block (``opt_block["stratification"]``)
        and resolves homodyne-mirrored defaults for any missing fields.  A missing
        or non-dict ``stratification`` entry yields all defaults.

        Parameters
        ----------
        opt_block
            The ``config.config["optimization"]`` mapping, or ``None``.

        Returns
        -------
        StratificationConfig
            A fully resolved configuration.
        """
        strat: dict[str, Any] = {}
        if isinstance(opt_block, dict):
            candidate = opt_block.get("stratification")
            if isinstance(candidate, dict):
                strat = candidate
        return cls(
            enabled=strat.get("enabled", "auto"),
            target_chunk_size=int(strat.get("target_chunk_size", 100_000)),
            max_imbalance_ratio=float(strat.get("max_imbalance_ratio", 5.0)),
            force_sequential_fallback=bool(strat.get("force_sequential_fallback", False)),
            check_memory_safety=bool(strat.get("check_memory_safety", True)),
            use_index_based=bool(strat.get("use_index_based", False)),
        )

    def is_disabled(self) -> bool:
        """Return ``True`` if stratification is explicitly turned off.

        Mirrors homodyne's check: ``enabled`` is the boolean ``False`` or the
        case-insensitive string ``"false"``.
        """
        if self.enabled is False:
            return True
        return isinstance(self.enabled, str) and self.enabled.lower() == "false"


# ---------------------------------------------------------------------------
# NLSQConfig
# ---------------------------------------------------------------------------


@dataclass
class NLSQConfig:
    """Master configuration for NLSQ fitting of heterodyne XPCS data.

    The heterodyne model has 14 parameters organised into two-component
    (signal + background) correlation functions.  This configuration covers
    the full pipeline: solver hyperparameters, multi-start, streaming /
    chunking, recovery on failure, and post-fit diagnostics.

    Notes
    -----
    Only the most load-bearing fields are documented below; the additional
    Fourier-reparameterization, hierarchical, regularization, gradient-monitor,
    CMA-ES, hybrid-streaming, and multi-start knobs are grouped inline at their
    declarations. The ``per_angle_mode`` field accepts the user-facing tokens
    ``"individual"``, ``"fourier"``, ``"auto"``, and ``"constant"``
    (``"independent"`` is a deprecated alias for ``"individual"``, normalised in
    :meth:`__post_init__`); ``"averaged"`` is *not* a user input — it is an
    internally resolved mode that ``auto`` produces when
    ``n_phi >= constant_scaling_threshold`` (default 3), falling back to
    ``"individual"`` below that threshold.

    This is the NLSQ-only solver config: there is intentionally no Bayesian /
    MCMC pathway, so no sampler fields exist here.

    Attributes
    ----------
    max_iterations : int
        Maximum number of optimiser iterations per fit.
    tolerance : float
        Convergence tolerance for the cost function.
    method : {"trf", "lm", "dogbox"}
        Trust-region algorithm variant passed to the ``nlsq CurveFit``
        optimizer. ``dogbox`` is coerced to ``trf`` by the strategy layer.
    multistart : bool
        Whether to run multi-start optimisation to avoid local minima.
    multistart_n : int
        Number of random starting points when *multistart* is enabled.
    verbose : int
        Verbosity level forwarded to the solver (0 = silent, 1 = summary,
        2 = detailed).
    use_jac : bool
        Whether to supply an analytic Jacobian to the solver.
    x_scale : str or list of float
        Parameter scaling strategy. ``"jac"`` uses the Jacobian diagonal; a
        list of floats provides explicit per-parameter scales.
    ftol : float
        Relative tolerance on the cost-function change.
    xtol : float
        Relative tolerance on the parameter step norm.
    gtol : float
        Absolute tolerance on the projected gradient norm.
    loss : {"linear", "soft_l1", "huber", "cauchy", "arctan"}
        Robust loss-function kernel.
    diff_step : float or None
        Finite-difference step size. ``None`` selects the solver default.
    max_nfev : int or None
        Per-angle cap on function evaluations passed to the underlying solver.
        ``None`` is unlimited.

        .. important::
           For the multi-angle joint-fit paths
           (``_fit_joint_constant_multi_phi`` in
           ``heterodyne_constant_mode.py``, ``_fit_joint_averaged_multi_phi``
           and ``_fit_joint_multi_phi`` in ``heterodyne_core.py``) the
           effective solver budget is ``max_nfev * n_phi`` — those paths run a
           single combined least-squares problem whose residual vector
           concatenates all angles, so the per-call cap is scaled by ``n_phi``
           to give each angle the same iteration budget it would have under
           independent fits. Single-angle paths (``_fit_local``,
           ``_fit_multistart``) pass ``max_nfev`` through unchanged.
    chunk_size : int or None
        Number of q-points per processing chunk. ``None`` means auto-select
        based on available memory.
    workflow : str
        High-level workflow preset; one of ``"auto"``, ``"auto_global"``,
        ``"hpc"``.
    goal : str
        Optimisation goal preset controlling the balance between speed,
        robustness, and solution quality; one of ``"fast"``, ``"robust"``,
        ``"quality"``, ``"memory_efficient"``.
    enable_streaming : bool
        Process data in a streaming fashion (chunk-by-chunk) rather than
        loading all q-points at once.
    streaming_chunk_size : int
        Number of q-points per streaming chunk when *enable_streaming* is
        ``True``.
    enable_stratified : bool
        Use stratified sampling across q-point subsets.
    target_chunk_size : int
        Target number of data points per stratified chunk.
    enable_recovery : bool
        Automatically retry failed fits with more aggressive regularisation
        (see *recovery_config*).
    max_recovery_attempts : int
        Maximum retries before a fit is declared failed.
    recovery_config : HybridRecoveryConfig
        Per-retry scaling parameters.
    enable_diagnostics : bool
        Emit structured convergence / quality diagnostics after each fit.
    enable_anti_degeneracy : bool
        Apply anti-degeneracy constraints to prevent parameter collapse (e.g.
        two identical relaxation modes).
    x_scale_map : dict of str to float
        Per-parameter scale overrides keyed by parameter name. Entries here are
        merged into (and override) the default Jacobian-based scaling.
    loss_weights : list of float or None
        Per-data-point loss weights. ``None`` uses uniform weighting.
    loss_scale : float
        Global scale factor applied to the loss-function value before passing
        to the solver.
    tr_solver : str or None
        Trust-region sub-problem solver override (``"exact"``, ``"lsmr"``, or
        ``None`` for the solver default).
    step_bound : float
        Upper bound on the step norm relative to the trust radius. ``0.0``
        defers to the solver default.
    per_angle_mode : {"individual", "fourier", "auto", "constant", "independent"}
        Per-angle scaling layout (see Notes). ``"independent"`` is a deprecated
        alias for ``"individual"``.
    constant_scaling_threshold : int
        ``n_phi`` threshold (default 3) at or above which ``auto`` resolves to
        the ``"averaged"`` scaling layout; below it ``auto`` uses
        ``"individual"``.
    use_nlsq_library : bool
        Prefer the ``nlsq`` library over the scipy fallback.
    n_params : int
        Number of model parameters. Fixed at 14 for heterodyne.
    analysis_mode : AnalysisMode
        Which physical model variant to use — one of ``"static_ref"``
        (reference beam treated as static background), ``"static_both"`` (both
        beams treated as static), or ``"two_component"`` (full two-component
        model, default).
    validation : NLSQValidationConfig
        Post-fit validation thresholds.
    execute_layers : bool
        Opt-in gate for the L2/L3 anti-degeneracy ESCAPE on the >=1M stratified-LS
        path. Default ``False`` runs the single baseline solve (byte-identical to
        the pre-escape path). ``True`` runs the keep-better-guarded hierarchical
        (+ regularization) escape after the baseline — it is EXPENSIVE (~3-5x the
        baseline fit wall-time) and never returns a worse result than the baseline,
        so enable it only for a genuinely-stuck / degenerate fit.
    """

    # ------------------------------------------------------------------
    # Existing / core solver fields
    # ------------------------------------------------------------------

    max_iterations: int = 1000
    tolerance: float = 1e-8
    method: Literal["trf", "lm", "dogbox"] = "trf"
    multistart: bool = False
    multistart_n: int = 10
    verbose: int = 1
    use_jac: bool = True
    x_scale: str | list[float] = "jac"
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1"

    # Advanced solver options
    diff_step: float | None = None
    max_nfev: int | None = None

    # Memory management
    chunk_size: int | None = None  # None for auto

    # ------------------------------------------------------------------
    # Workflow / goal presets
    # ------------------------------------------------------------------

    workflow: str = "auto"
    goal: str = "robust"

    # ------------------------------------------------------------------
    # Streaming and stratified sampling
    # ------------------------------------------------------------------

    enable_streaming: bool = False
    streaming_chunk_size: int = 50000
    enable_stratified: bool = False
    target_chunk_size: int = 10000

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    enable_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_config: HybridRecoveryConfig = field(default_factory=HybridRecoveryConfig)

    # ------------------------------------------------------------------
    # Diagnostics and anti-degeneracy
    # ------------------------------------------------------------------

    enable_diagnostics: bool = True
    enable_anti_degeneracy: bool = True

    # ------------------------------------------------------------------
    # Loss and scaling overrides
    # ------------------------------------------------------------------

    x_scale_map: dict[str, float] = field(default_factory=dict)
    loss_weights: list[float] | None = None
    loss_scale: float = 1.0
    tr_solver: str | None = None
    step_bound: float = 0.0

    # ------------------------------------------------------------------
    # Fourier reparameterization for per-angle scaling
    # ------------------------------------------------------------------

    per_angle_mode: PerAngleMode = "auto"
    fourier_order: int = 2
    fourier_auto_threshold: int = 6

    # ------------------------------------------------------------------
    # Hierarchical optimization
    # ------------------------------------------------------------------

    enable_hierarchical: bool = False
    # Opt-in L2/L3 anti-degeneracy escape on the >=1M stratified-LS path (default
    # OFF = byte-identical single solve; True = keep-better hierarchical escape,
    # expensive ~3-5x baseline wall-time). See the field docstring above.
    execute_layers: bool = False
    hierarchical_max_outer_iterations: int = 20
    hierarchical_inner_tolerance: float = 1e-6
    hierarchical_outer_tolerance: float = 1e-4

    # ------------------------------------------------------------------
    # Adaptive regularization
    # ------------------------------------------------------------------

    regularization_mode: Literal["none", "tikhonov", "adaptive"] = "none"
    group_variance_lambda: float = 0.01
    regularization_target_cv: float = 0.5

    # ------------------------------------------------------------------
    # Gradient collapse detection
    # ------------------------------------------------------------------

    enable_gradient_monitoring: bool = False
    gradient_ratio_threshold: float = 100.0
    gradient_consecutive_triggers: int = 3

    # ------------------------------------------------------------------
    # CMA-ES global search
    # ------------------------------------------------------------------

    enable_cmaes: bool = False
    cmaes_sigma0: float = 0.3
    cmaes_max_iterations: int = 1000
    cmaes_population_size: int | None = None
    cmaes_tolx: float = 1e-6
    cmaes_tolfun: float = 1e-8
    cmaes_diagonal_filtering: str = "remove"
    cmaes_anti_degeneracy: bool = False
    cmaes_warmstart_auto_skip: bool = True
    cmaes_warmstart_skip_threshold: float = 5.0
    cmaes_restart_strategy: str = "bipop"
    cmaes_max_restarts: int = 9

    # ------------------------------------------------------------------
    # Hybrid streaming optimizer
    # ------------------------------------------------------------------

    hybrid_enable: bool = False
    hybrid_warmup_fraction: float = 0.1
    hybrid_normalization: bool = True
    hybrid_method: Literal["lbfgs", "gauss_newton"] = "gauss_newton"
    hybrid_lbfgs_memory: int = 10
    hybrid_convergence_window: int = 5
    hybrid_convergence_threshold: float = 1e-6
    hybrid_max_phases: int = 4

    # ------------------------------------------------------------------
    # Multi-start extensions
    # ------------------------------------------------------------------

    sampling_strategy: Literal["lhs", "sobol", "random"] = "lhs"
    screen_keep_fraction: float = 0.5
    refine_top_k: int = 3

    # ------------------------------------------------------------------
    # Scaling threshold
    # ------------------------------------------------------------------

    constant_scaling_threshold: int = 3

    # ------------------------------------------------------------------
    # Backend and model identity
    # ------------------------------------------------------------------

    use_nlsq_library: bool = True
    n_params: int = 14  # heterodyne: 14 parameters
    analysis_mode: AnalysisMode = AnalysisMode.TWO_COMPONENT

    # ------------------------------------------------------------------
    # NLSQ package integration (mirrors homodyne wrapper.py)
    # ------------------------------------------------------------------

    nlsq_stability: str = "auto"  # 'auto', 'check', or 'off'
    nlsq_rescale_data: bool = False  # xdata is indices, not physical
    nlsq_x_scale: str | np.ndarray = "jac"  # trust-region scaling
    nlsq_memory_fraction: float = 0.75  # fraction of RAM for NLSQ
    nlsq_memory_fallback_gb: float = 16.0  # fallback if detection fails

    # ------------------------------------------------------------------
    # Post-fit validation
    # ------------------------------------------------------------------

    validation: NLSQValidationConfig = field(default_factory=NLSQValidationConfig)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate invariants that must hold immediately after construction."""
        # Deprecation alias: 'independent' → 'individual' (homodyne's canonical
        # vocabulary). Emit a DeprecationWarning and normalize so downstream
        # code only ever sees 'individual'. Slated for future removal.
        # TODO: remove deprecation alias
        if self.per_angle_mode == "independent":
            import warnings

            warnings.warn(
                "per_angle_mode='independent' is deprecated; use 'individual' "
                "(matches homodyne's canonical name). 'independent' will be "
                "removed in a future release.",
                DeprecationWarning,
                stacklevel=3,  # user -> synthesized __init__ -> __post_init__ -> warnings.warn
            )
            self.per_angle_mode = "individual"

        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.multistart_n < 1:
            raise ValueError("multistart_n must be >= 1")
        if self.streaming_chunk_size < 1:
            raise ValueError("streaming_chunk_size must be >= 1")
        if self.target_chunk_size < 1:
            raise ValueError("target_chunk_size must be >= 1")
        if self.max_recovery_attempts < 0:
            raise ValueError("max_recovery_attempts must be >= 0")
        if self.loss_scale <= 0:
            raise ValueError("loss_scale must be positive")
        if self.hierarchical_max_outer_iterations < 1:
            raise ValueError("hierarchical_max_outer_iterations must be >= 1")
        if self.gradient_consecutive_triggers < 1:
            raise ValueError("gradient_consecutive_triggers must be >= 1")
        if self.cmaes_sigma0 <= 0:
            raise ValueError("cmaes_sigma0 must be > 0")
        if self.cmaes_diagonal_filtering not in ("remove", "none"):
            raise ValueError(
                f"cmaes_diagonal_filtering must be 'remove' or 'none', "
                f"got {self.cmaes_diagonal_filtering!r}"
            )
        if self.cmaes_warmstart_skip_threshold <= 0:
            raise ValueError("cmaes_warmstart_skip_threshold must be > 0")
        if self.cmaes_restart_strategy not in ("bipop", "none"):
            raise ValueError(
                f"cmaes_restart_strategy must be 'bipop' or 'none', "
                f"got {self.cmaes_restart_strategy!r}"
            )
        if self.cmaes_max_restarts < 0:
            raise ValueError("cmaes_max_restarts must be >= 0")
        if not (0 < self.hybrid_warmup_fraction < 1):
            raise ValueError("hybrid_warmup_fraction must be in (0, 1)")
        if not (0 < self.screen_keep_fraction <= 1):
            raise ValueError("screen_keep_fraction must be in (0, 1]")
        if self.refine_top_k < 1:
            raise ValueError("refine_top_k must be >= 1")
        if self.constant_scaling_threshold >= self.fourier_auto_threshold:
            raise ValueError(
                f"constant_scaling_threshold ({self.constant_scaling_threshold}) must be "
                f"< fourier_auto_threshold ({self.fourier_auto_threshold}): "
                f"the auto-dispatch 'averaged' range would be empty or inverted"
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of configuration error strings.

        An empty list means the configuration is consistent.  Callers should
        treat a non-empty list as a hard error before launching a fit.

        Returns
        -------
        list of str
            Human-readable error strings, one per violation found.
        """
        errors: list[str] = []

        if self.workflow not in _VALID_WORKFLOWS:
            errors.append(
                f"workflow={self.workflow!r} is not valid; "
                f"must be one of {sorted(_VALID_WORKFLOWS)}"
            )

        if self.goal not in _VALID_GOALS:
            errors.append(f"goal={self.goal!r} is not valid; must be one of {sorted(_VALID_GOALS)}")

        if self.tolerance <= 0:
            errors.append(f"tolerance={self.tolerance} must be > 0")

        if self.streaming_chunk_size <= 0:
            errors.append(f"streaming_chunk_size={self.streaming_chunk_size} must be > 0")

        if self.analysis_mode not in _VALID_ANALYSIS_MODES:
            errors.append(
                f"analysis_mode={self.analysis_mode!r} is not valid; "
                f"must be one of {sorted(_VALID_ANALYSIS_MODES)}"
            )

        valid_per_angle_modes = (
            "individual",
            "fourier",
            "auto",
            "constant",
            "independent",
        )
        # Note: 'independent' is a deprecated alias for 'individual'; it is
        # normalized in __post_init__ and should not appear here at runtime.
        # We include it in the accepted tuple to keep validate() robust against
        # callers that construct an NLSQConfig and then mutate per_angle_mode.
        if self.per_angle_mode not in valid_per_angle_modes:
            user_facing_modes = ("individual", "fourier", "auto", "constant")
            errors.append(
                f"per_angle_mode={self.per_angle_mode!r} is not valid; "
                f"must be one of {user_facing_modes}"
            )
        if self.fourier_order < 1:
            errors.append(f"fourier_order={self.fourier_order} must be >= 1")
        if self.fourier_auto_threshold < 1:
            errors.append(f"fourier_auto_threshold={self.fourier_auto_threshold} must be >= 1")
        if self.constant_scaling_threshold >= self.fourier_auto_threshold:
            errors.append(
                f"constant_scaling_threshold={self.constant_scaling_threshold} must be "
                f"< fourier_auto_threshold={self.fourier_auto_threshold}: "
                f"the auto-dispatch 'averaged' range would be empty or inverted"
            )

        valid_regularization_modes = ("none", "tikhonov", "adaptive")
        if self.regularization_mode not in valid_regularization_modes:
            errors.append(
                f"regularization_mode={self.regularization_mode!r} is not valid; "
                f"must be one of {valid_regularization_modes}"
            )

        valid_hybrid_methods = ("lbfgs", "gauss_newton")
        if self.hybrid_method not in valid_hybrid_methods:
            errors.append(
                f"hybrid_method={self.hybrid_method!r} is not valid; "
                f"must be one of {valid_hybrid_methods}"
            )

        valid_sampling_strategies = ("lhs", "sobol", "random")
        if self.sampling_strategy not in valid_sampling_strategies:
            errors.append(
                f"sampling_strategy={self.sampling_strategy!r} is not valid; "
                f"must be one of {valid_sampling_strategies}"
            )

        if self.nlsq_stability not in _VALID_NLSQ_STABILITY:
            errors.append(
                f"nlsq_stability={self.nlsq_stability!r} is not valid; "
                f"must be one of {sorted(_VALID_NLSQ_STABILITY)}"
            )

        if not (0 < self.nlsq_memory_fraction <= 1):
            errors.append(f"nlsq_memory_fraction={self.nlsq_memory_fraction} must be in (0, 1]")

        if self.nlsq_memory_fallback_gb <= 0:
            errors.append(f"nlsq_memory_fallback_gb={self.nlsq_memory_fallback_gb} must be > 0")

        # Fields validated in __post_init__ that may be mutated after construction
        if self.max_iterations < 1:
            errors.append(f"max_iterations={self.max_iterations} must be >= 1")
        if self.multistart_n < 1:
            errors.append(f"multistart_n={self.multistart_n} must be >= 1")
        if self.max_recovery_attempts < 0:
            errors.append(f"max_recovery_attempts={self.max_recovery_attempts} must be >= 0")
        if self.loss_scale <= 0:
            errors.append(f"loss_scale={self.loss_scale} must be > 0")
        if self.cmaes_sigma0 <= 0:
            errors.append(f"cmaes_sigma0={self.cmaes_sigma0} must be > 0")
        if self.gradient_consecutive_triggers < 1:
            errors.append(
                f"gradient_consecutive_triggers={self.gradient_consecutive_triggers} must be >= 1"
            )

        # Advisory warnings — not errors, but worth surfacing at validate() time
        if self.gtol < 1e-7 and self.loss != "linear":
            logger.warning(
                "NLSQConfig: gtol=%.2e is very tight for loss=%r. "
                "Robust loss landscapes are harder — consider gtol >= 1e-6 "
                "to avoid premature max_nfev exhaustion.",
                self.gtol,
                self.loss,
            )

        if self.max_nfev is None:
            logger.debug(
                "NLSQConfig: max_nfev=None — nlsq defaults to 100×n_params "
                "(e.g. 1400 for 14 params). Set explicitly to override.",
            )

        return errors

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> NLSQConfig:
        """Construct an ``NLSQConfig`` from a plain dictionary.

        Nested sub-dictionaries under ``"recovery"`` and ``"validation"``
        are automatically parsed into their respective dataclasses.
        Unrecognised top-level keys are logged as warnings and ignored.

        Parameters
        ----------
        config
            Flat or nested configuration dictionary, e.g. loaded from a YAML
            file.

        Returns
        -------
        NLSQConfig
            A fully populated instance.
        """
        known_scalar_fields: dict[str, str] = {
            # Core solver
            "max_iterations": "int",
            "tolerance": "float",
            "method": "str",
            "multistart": "bool",
            "multistart_n": "int",
            "verbose": "int",
            "use_jac": "bool",
            "x_scale": "passthrough",  # str or list — handled separately
            "ftol": "float",
            "xtol": "float",
            "gtol": "float",
            "loss": "str",
            "diff_step": "float_or_none",
            "max_nfev": "int_or_none",
            "chunk_size": "int_or_none",
            # Workflow / goal
            "workflow": "str",
            "goal": "str",
            # Streaming / stratified
            "enable_streaming": "bool",
            "streaming_chunk_size": "int",
            "enable_stratified": "bool",
            "target_chunk_size": "int",
            # Recovery
            "enable_recovery": "bool",
            "max_recovery_attempts": "int",
            # Diagnostics
            "enable_diagnostics": "bool",
            "enable_anti_degeneracy": "bool",
            # Loss / scaling
            "loss_weights": "passthrough",  # list[float] | None
            "loss_scale": "float",
            "tr_solver": "str_or_none",
            "step_bound": "float",
            # Fourier reparameterization
            "per_angle_mode": "str",
            "fourier_order": "int",
            "fourier_auto_threshold": "int",
            # Hierarchical optimization
            "enable_hierarchical": "bool",
            "execute_layers": "bool",
            "hierarchical_max_outer_iterations": "int",
            "hierarchical_inner_tolerance": "float",
            "hierarchical_outer_tolerance": "float",
            # Adaptive regularization
            "regularization_mode": "str",
            "group_variance_lambda": "float",
            "regularization_target_cv": "float",
            # Gradient collapse detection
            "enable_gradient_monitoring": "bool",
            "gradient_ratio_threshold": "float",
            "gradient_consecutive_triggers": "int",
            # CMA-ES global search
            "enable_cmaes": "bool",
            "cmaes_sigma0": "float",
            "cmaes_max_iterations": "int",
            "cmaes_population_size": "int_or_none",
            "cmaes_tolx": "float",
            "cmaes_tolfun": "float",
            "cmaes_diagonal_filtering": "str",
            "cmaes_anti_degeneracy": "bool",
            "cmaes_warmstart_auto_skip": "bool",
            "cmaes_warmstart_skip_threshold": "float",
            "cmaes_restart_strategy": "str",
            "cmaes_max_restarts": "int",
            # Hybrid streaming optimizer
            "hybrid_enable": "bool",
            "hybrid_warmup_fraction": "float",
            "hybrid_normalization": "bool",
            "hybrid_method": "str",
            "hybrid_lbfgs_memory": "int",
            "hybrid_convergence_window": "int",
            "hybrid_convergence_threshold": "float",
            "hybrid_max_phases": "int",
            # Multi-start extensions
            "sampling_strategy": "str",
            "screen_keep_fraction": "float",
            "refine_top_k": "int",
            # Scaling threshold
            "constant_scaling_threshold": "int",
            # Backend / model
            "use_nlsq_library": "bool",
            "n_params": "int",
            "analysis_mode": "str",
            # NLSQ package integration
            "nlsq_stability": "str",
            "nlsq_rescale_data": "bool",
            "nlsq_x_scale": "passthrough",  # str or np.ndarray
            "nlsq_memory_fraction": "float",
            "nlsq_memory_fallback_gb": "float",
        }

        normalized_config = dict(config)

        def _set_from_nested(field_name: str, value: Any) -> None:
            if value is not _SENTINEL and field_name not in normalized_config:
                normalized_config[field_name] = value

        raw_anti_degeneracy = config.get("anti_degeneracy")
        if isinstance(raw_anti_degeneracy, dict):
            _set_from_nested(
                "per_angle_mode",
                raw_anti_degeneracy.get("per_angle_mode", _SENTINEL),
            )
            _set_from_nested(
                "fourier_order",
                raw_anti_degeneracy.get("fourier_order", _SENTINEL),
            )
            _set_from_nested(
                "fourier_auto_threshold",
                raw_anti_degeneracy.get("fourier_auto_threshold", _SENTINEL),
            )
            _set_from_nested(
                "constant_scaling_threshold",
                raw_anti_degeneracy.get("constant_scaling_threshold", _SENTINEL),
            )
            _set_from_nested(
                "execute_layers",
                raw_anti_degeneracy.get("execute_layers", _SENTINEL),
            )

            hierarchical = raw_anti_degeneracy.get("hierarchical")
            if isinstance(hierarchical, dict):
                _set_from_nested("enable_hierarchical", hierarchical.get("enable", _SENTINEL))
                _set_from_nested(
                    "hierarchical_max_outer_iterations",
                    hierarchical.get("max_outer_iterations", _SENTINEL),
                )
                _set_from_nested(
                    "hierarchical_inner_tolerance",
                    hierarchical.get("inner_tolerance", _SENTINEL),
                )
                _set_from_nested(
                    "hierarchical_outer_tolerance",
                    hierarchical.get("outer_tolerance", _SENTINEL),
                )
            elif hierarchical is not None:
                logger.warning(
                    "NLSQConfig.from_dict: anti_degeneracy.hierarchical must be a "
                    "dict, got %r — ignoring",
                    type(hierarchical).__name__,
                )

            regularization = raw_anti_degeneracy.get("regularization")
            if isinstance(regularization, dict):
                _set_from_nested("regularization_mode", regularization.get("mode", _SENTINEL))
                _set_from_nested("group_variance_lambda", regularization.get("lambda", _SENTINEL))
                _set_from_nested(
                    "regularization_target_cv",
                    regularization.get("target_cv", _SENTINEL),
                )
            elif regularization is not None:
                logger.warning(
                    "NLSQConfig.from_dict: anti_degeneracy.regularization must be a "
                    "dict, got %r — ignoring",
                    type(regularization).__name__,
                )

            gradient_monitoring = raw_anti_degeneracy.get("gradient_monitoring")
            if isinstance(gradient_monitoring, dict):
                _set_from_nested(
                    "enable_gradient_monitoring",
                    gradient_monitoring.get("enable", _SENTINEL),
                )
                _set_from_nested(
                    "gradient_ratio_threshold",
                    gradient_monitoring.get("ratio_threshold", _SENTINEL),
                )
                _set_from_nested(
                    "gradient_consecutive_triggers",
                    gradient_monitoring.get("consecutive_triggers", _SENTINEL),
                )
            elif gradient_monitoring is not None:
                logger.warning(
                    "NLSQConfig.from_dict: anti_degeneracy.gradient_monitoring must "
                    "be a dict, got %r — ignoring",
                    type(gradient_monitoring).__name__,
                )
        elif raw_anti_degeneracy is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'anti_degeneracy' must be a dict, got %r — ignoring",
                type(raw_anti_degeneracy).__name__,
            )

        raw_cmaes = config.get("cmaes")
        if isinstance(raw_cmaes, dict):
            _set_from_nested("enable_cmaes", raw_cmaes.get("enable", _SENTINEL))
            _set_from_nested(
                "cmaes_sigma0",
                raw_cmaes.get("sigma", raw_cmaes.get("sigma0", _SENTINEL)),
            )
            _set_from_nested(
                "cmaes_max_iterations",
                raw_cmaes.get(
                    "max_generations",
                    raw_cmaes.get("max_iterations", _SENTINEL),
                ),
            )
            _set_from_nested(
                "cmaes_population_size",
                raw_cmaes.get("popsize", raw_cmaes.get("population_size", _SENTINEL)),
            )
            _set_from_nested("cmaes_tolx", raw_cmaes.get("tol_x", raw_cmaes.get("tolx", _SENTINEL)))
            _set_from_nested(
                "cmaes_tolfun",
                raw_cmaes.get("tol_fun", raw_cmaes.get("tolfun", _SENTINEL)),
            )
            _set_from_nested(
                "cmaes_diagonal_filtering",
                raw_cmaes.get("diagonal_filtering", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_anti_degeneracy",
                raw_cmaes.get("anti_degeneracy", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_warmstart_auto_skip",
                raw_cmaes.get("warmstart_auto_skip", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_warmstart_skip_threshold",
                raw_cmaes.get("warmstart_skip_threshold", _SENTINEL),
            )
        elif raw_cmaes is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'cmaes' must be a dict, got %r — ignoring",
                type(raw_cmaes).__name__,
            )

        nested_keys = {
            "recovery",
            "validation",
            "x_scale_map",
            "anti_degeneracy",
            "cmaes",
        }

        # Canonical ``optimization.nlsq`` template sections that the heterodyne
        # solver-config deliberately does not translate into solver scalars.
        # They are owned by other layers — memory/strategy routing and the
        # shared NLSQ config in ``config.py`` consume ``memory_fraction``,
        # ``trust_region_scale``, ``hybrid_streaming``, ``quality_validation``,
        # ``diagnostics`` and ``progress``; heterodyne multi-start is not yet
        # wired (so ``multi_start`` is intentionally inert here). Listing them
        # keeps the "unrecognised key" warning a genuine typo detector instead
        # of firing for every documented template section on every run.
        known_ignored_keys = {
            "memory_fraction",
            "trust_region_scale",
            "progress",
            "diagnostics",
            "multi_start",
            "hybrid_streaming",
            "quality_validation",
        }

        # Warn on unrecognised keys
        all_known = set(known_scalar_fields) | nested_keys | known_ignored_keys
        for key in normalized_config:
            if key not in all_known:
                logger.warning("NLSQConfig.from_dict: unrecognised key %r — ignoring", key)

        kwargs: dict[str, Any] = {}

        # --- Parse scalar fields -----------------------------------------
        for field_name, kind in known_scalar_fields.items():
            raw = normalized_config.get(field_name, _SENTINEL)
            if raw is _SENTINEL:
                continue  # use dataclass default

            if kind == "float":
                kwargs[field_name] = safe_float(raw, 0.0)
            elif kind == "int":
                kwargs[field_name] = safe_int(raw, 0)
            elif kind == "bool":
                kwargs[field_name] = bool(raw)
            elif kind == "str":
                kwargs[field_name] = str(raw)
            elif kind == "float_or_none":
                kwargs[field_name] = None if raw is None else safe_float(raw, 0.0)
            elif kind == "int_or_none":
                kwargs[field_name] = None if raw is None else safe_int(raw, 0)
            elif kind == "str_or_none":
                kwargs[field_name] = None if raw is None else str(raw)
            elif kind == "passthrough":
                kwargs[field_name] = raw
            # no else branch needed — exhaustive set above

        # --- Parse x_scale_map -------------------------------------------
        raw_scale_map = normalized_config.get("x_scale_map")
        if isinstance(raw_scale_map, dict):
            kwargs["x_scale_map"] = {str(k): safe_float(v, 1.0) for k, v in raw_scale_map.items()}
        elif raw_scale_map is not None:
            logger.warning(
                "NLSQConfig.from_dict: x_scale_map must be a dict, got %r — ignoring",
                type(raw_scale_map).__name__,
            )

        # --- Parse nested recovery sub-dict ------------------------------
        raw_recovery = normalized_config.get("recovery")
        if isinstance(raw_recovery, dict):
            recovery = HybridRecoveryConfig(
                max_retries=safe_int(
                    raw_recovery.get("max_retries"), HybridRecoveryConfig.max_retries
                ),
                lr_decay=safe_float(raw_recovery.get("lr_decay"), HybridRecoveryConfig.lr_decay),
                lambda_growth=safe_float(
                    raw_recovery.get("lambda_growth"),
                    HybridRecoveryConfig.lambda_growth,
                ),
                trust_decay=safe_float(
                    raw_recovery.get("trust_decay"), HybridRecoveryConfig.trust_decay
                ),
                perturb_scale=safe_float(
                    raw_recovery.get("perturb_scale"),
                    HybridRecoveryConfig.perturb_scale,
                ),
            )
            kwargs["recovery_config"] = recovery
        elif raw_recovery is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'recovery' must be a dict, got %r — ignoring",
                type(raw_recovery).__name__,
            )

        # --- Parse nested validation sub-dict ----------------------------
        raw_validation = normalized_config.get("validation")
        if isinstance(raw_validation, dict):
            defaults = NLSQValidationConfig()
            validation = NLSQValidationConfig(
                chi2_warn_low=safe_float(
                    raw_validation.get("chi2_warn_low"), defaults.chi2_warn_low
                ),
                chi2_warn_high=safe_float(
                    raw_validation.get("chi2_warn_high"), defaults.chi2_warn_high
                ),
                chi2_fail_high=safe_float(
                    raw_validation.get("chi2_fail_high"), defaults.chi2_fail_high
                ),
                max_relative_uncertainty=safe_float(
                    raw_validation.get("max_relative_uncertainty"),
                    defaults.max_relative_uncertainty,
                ),
                correlation_warn=safe_float(
                    raw_validation.get("correlation_warn"), defaults.correlation_warn
                ),
            )
            kwargs["validation"] = validation
        elif raw_validation is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'validation' must be a dict, got %r — ignoring",
                type(raw_validation).__name__,
            )

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the configuration to a plain dictionary.

        Nested dataclasses are serialised as nested dicts, making the output
        suitable for round-tripping through YAML / JSON.

        Returns
        -------
        dict
            A fully populated dictionary representation.
        """
        return {
            # Core solver
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "method": self.method,
            "multistart": self.multistart,
            "multistart_n": self.multistart_n,
            "verbose": self.verbose,
            "use_jac": self.use_jac,
            "x_scale": self.x_scale,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "gtol": self.gtol,
            "loss": self.loss,
            "diff_step": self.diff_step,
            "max_nfev": self.max_nfev,
            "chunk_size": self.chunk_size,
            # Workflow / goal
            "workflow": self.workflow,
            "goal": self.goal,
            # Streaming / stratified
            "enable_streaming": self.enable_streaming,
            "streaming_chunk_size": self.streaming_chunk_size,
            "enable_stratified": self.enable_stratified,
            "target_chunk_size": self.target_chunk_size,
            # Recovery
            "enable_recovery": self.enable_recovery,
            "max_recovery_attempts": self.max_recovery_attempts,
            "recovery": {
                "max_retries": self.recovery_config.max_retries,
                "lr_decay": self.recovery_config.lr_decay,
                "lambda_growth": self.recovery_config.lambda_growth,
                "trust_decay": self.recovery_config.trust_decay,
                "perturb_scale": self.recovery_config.perturb_scale,
            },
            # Diagnostics
            "enable_diagnostics": self.enable_diagnostics,
            "enable_anti_degeneracy": self.enable_anti_degeneracy,
            # Loss / scaling
            "x_scale_map": dict(self.x_scale_map),
            "loss_weights": self.loss_weights,
            "loss_scale": self.loss_scale,
            "tr_solver": self.tr_solver,
            "step_bound": self.step_bound,
            # Fourier reparameterization
            "per_angle_mode": self.per_angle_mode,
            "fourier_order": self.fourier_order,
            "fourier_auto_threshold": self.fourier_auto_threshold,
            # Hierarchical optimization
            "enable_hierarchical": self.enable_hierarchical,
            "execute_layers": self.execute_layers,
            "hierarchical_max_outer_iterations": self.hierarchical_max_outer_iterations,
            "hierarchical_inner_tolerance": self.hierarchical_inner_tolerance,
            "hierarchical_outer_tolerance": self.hierarchical_outer_tolerance,
            # Adaptive regularization
            "regularization_mode": self.regularization_mode,
            "group_variance_lambda": self.group_variance_lambda,
            "regularization_target_cv": self.regularization_target_cv,
            # Gradient collapse detection
            "enable_gradient_monitoring": self.enable_gradient_monitoring,
            "gradient_ratio_threshold": self.gradient_ratio_threshold,
            "gradient_consecutive_triggers": self.gradient_consecutive_triggers,
            # CMA-ES global search
            "enable_cmaes": self.enable_cmaes,
            "cmaes_sigma0": self.cmaes_sigma0,
            "cmaes_max_iterations": self.cmaes_max_iterations,
            "cmaes_population_size": self.cmaes_population_size,
            "cmaes_tolx": self.cmaes_tolx,
            "cmaes_tolfun": self.cmaes_tolfun,
            "cmaes_diagonal_filtering": self.cmaes_diagonal_filtering,
            "cmaes_anti_degeneracy": self.cmaes_anti_degeneracy,
            "cmaes_warmstart_auto_skip": self.cmaes_warmstart_auto_skip,
            "cmaes_warmstart_skip_threshold": self.cmaes_warmstart_skip_threshold,
            "cmaes_restart_strategy": self.cmaes_restart_strategy,
            "cmaes_max_restarts": self.cmaes_max_restarts,
            # Hybrid streaming optimizer
            "hybrid_enable": self.hybrid_enable,
            "hybrid_warmup_fraction": self.hybrid_warmup_fraction,
            "hybrid_normalization": self.hybrid_normalization,
            "hybrid_method": self.hybrid_method,
            "hybrid_lbfgs_memory": self.hybrid_lbfgs_memory,
            "hybrid_convergence_window": self.hybrid_convergence_window,
            "hybrid_convergence_threshold": self.hybrid_convergence_threshold,
            "hybrid_max_phases": self.hybrid_max_phases,
            # Multi-start extensions
            "sampling_strategy": self.sampling_strategy,
            "screen_keep_fraction": self.screen_keep_fraction,
            "refine_top_k": self.refine_top_k,
            # Scaling threshold
            "constant_scaling_threshold": self.constant_scaling_threshold,
            # Backend / model
            "use_nlsq_library": self.use_nlsq_library,
            "n_params": self.n_params,
            "analysis_mode": self.analysis_mode,
            # NLSQ package integration
            "nlsq_stability": self.nlsq_stability,
            "nlsq_rescale_data": self.nlsq_rescale_data,
            "nlsq_x_scale": self.nlsq_x_scale,
            "nlsq_memory_fraction": self.nlsq_memory_fraction,
            "nlsq_memory_fallback_gb": self.nlsq_memory_fallback_gb,
            # Validation
            "validation": {
                "chi2_warn_low": self.validation.chi2_warn_low,
                "chi2_warn_high": self.validation.chi2_warn_high,
                "chi2_fail_high": self.validation.chi2_fail_high,
                "max_relative_uncertainty": self.validation.max_relative_uncertainty,
                "correlation_warn": self.validation.correlation_warn,
            },
        }
