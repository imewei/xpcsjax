"""Hierarchical Two-Stage Optimization for Anti-Degeneracy Defense.

This module implements alternating optimization between physical and per-angle
parameters, breaking the gradient cancellation cycle that causes structural
degeneracy in streaming optimization.

Part of Anti-Degeneracy Defense System v2.9.0.
See: docs/specs/anti-degeneracy-defense-v2.9.0.md

Algorithm::

    Initialize: params = [per_angle_params, physical_params]

    for outer_iter in range(max_outer_iterations):

        # Stage 1: Fit PHYSICAL params only
        freeze(per_angle_params)
        result1 = L-BFGS(
            loss_fn(physical_params | frozen_per_angle),
            physical_params
        )
        physical_params = result1.x

        # Stage 2: Fit PER-ANGLE params only
        freeze(physical_params)
        result2 = L-BFGS(
            loss_fn(per_angle_params | frozen_physical),
            per_angle_params
        )
        per_angle_params = result2.x

        # Check convergence
        if converged(physical_params, previous_physical_params):
            break

    return [per_angle_params, physical_params]

Why It Works
------------
1. In Stage 1, there are NO per-angle DoF to compete with physical params
2. gamma_dot_t0 gradient CANNOT cancel (no per-angle params to absorb signal)
3. Physical params converge to true values
4. Stage 2 only cleans up residuals with physical interpretation fixed
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

# jaxopt emits a DeprecationWarning at import time (it is no longer maintained).
# Suppress it at the callsite so -W error::DeprecationWarning in CI doesn't
# block collection. Tracked for migration in v0.2.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from jaxopt import LBFGSB

from xpcsjax.optimization.nlsq.config import safe_float, safe_int
from xpcsjax.optimization.nlsq.fourier_reparam import FourierReparameterizer
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


class _OptimizeResult(NamedTuple):
    """Compatibility result object for jaxopt.LBFGSB.

    Mimics scipy.optimize.OptimizeResult interface for seamless integration.
    """

    x: np.ndarray
    fun: float
    nit: int
    success: bool
    message: str = ""


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical optimization.

    Attributes
    ----------
    enable : bool
        Whether to enable hierarchical optimization. Default True.
    max_outer_iterations : int
        Maximum outer iterations. Default 5.
    outer_tolerance : float
        Convergence tolerance for physical parameters. Default 1e-6.
    physical_max_iterations : int
        Max iterations for Stage 1 (physical params). Default 100.
    physical_ftol : float
        Function tolerance for Stage 1. Default 1e-8.
    per_angle_max_iterations : int
        Max iterations for Stage 2 (per-angle params). Default 50.
    per_angle_ftol : float
        Function tolerance for Stage 2. Default 1e-6.
    log_stage_transitions : bool
        Whether to log stage transitions. Default True.
    save_intermediate_results : bool
        Whether to save intermediate results. Default False.
    """

    enable: bool = True
    max_outer_iterations: int = 5
    outer_tolerance: float = 1e-6

    # Stage 1: Physical parameter optimization
    physical_max_iterations: int = 100
    physical_ftol: float = 1e-8

    # Stage 2: Per-angle parameter optimization
    per_angle_max_iterations: int = 50
    per_angle_ftol: float = 1e-6

    # Callback options
    log_stage_transitions: bool = True
    save_intermediate_results: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict) -> HierarchicalConfig:
        """Create config from dictionary with safe type conversion."""
        return cls(
            enable=bool(config_dict.get("enable", True)),
            max_outer_iterations=safe_int(config_dict.get("max_outer_iterations"), 5),
            outer_tolerance=safe_float(config_dict.get("outer_tolerance"), 1e-6),
            physical_max_iterations=safe_int(
                config_dict.get("physical_max_iterations"), 100
            ),
            physical_ftol=safe_float(config_dict.get("physical_ftol"), 1e-8),
            per_angle_max_iterations=safe_int(
                config_dict.get("per_angle_max_iterations"), 50
            ),
            per_angle_ftol=safe_float(config_dict.get("per_angle_ftol"), 1e-6),
            log_stage_transitions=bool(config_dict.get("log_stage_transitions", True)),
            save_intermediate_results=bool(
                config_dict.get("save_intermediate_results", False)
            ),
        )


@dataclass
class HierarchicalResult:
    """Result from hierarchical optimization.

    Attributes
    ----------
    x : np.ndarray
        Optimized parameters.
    fun : float
        Final loss value.
    success : bool
        Whether optimization succeeded.
    n_outer_iterations : int
        Number of outer iterations performed.
    history : list
        History of each outer iteration.
    total_time : float
        Total optimization time in seconds.
    message : str
        Status message.
    """

    x: np.ndarray
    fun: float
    success: bool
    n_outer_iterations: int
    history: list[dict] = field(default_factory=list)
    total_time: float = 0.0
    message: str = ""


class HierarchicalOptimizer:
    """Two-stage hierarchical optimizer for decoupled fitting.

    This optimizer breaks the gradient cancellation problem by alternating
    between physical and per-angle parameter optimization:

    Stage 1: Physical parameters only
        - Per-angle parameters are frozen
        - gamma_dot_t0 gradient cannot be cancelled by per-angle absorption
        - Physical params converge to true values

    Stage 2: Per-angle parameters only
        - Physical parameters are frozen
        - Per-angle params absorb only experimental noise
        - Cannot change the physical interpretation

    Parameters
    ----------
    config : HierarchicalConfig
        Hierarchical optimization configuration.
    n_phi : int
        Number of unique phi angles.
    n_physical : int
        Number of physical parameters.
    fourier_reparameterizer : FourierReparameterizer, optional
        Fourier reparameterizer if using Fourier mode.

    Examples
    --------
    >>> config = HierarchicalConfig(max_outer_iterations=5)
    >>> optimizer = HierarchicalOptimizer(config, n_phi=23, n_physical=7)
    >>> result = optimizer.fit(loss_fn, grad_fn, p0, bounds)
    """

    def __init__(
        self,
        config: HierarchicalConfig,
        n_phi: int,
        n_physical: int,
        fourier_reparameterizer: FourierReparameterizer | None = None,
    ):
        """Initialize hierarchical optimizer.

        Parameters
        ----------
        config : HierarchicalConfig
            Configuration.
        n_phi : int
            Number of unique phi angles.
        n_physical : int
            Number of physical parameters.
        fourier_reparameterizer : FourierReparameterizer, optional
            Fourier reparameterizer for Fourier mode.
        """
        self.config = config
        self.n_phi = n_phi
        self.n_physical = n_physical
        self.fourier = fourier_reparameterizer

        # Determine parameter indices based on Fourier mode
        # When Fourier mode is active, per-angle params are Fourier coefficients
        # not 2 * n_phi independent values
        if self.fourier is not None:
            self.n_per_angle = self.fourier.n_coeffs
        else:
            self.n_per_angle = 2 * n_phi

        # Use numpy arrays for indices to support both NumPy and JAX array indexing
        # JAX arrays don't support Python list indexing (non-tuple sequence error)
        self.per_angle_indices: np.ndarray = np.arange(self.n_per_angle, dtype=np.intp)
        self.physical_indices: np.ndarray = np.arange(
            self.n_per_angle, self.n_per_angle + n_physical, dtype=np.intp
        )

        # Pre-allocate buffer for full parameter vector (Spec 006 - FR-002)
        # This avoids np.concatenate allocations on every loss function call,
        # providing ~10-15% speedup for L-BFGS phase
        self._full_params_buffer: np.ndarray = np.empty(
            self.n_per_angle + n_physical, dtype=np.float64
        )

        logger.debug(
            f"HierarchicalOptimizer initialized: "
            f"n_per_angle={self.n_per_angle}, n_physical={n_physical}, "
            f"fourier={'enabled' if self.fourier else 'disabled'}"
        )

    def _create_physical_loss(
        self,
        frozen_per_angle: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
    ) -> Callable[[np.ndarray], float]:
        """Create physical loss function with pre-allocated buffer (FR-002).

        Uses a local buffer copy to avoid interference between closures.

        Parameters
        ----------
        frozen_per_angle : np.ndarray
            Frozen per-angle parameters.
        loss_fn : callable
            Full loss function.

        Returns
        -------
        callable
            Physical-only loss function.
        """
        # Local copy to avoid shared-buffer interference between closures
        buffer = self._full_params_buffer.copy()
        buffer[self.per_angle_indices] = frozen_per_angle

        def physical_loss(physical_params: np.ndarray) -> float:
            buffer[self.physical_indices] = physical_params
            return loss_fn(buffer)

        return physical_loss

    def _create_physical_grad(
        self,
        frozen_per_angle: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create physical gradient function with pre-allocated buffer (FR-002).

        Uses a local buffer copy to avoid interference between closures.

        Parameters
        ----------
        frozen_per_angle : np.ndarray
            Frozen per-angle parameters.
        grad_fn : callable
            Full gradient function.

        Returns
        -------
        callable
            Physical-only gradient function.
        """
        # Local copy to avoid shared-buffer interference between closures
        buffer = self._full_params_buffer.copy()
        buffer[self.per_angle_indices] = frozen_per_angle

        def physical_grad(physical_params: np.ndarray) -> np.ndarray:
            buffer[self.physical_indices] = physical_params
            full_grad = grad_fn(buffer)
            return full_grad[self.physical_indices]

        return physical_grad

    def _create_per_angle_loss(
        self,
        frozen_physical: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
    ) -> Callable[[np.ndarray], float]:
        """Create per-angle loss function with pre-allocated buffer (FR-002).

        Uses a local buffer copy to avoid interference between closures.

        Parameters
        ----------
        frozen_physical : np.ndarray
            Frozen physical parameters.
        loss_fn : callable
            Full loss function.

        Returns
        -------
        callable
            Per-angle-only loss function.
        """
        # Local copy to avoid shared-buffer interference between closures
        buffer = self._full_params_buffer.copy()
        buffer[self.physical_indices] = frozen_physical

        def per_angle_loss(per_angle_params: np.ndarray) -> float:
            buffer[self.per_angle_indices] = per_angle_params
            return loss_fn(buffer)

        return per_angle_loss

    def _create_per_angle_grad(
        self,
        frozen_physical: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create per-angle gradient function with pre-allocated buffer (FR-002).

        Uses a local buffer copy to avoid interference between closures.

        Parameters
        ----------
        frozen_physical : np.ndarray
            Frozen physical parameters.
        grad_fn : callable
            Full gradient function.

        Returns
        -------
        callable
            Per-angle-only gradient function.
        """
        # Local copy to avoid shared-buffer interference between closures
        buffer = self._full_params_buffer.copy()
        buffer[self.physical_indices] = frozen_physical

        def per_angle_grad(per_angle_params: np.ndarray) -> np.ndarray:
            buffer[self.per_angle_indices] = per_angle_params
            full_grad = grad_fn(buffer)
            return full_grad[self.per_angle_indices]

        return per_angle_grad

    def fit(
        self,
        loss_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray] | None,
        p0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        outer_iteration_callback: Callable[[np.ndarray, int], None] | None = None,
    ) -> HierarchicalResult:
        """Run hierarchical optimization.

        Parameters
        ----------
        loss_fn : callable
            Loss function f(params) -> scalar.
        grad_fn : callable or None
            Gradient function g(params) -> gradient array. If None, uses
            finite differences.
        p0 : np.ndarray
            Initial parameters.
        bounds : tuple
            (lower_bounds, upper_bounds).
        outer_iteration_callback : callable or None
            Optional callback called at the start of each outer iteration.
            Signature: callback(current_params, outer_iter). Used for updating
            shear-sensitivity weights based on current phi0 estimate.

        Returns
        -------
        HierarchicalResult
            Optimization result with diagnostics.
        """
        start_time = time.perf_counter()
        current_params = p0.copy()
        history = []

        initial_loss = loss_fn(current_params)
        logger.info("=" * 60)
        logger.info("HIERARCHICAL OPTIMIZATION")
        logger.info("=" * 60)
        logger.info(f"Initial loss: {initial_loss:.6e}")
        logger.info(
            f"Parameter split: {self.n_per_angle} per-angle + "
            f"{self.n_physical} physical"
        )

        converged = False

        for outer_iter in range(self.config.max_outer_iterations):
            # Call outer iteration callback if provided (e.g., for shear weight updates)
            if outer_iteration_callback is not None:
                outer_iteration_callback(current_params, outer_iter)

            previous_physical = current_params[self.physical_indices].copy()
            iter_start = time.perf_counter()

            if self.config.log_stage_transitions:
                logger.info("-" * 40)
                logger.info(f"Outer iteration {outer_iter + 1}")

            # Stage 1: Fit physical parameters
            stage1_result = self._fit_physical_stage(
                loss_fn, grad_fn, current_params, bounds, outer_iter
            )
            current_params = stage1_result.x.copy()

            if self.config.log_stage_transitions:
                logger.info(
                    f"  Stage 1 (physical): loss={stage1_result.fun:.6e}, "
                    f"iters={stage1_result.nit}"
                )

            # Stage 2: Fit per-angle parameters
            stage2_result = self._fit_per_angle_stage(
                loss_fn, grad_fn, current_params, bounds, outer_iter
            )
            current_params = stage2_result.x.copy()

            if self.config.log_stage_transitions:
                logger.info(
                    f"  Stage 2 (per-angle): loss={stage2_result.fun:.6e}, "
                    f"iters={stage2_result.nit}"
                )

            iter_time = time.perf_counter() - iter_start

            # Record history
            history.append(
                {
                    "outer_iter": outer_iter,
                    "stage1_loss": float(stage1_result.fun),
                    "stage1_iterations": stage1_result.nit,
                    "stage2_loss": float(stage2_result.fun),
                    "stage2_iterations": stage2_result.nit,
                    "physical_params": current_params[self.physical_indices].copy(),
                    "time": iter_time,
                }
            )

            # Check convergence
            physical_change = np.linalg.norm(
                current_params[self.physical_indices] - previous_physical
            )
            relative_change = physical_change / (
                np.linalg.norm(previous_physical) + 1e-10
            )

            if self.config.log_stage_transitions:
                logger.info(
                    f"  Physical param change: {physical_change:.6e} "
                    f"(relative: {relative_change:.6e})"
                )

            if physical_change < self.config.outer_tolerance:
                converged = True
                logger.info(
                    f"Converged at outer iteration {outer_iter + 1} "
                    f"(change {physical_change:.6e} "
                    f"< tol {self.config.outer_tolerance})"
                )
                break

        total_time = time.perf_counter() - start_time
        final_loss = loss_fn(current_params)

        logger.info("=" * 60)
        logger.info("HIERARCHICAL OPTIMIZATION COMPLETE")
        logger.info(f"  Converged: {converged}")
        logger.info(f"  Outer iterations: {len(history)}")
        logger.info(f"  Final loss: {final_loss:.6e}")
        logger.info(f"  Improvement: {100 * (1 - final_loss / initial_loss):.2f}%")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info("=" * 60)

        return HierarchicalResult(
            x=current_params,
            fun=final_loss,
            success=converged,
            n_outer_iterations=len(history),
            history=history,
            total_time=total_time,
            message="Converged" if converged else "Max iterations reached",
        )

    def _fit_physical_stage(
        self,
        loss_fn: Callable,
        grad_fn: Callable | None,
        current_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        outer_iter: int,
    ) -> _OptimizeResult:
        """Stage 1: Optimize physical parameters with per-angle frozen.

        Performance Optimization (Spec 006 - FR-002):
        Uses pre-allocated buffer and in-place updates to avoid np.concatenate
        allocations on every loss/gradient call.

        Uses jaxopt.LBFGSB for JAX-native bounded L-BFGS optimization.

        Parameters
        ----------
        loss_fn : callable
            Full loss function.
        grad_fn : callable or None
            Full gradient function.
        current_params : np.ndarray
            Current full parameter vector.
        bounds : tuple
            Full parameter bounds.
        outer_iter : int
            Current outer iteration.

        Returns
        -------
        _OptimizeResult
            Optimization result with x containing full parameter vector.
        """
        frozen_per_angle = current_params[self.per_angle_indices].copy()

        # Use helper methods with pre-allocated buffer (FR-002 optimization)
        physical_loss = self._create_physical_loss(frozen_per_angle, loss_fn)
        physical_grad = None
        if grad_fn is not None:
            physical_grad = self._create_physical_grad(frozen_per_angle, grad_fn)

        # Extract physical bounds for jaxopt (lower, upper) tuple format
        physical_lower = jnp.asarray(bounds[0][self.physical_indices])
        physical_upper = jnp.asarray(bounds[1][self.physical_indices])
        physical_bounds = (physical_lower, physical_upper)

        # Create loss function with optional gradient for jaxopt
        if physical_grad is not None:
            # jaxopt expects (value, grad) when value_and_grad=True
            def value_and_grad_fn(x: jnp.ndarray) -> tuple[float, jnp.ndarray]:
                return physical_loss(np.asarray(x)), jnp.asarray(
                    physical_grad(np.asarray(x))
                )

            solver = LBFGSB(
                fun=value_and_grad_fn,
                value_and_grad=True,
                maxiter=self.config.physical_max_iterations,
                tol=self.config.physical_ftol,
                jit=False,  # Disable JIT for NumPy compatibility
            )
        else:
            # Let jaxopt compute gradients via autodiff
            def jax_loss_fn(x: jnp.ndarray) -> float:
                return physical_loss(np.asarray(x))

            solver = LBFGSB(
                fun=jax_loss_fn,
                maxiter=self.config.physical_max_iterations,
                tol=self.config.physical_ftol,
                jit=False,
            )

        # Run L-BFGS-B on physical params only
        x0 = jnp.asarray(current_params[self.physical_indices])
        result = solver.run(x0, bounds=physical_bounds)

        # Convert jaxopt result to compatible format
        optimized_params = np.asarray(result.params)
        final_loss = float(result.state.value)
        n_iterations = int(result.state.iter_num)
        converged = result.state.error < self.config.physical_ftol

        # Update full params
        full_result_x = current_params.copy()
        full_result_x[self.physical_indices] = optimized_params

        return _OptimizeResult(
            x=full_result_x,
            fun=final_loss,
            nit=n_iterations,
            success=converged,
            message="Converged" if converged else "Max iterations reached",
        )

    def _fit_per_angle_stage(
        self,
        loss_fn: Callable,
        grad_fn: Callable | None,
        current_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        outer_iter: int,
    ) -> _OptimizeResult:
        """Stage 2: Optimize per-angle parameters with physical frozen.

        Performance Optimization (Spec 006 - FR-002):
        Uses pre-allocated buffer and in-place updates to avoid np.concatenate
        allocations on every loss/gradient call.

        Uses jaxopt.LBFGSB for JAX-native bounded L-BFGS optimization.

        Parameters
        ----------
        loss_fn : callable
            Full loss function.
        grad_fn : callable or None
            Full gradient function.
        current_params : np.ndarray
            Current full parameter vector.
        bounds : tuple
            Full parameter bounds.
        outer_iter : int
            Current outer iteration.

        Returns
        -------
        _OptimizeResult
            Optimization result with x containing full parameter vector.
        """
        frozen_physical = current_params[self.physical_indices].copy()

        # Use helper methods with pre-allocated buffer (FR-002 optimization)
        per_angle_loss = self._create_per_angle_loss(frozen_physical, loss_fn)
        per_angle_grad = None
        if grad_fn is not None:
            per_angle_grad = self._create_per_angle_grad(frozen_physical, grad_fn)

        # Extract per-angle bounds for jaxopt (lower, upper) tuple format
        per_angle_lower = jnp.asarray(bounds[0][self.per_angle_indices])
        per_angle_upper = jnp.asarray(bounds[1][self.per_angle_indices])
        per_angle_bounds = (per_angle_lower, per_angle_upper)

        # Create loss function with optional gradient for jaxopt
        if per_angle_grad is not None:
            # jaxopt expects (value, grad) when value_and_grad=True
            def value_and_grad_fn(x: jnp.ndarray) -> tuple[float, jnp.ndarray]:
                return per_angle_loss(np.asarray(x)), jnp.asarray(
                    per_angle_grad(np.asarray(x))
                )

            solver = LBFGSB(
                fun=value_and_grad_fn,
                value_and_grad=True,
                maxiter=self.config.per_angle_max_iterations,
                tol=self.config.per_angle_ftol,
                jit=False,  # Disable JIT for NumPy compatibility
            )
        else:
            # Let jaxopt compute gradients via autodiff
            def jax_loss_fn(x: jnp.ndarray) -> float:
                return per_angle_loss(np.asarray(x))

            solver = LBFGSB(
                fun=jax_loss_fn,
                maxiter=self.config.per_angle_max_iterations,
                tol=self.config.per_angle_ftol,
                jit=False,
            )

        # Run L-BFGS-B on per-angle params only
        x0 = jnp.asarray(current_params[self.per_angle_indices])
        result = solver.run(x0, bounds=per_angle_bounds)

        # Convert jaxopt result to compatible format
        optimized_params = np.asarray(result.params)
        final_loss = float(result.state.value)
        n_iterations = int(result.state.iter_num)
        converged = result.state.error < self.config.per_angle_ftol

        # Update full params
        full_result_x = current_params.copy()
        full_result_x[self.per_angle_indices] = optimized_params

        return _OptimizeResult(
            x=full_result_x,
            fun=final_loss,
            nit=n_iterations,
            success=converged,
            message="Converged" if converged else "Max iterations reached",
        )

    def get_diagnostics(self) -> dict:
        """Get optimizer diagnostics.

        Returns
        -------
        dict
            Diagnostic information.
        """
        return {
            "enabled": self.config.enable,
            "n_phi": self.n_phi,
            "n_physical": self.n_physical,
            "n_per_angle": self.n_per_angle,
            "fourier_enabled": self.fourier is not None,
            "max_outer_iterations": self.config.max_outer_iterations,
            "outer_tolerance": self.config.outer_tolerance,
        }
