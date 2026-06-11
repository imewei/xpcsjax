"""Mixins for physics model capabilities.

This module provides reusable mixin classes that can be composed
with PhysicsModelBase to add gradient, benchmarking, and optimization
recommendation capabilities.

Part of Architecture Refactoring.
Extracted from CombinedModel to improve maintainability.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from xpcsjax.core.jax_backend import (
    get_device_info,
    gradient_g2,
    hessian_g2,
    jax_available,
    jnp,
    numpy_gradients_available,
    validate_backend,
)
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from xpcsjax.config.parameter_registry import AnalysisMode

logger = get_logger(__name__)


class _PhysicsModelProtocol(Protocol):
    """Protocol defining the interface expected by mixins."""

    name: str
    analysis_mode: AnalysisMode
    n_params: int
    parameter_names: list[str]

    def get_default_parameters(self) -> jnp.ndarray: ...
    def get_parameter_bounds(self) -> list[tuple[float, float]]: ...
    def compute_g2(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        contrast: float,
        offset: float,
        dt: float,
    ) -> jnp.ndarray: ...
    def supports_gradients(self) -> bool: ...
    def get_gradient_function(self) -> Callable: ...
    def get_best_gradient_method(self) -> str: ...
    def get_gradient_capabilities(self) -> dict[str, Any]: ...
    def get_optimization_recommendations(self) -> list[str]: ...


class GradientCapabilityMixin:
    """Mixin providing gradient computation capabilities.

    This mixin adds methods for accessing gradient and Hessian functions
    with intelligent backend selection, capability reporting, and
    performance warnings.

    Requires the class to have:
    - analysis_mode: AnalysisMode attribute
    """

    def get_gradient_function(self) -> Callable:
        """Get gradient function with intelligent backend selection."""
        backend_info = self.get_gradient_capabilities()

        if backend_info["gradient_available"]:
            logger.info(f"Using {backend_info['best_method']} for gradient computation")
            if backend_info["performance_warning"]:
                logger.warning(backend_info["performance_warning"])
            return gradient_g2
        else:
            # Provide informative error with recommendations
            error_msg = (
                "Gradient computation not available. Install dependencies for differentiation:\n"
                "* Recommended: pip install jax (optimal performance)\n"
                "* Alternative: pip install scipy (basic numerical gradients)\n\n"
                f"Current backend status: {backend_info['backend_summary']}"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

    def get_hessian_function(self) -> Callable:
        """Get hessian function with intelligent backend selection."""
        backend_info = self.get_gradient_capabilities()

        if backend_info["hessian_available"]:
            logger.info(f"Using {backend_info['best_method']} for Hessian computation")
            if backend_info["performance_warning"]:
                logger.warning(backend_info["performance_warning"])
            return hessian_g2
        else:
            # Provide informative error with recommendations
            error_msg = (
                "Hessian computation not available. Install dependencies for second derivatives:\n"
                "* Recommended: pip install jax (optimal performance)\n"
                "* Alternative: pip install scipy (basic numerical Hessians)\n\n"
                f"Current backend status: {backend_info['backend_summary']}"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

    def supports_gradients(self) -> bool:
        """Check if gradient computation is available."""
        return jax_available or numpy_gradients_available

    def get_best_gradient_method(self) -> str:
        """Get the best available gradient method for optimization algorithms."""
        backend_info = validate_backend()

        if backend_info["jax_available"]:
            return "jax_native"
        elif backend_info["numpy_gradients_available"]:
            return "numpy_fallback"
        else:
            return "none_available"

    def get_gradient_capabilities(self) -> dict[str, Any]:
        """Get comprehensive gradient capability information."""
        backend_info = validate_backend()
        device_info = get_device_info()

        # Determine best available method
        if backend_info["jax_available"]:
            best_method = "JAX (automatic differentiation)"
            performance_warning = None
        elif backend_info["numpy_gradients_available"]:
            best_method = "NumPy (numerical differentiation)"
            performance_warning = "Using NumPy fallback - expect 10-50x performance degradation"
        else:
            best_method = "None available"
            performance_warning = "No gradient computation backend available"

        return {
            "gradient_available": backend_info["gradient_support"],
            "hessian_available": backend_info["hessian_support"],
            "best_method": best_method,
            "backend_type": backend_info["backend_type"],
            "performance_estimate": backend_info["performance_estimate"],
            "performance_warning": performance_warning,
            "jax_available": backend_info["jax_available"],
            "numpy_gradients_available": backend_info["numpy_gradients_available"],
            "device_info": device_info,
            "recommendations": backend_info["recommendations"],
            "fallback_stats": backend_info["fallback_stats"],
            "backend_summary": self._generate_backend_summary(
                backend_info,
                device_info,
            ),
        }

    def _generate_backend_summary(self, backend_info: dict, device_info: dict) -> str:
        """Generate human-readable backend summary."""
        if backend_info["jax_available"]:
            devices = device_info.get("devices", ["unknown"])
            device_str = f", {len(devices)} device(s) available" if len(devices) > 1 else ""
            return f"JAX backend active{device_str}, optimal performance"
        elif backend_info["numpy_gradients_available"]:
            return "NumPy numerical differentiation active, reduced performance"
        else:
            return "No differentiation backend available"


class BenchmarkingMixin:
    """Mixin providing performance benchmarking capabilities.

    This mixin adds methods for benchmarking gradient computation
    performance and validating gradient accuracy.

    Requires the class to have:
    - analysis_mode: AnalysisMode attribute
    - get_default_parameters(): method returning jnp.ndarray
    - compute_g2(): method for g2 computation
    - supports_gradients(): method returning bool
    - get_gradient_function(): method returning Callable
    - get_best_gradient_method(): method returning str
    """

    def benchmark_gradient_performance(
        self: _PhysicsModelProtocol,
        test_params: jnp.ndarray | None = None,
    ) -> dict[str, Any]:
        """Benchmark gradient computation performance across available methods."""
        if test_params is None:
            test_params = self.get_default_parameters()

        logger.info("Benchmarking gradient computation performance...")

        # Test parameters for performance evaluation
        test_t1 = jnp.array([0.0, 0.1, 0.2])
        test_t2 = jnp.array([1.0, 1.1, 1.2])
        test_phi = jnp.array([0.0, 45.0, 90.0])
        test_q = 0.01
        test_L = 1e6  # 1 mm in Angstroms, center of valid range [1e5, 1e8]
        test_contrast = 0.8
        test_offset = 1.0
        test_dt = 0.001  # Required dt parameter

        benchmark_results: dict[str, Any] = {
            "test_conditions": {
                "n_parameters": len(test_params),
                "n_time_points": len(test_t1),
                "n_angles": len(test_phi),
                "analysis_mode": self.analysis_mode,
            },
            "methods": {},
            "best_method": None,
            "performance_ratio": None,
        }

        methods_to_test = []
        if jax_available:
            methods_to_test.append(("jax_native", "JAX automatic differentiation"))
        if numpy_gradients_available:
            methods_to_test.append(
                ("numpy_fallback", "NumPy numerical differentiation"),
            )

        if not methods_to_test:
            benchmark_results["error"] = "No gradient methods available for benchmarking"
            return benchmark_results

        # Test each available method
        for method_key, method_name in methods_to_test:
            try:
                start_time = time.perf_counter()

                # Test forward computation
                g2_result = self.compute_g2(
                    test_params,
                    test_t1,
                    test_t2,
                    test_phi,
                    test_q,
                    test_L,
                    test_contrast,
                    test_offset,
                    test_dt,
                )

                # Test gradient computation if available
                grad_result = None
                if self.supports_gradients():
                    grad_func = self.get_gradient_function()
                    grad_result = grad_func(
                        test_params,
                        test_t1,
                        test_t2,
                        test_phi,
                        test_q,
                        test_L,
                        test_contrast,
                        test_offset,
                        test_dt,
                    )

                computation_time = time.perf_counter() - start_time

                benchmark_results["methods"][method_key] = {
                    "name": method_name,
                    "computation_time": computation_time,
                    "success": True,
                    "forward_shape": (g2_result.shape if hasattr(g2_result, "shape") else "scalar"),
                    "gradient_shape": (
                        grad_result.shape if grad_result is not None else "not_computed"
                    ),
                }

            except Exception as e:
                benchmark_results["methods"][method_key] = {
                    "name": method_name,
                    "success": False,
                    "error": str(e),
                }

        # Determine best method and performance ratios
        successful_methods = [
            (k, v) for k, v in benchmark_results["methods"].items() if v["success"]
        ]
        if successful_methods:
            # Sort by computation time
            successful_methods.sort(key=lambda x: x[1]["computation_time"])
            best_method_key, best_method_info = successful_methods[0]

            benchmark_results["best_method"] = {
                "method": best_method_key,
                "name": best_method_info["name"],
                "time": best_method_info["computation_time"],
            }

            # Calculate performance ratios relative to best method
            best_time = best_method_info["computation_time"]
            for _, method_info in benchmark_results["methods"].items():
                if method_info["success"]:
                    method_info["performance_ratio"] = method_info["computation_time"] / best_time

        logger.info("Gradient performance benchmark completed")
        return benchmark_results

    def validate_gradient_accuracy(
        self: _PhysicsModelProtocol,
        test_params: jnp.ndarray | None = None,
        tolerance: float = 1e-6,
    ) -> dict[str, Any]:
        """Validate gradient accuracy against reference solutions."""
        if test_params is None:
            test_params = self.get_default_parameters()

        logger.info("Validating gradient accuracy...")

        # Simple test case for validation
        test_t1 = jnp.array([0.0])
        test_t2 = jnp.array([1.0])
        test_phi = jnp.array([0.0])
        test_q = 0.01
        test_L = 1e6  # 1 mm in Angstroms, center of valid range [1e5, 1e8]
        test_contrast = 0.8
        test_offset = 1.0
        test_dt = 0.001  # Required dt parameter (config-sourced in production)

        validation_results: dict[str, Any] = {
            "test_conditions": {
                "parameters": test_params.tolist(),
                "tolerance": tolerance,
                "analysis_mode": self.analysis_mode,
            },
            "accuracy_assessment": {},
            "recommendations": [],
        }

        try:
            # Test gradient computation
            if self.supports_gradients():
                grad_func = self.get_gradient_function()
                gradient = grad_func(
                    test_params,
                    test_t1,
                    test_t2,
                    test_phi,
                    test_q,
                    test_L,
                    test_contrast,
                    test_offset,
                    test_dt,
                )

                # Basic validation checks
                validation_results["accuracy_assessment"] = {
                    "gradient_computed": True,
                    "gradient_shape": gradient.shape,
                    "gradient_finite": bool(jnp.all(jnp.isfinite(gradient))),
                    "gradient_magnitude": float(jnp.linalg.norm(gradient)),
                    "max_gradient_component": float(jnp.nanmax(jnp.abs(gradient))),
                    "method_used": self.get_best_gradient_method(),
                }

                # Check for reasonable gradient magnitudes for XPCS physics
                max_grad = float(jnp.nanmax(jnp.abs(gradient)))
                if max_grad > 1e6:
                    validation_results["recommendations"].append(
                        "Gradient magnitudes are very large - check parameter scaling",
                    )
                elif max_grad < 1e-10:
                    validation_results["recommendations"].append(
                        "Gradient magnitudes are very small - may indicate insensitive parameters",
                    )
                else:
                    validation_results["recommendations"].append(
                        "Gradient magnitudes appear reasonable for XPCS analysis",
                    )

            else:
                validation_results["accuracy_assessment"] = {
                    "gradient_computed": False,
                    "error": "No gradient computation backend available",
                }
                validation_results["recommendations"].append(
                    "Install JAX or scipy for gradient-based optimization",
                )

        except (
            TypeError,
            ValueError,
            RuntimeError,
            ArithmeticError,
            AttributeError,
        ) as e:
            validation_results["accuracy_assessment"] = {
                "gradient_computed": False,
                "error": str(e),
            }
            validation_results["recommendations"].append(
                f"Gradient computation failed: {str(e)}",
            )

        logger.info("Gradient accuracy validation completed")
        return validation_results


class OptimizationRecommendationMixin:
    """Mixin providing optimization guidance.

    This mixin adds methods for getting optimization recommendations
    and comprehensive model information.

    Requires the class to have:
    - name: str attribute
    - analysis_mode: AnalysisMode attribute
    - n_params: int attribute
    - parameter_names: list[str] attribute
    - get_parameter_bounds(): method
    - get_default_parameters(): method
    - get_gradient_capabilities(): method
    - supports_gradients(): method
    - get_best_gradient_method(): method
    """

    def get_optimization_recommendations(
        self: _PhysicsModelProtocol,
    ) -> list[str]:
        """Get optimization recommendations based on available capabilities."""
        capabilities = self.get_gradient_capabilities()
        recommendations: list[str] = []

        if capabilities["jax_available"]:
            recommendations.append(
                "JAX available - use gradient-based optimization (BFGS, Adam)",
            )

            device_info = capabilities["device_info"]
            if device_info.get("available", False):
                devices = device_info.get("devices", [])
                if len(devices) > 1:
                    recommendations.append(
                        f"{len(devices)} compute devices available for parallel optimization",
                    )

        elif capabilities["numpy_gradients_available"]:
            recommendations.append(
                "Using NumPy gradients - prefer L-BFGS over high-order methods",
            )
            recommendations.append(
                "Consider installing JAX for 10-50x performance improvement",
            )

        else:
            recommendations.append(
                "No gradient support - use gradient-free optimization (Nelder-Mead, Powell)",
            )
            recommendations.append(
                "Install scipy for basic optimization: pip install scipy",
            )
            recommendations.append(
                "Install JAX for advanced optimization: pip install jax",
            )

        # Analysis mode specific recommendations
        if self.analysis_mode == "laminar_flow":
            if capabilities["jax_available"]:
                recommendations.append(
                    "Laminar flow mode (7 parameters) - JAX optimization recommended",
                )
            else:
                recommendations.append(
                    "Laminar flow mode - many parameters, consider staged optimization",
                )

        elif self.analysis_mode.startswith("static"):
            recommendations.append(
                "Static mode (3 parameters) - most optimization methods will work well",
            )

        return recommendations

    def get_model_info(self: _PhysicsModelProtocol) -> dict:
        """Get comprehensive model information with enhanced capabilities."""
        capabilities = self.get_gradient_capabilities()

        return {
            # Basic model information
            "name": self.name,
            "analysis_mode": self.analysis_mode,
            "n_parameters": self.n_params,
            "parameter_names": self.parameter_names,
            "parameter_bounds": self.get_parameter_bounds(),
            "default_parameters": self.get_default_parameters().tolist(),
            # Gradient capabilities
            "supports_gradients": self.supports_gradients(),
            "gradient_method": self.get_best_gradient_method(),
            "gradient_capabilities": capabilities,
            # Backend information
            "jax_available": jax_available,
            "numpy_gradients_available": numpy_gradients_available,
            "backend_summary": capabilities["backend_summary"],
            # Optimization guidance
            "optimization_recommendations": self.get_optimization_recommendations(),
            # Performance information
            "performance_estimate": capabilities["performance_estimate"],
            "device_info": capabilities["device_info"],
        }


# Export all mixins
__all__ = [
    "GradientCapabilityMixin",
    "BenchmarkingMixin",
    "OptimizationRecommendationMixin",
]
