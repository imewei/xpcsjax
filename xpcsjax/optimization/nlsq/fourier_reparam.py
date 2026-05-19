"""Fourier Reparameterization for Anti-Degeneracy Defense.

This module replaces n_phi independent per-angle contrast/offset values
with truncated Fourier series, dramatically reducing structural degeneracy.

Part of Anti-Degeneracy Defense System v2.9.0.
See: docs/specs/anti-degeneracy-defense-v2.9.0.md

Mathematical Formulation
------------------------
contrast(φ) = c₀ + Σₖ[cₖ×cos(kφ) + sₖ×sin(kφ)]    for k=1..order
offset(φ)   = o₀ + Σₖ[oₖ×cos(kφ) + tₖ×sin(kφ)]    for k=1..order

For order=2:
- Contrast: 5 coefficients [c₀, c₁, s₁, c₂, s₂]
- Offset: 5 coefficients [o₀, o₁, t₁, o₂, t₂]
- Total: 10 Fourier coefficients vs 2×n_phi independent params

Parameter Count Comparison::

    n_phi | Independent | Fourier (order=2) | Reduction
    ------|-------------|-------------------|----------
      2   |     4       |        4          |    0%
      3   |     6       |        6          |    0%
     10   |    20       |       10          |   50%
     23   |    46       |       10          |   78%
    100   |   200       |       10          |   95%

Note: For n_phi <= 2*(order+1), independent mode is used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FourierReparamConfig:
    """Configuration for Fourier reparameterization.

    Attributes
    ----------
    mode : str
        Per-angle parameter mode:
        - "independent": Use n_phi independent contrast/offset values
        - "fourier": Use truncated Fourier series
        - "auto": Use Fourier when n_phi > auto_threshold
    fourier_order : int
        Number of Fourier harmonics. Default 2.
        order=2 gives 5 coefficients per parameter (c0, c1, s1, c2, s2).
    auto_threshold : int
        Use Fourier when n_phi > this threshold in auto mode. Default 6.
    c0_bounds : tuple
        Bounds for mean contrast coefficient. Default (0.1, 0.8).
    ck_bounds : tuple
        Bounds for harmonic contrast amplitudes. Default (-0.2, 0.2).
    o0_bounds : tuple
        Bounds for mean offset coefficient. Default (0.5, 1.5).
    ok_bounds : tuple
        Bounds for harmonic offset amplitudes. Default (-0.3, 0.3).
    """

    mode: Literal["independent", "fourier", "auto"] = "auto"
    fourier_order: int = 2
    auto_threshold: int = 6

    # Bounds for Fourier coefficients
    c0_bounds: tuple[float, float] = (0.1, 0.8)  # Mean contrast
    ck_bounds: tuple[float, float] = (-0.2, 0.2)  # Harmonic amplitudes
    o0_bounds: tuple[float, float] = (0.5, 1.5)  # Mean offset
    ok_bounds: tuple[float, float] = (-0.3, 0.3)  # Harmonic amplitudes

    @classmethod
    def from_dict(cls, config_dict: dict) -> FourierReparamConfig:
        """Create config from dictionary."""
        return cls(
            mode=config_dict.get("per_angle_mode", "auto"),
            fourier_order=config_dict.get("fourier_order", 2),
            auto_threshold=config_dict.get("fourier_auto_threshold", 6),
            c0_bounds=tuple(config_dict.get("c0_bounds", (0.1, 0.8))),
            ck_bounds=tuple(config_dict.get("ck_bounds", (-0.2, 0.2))),
            o0_bounds=tuple(config_dict.get("o0_bounds", (0.5, 1.5))),
            ok_bounds=tuple(config_dict.get("ok_bounds", (-0.3, 0.3))),
        )


class FourierReparameterizer:
    """Handles conversion between Fourier coefficients and per-angle values.

    This class provides the core functionality for Fourier reparameterization:
    1. Convert per-angle values to Fourier coefficients (initialization)
    2. Convert Fourier coefficients to per-angle values (model evaluation)
    3. Compute Jacobian for covariance transformation

    The Fourier basis ensures smooth variation of contrast/offset with angle,
    preventing the optimizer from using per-angle parameters to absorb
    angle-dependent physical signals (like the shear term cos(φ₀-φ)).

    Parameters
    ----------
    phi_angles : np.ndarray
        Unique phi angles in radians, shape (n_phi,).
    config : FourierReparamConfig
        Fourier configuration.

    Attributes
    ----------
    n_phi : int
        Number of unique phi angles.
    n_coeffs : int
        Total number of Fourier coefficients (contrast + offset).
    n_coeffs_per_param : int
        Coefficients per parameter type (contrast or offset).
    use_fourier : bool
        Whether Fourier mode is active.

    Examples
    --------
    >>> phi_angles = np.linspace(-np.pi, np.pi, 23)
    >>> config = FourierReparamConfig(mode="fourier", fourier_order=2)
    >>> fourier = FourierReparameterizer(phi_angles, config)
    >>> # Convert initial per-angle values to Fourier
    >>> contrast = np.full(23, 0.3)
    >>> offset = np.full(23, 1.0)
    >>> fourier_coeffs = fourier.per_angle_to_fourier(contrast, offset)
    >>> # Convert back during model evaluation
    >>> contrast_out, offset_out = fourier.fourier_to_per_angle(fourier_coeffs)
    """

    def __init__(self, phi_angles: np.ndarray, config: FourierReparamConfig):
        """Initialize Fourier reparameterizer.

        Parameters
        ----------
        phi_angles : np.ndarray
            Unique phi angles in radians, shape (n_phi,).
        config : FourierReparamConfig
            Fourier configuration.
        """
        self.phi_angles = np.asarray(phi_angles, dtype=np.float64)
        self.config = config
        self.n_phi = len(phi_angles)
        self._basis_matrix: np.ndarray | None = None

        # Determine effective mode
        self.use_fourier = self._determine_mode()

        if self.use_fourier:
            # Number of coefficients per parameter: c0 + order×(ck, sk)
            self.n_coeffs_per_param = 1 + 2 * config.fourier_order
            self.n_coeffs = 2 * self.n_coeffs_per_param  # contrast + offset

            # Precompute Fourier basis matrix for efficiency
            self._basis_matrix = self._compute_basis_matrix()

            # Performance Optimization (Spec 001 - FR-009, T046): Compute explicit rcond
            # based on matrix dimensions and machine precision, rather than letting numpy
            # use its default which may be overly conservative or vary between versions.
            # rcond = max(m, n) * eps is the standard recommendation.
            self._rcond = (
                max(self.n_phi, self.n_coeffs_per_param) * np.finfo(np.float64).eps
            )

            logger.info(
                f"Fourier reparameterization enabled: "
                f"{self.n_coeffs} coefficients for {self.n_phi} angles "
                f"(order={config.fourier_order}, rcond={self._rcond:.2e})"
            )
        else:
            # Independent mode: n_phi per parameter
            self.n_coeffs_per_param = self.n_phi
            self.n_coeffs = 2 * self.n_phi
            self._basis_matrix = None

            logger.info(
                f"Independent per-angle mode: {self.n_coeffs} parameters "
                f"for {self.n_phi} angles"
            )

    def _determine_mode(self) -> bool:
        """Determine whether to use Fourier mode.

        Returns
        -------
        bool
            True if Fourier mode should be used.
        """
        if self.config.mode == "fourier":
            # Check if Fourier is feasible (need enough angles)
            min_angles = 1 + 2 * self.config.fourier_order
            if self.n_phi < min_angles:
                logger.warning(
                    f"Fourier mode requested but n_phi={self.n_phi} < "
                    f"min_angles={min_angles}. Falling back to independent mode."
                )
                return False
            return True

        elif self.config.mode == "independent":
            return False

        else:  # auto
            # Use Fourier when n_phi > threshold AND feasible
            min_angles = 1 + 2 * self.config.fourier_order
            use_fourier = (
                self.n_phi > self.config.auto_threshold and self.n_phi >= min_angles
            )
            if use_fourier:
                logger.debug(
                    f"Auto mode: using Fourier (n_phi={self.n_phi} > "
                    f"threshold={self.config.auto_threshold})"
                )
            else:
                logger.debug(
                    f"Auto mode: using independent (n_phi={self.n_phi} <= "
                    f"threshold={self.config.auto_threshold})"
                )
            return use_fourier

    def _compute_basis_matrix(self) -> np.ndarray:
        """Compute Fourier basis matrix B where values = B @ coeffs.

        Returns
        -------
        np.ndarray
            Basis matrix of shape (n_phi, n_coeffs_per_param).
        """
        order = self.config.fourier_order

        B = np.zeros((self.n_phi, self.n_coeffs_per_param))
        B[:, 0] = 1.0  # c0 term (constant)

        col = 1
        for k in range(1, order + 1):
            B[:, col] = np.cos(k * self.phi_angles)  # ck term
            B[:, col + 1] = np.sin(k * self.phi_angles)  # sk term
            col += 2

        return B

    def get_basis_matrix(self) -> np.ndarray | None:
        """Get the Fourier basis matrix for covariance transformation.

        Returns
        -------
        np.ndarray or None
            Basis matrix of shape (n_phi, n_coeffs_per_param) if in Fourier mode,
            None if in independent mode. The basis matrix B satisfies:
            per_angle_values = B @ fourier_coeffs

        Notes
        -----
        Used for transforming covariance from Fourier space to per-angle space:
        pcov_per_angle = B @ pcov_fourier @ B.T
        """
        return self._basis_matrix

    @property
    def order(self) -> int:
        """Get the Fourier order (number of harmonics).

        Returns
        -------
        int
            Fourier order from config.
        """
        return self.config.fourier_order

    def fourier_to_per_angle(
        self, fourier_coeffs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert Fourier coefficients to per-angle contrast/offset.

        Parameters
        ----------
        fourier_coeffs : np.ndarray
            Shape (n_coeffs,) = [c0,c1,s1,c2,s2,...,o0,o1,t1,o2,t2,...].

        Returns
        -------
        contrast : np.ndarray
            Per-angle contrast values, shape (n_phi,).
        offset : np.ndarray
            Per-angle offset values, shape (n_phi,).

        Raises
        ------
        ValueError
            If fourier_coeffs has wrong shape.
        """
        # Validate input array bounds
        fourier_coeffs = np.asarray(fourier_coeffs, dtype=np.float64)
        if fourier_coeffs.ndim != 1:
            raise ValueError(
                f"fourier_coeffs must be 1D array, got shape {fourier_coeffs.shape}"
            )
        if len(fourier_coeffs) != self.n_coeffs:
            raise ValueError(
                f"Expected {self.n_coeffs} Fourier coefficients, got {len(fourier_coeffs)}"
            )

        if not self.use_fourier:
            # Independent mode: first half is contrast, second half is offset
            contrast = fourier_coeffs[: self.n_phi].copy()
            offset = fourier_coeffs[self.n_phi :].copy()
            return contrast, offset

        n_half = self.n_coeffs_per_param
        contrast_coeffs = fourier_coeffs[:n_half]
        offset_coeffs = fourier_coeffs[n_half:]

        contrast = self._basis_matrix @ contrast_coeffs
        offset = self._basis_matrix @ offset_coeffs

        return contrast, offset

    def per_angle_to_fourier(
        self, contrast: np.ndarray, offset: np.ndarray
    ) -> np.ndarray:
        """Convert per-angle values to Fourier coefficients.

        Uses least squares fitting when n_phi > n_coeffs_per_param.

        Parameters
        ----------
        contrast : np.ndarray
            Per-angle contrast values, shape (n_phi,).
        offset : np.ndarray
            Per-angle offset values, shape (n_phi,).

        Returns
        -------
        np.ndarray
            Fourier coefficients, shape (n_coeffs,).

        Raises
        ------
        ValueError
            If contrast or offset has wrong shape.
        """
        # Validate input array bounds
        contrast = np.asarray(contrast, dtype=np.float64)
        offset = np.asarray(offset, dtype=np.float64)

        if contrast.ndim != 1:
            raise ValueError(f"contrast must be 1D array, got shape {contrast.shape}")
        if offset.ndim != 1:
            raise ValueError(f"offset must be 1D array, got shape {offset.shape}")
        if len(contrast) != self.n_phi:
            raise ValueError(
                f"Expected {self.n_phi} contrast values, got {len(contrast)}"
            )
        if len(offset) != self.n_phi:
            raise ValueError(f"Expected {self.n_phi} offset values, got {len(offset)}")

        if not self.use_fourier:
            # Independent mode: just concatenate
            return np.concatenate([contrast, offset])

        # Invariant: ``use_fourier=True`` implies the basis matrix was built
        # in ``__init__``. mypy can't see the invariant through ``Optional``
        # — narrow with an explicit assert that doubles as a runtime guard.
        assert self._basis_matrix is not None, (
            "use_fourier=True requires _basis_matrix to be initialized"
        )

        # Least squares: B @ coeffs = values
        # coeffs = (B^T B)^{-1} B^T values = lstsq solution
        # Performance Optimization (Spec 001 - FR-009, T047): Use precomputed rcond
        contrast_coeffs, residuals_c, rank_c, s_c = np.linalg.lstsq(
            self._basis_matrix, contrast, rcond=float(self._rcond)
        )
        offset_coeffs, residuals_o, rank_o, s_o = np.linalg.lstsq(
            self._basis_matrix, offset, rcond=float(self._rcond)
        )

        # Log fit quality if there are residuals
        if len(residuals_c) > 0 and residuals_c[0] > 0.01:
            rms_c = np.sqrt(residuals_c[0] / self.n_phi)
            logger.debug(f"Fourier fit residual (contrast): {rms_c:.4f}")
        if len(residuals_o) > 0 and residuals_o[0] > 0.01:
            rms_o = np.sqrt(residuals_o[0] / self.n_phi)
            logger.debug(f"Fourier fit residual (offset): {rms_o:.4f}")

        return np.concatenate([contrast_coeffs, offset_coeffs])

    def get_jacobian_transform(self) -> np.ndarray:
        """Get Jacobian of transformation: d(per_angle)/d(fourier).

        Used for covariance transformation back to per-angle space:
            Cov_per_angle = J @ Cov_fourier @ J.T

        Returns
        -------
        np.ndarray
            Jacobian matrix of shape (2*n_phi, n_coeffs).
        """
        if not self.use_fourier:
            # Independent mode: Jacobian is identity
            return np.eye(self.n_coeffs)

        n_half = self.n_coeffs_per_param
        jacobian = np.zeros((2 * self.n_phi, self.n_coeffs))

        # d(contrast_i)/d(contrast_coeffs)
        jacobian[: self.n_phi, :n_half] = self._basis_matrix

        # d(offset_i)/d(offset_coeffs)
        jacobian[self.n_phi :, n_half:] = self._basis_matrix

        return jacobian

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds for Fourier coefficients.

        Returns
        -------
        lower : np.ndarray
            Lower bounds, shape (n_coeffs,).
        upper : np.ndarray
            Upper bounds, shape (n_coeffs,).
        """
        if not self.use_fourier:
            # Independent mode: use standard per-angle bounds
            # Contrast bounds: typically (0, 1)
            # Offset bounds: typically (0, 2)
            contrast_lower = np.full(self.n_phi, self.config.c0_bounds[0])
            contrast_upper = np.full(self.n_phi, self.config.c0_bounds[1])
            offset_lower = np.full(self.n_phi, self.config.o0_bounds[0])
            offset_upper = np.full(self.n_phi, self.config.o0_bounds[1])

            lower = np.concatenate([contrast_lower, offset_lower])
            upper = np.concatenate([contrast_upper, offset_upper])
            return lower, upper

        n_half = self.n_coeffs_per_param

        lower = np.zeros(self.n_coeffs)
        upper = np.zeros(self.n_coeffs)

        # Contrast coefficients
        lower[0] = self.config.c0_bounds[0]  # c0 (mean contrast)
        upper[0] = self.config.c0_bounds[1]
        for i in range(1, n_half):
            lower[i] = self.config.ck_bounds[0]  # ck, sk harmonics
            upper[i] = self.config.ck_bounds[1]

        # Offset coefficients
        lower[n_half] = self.config.o0_bounds[0]  # o0 (mean offset)
        upper[n_half] = self.config.o0_bounds[1]
        for i in range(n_half + 1, self.n_coeffs):
            lower[i] = self.config.ok_bounds[0]  # ok, tk harmonics
            upper[i] = self.config.ok_bounds[1]

        return lower, upper

    def get_initial_coefficients(
        self, contrast_init: float | np.ndarray, offset_init: float | np.ndarray
    ) -> np.ndarray:
        """Get initial Fourier coefficients from initial values.

        Parameters
        ----------
        contrast_init : float or np.ndarray
            Initial contrast (scalar for uniform, array for per-angle).
        offset_init : float or np.ndarray
            Initial offset (scalar for uniform, array for per-angle).

        Returns
        -------
        np.ndarray
            Initial Fourier coefficients.
        """
        # Handle scalar inputs. ``np.isscalar`` doesn't narrow ``Any`` to a
        # SupportsFloat in mypy's stubs (it just returns ``bool``); coerce
        # via ``np.asarray(...).item()`` which always yields a Python scalar.
        if np.isscalar(contrast_init):
            contrast = np.full(self.n_phi, float(np.asarray(contrast_init).item()))
        else:
            contrast = np.asarray(contrast_init)

        if np.isscalar(offset_init):
            offset = np.full(self.n_phi, float(np.asarray(offset_init).item()))
        else:
            offset = np.asarray(offset_init)

        return self.per_angle_to_fourier(contrast, offset)

    def get_coefficient_labels(self) -> list[str]:
        """Get parameter labels for Fourier coefficients.

        Returns
        -------
        list of str
            Parameter labels.
        """
        if not self.use_fourier:
            labels = [f"contrast[{i}]" for i in range(self.n_phi)]
            labels += [f"offset[{i}]" for i in range(self.n_phi)]
            return labels

        labels = ["contrast_c0"]
        for k in range(1, self.config.fourier_order + 1):
            labels.append(f"contrast_c{k}")
            labels.append(f"contrast_s{k}")

        labels.append("offset_c0")
        for k in range(1, self.config.fourier_order + 1):
            labels.append(f"offset_c{k}")
            labels.append(f"offset_s{k}")

        return labels

    def to_fourier(self, per_angle_values: np.ndarray) -> np.ndarray:
        """Convert a single per-angle array to Fourier coefficients.

        Convenience method for transforming one group (contrast or offset)
        at a time, rather than both together.

        Parameters
        ----------
        per_angle_values : np.ndarray
            Per-angle values, shape (n_phi,).

        Returns
        -------
        np.ndarray
            Fourier coefficients, shape (n_coeffs_per_param,).

        Raises
        ------
        ValueError
            If per_angle_values has wrong shape.
        """
        # Validate input array bounds
        per_angle_values = np.asarray(per_angle_values, dtype=np.float64)
        if per_angle_values.ndim != 1:
            raise ValueError(
                f"per_angle_values must be 1D array, got shape {per_angle_values.shape}"
            )
        if len(per_angle_values) != self.n_phi:
            raise ValueError(
                f"Expected {self.n_phi} values, got {len(per_angle_values)}"
            )

        if not self.use_fourier:
            # Independent mode: return as-is
            return per_angle_values.copy()

        # Invariant: see per_angle_to_fourier above — use_fourier=True
        # guarantees _basis_matrix was built.
        assert self._basis_matrix is not None, (
            "use_fourier=True requires _basis_matrix to be initialized"
        )

        # Least squares fit
        # Performance Optimization (Spec 001 - FR-009, T047): Use precomputed rcond
        coeffs, _, _, _ = np.linalg.lstsq(
            self._basis_matrix, per_angle_values, rcond=self._rcond
        )
        return coeffs

    def from_fourier(self, fourier_coeffs: np.ndarray) -> np.ndarray:
        """Convert Fourier coefficients to per-angle values for a single group.

        Convenience method for transforming one group (contrast or offset)
        at a time, rather than both together.

        Parameters
        ----------
        fourier_coeffs : np.ndarray
            Fourier coefficients, shape (n_coeffs_per_param,).

        Returns
        -------
        np.ndarray
            Per-angle values, shape (n_phi,).

        Raises
        ------
        ValueError
            If fourier_coeffs has wrong shape.
        """
        # Validate input array bounds
        fourier_coeffs = np.asarray(fourier_coeffs, dtype=np.float64)
        if fourier_coeffs.ndim != 1:
            raise ValueError(
                f"fourier_coeffs must be 1D array, got shape {fourier_coeffs.shape}"
            )
        if len(fourier_coeffs) != self.n_coeffs_per_param:
            raise ValueError(
                f"Expected {self.n_coeffs_per_param} coefficients, got {len(fourier_coeffs)}"
            )

        if not self.use_fourier:
            # Independent mode: return as-is
            return fourier_coeffs.copy()

        # Matrix multiply to get per-angle values
        return self._basis_matrix @ fourier_coeffs

    def get_diagnostics(self) -> dict:
        """Get Fourier reparameterization diagnostics.

        Returns
        -------
        dict
            Diagnostic information.
        """
        return {
            "use_fourier": self.use_fourier,
            "mode": self.config.mode,
            "n_phi": self.n_phi,
            "n_coeffs": self.n_coeffs,
            "n_coeffs_per_param": self.n_coeffs_per_param,
            "fourier_order": self.config.fourier_order if self.use_fourier else None,
            "reduction_ratio": (
                self.n_coeffs / (2 * self.n_phi) if self.use_fourier else 1.0
            ),
        }


def create_fourier_model_wrapper(
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    fourier: FourierReparameterizer,
    n_physical: int,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create a model function wrapper that handles Fourier conversion.

    The wrapper converts Fourier coefficients to per-angle values before
    calling the underlying model function.

    Parameters
    ----------
    model_fn : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Original model function that expects per-angle parameters:
        f(params, x) where params = [contrast_per_angle, offset_per_angle, physical]
    fourier : FourierReparameterizer
        Fourier reparameterizer instance.
    n_physical : int
        Number of physical parameters.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], np.ndarray]
        Wrapped model function that accepts Fourier parameters:
        f(params, x) where params = [fourier_coeffs, physical]
    """

    def wrapped_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Model wrapper that converts Fourier to per-angle."""
        # Split params into Fourier coefficients and physical
        n_fourier = fourier.n_coeffs
        fourier_coeffs = params[:n_fourier]
        physical_params = params[n_fourier:]

        # Convert Fourier to per-angle
        contrast, offset = fourier.fourier_to_per_angle(fourier_coeffs)

        # Reconstruct full per-angle parameter vector
        full_params = np.concatenate([contrast, offset, physical_params])

        # Call original model
        return model_fn(full_params, x)

    return wrapped_model
