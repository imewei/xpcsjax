"""Shared covariance post-processing for the NLSQ result paths.

The solver-side covariance (nlsq ``curve_fit`` pcov) is ``s² · (JᵀJ)⁻¹`` with
``s² = cost / (ysize − n_params)`` where ``ysize`` is the length of the residual
VECTOR the solver saw. On the stratification engine
(:class:`~xpcsjax.optimization.nlsq.strategies.residual_jit.StratifiedResidualFunctionJIT`)
that vector carries chunk padding and the zero-masked ``t1 == t2`` rows, none
of which are observations, so ``ysize`` over-counts the degrees of freedom and
the reported uncertainties are too small by ``sqrt((n_valid − p) / (ysize − p))``.
:func:`rescale_covariance_dof` undoes exactly that factor after the solve, so
the solve itself (and every rtol-gated parity baseline on it) is untouched.
"""

from __future__ import annotations

import numpy as np


def rescale_covariance_dof(
    pcov: np.ndarray,
    *,
    n_rows_solver: int,
    n_valid: int,
    n_params: int,
) -> np.ndarray:
    """Re-express a solver covariance on the valid-observation dof.

    Parameters
    ----------
    pcov : np.ndarray
        Solver covariance, already scaled by ``cost / (n_rows_solver − n_params)``.
    n_rows_solver : int
        Length of the residual vector the solver scaled ``pcov`` with
        (padding and masked rows included).
    n_valid : int
        Number of genuine observations in that vector.
    n_params : int
        Optimizer parameter count used in the solver's dof.

    Returns
    -------
    np.ndarray
        ``pcov · (n_rows_solver − n_params) / (n_valid − n_params)``. Returned
        unchanged when the counts agree or when either dof is non-positive
        (the solver already reports an all-``inf`` covariance for that case).
    """
    pcov = np.asarray(pcov, dtype=np.float64)
    dof_solver = int(n_rows_solver) - int(n_params)
    dof_valid = int(n_valid) - int(n_params)
    if dof_solver == dof_valid or dof_solver <= 0 or dof_valid <= 0:
        return pcov
    return pcov * (dof_solver / dof_valid)


def finalize_covariance(
    pcov: np.ndarray | None, n_params: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Turn a solver covariance into ``(covariance, uncertainties, is_placeholder)``.

    One rule for every path: a covariance is REAL only when every entry is
    finite and every variance is strictly positive. Anything else — the
    solver's all-``inf`` singular marker, a pseudo-inverse whose null-space
    directions read as exactly ``0.0`` variance ("infinite precision"), a
    negative diagonal from a non-PD Hessian, or no covariance at all — is
    reported as all-``NaN`` with ``is_placeholder=True`` so it can never be
    mistaken for a measured uncertainty.
    """
    nan_cov = np.full((n_params, n_params), np.nan, dtype=np.float64)
    nan_unc = np.full(n_params, np.nan, dtype=np.float64)
    if pcov is None:
        return nan_cov, nan_unc, True
    pcov = np.asarray(pcov, dtype=np.float64)
    if pcov.shape != (n_params, n_params):
        return nan_cov, nan_unc, True
    diag = np.diag(pcov)
    if not (np.all(np.isfinite(pcov)) and np.all(diag > 0.0)):
        return nan_cov, nan_unc, True
    return pcov, np.sqrt(diag), False


_EPS = float(np.finfo(np.float64).eps)


def _rho(loss: str, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rho, rho', rho'')`` of the scipy/nlsq robust losses at ``z = (f/f_scale)²``."""
    if loss == "linear":
        return z, np.ones_like(z), np.zeros_like(z)
    if loss == "huber":
        big = z > 1.0
        rho0 = np.where(big, 2.0 * np.sqrt(z) - 1.0, z)
        rho1 = np.where(big, z**-0.5, 1.0)
        rho2 = np.where(big, -0.5 * z**-1.5, 0.0)
        return rho0, rho1, rho2
    if loss == "soft_l1":
        t = 1.0 + z
        return 2.0 * (np.sqrt(t) - 1.0), t**-0.5, -0.5 * t**-1.5
    if loss == "cauchy":
        return np.log1p(z), 1.0 / (1.0 + z), -1.0 / (1.0 + z) ** 2
    if loss == "arctan":
        return np.arctan(z), 1.0 / (1.0 + z**2), -2.0 * z / (1.0 + z**2) ** 2
    raise ValueError(f"unknown robust loss {loss!r}")


def robust_scale_residual_jacobian(
    f: np.ndarray, J: np.ndarray, *, loss: str = "linear", f_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Mirror nlsq/scipy ``scale_for_robust_loss_function`` on the host.

    Returns ``(f_scaled, J_scaled, cost)`` where ``cost = Σ ρ`` (nlsq's
    ``2 * res.cost``), so ``cost / (n − p) · (J_scaledᵀ J_scaled)⁻¹`` is the
    SAME estimator nlsq's ``curve_fit`` reports for a solve run with ``loss``.
    For ``loss="linear"`` this is the identity and ``cost`` is the plain SSR.
    """
    f = np.asarray(f, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    if loss == "linear":
        return f, J, float(np.sum(f * f))
    z = (f / f_scale) ** 2
    rho0, rho1, rho2 = _rho(loss, z)
    rho0 = rho0 * f_scale**2
    rho2 = rho2 / f_scale**2
    j_scale = rho1 + 2.0 * rho2 * f * f
    j_scale = np.sqrt(np.where(j_scale < _EPS, _EPS, j_scale))
    return f * rho1 / j_scale, J * j_scale[:, None], float(np.sum(rho0))


def gauss_newton_covariance(
    f: np.ndarray,
    J: np.ndarray,
    *,
    n_valid: int,
    n_params: int,
    loss: str = "linear",
    f_scale: float = 1.0,
) -> np.ndarray:
    """Strict ``s² (JᵀJ)⁻¹`` on the same robust-loss convention as the solver.

    ``s² = cost / max(n_valid − n_params, 1)``. No pseudo-inverse: a singular
    ``JᵀJ`` raises ``numpy.linalg.LinAlgError`` and a non-finite or
    non-positive-diagonal result raises ``ValueError`` — callers map both to
    :func:`finalize_covariance`'s all-``NaN`` placeholder.
    """
    _, J_s, cost = robust_scale_residual_jacobian(f, J, loss=loss, f_scale=f_scale)
    s2 = cost / max(int(n_valid) - int(n_params), 1)
    JTJ = J_s.T @ J_s
    del J_s
    pcov = np.linalg.inv(JTJ) * s2
    if not (np.all(np.isfinite(pcov)) and np.all(np.diag(pcov) > 0.0)):
        raise ValueError("covariance non-finite or non-positive diagonal at popt")
    return pcov
