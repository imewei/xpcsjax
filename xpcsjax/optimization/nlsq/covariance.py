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

from collections.abc import Callable

import jax
import jax.numpy as jnp
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


# Column-block width for the host covariance Jacobian (see
# _chunked_jacfwd_dense). n_params is small (16 for two_component); 4 blocks
# of 4 tangents cap the forward-AD tangent width at 4 instead of n_params
# while staying byte-identical.
_COV_JACFWD_COL_BLOCK = 4


def _chunked_jacfwd_dense(
    fn: Callable[[np.ndarray], jnp.ndarray],
    x: np.ndarray,
    *,
    col_block: int = _COV_JACFWD_COL_BLOCK,
) -> np.ndarray:
    """Column-blocked forward-mode Jacobian, numerically identical to ``jax.jacfwd``.

    ``jax.jacfwd(fn)(x)`` builds the full ``(n_out, n_in)`` Jacobian by pushing
    all ``n_in`` basis tangents through ``fn`` at once, so every intermediate of
    ``fn`` is materialised at width ``n_in``. For the heterodyne stratified-LS
    covariance ``n_out`` is the full support (~23M points) and ``n_in`` is the
    16-ish joint parameters, so that ``n_in``-wide tangent is the dominant
    transient -- the ~3 GB+ spike that drives the post-solve memory peak.

    Computing the columns in small blocks (each a ``vmap``'d JVP over ``col_block``
    basis vectors, moved to host and released before the next block) yields the
    SAME columns -- ``jvp`` is exact and column order is preserved -- while capping
    the live tangent width at ``col_block``. The assembled ``J`` (and therefore
    ``JᵀJ`` and the covariance) is byte-identical to ``jax.jacfwd`` up to XLA
    fusion noise (<= ULP); this only affects the post-solve covariance, never the
    fit trajectory.

    Also used by ``strategies/stratified_ls.py``'s (>=1 M point) post-solve
    Jacobian, which had the same full-width ``jax.jacfwd`` memory spike as the
    heterodyne stratified-LS path this was first written for.

    Parameters
    ----------
    fn : callable
        ``params (n_in,) -> residuals (n_out,)`` (the joint residual). May be
        ``jax.jit``-wrapped.
    x : np.ndarray
        Point at which to evaluate the Jacobian (the converged ``popt``).
    col_block : int, optional
        Number of parameter columns evaluated per block. Defaults to
        :data:`_COV_JACFWD_COL_BLOCK`.

    Returns
    -------
    np.ndarray, (n_out, n_in) float64
        The dense Jacobian, matching ``np.asarray(jax.jacfwd(fn)(x))``.
    """
    x_jax = jnp.asarray(x, dtype=jnp.float64)
    n_in = int(x_jax.shape[0])
    eye = jnp.eye(n_in, dtype=x_jax.dtype)

    def _jvp_col(tangent: jnp.ndarray) -> jnp.ndarray:
        # jvp(fn, primal, e_j)[1] == d fn / d x_j == column j of the Jacobian.
        return jax.jvp(fn, (x_jax,), (tangent,))[1]

    jt: np.ndarray | None = None  # (n_in, n_out), filled block by block
    for c0 in range(0, n_in, max(1, col_block)):
        tangents = eye[c0 : c0 + max(1, col_block)]  # (b, n_in)
        # (b, n_out): row r is column (c0 + r) of J. Pull to host and let the
        # device buffer for this block free before the next block allocates.
        block_cols = np.asarray(jax.vmap(_jvp_col)(tangents), dtype=np.float64)
        if jt is None:
            # Preallocate once (no concatenate copy): at >=1 M points x tens of
            # params the dense J is several GB, so a second copy is the peak.
            jt = np.empty((n_in, block_cols.shape[1]), dtype=np.float64)
        jt[c0 : c0 + block_cols.shape[0]] = block_cols

    if jt is None:  # n_in == 0
        return np.empty((0, 0), dtype=np.float64)
    # (n_in, n_out) -> (n_out, n_in) as a view.
    return jt.T
