"""NLSQ Physics Backend - Meshgrid Computations Only
======================================================

This module provides meshgrid physics computations specifically for
Nonlinear Least Squares (NLSQ) optimization. It contains ONLY meshgrid functions
and has NO element-wise paired array computations.

Key Design Principles:
- Meshgrid computations for 2D correlation matrices
- No element-wise modes, no dispatchers
- Direct JIT-compiled functions only

Physical Model:
g₂(φ,t₁,t₂) = offset + contrast × [g₁(φ,t₁,t₂)]²
g₁_total = g₁_diffusion × g₁_shear

Usage:
  from xpcsjax.core.physics_nlsq import compute_g2_scaled
"""

import jax.numpy as jnp
from jax import jit

from xpcsjax.core.physics_utils import (
    PI,
    safe_sinc,
)
from xpcsjax.core.physics_utils import (
    calculate_diffusion_coefficient as _calculate_diffusion_coefficient_impl_jax,
)
from xpcsjax.core.physics_utils import (
    calculate_shear_rate as _calculate_shear_rate_impl_jax,
)
from xpcsjax.core.physics_utils import (
    create_time_integral_matrix as _create_time_integral_matrix_impl_jax,
)
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

# JAX availability flag (for backward compatibility)
jax_available = True


# =============================================================================
# MESHGRID CORE PHYSICS (NLSQ ONLY)
# =============================================================================


@jit
def _compute_g1_diffusion_meshgrid(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    dt: float,
) -> jnp.ndarray:
    """Meshgrid diffusion computation for NLSQ optimization.

    Computes g1_diffusion for 2D meshgrid (all combinations of t1[i], t2[j]).
    This is the ONLY diffusion function for NLSQ - no element-wise mode.

    Args:
        params: Physical parameters [D0, alpha, D_offset, ...]
        t1: Time meshgrid (2D) or 1D time array (PHYSICAL TIME in seconds)
        t2: Time meshgrid (2D) or 1D time array (PHYSICAL TIME in seconds)
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt
        dt: Time step [seconds] - used ONLY for wavevector_q_squared_half_dt calculation

    Returns:
        Diffusion contribution to g1 (2D array: (n_times, n_times))

    Note:
        The data loader (xpcs_loader.py) converts frame indices to physical time:
        time_1d = np.linspace(0, dt * (end_frame - start_frame), matrix_size)
        So t1/t2 arrays contain physical time [0.0, 0.1, 0.2, ...], NOT frame indices.
    """
    D0, alpha, D_offset = params[0], params[1], params[2]

    # Extract time array (t1 and t2 should be identical)
    # Handle all dimensionality cases: 0D (scalar), 1D arrays, and 2D meshgrids
    if t1.ndim == 2:
        # For meshgrid with indexing="ij": t1 varies along rows (axis 0), constant along columns
        # So extract first COLUMN to get unique t1 values
        time_array = t1[:, 0]  # Extract first column for unique t1 values (in seconds)
    elif t1.ndim == 0:
        # Handle 0-dimensional (scalar) input
        time_array = jnp.atleast_1d(t1)
    else:
        # Handle 1D and other cases
        time_array = jnp.atleast_1d(t1)

    # CRITICAL FIX (Nov 11, 2025): time_array is ALREADY physical time in seconds
    # Data loader converts: time_1d = np.linspace(0, dt*(end-start), size)
    # Result: time_array = [0.0, 0.1, 0.2, ...] seconds (NOT frame indices!)
    # DO NOT multiply by dt - that would cause 10× time scale error!

    # Calculate D(t) at each time point (time_array already in seconds)
    D_t = _calculate_diffusion_coefficient_impl_jax(time_array, D0, alpha, D_offset)

    # Create diffusion integral matrix using cumulative sums
    # This gives matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ |∫D(t)dt from i to j|
    D_integral = _create_time_integral_matrix_impl_jax(D_t)

    # Compute g1 correlation using log-space for numerical stability
    # This matches reference: g1 = exp(-wavevector_q_squared_half_dt * D_integral)
    #
    # LOG-SPACE CALCULATION FIX (Oct 2025):
    # Computing in log-space preserves precision across full dynamic range.
    # Old approach: clip(g1, 1e-10, 1.0) caused artificial plateaus (~16% of data)
    # New approach: clip in log-space, then exp() - no artificial plateaus
    log_g1 = -wavevector_q_squared_half_dt * D_integral

    # Clip in log-space to prevent numerical overflow/underflow
    # -700 → exp(-700) ≈ 1e-304 (near machine precision)
    # 0 → exp(0) = 1.0 (maximum physical value)
    log_g1_bounded = jnp.clip(log_g1, -700.0, 0.0)

    # Compute exponential — log_g1_bounded is already clipped to [-700, 0],
    # so jnp.exp is safe (no overflow risk).
    g1_result = jnp.exp(log_g1_bounded)

    # P1-2: Removed jnp.minimum(g1_result, 1.0) — the log-space clip above
    # (jnp.clip(log_g1, -700, 0)) already guarantees g1 = exp(log_g1) ≤ 1.0.
    # The hard min killed gradients at g1=1.0 (diagonal elements), harming NLSQ Jacobians.
    return g1_result


@jit
def _compute_g1_shear_meshgrid(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    sinc_prefactor: float,
    dt: float,
) -> jnp.ndarray:
    """Meshgrid shear computation for NLSQ optimization.

    Computes g1_shear for 2D meshgrid (all combinations of t1[i], t2[j])
    across all phi angles. This is the ONLY shear function for NLSQ - no element-wise mode.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1: Time meshgrid (2D) or 1D time array (PHYSICAL TIME in seconds)
        t2: Time meshgrid (2D) or 1D time array (PHYSICAL TIME in seconds)
        phi: Scattering angles (1D array)
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt
        dt: Time step [seconds] - used ONLY for sinc_prefactor calculation

    Returns:
        Shear contribution to g1 (3D array: (n_phi, n_times, n_times))

    Note:
        The data loader (xpcs_loader.py) converts frame indices to physical time.
        So t1/t2 arrays contain physical time [0.0, 0.1, 0.2, ...], NOT frame indices.
    """
    # Check params length - if < 7, we're in static mode (no shear)
    if params.shape[0] < 7:
        # Return ones for all phi angles and time combinations (g1_shear = 1)
        phi_array = jnp.atleast_1d(phi)
        n_phi = phi_array.shape[0]
        n_times = t1.shape[0]
        return jnp.ones((n_phi, n_times, n_times))

    gamma_dot_0, beta, gamma_dot_offset, phi0 = (
        params[3],
        params[4],
        params[5],
        params[6],
    )

    # Extract time array from t1 and t2 (should be identical)
    # Handle all dimensionality cases: 0D (scalar), 1D arrays, and 2D meshgrids
    if t1.ndim == 2:
        # For meshgrid with indexing="ij": t1 varies along rows (axis 0), constant along columns
        # So extract first COLUMN to get unique t1 values
        time_array = t1[:, 0]  # Extract first column for unique t1 values (in seconds)
    elif t1.ndim == 0:
        # Handle 0-dimensional (scalar) input
        time_array = jnp.atleast_1d(t1)
    else:
        # Handle 1D and other cases
        time_array = jnp.atleast_1d(t1)

    # CRITICAL FIX (Nov 11, 2025): time_array is ALREADY physical time in seconds
    # Data loader converts: time_1d = np.linspace(0, dt*(end-start), size)
    # Result: time_array = [0.0, 0.1, 0.2, ...] seconds (NOT frame indices!)
    # DO NOT multiply by dt - that would cause 10× time scale error!

    # Calculate γ̇(t) at each time point (time_array already in seconds)
    gamma_t = _calculate_shear_rate_impl_jax(
        time_array,
        gamma_dot_0,
        beta,
        gamma_dot_offset,
    )

    # Create shear integral matrix using cumulative sums
    # This gives matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ |∫γ̇(t)dt from i to j|
    # Create shear integral matrix using cumulative sums
    gamma_integral = _create_time_integral_matrix_impl_jax(gamma_t)

    # Ensure phi is a 1D array regardless of input shape.
    # Handles (1, 1, 1, 23), (23,), scalar, etc. uniformly.
    # reshape(-1) avoids a Python while loop that would cause JIT retracing.
    phi_array = jnp.asarray(phi, dtype=jnp.result_type(phi)).reshape(-1)
    n_phi = phi_array.shape[0]
    n_times = time_array.shape[0]

    # Vectorized computation: compute all phi angles at once
    # angle_diff shape: (n_phi,)
    angle_diff = jnp.deg2rad(phi0 - phi_array)  # Use phi_array for consistency
    cos_term = jnp.cos(angle_diff)  # shape: (n_phi,)

    # Broadcast: prefactor shape (n_phi,), gamma_integral shape (n_times, n_times)
    # Need to expand prefactor to (n_phi, 1, 1) for proper broadcasting
    prefactor = sinc_prefactor * cos_term[:, None, None]  # shape: (n_phi, 1, 1)

    # Ensure gamma_integral has the expected 2D shape.
    # Note: evaluated at JIT trace time only (ndim is concrete during tracing).
    # Will not trigger at runtime for traced arrays with unexpected shapes.
    if gamma_integral.ndim != 2:
        raise ValueError(
            f"gamma_integral should be 2D, got shape {gamma_integral.shape}",
        )

    # Compute phase matrix for all phi angles: shape (n_phi, n_times, n_times)
    # P2: Removed try/except — JAX traces through try/except at compile time;
    # exceptions cannot be caught at JIT runtime.
    phase = prefactor * gamma_integral  # Broadcast: (n_phi, 1, 1) * (n_times, n_times)

    # Compute sinc² values: [sinc(Φ)]² for all phi angles
    sinc_val = safe_sinc(phase)
    sinc2_result: jnp.ndarray = sinc_val**2

    return sinc2_result


@jit
def _compute_g1_total_meshgrid(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    sinc_prefactor: float,
    dt: float,
) -> jnp.ndarray:
    """Meshgrid total g1 computation for NLSQ optimization.

    Computes g1_total = g1_diffusion × g1_shear for 2D meshgrid.
    This is the ONLY g1_total function for NLSQ - no element-wise mode.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1: Time meshgrid (2D) or 1D time array (PHYSICAL TIME in seconds)
        t2: Time meshgrid (2D) or 1D time array (PHYSICAL TIME in seconds)
        phi: Scattering angles
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt
        dt: Time step [seconds]

    Returns:
        Total g1 correlation function (3D array: (n_phi, n_times, n_times))
    """
    # Compute diffusion contribution: shape (n_times, n_times)
    g1_diff = _compute_g1_diffusion_meshgrid(params, t1, t2, wavevector_q_squared_half_dt, dt)

    # Compute shear contribution: shape (n_phi, n_times, n_times)
    g1_shear = _compute_g1_shear_meshgrid(params, t1, t2, phi, sinc_prefactor, dt)

    # Broadcast g1_diff from (n_times, n_times) to (n_phi, n_times, n_times)
    n_phi = g1_shear.shape[0]
    g1_diff_broadcasted = jnp.broadcast_to(
        g1_diff[None, :, :],
        (n_phi, g1_diff.shape[0], g1_diff.shape[1]),
    )

    # Multiply: g₁_total[phi, i, j] = g₁_diffusion[i, j] × g₁_shear[phi, i, j]
    # P2: Removed try/except — JAX traces through try/except at compile time;
    # exceptions cannot be caught at JIT runtime. Dead code that added confusion.
    g1_total = g1_diff_broadcasted * g1_shear

    # P1-2: Keep only gradient-safe lower floor. Upper clip removed — g1_diff is
    # already bounded ≤ 1.0 from log-space clip, and g1_shear (sinc²) is naturally
    # bounded ≤ 1.0. Hard clips kill gradients at boundaries, harming NLSQ Jacobians.
    epsilon = 1e-10
    g1_bounded: jnp.ndarray = jnp.where(g1_total > epsilon, g1_total, epsilon)

    return g1_bounded


@jit
def _compute_g2_scaled_meshgrid(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    sinc_prefactor: float,
    contrast: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """Meshgrid g2 computation for NLSQ optimization.

    Core homodyne equation: g₂ = offset + contrast × [g₁]²

    The homodyne scattering equation is g₂ = 1 + β×g₁², where the baseline "1"
    is the constant background. In our implementation, this baseline is included
    in the offset parameter (offset ≈ 1.0 for physical measurements).

    For theoretical fits: Use offset=1.0, contrast=1.0 to get g₂ = 1 + g₁²
    For experimental fits: offset and contrast are free parameters centered around 1.0 and 0.5

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time points for correlation calculation (PHYSICAL TIME in seconds)
        phi: Scattering angles
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt
        contrast: Contrast parameter (β in literature) - typically [0, 1]
        offset: Baseline level (includes the "1" from physics) - typically ~1.0
        dt: Time step [seconds]

    Returns:
        g2 correlation function with scaled fitting and physical bounds applied
    """
    g1 = _compute_g1_total_meshgrid(
        params,
        t1,
        t2,
        phi,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
        dt,
    )

    # Homodyne physics: g₂ = offset + contrast × [g₁]²
    # The baseline "1" is included in the offset parameter (offset ≈ 1.0 for physical data)
    # NOTE: Bounds enforcement is handled by NLSQ optimization bounds, NOT by clipping here.
    # Clipping parameters inside the physics function creates a mismatch between:
    # - The parameter values NLSQ stores (unclipped)
    # - The parameter values actually used in computation (clipped)
    # This causes visualization to use different values than optimization used!
    g2 = offset + contrast * g1**2

    # P0-3: Removed hard jnp.clip(g2, 0.5, 2.5) — it kills gradients at boundaries.
    # For NLSQ (TRF optimizer), the bounds are enforced via parameter bounds, not g2 clipping.
    # Physical range (0.5-2.5) is enforced through parameter priors instead.
    result: jnp.ndarray = g2
    return result


# =============================================================================
# PUBLIC API FUNCTIONS (NLSQ ONLY)
# =============================================================================
# Note: apply_diagonal_correction is imported from physics_utils.py
# to eliminate code duplication across backends.


def compute_g2_scaled(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: float | jnp.ndarray,
    q: float,
    L: float,
    contrast: float | jnp.ndarray,
    offset: float | jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Compute g2 for NLSQ using meshgrid 2D matrices.

    NLSQ-specific function - creates meshgrid for 2D correlation matrices.
    No element-wise mode, direct meshgrid computation only.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time points for correlation calculation (1D arrays will be converted to meshgrid)
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        contrast: Contrast parameter (β in literature)
        offset: Baseline offset
        dt: Time step from configuration [s] (REQUIRED)

    Returns:
        g2 correlation function with scaled fitting and physical bounds applied
    """
    # Handle 1D time arrays by creating meshgrids
    if t1.ndim == 1 and t2.ndim == 1:
        # Normal time vectors: create 2D meshgrids for all (t1[i], t2[j]) pairs
        # CRITICAL: Must match caller's convention: t1_grid, t2_grid = meshgrid(t, t, 'ij')
        t1_grid, t2_grid = jnp.meshgrid(t1, t2, indexing="ij")
        t1 = t1_grid
        t2 = t2_grid

    # Compute the physics factors using configuration dt.
    # dt is required (callers must validate before calling; None is not accepted).
    dt_value = dt
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt_value
    sinc_prefactor = 0.5 / PI * q * L * dt_value

    result: jnp.ndarray = _compute_g2_scaled_meshgrid(
        params,
        t1,
        t2,
        phi,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
        contrast,
        offset,
        dt_value,
    )
    return result
