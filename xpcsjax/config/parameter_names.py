"""Parameter name constants for homodyne analysis.

Centralized parameter name definitions to ensure consistency across
model definitions and result processing.

This module defines the canonical parameter names and ordering for
both analysis modes (static_isotropic and laminar_flow).

Usage::

    from xpcsjax.config.parameter_names import (
        STATIC_ISOTROPIC_PARAMS,
        LAMINAR_FLOW_PARAMS,
        get_parameter_names
    )

Notes
-----
Version history:

- v2.1.1 (Nov 2025): Created to prevent parameter name mismatches.
- Fixed bug where the pjit backend used ``gamma_dot_0`` instead of
  ``gamma_dot_t0``.
"""


# =============================================================================
# PARAMETER NAME CONSTANTS
# =============================================================================

# Scaling parameters (common to all modes)
SCALING_PARAMS = ["contrast", "offset"]

# Static isotropic diffusion parameters (3 physical parameters)
STATIC_PHYSICAL_PARAMS = ["D0", "alpha", "D_offset"]

# Laminar flow shear parameters (4 additional parameters)
FLOW_PARAMS = ["gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]

# Complete parameter sets for each analysis mode
STATIC_ISOTROPIC_PARAMS = SCALING_PARAMS + STATIC_PHYSICAL_PARAMS
LAMINAR_FLOW_PARAMS = SCALING_PARAMS + STATIC_PHYSICAL_PARAMS + FLOW_PARAMS

# Parameter counts
NUM_PARAMS_STATIC = len(STATIC_ISOTROPIC_PARAMS)  # 5
NUM_PARAMS_LAMINAR = len(LAMINAR_FLOW_PARAMS)  # 9

# =============================================================================
# PARAMETER DESCRIPTIONS
# =============================================================================

PARAMETER_DESCRIPTIONS = {
    # Scaling parameters
    "contrast": "Contrast factor for c2 = contrast × c1² + offset",
    "offset": "Baseline offset for c2 = contrast × c1² + offset",
    # Diffusion parameters
    "D0": "Diffusion coefficient amplitude [Å²/s]",
    "alpha": "Anomalous diffusion exponent (α < 0: subdiffusion, α > 0: superdiffusion)",
    "D_offset": "Baseline diffusion offset [Å²/s]",
    # Flow parameters (laminar flow only)
    "gamma_dot_t0": "Shear rate amplitude at t=0 [s⁻¹]",
    "beta": "Shear rate time exponent (γ̇(t) = γ̇₀ × t^β)",
    "gamma_dot_t_offset": "Baseline shear rate offset [s⁻¹]",
    "phi0": "Flow direction angle [degrees]",
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Re-export the canonical AnalysisMode literal (defined in parameter_registry)
# so the type hints in this module's function signatures stay in sync with the
# full set of supported modes — previously this duplicate was missing
# ``two_component`` and silently filtered heterodyne values out of any guard
# that imported it.
from xpcsjax.config.parameter_registry import AnalysisMode as AnalysisMode  # noqa: E402


def get_parameter_names(analysis_mode: AnalysisMode) -> list[str]:
    """Get parameter names for specified analysis mode.

    Parameters
    ----------
    analysis_mode : str
        Analysis mode: 'static_isotropic' or 'laminar_flow'

    Returns
    -------
    list of str
        Ordered list of parameter names

    Examples
    --------
    >>> get_parameter_names('static_isotropic')
    ['contrast', 'offset', 'D0', 'alpha', 'D_offset']

    >>> get_parameter_names('laminar_flow')
    ['contrast', 'offset', 'D0', 'alpha', 'D_offset',
     'gamma_dot_t0', 'beta', 'gamma_dot_t_offset', 'phi0']
    """
    if "static" in analysis_mode.lower():
        return STATIC_ISOTROPIC_PARAMS.copy()
    elif "laminar" in analysis_mode.lower() or "flow" in analysis_mode.lower():
        return LAMINAR_FLOW_PARAMS.copy()
    else:
        raise ValueError(
            f"Unknown analysis mode: {analysis_mode}. Expected 'static_isotropic' or 'laminar_flow'"
        )


def get_physical_param_names(analysis_mode: AnalysisMode) -> list[str]:
    """Get physical parameter names only (without scaling params).

    Unlike get_parameter_names() which includes contrast/offset, this
    returns only the physical model parameters for result formatting.

    Parameters
    ----------
    analysis_mode : str
        Analysis mode: 'static_anisotropic', 'static_isotropic', or 'laminar_flow'

    Returns
    -------
    list of str
        Ordered list of physical parameter names (without contrast/offset)

    Examples
    --------
    >>> get_physical_param_names('static_anisotropic')
    ['D0', 'alpha', 'D_offset']

    >>> get_physical_param_names('laminar_flow')
    ['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta', 'gamma_dot_t_offset', 'phi0']
    """
    if "static" in analysis_mode.lower():
        return STATIC_PHYSICAL_PARAMS.copy()
    elif "laminar" in analysis_mode.lower() or "flow" in analysis_mode.lower():
        return (STATIC_PHYSICAL_PARAMS + FLOW_PARAMS).copy()
    else:
        raise ValueError(
            f"Unknown analysis mode: {analysis_mode}. "
            f"Expected 'static_anisotropic', 'static_isotropic', or 'laminar_flow'"
        )


def get_num_parameters(analysis_mode: AnalysisMode) -> int:
    """Get number of parameters for analysis mode.

    Parameters
    ----------
    analysis_mode : str
        Analysis mode: 'static_isotropic' or 'laminar_flow'

    Returns
    -------
    int
        Number of parameters

    Examples
    --------
    >>> get_num_parameters('static_isotropic')
    5

    >>> get_num_parameters('laminar_flow')
    9
    """
    return len(get_parameter_names(analysis_mode))


def validate_parameter_names(
    param_names: list[str], analysis_mode: AnalysisMode, strict: bool = True
) -> None:
    """Validate parameter names against expected names for analysis mode.

    Parameters
    ----------
    param_names : list of str
        Parameter names to validate
    analysis_mode : str
        Analysis mode: 'static_isotropic' or 'laminar_flow'
    strict : bool, default=True
        If True, require exact match of names and order
        If False, only check that all expected names are present

    Raises
    ------
    ValueError
        If parameter names don't match expected names

    Examples
    --------
    >>> validate_parameter_names(
    ...     ['contrast', 'offset', 'D0', 'alpha', 'D_offset'],
    ...     'static_isotropic'
    ... )  # No error

    >>> validate_parameter_names(
    ...     ['D0', 'alpha'],  # Missing parameters
    ...     'static_isotropic'
    ... )
    ValueError: Missing parameters: ['contrast', 'offset', 'D_offset']
    """
    expected = get_parameter_names(analysis_mode)

    if strict:
        # Exact match required
        if param_names != expected:
            raise ValueError(
                f"Parameter names don't match expected order for {analysis_mode}.\n"
                f"Expected: {expected}\n"
                f"Got: {param_names}"
            )
    else:
        # Check all expected names are present
        missing = set(expected) - set(param_names)
        if missing:
            raise ValueError(
                f"Missing parameters for {analysis_mode}: {sorted(missing)}\n"
                f"Expected: {expected}\n"
                f"Got: {param_names}"
            )

        extra = set(param_names) - set(expected)
        if extra:
            raise ValueError(
                f"Unexpected parameters for {analysis_mode}: {sorted(extra)}\n"
                f"Expected: {expected}\n"
                f"Got: {param_names}"
            )


def get_parameter_description(param_name: str) -> str:
    """Get description for parameter.

    Parameters
    ----------
    param_name : str
        Parameter name

    Returns
    -------
    str
        Parameter description

    Examples
    --------
    >>> get_parameter_description('D0')
    'Diffusion coefficient amplitude [Å²/s]'
    """
    if param_name not in PARAMETER_DESCRIPTIONS:
        raise ValueError(
            f"Unknown parameter: {param_name}. "
            f"Valid parameters: {list(PARAMETER_DESCRIPTIONS.keys())}"
        )
    return PARAMETER_DESCRIPTIONS[param_name]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def verify_samples_dict(samples_dict: dict, analysis_mode: AnalysisMode) -> None:
    """Verify parameter dictionary contains all expected parameters.

    This function validates that a parameter dictionary has all required
    parameter names for the given analysis mode.

    Parameters
    ----------
    samples_dict : dict
        Dictionary of parameter values (parameter_name -> values array)
    analysis_mode : str
        Analysis mode: 'static_isotropic' or 'laminar_flow'

    Raises
    ------
    KeyError
        If any expected parameters are missing from samples

    Examples
    --------
    >>> samples = {'contrast': [...], 'offset': [...], ...}
    >>> verify_samples_dict(samples, 'laminar_flow')
    """
    expected = get_parameter_names(analysis_mode)
    missing = [p for p in expected if p not in samples_dict]

    if missing:
        available = list(samples_dict.keys())
        raise KeyError(
            f"Missing parameters in parameter dictionary for {analysis_mode}:\n"
            f"Missing: {missing}\n"
            f"Expected: {expected}\n"
            f"Available: {available}\n\n"
            f"This indicates a parameter name mismatch between model "
            f"definition and sample extraction. Check that the sampler model "
            f"uses the same parameter names as defined in parameter_names.py"
        )


__all__ = [
    # Constants
    "SCALING_PARAMS",
    "STATIC_PHYSICAL_PARAMS",
    "FLOW_PARAMS",
    "STATIC_ISOTROPIC_PARAMS",
    "LAMINAR_FLOW_PARAMS",
    "NUM_PARAMS_STATIC",
    "NUM_PARAMS_LAMINAR",
    "PARAMETER_DESCRIPTIONS",
    # Functions
    "get_parameter_names",
    "get_physical_param_names",
    "get_num_parameters",
    "validate_parameter_names",
    "get_parameter_description",
    "verify_samples_dict",
]
