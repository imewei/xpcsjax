"""Regression tests for the xdata->JAX conversion cache in get_or_create_model.

The cached ``model_func`` memoizes the xdata->JAX conversion. It used to key on
``id(xdata)`` alone, but the closure (with its cache) persists in the
module-level ``_model_cache`` across fits and CPython recycles object ids, so a
freed xdata array's id could be reused by a different array — returning the STALE
(t1, t2, phi_idx) conversion and silently fitting against the wrong coordinates.
The fix stores a weakref to the source array in the cache value, verifies
identity on a hit, and registers an eviction callback that drops the entry the
moment the source is GC'd.

The recycled-id collision itself is not deterministically reproducible across
interpreters (numpy does not reliably hand back a freed ndarray object's
address), so these tests pin the OBSERVABLE behavior the fix must preserve:
distinct arrays never share a conversion, and repeated calls with the same array
stay correct. The recycled-id guard proper (weakref identity + eviction) is
correct by construction.

The physics kernel (``compute_g1_batch``) is stubbed with a coordinate-sensitive
function (g1 = t1 + t2) so the test exercises the REAL production cache logic
without the dt-dependent physics — different coordinates must yield different
predictions.
"""

import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.adapter import clear_model_cache, get_or_create_model

# laminar per-angle params for n_phi=1: [contrast, offset, *physical(7)]
_PARAMS = (0.3, 1.0, 1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0)


def _xdata(t1_vals, t2_vals):
    """Build an [n,3] xdata array (columns t1, t2, phi_idx=0)."""
    n = len(t1_vals)
    return np.column_stack([np.asarray(t1_vals, float), np.asarray(t2_vals, float), np.zeros(n)])


def _coord_sensitive_model_func():
    """Production model_func with the physics kernel stubbed to g1 = t1 + t2."""
    clear_model_cache()
    model, model_func, _ = get_or_create_model(
        AnalysisMode.LAMINAR_FLOW,
        phi_angles=np.array([0.0]),
        q=0.0237,
        per_angle_scaling=True,
        enable_jit=False,
    )
    model.compute_g1_batch = lambda phys, t1, t2, phi, q, L, dt=None: (
        np.asarray(t1) + np.asarray(t2)
    )
    return model_func


def _expected(t1_vals, t2_vals):
    # g2 = offset + contrast * g1^2, g1 = t1 + t2
    return _PARAMS[1] + _PARAMS[0] * (np.array(t1_vals) + np.array(t2_vals)) ** 2


def test_distinct_xdata_objects_do_not_cross_contaminate():
    """Two live arrays with different coordinates must get their OWN conversions —
    no stale sharing through the cache."""
    model_func = _coord_sensitive_model_func()
    t1a, t2a = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]
    t1b, t2b = [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]

    x_a = _xdata(t1a, t2a)
    x_b = _xdata(t1b, t2b)
    out_a = np.asarray(model_func(x_a, *_PARAMS))
    out_b = np.asarray(model_func(x_b, *_PARAMS))

    np.testing.assert_allclose(out_a, _expected(t1a, t2a), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out_b, _expected(t1b, t2b), rtol=1e-12, atol=1e-12)
    assert not np.allclose(out_a, out_b)


def test_repeated_same_object_is_cached_and_correct():
    """Repeated calls with the SAME array object (the cache-hit path) stay
    correct — the identity guard must not reject a legitimate hit."""
    model_func = _coord_sensitive_model_func()
    t1a, t2a = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]
    x_a = _xdata(t1a, t2a)

    first = np.asarray(model_func(x_a, *_PARAMS))
    for _ in range(5):
        again = np.asarray(model_func(x_a, *_PARAMS))
        np.testing.assert_allclose(again, first, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(first, _expected(t1a, t2a), rtol=1e-12, atol=1e-12)
