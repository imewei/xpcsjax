"""Regression test for integer-dtype truncation in enhanced diagonal correction.

Bug: ``_correct_diagonal_enhanced`` builds its working buffer with
``np.array(v)`` (preserving the source dtype) and then assigns the *float*
results of ``apply_diagonal_correction`` back into ``corrected_data["c2_exp"][i]``.
If the source ``c2_exp`` is an integer dtype, that item-assignment silently
truncates the fractional corrected diagonal — a silent data-loss / integrity
violation. ``_normalize_data`` already guards this with an int->float64 upcast;
``_correct_diagonal_enhanced`` was the inconsistent sibling.
"""

import numpy as np

from xpcsjax.data.preprocessing import PreprocessingPipeline


def _integer_c2_with_fractional_correction():
    """One 8x8 integer matrix whose basic diagonal correction is fractional.

    The basic correction replaces each diagonal element with the mean of its
    first off-diagonal neighbours. With an alternating 1/2 super-diagonal those
    means are 1.5 — non-integer, so an integer buffer would truncate them to 1.
    """
    n = 8
    base = np.full((n, n), 2, dtype=np.int64)
    # Alternating first off-diagonal: 1,2,1,2,... (kept symmetric)
    for k in range(n - 1):
        v = 1 if k % 2 == 0 else 2
        base[k, k + 1] = v
        base[k + 1, k] = v
    return base


def test_diagonal_correction_upcasts_integer_c2_no_truncation():
    pipeline = PreprocessingPipeline({})
    matrix = _integer_c2_with_fractional_correction()
    data = {"c2_exp": np.stack([matrix])}  # shape (1, 8, 8), int dtype

    corrected = pipeline._correct_diagonal_enhanced(data, {"method": "basic"})

    c2 = corrected["c2_exp"]
    # The buffer must be upcast so float corrections survive.
    assert np.issubdtype(np.asarray(c2).dtype, np.floating), (
        f"c2_exp not upcast to float: dtype={np.asarray(c2).dtype}"
    )
    # At least one corrected diagonal value must retain a fractional part
    # (proving no integer truncation occurred).
    diag = np.diag(np.asarray(c2)[0])
    frac = np.abs(diag - np.round(diag))
    assert np.max(frac) > 1e-9, f"corrected diagonal was truncated to integers: diag={diag}"
