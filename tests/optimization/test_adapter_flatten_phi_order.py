"""Regression test for phi-broadcast ordering in NLSQAdapter._flatten_xpcs_data.

When phi is supplied per-angle (``len(phi) != len(t1)``), the angle column was
rebuilt from ``np.unique(phi)`` (SORTED ascending) while g2/t1/t2 stay in their
incoming angle-major order. For an unsorted phi this mis-paired each time-block
with the wrong scattering angle. The fix broadcasts the INCOMING phi order, which
for already-sorted phi is identical to the previous behavior.
"""

import numpy as np

from xpcsjax.optimization.nlsq.adapter import NLSQAdapter


def _data_unsorted_phi():
    # 3 angles in NON-ascending order; each angle has a 2x2 time block whose g2
    # values encode the true angle, so misalignment is detectable.
    phi = np.array([90.0, 0.0, 45.0])
    n_t = 4  # points per angle (raveled 2x2)
    t1 = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), len(phi))
    t2 = np.tile(np.array([2.0, 3.0, 4.0, 5.0]), len(phi))
    # g2 block i carries its incoming angle value as a tag.
    g2 = np.concatenate([np.full(n_t, ang) for ang in phi])
    return {"phi": phi, "t1": t1, "t2": t2, "g2": g2}


def test_flatten_preserves_incoming_phi_block_order():
    adapter = NLSQAdapter()
    xdata, ydata, n_phi = adapter._flatten_xpcs_data(_data_unsorted_phi())

    assert n_phi == 3
    phi_unique = np.array([0.0, 45.0, 90.0])  # sorted unique
    phi_idx = xdata[:, 2].astype(int)

    # Each row's resolved angle (phi_unique[idx]) must equal the g2 tag for that
    # row (the incoming angle of its block). The pre-fix sorted broadcast paired
    # block 0 (g2=90) with phi_unique[0]=0 -> mismatch.
    resolved_angle = phi_unique[phi_idx]
    np.testing.assert_array_equal(resolved_angle, ydata)


def test_flatten_sorted_phi_unchanged():
    # Sorted phi: incoming order == unique order, so behavior is unchanged.
    phi = np.array([0.0, 45.0, 90.0])
    t1 = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), 3)
    t2 = np.tile(np.array([2.0, 3.0, 4.0, 5.0]), 3)
    g2 = np.concatenate([np.full(4, ang) for ang in phi])
    adapter = NLSQAdapter()
    xdata, ydata, _ = adapter._flatten_xpcs_data({"phi": phi, "t1": t1, "t2": t2, "g2": g2})
    resolved = np.array([0.0, 45.0, 90.0])[xdata[:, 2].astype(int)]
    np.testing.assert_array_equal(resolved, ydata)


def test_flatten_duplicate_phi_does_not_crash():
    """Duplicate angles in phi must not crash the broadcast (agy review, P2).

    ``n_phi`` counts UNIQUE angles, but g2 carries one block per phi ENTRY. The
    fix's ``np.repeat(np.asarray(phi), len(t1)//n_phi)`` over-counted blocks when
    ``len(phi) > n_phi`` (duplicate angles) and produced an over-long phi column
    -> ``ValueError`` on the column stack. The block count must key on
    ``len(phi)`` so duplicate angles map to their shared unique index instead.
    """
    phi = np.array([10.0, 10.0, 20.0])  # angle 10 appears twice -> 2 unique
    n_t = 4
    t1 = np.tile(np.arange(n_t, dtype=float), len(phi))
    t2 = t1.copy()
    # Tag each block with its incoming angle so we can verify the mapping.
    g2 = np.concatenate([np.full(n_t, ang) for ang in phi])
    adapter = NLSQAdapter()
    xdata, ydata, n_phi = adapter._flatten_xpcs_data(
        {"phi": phi, "t1": t1, "t2": t2, "g2": g2}
    )
    assert n_phi == 2  # only two distinct angles
    assert xdata.shape == (len(t1), 3)
    # Each row resolves to its block's incoming angle; the two angle-10 blocks
    # share phi_idx 0, the angle-20 block is phi_idx 1.
    phi_unique = np.array([10.0, 20.0])
    resolved = phi_unique[xdata[:, 2].astype(int)]
    np.testing.assert_array_equal(resolved, ydata)


def test_flatten_non_rectangular_raises():
    # 13 time points cannot split evenly across 3 angles -> clear error.
    phi = np.array([0.0, 45.0, 90.0])
    t1 = np.arange(13, dtype=float)
    data = {"phi": phi, "t1": t1, "t2": t1.copy(), "g2": t1.copy()}
    adapter = NLSQAdapter()
    try:
        adapter._flatten_xpcs_data(data)
    except ValueError as e:
        assert "per angle" in str(e) or "rectangular" in str(e)
    else:
        raise AssertionError("expected ValueError for non-rectangular layout")
