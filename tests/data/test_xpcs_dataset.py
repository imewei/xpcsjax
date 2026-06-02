"""Typed XpcsDataset at the load/fit boundary.

Quality-gate type-design finding: ``fit_nlsq``/``load_xpcs_data`` exchanged a
bare ``dict[str, Any]``, so wrong/missing keys surfaced as errors deep in the
fit. ``XpcsDataset`` gives the boundary a named schema and typed accessors
(``.c2`` / ``.phi``) while remaining a ``dict`` subclass, so existing
key-indexed access (``data["c2_exp"]``) keeps working — the type is additive,
not a breaking change.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.data import XpcsDataset


def _raw():
    return {
        "c2_exp": np.ones((2, 5, 5)),
        "phi_angles_list": np.array([0.0, 90.0]),
        "t1": np.arange(5.0),
        "t2": np.arange(5.0),
    }


def test_is_dict_subclass_backward_compatible():
    ds = XpcsDataset(_raw())
    assert isinstance(ds, dict)
    # Existing key-indexed access still works unchanged.
    assert ds["c2_exp"].shape == (2, 5, 5)


def test_typed_accessors_resolve_canonical_and_alias_keys():
    ds = XpcsDataset(_raw())
    assert ds.c2.shape == (2, 5, 5)
    assert np.array_equal(ds.phi, np.array([0.0, 90.0]))
    # alias: ``c2`` and ``phi`` keys are accepted too
    ds2 = XpcsDataset({"c2": np.zeros((1, 3, 3)), "phi": np.array([0.0])})
    assert ds2.c2.shape == (1, 3, 3)
    assert ds2.phi.shape == (1,)


def test_missing_correlation_raises_clear_error():
    ds = XpcsDataset({"phi": np.array([0.0])})
    with pytest.raises(KeyError, match="correlation"):
        _ = ds.c2
