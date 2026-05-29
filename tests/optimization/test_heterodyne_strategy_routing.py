"""Routing tests for the heterodyne standard-tier stratification gate.

The gate (in ``_fit_nlsq_heterodyne``) mirrors homodyne: it *decides* on
stratification above 100k points but only *engages* the stratified-LS solver at
>= 1M points. These tests pin the >=1M boundary by patching the solver and the
point estimator so no large array is ever allocated.
"""

import numpy as np  # noqa: F401  (kept for parity / future array fixtures)


def test_sub_1M_does_not_stratify(monkeypatch):  # noqa: N802 - "1M" pins the >=1M point boundary
    import xpcsjax.optimization.nlsq as nlsq_pkg
    import xpcsjax.optimization.nlsq.heterodyne_stratified_ls as hsl

    called = {"strat": False}
    monkeypatch.setattr(
        hsl,
        "fit_heterodyne_stratified_least_squares",
        lambda **k: called.__setitem__("strat", True),
    )
    from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data

    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=20)  # ~1.2k points
    nlsq_pkg.fit_nlsq(data, cfg)
    assert called["strat"] is False


def test_ge_1M_stratifies(monkeypatch):  # noqa: N802 - "1M" pins the >=1M point boundary
    import xpcsjax.optimization.nlsq as nlsq_pkg
    import xpcsjax.optimization.nlsq.heterodyne_stratified_ls as hsl

    sentinel = object()
    called = {"strat": False}

    def _fake(**k):
        called["strat"] = True
        return sentinel

    monkeypatch.setattr(hsl, "fit_heterodyne_stratified_least_squares", _fake)
    # Force point count over 1M without allocating a huge array:
    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq._estimate_heterodyne_points",
        lambda c2, phi: 2_000_000,
    )
    from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data

    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=20)
    result = nlsq_pkg.fit_nlsq(data, cfg)
    assert called["strat"] is True
    assert result is sentinel
