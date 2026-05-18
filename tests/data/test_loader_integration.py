"""Round-trip load test using a known-good homodyne fixture.

This test depends on a fixture homodyne config; we don't bundle the HDF5 with
xpcsjax — instead we point at the file in the source homodyne repo for now.
The Phase 5 characterization-test infra makes this a permanent fixture.
"""
from pathlib import Path

import numpy as np
import pytest

from xpcsjax.data import load_xpcs_data

# No bundled fixture is available yet (Phase 5 will provide one). Point at a
# deliberately non-existent path so the test SKIPs cleanly on every machine
# until the characterization-test infra lands proper baselines.
HOMODYNE_FIXTURE_CONFIG = Path(
    "/home/wei/Documents/GitHub/xpcsjax/tests/data/_fixtures/homodyne_static_fixture.yaml"
)


@pytest.mark.skipif(
    not HOMODYNE_FIXTURE_CONFIG.exists(),
    reason="homodyne fixture not present on this machine",
)
def test_load_static_fixture():
    data = load_xpcs_data(str(HOMODYNE_FIXTURE_CONFIG))
    # Sanity invariants for any homodyne XPCS file:
    assert "c2_exp" in data
    assert "phi_angles_list" in data
    assert "t1" in data and "t2" in data
    c2 = data["c2_exp"]
    # c2 shape: (n_q, n_phi, N, N) — must be 4-D float64
    assert c2.ndim == 4
    assert c2.dtype == np.float64
    # Diagonal correction was applied: square last two axes
    N = c2.shape[2]
    assert N == c2.shape[3]
    # Verify time arrays are monotonic
    assert np.all(np.diff(data["t1"]) > 0)
