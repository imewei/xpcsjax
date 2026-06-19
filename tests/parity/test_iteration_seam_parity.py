"""on_iteration=None must be byte-identical to the legacy call (parity).

Pins the structural invariant that passing ``on_iteration=None`` explicitly
produces bit-identical results to omitting it entirely.  This guards the E2.2
observer seam: ``_build_homodyne_l4_callback``'s ``on_iteration is None`` branch
must return today's exact callback object unchanged, so the engine trajectory
never diverges.
"""

import numpy as np

from xpcsjax import fit_nlsq


def test_on_iteration_none_is_byte_identical(tiny_laminar_config_and_data):
    config, data = tiny_laminar_config_and_data
    legacy = fit_nlsq(data, config)
    explicit_none = fit_nlsq(data, config, on_iteration=None)
    np.testing.assert_array_equal(
        np.asarray(legacy.parameters), np.asarray(explicit_none.parameters)
    )
    assert legacy.chi_squared == explicit_none.chi_squared
    assert legacy.iterations == explicit_none.iterations
