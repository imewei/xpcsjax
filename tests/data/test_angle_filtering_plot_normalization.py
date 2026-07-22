"""Regression test: apply_angle_filtering_for_plot must normalize phi_angles
and the config's phi_filtering.target_ranges the same way
apply_angle_filtering_for_optimization does, before calling the shared
apply_angle_filtering() core function.

Before the fix, the plot wrapper passed raw (unnormalized) angles/ranges
straight through. For a target range that crosses the +-180 boundary (e.g.
[170, 190] deg, which normalizes to the wrapped [170, -170]), the raw
comparison finds zero matches and silently falls back to "plot all angles" --
diverging from the optimization path, which normalizes first and correctly
selects the wrapped-range match.
"""

import numpy as np

from xpcsjax.data.angle_filtering import (
    apply_angle_filtering_for_optimization,
    apply_angle_filtering_for_plot,
)


def _config():
    return {
        "phi_filtering": {
            "enabled": True,
            "target_ranges": [{"min_angle": 170.0, "max_angle": 190.0}],
        }
    }


def test_plot_and_optimization_paths_select_same_angles_across_wrap():
    # 200 deg normalizes to -160 deg; only -170 deg falls inside the wrapped
    # [170, -170] range once both angles and range are normalized.
    phi_angles = np.array([200.0, 10.0, -170.0])
    c2_exp = np.zeros((3, 4, 4))
    config = _config()

    plot_indices, plot_phi, _plot_c2 = apply_angle_filtering_for_plot(
        phi_angles, c2_exp, {"config": config}
    )

    opt_result = apply_angle_filtering_for_optimization(
        {"phi_angles_list": phi_angles.copy(), "c2_exp": c2_exp.copy()}, config
    )

    assert plot_indices == [2], f"plot path did not select the wrapped-range match: {plot_indices}"
    assert np.allclose(plot_phi, [-170.0])
    assert np.allclose(opt_result["phi_angles_list"], [-170.0])
