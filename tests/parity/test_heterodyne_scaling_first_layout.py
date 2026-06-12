# tests/parity/test_heterodyne_scaling_first_layout.py
"""Phase 1+2 — heterodyne scaling-first joint layout helpers."""
from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.heterodyne_core import (
    _joint_param_names_scaling_first,
)


def test_individual_names_scaling_first():
    names = _joint_param_names_scaling_first(
        mode="individual", physics_names=["D0", "alpha"], n_phi=3
    )
    # scaling head [c0,c1,c2,o0,o1,o2] then physics tail [D0, alpha]
    assert names == [
        "contrast_0", "contrast_1", "contrast_2",
        "offset_0", "offset_1", "offset_2",
        "D0", "alpha",
    ]


def test_averaged_names_scaling_first():
    names = _joint_param_names_scaling_first(
        mode="averaged", physics_names=["D0", "alpha"], n_phi=5
    )
    assert names == ["contrast_avg", "offset_avg", "D0", "alpha"]


def test_constant_names_physics_only():
    names = _joint_param_names_scaling_first(
        mode="constant", physics_names=["D0", "alpha"], n_phi=4
    )
    assert names == ["D0", "alpha"]


def test_rejects_fourier():
    with pytest.raises(ValueError, match="unknown per_angle_mode"):
        _joint_param_names_scaling_first(
            mode="fourier", physics_names=["D0"], n_phi=4
        )


def test_split_individual_scaling_first():
    from xpcsjax.optimization.nlsq.heterodyne_core import (
        _split_scaling_first_joint,
    )
    # [c0,c1, o0,o1, D0,alpha,beta]  (n_phi=2, n_physics=3)
    x = np.array([0.1, 0.2, 1.0, 1.1, 5.0, 6.0, 7.0])
    physics, contrast, offset = _split_scaling_first_joint(
        x, mode="individual", n_phi=2, n_physics=3
    )
    np.testing.assert_array_equal(physics, [5.0, 6.0, 7.0])
    np.testing.assert_array_equal(contrast, [0.1, 0.2])
    np.testing.assert_array_equal(offset, [1.0, 1.1])


def test_split_averaged_broadcasts():
    from xpcsjax.optimization.nlsq.heterodyne_core import (
        _split_scaling_first_joint,
    )
    # [c_avg, o_avg, D0, alpha]  (n_phi=4, n_physics=2)
    x = np.array([0.3, 1.2, 5.0, 6.0])
    physics, contrast, offset = _split_scaling_first_joint(
        x, mode="averaged", n_phi=4, n_physics=2
    )
    np.testing.assert_array_equal(physics, [5.0, 6.0])
    np.testing.assert_array_equal(contrast, [0.3, 0.3, 0.3, 0.3])
    np.testing.assert_array_equal(offset, [1.2, 1.2, 1.2, 1.2])


def test_split_constant_uses_frozen():
    from xpcsjax.optimization.nlsq.heterodyne_core import (
        _split_scaling_first_joint,
    )
    # [D0, alpha] only; frozen scaling supplied
    x = np.array([5.0, 6.0])
    physics, contrast, offset = _split_scaling_first_joint(
        x, mode="constant", n_phi=3, n_physics=2,
        frozen_contrast=np.array([0.4, 0.5, 0.6]),
        frozen_offset=np.array([1.4, 1.5, 1.6]),
    )
    np.testing.assert_array_equal(physics, [5.0, 6.0])
    np.testing.assert_array_equal(contrast, [0.4, 0.5, 0.6])
    np.testing.assert_array_equal(offset, [1.4, 1.5, 1.6])


def test_build_joint_problem_x0_is_scaling_first():
    """x0 places scaling at the HEAD, physics at the TAIL, for individual mode."""
    import numpy as np

    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import _build_joint_problem

    # The stateful HeterodyneModel exposes t/q/dt as read-only properties — the
    # fixture configures them (and a self-consistent c2/phi); we use those
    # directly. The scaling-first layout assertions are the load-bearing part.
    n_phi = 4
    model, c2, phi = make_synthetic_two_component(n_phi=n_phi, n_t=12)
    cfg = NLSQConfig(per_angle_mode="individual")

    prob = _build_joint_problem(model, c2, phi, cfg, None)
    n_physics = model.param_manager.n_varying
    x0 = np.asarray(prob.x0, dtype=np.float64)
    # scaling head is 2*n_phi, physics tail is n_physics
    assert len(x0) == 2 * n_phi + n_physics
    assert prob.meta["scaling_first"] is True
    assert prob.meta["resolved_mode"] == "individual"
    # the joint names are scaling-first
    assert prob.meta["joint_param_names"][:1] == ["contrast_0"]
    assert prob.meta["joint_param_names"][-n_physics:] == list(
        model.param_manager.varying_names
    )


@pytest.mark.parametrize("mode", ["constant", "averaged", "individual"])
def test_result_builder_roundtrips_scaling_first(mode):
    """An escape-style scaling-first x_final reconstructs the right per-angle
    scaling AND the right physics for every mode (the §Risk-3 1e6-heatmap class)."""
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import (
        _build_joint_problem,
        _build_joint_result,
    )

    # Fixture builds a fully-configured model with self-consistent c2/phi; t/q/dt
    # are read-only properties we do not override.
    n_phi = 4
    model, c2, phi = make_synthetic_two_component(n_phi=n_phi, n_t=12)
    cfg = NLSQConfig(per_angle_mode=mode)
    prob = _build_joint_problem(model, c2, phi, cfg, None)

    # Construct a scaling-first x_final with KNOWN, distinguishable values.
    n_physics = model.param_manager.n_varying
    known_physics = model.param_manager.get_initial_values()
    if mode == "individual":
        head = np.concatenate([
            0.10 + 0.01 * np.arange(n_phi),      # contrast per angle
            1.20 + 0.01 * np.arange(n_phi),      # offset per angle
        ])
        x_final = np.concatenate([head, known_physics])
    elif mode == "averaged":
        x_final = np.concatenate([[0.33, 1.27], known_physics])
    else:  # constant: physics-only vector
        x_final = np.asarray(known_physics, dtype=np.float64)

    plan = prob.meta["plan"]
    result = _build_joint_result(
        model, prob, c2, x_final, phi, cfg, None,
    )
    # parameters surface is canonical scaling-first: physics live in the TAIL
    params = np.asarray(result.parameters, dtype=np.float64)
    np.testing.assert_allclose(params[-n_physics:], known_physics, rtol=0, atol=0)
    if mode == "individual":
        np.testing.assert_allclose(params[:n_phi], 0.10 + 0.01 * np.arange(n_phi))
        np.testing.assert_allclose(
            params[n_phi : 2 * n_phi], 1.20 + 0.01 * np.arange(n_phi)
        )
        # model.scaling reflects the SAME per-angle values (not transposed)
        np.testing.assert_allclose(model.scaling.contrast, 0.10 + 0.01 * np.arange(n_phi))
    elif mode == "averaged":
        # the scaling head carries the distinguishable [c_avg, o_avg] pair...
        np.testing.assert_allclose(params[:2], [0.33, 1.27])
        # ...broadcast onto every per-angle scaling slot (not transposed).
        np.testing.assert_allclose(model.scaling.contrast, np.full(n_phi, 0.33))
        np.testing.assert_allclose(model.scaling.offset, np.full(n_phi, 1.27))
    else:  # constant: scaling is frozen from the plan's quantile estimate
        np.testing.assert_allclose(model.scaling.contrast, plan.frozen_contrast)
        np.testing.assert_allclose(model.scaling.offset, plan.frozen_offset)
        # physics-only vector: the whole parameter surface IS the physics tail
        np.testing.assert_allclose(params, known_physics, rtol=0, atol=0)
