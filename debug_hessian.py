import numpy as np
import jax
import jax.numpy as jnp
import logging
logging.basicConfig(level=logging.WARNING)

from tests.optimization.test_heterodyne_hybrid_streaming import _make_synthetic_heterodyne
from xpcsjax.optimization.nlsq.heterodyne_stratified_data import build_heterodyne_stratified_data
from xpcsjax.optimization.nlsq.strategies import heterodyne_hybrid_streaming as hs

model, c2, phi = _make_synthetic_heterodyne(n_phi=4, n_t=6)
rng = np.random.default_rng(seed=20260821)
c2_noisy = c2 + rng.normal(0.0, 1e-3, size=c2.shape)

strat = build_heterodyne_stratified_data(model, c2_noisy, phi, weights=None)
lo, hi = model.param_manager.get_bounds()

popt, pcov, info = hs.fit_with_stratified_hybrid_streaming_heterodyne(
    stratified_data=strat,
    model=model,
    physical_param_names=list(model.param_manager.varying_names),
    initial_params=np.asarray(model.param_manager.get_initial_values(), dtype=np.float64),
    bounds=(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)),
    hybrid_config={
        "warmup_iterations": 5,
        "max_warmup_iterations": 10,
        "gauss_newton_max_iterations": 5,
        "verbose": 0,
    },
    anti_degeneracy_config={
        "per_angle_mode": "individual",
        "hierarchical": {"enable": True, "max_outer_iterations": 2},
    },
)
print("popt=", popt)
print("pcov placeholder=", info["covariance_is_placeholder"])
print("pcov[:3,:3]=", pcov[:3, :3])
