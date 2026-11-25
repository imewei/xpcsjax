# codex closing check of PR #81
## Per-finding status
- c-F1 / a-F4 (engine route mask weights): FIXED — The engine rejects non-positive and non-finite weights before dropping uniform weights. The regression covers both an all-zero mask and a mix of zeros and tiny positive weights.
- a-F1 (zero-DOF crash): FIXED — Individual, averaged, and constant joint builders set quality to `unknown` when masking leaves the statistic undefined. The regression exercises all three joint modes.
- c-F2 / a-F2 (large-path zero DOF, placeholder flag): FIXED — Both large-data paths provide total and contributing row counts. The shared result builder returns NaN when masking exhausts reporting DOF. Stratified LS also guards both covariance sources and marks their NaN outputs as placeholders. The probe below returned NaN and `unknown` through the streaming result builder with 4 contributing rows out of 126.
- c-F3 (duplicate phi sigma): FIXED — Repeated phi measurements take sigma from their original flattened measurement rows before the support mask is applied. The regression's contributing count is 330.
- c-F4 / a-F3 (streaming SSR): FIXED — Both fitted and frozen-baseline SSR now exclude masked pixels. Retained pixels use the streaming path's raw SSR convention.
- c-F5 / a-F5 (Layer 3 off switch): FIXED for the explicit nested false override of the flat heterodyne mode, absence of L3 callback groups, and `mode: "none"` on both streaming paths. DECLINED-OK for laminar's explicitly enabled flat optimizer penalty, which is distinct from the adaptive Layer 3 control.
- c-F6 (NumPy frame counts): FIXED — Clamp metadata accepts NumPy integer requests and returns a Python integer. The original cache-disabled case reports the 20 dropped frames.
- c-F7 / a-F7 / a-F6 (tests, config=None, cache fallback): FIXED — The regression checks the actual escalation for both an incompatible object and `None`, and cache metadata uses the configured or derived window when applied bounds are unset.
## New issues introduced by the fixes
### F1: An explicit null now turns on Layer 3 in heterodyne hybrid streaming
- Must-fix: yes
- Location: xpcsjax/optimization/nlsq/strategies/hybrid_streaming_layers.py:56
- Scenario: Set `optimization.nlsq.anti_degeneracy` to use `per_angle_mode: individual`, `hierarchical.enable: false`, and `regularization: {enable: null, mode: relative}`, and take the two-component hybrid-streaming path. The previous heterodyne gate treated the supplied null as false. The new shared gate enables group-variance regularization, changing the objective for an existing accepted configuration. The in-package dispatcher passes the nested config through at xpcsjax/optimization/nlsq/__init__.py:904.
- Evidence: On `2a8879f40a65082413b84776a794375512645138`, this probe captures the actual optimizer configuration before the solve. It printed `optimizer L3 False` for false and `optimizer L3 True` for null, with group-variance lambda 10 and groups `[(0, 3), (3, 6)]` logged for null. The pre-fix predicate was `reg_cfg_dict.get("enable", True)` in a boolean condition.
  ```sh
  rtk uv run --no-sync python -B -c 'import numpy as np
  from unittest.mock import patch
  from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
  from xpcsjax.optimization.nlsq import heterodyne_stratified_data as data
  from xpcsjax.optimization.nlsq.strategies import heterodyne_hybrid_streaming as stream
  from xpcsjax.optimization.nlsq.heterodyne_result_builder import build_hybrid_streaming_result
  m = HeterodyneModel.from_config(dict(analyzer_parameters=dict(dt=0.1,start_frame=1,end_frame=8,scattering=dict(wavevector_q=0.01))))
  pm = m.param_manager
  phi = np.array([0.,45.,90.])
  c = np.ones((3,8,8))
  w = np.zeros_like(c)
  w[0,1,2:6] = 1
  sd = data.build_heterodyne_stratified_data(m,c,phi,w)
  class Capture:
      def __init__(self, cfg):
          self.enabled = cfg.enable_group_variance_regularization
      def fit(self, **kwargs):
          p = kwargs["p0"]
          return dict(x=p,pcov=np.eye(len(p)),success=True,optimizer_l3=self.enabled)
  for enable in (False,None):
      ad = dict(per_angle_mode="individual",hierarchical=dict(enable=False),gradient_monitoring=dict(enable=False),regularization=dict(enable=enable,mode="relative"))
      with patch.object(stream,"AdaptiveHybridStreamingOptimizer",Capture):
          p,cov,info = stream.fit_with_stratified_hybrid_streaming_heterodyne(stratified_data=sd,model=m,physical_param_names=list(pm.varying_names),initial_params=pm.get_initial_values(),bounds=pm.get_bounds(),anti_degeneracy_config=ad)
      r = build_hybrid_streaming_result(model=m,popt=p,pcov=cov,info=info,phi_angles=phi)
      print("enable",enable,"optimizer L3",info["optimizer_l3"],"rows",info["n_data_points"],info["n_data_rows"],"reduced chi2",r.reduced_chi_squared,"quality",r.quality_flag)'
  ```
- Suggested fix: Preserve the heterodyne streaming gate's treatment of supplied falsey values while retaining the new `mode: "none"` check and the default-on behavior when `enable` is absent.
## Verdict
Must-fix count: 1
END_OF_REPORT
