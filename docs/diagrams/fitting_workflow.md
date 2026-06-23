# xpcsjax Fitting Workflow

The NLSQ fitting pipeline, derived from the graphify knowledge graph hyperedges
(`Two-Function Fit Pipeline`, `Heterodyne Memory Tier Dispatch`, `Memory routing
strategy tiers`, `Five-Layer Anti-Degeneracy Stack`) in
`graphify-out/GRAPH_REPORT.md`.

A fit enters through `fit_nlsq`, is routed by memory budget to one of several
strategy paths, and every in-scope path runs the 5-layer anti-degeneracy defense
before producing an `OptimizationResult`.

```mermaid
graph TB
    START(["Raw C2 data + mode config"]) --> LOAD["load_xpcs_data"]
    LOAD --> CFG["ConfigManager: resolve analysis_mode + bounds"]
    CFG --> FIT["fit_nlsq"]
    FIT --> FMP["fit_nlsq_multi_phi: resolve effective per-angle mode"]
    FMP --> SEL{"select_nlsq_strategy<br/>memory budget?"}

    SEL -->|"global escape enabled"| CMAES["CMA-ES escape (seed-pinned, keep-better)"]
    SEL -->|"multistart enabled"| MULTI["LHS multistart"]
    SEL -->|"very large, streaming"| HYB["hybrid-streaming"]
    SEL -->|">= 1M points"| STRAT["stratified-LS (double-chunking)"]
    SEL -->|"fits in memory"| INMEM["in-memory engine route (StratifiedResidualFunctionJIT)"]

    CMAES --> AD
    MULTI --> AD
    HYB --> AD
    STRAT --> AD
    INMEM --> AD

    subgraph AD["Anti-Degeneracy Defense (per iteration)"]
        L1["L1 Per-Angle Reparameterization (all modes)"]
        L2["L2 Hierarchical Optimization (all modes)"]
        L3["L3 Adaptive CV Regularization (all modes)"]
        L4["L4 Gradient-Collapse Monitor (diagnostic)"]
        L5["L5 Shear-Sensitivity Weighting (laminar_flow only)"]
        L1 --> L2 --> L3 --> L4 --> L5
    end

    AD --> SOLVE["NLSQ CurveFit: trust-region least squares"]
    SOLVE --> CONV{"converged or<br/>keep-better floor?"}
    CONV -->|"no, refine"| SOLVE
    CONV -->|"yes"| OR["OptimizationResult (params, chi2, uncertainties, diagnostics)"]
    OR --> VIZ["generate_nlsq_plots / GUI"]

    classDef io fill:#d3f9d8,stroke:#2f9e44,color:#13501f;
    classDef route fill:#ffe3e3,stroke:#c92a2a,color:#5c1212;
    classDef path fill:#ffe8cc,stroke:#d9480f,color:#5c2208;
    classDef layer fill:#f3d9fa,stroke:#862e9c,color:#3d1349;
    classDef solve fill:#c5f6fa,stroke:#0c8599,color:#063d45;
    class START,LOAD,CFG,OR,VIZ io;
    class SEL,CONV route;
    class CMAES,MULTI,HYB,STRAT,INMEM path;
    class L1,L2,L3,L4,L5 layer;
    class FIT,FMP,SOLVE solve;
```

## Notes

- **Dispatch order** inside the strategy router is: CMA-ES escape -> multistart ->
  hybrid-streaming -> stratified-LS (activates at >= 1M points) -> in-memory
  engine route. The first applicable path wins.
- **Anti-degeneracy layers L1-L4** are active for all analysis modes. **L5
  (shear-sensitivity weighting)** is `laminar_flow`-only by design — the static
  modes have no flow direction and `two_component` (heterodyne) has no shear rate,
  so L5 short-circuits there.
- **L4 is strictly diagnostic**: monitor-on vs monitor-off is bit-identical
  (the homodyne rtol=1e-10 parity baselines included).
- **Global escapes** (CMA-ES / multistart) keep-better vs the plain NLSQ joint fit
  and fall back to it on failure; an escape result is tagged
  `nlsq_diagnostics["global_escape"]` and carries NaN covariance by construction.
- **Engine route**: the in-memory in-scope-mode path routes through the shared
  `StratifiedResidualFunctionJIT` engine for procedural parity between homodyne
  and heterodyne (no-worse SSR, not bit-identical).
