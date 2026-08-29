# xpcsjax Architecture

Layered architecture of the xpcsjax package, derived from the graphify knowledge
graph (`graphify-out/GRAPH_REPORT.md`): the named community hubs and the top god
nodes (`AnalysisMode`, `OptimizationResult`, `ConfigManager`, `NLSQConfig`,
`fit_nlsq_multi_phi`, `ParameterManager`, `StratifiedResidualFunctionJIT`).

Data flows top-to-bottom: configuration and raw data enter at the top, the core
physics models and the NLSQ engine sit in the middle, and results / visualization /
GUI consume the fit at the bottom.

```mermaid
graph TB
    subgraph api["Public API and Configuration"]
        CM["ConfigManager"]
        AM["AnalysisMode (4 modes)"]
        PREG["parameter_registry (single source of truth)"]
        PM["ParameterManager (bounds, per-angle expansion)"]
        NCFG["NLSQConfig"]
        CM --> AM
        CM --> PREG
        PREG --> PM
        CM --> NCFG
    end

    subgraph data["Data Layer"]
        LOAD["load_xpcs_data"]
        DQC["DataQualityController"]
        MEM["AdvancedMemoryManager"]
        LOAD --> DQC
        LOAD --> MEM
    end

    subgraph core["Core Physics Models (JAX kernels)"]
        HOM["HomodyneModel (static_* / laminar_flow)"]
        HET["HeterodyneModel (two_component, 14 params)"]
        KG2["compute_g2_scaled (homodyne g1/g2)"]
        KC2["compute_c2_heterodyne (reference + sample)"]
        HOM --> KG2
        HET --> KC2
    end

    subgraph engine["NLSQ Optimization Engine"]
        FIT["fit_nlsq (single entry)"]
        FMP["fit_nlsq_multi_phi"]
        SEL["select_nlsq_strategy (memory routing)"]
        ADC["AntiDegeneracyController (L1-L5)"]
        ADAPT["NLSQ adapters (CurveFit, trust-region)"]
        SRJ["StratifiedResidualFunctionJIT (engine route)"]
        FIT --> FMP
        FMP --> SEL
        SEL --> ADC
        ADC --> ADAPT
        ADC --> SRJ
    end

    subgraph svc["service (headless orchestration seam)"]
        SFIT["service.fit"]
        SDATA["service.data"]
        SCFG["service.config"]
        SPLOTS["service.plots"]
        SPERSIST["service.persist"]
        SEVT["service.events"]
    end

    subgraph front["Front Ends"]
        CLI["cli (argparse)"]
        WORKER["gui.ipc.worker (subprocess, JAX-free MainWindow)"]
    end

    subgraph out["Results, Visualization, GUI"]
        OR["OptimizationResult"]
        VIZ["generate_nlsq_plots"]
        GUI["MainWindow (PySide6 workbench)"]
        OR --> VIZ
        OR --> GUI
    end

    api --> engine
    data --> engine
    AM --> core
    core --> engine
    engine --> OR

    CLI --> svc
    WORKER --> svc
    svc --> SFIT
    SFIT --> FIT
    SDATA --> LOAD
    SCFG --> CM
    SPLOTS --> VIZ
    GUI -.->|IPC| WORKER

    classDef cfg fill:#e7f5ff,stroke:#1971c2,color:#0b3d66;
    classDef dat fill:#fff4e6,stroke:#e67700,color:#663d00;
    classDef phys fill:#e5dbff,stroke:#5f3dc4,color:#2d1a66;
    classDef eng fill:#c5f6fa,stroke:#0c8599,color:#063d45;
    classDef res fill:#d3f9d8,stroke:#2f9e44,color:#13501f;
    classDef svcCls fill:#fff9db,stroke:#f08c00,color:#663d00;
    class CM,AM,PREG,PM,NCFG cfg;
    class LOAD,DQC,MEM dat;
    class HOM,HET,KG2,KC2 phys;
    class FIT,FMP,SEL,ADC,ADAPT,SRJ eng;
    class OR,VIZ,GUI res;
    class SFIT,SDATA,SCFG,SPLOTS,SPERSIST,SEVT,CLI,WORKER svcCls;
```

## Notes

- **`ConfigManager` + `AnalysisMode`** drive config-based dispatch to the right
  physics model. The four modes are `static_anisotropic`, `static_isotropic`,
  `laminar_flow` (homodyne) and `two_component` (heterodyne).
- **`parameter_registry`** is the single source of truth for parameter names,
  bounds and physical constraints; both `ConfigManager` and the NLSQ bounds
  builder read from it via `ParameterManager`.
- **The engine owns strategy; the upstream NLSQ library owns the solve.**
  xpcsjax routes memory (`select_nlsq_strategy`) and runs the anti-degeneracy
  controller; NLSQ's `CurveFit` performs the trust-region least-squares solve.
- **`OptimizationResult`** is the cross-cutting bridge node (182 edges) consumed
  by visualization and the GUI workbench.
- **`service`** is the headless, argparse/Qt-free orchestration seam shared by
  the CLI and the GUI worker subprocess — both front ends call into
  `service.fit` / `service.data` / `service.config` / `service.plots` rather
  than the optimization/data/config/viz layers directly. `MainWindow` never
  touches JAX; it talks to `gui.ipc.worker` (a separate subprocess) over IPC,
  and the worker is the one that calls into `service`.
