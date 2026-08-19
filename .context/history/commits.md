# Commit Decision History

> 此文件是 `commits.jsonl` 的人类可读视图，可由工具重生成。
> Canonical store: `commits.jsonl` (JSONL, append-only)

| Date | Context-Id | Commit | Summary | Decisions | Bugs | Risk |
|------|-----------|--------|---------|-----------|------|------|
| 2026-08-18 | 0355965b | 31653dc | docs(diagrams): sync architecture + fitting-workflow diagrams with current impl | added `service` seam to architecture.md; fixed L2 gate label in fitting_workflow.md; graphify rebuild refused by shrink-guard, left untouched | — | low |
| 2026-08-18 | 490661dc | — | docs(diagrams): add print-ready fitting workflow figure (PNG/PDF) | rebuilt mermaid as narrow vertical pipeline (invisible rank links + pinned `direction LR`) to fit a 3.375in single journal column at 300dpi; moved anti-degeneracy caption out of subgraph title to stop border/arrow overlap | landscape figure didn't fit single column; subgraph title overlapped box border at large print font | low |
| 2026-08-19 | f5429907 | — | fix(optimization): skip heterodyne baseline solve that would OOM | reused existing `select_nlsq_strategy()` with the TRUE joint param count (n_scaling+n_physics) instead of a new heuristic; skip only when the L2 hierarchical escape is configured to pick up afterward; reused `build_failed_result()` so the skip is byte-identical in shape to the existing OOM path | `per_angle_mode=individual` at >=1M points (60 joint params x 23M points) tried to allocate a 114GB dense Jacobian against a 41.6GB budget; no gate checked the true joint param count before calling `adapter.fit()` | low |
