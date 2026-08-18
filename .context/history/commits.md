# Commit Decision History

> 此文件是 `commits.jsonl` 的人类可读视图，可由工具重生成。
> Canonical store: `commits.jsonl` (JSONL, append-only)

| Date | Context-Id | Commit | Summary | Decisions | Bugs | Risk |
|------|-----------|--------|---------|-----------|------|------|
| 2026-08-18 | 0355965b | 31653dc | docs(diagrams): sync architecture + fitting-workflow diagrams with current impl | added `service` seam to architecture.md; fixed L2 gate label in fitting_workflow.md; graphify rebuild refused by shrink-guard, left untouched | — | low |
| 2026-08-18 | 490661dc | — | docs(diagrams): add print-ready fitting workflow figure (PNG/PDF) | rebuilt mermaid as narrow vertical pipeline (invisible rank links + pinned `direction LR`) to fit a 3.375in single journal column at 300dpi; moved anti-degeneracy caption out of subgraph title to stop border/arrow overlap | landscape figure didn't fit single column; subgraph title overlapped box border at large print font | low |
