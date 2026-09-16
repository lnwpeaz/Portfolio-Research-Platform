# Repository audit and implementation plan

Inspection preceded presentation/code changes. Baseline: 66 tests passed in
53.59 seconds, with one joblib physical-core detection warning.

## Inventory

| Area | Existing implementation and finding |
|---|---|
| Structure | Root CLI scripts, `src/{data,universe,features,models,portfolio,backtest,evaluation,live}`, 13 test modules, five notebooks, `data`, `reports`, `paper` |
| Execution | `main.py`, `ml_main.py`, `comparison_main.py`, `research_main.py`, both robustness scripts, live/replay, and three institutional entrypoints |
| Signals | 3M and 12–1 monthly momentum; Top-5 equal, max-Sharpe and min-volatility variants; ML probability ranking |
| ML | Five explicit features; HistGradientBoostingClassifier; 36-month expanding training minimum; strict target-date embargo |
| Portfolio | Constrained long-only optimization, 252-price history, cap projection, explicit equal-weight fallbacks |
| Backtest | Shared momentum/score engines, next-period realization dates, fixed selection before outcomes, drift-aware one-way costs |
| Robustness | Leave-one-out, four calendar subperiods, costs, one-month delay; bootstrap, contributor/outlier counterfactuals, optimizer and ML stability, rolling consistency |
| Live | Snapshot cutoff, pure feature output, embargoed training, input hashes, immutable recommendations, isolated replay and institutional paper namespaces |
| Figures/reports | Root CSV/PNG performance, risk, attribution, ranking, concentration and robustness artifacts; dated live/replay trees; institutional current and historical outputs |
| Tests | Timing, leakage, constraints, metrics, robustness identities, deterministic live predictions, frozen paper evaluation, providers and cache contracts |
| Documentation | Operational README and detailed AUDIT; no concise research report or interview-oriented methodology reference |
| Generated artifacts | Approximately 19 MB data, 5.7 MB reports, 24 KB paper, 152 KB notebooks at inspection; bundled raw inputs total approximately 2.5 MB |
| Version control | No project-local `.git`; enclosing repository reports no tracked project files. No claim that any generated artifact is committed; no Git staging or repository initialization performed |
| Legacy/duplication | Notebook 04 contains old outcome dropping/date alignment; 05 consumes its stale backtest; 03 is a same-day exploratory calculation. The ignored `data/processed/backtest.parquet` is the obsolete notebook output. Duplicate selector/optimizer equal-weight helpers and older factor utilities retained to avoid unnecessary API changes |

## Audit/documentation discrepancies

- Optimizer condition/eigenvalue diagnostics already exist despite early audit
  wording treating them as future work. Stored historical solver objective
  values remain unavailable.
- Institutional and legacy fixed20 paths are distinct; current S&P membership
  is not historical PIT coverage. The dated 503-member acquisition status is
  historical context, not a claim about today's membership.
- Legacy live metadata hashes a snapshot but does not persist its input panel.
- `.gitignore` excluded all Parquet files, including reproduction inputs.
- Historical scripts did not persist a shared config/source/input manifest.
- Historical charts lacked visible bias/benchmark labels.
- Phase 2 reads Phase 1 CSVs; execution order is a real dependency.
- Hard-coded ML performance regression guards intentionally depend on the
  bundled data and environment. They are not portable acceptance thresholds
  for a different dataset.

## Implemented plan

1. Preserve official model, signals, parameters, engine, costs and report paths.
2. Replace the landing page; retain operational detail in `docs/OPERATIONS.md`.
3. Add research, methodology, reproduction and output guides.
4. Mark all early notebooks as superseded; retain their history.
5. Add report provenance and a snapshot of the installed dependencies.
6. Label historical figures; surface a small representative set in README.
7. Keep small bundled inputs and root examples visible to Git; ignore local
   caches, repeated runs, paper state and validation logs.
8. Execute tests and all requested CLIs; compare historical CSV hashes and
   deterministic replay outputs. Record actual outcomes in `VALIDATION.md`.

No large refactor, strategy redesign, parameter search, external data purchase,
or point-in-time data reconstruction is part of this update.
