# TabularXAI-MCP

An MCP server, CLI, and Python library that runs a complete tabular-ML
benchmark — dataset profiling, leakage guarding, a LightGBM-vs-PyTorch
cross-validation arena, calibration diagnostics, SHAP explainability with
cross-model attribution agreement — and emits **two artifacts from one run**:

| Artifact | Audience | Contract |
|---|---|---|
| `manifest.json` | LLM agents | Strictly typed (`RunManifest` pydantic schema), compact by construction — warnings capped and truncated, attributions capped at 10, error indices capped at 5 per category; estimated token count written into the manifest itself and kept under ~2,000. |
| `report.html` | Humans | Single-file interactive Plotly dashboard. Standalone: opens in any browser with no server, no package install. |

The point is the loop: an agent calls the tool, reads the compact manifest,
surfaces the `report.html` path for the human, and can then answer follow-ups
by re-reading artifacts — never by re-running training.

## Pipeline

```
CSV/Parquet ─▶ profiler + leakage guard ─▶ (stratified) K-fold arena ─▶ SHAP ─▶ manifest.json
                 │                           LightGBM vs PyTorch MLP          └▶ report.html
                 │                           identical folds, leak-free
                 ├─ target-leakage flags     per-fold preprocessing
                 ├─ collinear clusters       pooled OOF metrics + calibration
                 ├─ ID-like columns          winner + written rationale
                 └─ high-missingness flags   Spearman agreement of attributions
```

**Guards** (constants in `engine/profiler.py`): single-feature
|Pearson r| ≥ 0.95 or single-feature ROC-AUC ≥ 0.98 against the target flags
a leak; pairwise |r| ≥ 0.90 forms collinear clusters; ≥ 30 % missing flags a
column; high-cardinality integer columns are treated as IDs. ID-like columns
are auto-dropped; leaky columns are flagged by default and auto-dropped with
`auto_drop_leakage` / `--drop-leaky` — always reported either way.

**Arena**: both families train on identical (stratified) K-folds; every
imputer/scaler/encoder is fit on the fold's train partition only. Metrics per
fold (mean±std) and pooled out-of-fold: ROC-AUC, PR-AUC, F1, Brier, accuracy
(binary); macro-F1, log-loss (multiclass); RMSE, MAE, R² (regression).
Calibration: reliability curves + ECE per model.

**NN baseline**: PyTorch residual MLP — linear stem → N×(Linear → LayerNorm →
SiLU → Dropout + skip) → head; AdamW, cosine LR decay, early stopping with
best-weight restore, CPU/CUDA auto-detect.

**Explainability**: `TreeExplainer` for LightGBM, `GradientExplainer`
(Integrated-Gradients fallback) for the MLP, computed on OOF rows; global
mean|SHAP| and per-sample drivers; cross-model agreement = Spearman ρ between
GBDT gain importances and MLP SHAP importances. Per-row explanations after the
fact come from the saved fold checkpoints + SHAP matrices — no retraining.

## Install

Python ≥ 3.11. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # runtime deps (torch, lightgbm, shap, plotly, ...)
uv sync --extra dev     # + pytest
```

## CLI

```bash
tabular-xai generate-demo --rows 2000 --output-dir examples/demo_run
tabular-xai run --data my.csv --target y --n-splits 5 --drop-leaky
tabular-xai profile --data my.csv --target y
tabular-xai inspect-errors --run-dir examples/demo_run [--row 152 --top-k 3]
```

`run` also accepts `--config experiment.yaml` (a serialized
`ExperimentConfig`); CLI flags override it.

## MCP

The server is named `tabular-xai` and exposes:

- **Tools** — `profile_tabular_dataset`, `run_tabular_benchmark`
  (incl. `drop_columns` / `drop_leaky_columns`), `explain_specific_rows`
  (`mode="worst_errors"|"explicit"`), `compare_feature_attributions`,
  `list_runs`.
- **Resource** — `xai-run://manifest/{run_id}` returns the compact manifest
  directly.

Register it (Claude Desktop / Cursor / any MCP client):

```json
{
  "mcpServers": {
    "tabular-xai": { "command": "tabular-xai-mcp" }
  }
}
```

or, from a checkout without installing: `uv run --directory /path/to/repo
tabular-xai-mcp`. Tool descriptions tell the calling model that the human
dashboard lives at `artifact_paths.report_html` and to surface that path.

## Example run

`examples/demo_run/` is the output of `tabular-xai generate-demo` on the
bundled synthetic generator (2,000 rows, binary task, seed 7 — deterministic,
reproducible with one command). The synthetic dataset plants realistic
pathologies: nonlinear interactions, a collinear proxy (r ≈ 0.93), a leaky
post-outcome sensor (r ≈ 0.98), an ID column, a 35 %-missing noise column,
high-cardinality categoricals, and noise decoys. The run shows the guard
firing on all of them, dropping the leak and the ID automatically, and an
honest OOF ROC-AUC around 0.8 instead of the leak-inflated 1.0.

## Architecture rules

Enforced by layout and tests, not vibes:

1. `engine/` never imports `mcp` — the ML core is plain Python; the MCP
   layer (`mcp_server.py`) is a thin, testable shell that defers engine
   imports into tool bodies.
2. `engine/config.py` imports only pydantic/stdlib/PyYAML — schemas are
   importable without torch.
3. `import tabular_xai` stays light: no torch, no lightgbm at import time.
4. `reporting/` never imports the MCP SDK; the HTML report is standalone
   except a single Plotly CDN tag.
5. Determinism: fixed seeds, deterministic LightGBM (`deterministic=True`,
   `force_row_wise=True`), single-threaded NN training in tests.

## Tests

```bash
uv run pytest
```

16 tests across the four spec'd files: profiler/leakage pathologies
(including a pandas-3 `str`-dtype regression test), binary + multiclass and
regression end-to-end arenas (schema validation, fold/metric integrity, OOF
coverage, token budget, auto-drop-leakage behavior), and direct invocation of
the MCP tool functions plus FastMCP registration checks. A session-scoped
fixture keeps real end-to-end runs fast (≈40 s, CPU-only).

## License

Apache-2.0 — see `LICENSE` and `NOTICE`.
