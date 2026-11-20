"""FastMCP server exposing the engine as MCP Tools + a manifest Resource.

Tool functions are plain callables decorated with @mcp.tool() (FastMCP's
decorator returns the original function), so tests can invoke them
directly without an MCP client. Engine imports are deferred into the tool
bodies to keep server startup instant and failure messages sharp.
"""
from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "tabular-xai",
    instructions=(
        "Trains and benchmarks tabular models (LightGBM vs a PyTorch residual MLP) "
        "under identical cross-validation, checks calibration, computes SHAP "
        "attributions and their cross-model agreement, and returns a compact JSON "
        "manifest for agents plus an interactive report.html for humans. Always "
        "surface the report.html path from artifact_paths to the user."
    ),
)

RunId = str


def _find_run(output_dir: str, run_id: str) -> Path:
    base = Path(output_dir).expanduser()
    if (base / "manifest.json").exists():  # caller pointed straight at a run dir
        return base
    exact = base / run_id
    if (exact / "manifest.json").exists():
        return exact
    if base.is_dir():
        candidates = [d for d in sorted(base.iterdir()) if (d / "manifest.json").exists()]
        if run_id in ("", "latest", "last") and candidates:
            return candidates[-1]
        matches = [d for d in candidates if run_id in d.name]
        if len(matches) == 1:
            return matches[0]
    known = [d.name for d in base.glob("*") if (d / "manifest.json").exists()] if base.is_dir() else []
    raise ValueError(f"run '{run_id}' not found under '{base}'. Known runs: {known or 'none'}")


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@mcp.tool()
def profile_tabular_dataset(data_path: str, target_column: str) -> dict:
    """Profile a tabular dataset BEFORE modelling.

    Returns the column schema with inferred roles, the recommended task_type
    (binary_classification | multiclass_classification | regression), missingness
    statistics, collinear feature clusters (|r| > 0.90), ID-like columns, and
    potential_target_leakage flags (|r| > 0.95 or single-feature AUC > 0.98
    against the target). Use this first to decide what to drop before calling
    run_tabular_benchmark.

    Args:
        data_path: Absolute or relative path to a CSV or Parquet file.
        target_column: Name of the column to predict.
    """
    from .engine.profiler import load_prepared_dataset, profile_dataset

    df, notes = load_prepared_dataset(data_path, target_column)
    prof = profile_dataset(df, target_column)
    out = json.loads(prof.model_dump_json())
    out["usage_hint"] = (
        "Review potential_target_leakage and collinear_feature_clusters; pass the "
        f"columns you want excluded via run_tabular_benchmark(drop_columns=...). Notes: {notes}"
    )
    return out


@mcp.tool()
def run_tabular_benchmark(
    data_path: str,
    target_column: str,
    task_type: str = "auto",
    output_dir: str = "runs",
    n_splits: int = 5,
    nn_epochs: int = 40,
    drop_columns: list[str] | None = None,
    drop_leaky_columns: bool = False,
) -> dict:
    """Run the full tabular benchmark: CV arena, calibration, SHAP, dual artifacts.

    Trains a LightGBM baseline and a PyTorch residual MLP under identical
    (stratified) K-fold splits, computes out-of-fold metrics (ROC-AUC/PR-AUC/
    F1/Brier/ECE for binary; macro-F1/LogLoss for multiclass; RMSE/MAE/R2 for
    regression), global and per-sample SHAP values, and cross-model attribution
    agreement (Spearman). ID-like columns are auto-dropped.

    Returns the compact manifest dict: model_comparison (per-fold mean±std and
    pooled OOF metrics), winner + rationale, calibration_summary, top_features,
    attribution_agreement_spearman, worst_error_indices, and artifact_paths.
    The human-facing interactive dashboard is at artifact_paths.report_html —
    tell the user they can open it in a browser.

    Args:
        data_path: CSV or Parquet dataset path.
        target_column: Column to predict.
        task_type: 'auto' or an explicit task name.
        output_dir: Directory that will contain <run_id>/ artifacts.
        n_splits: Number of CV folds (2-10).
        nn_epochs: Max training epochs for the PyTorch MLP.
        drop_columns: Columns to exclude from modelling (e.g. flagged leaks).
        drop_leaky_columns: Auto-exclude every column the leakage guard flags.
    """
    from .engine.config import ExperimentConfig, NNParams
    from .engine.trainer import run_benchmark

    cfg = ExperimentConfig(
        data_path=data_path,
        target_column=target_column,
        task_type=task_type,  # type: ignore[arg-type]
        n_splits=n_splits,
        output_dir=output_dir,
        nn=NNParams(epochs=nn_epochs),
        drop_columns=list(drop_columns or []),
        auto_drop_leakage=drop_leaky_columns,
    )
    return run_benchmark(cfg)


@mcp.tool()
def explain_specific_rows(
    run_id: RunId,
    mode: str = "worst_errors",
    row_indices: list[int] | None = None,
    top_k_features: int = 5,
    output_dir: str = "runs",
) -> dict:
    """Explain individual predictions from a completed run.

    mode='worst_errors' uses the run's worst false positives/negatives,
    confident misclassifications or largest residuals; mode='explicit' explains
    the given row_indices (positions in the dataset as loaded, after row
    filtering — see manifest worst_error_indices for the same index space).

    For each row returns original feature values, the true target/class, each
    model family's out-of-fold prediction/probabilities, and its top-k signed
    local SHAP contributions — enough to narrate why a specific prediction
    failed.

    Args:
        run_id: Run identifier, or 'latest' for the newest run in output_dir.
        mode: 'worst_errors' or 'explicit'.
        row_indices: Row positions to explain when mode='explicit'.
        top_k_features: Max SHAP drivers per model per row.
        output_dir: Directory holding run folders.
    """
    from .engine.explain import explain_rows

    run_dir = _find_run(output_dir, run_id)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if mode == "explicit":
        if not row_indices:
            raise ValueError("mode='explicit' requires row_indices")
        positions = [int(r) for r in row_indices]
    elif mode == "worst_errors":
        positions = sorted({p for lst in manifest["worst_error_indices"].values() for p in lst})
    else:
        raise ValueError(f"unknown mode '{mode}' (use 'worst_errors' or 'explicit')")
    result = explain_rows(run_dir, positions, top_k=top_k_features)
    result["report_html"] = manifest["artifact_paths"].get("report_html", "")
    return result


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #


@mcp.resource("xai://runs/{run_id}/manifest")
def run_manifest_resource(run_id: str) -> str:
    """Raw compact manifest.json for a run (search 'runs' directory). Agent-ready."""
    run_dir = _find_run("runs", run_id)
    return (run_dir / "manifest.json").read_text(encoding="utf-8")


def main() -> None:
    """Console entrypoint: serve the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
