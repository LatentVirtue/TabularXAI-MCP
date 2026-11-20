"""Typer CLI: `profile`, `run`, `generate-demo`, `inspect-errors`.

The CLI is a thin shell over the engine — every command maps 1:1 onto an
engine function so CLI and MCP server stay behaviourally identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    help="Automated tabular ML arena (LightGBM vs PyTorch MLP), calibration "
         "diagnostics and SHAP explainability with dual-artifact output.",
    no_args_is_help=True,
)


def _echo_json(payload: dict) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def _summarize(manifest: dict) -> None:
    cmp_ = manifest["model_comparison"]
    winner = manifest["winner"]
    pm = manifest["winning_metric"]
    typer.echo("")
    typer.echo(f"  run id        {manifest['run_id']}")
    typer.echo(f"  task          {manifest['task_type']}")
    typer.echo(f"  winner        {winner}  ({pm} = {cmp_[winner]['oof'][pm]:.4f})")
    for fam, blob in cmp_.items():
        typer.echo(f"    {fam:<14} oof {pm} = {blob['oof'].get(pm, float('nan')):.4f}")
    if manifest["dataset_warnings"]:
        typer.echo(f"  warnings      {len(manifest['dataset_warnings'])} finding(s)")
        for w in manifest["dataset_warnings"][:6]:
            typer.echo(f"    - {w}")
    typer.echo(f"  agreement rho {manifest['attribution_agreement_spearman']}")
    typer.echo("")
    typer.echo(manifest.get("usage_hint", ""))


@app.command()
def profile(
    data_path: str = typer.Option(..., "--data", help="CSV or Parquet dataset."),
    target_column: str = typer.Option(..., "--target", help="Target column name."),
    drop_columns: list[str] = typer.Option([], "--drop", help="Column to exclude (repeatable)."),
    task_type: str = typer.Option("auto", "--task", help="auto|binary_classification|multiclass_classification|regression"),
) -> None:
    """Profile a dataset: schema, task inference, leakage/collinearity/ID/missingness guards."""
    from .engine.profiler import load_prepared_dataset, profile_dataset

    df, notes = load_prepared_dataset(data_path, target_column)
    prof = profile_dataset(df, target_column, list(drop_columns), task_type)  # type: ignore[arg-type]
    out = json.loads(prof.model_dump_json(indent=2))
    out["_notes"] = notes
    _echo_json(out)


@app.command()
def run(
    data_path: str = typer.Option(None, "--data", help="CSV or Parquet dataset."),
    target_column: str = typer.Option(None, "--target", help="Target column name."),
    task_type: str = typer.Option("auto", "--task"),
    output_dir: str = typer.Option("runs", "--out", help="Run artifacts land in <out>/<run_id>/."),
    n_splits: int = typer.Option(5, "--n-splits", min=2, max=10),
    nn_epochs: int = typer.Option(40, "--nn-epochs"),
    seed: int = typer.Option(42, "--seed"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Deterministic run id (default: timestamped)."),
    drop_columns: list[str] = typer.Option([], "--drop"),
    drop_leaky: bool = typer.Option(False, "--drop-leaky", help="Auto-drop columns the leakage guard flags."),
    config: Optional[str] = typer.Option(None, "--config", help="YAML/JSON ExperimentConfig; CLI flags override it."),
) -> None:
    """Run the full head-to-head arena + SHAP + dual-artifact pipeline."""
    from .engine.config import ExperimentConfig, NNParams
    from .engine.trainer import run_benchmark

    if config:
        cfg = ExperimentConfig.from_file(config)
        if data_path:
            cfg.data_path = data_path
        if target_column:
            cfg.target_column = target_column
    else:
        if not data_path or not target_column:
            typer.echo("--data and --target are required (or pass --config).", err=True)
            raise typer.Exit(2)
        cfg = ExperimentConfig(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,  # type: ignore[arg-type]
            n_splits=n_splits,
            seed=seed,
            drop_columns=list(drop_columns),
            auto_drop_leakage=drop_leaky,
            output_dir=output_dir,
            run_id=run_id,
            nn=NNParams(epochs=nn_epochs),
        )
    manifest = run_benchmark(cfg)
    _summarize(manifest)


@app.command("generate-demo")
def generate_demo(
    output_dir: str = typer.Option("examples/demo_run", help="Artifacts land directly in this directory."),
    task: str = typer.Option("binary_classification", "--task", help="Task for the synthetic generator."),
    rows: int = typer.Option(1500, "--rows", min=100),
    seed: int = typer.Option(7, "--seed"),
    nn_epochs: int = typer.Option(25, "--nn-epochs"),
) -> None:
    """Generate the synthetic benchmark dataset and run the full pipeline into it."""
    from .engine.config import ExperimentConfig, NNParams
    from .engine.synthetic import generate_dataset
    from .engine.trainer import run_benchmark

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "synthetic_dataset.csv"
    df = generate_dataset(task=task, n_rows=rows, seed=seed)  # type: ignore[arg-type]
    df.to_csv(csv_path, index=False)
    typer.echo(f"synthetic dataset ({task}, {len(df)} rows) -> {csv_path}")

    cfg = ExperimentConfig(
        data_path=str(csv_path),
        target_column="target",
        task_type=task,  # type: ignore[arg-type]
        n_splits=5,
        seed=seed,
        auto_drop_leakage=True,  # demo: show the guard firing, then act on it
        output_dir=str(out.parent),
        run_id=out.name,
        nn=NNParams(epochs=nn_epochs),
    )
    manifest = run_benchmark(cfg)
    _summarize(manifest)


@app.command("inspect-errors")
def inspect_errors(
    run_dir: str = typer.Option(..., "--run-dir", help="A completed run directory (contains manifest.json)."),
    rows: list[int] = typer.Option([], "--row", help="Explicit row position(s); default: the run's worst errors."),
    top_k: int = typer.Option(5, "--top-k", help="Top-k local SHAP drivers per model."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Explain individual predictions: values, OOF predictions, local SHAP drivers."""
    from .engine.explain import explain_rows

    rd = Path(run_dir)
    manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    positions = list(rows)
    if not positions:
        positions = sorted({p for lst in manifest["worst_error_indices"].values() for p in lst})
    result = explain_rows(rd, positions, top_k=top_k)
    if as_json:
        _echo_json(result)
        return
    for row in result["rows"]:
        typer.echo(f"\nrow {row['position']} (fold {row['fold']})  true={row.get('true_class') or row.get('true_target')}")
        for fam, m in row["models"].items():
            if "error" in m:
                typer.echo(f"  {fam}: error {m['error']}")
                continue
            typer.echo(f"  {fam}: pred={m['prediction']} probs={m['probabilities']}")
            for c in m["top_contributions"]:
                typer.echo(f"    {c['feature']:<22} {c['shap_contribution']:+.4f}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
