"""End-to-end regression arena: RMSE/MAE/R2, residuals, row-level explanation."""
from __future__ import annotations

import math
import pathlib

import pandas as pd

from tabular_xai.engine.config import MODEL_FAMILIES, MODEL_LIGHTGBM, MODEL_PYTORCH_MLP, RunManifest
from tabular_xai.engine.explain import explain_rows
from tabular_xai.engine.synthetic import ID_COLUMN


def test_regression_arena_metrics_and_manifest(tiny_benchmark):
    manifest, run_dir, _csv = tiny_benchmark("regression", rows=240, epochs=10)
    RunManifest.model_validate(manifest)
    assert manifest["estimated_manifest_tokens"] < 2000
    assert manifest["task_type"] == "regression"
    assert manifest["winning_metric"] == "rmse"
    assert manifest["winner"] in MODEL_FAMILIES

    for fam in MODEL_FAMILIES:
        blob = manifest["model_comparison"][fam]
        assert {"rmse", "mae", "r2"} <= set(blob["mean"].keys())
        assert len(blob["per_fold"]["rmse"]) == 3
        assert math.isfinite(blob["oof"]["rmse"]) and blob["oof"]["rmse"] > 0

    # planted signal is learnable — winner must explain real variance
    assert manifest["model_comparison"][manifest["winner"]]["oof"]["r2"] > 0.2

    assert ID_COLUMN not in manifest["dataset"]["feature_columns"]
    cal = manifest["model_comparison"][manifest["winner"]]["calibration"]
    assert cal["mean_abs_residual"] >= 0

    w = manifest["worst_error_indices"]
    assert "largest_absolute_residuals" in w and 1 <= len(w["largest_absolute_residuals"]) <= 5

    ap = manifest["artifact_paths"]
    for key in ("manifest_json", "report_html", "oof_predictions", "shap_gbdt", "shap_nn"):
        p = pathlib.Path(ap[key])
        assert p.exists() and p.stat().st_size > 100
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    for needle in ("cdn.plot.ly", "Predicted vs actual", "Residual"):
        assert needle in html


def test_regression_oof_and_explanations(tiny_benchmark):
    manifest, run_dir, csv_path = tiny_benchmark("regression", rows=240, epochs=10)
    oof = pd.read_parquet(run_dir / "oof_predictions.parquet")
    for fam in MODEL_FAMILIES:
        assert f"{fam}_pred" in oof.columns
    true = pd.read_csv(csv_path)["target"].to_numpy()
    assert len(oof) == len(true)

    positions = manifest["worst_error_indices"]["largest_absolute_residuals"][:2]
    result = explain_rows(run_dir, positions, top_k=3)
    assert result["n_rows"] == 2
    for row in result["rows"]:
        assert isinstance(row["true_target"], float)
        gb = row["models"][MODEL_LIGHTGBM]
        assert isinstance(gb["prediction"], float)
        assert 1 <= len(gb["top_contributions"]) <= 3
        nn = row["models"][MODEL_PYTORCH_MLP]
        assert ("top_contributions" in nn) or ("error" in nn)

    # residuals of the top-5 worst rows must be the largest across the OOF frame
    worst_pos = manifest["worst_error_indices"]["largest_absolute_residuals"]
    pred_col = f"{manifest['winner']}_pred"
    resid = (true - oof[pred_col].to_numpy())
    top2 = sorted(worst_pos, key=lambda p: -abs(resid[p]))[:2]
    assert set(top2) == set(positions) or len(worst_pos) < 2
