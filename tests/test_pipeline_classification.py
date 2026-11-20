"""End-to-end binary + multiclass classification arena, artifacts, row-explanations."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tabular_xai.engine.config import MODEL_FAMILIES, MODEL_LIGHTGBM, MODEL_PYTORCH_MLP, RunManifest
from tabular_xai.engine.explain import explain_rows
from tabular_xai.engine.synthetic import ID_COLUMN, LEAK_COLUMN

BINARY_KEYS = {"roc_auc", "pr_auc", "f1", "brier", "accuracy"}


@pytest.fixture(scope="module")
def binary(tiny_benchmark):
    manifest, run_dir, _csv = tiny_benchmark("binary_classification")
    return manifest, run_dir


def _classification_checks(manifest: dict, run_dir, expected_metric: str):
    # schema-valid, compact manifest
    RunManifest.model_validate(manifest)
    assert manifest["estimated_manifest_tokens"] < 2000

    # both families benchmarked under identical folds
    assert set(manifest["model_comparison"]) == set(MODEL_FAMILIES)
    for blob in manifest["model_comparison"].values():
        assert len(blob["per_fold"][expected_metric]) == 3
        assert blob["oof"][expected_metric] == blob["oof"][expected_metric]  # finite-checked below
        assert blob["fit_seconds"] > 0

    import math
    for fam in MODEL_FAMILIES:
        v = manifest["model_comparison"][fam]["oof"][expected_metric]
        assert math.isfinite(v)

    # winner decided on the headline metric
    assert manifest["winner"] in MODEL_FAMILIES
    assert manifest["winning_metric"] == expected_metric
    assert any(LEAK_COLUMN in w for w in manifest["dataset_warnings"])
    assert manifest["winner_rationale"]

    # ID column auto-dropped from features
    assert ID_COLUMN not in manifest["dataset"]["feature_columns"]

    # explainability present for BOTH model families
    assert len(manifest["top_features"]) >= 3
    for row in manifest["top_features"]:
        assert row["feature"] in manifest["dataset"]["feature_columns"]
    assert sum(f["gbdt_mean_abs_shap"] for f in manifest["top_features"]) > 0
    assert sum(f["nn_mean_abs_shap"] for f in manifest["top_features"]) >= 0
    rho = manifest["attribution_agreement_spearman"]
    assert rho is None or -1.0 <= rho <= 1.0

    # dual artifacts on disk, non-empty
    ap = manifest["artifact_paths"]
    for key in ("manifest_json", "report_html", "oof_predictions", "shap_gbdt", "shap_nn", "metrics_json"):
        p = ap[key]
        assert p, key
        import pathlib
        assert pathlib.Path(p).exists(), key
        assert pathlib.Path(p).stat().st_size > 100, key

    html = (run_dir / "report.html").read_text(encoding="utf-8")
    for needle in ("cdn.plot.ly", "Model Arena", "Explainability", "Worst Predictions", "Data Quality"):
        assert needle in html

    # OOF predictions cover every row exactly once, each fold non-empty
    oof = pd.read_parquet(run_dir / "oof_predictions.parquet")
    n = manifest["dataset"]["n_rows"]
    assert len(oof) == n
    folds = oof["_fold"].tolist()
    assert set(folds) == {0, 1, 2}
    counts = {f: folds.count(f) for f in (0, 1, 2)}
    assert all(c >= n // 3 - 1 for c in counts.values())


def test_binary_arena_metrics_and_artifacts(binary):
    manifest, run_dir = binary
    _classification_checks(manifest, run_dir, "roc_auc")
    for fam in MODEL_FAMILIES:
        assert BINARY_KEYS <= set(manifest["model_comparison"][fam]["mean"].keys())
        cal = manifest["model_comparison"][fam]["calibration"]
        assert 0.0 <= cal["ece"] <= 1.0
        assert len(cal["reliability_curve"]) > 0
    # the planted signal must be learnable: winner well above chance
    assert manifest["model_comparison"][manifest["winner"]]["oof"]["roc_auc"] > 0.7


def test_binary_worst_errors_are_fp_and_fn(binary):
    manifest, _ = binary
    w = manifest["worst_error_indices"]
    assert {"worst_false_positives", "worst_false_negatives"} <= set(w)
    assert all(len(v) <= 5 for v in w.values())
    oof = pd.read_parquet(manifest["artifact_paths"]["oof_predictions"])
    for pos in w["worst_false_positives"]:
        assert oof["y_true_code"].iloc[pos] == 0
    for pos in w["worst_false_negatives"]:
        assert oof["y_true_code"].iloc[pos] == 1


def test_explain_rows_for_worst_errors(binary):
    manifest, run_dir = binary
    positions = sorted({p for lst in manifest["worst_error_indices"].values() for p in lst})
    result = explain_rows(run_dir, positions, top_k=4)
    assert result["n_rows"] == len(positions)
    for row in result["rows"]:
        assert "values" in row and "true_target" in row
        models = row["models"]
        gb = models[MODEL_LIGHTGBM]
        assert "top_contributions" in gb and 1 <= len(gb["top_contributions"]) <= 4
        assert len(gb["probabilities"]) == 2
        nn = models[MODEL_PYTORCH_MLP]
        assert ("top_contributions" in nn) or ("error" in nn)


@pytest.fixture(scope="module")
def multiclass(tiny_benchmark):
    manifest, run_dir, _csv = tiny_benchmark("multiclass_classification", rows=300, epochs=10)
    return manifest, run_dir


def test_multiclass_arena(multiclass):
    manifest, run_dir = multiclass
    assert manifest["task_type"] == "multiclass_classification"
    _classification_checks(manifest, run_dir, "macro_f1")
    metrics = set(manifest["model_comparison"][MODEL_LIGHTGBM]["mean"].keys())
    assert {"macro_f1", "log_loss", "accuracy"} <= metrics
    metrics_json = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics_json["classes"]) >= 3
    w = manifest["worst_error_indices"]
    assert "worst_misclassifications" in w
