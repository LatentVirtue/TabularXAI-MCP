"""Profiler + leakage-guard behaviour on the planted synthetic pathologies."""
from __future__ import annotations

from tabular_xai.engine.config import TaskType
from tabular_xai.engine.profiler import (
    COLLINEARITY_THRESHOLD,
    LEAK_R_THRESHOLD,
    infer_task_type,
    profile_dataset,
)
from tabular_xai.engine.synthetic import (
    COLLINEAR_PROXY,
    COLLINEAR_SOURCE,
    ID_COLUMN,
    LEAK_COLUMN,
    MISSING_COLUMN,
    generate_dataset,
)


def _profile(task: str, rows: int = 500):
    df = generate_dataset(task=task, n_rows=rows, seed=11)
    return df, profile_dataset(df, "target")


def test_binary_task_inference_and_leakage_flag():
    df, prof = _profile("binary_classification")
    assert infer_task_type(df, "target") is TaskType.BINARY_CLASSIFICATION
    assert prof.task_type is TaskType.BINARY_CLASSIFICATION
    leaked = {f.column: f for f in prof.potential_target_leakage}
    assert LEAK_COLUMN in leaked, "planted leaky sensor must be flagged"
    assert LEAK_R_THRESHOLD <= 0.95  # sanity on threshold constant
    assert any(f.association >= LEAK_R_THRESHOLD for f in prof.potential_target_leakage)


def test_collinear_cluster_detected():
    _, prof = _profile("binary_classification")
    clusters = [set(c) for c in prof.collinear_feature_clusters]
    assert any({COLLINEAR_SOURCE, COLLINEAR_PROXY} <= c for c in clusters), (
        f"{COLLINEAR_SOURCE} and {COLLINEAR_PROXY} must land in one cluster; got {clusters}"
    )
    assert any("collinear" in w.lower() for w in prof.warnings)


def test_id_like_and_high_missingness_flags():
    _, prof = _profile("binary_classification", rows=600)
    assert ID_COLUMN in prof.id_like_columns
    assert ID_COLUMN not in prof.numeric_feature_columns
    assert ID_COLUMN not in prof.categorical_feature_columns
    assert MISSING_COLUMN in prof.high_missing_columns
    assert prof.high_missing_columns[MISSING_COLUMN] > 25.0
    assert any("missing" in w.lower() for w in prof.warnings)


def test_multiclass_and_regression_inference():
    df_m, prof_m = _profile("multiclass_classification", rows=400)
    assert infer_task_type(df_m, "target") is TaskType.MULTICLASS_CLASSIFICATION
    assert prof_m.class_balance and len(prof_m.class_balance) >= 3

    df_r, prof_r = _profile("regression", rows=400)
    assert infer_task_type(df_r, "target") is TaskType.REGRESSION
    assert prof_r.class_balance is None
    assert prof_r.target_summary  # mean/std etc. populated for regression


def test_text_columns_stay_categorical_never_id_like():
    # Regression guard: pandas >=3 backs text with a dedicated `str` dtype,
    # not object. Text columns must classify as categorical and must never
    # fall through to the id_like/unusable bucket.
    _, prof = _profile("binary_classification", rows=600)
    assert prof.categorical_feature_columns, "synthetic generator plants string categoricals"
    assert ID_COLUMN in prof.id_like_columns
    assert ID_COLUMN not in prof.categorical_feature_columns
    assert not set(prof.id_like_columns) & set(prof.categorical_feature_columns)


def test_auto_drop_leakage_flag_removes_sensor(tiny_benchmark):
    """Guard fires, run acts on it: leaky sensor leaves the feature space,
    warnings still name it, and the headline metric is no longer leak-inflated."""
    manifest, _run_dir, _csv = tiny_benchmark("binary_classification", auto_drop_leakage=True)
    assert LEAK_COLUMN in manifest["dataset"]["dropped_columns"]
    assert LEAK_COLUMN not in manifest["dataset"]["feature_columns"]
    assert any("leak" in w.lower() for w in manifest["dataset_warnings"])
    winner = manifest["winner"]
    assert manifest["model_comparison"][winner]["oof"]["roc_auc"] < 0.97, (
        "with the planted leak excluded, OOF AUC must reflect the real signal"
    )


def test_role_partition_is_consistent():
    _, prof = _profile("binary_classification", rows=400)
    roles = {c.name: c.role for c in prof.columns}
    assert roles["target"] == "target"
    feature_names = set(prof.numeric_feature_columns) | set(prof.categorical_feature_columns)
    target_cols = {c.name for c in prof.columns if c.role == "target"}
    assert "target" in target_cols
    assert "target" not in feature_names
    for name, role in roles.items():
        if role == "id_like":
            assert name not in feature_names
    assert COLLINEARITY_THRESHOLD == 0.90
