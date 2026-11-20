"""K-Fold / Stratified-K-Fold arena orchestrator.

Runs BOTH LightGBM and the PyTorch residual MLP under identical splits with
leak-free per-fold preprocessing, computes per-fold + pooled out-of-fold
metrics with mean/std, calibration diagnostics (ECE, reliability), saves all
raw artifacts (OOF predictions, model checkpoints, metrics.json) and finally
delegates to explainability + reporting (dual artifacts).

STRICT RULE: no MCP imports here.
"""
from __future__ import annotations

import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from ..reporting.html_report import write_report
from ..reporting.manifest import build_manifest, save_manifest
from . import preprocessing
from .config import (
    MODEL_FAMILIES,
    MODEL_LIGHTGBM,
    MODEL_PYTORCH_MLP,
    PRIMARY_METRIC,
    ExperimentConfig,
    TaskType,
    metric_higher_is_better,
    utc_now_iso,
)
from .explain import FoldBundle, run_explainability
from .models_gbdt import make_gbdt
from .models_nn import TorchTabularModel
from .profiler import encode_target, load_prepared_dataset, profile_dataset

CALIBRATION_BINS = 10


def _safe(fn, default=float("nan")) -> float:
    try:
        v = float(fn())
        return v if np.isfinite(v) else default
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# Metric packs
# --------------------------------------------------------------------------- #


def _binary_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    p1 = proba[:, 1]
    pred = (p1 >= 0.5).astype(int)
    return {
        "roc_auc": _safe(lambda: roc_auc_score(y_true, p1)),
        "pr_auc": _safe(lambda: average_precision_score(y_true, p1)),
        "f1": _safe(lambda: f1_score(y_true, pred, zero_division=0)),
        "brier": _safe(lambda: brier_score_loss(y_true, p1)),
        "accuracy": _safe(lambda: accuracy_score(y_true, pred)),
    }


def _multiclass_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    n_cls = proba.shape[1]
    return {
        "macro_f1": _safe(lambda: f1_score(y_true, pred, average="macro", zero_division=0)),
        "log_loss": _safe(
            lambda: log_loss(y_true, np.clip(proba, 1e-9, 1.0), labels=list(range(n_cls)))
        ),
        "accuracy": _safe(lambda: accuracy_score(y_true, pred)),
    }


def _regression_metrics(y_true: np.ndarray, values: np.ndarray) -> dict:
    return {
        "rmse": _safe(lambda: float(np.sqrt(mean_squared_error(y_true, values)))),
        "mae": _safe(lambda: mean_absolute_error(y_true, values)),
        "r2": _safe(lambda: r2_score(y_true, values)),
    }


def _metric_pack(task: TaskType, y_true: np.ndarray, proba: np.ndarray) -> dict:
    if task is TaskType.BINARY_CLASSIFICATION:
        return _binary_metrics(y_true, proba)
    if task is TaskType.MULTICLASS_CLASSIFICATION:
        return _multiclass_metrics(y_true, proba)
    return _regression_metrics(y_true, proba[:, 0])


# --------------------------------------------------------------------------- #
# Calibration diagnostics
# --------------------------------------------------------------------------- #


def binary_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = CALIBRATION_BINS) -> tuple[float, list[dict]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    curve = []
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(probs[mask].mean())
        frac_pos = float(y_true[mask].mean())
        ece += (count / n) * abs(frac_pos - mean_pred)
        curve.append(
            {
                "bin_center": round((edges[b] + edges[b + 1]) / 2, 3),
                "mean_predicted": round(mean_pred, 4),
                "observed_rate": round(frac_pos, 4),
                "count": count,
            }
        )
    return round(float(ece), 5), curve


def top_class_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = CALIBRATION_BINS) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count:
            ece += (count / n) * abs(correct[mask].mean() - conf[mask].mean())
    return round(float(ece), 5)


def _calibration(task: TaskType, y: np.ndarray, proba: np.ndarray) -> dict:
    if task is TaskType.BINARY_CLASSIFICATION:
        ece, curve = binary_ece(y, proba[:, 1])
        return {
            "ece": ece,
            "ece_n_bins": CALIBRATION_BINS,
            "brier": _safe(lambda: brier_score_loss(y, proba[:, 1])),
            "reliability_curve": curve,
        }
    if task is TaskType.MULTICLASS_CLASSIFICATION:
        return {
            "ece_top_class": top_class_ece(y, proba),
            "log_loss": _safe(lambda: log_loss(y, np.clip(proba, 1e-9, 1.0), labels=list(range(proba.shape[1])))),
        }
    residuals = y - proba[:, 0]
    return {
        "mean_residual": round(float(residuals.mean()), 5),
        "mean_abs_residual": round(float(np.abs(residuals).mean()), 5),
        "p90_abs_residual": round(float(np.percentile(np.abs(residuals), 90)), 5),
        "std_residual": round(float(residuals.std()), 5),
    }


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def _make_folds(y: np.ndarray, task: TaskType, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if task.is_classification:
        min_class = int(pd.Series(y).value_counts().min())
        n_splits = max(2, min(n_splits, min_class))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y))))


def _predict_matrix(model, task: TaskType, X_enc: np.ndarray) -> np.ndarray:
    if task is TaskType.REGRESSION:
        return np.asarray(model.predict(X_enc), dtype=float).reshape(-1, 1)
    return np.asarray(model.predict_proba(X_enc), dtype=float)


def _worst_positions(task: TaskType, y: np.ndarray, proba: np.ndarray, k: int = 5) -> dict[str, list[int]]:
    if task is TaskType.BINARY_CLASSIFICATION:
        p1 = proba[:, 1]
        fp_order = np.argsort(-p1)
        fp_idx = fp_order[y[fp_order] == 0][:k]
        fn_order = np.argsort(p1)
        fn_idx = fn_order[y[fn_order] == 1][:k]
        return {
            "worst_false_positives": [int(i) for i in fp_idx],
            "worst_false_negatives": [int(i) for i in fn_idx],
        }
    if task is TaskType.MULTICLASS_CLASSIFICATION:
        pred = proba.argmax(axis=1)
        wrong = np.where(pred != y)[0]
        conf_wrong = proba[wrong, pred[wrong]]
        order = wrong[np.argsort(-conf_wrong)][:k]
        return {"worst_misclassifications": [int(i) for i in order]}
    residual = np.abs(y - proba[:, 0])
    order = np.argsort(-residual)[:k]
    return {"largest_absolute_residuals": [int(i) for i in order]}


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def run_benchmark(config: ExperimentConfig) -> dict:
    """Execute the full arena + SHAP + dual-artifact pipeline. Returns the
    manifest as a plain dict."""
    t_start = time.perf_counter()
    run_id = config.resolved_run_id()
    run_dir = Path(config.output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    rng = np.random.default_rng(config.seed)

    # ---- data + profile ----------------------------------------------------
    df, notes = load_prepared_dataset(config.data_path, config.target_column)
    profile = profile_dataset(df, config.target_column, config.drop_columns, config.task_type)
    task = TaskType(config.task_type) if config.task_type != "auto" else profile.task_type
    y, classes = encode_target(df, config.target_column, task)
    n = len(df)

    drop_set = set(config.drop_columns)
    auto_drop = list(profile.id_like_columns) if config.auto_drop_id_columns else []
    leak_drop = (
        [f.column for f in profile.potential_target_leakage] if config.auto_drop_leakage else []
    )
    excluded = drop_set | set(auto_drop) | set(leak_drop)
    feature_cols = [
        c for c in profile.numeric_feature_columns + profile.categorical_feature_columns
        if c not in excluded
    ]
    if auto_drop:
        notes.append("Auto-dropped ID-like columns: " + ", ".join(auto_drop))
    if leak_drop:
        notes.append("Auto-dropped leaky columns (guard fired): " + ", ".join(leak_drop))
    if drop_set:
        notes.append("Dropped per config: " + ", ".join(sorted(drop_set)))
    if not feature_cols:
        raise ValueError("No usable feature columns remain after exclusions.")
    X = df[feature_cols]
    numeric_sel = [c for c in feature_cols if c in set(profile.numeric_feature_columns)]
    categorical_sel = [c for c in feature_cols if c in set(profile.categorical_feature_columns)]

    # ---- arena --------------------------------------------------------------
    folds = _make_folds(y, task, config.n_splits, config.seed)
    width = 1 if task is TaskType.REGRESSION else (2 if task is TaskType.BINARY_CLASSIFICATION else len(classes))
    oof_probas: dict[str, np.ndarray] = {f: np.full((n, width), np.nan) for f in MODEL_FAMILIES}
    oof_fold_of = np.full(n, -1, dtype=int)
    bundles_by_family: dict[str, list[FoldBundle]] = {}
    per_model: dict[str, dict] = {}

    for family in MODEL_FAMILIES:
        for_nn = family == MODEL_PYTORCH_MLP
        fold_metrics: list[dict] = []
        fit_seconds = 0.0
        n_parameters = None
        for k, (tr, va) in enumerate(folds):
            prep = preprocessing.build_column_transformer(numeric_sel, categorical_sel, for_nn=for_nn)
            Xt = np.asarray(prep.fit_transform(X.iloc[tr]), dtype=np.float64)
            Xv = np.asarray(prep.transform(X.iloc[va]), dtype=np.float64)

            t1 = time.perf_counter()
            if family == MODEL_LIGHTGBM:
                model = make_gbdt(task, config.gbdt, config.seed)
                model.fit(Xt, y[tr])
            else:
                model = TorchTabularModel(task, config.nn, seed=config.seed)
                model.n_classes_ = width if task.is_classification else 0
                model.classes_ = classes
                model.fit(Xt, y[tr], Xv, y[va])
                n_parameters = model.n_parameters_
            fit_seconds += time.perf_counter() - t1

            proba = _predict_matrix(model, task, Xv)
            oof_probas[family][va] = proba
            oof_fold_of[va] = k
            fold_metrics.append(_metric_pack(task, y[va], proba))

            bg_n = min(32, Xt.shape[0])
            bundle = FoldBundle(
                family=family,
                fold=k,
                prep=prep,
                model=model,
                bg_sample=Xt[rng.choice(Xt.shape[0], size=bg_n, replace=False)].astype(np.float32),
                val_positions=va,
            )
            bundles_by_family.setdefault(family, []).append(bundle)
            with open(run_dir / "models" / f"fold{k}_{family}.pkl", "wb") as fh:
                pickle.dump(bundle, fh)

        oof = _metric_pack(task, y, oof_probas[family])
        keys = fold_metrics[0].keys()
        mean = {m: _safe(lambda m=m: np.nanmean([fm[m] for fm in fold_metrics])) for m in keys}
        std = {m: _safe(lambda m=m: np.nanstd([fm[m] for fm in fold_metrics])) for m in keys}
        per_model[family] = {
            "per_fold": {m: [round(float(fm[m]), 6) for fm in fold_metrics] for m in keys},
            "mean": {m: round(v, 6) for m, v in mean.items()},
            "std": {m: round(v, 6) for m, v in std.items()},
            "oof": {m: round(v, 6) for m, v in oof.items()},
            "calibration": {k: (v if k != "reliability_curve" else v) for k, v in _calibration(task, y, oof_probas[family]).items()},
            "fit_seconds": round(fit_seconds, 3),
            "n_parameters": n_parameters,
        }

        # final full-data fit (used as reference checkpoints)
        prep_full = preprocessing.build_column_transformer(numeric_sel, categorical_sel, for_nn=for_nn)
        Xf = np.asarray(prep_full.fit_transform(X), dtype=np.float64)
        if family == MODEL_LIGHTGBM:
            model_full = make_gbdt(task, config.gbdt, config.seed)
            model_full.fit(Xf, y)
        else:
            model_full = TorchTabularModel(task, config.nn, seed=config.seed)
            model_full.n_classes_ = width if task.is_classification else 0
            model_full.classes_ = classes
            model_full.fit(Xf, y)
        with open(run_dir / "models" / f"final_{family}.pkl", "wb") as fh:
            pickle.dump({"prep": prep_full, "model": model_full}, fh)

    # ---- winner -------------------------------------------------------------
    primary = PRIMARY_METRIC[task]
    higher_better = metric_higher_is_better(primary)
    cand = {
        f: per_model[f]["oof"].get(primary, float("nan"))
        for f in MODEL_FAMILIES
        if np.isfinite(per_model[f]["oof"].get(primary, float("nan")))
    }
    if not cand:
        raise RuntimeError("No finite primary metric could be computed for any model family.")
    winner = max(cand, key=lambda f: cand[f] if higher_better else -cand[f])
    loser = [f for f in MODEL_FAMILIES if f != winner][0]
    rationale = (
        f"'{winner}' won on pooled OOF {primary}: {cand[winner]:.4f} vs "
        f"{cand.get(loser, float('nan')):.4f} for '{loser}'."
    )

    worst = _worst_positions(task, y, oof_probas[winner])

    # ---- explainability -----------------------------------------------------
    all_worst_positions = sorted({p for lst in worst.values() for p in lst})
    expl = run_explainability(
        X,
        feature_cols,
        bundles_by_family,
        task,
        config.seed,
        config.max_shap_rows,
        all_worst_positions,
        run_dir,
    )

    # ---- raw artifacts ------------------------------------------------------
    oof_df = pd.DataFrame({"y_true_code": y.astype(int) if task.is_classification else y.astype(float)})
    if task.is_classification:
        oof_df["y_true"] = [classes[c] for c in y]
    for family in MODEL_FAMILIES:
        proba = oof_probas[family]
        for c in range(width):
            col_name = f"{family}_pred" if task is TaskType.REGRESSION else f"{family}_proba_{c}"
            oof_df[col_name] = proba[:, c]
        if task is TaskType.BINARY_CLASSIFICATION:
            oof_df[f"{family}_pred"] = (oof_df[f"{family}_proba_1"] >= 0.5).astype(int)
        elif task is TaskType.MULTICLASS_CLASSIFICATION:
            oof_df[f"{family}_pred"] = np.argmax(proba, axis=1)
    oof_df["_pos"] = np.arange(n, dtype=int)
    oof_df["_fold"] = oof_fold_of
    oof_df = oof_df.reset_index(drop=True)
    oof_df.to_parquet(run_dir / "oof_predictions.parquet", index=False)

    metrics_blob = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "task_type": task.value,
        "classes": classes,
        "n_splits": len(folds),
        "fold_sizes": [[int(len(tr)), int(len(va))] for tr, va in folds],
        "primary_metric": primary,
        "winner": winner,
        "per_model": per_model,
        "attribution_agreement_spearman": expl["attribution_agreement_spearman"],
        "profile": profile.model_dump(mode="json"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_blob, indent=2), encoding="utf-8")
    (run_dir / "config_snapshot.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")

    dataset_copy = run_dir / ("dataset_source" + Path(config.data_path).suffix.lower())
    if Path(config.data_path).resolve() != dataset_copy.resolve():
        shutil.copy2(config.data_path, dataset_copy)

    # ---- manifest + report (dual artifact) ----------------------------------
    artifact_paths = {
        "manifest_json": (run_dir / "manifest.json").resolve().as_posix(),
        "report_html": (run_dir / "report.html").resolve().as_posix(),
        "oof_predictions": (run_dir / "oof_predictions.parquet").resolve().as_posix(),
        "shap_gbdt": expl["shap_paths"][MODEL_LIGHTGBM],
        "shap_nn": expl["shap_paths"][MODEL_PYTORCH_MLP],
        "metrics_json": (run_dir / "metrics.json").resolve().as_posix(),
        "models_dir": (run_dir / "models").resolve().as_posix(),
        "dataset_copy": dataset_copy.resolve().as_posix(),
    }

    manifest = build_manifest(
        run_id=run_id,
        task=task,
        config=config,
        profile=profile,
        feature_cols=feature_cols,
        dropped=sorted(excluded),
        notes=notes,
        per_model=per_model,
        winner=winner,
        winning_metric=primary,
        rationale=rationale,
        worst=worst,
        expl=expl,
        artifact_paths=artifact_paths,
    )
    tokens = manifest.estimated_token_count()
    manifest_dict = manifest.model_dump(mode="json")
    manifest_dict["estimated_manifest_tokens"] = tokens
    (run_dir / "manifest.json").write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")

    report_ctx = {
        "manifest": manifest_dict,
        "profile": profile.model_dump(mode="json"),
        "task": task,
        "classes": classes,
        "feature_cols": feature_cols,
        "oof_df": oof_df,
        "per_model": per_model,
        "winner": winner,
        "expl": expl,
        "worst": worst,
        "df": df,
        "run_dir": run_dir,
    }
    write_report(report_ctx)

    out = dict(manifest_dict)
    out["runtime_seconds"] = round(time.perf_counter() - t_start, 2)
    out["usage_hint"] = (
        f"Human-facing interactive dashboard: open {artifact_paths['report_html']} in a browser. "
        "Raw per-row data lives in the parquet files under artifact_paths."
    )
    return out
