"""SHAP explainability + cross-model attribution agreement.

Global mean|SHAP| and per-sample attributions for BOTH model families on
out-of-fold (held-out per fold) rows, plus a Spearman rank-agreement metric
between LightGBM gain importances and PyTorch-MLP SHAP importances.

SHAP backends: TreeExplainer for LightGBM; shap.GradientExplainer for the
PyTorch MLP with a deterministic Integrated-Gradients fallback (same math,
zero version fragility).

STRICT RULE: this module never imports anything MCP-related.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .config import MODEL_LIGHTGBM, MODEL_PYTORCH_MLP, TaskType
from .models_gbdt import gain_importance

# --------------------------------------------------------------------------- #
# Fold bundles (built by the trainer, consumed here)
# --------------------------------------------------------------------------- #


@dataclass
class FoldBundle:
    family: str
    fold: int
    prep: ColumnTransformer
    model: Any
    bg_sample: np.ndarray  # background rows (encoded) for gradient methods
    val_positions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.prep.transform(X_df), dtype=np.float64)

    def predict_matrix(self, X_df: pd.DataFrame, task: TaskType) -> np.ndarray:
        X = self.transform(X_df)
        if task is TaskType.REGRESSION:
            return np.asarray(self.model.predict(X), dtype=float).reshape(-1, 1)
        return np.asarray(self.model.predict_proba(X), dtype=float)


# --------------------------------------------------------------------------- #
# Encoded -> base feature mapping (from a fitted ColumnTransformer)
# --------------------------------------------------------------------------- #


def encoded_base_columns(prep: ColumnTransformer) -> list[str]:
    """For each encoded output column of a FITTED ColumnTransformer, return
    the original (base) column it was derived from."""
    bases: list[str] = []
    for t_name, trans, cols in prep.transformers_:
        if t_name == "remainder" or trans == "drop" or trans == "passthrough":
            continue
        cols = list(cols)
        if isinstance(trans, Pipeline) and "onehot" in getattr(trans, "named_steps", {}):
            onehot = trans.named_steps["onehot"]
            for col, cats in zip(cols, onehot.categories_):
                bases.extend([str(col)] * len(cats))
        else:
            bases.extend([str(c) for c in cols])
    n_out = len(prep.get_feature_names_out())
    if len(bases) != n_out:
        raise RuntimeError(f"encoded/base column mapping mismatch: {len(bases)} vs {n_out}")
    return bases


def _as_3d(vals) -> np.ndarray:
    """Normalize shap outputs to (n_samples, n_features, n_outputs)."""
    if isinstance(vals, (list, tuple)):
        arrs = [np.asarray(v, dtype=np.float64) for v in vals]
        if len(arrs) == 1:
            a = arrs[0]
            return a[:, :, None] if a.ndim == 2 else a
        if all(a.ndim == 2 for a in arrs):
            return np.stack(arrs, axis=-1)
        return arrs[0]
    arr = np.asarray(vals, dtype=np.float64)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


# --------------------------------------------------------------------------- #
# SHAP backends
# --------------------------------------------------------------------------- #


def shap_gbdt(bundle: FoldBundle, X_enc: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(bundle.model)
    try:
        vals = explainer.shap_values(X_enc, check_additivity=False)
    except TypeError:  # older shap signature
        vals = explainer.shap_values(X_enc)
    return _as_3d(vals)


def _integrated_gradients(module: torch.nn.Module, X: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Deterministic IG fallback: attr = (x - baseline) * integral(grad)."""
    module.eval()
    x0 = torch.tensor(bg.mean(axis=0), dtype=torch.float32).unsqueeze(0)
    x = torch.tensor(X, dtype=torch.float32)
    steps = 24
    n, feat = X.shape
    with torch.enable_grad():
        xin = (x0 + 1.0 * (x - x0)).requires_grad_(True)
        n_out = module(xin).shape[1]
    accum: list[np.ndarray] = []
    with torch.enable_grad():
        for c in range(n_out):
            g_sum = np.zeros_like(X)
            for a in np.linspace(0.0, 1.0, steps):
                xin = (x0 + float(a) * (x - x0)).requires_grad_(True)
                out = module(xin)
                (g,) = torch.autograd.grad(out[:, c].sum(), xin)
                g_sum += g.numpy()
            accum.append((X - x0.numpy()) * (g_sum / steps))
    return np.stack(accum, axis=-1)


def shap_nn(bundle: FoldBundle, X_enc: np.ndarray) -> np.ndarray:
    module: torch.nn.Module = bundle.model.model
    bg = np.clip(np.asarray(bundle.bg_sample, dtype=np.float32), -20.0, 20.0)
    X = np.clip(np.asarray(X_enc, dtype=np.float32), -20.0, 20.0)
    try:
        explainer = shap.GradientExplainer(module, torch.as_tensor(bg))
        vals = explainer.shap_values(torch.as_tensor(X))
        return _as_3d(vals)
    except Exception:
        return _integrated_gradients(module, X, bg)


def compute_shap(bundle: FoldBundle, X_enc: np.ndarray) -> np.ndarray:
    if bundle.family == MODEL_LIGHTGBM:
        return shap_gbdt(bundle, X_enc)
    return shap_nn(bundle, X_enc)


# --------------------------------------------------------------------------- #
# Full-run explainability pass
# --------------------------------------------------------------------------- #


def _aggregate(vec: np.ndarray, base_index: np.ndarray, n_base: int) -> np.ndarray:
    return np.bincount(base_index, weights=vec, minlength=n_base)


def run_explainability(
    X_df: pd.DataFrame,
    base_features: list[str],
    bundles_by_family: dict[str, list[FoldBundle]],
    task: TaskType,
    seed: int,
    max_shap_rows: int,
    force_include_positions: list[int],
    run_dir: Path,
) -> dict:
    """Per-fold SHAP on the held-out rows, saved as parquet matrices, with
    aggregated global importances, the top-10 feature table and the
    cross-model agreement (Spearman) metric."""
    rng = np.random.default_rng(seed + 17)
    n_folds = max(1, len(next(iter(bundles_by_family.values()))))
    per_fold_quota = max(16, max_shap_rows // n_folds)
    force_set = set(int(p) for p in force_include_positions)

    frames: dict[str, pd.DataFrame] = {}
    importance: dict[str, list[np.ndarray]] = {}
    gains: list[np.ndarray] = []
    encoded_to_base: dict[str, str] = {}

    for family, bundles in bundles_by_family.items():
        rows: list[pd.DataFrame] = []
        imp_folds: list[np.ndarray] = []
        for b in bundles:
            prep = b.prep
            bases = encoded_base_columns(prep)
            names = [str(x) for x in prep.get_feature_names_out()]
            encoded_to_base.update(dict(zip(names, bases)))
            base_index = np.array([base_features.index(c) for c in bases])
            n_base = len(base_features)

            positions = np.asarray(b.val_positions, dtype=int)
            if len(positions) > per_fold_quota:
                chosen = rng.choice(positions, size=per_fold_quota, replace=False)
                forced_in_fold = np.array([p for p in force_set if p in set(positions.tolist())], dtype=int)
                chosen = np.unique(np.concatenate([chosen, forced_in_fold]))
            else:
                chosen = positions
            if len(chosen) == 0:
                continue
            X_enc = b.transform(X_df.iloc[chosen])
            vals3 = compute_shap(b, X_enc)
            signed = vals3.mean(axis=-1)  # (rows, encoded_feat)

            frame = pd.DataFrame(signed.astype(np.float32), columns=names)
            frame.insert(0, "_fold", b.fold)
            frame.insert(0, "_pos", chosen)
            rows.append(frame)

            global_abs = np.abs(vals3).mean(axis=(-1, 0))  # per encoded feature
            imp_folds.append(_aggregate(global_abs, base_index, n_base))

            if family == MODEL_LIGHTGBM:
                g = np.asarray(gain_importance(b.model), dtype=float)
                gains.append(_aggregate(g, base_index, n_base))

        if not rows:
            frames[family] = pd.DataFrame(columns=["_pos", "_fold"])
            importance[family] = [np.zeros(len(base_features))]
            continue
        frames[family] = pd.concat(rows, ignore_index=True)
        importance[family] = imp_folds

    gbdt_shap = np.mean(importance.get(MODEL_LIGHTGBM, [np.zeros(len(base_features))]), axis=0)
    nn_shap = np.mean(importance.get(MODEL_PYTORCH_MLP, [np.zeros(len(base_features))]), axis=0)
    gbdt_gain = np.mean(gains, axis=0) if gains else np.zeros(len(base_features))

    # --- save raw SHAP matrices --------------------------------------------
    shap_paths: dict[str, str] = {}
    for family, fname in (
        (MODEL_LIGHTGBM, "shap_values_gbdt.parquet"),
        (MODEL_PYTORCH_MLP, "shap_values_nn.parquet"),
    ):
        p = run_dir / fname
        frames[family].to_parquet(p, index=False)
        shap_paths[family] = p.resolve().as_posix()

    # --- top features (max 10, keeps the manifest compact) -----------------
    order = np.argsort(-np.maximum(gbdt_shap, nn_shap))
    top_features = [
        {
            "feature": base_features[i],
            "gbdt_gain_importance": round(float(gbdt_gain[i]), 6),
            "gbdt_mean_abs_shap": round(float(gbdt_shap[i]), 6),
            "nn_mean_abs_shap": round(float(nn_shap[i]), 6),
        }
        for i in order[:10]
    ]

    # --- cross-model agreement: LightGBM gain vs NN SHAP --------------------
    agreement = None
    if np.std(gbdt_gain) > 0 and np.std(nn_shap) > 0:
        rho = spearmanr(gbdt_gain, nn_shap).statistic
        if np.isfinite(rho):
            agreement = round(float(rho), 4)

    return {
        "shap_frames": frames,
        "shap_paths": shap_paths,
        "base_features": base_features,
        "encoded_to_base": encoded_to_base,
        "gbdt_mean_abs_shap": gbdt_shap,
        "nn_mean_abs_shap": nn_shap,
        "gbdt_gain_importance": gbdt_gain,
        "top_features": top_features,
        "attribution_agreement_spearman": agreement,
    }


# --------------------------------------------------------------------------- #
# Row-level explanations (used by `explain_specific_rows` MCP tool and CLI)
# --------------------------------------------------------------------------- #


def _load_bundle(run_dir: Path, family: str, fold: int) -> FoldBundle:
    import pickle

    with open(run_dir / "models" / f"fold{fold}_{family}.pkl", "rb") as fh:
        return pickle.load(fh)


def explain_rows(
    run_dir: str | Path,
    row_positions: list[int],
    top_k: int = 5,
    families: tuple[str, ...] = (MODEL_LIGHTGBM, MODEL_PYTORCH_MLP),
) -> dict:
    """Explain specific rows using the fold model that held them out.

    Returns per-row: original feature values, true target, predictions of
    both model families (out-of-fold) and each model's top-k local SHAP
    contributions.
    """
    import json

    from .profiler import encode_target

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    task = TaskType(manifest["task_type"])
    config = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))

    df = pd.read_csv(next(run_dir.glob("dataset_source.*"))) if any(
        run_dir.glob("dataset_source.*")
    ) else pd.read_parquet(next(run_dir.glob("dataset_source.*")))
    df = df[df[config["target_column"]].notna()].reset_index(drop=True)
    y, classes = encode_target(df, config["target_column"], task)

    oof = pd.read_parquet(run_dir / "oof_predictions.parquet")
    base_features = [c for c in df.columns if c != config["target_column"]]

    bundle_cache: dict[tuple[str, int], FoldBundle] = {}
    out_rows = []
    for pos in row_positions:
        pos = int(pos)
        if not (0 <= pos < len(df)):
            out_rows.append({"position": pos, "error": "position out of range"})
            continue
        fold = int(oof["_fold"].iloc[pos])
        row_df = df.iloc[[pos]]
        entry: dict[str, Any] = {
            "position": pos,
            "fold": fold,
            "values": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                       for k, v in row_df.iloc[0].items()},
            "true_target": (int(y[pos]) if task.is_classification else float(y[pos])),
            "true_class": classes[int(y[pos])] if task.is_classification else None,
            "models": {},
        }
        for family in families:
            key = (family, fold)
            if key not in bundle_cache:
                bundle_cache[key] = _load_bundle(run_dir, family, fold)
            bundle = bundle_cache[key]
            try:
                proba = bundle.predict_matrix(row_df, task)[0]
                X_enc = bundle.transform(row_df)
                vals3 = compute_shap(bundle, X_enc)[0]  # (encoded_feat, outputs)
                signed = vals3.mean(axis=-1)
                bases = encoded_base_columns(bundle.prep)
                agg: dict[str, float] = {}
                for bcol, v in zip(bases, signed):
                    agg[bcol] = agg.get(bcol, 0.0) + float(v)
                ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
                if task is TaskType.REGRESSION:
                    pred = float(proba[0])
                elif task is TaskType.BINARY_CLASSIFICATION:
                    pred = int(proba[1] >= 0.5)
                else:
                    pred = int(np.argmax(proba))
                entry["models"][family] = {
                    "prediction": pred,
                    "probabilities": [round(float(x), 6) for x in np.atleast_1d(proba)],
                    "oof_prediction_from_run": (
                        float(oof[f"{family}_pred"].iloc[pos])
                        if f"{family}_pred" in oof.columns
                        else None
                    ),
                    "top_contributions": [
                        {"feature": f, "shap_contribution": round(v, 6)} for f, v in ranked
                    ],
                }
            except Exception as exc:  # keep partial results usable
                entry["models"][family] = {"error": f"{type(exc).__name__}: {exc}"}
        out_rows.append(entry)

    return {
        "run_id": manifest["run_id"],
        "task_type": manifest["task_type"],
        "classes": classes,
        "n_rows": len(out_rows),
        "rows": out_rows,
        "usage_hint": (
            "These are out-of-fold attributions from the model that held out each row. "
            "Full per-row matrices: see shap_values_gbdt.parquet / shap_values_nn.parquet "
            "in the run directory."
        ),
    }
