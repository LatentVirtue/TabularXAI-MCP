"""Dataset profiler and leakage guard.

Detects, before any model ever sees the data:
  * target leakage: single features with |r| > 0.95 against the target, or
    single-feature ROC-AUC > 0.98 (classification),
  * collinear feature clusters: connected groups of input features with
    pairwise |r| > 0.90 (SHAP will split credit across the proxies),
  * ID-like high-cardinality columns,
  * high-missingness columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import (
    ColumnProfile,
    DatasetProfile,
    LeakageFlag,
    TaskName,
    TaskType,
)

LEAK_R_THRESHOLD = 0.95
LEAK_AUC_THRESHOLD = 0.98
COLLINEARITY_THRESHOLD = 0.90
HIGH_MISSING_PCT = 30.0
ID_MIN_UNIQUE = 50
ID_UNIQUE_RATIO = 0.8
MULTICLASS_MAX_CLASSES = 20


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_dataset(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(p, sep=sep)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported input format '{suffix}'. Use .csv, .tsv or .parquet.")


def load_prepared_dataset(path: str | Path, target_column: str) -> tuple[pd.DataFrame, list[str]]:
    """Load a dataset and drop rows with a missing target (the one hard
    requirement for CV). Returns (df, notes)."""
    df = load_dataset(path)
    notes: list[str] = []
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found. Available: {list(df.columns)}"
        )
    n_before = len(df)
    df = df[df[target_column].notna()].reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped:
        notes.append(f"Dropped {dropped}/{n_before} rows with a missing target value.")
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} usable rows remain; need at least 30.")
    return df, notes


# --------------------------------------------------------------------------- #
# Task inference / target encoding
# --------------------------------------------------------------------------- #


def infer_task_type(df: pd.DataFrame, target_column: str) -> TaskType:
    s = df[target_column]
    n_unique = int(s.nunique(dropna=True))
    if n_unique == 2:
        return TaskType.BINARY_CLASSIFICATION
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_float_dtype(s) or n_unique > MULTICLASS_MAX_CLASSES:
            return TaskType.REGRESSION
        return TaskType.MULTICLASS_CLASSIFICATION
    # non-numeric target
    if n_unique <= MULTICLASS_MAX_CLASSES:
        return TaskType.MULTICLASS_CLASSIFICATION
    raise ValueError(
        f"Target '{target_column}' has {n_unique} non-numeric categories; "
        "use a drop_columns/exclusion strategy or a numeric encoding first."
    )


def encode_target(
    df: pd.DataFrame, target_column: str, task: TaskType
) -> tuple[np.ndarray, list[str]]:
    """Return (encoded_target, class_labels). Classification targets become
    integer codes with sorted class labels; regression stays float."""
    s = df[target_column]
    if not task.is_classification:
        return s.astype(float).to_numpy(), []
    labels = np.array(sorted(s.astype(str).unique()))
    lookup = {lab: i for i, lab in enumerate(labels)}
    codes = s.astype(str).map(lookup).to_numpy(dtype=int)
    return codes, [str(x) for x in labels]


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _single_feature_auc(x: np.ndarray, y: np.ndarray, task: TaskType) -> float:
    """Best achievable single-feature discrimination (as a raw score)."""
    best = 0.5
    try:
        if task is TaskType.BINARY_CLASSIFICATION:
            auc = roc_auc_score(y, x)
            best = max(best, auc, 1.0 - auc)
        else:
            for k in np.unique(y):
                auc = roc_auc_score((y == k).astype(int), x)
                best = max(best, auc, 1.0 - auc)
    except ValueError:
        return 0.5
    return float(best)


class _UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def profile_dataset(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: list[str] | None = None,
    task_type: TaskName = "auto",
) -> DatasetProfile:
    """Inspect a (target-complete) DataFrame and produce a DatasetProfile."""
    drop_columns = [c for c in (drop_columns or []) if c != target_column]
    warnings: list[str] = []

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    inferred = infer_task_type(df, target_column)
    if task_type == "auto":
        task = inferred
    else:
        task = TaskType(task_type)
        if task is not inferred:
            warnings.append(
                f"Explicit task_type='{task.value}' overrides inferred '{inferred.value}'."
            )

    n_rows, n_columns = df.shape
    y_enc, classes = encode_target(df, target_column, task)

    columns: list[ColumnProfile] = []
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    id_like: list[str] = []

    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        missing_pct = round(float(s.isna().mean()) * 100.0, 2)
        n_unique = int(s.nunique(dropna=True))
        if col == target_column:
            columns.append(
                ColumnProfile(
                    name=col, dtype=dtype, role="target", n_unique=n_unique, missing_pct=missing_pct
                )
            )
            continue
        if col in drop_columns:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
        ratio = n_unique / max(1, n_rows)
        # pandas 3 uses a dedicated `str` dtype; object columns may also hold
        # strings — classify both as categorical, never as ID-like.
        is_stringish = isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s)
        if not is_stringish and pd.api.types.is_object_dtype(s):
            vals = s.dropna().head(50).tolist()
            is_stringish = bool(vals) and all(isinstance(v, str) for v in vals)
        if is_stringish:
            role = "categorical_feature"
            categorical_features.append(col)
        elif pd.api.types.is_integer_dtype(s) and n_unique >= ID_MIN_UNIQUE and ratio > ID_UNIQUE_RATIO:
            role = "id_like"
            id_like.append(col)
        elif is_numeric:
            role = "numeric_feature"
            numeric_features.append(col)
        else:
            role = "id_like"  # unhandled dtype -> treat as unusable, keep flagged
            id_like.append(col)
        columns.append(
            ColumnProfile(
                name=col, dtype=dtype, role=role, n_unique=n_unique, missing_pct=missing_pct
            )
        )

    high_missing = {
        cp.name: cp.missing_pct
        for cp in columns
        if cp.missing_pct >= HIGH_MISSING_PCT and cp.role != "id_like"
    }
    for name, pct in high_missing.items():
        warnings.append(f"Column '{name}' is {pct:.1f}% missing (median/most-frequent imputed per fold).")
    if id_like:
        warnings.append(
            "ID-like column(s) detected and excluded from modeling: " + ", ".join(id_like)
        )

    # ---- leakage detection over numeric features ----
    leakage_flags: list[LeakageFlag] = []
    for col in numeric_features:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(x)
        if mask.sum() < 30:
            continue
        r = _pearson_r(x[mask], y_enc[mask])
        auc = _single_feature_auc(x[mask], y_enc, task) if task.is_classification else 0.5
        if abs(r) > LEAK_R_THRESHOLD:
            leakage_flags.append(
                LeakageFlag(
                    column=col,
                    reason=f"|Pearson r| = {abs(r):.3f} with target (> {LEAK_R_THRESHOLD})",
                    association=round(abs(r), 4),
                )
            )
        elif auc > LEAK_AUC_THRESHOLD:
            leakage_flags.append(
                LeakageFlag(
                    column=col,
                    reason=f"single-feature ROC-AUC = {auc:.3f} (> {LEAK_AUC_THRESHOLD})",
                    association=round(auc, 4),
                )
            )
    for flag in leakage_flags:
        warnings.append(
            f"potential_target_leakage: '{flag.column}' {flag.reason}. "
            "Metrics will be optimistic unless this column is dropped."
        )

    # ---- collinearity clusters over numeric features ----
    clusters: list[list[str]] = []
    if len(numeric_features) >= 2:
        corr = df[numeric_features].corr().to_numpy()
        uf = _UnionFind(numeric_features)
        n_f = len(numeric_features)
        for i in range(n_f):
            for j in range(i + 1, n_f):
                v = corr[i, j]
                if np.isfinite(v) and abs(v) > COLLINEARITY_THRESHOLD:
                    uf.union(numeric_features[i], numeric_features[j])
        groups: dict[str, list[str]] = {}
        for f in numeric_features:
            groups.setdefault(uf.find(f), []).append(f)
        clusters = sorted((g for g in groups.values() if len(g) >= 2), key=len, reverse=True)
    for cluster in clusters:
        warnings.append(
            "collinear_feature_cluster: "
            + ", ".join(sorted(cluster))
            + f" (pairwise |r| > {COLLINEARITY_THRESHOLD}) — SHAP credit may split across these proxies."
        )

    class_balance: dict[str, int] | None = None
    target_summary: dict[str, float] = {}
    if task.is_classification:
        counts = pd.Series(y_enc).value_counts().sort_index()
        class_balance = {classes[i]: int(counts.get(i, 0)) for i in range(len(classes))}
        minority = min(class_balance.values())
        if minority < 5 * int(2):  # extremely small classes are CV-hostile
            warnings.append(f"Minority class has only {minority} rows; stratified CV may be unstable.")
    else:
        t = df[target_column].astype(float)
        target_summary = {
            "mean": float(t.mean()),
            "std": float(t.std()),
            "min": float(t.min()),
            "max": float(t.max()),
        }

    return DatasetProfile(
        n_rows=n_rows,
        n_columns=n_columns,
        target_column=target_column,
        task_type=task,
        columns=columns,
        numeric_feature_columns=numeric_features,
        categorical_feature_columns=categorical_features,
        id_like_columns=id_like,
        potential_target_leakage=leakage_flags,
        collinear_feature_clusters=clusters,
        high_missing_columns=high_missing,
        class_balance=class_balance,
        target_summary=target_summary,
        warnings=warnings,
    )
