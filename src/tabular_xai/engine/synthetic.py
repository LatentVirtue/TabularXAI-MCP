"""Realistic synthetic tabular benchmark generator.

Every generated dataset intentionally contains:
  * nonlinear feature interactions (x1*x2, quadratic, tanh saturation),
  * one redundant collinear proxy (`x1_proxy`, r≈0.93 with x1) so SHAP
    attribution splits across redundant proxies, exactly like real data,
  * one subtle leaky post-outcome sensor (`recent_readout`) that a naive
    pipeline will happily rank as the top feature,
  * a high-cardinality `record_id` column,
  * a noisy column with ~35% missingness,
  * and pure-noise decoys.

This gives the profiler and the arena something concrete to be tested on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import TaskType

LEAK_COLUMN = "recent_readout"
COLLINEAR_SOURCE = "x1"
COLLINEAR_PROXY = "x1_proxy"
ID_COLUMN = "record_id"
MISSING_COLUMN = "partial_feature"


def generate_dataset(
    task: str | TaskType = TaskType.BINARY_CLASSIFICATION,
    n_rows: int = 2000,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic dataset with planted structure for the given task."""
    task = TaskType(task)
    rng = np.random.default_rng(seed)
    n = n_rows

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    x4 = rng.normal(0, 1, n)
    x5 = rng.normal(0, 1, n)
    x6 = rng.normal(0, 1, n)
    x7 = rng.uniform(0, 10, n)
    cat_a = rng.choice(["alpha", "bravo", "charlie"], size=n)
    cat_b = rng.choice(["yes", "no"], size=n)

    cat_a_eff = {"alpha": -0.8, "bravo": 0.2, "charlie": 1.0}
    a = np.array([cat_a_eff[c] for c in cat_a])
    b = np.where(cat_b == "yes", 1.0, 0.0)

    common_signal = (
        1.1 * x1
        + 0.7 * x1 * x2
        + 0.8 * (x3**2 - 1.0)
        - 0.9 * x4
        + 0.35 * np.tanh(3.0 * (x7 - 5.0))
    )

    df = pd.DataFrame(
        {
            ID_COLUMN: rng.integers(100_000, 999_999, size=n),
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5,
            "x6": x6,
            "x7_score": x7,
            "cat_a": cat_a,
            "cat_b": cat_b,
            COLLINEAR_PROXY: 0.9 * x1 + rng.normal(0, 0.35, n),
            "x8_noise": rng.normal(0, 1, n),
            MISSING_COLUMN: rng.normal(0, 1, n),
            "x9_noise": rng.normal(0, 1, n),
        }
    )
    # Introduce ~35% missingness into one noise column (above the profiler's
    # high-missingness flag threshold so the guard demonstrably fires).
    miss_mask = rng.random(n) < 0.36
    df[MISSING_COLUMN] = df[MISSING_COLUMN].mask(miss_mask)

    if task is TaskType.BINARY_CLASSIFICATION:
        latent = common_signal - 0.9 * x5 + 0.8 * a - 0.6 * b + rng.normal(0, 0.8, n)
        p = 1.0 / (1.0 + np.exp(-latent))
        y = (rng.random(n) < p).astype(int)
        df["target"] = y
        # Post-outcome sensor: measured *after* the outcome is decided — near-copy of
        # the label itself, so single-feature ROC-AUC ≈ 0.999 and the guard fires.
        df[LEAK_COLUMN] = (2.0 * y - 1.0) + rng.normal(0, 0.18, n)
    elif task is TaskType.MULTICLASS_CLASSIFICATION:
        z0 = 1.0 * x1 + 0.6 * x1 * x2 + 0.9 * a + rng.normal(0, 0.6, n)
        z1 = -0.8 * x2 + 0.7 * (x3**2 - 1.0) + 0.6 * b + rng.normal(0, 0.6, n)
        z2 = 0.6 * x4 * x5 - 0.5 * x6 + 0.08 * (x7 - 5.0) ** 2 + rng.normal(0, 0.6, n)
        z = np.stack([z0, z1, z2], axis=1)
        probs = np.exp(z - z.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        y = np.array([rng.choice(3, p=p) for p in probs])
        df["target"] = y
        # Leaky sensor: reads the winning class code after the draw (post-outcome).
        df[LEAK_COLUMN] = y.astype(float) + rng.normal(0, 0.2, n)
    else:  # regression
        y = (
            common_signal * 1.8
            + 1.5 * a
            + 1.2 * b
            - 0.6 * x5
            + rng.normal(0, 1.1, n)
        )
        df["target"] = y
        # Leaky sensor: the outcome measured again with small error (r ≈ 0.96).
        df[LEAK_COLUMN] = y + rng.normal(0, 0.35, n)

    return df


def write_csv(
    path: str | Path,
    task: str | TaskType = TaskType.BINARY_CLASSIFICATION,
    n_rows: int = 2000,
    seed: int = 7,
) -> Path:
    """Generate the dataset and write it as CSV. Returns the written path."""
    df = generate_dataset(task=task, n_rows=n_rows, seed=seed)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p
