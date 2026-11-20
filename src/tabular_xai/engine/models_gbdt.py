"""LightGBM baseline factory (classification & regression)."""
from __future__ import annotations

import lightgbm as lgb

from .config import GBDTParams, TaskType


def make_gbdt(task: TaskType, params: GBDTParams, seed: int):
    """Fresh (unfitted) LightGBM estimator with deterministic settings."""
    common = dict(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        num_leaves=params.num_leaves,
        min_child_samples=params.min_child_samples,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=-1,
        verbosity=-1,
    )
    if task is TaskType.REGRESSION:
        return lgb.LGBMRegressor(**common)
    return lgb.LGBMClassifier(**common)


def gain_importance(estimator: lgb.sklearn.LGBMModel) -> list[float]:
    """Gain-based feature importances from a fitted LightGBM estimator."""
    booster = estimator.booster_
    return [float(v) for v in booster.feature_importance(importance_type="gain")]
