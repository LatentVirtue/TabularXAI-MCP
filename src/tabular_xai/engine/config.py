"""Typed schemas for experiment configuration, dataset profiles and run manifests.

Strict rule: this module depends on pydantic/stdlib/PyYAML only. It never
imports numpy, pandas, torch, lightgbm, shap or the MCP SDK, so it can be
imported cheaply by thin clients, tests and tooling.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field

MODEL_LIGHTGBM = "lightgbm"
MODEL_PYTORCH_MLP = "pytorch_mlp"
MODEL_FAMILIES: tuple[str, ...] = (MODEL_LIGHTGBM, MODEL_PYTORCH_MLP)

TaskName = Literal["auto", "binary_classification", "multiclass_classification", "regression"]


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        return self in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)


#: Headline metric used to pick the arena winner (evaluated on pooled OOF predictions).
PRIMARY_METRIC: dict[TaskType, str] = {
    TaskType.BINARY_CLASSIFICATION: "roc_auc",
    TaskType.MULTICLASS_CLASSIFICATION: "macro_f1",
    TaskType.REGRESSION: "rmse",
}

_LOWER_IS_BETTER = {"rmse", "mae", "log_loss", "brier", "ece", "mean_abs_residual"}


def metric_higher_is_better(metric: str) -> bool:
    return metric not in _LOWER_IS_BETTER


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #


class GBDTParams(BaseModel):
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 10


class NNParams(BaseModel):
    epochs: int = 40
    hidden_dim: int = 128
    n_blocks: int = 2
    dropout: float = 0.1
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    patience: int = 10


class ExperimentConfig(BaseModel):
    """Full definition of one benchmark run. Accepts dict/JSON/YAML input."""

    data_path: str
    target_column: str
    task_type: TaskName = "auto"
    n_splits: int = Field(default=5, ge=2, le=10)
    seed: int = 42
    drop_columns: list[str] = Field(default_factory=list)
    auto_drop_id_columns: bool = True
    #: When the leakage guard fires, exclude those columns from modelling too.
    #: They are still profiled and reported — only training use is disabled.
    auto_drop_leakage: bool = False
    output_dir: str = "runs"
    run_id: Optional[str] = None
    gbdt: GBDTParams = Field(default_factory=GBDTParams)
    nn: NNParams = Field(default_factory=NNParams)
    #: Cap on rows for which per-sample SHAP matrices are computed.
    max_shap_rows: int = Field(default=2000, ge=16)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        else:
            data = yaml.safe_load(text)  # JSON is valid YAML
        return cls.model_validate(data)

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:6]}"


# --------------------------------------------------------------------------- #
# Dataset profile (profiler output)
# --------------------------------------------------------------------------- #

ColumnRole = Literal["target", "numeric_feature", "categorical_feature", "id_like"]


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    role: ColumnRole
    n_unique: int
    missing_pct: float


class LeakageFlag(BaseModel):
    column: str
    reason: str
    association: float  # |Pearson r| or single-feature ROC-AUC, per `reason`


class DatasetProfile(BaseModel):
    n_rows: int
    n_columns: int
    target_column: str
    task_type: TaskType
    columns: list[ColumnProfile]
    numeric_feature_columns: list[str] = Field(default_factory=list)
    categorical_feature_columns: list[str] = Field(default_factory=list)
    id_like_columns: list[str] = Field(default_factory=list)
    potential_target_leakage: list[LeakageFlag] = Field(default_factory=list)
    collinear_feature_clusters: list[list[str]] = Field(default_factory=list)
    high_missing_columns: dict[str, float] = Field(default_factory=dict)
    class_balance: Optional[dict[str, int]] = None
    target_summary: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Run manifest (the compact machine/agent artifact)
# --------------------------------------------------------------------------- #


class DatasetSection(BaseModel):
    path: str
    n_rows: int
    target_column: str
    task_type: TaskType
    n_features: int
    feature_columns: list[str]
    dropped_columns: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModelMetricsSection(BaseModel):
    per_fold: dict[str, list[float]] = Field(default_factory=dict)
    mean: dict[str, float] = Field(default_factory=dict)
    std: dict[str, float] = Field(default_factory=dict)
    oof: dict[str, float] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)
    fit_seconds: float = 0.0
    n_parameters: Optional[int] = None


class FeatureAttribution(BaseModel):
    feature: str
    gbdt_gain_importance: float = 0.0
    gbdt_mean_abs_shap: float = 0.0
    nn_mean_abs_shap: float = 0.0


class RunManifest(BaseModel):
    """Compact (< ~2,000 tokens) digest an LLM agent can act on directly."""

    schema_version: str = "1.0"
    run_id: str
    created_at: str
    task_type: TaskType
    dataset: DatasetSection
    dataset_warnings: list[str] = Field(default_factory=list)
    model_comparison: dict[str, ModelMetricsSection] = Field(default_factory=dict)
    winner: str = ""
    winning_metric: str = ""
    winner_rationale: str = ""
    calibration_summary: dict[str, Any] = Field(default_factory=dict)
    attribution_agreement_spearman: Optional[float] = None
    top_features: list[FeatureAttribution] = Field(default_factory=list, max_length=10)
    worst_error_indices: dict[str, list[int]] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)

    def estimated_token_count(self) -> int:
        """Cheap proxy for LLM tokens: ~4 characters/token. Keep below 2,000."""
        return math.ceil(len(self.model_dump_json()) / 4)
