"""Builds the compact, strictly-typed `manifest.json` agent artifact.

The manifest is validated against the RunManifest schema and is guaranteed
compact (< ~2,000 estimated tokens): warnings are capped/truncated, feature
attribution rows are capped at 10, error indices at 5 per category.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..engine.config import (
    MODEL_LIGHTGBM,
    DatasetProfile,
    ExperimentConfig,
    FeatureAttribution,
    DatasetSection,
    ModelMetricsSection,
    RunManifest,
    TaskType,
    utc_now_iso,
)

MAX_WARNINGS = 10
WARNING_MAX_CHARS = 200


def _compact_warnings(warnings: list[str], extra: list[str]) -> list[str]:
    out = [str(w)[:WARNING_MAX_CHARS] for w in list(warnings) + list(extra)]
    if len(out) > MAX_WARNINGS:
        extra_n = len(out) - MAX_WARNINGS
        out = out[:MAX_WARNINGS]
        out.append(f"(+{extra_n} more warnings — see metrics.json)")
    return out


def _calibration_summary(per_model: dict, winner: str) -> dict:
    calib = dict(per_model[winner].get("calibration", {}))
    calib.pop("reliability_curve", None)  # kept in metrics.json / report only
    return calib


def build_manifest(
    run_id: str,
    task: TaskType,
    config: ExperimentConfig,
    profile: DatasetProfile,
    feature_cols: list[str],
    dropped: list[str],
    notes: list[str],
    per_model: dict,
    winner: str,
    winning_metric: str,
    rationale: str,
    worst: dict[str, list[int]],
    expl: dict,
    artifact_paths: dict[str, str],
) -> RunManifest:
    leakage = [f"potential_target_leakage: {f.column} ({f.reason})" for f in profile.potential_target_leakage]
    clusters = ["collinear_feature_cluster: " + ", ".join(c) for c in profile.collinear_feature_clusters]

    model_sections = {
        family: ModelMetricsSection(**blob) for family, blob in per_model.items()
    }
    top_features = [FeatureAttribution(**row) for row in expl["top_features"][:10]]

    manifest = RunManifest(
        run_id=run_id,
        created_at=utc_now_iso(),
        task_type=task,
        dataset=DatasetSection(
            path=str(Path(config.data_path).resolve()),
            n_rows=profile.n_rows,
            target_column=config.target_column,
            task_type=task,
            n_features=len(feature_cols),
            feature_columns=feature_cols,
            dropped_columns=dropped,
            notes=notes,
        ),
        dataset_warnings=_compact_warnings(leakage + clusters, notes),
        model_comparison=model_sections,
        winner=winner,
        winning_metric=winning_metric,
        winner_rationale=rationale,
        calibration_summary=_calibration_summary(per_model, winner),
        attribution_agreement_spearman=expl["attribution_agreement_spearman"],
        top_features=top_features,
        worst_error_indices=worst,
        artifact_paths=artifact_paths,
    )
    tokens = manifest.estimated_token_count()
    if tokens > 2000:
        raise RuntimeError(
            f"manifest.json would be ~{tokens} estimated tokens (> 2,000 budget). "
            "Reduce warnings/features surfaced — full detail belongs in metrics.json."
        )
    return manifest


def save_manifest(manifest: RunManifest, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_manifest(path: str | Path) -> RunManifest:
    return RunManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
