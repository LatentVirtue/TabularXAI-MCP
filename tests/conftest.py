"""Shared fixtures: tiny end-to-end benchmark runs reused across test modules.

Kept deliberately small (a few hundred rows, 3 folds, few epochs) so the
whole suite stays fast on CPU while still exercising the real pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tabular_xai.engine.config import GBDTParams, ExperimentConfig, NNParams
from tabular_xai.engine.synthetic import generate_dataset


@pytest.fixture(scope="session")
def tiny_benchmark(tmp_path_factory):
    """Factory running the real pipeline end-to-end on tiny synthetic data.

    Returns (manifest_dict, run_dir, csv_path) for the requested task.
    """

    def _run(task: str, rows: int = 260, n_splits: int = 3, epochs: int = 8, seed: int = 5, **cfg_overrides):
        from tabular_xai.engine.trainer import run_benchmark

        tmp = tmp_path_factory.mktemp(f"bench_{task}")
        df = generate_dataset(task=task, n_rows=rows, seed=seed)
        csv_path = tmp / "dataset.csv"
        df.to_csv(csv_path, index=False)
        run_id = f"test-{task}"
        cfg = ExperimentConfig(
            data_path=str(csv_path),
            target_column="target",
            task_type=task,
            n_splits=n_splits,
            seed=seed,
            output_dir=str(tmp / "runs"),
            run_id=run_id,
            gbdt=GBDTParams(n_estimators=200),
            nn=NNParams(epochs=epochs, hidden_dim=64, n_blocks=1, patience=5),
            max_shap_rows=120,
            **cfg_overrides,
        )
        manifest = run_benchmark(cfg)
        return manifest, tmp / "runs" / run_id, csv_path

    return _run
