"""Direct invocation of the MCP server's tool functions and resource template,
plus registration checks against the FastMCP app. No MCP client transport
needed — FastMCP's @tool decorator returns the original function.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from tabular_xai.mcp_server import (
    explain_specific_rows,
    mcp,
    profile_tabular_dataset,
    run_manifest_resource,
    run_tabular_benchmark,
)
from tabular_xai.engine.synthetic import LEAK_COLUMN, generate_dataset


def test_tools_and_resource_are_registered():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert {
        "profile_tabular_dataset",
        "run_tabular_benchmark",
        "explain_specific_rows",
    } <= names
    templates = asyncio.run(mcp.list_resource_templates())
    assert any("{run_id}" in t.uriTemplate for t in templates)
    # every tool must carry a description for LLM clients
    for t in tools:
        assert t.description and len(t.description) > 50


def test_profile_tool_flags_pathologies(tmp_path):
    df = generate_dataset("binary_classification", n_rows=400, seed=3)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    out = profile_tabular_dataset(str(csv), "target")
    assert out["task_type"] == "binary_classification"
    assert any(LEAK_COLUMN == f["column"] for f in out["potential_target_leakage"])
    assert out["collinear_feature_clusters"]
    assert "usage_hint" in out


def test_run_and_explain_and_resource_flow(tmp_path, monkeypatch):
    """Full agent flow: profile -> run -> explain worst -> read manifest resource."""
    df = generate_dataset("binary_classification", n_rows=220, seed=4)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    runs_dir = tmp_path / "runs"

    manifest = run_tabular_benchmark(
        data_path=str(csv),
        target_column="target",
        task_type="binary_classification",
        output_dir=str(runs_dir),
        n_splits=3,
        nn_epochs=5,
    )
    assert manifest["run_id"]
    assert "report.html" in manifest["usage_hint"] or manifest["artifact_paths"]["report_html"]
    from pathlib import Path
    assert Path(manifest["artifact_paths"]["report_html"]).exists()

    explained = explain_specific_rows(manifest["run_id"], mode="worst_errors",
                                      output_dir=str(runs_dir))
    assert explained["n_rows"] >= 1
    for row in explained["rows"]:
        assert "values" in row and "models" in row

    explicit = explain_specific_rows(manifest["run_id"], mode="explicit",
                                     row_indices=[0, 1], top_k_features=3,
                                     output_dir=str(runs_dir))
    assert explicit["n_rows"] == 2

    with pytest.raises(ValueError):
        explain_specific_rows("no-such-run", output_dir=str(runs_dir))
    with pytest.raises(ValueError):
        explain_specific_rows(manifest["run_id"], mode="explicit", output_dir=str(runs_dir))

    # 'latest' resolves inside a runs directory
    latest = explain_specific_rows("latest", mode="explicit", row_indices=[0],
                                   output_dir=str(runs_dir))
    assert latest["run_id"] == manifest["run_id"]

    # resource: serve raw manifest text from CWD-relative runs/
    monkeypatch.chdir(tmp_path)
    raw = run_manifest_resource(manifest["run_id"])
    parsed = json.loads(raw)
    assert parsed["run_id"] == manifest["run_id"]
    assert parsed["winner"] in ("lightgbm", "pytorch_mlp")
