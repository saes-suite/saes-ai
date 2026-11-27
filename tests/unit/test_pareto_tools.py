"""Unit tests for SAES-AI Pareto & EAF plugin tools and navigation actions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from saes_ai.chat.tools import TOOL_REGISTRY, ToolContext, execute_tool
from saes_ai.mcp.tools import SAES_MCP_TOOLS

try:
    HAS_PARETO_PLUGIN = importlib.util.find_spec("saes.plugins.pareto") is not None
except ModuleNotFoundError:
    HAS_PARETO_PLUGIN = False


def test_pareto_tools_end_to_end(tmp_path: Path) -> None:
    ctx = ToolContext(datasets_dir=tmp_path)

    if not HAS_PARETO_PLUGIN:
        # Graceful degradation when running against saes < 2.1.0 without saes.plugins.pareto
        res = execute_tool(ctx, "list_pareto_studies", {})
        assert res["ok"] is False
        assert "saes.plugins.pareto" in res["error"]
        pytest.skip("Installed saes version does not include saes.plugins.pareto yet")

    # 1. Create demo study and list studies
    res_list = execute_tool(ctx, "list_pareto_studies", {"create_demo": True, "demo_name": "ai_demo"})
    assert res_list["ok"] is True
    assert res_list["count"] == 1
    assert res_list["studies"][0]["name"] == "ai_demo"

    # 2. Inspect study summary
    res_inspect = execute_tool(ctx, "inspect_pareto_study", {"study_name": "ai_demo"})
    assert res_inspect["ok"] is True
    assert "ZDT1" in res_inspect["instances"]
    assert "DTLZ2" in res_inspect["instances"]

    # 3. Generate inline Pareto attainment plot artifact
    res_plot = execute_tool(
        ctx,
        "generate_pareto_plot",
        {"study_name": "ai_demo", "instance": "ZDT1", "mode": "attainment"},
    )
    assert res_plot["ok"] is True
    assert res_plot["rendered_inline"] is True
    assert any(a.get("plot_type") == "pareto:attainment" for a in ctx.artifacts)

    # 4. Run pairwise EAF difference + KS-EAF permutation test (with inline heatmap artifact)
    res_eaf = execute_tool(
        ctx,
        "run_pareto_eaf_comparison",
        {
            "study_name": "ai_demo",
            "instance": "ZDT1",
            "algo_a": "NSGA-II",
            "algo_b": "MOEA/D",
            "n_permutations": 40,
            "include_plot": True,
        },
    )
    assert res_eaf["ok"] is True
    assert 0.0 <= res_eaf["ks_statistic"] <= 1.0
    assert 0.0 < res_eaf["p_value"] <= 1.0
    assert any(a.get("plot_type") == "pareto:eaf-diff" for a in ctx.artifacts)

    # 5. Compute Set Coverage C(A, B) & Representative Runs
    res_cov = execute_tool(ctx, "run_pareto_coverage", {"study_name": "ai_demo", "instance": "ZDT1"})
    assert res_cov["ok"] is True
    assert len(res_cov["representative_runs"]) == 4

    # 6. Extract quality indicators into canonical SAES CSV and run ranking on it!
    res_ext = execute_tool(
        ctx,
        "extract_pareto_indicators",
        {
            "study_name": "ai_demo",
            "dataset_filename": "ai_extracted.csv",
            "indicators": ["HV", "IGD+", "EPS"],
        },
    )
    assert res_ext["ok"] is True
    assert res_ext["filename"] == "ai_extracted.csv"
    assert (tmp_path / "ai_extracted.csv").is_file()
    assert any(a.get("type") == "refresh_datasets" for a in ctx.ui_actions)

    res_rank = execute_tool(
        ctx,
        "run_ranking_analysis",
        {"dataset_name": "ai_extracted.csv", "metric": "IGD+"},
    )
    assert res_rank["ok"] is True
    assert "friedman_p_value" in res_rank
    assert len(res_rank["ranks"]) == 4


def test_pareto_navigation_and_registry(tmp_path: Path) -> None:
    ctx = ToolContext(datasets_dir=tmp_path)
    res_nav = execute_tool(
        ctx,
        "navigate_ui",
        {"view": "pareto-visualize", "pareto_study": "ai_demo"},
    )
    assert res_nav["ok"] is True
    assert res_nav["action"]["view"] == "pareto-visualize"
    assert res_nav["action"]["pareto_study"] == "ai_demo"

    for name in (
        "list_pareto_studies",
        "inspect_pareto_study",
        "generate_pareto_plot",
        "run_pareto_eaf_comparison",
        "run_pareto_coverage",
        "extract_pareto_indicators",
    ):
        assert name in TOOL_REGISTRY

    mcp_names = {t.name for t in SAES_MCP_TOOLS}
    assert {"run_pareto_eaf_comparison", "run_pareto_coverage", "extract_pareto_indicators"}.issubset(mcp_names)
