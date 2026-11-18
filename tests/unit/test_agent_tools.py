"""Unit tests for SAES-AI agent tools, agentic loop, and UI actions."""

from pathlib import Path

from fastapi.testclient import TestClient

from saes_ai.chat.agent import StatisticalChatAgent
from saes_ai.chat.tools import ToolContext, execute_tool
from saes_ai.server.app import create_app


def _sample_valid_csv() -> str:
    rows = ["algorithm,instance,metric,run_id,value,maximize"]
    for inst in ("ZDT1", "ZDT2"):
        for run_id in range(6):
            rows.append(f"NSGA-II,{inst},HV,{run_id},{0.80 + 0.01 * run_id},true")
            rows.append(f"MOEA/D,{inst},HV,{run_id},{0.72 + 0.01 * run_id},true")
    return "\n".join(rows) + "\n"


def test_create_list_inspect_edit_delete_dataset(tmp_path: Path) -> None:
    ctx = ToolContext(datasets_dir=tmp_path)

    # 1. Create dataset from structured rows
    created = execute_tool(
        ctx,
        "create_dataset",
        {
            "filename": "synth.csv",
            "csv_content": _sample_valid_csv(),
        },
    )
    assert created["ok"] is True
    assert created["saved"] is True
    assert created["is_valid"] is True
    assert any(a["type"] == "refresh_datasets" for a in ctx.ui_actions)
    assert any(a["type"] == "select_dataset" and a["dataset_name"] == "synth.csv" for a in ctx.ui_actions)

    # 2. Overwrite guard
    dup = execute_tool(ctx, "create_dataset", {"filename": "synth.csv", "csv_content": _sample_valid_csv()})
    assert dup["saved"] is False

    # 3. List datasets
    listed = execute_tool(ctx, "list_datasets", {})
    assert listed["datasets_count"] == 1
    assert listed["datasets"][0]["filename"] == "synth.csv"
    assert listed["datasets"][0]["is_valid"] is True

    # 4. Inspect saved dataset
    insp = execute_tool(ctx, "inspect_dataset", {"dataset_name": "synth.csv"})
    assert insp["is_valid"] is True
    assert insp["row_count"] == 24

    # 5. Delete dataset
    deleted = execute_tool(ctx, "delete_dataset", {"dataset_name": "synth.csv"})
    assert deleted["deleted"] is True
    assert execute_tool(ctx, "list_datasets", {})["datasets_count"] == 0


def test_inspect_and_repair_staged_invalid_csv(tmp_path: Path) -> None:
    bad_csv = (
        "algo,problem,indicator,run,score,max\n"
        "NSGA-II,ZDT1,HV,0,0.85,true\n"
        "NSGA-II,ZDT1,HV,1,0.87,true\n"
        "MOEA/D,ZDT1,HV,0,bad_num,true\n"
        "MOEA/D,ZDT1,HV,0,0.75,true\n"
        "MOEA/D,ZDT1,HV,1,0.77,true\n"
    )
    ctx = ToolContext(datasets_dir=tmp_path, staged_csv=bad_csv, staged_filename="raw_upload.csv")

    insp = execute_tool(ctx, "inspect_dataset", {"use_staged": True})
    assert insp["ok"] is True
    assert insp["is_valid"] is False
    assert set(insp["missing_required_columns"]) == {"algorithm", "instance", "metric", "run_id", "value", "maximize"}
    suggested = insp["suggested_column_mapping"]
    assert suggested["algo"] == "algorithm"
    assert suggested["score"] == "value"

    # Repair using edit_dataset
    edited = execute_tool(
        ctx,
        "edit_dataset",
        {
            "use_staged": True,
            "target_filename": "repaired.csv",
            "column_mapping": suggested,
            "drop_invalid_rows": True,
            "save": True,
        },
    )
    assert edited["ok"] is True
    assert edited["is_valid"] is True
    assert edited["row_count"] == 4
    assert (tmp_path / "repaired.csv").exists()


def test_kpis_explore_and_full_saes_tools(tmp_path: Path) -> None:
    ctx = ToolContext(datasets_dir=tmp_path)
    execute_tool(ctx, "create_dataset", {"filename": "bench.csv", "csv_content": _sample_valid_csv()})

    kpis = execute_tool(ctx, "get_dataset_kpis", {"dataset_name": "bench.csv"})
    assert kpis["ok"] is True
    assert kpis["algorithms_count"] == 2
    assert kpis["metric_kpis"]["HV"]["best_overall_by_mean"] == "NSGA-II"

    exp = execute_tool(ctx, "explore_dataset", {"dataset_name": "bench.csv"})
    assert exp["ok"] is True
    assert exp["normality_total_checks"] >= 2
    assert len(exp["recommendations"]) >= 1

    analysis = execute_tool(ctx, "run_statistical_analysis", {"dataset_name": "bench.csv"})
    assert analysis["ok"] is True
    assert analysis["cells_count"] == 2

    ranking = execute_tool(ctx, "run_ranking_analysis", {"dataset_name": "bench.csv", "metric": "HV"})
    assert ranking["ok"] is True
    assert len(ranking["ranks"]) == 2

    table = execute_tool(ctx, "generate_statistical_table", {"dataset_name": "bench.csv", "metric": "HV"})
    assert table["ok"] is True
    assert "NSGA-II" in table["markdown_preview"]

    prof = execute_tool(ctx, "compute_performance_profiles", {"dataset_name": "bench.csv", "metric": "HV"})
    assert prof["ok"] is True
    assert len(prof["global_meta_ranking"]) == 2

    pwr = execute_tool(ctx, "analyze_statistical_power", {"dataset_name": "bench.csv", "metric": "HV"})
    assert pwr["ok"] is True
    assert pwr["overall_power_health"]

    repro = execute_tool(ctx, "generate_reproducibility_package", {"dataset_name": "bench.csv"})
    assert repro["ok"] is True
    assert "saes" in repro["python_script_preview"].lower()

    nav = execute_tool(
        ctx,
        "navigate_ui",
        {"view": "visualize", "tab": "stats", "dataset_name": "bench.csv", "alpha": 0.01},
    )
    assert nav["ok"] is True
    assert ctx.ui_actions[-1]["tab"] == "stats"


def test_generate_plot_and_table_artifacts(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("SAES_DATASETS_DIR", str(tmp_path))  # type: ignore[attr-defined]

    csv_content = (
        "algorithm,instance,metric,run_id,value,maximize\n"
        "A1,ZDT1,HV,0,0.85,true\n"
        "A1,ZDT1,HV,1,0.87,true\n"
        "A1,ZDT1,HV,2,0.86,true\n"
        "A2,ZDT1,HV,0,0.75,true\n"
        "A2,ZDT1,HV,1,0.77,true\n"
        "A2,ZDT1,HV,2,0.76,true\n"
        "A1,ZDT2,HV,0,0.81,true\n"
        "A1,ZDT2,HV,1,0.82,true\n"
        "A1,ZDT2,HV,2,0.80,true\n"
        "A2,ZDT2,HV,0,0.71,true\n"
        "A2,ZDT2,HV,1,0.72,true\n"
        "A2,ZDT2,HV,2,0.70,true\n"
    )
    (tmp_path / "bench.csv").write_text(csv_content, encoding="utf-8")

    ctx = ToolContext(datasets_dir=tmp_path, active_dataset_name="bench.csv")

    plot_res = execute_tool(ctx, "generate_plot", {"plot_type": "boxplot", "metric": "HV"})
    assert plot_res["ok"] is True
    assert plot_res["displayed_in_chat"] is True
    assert len(ctx.artifacts) == 1
    plot_art = ctx.artifacts[0]
    assert plot_art["kind"] == "plot"
    assert plot_art["plot_type"] == "boxplot"
    if plot_art["image_data_url"] is not None:
        assert plot_art["image_data_url"].startswith("data:image/png;base64,")
    assert "data" in plot_art["plotly_figure"]
    assert "layout" in plot_art["plotly_figure"]
    assert plot_art["download_formats"] == ["png", "svg", "pdf"]

    tbl_res = execute_tool(ctx, "generate_statistical_table", {"metric": "HV"})
    assert tbl_res["ok"] is True
    assert tbl_res["displayed_in_chat"] is True
    assert len(ctx.artifacts) == 2
    tbl_art = ctx.artifacts[1]
    assert tbl_art["kind"] == "table"
    assert "|" in tbl_art["markdown"]
    assert "A1" in tbl_art["csv_content"]
    assert "pdf" in tbl_art["download_formats"]

    agent = StatisticalChatAgent(provider="local", tool_context=ctx)
    events = list(agent._run_tool_and_emit("generate_plot", {"plot_type": "critical-distance", "metric": "HV"}))
    assert any(e["type"] == "artifact" and e["artifact"]["plot_type"] == "critical-distance" for e in events)

    app = create_app()
    client = TestClient(app)
    res_chat = client.post(
        "/api/v1/chat",
        json={
            "query": "Show me a boxplot of this dataset",
            "dataset_name": "bench.csv",
            "provider": "local",
        },
    )
    assert res_chat.status_code == 200
