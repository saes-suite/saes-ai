"""Agent tool registry and execution engine for SAES-AI.

Every tool delegates domain computation to the `saes` Python API and
returns structured JSON-serializable dictionaries for LLM consumption,
while optionally emitting structured UI actions for `saes-ui`.
"""

from __future__ import annotations

import base64
import importlib
import inspect
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import saes
from saes.domain.models import Dataset
from saes.io.csv_reader import CSVReader
from saes.stats.ranking import BayesianRankingTest, FriedmanNemenyiTest
from saes.validation.validator import REQUIRED_COLUMNS, DatasetValidator

MAX_SAMPLE_ROWS = 12
MAX_ISSUES_IN_RESPONSE = 25


@dataclass
class ToolContext:
    """Runtime context passed to every agent tool invocation."""

    datasets_dir: Path
    active_dataset_name: str | None = None
    staged_csv: str | None = None
    staged_filename: str | None = None
    alpha: float = 0.05
    current_view: str | None = None
    current_tab: str | None = None
    ui_actions: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def emit_ui_action(self, action: dict[str, Any]) -> None:
        """Queue a UI action to be sent to the frontend client."""
        self.ui_actions.append(action)

    def emit_artifact(self, artifact: dict[str, Any]) -> None:
        """Queue a rich inline artifact (plot, diagram, image, table) for the chat UI."""
        self.artifacts.append(artifact)


@dataclass
class ToolSpec:
    """Provider-agnostic specification of an agent tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[ToolContext, dict[str, Any]], dict[str, Any]]

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI / Ollama tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert to Anthropic Claude tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_gemini_declaration(self) -> dict[str, Any]:
        """Convert to Google Gemini functionDeclaration format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _filter_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Forward only kwargs supported by the installed saes callable signature."""
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _import_saes_server_helper(name: str, feature_label: str) -> Callable[..., Any]:
    """Dynamically resolve a saes >= 2.0.0 orchestration helper without breaking 0.3.x builds."""
    try:
        mod = importlib.import_module("saes.server.helpers")
        fn: Callable[..., Any] | None = getattr(mod, name, None)
    except ImportError:
        fn = None
    if fn is None:
        raise RuntimeError(f"{feature_label} requires saes >= 2.0.0.")
    return fn


def _sanitize_csv_filename(name: str) -> str:
    safe = Path(name.strip()).name
    if not safe:
        safe = "dataset.csv"
    if not safe.lower().endswith(".csv"):
        safe = f"{safe}.csv"
    return safe


def _resolve_dataset_csv(
    ctx: ToolContext,
    dataset_name: str | None = None,
    use_staged: bool = False,
) -> tuple[str, str, Path | None]:
    """Resolve CSV text and filename from staged upload, explicit name, or active dataset."""
    if use_staged and ctx.staged_csv is not None:
        return ctx.staged_csv, ctx.staged_filename or "uploaded_dataset.csv", None

    target = dataset_name or ctx.active_dataset_name
    if target:
        safe = _sanitize_csv_filename(target)
        file_path = ctx.datasets_dir / safe
        if file_path.exists() and file_path.is_file():
            return file_path.read_text(encoding="utf-8", errors="replace"), safe, file_path
        if ctx.staged_csv is not None and (ctx.staged_filename == target or ctx.staged_filename == safe):
            return ctx.staged_csv, safe, None
        raise FileNotFoundError(f"Dataset '{safe}' not found in SAES storage.")

    if ctx.staged_csv is not None:
        return ctx.staged_csv, ctx.staged_filename or "uploaded_dataset.csv", None

    raise ValueError(
        "No dataset specified. Pass `dataset_name` (use `list_datasets` to see available datasets) "
        "or ask the user which dataset they want to use."
    )


def _load_validated_dataset(
    ctx: ToolContext,
    dataset_name: str | None = None,
) -> tuple[Dataset, str]:
    """Load and validate a dataset via `saes`."""
    csv_text, fname, _ = _resolve_dataset_csv(ctx, dataset_name=dataset_name, use_staged=False)
    dataset = saes.load(io.StringIO(csv_text))
    report = saes.validate(dataset)
    if not report.is_valid:
        errs = "; ".join(e.message for e in report.errors[:5])
        raise ValueError(
            f"Dataset '{fname}' has validation errors and cannot be analyzed yet: {errs}. "
            "Use `inspect_dataset` and `edit_dataset` to fix it first."
        )
    return dataset, fname


def _suggest_column_mapping(columns: list[str]) -> dict[str, str]:
    """Heuristic suggestion mapping arbitrary CSV headers to SAES required columns."""
    aliases: dict[str, list[str]] = {
        "algorithm": ["algorithm", "algo", "alg", "solver", "method", "model", "approach"],
        "instance": ["instance", "problem", "benchmark", "task", "dataset", "function", "domain"],
        "metric": ["metric", "indicator", "measure", "kpi", "objective", "score_type"],
        "run_id": ["run_id", "run", "seed", "rep", "repetition", "iteration", "trial", "exec"],
        "value": ["value", "score", "result", "fitness", "performance", "val", "metric_value"],
        "maximize": ["maximize", "max", "higher_is_better", "direction", "is_max"],
    }
    lower_to_orig = {c.strip().lower(): c for c in columns}
    mapping: dict[str, str] = {}
    used_sources: set[str] = set()

    for req_col, candidates in aliases.items():
        for cand in candidates:
            if cand in lower_to_orig and lower_to_orig[cand] not in used_sources:
                src = lower_to_orig[cand]
                mapping[src] = req_col
                used_sources.add(src)
                break
    return mapping


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------


def _tool_list_datasets(ctx: ToolContext, _args: dict[str, Any]) -> dict[str, Any]:
    ctx.datasets_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(ctx.datasets_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    items: list[dict[str, Any]] = []

    for f in files:
        stat = f.stat()
        entry: dict[str, Any] = {
            "filename": f.name,
            "size_bytes": stat.st_size,
        }
        try:
            df = CSVReader().read(f)
            val = DatasetValidator().validate(df)
            entry["row_count"] = int(len(df))
            entry["is_valid"] = bool(val.is_valid)
            entry["error_count"] = len(val.errors)
            entry["warning_count"] = len(val.warnings)
            if "algorithm" in df.columns:
                entry["algorithms"] = sorted(str(x) for x in df["algorithm"].dropna().unique())[:10]
            if "instance" in df.columns:
                entry["instances_count"] = int(df["instance"].dropna().nunique())
            if "metric" in df.columns:
                entry["metrics"] = sorted(str(x) for x in df["metric"].dropna().unique())[:10]
        except Exception as exc:
            entry["is_valid"] = False
            entry["parse_error"] = str(exc)
        items.append(entry)

    result: dict[str, Any] = {
        "datasets_count": len(items),
        "datasets": items,
    }
    if ctx.staged_csv is not None:
        result["staged_upload"] = {
            "filename": ctx.staged_filename or "uploaded_dataset.csv",
            "bytes": len(ctx.staged_csv.encode("utf-8", errors="ignore")),
        }
    return result


def _tool_inspect_dataset(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset_name = args.get("dataset_name")
    use_staged = bool(args.get("use_staged", False))
    csv_text, fname, file_path = _resolve_dataset_csv(ctx, dataset_name=dataset_name, use_staged=use_staged)

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        return {
            "filename": fname,
            "is_valid": False,
            "parse_error": f"Failed to parse CSV: {exc!s}",
            "raw_preview": csv_text[:800],
        }

    columns = [str(c) for c in df.columns]
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in columns]
    extra_cols = [c for c in columns if c not in REQUIRED_COLUMNS]

    val_report = DatasetValidator().validate(df)
    issues = [
        {
            "severity": i.severity,
            "issue_type": i.issue_type,
            "message": i.message,
            "column": i.column,
            "row_index": i.row_index,
        }
        for i in val_report.issues[:MAX_ISSUES_IN_RESPONSE]
    ]

    sample_rows = json.loads(df.head(MAX_SAMPLE_ROWS).to_json(orient="records"))

    response: dict[str, Any] = {
        "filename": fname,
        "source": "saved_storage" if file_path is not None else "staged_chat_upload",
        "row_count": int(len(df)),
        "columns": columns,
        "required_columns": REQUIRED_COLUMNS,
        "missing_required_columns": missing_cols,
        "extra_columns": extra_cols,
        "is_valid": bool(val_report.is_valid),
        "error_count": len(val_report.errors),
        "warning_count": len(val_report.warnings),
        "issues": issues,
        "sample_rows": sample_rows,
    }
    if missing_cols and len(columns) >= len(REQUIRED_COLUMNS):
        response["suggested_column_mapping"] = _suggest_column_mapping(columns)
    return response


def _tool_create_dataset(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    filename = _sanitize_csv_filename(str(args.get("filename", "generated_benchmark.csv")))
    overwrite = bool(args.get("overwrite", False))
    csv_content = args.get("csv_content")
    rows = args.get("rows")

    target_path = ctx.datasets_dir / filename
    if target_path.exists() and not overwrite:
        return {
            "saved": False,
            "error": f"Dataset '{filename}' already exists. Pass overwrite=true or choose a new filename.",
        }

    if isinstance(rows, list) and rows:
        df = pd.DataFrame(rows)
        csv_text = df.to_csv(index=False)
    elif isinstance(csv_content, str) and csv_content.strip():
        csv_text = csv_content.strip() + "\n"
        df = pd.read_csv(io.StringIO(csv_text))
    else:
        return {
            "saved": False,
            "error": "Provide either `csv_content` (CSV string) or `rows` (list of row objects).",
        }

    val_report = DatasetValidator().validate(df)
    ctx.datasets_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_text(csv_text, encoding="utf-8")
    ctx.active_dataset_name = filename

    ctx.emit_ui_action({"type": "refresh_datasets"})

    return {
        "saved": True,
        "filename": filename,
        "row_count": int(len(df)),
        "is_valid": bool(val_report.is_valid),
        "error_count": len(val_report.errors),
        "warning_count": len(val_report.warnings),
        "issues": [
            {"severity": i.severity, "issue_type": i.issue_type, "message": i.message}
            for i in val_report.issues[:MAX_ISSUES_IN_RESPONSE]
        ],
    }


def _tool_edit_dataset(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset_name = args.get("dataset_name")
    use_staged = bool(args.get("use_staged", False))
    csv_text, src_filename, _ = _resolve_dataset_csv(ctx, dataset_name=dataset_name, use_staged=use_staged)

    target_filename = _sanitize_csv_filename(str(args.get("target_filename") or src_filename))
    save = bool(args.get("save", True))

    new_csv_content = args.get("new_csv_content")
    if isinstance(new_csv_content, str) and new_csv_content.strip():
        df = pd.read_csv(io.StringIO(new_csv_content))
    else:
        df = pd.read_csv(io.StringIO(csv_text))

    applied_edits: list[str] = []

    # 1. Column mapping / renaming
    column_mapping = args.get("column_mapping")
    if isinstance(column_mapping, dict) and column_mapping:
        df = df.rename(columns={str(k): str(v) for k, v in column_mapping.items()})
        applied_edits.append(f"Renamed columns: {column_mapping}")

    # 2. Keep only required columns if requested
    if args.get("drop_extra_columns"):
        keep = [c for c in REQUIRED_COLUMNS if c in df.columns]
        df = df[keep]
        applied_edits.append("Dropped non-SAES extra columns")

    # 3. Fill missing values or add missing default columns
    fill_missing = args.get("fill_missing")
    if isinstance(fill_missing, dict):
        for col, fill_val in fill_missing.items():
            if col not in df.columns:
                df[col] = fill_val
                applied_edits.append(f"Added column '{col}' with default {fill_val!r}")
            else:
                df[col] = df[col].fillna(fill_val)
                applied_edits.append(f"Filled missing values in '{col}' with {fill_val!r}")

    # 4. Specific row updates
    row_updates = args.get("row_updates")
    if isinstance(row_updates, list):
        for upd in row_updates:
            if not isinstance(upd, dict) or "row_index" not in upd:
                continue
            ridx = int(upd["row_index"])
            if 0 <= ridx < len(df):
                for k, v in upd.items():
                    if k != "row_index" and k in df.columns:
                        df.at[ridx, k] = v
                applied_edits.append(f"Updated row {ridx}")

    # 5. Drop invalid / null rows in required columns
    if args.get("drop_invalid_rows"):
        before_n = len(df)
        req_present = [c for c in REQUIRED_COLUMNS if c in df.columns]
        if req_present:
            df = df.dropna(subset=req_present)
        if "value" in df.columns:
            numeric_vals = pd.to_numeric(df["value"], errors="coerce")
            df = df[numeric_vals.notna() & np.isfinite(numeric_vals)]
            df = df.copy()
            df["value"] = pd.to_numeric(df["value"])
        if "run_id" in df.columns:
            run_vals = pd.to_numeric(df["run_id"], errors="coerce")
            df = df[run_vals.notna() & (run_vals >= 0) & (run_vals == run_vals.astype("Int64"))]
            df = df.copy()
            df["run_id"] = pd.to_numeric(df["run_id"]).astype(int)
        df = df.reset_index(drop=True)
        applied_edits.append(f"Dropped {before_n - len(df)} invalid row(s)")

    val_report = DatasetValidator().validate(df)
    updated_csv = df.to_csv(index=False)

    if use_staged or ctx.staged_csv is not None and src_filename == ctx.staged_filename:
        ctx.staged_csv = updated_csv
        ctx.staged_filename = target_filename

    saved_path: str | None = None
    if save:
        ctx.datasets_dir.mkdir(parents=True, exist_ok=True)
        out_path = ctx.datasets_dir / target_filename
        out_path.write_text(updated_csv, encoding="utf-8")
        saved_path = str(out_path)
        ctx.active_dataset_name = target_filename
        ctx.emit_ui_action({"type": "refresh_datasets"})

    return {
        "filename": target_filename,
        "saved": save,
        "saved_path": saved_path,
        "applied_edits": applied_edits,
        "row_count": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "is_valid": bool(val_report.is_valid),
        "error_count": len(val_report.errors),
        "warning_count": len(val_report.warnings),
        "remaining_issues": [
            {
                "severity": i.severity,
                "issue_type": i.issue_type,
                "message": i.message,
                "column": i.column,
                "row_index": i.row_index,
            }
            for i in val_report.issues[:MAX_ISSUES_IN_RESPONSE]
        ],
    }


def _tool_delete_dataset(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    safe = _sanitize_csv_filename(str(args["dataset_name"]))
    file_path = ctx.datasets_dir / safe
    if not file_path.exists():
        return {"deleted": False, "error": f"Dataset '{safe}' does not exist."}
    file_path.unlink()
    if ctx.active_dataset_name == safe:
        ctx.active_dataset_name = None
    ctx.emit_ui_action({"type": "refresh_datasets"})
    return {"deleted": True, "filename": safe}


def _tool_get_dataset_kpis(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    df = dataset.df

    algorithms = sorted(str(a) for a in df["algorithm"].unique())
    instances = sorted(str(i) for i in df["instance"].unique())
    metrics = sorted(str(m) for m in df["metric"].unique())

    cell_counts = df.groupby(["instance", "metric", "algorithm"]).size()
    min_runs = int(cell_counts.min()) if not cell_counts.empty else 0
    max_runs = int(cell_counts.max()) if not cell_counts.empty else 0

    metric_kpis: dict[str, Any] = {}
    for met in metrics:
        sub = df[df["metric"] == met]
        maximize = bool(sub["maximize"].iloc[0])
        agg = sub.groupby("algorithm")["value"].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
        agg = agg.sort_values("mean", ascending=not maximize)

        # Win count across instances
        inst_means = sub.groupby(["instance", "algorithm"])["value"].mean().reset_index()
        wins: dict[str, int] = {a: 0 for a in algorithms}
        for _, grp in inst_means.groupby("instance"):
            best_val = grp["value"].max() if maximize else grp["value"].min()
            winners = grp[grp["value"] == best_val]["algorithm"].tolist()
            for w in winners:
                wins[str(w)] = wins.get(str(w), 0) + 1

        algo_stats = []
        for _, row in agg.iterrows():
            algo_name = str(row["algorithm"])
            algo_stats.append(
                {
                    "algorithm": algo_name,
                    "n": int(row["count"]),
                    "mean": round(float(row["mean"]), 6),
                    "std": round(float(row["std"]) if pd.notna(row["std"]) else 0.0, 6),
                    "median": round(float(row["median"]), 6),
                    "min": round(float(row["min"]), 6),
                    "max": round(float(row["max"]), 6),
                    "instance_wins": wins.get(algo_name, 0),
                }
            )

        metric_kpis[met] = {
            "maximize": maximize,
            "best_overall_by_mean": algo_stats[0]["algorithm"] if algo_stats else None,
            "algorithms": algo_stats,
        }

    return {
        "filename": fname,
        "total_rows": int(len(df)),
        "algorithms_count": len(algorithms),
        "algorithms": algorithms,
        "instances_count": len(instances),
        "instances": instances,
        "metrics_count": len(metrics),
        "metrics": metrics,
        "runs_per_cell_min": min_runs,
        "runs_per_cell_max": max_runs,
        "is_balanced_design": min_runs == max_runs,
        "metric_kpis": metric_kpis,
    }


def _tool_explore_dataset(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    metrics = args.get("metrics")
    alpha = float(args.get("alpha", ctx.alpha))
    exp = saes.explore(dataset, metrics=metrics, alpha=alpha)
    diag = exp.context_diagnostics

    normality_summary: list[dict[str, Any]] = [
        {
            "metric": dp.metric,
            "instance": dp.instance,
            "algorithm": dp.algorithm,
            "mean": round(float(dp.mean), 6),
            "std": round(float(dp.std), 6),
            "median": round(float(dp.median), 6),
            "iqr": round(float(dp.iqr), 6),
            "shapiro_p": getattr(dp, "shapiro_p", getattr(dp, "shapiro_p_value", None)),
            "anderson_p": getattr(dp, "anderson_p", None),
            "is_normal": getattr(dp, "is_normal", None),
        }
        for dp in exp.distribution_profiles[:30]
    ]

    recs = [
        {
            "test_id": r.test_id,
            "test_name": r.test_name,
            "category": r.category,
            "suitability": r.suitability,
            "suitability_score": r.suitability_score,
            "suggested_table_type": r.suggested_table_type,
            "target_tab": getattr(r, "target_tab", "stats"),
            "target_mode": getattr(r, "target_mode", ""),
            "rationale": r.rationale,
            "assumptions_satisfied": r.assumptions_satisfied,
            "assumptions_violated": r.assumptions_violated,
        }
        for r in exp.recommendations
    ]

    return {
        "filename": fname,
        "alpha": alpha,
        "total_observations": diag.total_observations,
        "k_algorithms": diag.k_algorithms,
        "n_instances": diag.n_instances,
        "runs_per_cell": diag.runs_per_cell,
        "is_paired": diag.is_paired,
        "is_balanced": diag.is_balanced,
        "can_run_cell_tests": getattr(diag, "can_run_cell_tests", diag.runs_per_cell >= 2),
        "can_run_blocked_omnibus": getattr(diag, "can_run_blocked_omnibus", diag.n_instances >= 2),
        "normality_passed_count": diag.normality_pass_count,
        "normality_total_checks": diag.normality_total_groups,
        "normality_rate": round(float(diag.normality_rate), 4),
        "distribution_profiles": normality_summary,
        "recommendations": recs,
    }


def _tool_run_statistical_analysis(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    alpha = float(args.get("alpha", ctx.alpha))
    paired = bool(args.get("paired", False))
    analyze_kwargs = _filter_supported_kwargs(
        saes.analyze, {"alpha": alpha, "generate_plots": False, "paired": paired}
    )
    report = saes.analyze(dataset, **analyze_kwargs)

    cells = []
    for cell in report.cell_analyses:
        res = cell.result
        cells.append(
            {
                "instance": cell.instance,
                "metric": cell.metric,
                "test_name": res.test_name,
                "statistic": round(float(res.statistic), 5),
                "p_value": float(res.p_value),
                "is_significant": bool(res.is_significant),
                "effect_size": (
                    {
                        "name": res.effect_size.name,
                        "value": round(float(res.effect_size.value), 4),
                        "ci_lower": getattr(res.effect_size, "ci_lower", None),
                        "ci_upper": getattr(res.effect_size, "ci_upper", None),
                        "magnitude": res.effect_size.magnitude,
                    }
                    if res.effect_size
                    else None
                ),
                "assumptions": [
                    {
                        "name": a.name,
                        "p_value": float(a.p_value),
                        "passed": bool(a.passed),
                    }
                    for a in res.assumptions
                ],
                "warnings": list(res.warnings),
            }
        )

    return {
        "filename": fname,
        "alpha": alpha,
        "paired": paired,
        "cells_count": len(cells),
        "significant_cells_count": sum(1 for c in cells if c["is_significant"]),
        "cells": cells[:40],
    }


def _tool_run_ranking_analysis(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    metric = str(args["metric"])
    alpha = float(args.get("alpha", ctx.alpha))
    scope = str(args.get("scope", "all-instances"))
    instance = args.get("instance")
    rope_percent = float(args.get("rope_percent", 1.0))
    rope_abs = float(args["rope"]) if args.get("rope") is not None else None
    prior_strength = float(args.get("prior_strength", 1.0))

    cd_res = FriedmanNemenyiTest().run(df=dataset.df, metric=metric, alpha=alpha, scope=scope, instance=instance)
    bayes_kwargs = _filter_supported_kwargs(
        BayesianRankingTest.run,
        {
            "df": dataset.df,
            "metric": metric,
            "scope": scope,
            "instance": instance,
            "rope_percent": rope_percent,
            "rope": rope_abs,
            "prior_strength": prior_strength,
        },
    )
    bayes_res = BayesianRankingTest().run(**bayes_kwargs)

    out: dict[str, Any] = {
        "filename": fname,
        "metric": metric,
        "scope_label": cd_res.scope_label,
        "friedman_stat": round(float(cd_res.friedman_stat), 4),
        "friedman_p_value": float(cd_res.friedman_p_value),
        "iman_davenport_f": round(float(cd_res.iman_davenport_f), 4),
        "iman_davenport_p_value": float(cd_res.iman_davenport_p_value),
        "nemenyi_cd": round(float(cd_res.cd), 4),
        "ranks": [{"algorithm": r.algorithm, "avg_rank": round(float(r.avg_rank), 3)} for r in cd_res.ranks],
        "bayesian_ranking": [
            {
                "algorithm": b.algorithm,
                "prob_best_pct": round(float(b.prob_best) * 100, 2),
                "mean_rank": round(float(b.mean_rank), 3),
                "ci_95": [round(float(b.ci_lower), 3), round(float(b.ci_upper), 3)],
            }
            for b in bayes_res.items
        ],
    }

    # Include extended blocked omnibus & Plackett-Luce models when available in saes >= 2.0
    stats_ns = getattr(saes, "statistics", None)
    if stats_ns is not None:
        for key, fn_name in (
            ("friedman_aligned", "friedman_aligned_ranks"),
            ("quade", "quade"),
            ("rm_anova", "rm_anova"),
        ):
            fn = getattr(stats_ns, fn_name, None)
            if fn is not None:
                try:
                    res = fn(dataset, metric=metric, alpha=alpha, scope=scope, instance=instance)
                    out[key] = {
                        "test_name": res.test_name,
                        "statistic": round(float(res.statistic), 4),
                        "p_value": float(res.p_value),
                        "is_significant": bool(res.is_significant),
                        "sphericity_epsilon": getattr(res, "sphericity_epsilon", None),
                    }
                except Exception:
                    pass

        pl_fn = getattr(stats_ns, "bayesian_plackett_luce", None)
        if pl_fn is not None:
            try:
                pl_res = pl_fn(dataset, metric=metric, scope=scope, instance=instance)
                out["bayesian_plackett_luce"] = [
                    {
                        "algorithm": it.algorithm,
                        "prob_best_pct": round(float(it.prob_best) * 100, 2),
                        "expected_rank": round(float(it.expected_rank), 3),
                        "ci_95": [round(float(it.rank_ci_lower), 3), round(float(it.rank_ci_upper), 3)],
                    }
                    for it in pl_res.items
                ]
            except Exception:
                pass

    return out


def _tool_generate_statistical_table(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    metric = str(args["metric"])
    table_type = str(args.get("table_type", "instance_algorithm"))
    test_type = str(args.get("test_type", "auto"))
    posthoc = str(args.get("posthoc_correction", "holm"))
    ref_algo = args.get("reference_algorithm")
    alpha = float(args.get("alpha", ctx.alpha))
    paired = bool(args.get("paired", False))

    table_kwargs = _filter_supported_kwargs(
        saes.generate_table,
        {
            "metric": metric,
            "table_type": table_type,
            "test_type": test_type,
            "posthoc_correction": posthoc,
            "reference_algorithm": ref_algo,
            "alpha": alpha,
            "paired": paired,
        },
    )
    table = saes.generate_table(dataset, **table_kwargs)
    md_export = saes.export_table(table, export_format="markdown")
    md_text = md_export.decode("utf-8", errors="replace") if isinstance(md_export, bytes) else str(md_export)
    csv_export = saes.export_table(table, export_format="csv")
    csv_text = csv_export.decode("utf-8", errors="replace") if isinstance(csv_export, bytes) else str(csv_export)

    subtitle = (
        f"{table.resolved_omnibus_test} / {table.resolved_pairwise_test} · "
        f"{table.posthoc_correction} · α = {alpha}"
    )
    ctx.emit_artifact(
        {
            "id": f"table-{uuid4().hex[:8]}",
            "kind": "table",
            "title": f"Statistical Table ({table.table_type}) — {metric}",
            "subtitle": subtitle,
            "filename": fname,
            "metric": metric,
            "table_type": table.table_type,
            "test_type": test_type,
            "posthoc_correction": table.posthoc_correction,
            "reference_algorithm": table.reference_algorithm,
            "alpha": alpha,
            "markdown": md_text,
            "csv_content": csv_text,
            "download_formats": ["csv", "latex", "markdown", "html", "pdf", "docx"],
        }
    )

    return {
        "filename": fname,
        "metric": metric,
        "table_type": table.table_type,
        "resolved_omnibus_test": table.resolved_omnibus_test,
        "resolved_pairwise_test": table.resolved_pairwise_test,
        "posthoc_correction": table.posthoc_correction,
        "reference_algorithm": table.reference_algorithm,
        "displayed_in_chat": True,
        "available_downloads": ["csv", "latex", "markdown", "html", "pdf", "docx"],
        "markdown_preview": md_text[:3500],
    }


def _tool_generate_plot(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    plot_type = str(args.get("plot_type", "boxplot")).strip().lower()
    metric = str(args.get("metric") or (dataset.metrics[0] if dataset.metrics else "value"))
    if metric not in dataset.metrics:
        raise ValueError(f"Metric '{metric}' not found in '{fname}'. Available metrics: {dataset.metrics}")

    is_ranking = plot_type in {
        "critical-distance",
        "bayesian-rank",
        "bayesian-sign",
        "bayesian-signed-rank",
        "plackett-luce",
    }
    scope = str(args.get("scope", "all-instances" if is_ranking else "single-instance"))
    instance = args.get("instance")
    if not instance and (not is_ranking or scope != "all-instances"):
        instance = dataset.instances[0] if dataset.instances else None

    alpha = float(args.get("alpha", ctx.alpha))
    rope_percent = float(args.get("rope_percent", 1.0))
    palette = args.get("palette")

    plot_kwargs: dict[str, Any] = {"alpha": alpha, "scope": scope, "rope_percent": rope_percent}
    if args.get("rope") is not None:
        plot_kwargs["rope"] = float(args["rope"])
    if args.get("prior_strength") is not None:
        plot_kwargs["prior_strength"] = float(args["prior_strength"])
    if palette:
        plot_kwargs["palette"] = str(palette)

    fig_plotly = saes.plot(
        dataset,
        plot_type=plot_type,
        metric=metric,
        instance=instance,
        interactive=True,
        **plot_kwargs,
    )
    plotly_json = json.loads(fig_plotly.to_json())

    image_data_url: str | None = None
    try:
        png_bytes, _, _ = saes.export_plot(
            dataset,
            plot_type=plot_type,
            metric=metric,
            instance=instance,
            export_format="png",
            dpi=150,
            **plot_kwargs,
        )
        image_b64 = base64.b64encode(png_bytes).decode("ascii")
        image_data_url = f"data:image/png;base64,{image_b64}"
    except Exception:
        # Standalone saes-ai environments without matplotlib still serve interactive Plotly specs
        image_data_url = None

    label_map = {
        "boxplot": "Box Plot",
        "violin": "Violin Plot",
        "critical-distance": "Critical Distance (CD) Diagram",
        "bayesian-rank": "Bayesian Posterior Rank Plot",
        "bayesian-sign": "Bayesian Sign Test Simplex",
        "bayesian-signed-rank": "Bayesian Signed-Rank Simplex",
        "plackett-luce": "Bayesian Plackett-Luce Posterior Ranking",
        "effect-size-forest": "Standardized Effect-Size Forest Plot (95% CI)",
        "barplot": "Bar Plot",
        "lineplot": "Line Plot",
        "ecdf": "ECDF Plot",
        "scatter": "Scatter Plot",
    }
    plot_label = label_map.get(plot_type, plot_type.replace("-", " ").title())
    inst_suffix = f" ({instance})" if instance and (not is_ranking or scope != "all-instances") else ""
    title = f"{plot_label} — {metric}{inst_suffix}"

    ctx.emit_artifact(
        {
            "id": f"plot-{uuid4().hex[:8]}",
            "kind": "plot",
            "plot_type": plot_type,
            "title": title,
            "subtitle": f"Dataset: {fname} · α = {alpha}",
            "filename": fname,
            "metric": metric,
            "instance": instance,
            "scope": scope,
            "alpha": alpha,
            "rope_percent": rope_percent,
            "palette": palette,
            "plotly_figure": plotly_json,
            "image_data_url": image_data_url,
            "download_formats": ["png", "svg", "pdf"],
        }
    )

    return {
        "filename": fname,
        "plot_type": plot_type,
        "metric": metric,
        "instance": instance,
        "scope": scope,
        "title": title,
        "displayed_in_chat": True,
        "available_downloads": ["png", "svg", "pdf"],
    }


def _tool_compute_performance_profiles(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    metric = str(args["metric"])
    alpha = float(args.get("alpha", ctx.alpha))
    paired = bool(args.get("paired", False))
    posthoc_correction = str(args.get("posthoc_correction", "holm"))
    prof_kwargs = _filter_supported_kwargs(
        saes.compute_performance_profiles,
        {
            "metric": metric,
            "alpha": alpha,
            "paired": paired,
            "posthoc_correction": posthoc_correction,
        },
    )
    prof = saes.compute_performance_profiles(dataset, **prof_kwargs)

    return {
        "filename": fname,
        "metric": metric,
        "maximize": prof.maximize,
        "paired": paired,
        "posthoc_correction": posthoc_correction,
        "curves_summary": [
            {
                "algorithm": c.algorithm,
                "efficiency_rho1": round(float(c.efficiency), 4),
                "robustness": round(float(c.robustness), 4),
            }
            for c in sorted(prof.curves, key=lambda x: x.efficiency, reverse=True)
        ],
        "global_meta_ranking": [
            {
                "rank": g.rank,
                "algorithm": g.algorithm,
                "copeland_score": round(float(g.copeland_score), 4),
                "net_score": g.net_score,
                "total_wins": g.total_wins,
                "total_ties": g.total_ties,
                "total_losses": g.total_losses,
            }
            for g in prof.meta_rankings
        ],
    }


def _tool_analyze_statistical_power(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    metric = str(args["metric"])
    target_power = float(args.get("target_power", 0.8))
    alpha = float(args.get("alpha", ctx.alpha))
    paired = bool(args.get("paired", False))

    pwr_kwargs = _filter_supported_kwargs(
        saes.analyze_statistical_power,
        {"metric": metric, "target_power": target_power, "alpha": alpha, "paired": paired},
    )
    pwr = saes.analyze_statistical_power(dataset, **pwr_kwargs)
    sensitivity_items = getattr(pwr, "sensitivity_audit", []) or []
    return {
        "filename": fname,
        "metric": metric,
        "alpha": pwr.alpha,
        "target_power": pwr.target_power,
        "target_effect_size": pwr.target_effect_size,
        "paired": paired,
        "overall_power_health": pwr.overall_power_health,
        "sample_size_recommendations": [
            {
                "effect_size_label": s.effect_size_label,
                "effect_size": round(float(s.effect_size), 4),
                "recommended_n_independent": s.recommended_n_independent,
                "recommended_n_paired": s.recommended_n_paired,
                "recommended_n_nonparametric": s.recommended_n_nonparametric,
            }
            for s in pwr.sample_size_recommendations[:10]
        ],
        "post_hoc_audit_count": len(pwr.post_hoc_audit),
        "sensitivity_mdes_preview": [
            {
                "instance": it.instance,
                "comparison": f"{it.algo_1} vs {it.algo_2}",
                "n_samples": it.n_samples,
                "mdes_cohens_d": round(float(it.mdes_cohens_d), 4),
                "observed_cohens_d": round(float(it.observed_cohens_d), 4),
            }
            for it in sensitivity_items[:15]
        ],
    }


def _tool_run_posthoc_analysis(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    run_posthoc_for_dataset = _import_saes_server_helper("run_posthoc_for_dataset", "Post-hoc analysis")

    res = run_posthoc_for_dataset(
        dataset=dataset,
        metric=args.get("metric"),
        instance=args.get("instance"),
        method=str(args.get("method", "auto")),
        control=args.get("control"),
        p_adjust=str(args.get("p_adjust", "holm")),
        alpha=float(args.get("alpha", ctx.alpha)),
    )
    return {
        "filename": fname,
        "method": res.method,
        "p_adjust": res.p_adjust,
        "alpha": res.alpha,
        "significant_pairs": sum(1 for c in res.comparisons if c.is_significant),
        "total_pairs": len(res.comparisons),
        "comparisons": [c.model_dump() for c in res.comparisons[:40]],
        "warnings": list(res.warnings),
    }


def _tool_run_tost_equivalence(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    run_tost_for_dataset = _import_saes_server_helper("run_tost_for_dataset", "TOST equivalence testing")

    res = run_tost_for_dataset(
        dataset=dataset,
        metric=args.get("metric"),
        algo_a=args.get("algo_a"),
        algo_b=args.get("algo_b"),
        low_eqbound=float(args.get("low_eqbound", 0.05)),
        high_eqbound=float(args["high_eqbound"]) if args.get("high_eqbound") is not None else None,
        scope=str(args.get("scope", "single-instance")),
        instance=args.get("instance"),
        alpha=float(args.get("alpha", ctx.alpha)),
        paired=bool(args.get("paired", False)),
        equal_var=bool(args.get("equal_var", False)),
    )
    return {"filename": fname, **res.model_dump()}


def _tool_run_bayesian_correlated_ttest(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    run_bayesian_correlated_t_for_dataset = _import_saes_server_helper(
        "run_bayesian_correlated_t_for_dataset", "Bayesian correlated t-test"
    )

    res = run_bayesian_correlated_t_for_dataset(
        dataset=dataset,
        metric=args.get("metric"),
        algo_a=args.get("algo_a"),
        algo_b=args.get("algo_b"),
        scope=str(args.get("scope", "single-instance")),
        instance=args.get("instance"),
        rope=float(args.get("rope", 0.01)),
        rho=float(args["rho"]) if args.get("rho") is not None else None,
    )
    dumped = res.model_dump(exclude={"curve_x", "curve_y"})
    return {"filename": fname, **dumped}


def _tool_compute_effect_size_forest(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    stats_ns = getattr(saes, "statistics", None)
    forest_fn = getattr(stats_ns, "effect_size_forest", None) if stats_ns is not None else None
    if forest_fn is None:
        raise RuntimeError("Effect-size forest requires saes >= 2.0.0.")

    metric = str(args.get("metric") or (dataset.metrics[0] if dataset.metrics else "value"))
    res = forest_fn(
        dataset=dataset,
        metric=metric,
        scope=str(args.get("scope", "single-instance")),
        instance=args.get("instance"),
        reference_algorithm=args.get("reference_algorithm"),
        alpha=float(args.get("alpha", ctx.alpha)),
        paired=bool(args.get("paired", False)),
        p_adjust=str(args.get("p_adjust", "holm")),
    )
    return {"filename": fname, **res.model_dump()}


def _tool_generate_reproducibility(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    dataset, fname = _load_validated_dataset(ctx, dataset_name=args.get("dataset_name"))
    alpha = float(args.get("alpha", ctx.alpha))
    repro = saes.generate_reproducibility_artifacts(dataset, alpha=alpha)
    return {
        "filename": fname,
        "environment": {
            "python_version": repro.manifest.python_version,
            "saes_version": repro.manifest.saes_version,
            "platform": repro.manifest.platform,
            "row_count": repro.manifest.row_count,
            "runs_per_cell": repro.manifest.runs_per_cell,
        },
        "python_script_preview": repro.python_script[:1500],
    }


def _tool_navigate_ui(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    view = str(args.get("view", "visualize"))
    tab = args.get("tab")
    dataset_name = args.get("dataset_name")
    alpha = args.get("alpha")
    metric = args.get("metric")
    test_type = args.get("test_type")
    table_type = args.get("table_type")

    if dataset_name:
        safe = _sanitize_csv_filename(str(dataset_name))
        ctx.active_dataset_name = safe
        dataset_name = safe

    action: dict[str, Any] = {
        "type": "navigate_ui",
        "view": view,
    }
    if tab:
        action["tab"] = tab
    if dataset_name:
        action["dataset_name"] = dataset_name
    if alpha is not None:
        action["alpha"] = float(alpha)
        ctx.alpha = float(alpha)
    if metric:
        action["metric"] = metric
    if test_type:
        action["test_type"] = test_type
    if table_type:
        action["table_type"] = table_type

    ctx.emit_ui_action(action)
    return {"navigated": True, "action": action}


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


SAES_AGENT_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="list_datasets",
        description=(
            "List all uploaded benchmark CSV datasets in SAES storage, including validity status, "
            "row counts, algorithms, and metrics."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_tool_list_datasets,
    ),
    ToolSpec(
        name="inspect_dataset",
        description=(
            "Inspect and validate a dataset (either saved in storage or uploaded/staged in chat). "
            "Checks SAES required columns (algorithm, instance, metric, run_id, value, maximize), "
            "returns validation errors/warnings, sample rows, and column mapping suggestions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "Filename of saved dataset (e.g., 'benchmark.csv')."},
                "use_staged": {
                    "type": "boolean",
                    "description": "Set true to inspect the CSV file just uploaded in chat.",
                    "default": False,
                },
            },
            "required": [],
        },
        handler=_tool_inspect_dataset,
    ),
    ToolSpec(
        name="create_dataset",
        description=(
            "Create a new SAES benchmark CSV dataset from structured rows or CSV text, validate it, "
            "and save it to SAES storage."
        ),
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Target CSV filename (e.g., 'my_experiment.csv')."},
                "csv_content": {
                    "type": "string",
                    "description": "Full CSV content with header: algorithm,instance,metric,run_id,value,maximize.",
                },
                "rows": {
                    "type": "array",
                    "description": (
                        "Alternative to csv_content: list of row objects with "
                        "algorithm, instance, metric, run_id, value, maximize."
                    ),
                    "items": {"type": "object"},
                },
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["filename"],
        },
        handler=_tool_create_dataset,
    ),
    ToolSpec(
        name="edit_dataset",
        description=(
            "Edit or repair a dataset (saved or staged chat upload). Can rename/map columns, fill missing values, "
            "update specific rows by row_index, drop invalid rows, or replace CSV content, then re-validate and save."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "Saved dataset filename to edit."},
                "use_staged": {
                    "type": "boolean",
                    "description": "Set true to edit the staged chat upload.",
                    "default": False,
                },
                "target_filename": {"type": "string", "description": "Filename to save the edited dataset as."},
                "column_mapping": {
                    "type": "object",
                    "description": "Map old column names to SAES names, e.g. {'algo': 'algorithm'}.",
                },
                "drop_extra_columns": {"type": "boolean", "default": False},
                "fill_missing": {
                    "type": "object",
                    "description": "Default values for missing columns/cells, e.g. {'maximize': true}.",
                },
                "row_updates": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of {'row_index': 0, 'value': 0.5, ...}.",
                },
                "drop_invalid_rows": {"type": "boolean", "default": False},
                "new_csv_content": {"type": "string", "description": "Optional complete replacement CSV text."},
                "save": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        handler=_tool_edit_dataset,
    ),
    ToolSpec(
        name="delete_dataset",
        description="Delete a dataset from SAES storage when explicitly requested by the user.",
        parameters={
            "type": "object",
            "properties": {"dataset_name": {"type": "string"}},
            "required": ["dataset_name"],
        },
        handler=_tool_delete_dataset,
    ),
    ToolSpec(
        name="get_dataset_kpis",
        description=(
            "Extract key performance indicators (KPIs) and descriptive statistics: algorithms, "
            "instances, metrics, runs/cell, mean, std, median, min, max, and instance win counts."
        ),
        parameters={
            "type": "object",
            "properties": {"dataset_name": {"type": "string"}},
            "required": [],
        },
        handler=_tool_get_dataset_kpis,
    ),
    ToolSpec(
        name="explore_dataset",
        description=(
            "Run assumption diagnostics (Shapiro-Wilk, D'Agostino, KS normality; Levene, Bartlett homoscedasticity) "
            "and get smart statistical test recommendations based on dataset properties."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "alpha": {"type": "number", "default": 0.05},
            },
            "required": [],
        },
        handler=_tool_explore_dataset,
    ),
    ToolSpec(
        name="run_statistical_analysis",
        description=(
            "Run SAES automated statistical hypothesis testing across all experimental cells "
            "(parametric & non-parametric) with exact effect-size 95% CIs and assumption checks."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "alpha": {"type": "number", "default": 0.05},
                "paired": {
                    "type": "boolean",
                    "description": (
                        "Default false (independent stochastic runs). Set true only for matched CV folds/CRN."
                    ),
                    "default": False,
                },
            },
            "required": [],
        },
        handler=_tool_run_statistical_analysis,
    ),
    ToolSpec(
        name="run_ranking_analysis",
        description=(
            "Run blocked multi-instance omnibus tests (Friedman + Nemenyi CD, Iman-Davenport, "
            "Friedman Aligned-Ranks, Quade, RM-ANOVA GG) and Bayesian ranking (Simplex ROPE & Plackett-Luce)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "Metric name to rank algorithms on."},
                "dataset_name": {"type": "string"},
                "alpha": {"type": "number", "default": 0.05},
                "scope": {"type": "string", "enum": ["all-instances", "single-instance"], "default": "all-instances"},
                "instance": {"type": "string"},
                "rope_percent": {"type": "number", "default": 1.0},
                "rope": {"type": "number", "description": "Optional exact absolute ROPE half-width in metric units."},
                "prior_strength": {"type": "number", "default": 1.0},
            },
            "required": ["metric"],
        },
        handler=_tool_run_ranking_analysis,
    ),
    ToolSpec(
        name="generate_statistical_table",
        description=(
            "Generate publication-grade statistical benchmark tables (Friedman, Aligned-Ranks, Quade, "
            "RM-ANOVA, Brunner-Munzel, Wilcoxon, Kruskal, ANOVA, Welch, t-test) with 13 multiplicity adjustments."
        ),
        parameters={
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "dataset_name": {"type": "string"},
                "table_type": {
                    "type": "string",
                    "enum": ["instance_algorithm", "pairwise_matrix", "omnibus_posthoc"],
                    "default": "instance_algorithm",
                },
                "test_type": {
                    "type": "string",
                    "enum": [
                        "auto",
                        "wilcoxon",
                        "brunner_munzel",
                        "mann_whitney",
                        "friedman",
                        "friedman_aligned",
                        "quade",
                        "rm_anova",
                        "kruskal",
                        "anova_oneway",
                        "anova_welch",
                        "ttest_welch",
                        "ttest_ind",
                        "ttest_rel",
                    ],
                    "default": "auto",
                },
                "posthoc_correction": {
                    "type": "string",
                    "enum": [
                        "holm",
                        "bonferroni",
                        "hochberg",
                        "hommel",
                        "finner",
                        "holland",
                        "rom",
                        "li",
                        "fdr_bh",
                        "fdr_by",
                        "shaffer",
                        "bergmann_hommel",
                        "none",
                    ],
                    "default": "holm",
                },
                "reference_algorithm": {"type": "string"},
                "alpha": {"type": "number", "default": 0.05},
                "paired": {"type": "boolean", "default": False},
            },
            "required": ["metric"],
        },
        handler=_tool_generate_statistical_table,
    ),
    ToolSpec(
        name="generate_plot",
        description=(
            "Generate a SAES statistical plot, diagram, or image (boxplot, violin, critical-distance, "
            "bayesian-rank, bayesian-sign, bayesian-signed-rank, plackett-luce, effect-size-forest, "
            "barplot, lineplot, ecdf, scatter) and display it directly inside the chatbot with PNG/SVG/PDF downloads."
        ),
        parameters={
            "type": "object",
            "properties": {
                "plot_type": {
                    "type": "string",
                    "enum": [
                        "boxplot",
                        "violin",
                        "critical-distance",
                        "bayesian-rank",
                        "bayesian-sign",
                        "bayesian-signed-rank",
                        "plackett-luce",
                        "effect-size-forest",
                        "barplot",
                        "lineplot",
                        "ecdf",
                        "scatter",
                    ],
                    "default": "boxplot",
                },
                "metric": {"type": "string"},
                "instance": {"type": "string"},
                "scope": {
                    "type": "string",
                    "enum": ["all-instances", "single-instance"],
                },
                "alpha": {"type": "number", "default": 0.05},
                "rope_percent": {"type": "number", "default": 1.0},
                "rope": {"type": "number"},
                "prior_strength": {"type": "number"},
                "palette": {"type": "string"},
                "dataset_name": {"type": "string"},
            },
            "required": ["plot_type"],
        },
        handler=_tool_generate_plot,
    ),
    ToolSpec(
        name="run_posthoc_analysis",
        description=(
            "Run cell-level multiple-comparison post-hoc tests (auto, tukey_hsd, games_howell, "
            "dunnett vs control, dunn, conover_iman) with exact 95% CIs and FWER/FDR adjustments."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "metric": {"type": "string"},
                "instance": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": ["auto", "tukey_hsd", "games_howell", "dunnett", "dunn", "conover_iman"],
                    "default": "auto",
                },
                "control": {"type": "string", "description": "Control baseline algorithm for Dunnett's test."},
                "p_adjust": {"type": "string", "default": "holm"},
                "alpha": {"type": "number", "default": 0.05},
            },
            "required": [],
        },
        handler=_tool_run_posthoc_analysis,
    ),
    ToolSpec(
        name="run_tost_equivalence",
        description=(
            "Run Schuirmann (1987) / Lakens (2017) Two One-Sided Tests (TOST) to test formal "
            "statistical equivalence between two algorithms within [-delta, +delta]."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "metric": {"type": "string"},
                "algo_a": {"type": "string"},
                "algo_b": {"type": "string"},
                "low_eqbound": {"type": "number", "default": 0.05},
                "high_eqbound": {"type": "number"},
                "scope": {"type": "string", "enum": ["single-instance", "all-instances"], "default": "single-instance"},
                "instance": {"type": "string"},
                "alpha": {"type": "number", "default": 0.05},
                "paired": {"type": "boolean", "default": False},
                "equal_var": {"type": "boolean", "default": False},
            },
            "required": ["metric"],
        },
        handler=_tool_run_tost_equivalence,
    ),
    ToolSpec(
        name="run_bayesian_correlated_ttest",
        description=(
            "Run the Nadeau & Bengio (2003) / Benavoli et al. (2017) Bayesian Correlated t-test "
            "for overlapping k-fold cross-validation or paired benchmark differences with ROPE."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "metric": {"type": "string"},
                "algo_a": {"type": "string"},
                "algo_b": {"type": "string"},
                "scope": {"type": "string", "enum": ["single-instance", "all-instances"], "default": "single-instance"},
                "instance": {"type": "string"},
                "rope": {"type": "number", "default": 0.01},
                "rho": {"type": "number", "description": "CV correlation n_test/n_train (e.g., 0.111 for 10-fold CV)."},
            },
            "required": ["metric"],
        },
        handler=_tool_run_bayesian_correlated_ttest,
    ),
    ToolSpec(
        name="compute_effect_size_forest",
        description=(
            "Compute pairwise standardized effect sizes (Hedges' g, Cohen's d, Vargha-Delaney A12, "
            "Cliff's delta) with exact 95% confidence intervals against a reference baseline."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "metric": {"type": "string"},
                "scope": {"type": "string", "enum": ["single-instance", "all-instances"], "default": "single-instance"},
                "instance": {"type": "string"},
                "reference_algorithm": {"type": "string"},
                "alpha": {"type": "number", "default": 0.05},
                "paired": {"type": "boolean", "default": False},
                "p_adjust": {"type": "string", "default": "holm"},
            },
            "required": ["metric"],
        },
        handler=_tool_compute_effect_size_forest,
    ),
    ToolSpec(
        name="compute_performance_profiles",
        description=(
            "Compute Dolan-Moré performance profiles, efficiency/robustness, multiplicity-adjusted "
            "win-tie-loss matrix, and Copeland meta-ranking for a metric."
        ),
        parameters={
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "dataset_name": {"type": "string"},
                "alpha": {"type": "number", "default": 0.05},
                "paired": {"type": "boolean", "default": False},
                "posthoc_correction": {"type": "string", "default": "holm"},
            },
            "required": ["metric"],
        },
        handler=_tool_compute_performance_profiles,
    ),
    ToolSpec(
        name="analyze_statistical_power",
        description=(
            "Analyze statistical power, a priori sample sizes, and Minimum Detectable Effect Size "
            "(MDES) sensitivity audits for a metric."
        ),
        parameters={
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "dataset_name": {"type": "string"},
                "target_power": {"type": "number", "default": 0.8},
                "alpha": {"type": "number", "default": 0.05},
                "paired": {"type": "boolean", "default": False},
            },
            "required": ["metric"],
        },
        handler=_tool_analyze_statistical_power,
    ),
    ToolSpec(
        name="generate_reproducibility_package",
        description=(
            "Generate environment manifest, methodology paragraph, BibTeX citation, "
            "and reproducible Python code snippet."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "alpha": {"type": "number", "default": 0.05},
            },
            "required": [],
        },
        handler=_tool_generate_reproducibility,
    ),
    ToolSpec(
        name="navigate_ui",
        description=(
            "Navigate the SAES UI when the user explicitly asks to open a view or workspace tab, "
            "optionally opening a dataset, changing alpha, or pre-configuring the Statistical Workbench."
        ),
        parameters={
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": ["hub", "editor", "visualize"], "description": "Top-level UI view."},
                "tab": {
                    "type": "string",
                    "enum": ["data", "stats", "charts", "profiles", "power", "reproducibility"],
                    "description": "Workspace tab when view='visualize'.",
                },
                "dataset_name": {"type": "string", "description": "Dataset filename to select and open."},
                "alpha": {
                    "type": "number",
                    "description": "Update global significance level alpha (e.g. 0.01, 0.05, 0.10).",
                },
                "metric": {"type": "string", "description": "Pre-select metric in Statistical Workbench."},
                "test_type": {"type": "string", "description": "Pre-select statistical test in Statistical Workbench."},
                "table_type": {"type": "string", "enum": ["instance_algorithm", "pairwise_matrix", "omnibus_posthoc"]},
            },
            "required": ["view"],
        },
        handler=_tool_navigate_ui,
    ),
]

TOOL_REGISTRY: dict[str, ToolSpec] = {t.name: t for t in SAES_AGENT_TOOLS}


def execute_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a registered tool safely and return a JSON-serializable result."""
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    try:
        result = spec.handler(ctx, arguments)
        return {"ok": True, **result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
