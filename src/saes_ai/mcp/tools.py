"""Tool definitions and schemas for the SAES Model Context Protocol (MCP) server.

NOTE: This is the prepared structural architecture for MCP integration.
Implementation is stubbed until the MCP server feature phase is activated.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MCPToolSpec:
    """Specification of an MCP tool exposed to LLMs."""

    name: str
    description: str
    parameters_schema: dict[str, Any]


SAES_MCP_TOOLS: list[MCPToolSpec] = [
    MCPToolSpec(
        name="validate_benchmark_csv",
        description="Validate schema, column names, data types, and value constraints of a benchmark CSV file.",
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the CSV benchmark file on disk.",
                },
            },
            "required": ["file_path"],
        },
    ),
    MCPToolSpec(
        name="run_statistical_analysis",
        description="Run automated statistical evaluation (normality, homoscedasticity, t-test/ANOVA, effect sizes).",
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to benchmark CSV.",
                },
                "alpha": {
                    "type": "number",
                    "description": "Significance level (default: 0.05).",
                    "default": 0.05,
                },
                "instance": {
                    "type": "string",
                    "description": "Optional specific problem instance to analyze (e.g. 'ZDT1').",
                },
                "metric": {
                    "type": "string",
                    "description": "Optional specific metric to analyze (e.g. 'HV').",
                },
            },
            "required": ["file_path"],
        },
    ),
    MCPToolSpec(
        name="check_assumptions",
        description="Test normality (Shapiro-Wilk) and homoscedasticity (Levene's test) for experimental groups.",
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to benchmark CSV."},
                "instance": {"type": "string", "description": "Problem instance identifier."},
                "metric": {"type": "string", "description": "Metric name."},
                "alpha": {"type": "number", "default": 0.05},
            },
            "required": ["file_path", "instance", "metric"],
        },
    ),
    MCPToolSpec(
        name="generate_distribution_plot",
        description=(
            "Generate statistical visualization (boxplot, violin, critical-distance, bayesian-rank) "
            "for benchmark results."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to benchmark CSV."},
                "plot_type": {
                    "type": "string",
                    "enum": [
                        "boxplot",
                        "violin",
                        "critical-distance",
                        "bayesian-rank",
                        "barplot",
                        "ecdf",
                        "scatterplot",
                    ],
                    "description": "Type of distribution or ranking plot to generate.",
                },
                "instance": {"type": "string", "description": "Problem instance identifier."},
                "metric": {"type": "string", "description": "Metric name."},
                "output_path": {"type": "string", "description": "File path to save the generated plot image."},
            },
            "required": ["file_path", "plot_type", "metric"],
        },
    ),
    MCPToolSpec(
        name="run_ranking_analysis",
        description="Run Friedman + Nemenyi Critical Distance (CD) and Bayesian Posterior Ranking across algorithms.",
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to benchmark CSV."},
                "metric": {"type": "string", "description": "Metric name to rank algorithms on."},
                "alpha": {"type": "number", "default": 0.05},
                "scope": {"type": "string", "enum": ["all-instances", "single-instance"], "default": "all-instances"},
                "instance": {"type": "string", "description": "Optional problem instance if scope='single-instance'."},
                "rope_percent": {"type": "number", "default": 1.0},
            },
            "required": ["file_path", "metric"],
        },
    ),
    MCPToolSpec(
        name="generate_benchmark_table",
        description=(
            "Generate publication-grade ANOVA, Friedman, Wilcoxon, Kruskal-Wallis, or Pairwise statistical "
            "benchmark tables with LaTeX, CSV, Markdown, PDF, and DOCX exports."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to benchmark CSV."},
                "metric": {"type": "string", "description": "Metric name."},
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
                        "mann_whitney",
                        "friedman",
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
                    "enum": ["holm", "bonferroni", "hochberg", "fdr_bh", "none"],
                    "default": "holm",
                },
                "reference_algorithm": {"type": "string", "description": "Control/reference algorithm name."},
                "alpha": {"type": "number", "default": 0.05},
            },
            "required": ["file_path", "metric"],
        },
    ),
    MCPToolSpec(
        name="explore_dataset_distributions",
        description=(
            "Explore dataset observations, reveal multi-metric pivots, evaluate normality (Shapiro-Wilk, "
            "D'Agostino, KS) and homoscedasticity (Levene, Bartlett), and get smart statistical test recommendations."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to benchmark CSV."},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "algorithms": {"type": "array", "items": {"type": "string"}},
                "instances": {"type": "array", "items": {"type": "string"}},
                "alpha": {"type": "number", "default": 0.05},
            },
            "required": ["file_path"],
        },
    ),
]
