"""Markdown readiness report for AIES NIDS demos and viva checks."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.scenario_lab import scenario_description, scenario_names, scenario_targets


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def _markdown_table(df: pd.DataFrame | None, limit: int | None = None) -> list[str]:
    if df is None or df.empty:
        return ["_No rows._"]
    view = df.head(limit).copy() if limit else df.copy()
    headers = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        cells = [
            str(row.get(col, "")).replace("\n", " ").replace("|", "/")
            for col in view.columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _manifest_lines(title: str, manifest: dict[str, Any]) -> list[str]:
    if not manifest:
        return [f"### {title}", "", "- Manifest not found.", ""]
    classes = ", ".join(str(c) for c in manifest.get("classes", [])) or "n/a"
    return [
        f"### {title}",
        "",
        f"- Model: {manifest.get('model_name', 'Unknown')}",
        f"- Macro-F1: {float(manifest.get('macro_f1', 0.0)):.3f}",
        f"- Accuracy: {float(manifest.get('accuracy', 0.0)):.3f}",
        f"- Features: {manifest.get('n_features', 'n/a')}",
        f"- Classes: {classes}",
        f"- Train time: {float(manifest.get('train_time_sec', 0.0)):.1f}s",
        "",
    ]


def build_readiness_report(project_root: Path, health_df: pd.DataFrame, health_meta: dict[str, Any]) -> str:
    """Build a portable pre-demo report that can be exported from the UI."""
    root = Path(project_root)
    model_manifest = _load_json(root / "models" / "model_manifest.json")
    nsl_manifest = _load_json(root / "models" / "nslkdd_manifest.json")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# AIES NIDS Readiness Report",
        "",
        f"Generated: {generated}",
        f"Project root: `{root}`",
        "",
        "## Verdict",
        f"- Status: {health_meta.get('status', 'UNKNOWN')}",
        f"- PASS: {health_meta.get('pass', 0)}",
        f"- WARN: {health_meta.get('warn', 0)}",
        f"- FAIL: {health_meta.get('fail', 0)}",
        "",
        "## System Health",
        *_markdown_table(health_df),
        "",
        "## Model Artifacts",
        *_manifest_lines("CICIDS2017 Primary Model", model_manifest),
        *_manifest_lines("NSL-KDD Secondary Benchmark", nsl_manifest),
        "## Scenario Presets",
    ]

    for name in scenario_names():
        targets = scenario_targets(1000, name)
        mix = ", ".join(f"{label}={count}" for label, count in targets.items())
        lines.extend([
            f"### {name}",
            "",
            f"- Purpose: {scenario_description(name)}",
            f"- 1000-row target mix: {mix}",
            "",
        ])

    lines.extend([
        "## Recommended Demo Flow",
        "",
        "1. Run Analysis: Mixed Attack Lab / Executive mixed / 1000 rows.",
        "2. Overview: explain current-session threat posture and severity mix.",
        "3. Forensics: show Case Board, incident queue, evidence, playbook, and exports.",
        "4. SOC Brief: show data quality, timeline, attack surface, remediation, and markdown export.",
        "5. AIES Theory: show forward chaining, backward chaining, certainty factors, and hybrid comparison.",
        "",
        "## Operator Checklist",
        "",
        "- Full checklist: `docs/OPERATOR_LAUNCH_CHECKLIST.md`",
        "- Start command: `streamlit run app.py`",
        "- Validation command: `python validate_project.py`",
        "- Best first preset: `Mixed Attack Lab / Executive mixed / 1000 rows`",
        "",
        "## Honest Scope",
        "",
        "- Replay/batch mode only; real packet capture is a production adapter task.",
        "- Geographic attribution is synthetic because CICIDS2017 strips real IP context.",
        "- Compliance mapping is a demonstration, not a legal audit.",
        "- This is a high-quality academic/SOC prototype, not a hardened multi-tenant SaaS backend yet.",
        "",
    ])
    return "\n".join(lines)
