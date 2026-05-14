"""Local readiness checks for AIES NIDS.

These checks are intentionally local and non-destructive. They help the operator
answer a simple question before a demo or viva: is the system ready to run?
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _status(ok: bool, warn: bool = False) -> str:
    if ok and not warn:
        return "PASS"
    if ok and warn:
        return "WARN"
    return "FAIL"


def _mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2) if path.exists() and path.is_file() else 0.0


def project_health(project_root: Path) -> pd.DataFrame:
    """Return a compact health table for core local artifacts."""
    root = Path(project_root)
    checks: list[dict[str, Any]] = []

    cic_model = root / "models" / "nids_model.pkl"
    nsl_model = root / "models" / "nslkdd_model.pkl"
    manifest = root / "models" / "model_manifest.json"
    nsl_manifest = root / "models" / "nslkdd_manifest.json"
    data_dir = root / "data"
    reports_dir = root / "reports"
    assets_dir = root / "assets"
    hero = assets_dir / "aies-cyber-hero.png"
    validator = root / "validate_project.py"

    csv_count = len(list(data_dir.glob("*.csv"))) if data_dir.exists() else 0
    nsl_count = len(list((data_dir / "nslkdd").glob("*.txt"))) if (data_dir / "nslkdd").exists() else 0

    checks.extend([
        {
            "Area": "Model",
            "Check": "CICIDS model artifact",
            "Status": _status(cic_model.exists()),
            "Detail": f"{_mb(cic_model)} MB" if cic_model.exists() else "models/nids_model.pkl missing",
        },
        {
            "Area": "Model",
            "Check": "CICIDS manifest",
            "Status": _status(manifest.exists()),
            "Detail": "model_manifest.json present" if manifest.exists() else "manifest missing",
        },
        {
            "Area": "Benchmark",
            "Check": "NSL-KDD model and manifest",
            "Status": _status(nsl_model.exists() and nsl_manifest.exists(), warn=not nsl_model.exists() or not nsl_manifest.exists()),
            "Detail": "secondary benchmark ready" if nsl_model.exists() and nsl_manifest.exists() else "secondary benchmark incomplete",
        },
        {
            "Area": "Data",
            "Check": "CICIDS CSV files",
            "Status": _status(csv_count > 0, warn=csv_count < 4 and csv_count > 0),
            "Detail": f"{csv_count} CSV files in data/",
        },
        {
            "Area": "Data",
            "Check": "NSL-KDD text files",
            "Status": _status(nsl_count >= 2, warn=nsl_count == 1),
            "Detail": f"{nsl_count} NSL-KDD files",
        },
        {
            "Area": "Storage",
            "Check": "Reports directory",
            "Status": _status(reports_dir.exists()),
            "Detail": str(reports_dir),
        },
        {
            "Area": "Storage",
            "Check": "SQLite targets",
            "Status": _status(reports_dir.exists()),
            "Detail": "alerts.db and audit.db are created on demand",
        },
        {
            "Area": "UI",
            "Check": "Cyber hero asset",
            "Status": _status(hero.exists()),
            "Detail": f"{_mb(hero)} MB" if hero.exists() else "assets/aies-cyber-hero.png missing",
        },
        {
            "Area": "Docs",
            "Check": "README, blueprint, viva runbook",
            "Status": _status(
                (root / "README.md").exists()
                and (root / "docs" / "PROJECT_BLUEPRINT.md").exists()
                and (root / "docs" / "VIVA_AND_DEMO_RUNBOOK.md").exists()
                and (root / "docs" / "OPERATOR_LAUNCH_CHECKLIST.md").exists()
            ),
            "Detail": "operator and presentation docs present",
        },
        {
            "Area": "Ops",
            "Check": "Validation command",
            "Status": _status(validator.exists()),
            "Detail": "python validate_project.py" if validator.exists() else "validator missing",
        },
        {
            "Area": "SOC",
            "Check": "Case-management module",
            "Status": _status((root / "src" / "case_management.py").exists()),
            "Detail": "case roll-up ready",
        },
        {
            "Area": "Assurance",
            "Check": "Model-risk module",
            "Status": _status((root / "src" / "model_risk.py").exists()),
            "Detail": "confidence and review gate ready",
        },
        {
            "Area": "Data",
            "Check": "Scenario lab module",
            "Status": _status((root / "src" / "scenario_lab.py").exists()),
            "Detail": "mixed attack presets ready",
        },
        {
            "Area": "Ops",
            "Check": "Readiness report module",
            "Status": _status((root / "src" / "readiness_report.py").exists()),
            "Detail": "pre-viva export ready",
        },
    ])

    return pd.DataFrame(checks)


def health_summary(health: pd.DataFrame) -> dict[str, Any]:
    if health is None or health.empty:
        return {"status": "FAIL", "pass": 0, "warn": 0, "fail": 1}
    counts = health["Status"].value_counts().to_dict()
    fail = int(counts.get("FAIL", 0))
    warn = int(counts.get("WARN", 0))
    passed = int(counts.get("PASS", 0))
    status = "FAIL" if fail else "WARN" if warn else "PASS"
    return {"status": status, "pass": passed, "warn": warn, "fail": fail}
