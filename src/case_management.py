"""Incident case roll-up for AIES NIDS.

Per-flow alerts are useful evidence, but SOC analysts work cases. This module
groups related detections into a compact case board with priorities, owners,
and first response actions.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Informational": 0}

CASE_OWNER = {
    "DDoS": "Network / Edge",
    "DoS": "Network / Platform",
    "Botnet": "Incident Response",
    "Infiltration": "Incident Response / DLP",
    "BruteForce": "Identity / SOC",
    "WebAttack": "AppSec / Web",
    "PortScan": "Network / Vulnerability",
}

CASE_ACTION = {
    "DDoS": "Engage scrubbing/CDN protection and apply edge rate limits.",
    "DoS": "Throttle offending service path and validate resource pressure.",
    "Botnet": "Quarantine host and sinkhole suspicious C2 destinations.",
    "Infiltration": "Isolate affected host and preserve forensic evidence.",
    "BruteForce": "Enable lockout/MFA and inspect successful logins.",
    "WebAttack": "Enable WAF rules and review targeted endpoints.",
    "PortScan": "Validate exposed services and close unintended ports.",
}


def _priority(max_severity: str, count: int, median_confidence: float) -> str:
    if max_severity == "Critical" or (max_severity == "High" and count >= 50):
        return "P1"
    if max_severity == "High" or (max_severity == "Medium" and median_confidence >= 0.85):
        return "P2"
    if max_severity == "Medium" or count >= 25:
        return "P3"
    return "P4"


def _status(priority: str) -> str:
    return {
        "P1": "Containment",
        "P2": "Investigation",
        "P3": "Watchlist",
        "P4": "Monitor",
    }.get(priority, "Monitor")


def build_case_board(enriched: pd.DataFrame) -> pd.DataFrame:
    """Group non-BENIGN enriched rows into analyst-facing cases."""
    if enriched is None or enriched.empty:
        return pd.DataFrame(columns=[
            "Case", "Priority", "Status", "Class", "Service", "MITRE",
            "Events", "Critical", "High", "Median confidence", "Window",
            "Owner", "First action",
        ])

    df = enriched[enriched["predicted_class"] != "BENIGN"].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "Case", "Priority", "Status", "Class", "Service", "MITRE",
            "Events", "Critical", "High", "Median confidence", "Window",
            "Owner", "First action",
        ])

    for col in ["service", "mitre_id", "country", "kill_chain_stage"]:
        if col not in df.columns:
            df[col] = "Unknown"
    df["_sev_rank"] = df["severity"].map(SEVERITY_RANK).fillna(0).astype(int)

    rows: list[dict[str, Any]] = []
    group_cols = ["predicted_class", "service", "mitre_id"]
    for idx, ((cls, service, mitre), part) in enumerate(df.groupby(group_cols, sort=False), start=1):
        part = part.copy()
        max_rank = int(part["_sev_rank"].max())
        max_severity = next((s for s, r in SEVERITY_RANK.items() if r == max_rank), "Informational")
        median_conf = float(part["confidence"].median()) if "confidence" in part else 0.0
        priority = _priority(max_severity, len(part), median_conf)
        countries = ", ".join(part["country"].value_counts().head(3).index.astype(str))
        stages = ", ".join(part["kill_chain_stage"].value_counts().head(2).index.astype(str))
        case_id = f"CASE-{idx:03d}-{str(cls).upper()[:3]}"
        rows.append({
            "Case": case_id,
            "Priority": priority,
            "Status": _status(priority),
            "Class": str(cls),
            "Service": str(service),
            "MITRE": str(mitre),
            "Events": int(len(part)),
            "Critical": int((part["severity"] == "Critical").sum()),
            "High": int((part["severity"] == "High").sum()),
            "Median confidence": round(median_conf, 3),
            "Window": f"{int(part['flow_idx'].min())}-{int(part['flow_idx'].max())}",
            "Top countries": countries,
            "Kill chain": stages,
            "Owner": CASE_OWNER.get(str(cls), "SOC"),
            "First action": CASE_ACTION.get(str(cls), "Review evidence and assign analyst."),
        })

    out = pd.DataFrame(rows)
    rank = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    out["_rank"] = out["Priority"].map(rank).fillna(9)
    return out.sort_values(["_rank", "Events"], ascending=[True, False]).drop(columns=["_rank"]).reset_index(drop=True)


def case_summary(case_board: pd.DataFrame) -> dict[str, Any]:
    if case_board is None or case_board.empty:
        return {"cases": 0, "p1": 0, "p2": 0, "top_owner": "—", "status": "NO CASES"}
    counts = case_board["Priority"].value_counts().to_dict()
    owner = case_board["Owner"].value_counts().idxmax() if "Owner" in case_board else "SOC"
    status = "CONTAINMENT" if counts.get("P1", 0) else "INVESTIGATE" if counts.get("P2", 0) else "MONITOR"
    return {
        "cases": int(len(case_board)),
        "p1": int(counts.get("P1", 0)),
        "p2": int(counts.get("P2", 0)),
        "top_owner": str(owner),
        "status": status,
    }


def cases_markdown(case_board: pd.DataFrame) -> str:
    if case_board is None or case_board.empty:
        return "# AIES NIDS Case Board\n\nNo active cases.\n"
    lines = ["# AIES NIDS Case Board", ""]
    for _, row in case_board.iterrows():
        lines.extend([
            f"## {row['Case']} | {row['Priority']} | {row['Class']} on {row['Service']}",
            f"- Status: {row['Status']}",
            f"- Events: {row['Events']} ({row['Critical']} critical, {row['High']} high)",
            f"- MITRE: {row['MITRE']}",
            f"- Window: {row['Window']}",
            f"- Owner: {row['Owner']}",
            f"- First action: {row['First action']}",
            "",
        ])
    return "\n".join(lines)
