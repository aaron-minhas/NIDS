"""Mission-control analytics for the AIES NIDS dashboard.

The UI should not only say "this flow is DDoS"; it should explain the
operational picture an analyst would care about: data quality, campaign shape,
priority, attack surface, and response plan.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Informational": 0}

CLASS_PLAYBOOK = {
    "DDoS": {
        "priority": "P1",
        "owner": "Network / Edge",
        "objective": "Preserve service availability",
        "actions": [
            "Engage upstream scrubbing or CDN DDoS protection.",
            "Apply edge rate limits for abusive 5-tuples and high-volume ASNs.",
            "Capture packet samples for post-incident attribution.",
        ],
    },
    "DoS": {
        "priority": "P1",
        "owner": "Network / Platform",
        "objective": "Stop resource exhaustion",
        "actions": [
            "Block or throttle the dominant source at the perimeter.",
            "Tune server connection and request timeouts.",
            "Check application and OS resource saturation metrics.",
        ],
    },
    "Botnet": {
        "priority": "P1",
        "owner": "Incident Response",
        "objective": "Contain compromised hosts",
        "actions": [
            "Quarantine suspected hosts before reimaging.",
            "Sinkhole suspicious domains or destinations at DNS.",
            "Hunt for the same beacon pattern across adjacent hosts.",
        ],
    },
    "BruteForce": {
        "priority": "P2",
        "owner": "Identity / SOC",
        "objective": "Prevent account takeover",
        "actions": [
            "Enable lockout or step-up MFA on targeted services.",
            "Block the attacking source for a fixed investigation window.",
            "Audit successful logins from the same origin.",
        ],
    },
    "WebAttack": {
        "priority": "P2",
        "owner": "AppSec / Web",
        "objective": "Protect public-facing applications",
        "actions": [
            "Turn on WAF rules for SQLi, XSS, and command injection.",
            "Review vulnerable endpoints and request payload logs.",
            "Patch or virtual-patch exposed frameworks.",
        ],
    },
    "Infiltration": {
        "priority": "P1",
        "owner": "Incident Response / DLP",
        "objective": "Stop lateral movement and data loss",
        "actions": [
            "Isolate affected host and preserve volatile evidence.",
            "Review DLP and privilege audit logs for the last 7 days.",
            "Rotate credentials used by the suspected account.",
        ],
    },
    "PortScan": {
        "priority": "P3",
        "owner": "Network / Vulnerability Management",
        "objective": "Reduce exposed attack surface",
        "actions": [
            "Inventory probed services and close unintended exposure.",
            "Temporarily block source networks with repeated scans.",
            "Schedule vulnerability validation on discovered ports.",
        ],
    },
}


def _pct(part: float, whole: float) -> float:
    return round((float(part) / float(whole) * 100.0), 2) if whole else 0.0


def _safe_value_counts(series: pd.Series | None) -> dict[str, int]:
    if series is None or series.empty:
        return {}
    return {str(k): int(v) for k, v in series.value_counts().to_dict().items()}


def severity_counts_from_enriched(enriched: pd.DataFrame, include_info: bool = False) -> dict[str, int]:
    """Return severity counts from the current in-memory analysis result."""
    if enriched is None or enriched.empty or "severity" not in enriched.columns:
        return {}
    s = enriched["severity"].fillna("Informational").astype(str)
    if not include_info:
        s = s[s != "Informational"]
    return _safe_value_counts(s)


def event_feed_from_enriched(enriched: pd.DataFrame, limit: int = 500) -> pd.DataFrame:
    """Convert current session enriched rows into the same shape as alert DB rows."""
    if enriched is None or enriched.empty:
        return pd.DataFrame(columns=["ts_utc", "severity", "pred_class", "title", "confidence"])
    df = enriched.copy()
    df["_rank"] = df["severity"].map(SEVERITY_RANK).fillna(0).astype(int)
    df = df[df["severity"] != "Informational"].sort_values(
        ["_rank", "confidence"], ascending=[False, False]
    ).head(limit)
    return pd.DataFrame({
        "ts_utc": df["flow_idx"].map(lambda x: f"FLOW {int(x):06d}"),
        "severity": df["severity"].astype(str),
        "pred_class": df["predicted_class"].astype(str),
        "title": df["advisory_title"].astype(str),
        "confidence": df["confidence"].astype(float),
    })


def build_data_quality_report(raw_df: pd.DataFrame, sample_profile: dict[str, int] | None = None) -> dict[str, Any]:
    """Score whether the analysed batch is representative and clean enough."""
    if raw_df is None or raw_df.empty:
        return {
            "score": 0,
            "grade": "No data",
            "rows": 0,
            "columns": 0,
            "missing_pct": 0.0,
            "duplicate_pct": 0.0,
            "infinite_values": 0,
            "distinct_labels": 0,
            "notes": ["No rows were available for analysis."],
        }

    rows, cols = int(raw_df.shape[0]), int(raw_df.shape[1])
    missing_cells = int(raw_df.isna().sum().sum())
    missing_pct = _pct(missing_cells, rows * max(cols, 1))
    duplicate_rows = int(raw_df.duplicated().sum())
    duplicate_pct = _pct(duplicate_rows, rows)

    numeric = raw_df.select_dtypes(include=[np.number])
    infinite_values = 0
    if not numeric.empty:
        arr = numeric.to_numpy(dtype=float, copy=False)
        infinite_values = int(np.isinf(arr).sum())

    profile = sample_profile or {}
    distinct_labels = len([v for v in profile.values() if int(v) > 0])

    score = 100.0
    score -= min(35.0, missing_pct * 1.2)
    score -= min(20.0, duplicate_pct * 2.0)
    score -= min(20.0, infinite_values * 0.5)
    if not profile:
        score -= 10.0
    elif distinct_labels <= 1:
        score -= 18.0
    elif distinct_labels < 4:
        score -= 6.0

    score = int(max(0, min(100, round(score))))
    grade = "Excellent" if score >= 90 else "Good" if score >= 75 else "Review" if score >= 55 else "Weak"

    notes = []
    if missing_pct > 1:
        notes.append(f"{missing_pct:.2f}% cells are missing; imputer handled them for inference.")
    if duplicate_pct > 1:
        notes.append(f"{duplicate_pct:.2f}% rows are duplicates; review sample source if this is unexpected.")
    if infinite_values:
        notes.append(f"{infinite_values} infinite numeric values were detected before preprocessing.")
    if distinct_labels <= 1 and profile:
        notes.append("Selected sample has one visible source label; Mixed Attack Lab gives a richer demo.")
    if not notes:
        notes.append("Batch is clean enough for demo-grade analysis.")

    return {
        "score": score,
        "grade": grade,
        "rows": rows,
        "columns": cols,
        "missing_pct": missing_pct,
        "duplicate_pct": duplicate_pct,
        "infinite_values": infinite_values,
        "distinct_labels": distinct_labels,
        "profile": profile,
        "notes": notes,
    }


def build_command_brief(
    enriched: pd.DataFrame,
    summary: dict[str, Any] | None,
    quality: dict[str, Any] | None,
    source_label: str = "",
    sampling_mode: str = "",
    elapsed_sec: float = 0.0,
) -> dict[str, Any]:
    """Create an executive SOC brief from enriched predictions."""
    if enriched is None or enriched.empty:
        return {
            "status": "NO DATA",
            "headline": "Run an analysis to generate the command brief.",
            "key_findings": [],
            "next_actions": [],
        }

    summary = summary or {}
    quality = quality or {}
    total = int(len(enriched))
    attack_df = enriched[enriched["predicted_class"] != "BENIGN"].copy()
    attack_count = int(len(attack_df))
    attack_ratio = attack_count / total if total else 0.0

    sev_counts = severity_counts_from_enriched(enriched, include_info=True)
    critical = int(sev_counts.get("Critical", 0))
    high = int(sev_counts.get("High", 0))
    medium = int(sev_counts.get("Medium", 0))
    low_conf = int((enriched["confidence"].astype(float) < 0.65).sum()) if "confidence" in enriched else 0
    median_conf = float(enriched["confidence"].median()) if "confidence" in enriched else 0.0

    class_counts = _safe_value_counts(attack_df["predicted_class"]) if not attack_df.empty else {}
    top_attack, top_attack_count = next(iter(class_counts.items()), ("None", 0))
    service_counts = _safe_value_counts(attack_df["service"]) if "service" in attack_df else {}
    country_counts = _safe_value_counts(attack_df["country"]) if "country" in attack_df else {}
    top_service = next(iter(service_counts.items()), ("Unknown", 0))
    top_country = next(iter(country_counts.items()), ("Unknown", 0))
    technique_count = int(attack_df["mitre_id"].nunique()) if "mitre_id" in attack_df else 0

    if critical:
        status = "CONTAINMENT REQUIRED"
    elif high:
        status = "INVESTIGATE NOW"
    elif medium:
        status = "ELEVATED WATCH"
    else:
        status = "NORMAL MONITORING"

    headline = (
        f"{attack_count:,} non-BENIGN flows across {len(class_counts)} attack families; "
        f"{top_attack} is the leading class ({top_attack_count:,} flows)."
        if attack_count
        else "No non-BENIGN flows detected in the current batch."
    )

    key_findings = [
        f"Attack ratio is {attack_ratio:.1%} from {total:,} analysed flows.",
        f"Severity mix: {critical:,} critical, {high:,} high, {medium:,} medium.",
        f"Dominant exposed service is {top_service[0]} ({top_service[1]:,} flows).",
        f"Top synthetic origin is {top_country[0]} ({top_country[1]:,} flows).",
        f"{technique_count} MITRE techniques mapped with median model confidence {median_conf:.3f}.",
    ]
    if low_conf:
        key_findings.append(f"{low_conf:,} low-confidence flows should be manually reviewed.")

    next_actions = []
    for cls in class_counts.keys():
        plan = CLASS_PLAYBOOK.get(cls)
        if not plan:
            continue
        for action in plan["actions"]:
            if action not in next_actions:
                next_actions.append(action)
        if len(next_actions) >= 7:
            break

    return {
        "status": status,
        "headline": headline,
        "source": source_label,
        "sampling": sampling_mode,
        "elapsed_sec": round(float(elapsed_sec or 0.0), 2),
        "total_flows": total,
        "attack_count": attack_count,
        "attack_ratio": attack_ratio,
        "critical": critical,
        "high": high,
        "medium": medium,
        "low_confidence": low_conf,
        "median_confidence": round(median_conf, 3),
        "top_attack": top_attack,
        "top_attack_count": top_attack_count,
        "top_service": top_service[0],
        "top_country": top_country[0],
        "quality_score": quality.get("score", 0),
        "quality_grade": quality.get("grade", "Unknown"),
        "key_findings": key_findings,
        "next_actions": next_actions or ["Continue monitoring; no urgent containment action is required."],
    }


def build_campaign_timeline(enriched: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    """Bucket flows by order to show how the scenario unfolds over time."""
    if enriched is None or enriched.empty:
        return pd.DataFrame(columns=["Window", "Dominant stage", "Top class", "Events", "Critical", "High"])
    df = enriched[enriched["predicted_class"] != "BENIGN"].copy()
    if df.empty:
        return pd.DataFrame(columns=["Window", "Dominant stage", "Top class", "Events", "Critical", "High"])
    df["flow_idx"] = df["flow_idx"].astype(int)
    bins = max(1, min(int(bins), len(df)))
    df["_window"] = pd.cut(df["flow_idx"], bins=bins, duplicates="drop")
    rows = []
    for idx, part in df.groupby("_window", observed=True):
        stage_counts = Counter(part["kill_chain_stage"].fillna("Unknown"))
        class_counts = Counter(part["predicted_class"].fillna("Unknown"))
        lo = int(part["flow_idx"].min())
        hi = int(part["flow_idx"].max())
        rows.append({
            "Window": f"{lo}-{hi}",
            "Dominant stage": stage_counts.most_common(1)[0][0],
            "Top class": class_counts.most_common(1)[0][0],
            "Events": int(len(part)),
            "Critical": int((part["severity"] == "Critical").sum()),
            "High": int((part["severity"] == "High").sum()),
        })
    return pd.DataFrame(rows)


def build_attack_surface(enriched: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    """Summarise services and ports receiving attack traffic."""
    if enriched is None or enriched.empty:
        return pd.DataFrame(columns=["Service", "Port", "Events", "Critical/High", "Classes"])
    df = enriched[enriched["predicted_class"] != "BENIGN"].copy()
    if df.empty:
        return pd.DataFrame(columns=["Service", "Port", "Events", "Critical/High", "Classes"])
    group_cols = ["service", "dest_port"]
    rows = []
    for (service, port), part in df.groupby(group_cols, dropna=False):
        classes = ", ".join(part["predicted_class"].value_counts().head(3).index.astype(str))
        rows.append({
            "Service": str(service),
            "Port": int(port) if pd.notna(port) else 0,
            "Events": int(len(part)),
            "Critical/High": int(part["severity"].isin(["Critical", "High"]).sum()),
            "Classes": classes,
        })
    return pd.DataFrame(rows).sort_values(["Critical/High", "Events"], ascending=False).head(limit)


def build_remediation_plan(enriched: pd.DataFrame) -> pd.DataFrame:
    """Deduplicated response plan based on the classes that actually appeared."""
    if enriched is None or enriched.empty:
        return pd.DataFrame(columns=["Priority", "Class", "Owner", "Objective", "First action"])
    present = [
        c for c in enriched["predicted_class"].value_counts().index.astype(str).tolist()
        if c != "BENIGN" and c in CLASS_PLAYBOOK
    ]
    rows = []
    for cls in present:
        plan = CLASS_PLAYBOOK[cls]
        rows.append({
            "Priority": plan["priority"],
            "Class": cls,
            "Owner": plan["owner"],
            "Objective": plan["objective"],
            "First action": plan["actions"][0],
        })
    rank = {"P1": 1, "P2": 2, "P3": 3}
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_rank"] = out["Priority"].map(rank).fillna(9)
    return out.sort_values(["_rank", "Class"]).drop(columns=["_rank"]).reset_index(drop=True)


def _markdown_table(df: pd.DataFrame | None, columns: list[str] | None = None, limit: int = 10) -> list[str]:
    if df is None or df.empty:
        return ["_No rows._"]

    view = df.copy()
    if columns:
        existing = [c for c in columns if c in view.columns]
        view = view[existing] if existing else view
    view = view.head(limit)

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


def build_brief_markdown(
    brief: dict[str, Any],
    quality: dict[str, Any],
    remediation: pd.DataFrame,
    timeline: pd.DataFrame | None = None,
    surface: pd.DataFrame | None = None,
    case_board: pd.DataFrame | None = None,
    model_risk: dict[str, Any] | None = None,
) -> str:
    """Portable analyst brief for viva, SOC handoff, or report appendix."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# AIES NIDS Analyst Brief",
        "",
        f"Generated: {ts}",
        f"Status: {brief.get('status', 'UNKNOWN')}",
        f"Source: {brief.get('source', '-')}",
        f"Sampling: {brief.get('sampling', '-')}",
        "",
        "## Executive Summary",
        brief.get("headline", "No headline available."),
        "",
        "## Key Findings",
    ]
    for item in brief.get("key_findings", []):
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Data Quality",
        f"- Score: {quality.get('score', 0)}/100 ({quality.get('grade', 'Unknown')})",
        f"- Rows: {quality.get('rows', 0):,}",
        f"- Columns: {quality.get('columns', 0):,}",
        f"- Missing cells: {quality.get('missing_pct', 0):.2f}%",
        f"- Duplicates: {quality.get('duplicate_pct', 0):.2f}%",
        "",
        "## Model Risk Gate",
    ])
    if model_risk:
        lines.extend([
            f"- Status: {model_risk.get('status', 'UNKNOWN')}",
            f"- Median confidence: {model_risk.get('median_confidence', 0):.4f}",
            f"- P10 confidence: {model_risk.get('p10_confidence', 0):.4f}",
            f"- Low-confidence flows: {model_risk.get('low_confidence', 0):,}",
            f"- Review rate: {model_risk.get('review_rate', 0):.2%}",
        ])
    else:
        lines.append("- Model risk was not available for this run.")
    lines.extend([
        "",
        "## Campaign Timeline",
        *_markdown_table(timeline, ["Window", "Dominant stage", "Top class", "Events", "Critical", "High"], limit=8),
        "",
        "## Attack Surface",
        *_markdown_table(surface, ["Service", "Port", "Events", "Critical/High", "Classes"], limit=10),
        "",
        "## Case Board",
        *_markdown_table(
            case_board,
            ["Case", "Priority", "Status", "Class", "Service", "MITRE", "Events", "Owner", "First action"],
            limit=12,
        ),
        "",
        "## Response Plan",
    ])
    if remediation is not None and not remediation.empty:
        for _, row in remediation.iterrows():
            lines.append(
                f"- [{row['Priority']}] {row['Class']} | {row['Owner']} | {row['First action']}"
            )
    else:
        lines.append("- Continue monitoring.")
    lines.extend(["", "## Analyst Next Actions"])
    for action in brief.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"
