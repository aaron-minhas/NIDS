"""Compliance framework mapping: NIST CSF, ISO 27001 Annex A, CIS Controls v8."""
from __future__ import annotations
from typing import Dict, List
import pandas as pd

# ============================================================================
# NIST Cybersecurity Framework (CSF) v1.1
# 5 Functions -> Categories -> Subcategories
# We map our detection capabilities to relevant subcategories.
# ============================================================================
NIST_CSF: Dict[str, dict] = {
    "DE.AE-1": {"function": "Detect", "category": "Anomalies and Events", "name": "Baseline of network operations established", "covered_by": ["BENIGN"]},
    "DE.AE-2": {"function": "Detect", "category": "Anomalies and Events", "name": "Detected events analyzed for attack targets/methods", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "DE.AE-3": {"function": "Detect", "category": "Anomalies and Events", "name": "Event data aggregated and correlated", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "DE.CM-1": {"function": "Detect", "category": "Continuous Monitoring", "name": "Network monitored to detect potential events", "covered_by": ["BENIGN","DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "DE.CM-4": {"function": "Detect", "category": "Continuous Monitoring", "name": "Malicious code detected", "covered_by": ["Botnet","Infiltration","WebAttack"]},
    "DE.CM-7": {"function": "Detect", "category": "Continuous Monitoring", "name": "Monitoring for unauthorized personnel/connections", "covered_by": ["BruteForce","Infiltration","PortScan"]},
    "DE.DP-4": {"function": "Detect", "category": "Detection Processes", "name": "Event detection information communicated", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "RS.AN-1": {"function": "Respond", "category": "Analysis", "name": "Notifications from detection systems investigated", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "RS.AN-3": {"function": "Respond", "category": "Analysis", "name": "Forensics performed", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "RS.MI-1": {"function": "Respond", "category": "Mitigation", "name": "Incidents contained", "covered_by": ["DDoS","DoS","BruteForce","Botnet"]},
    "PR.AC-5": {"function": "Protect", "category": "Access Control", "name": "Network integrity protected", "covered_by": ["DDoS","DoS","BruteForce","Infiltration"]},
    "PR.IP-1": {"function": "Protect", "category": "Information Protection", "name": "Baseline configuration created/maintained", "covered_by": ["BENIGN"]},
    "ID.RA-1": {"function": "Identify", "category": "Risk Assessment", "name": "Asset vulnerabilities identified", "covered_by": ["PortScan","WebAttack"]},
    "ID.RA-3": {"function": "Identify", "category": "Risk Assessment", "name": "Threats internal/external identified", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
}

# ============================================================================
# ISO/IEC 27001:2022 Annex A controls relevant to NIDS
# (Subset: 14 controls applicable to network monitoring / intrusion detection)
# ============================================================================
ISO_27001: Dict[str, dict] = {
    "A.5.7":  {"name": "Threat intelligence", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "A.5.25": {"name": "Assessment and decision on information security events", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "A.5.26": {"name": "Response to information security incidents", "covered_by": ["DDoS","DoS","BruteForce","Botnet","WebAttack","Infiltration"]},
    "A.5.27": {"name": "Learning from information security incidents", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "A.5.28": {"name": "Collection of evidence", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "A.8.7":  {"name": "Protection against malware", "covered_by": ["Botnet","Infiltration"]},
    "A.8.15": {"name": "Logging", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "A.8.16": {"name": "Monitoring activities", "covered_by": ["BENIGN","DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "A.8.20": {"name": "Network security", "covered_by": ["DDoS","DoS","PortScan","Infiltration"]},
    "A.8.21": {"name": "Security of network services", "covered_by": ["WebAttack","BruteForce","Infiltration"]},
    "A.8.22": {"name": "Segregation of networks", "covered_by": ["Infiltration","Botnet"]},
    "A.8.23": {"name": "Web filtering", "covered_by": ["WebAttack","Botnet"]},
    "A.8.32": {"name": "Change management", "covered_by": ["BENIGN"]},
    "A.5.30": {"name": "ICT readiness for business continuity", "covered_by": ["DDoS","DoS"]},
}

# ============================================================================
# CIS Critical Security Controls v8 (subset relevant to NIDS)
# ============================================================================
CIS_CONTROLS: Dict[str, dict] = {
    "CIS 8.1":  {"name": "Establish and maintain an audit log management process", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "CIS 8.2":  {"name": "Collect audit logs", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "CIS 8.11": {"name": "Conduct audit log reviews", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "CIS 13.1": {"name": "Centralize security event alerting", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "CIS 13.2": {"name": "Deploy host-based intrusion detection (HIDS)", "covered_by": ["BruteForce","Infiltration","Botnet"]},
    "CIS 13.3": {"name": "Deploy network intrusion detection (NIDS)", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "CIS 13.4": {"name": "Perform traffic filtering between network segments", "covered_by": ["DDoS","DoS","Infiltration"]},
    "CIS 13.5": {"name": "Manage access control for remote assets", "covered_by": ["BruteForce","Infiltration"]},
    "CIS 13.6": {"name": "Collect network traffic flow logs", "covered_by": ["DDoS","DoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]},
    "CIS 13.10": {"name": "Perform application layer filtering", "covered_by": ["WebAttack","Botnet"]},
    "CIS 17.1": {"name": "Designate personnel to manage incident handling", "covered_by": ["DDoS","DoS","BruteForce","Botnet","WebAttack","Infiltration"]},
    "CIS 17.7": {"name": "Conduct post-incident reviews", "covered_by": ["DDoS","DoS","BruteForce","Botnet","WebAttack","Infiltration"]},
}

FRAMEWORKS = {
    "NIST CSF v1.1": NIST_CSF,
    "ISO/IEC 27001:2022": ISO_27001,
    "CIS Controls v8": CIS_CONTROLS,
}

def compute_coverage(detected_classes: List[str], framework: str = "NIST CSF v1.1") -> pd.DataFrame:
    """For each control, compute whether system has detected at least one applicable class."""
    fw = FRAMEWORKS.get(framework, NIST_CSF)
    detected_set = set(detected_classes)
    rows = []
    for ctrl_id, info in fw.items():
        covered_classes = set(info["covered_by"])
        intersect = detected_set & covered_classes
        coverage_pct = (len(intersect) / max(len(covered_classes), 1)) * 100
        status = "Active" if intersect else "Not exercised"
        rows.append({
            "control": ctrl_id,
            "function": info.get("function", info.get("category", "—")),
            "name": info["name"],
            "applicable_classes": len(covered_classes),
            "detected_classes": len(intersect),
            "coverage_pct": round(coverage_pct, 1),
            "status": status,
            "evidence": ", ".join(sorted(intersect)) if intersect else "—",
        })
    return pd.DataFrame(rows)

def overall_posture_score(detected_classes: List[str], framework: str = "NIST CSF v1.1") -> dict:
    """Aggregate coverage % across all controls in a framework."""
    df = compute_coverage(detected_classes, framework)
    avg = df["coverage_pct"].mean() if len(df) else 0
    active = (df["status"] == "Active").sum()
    total = len(df)
    if avg >= 75: rating, color = "Strong", "#16A34A"
    elif avg >= 50: rating, color = "Moderate", "#F59E0B"
    elif avg >= 25: rating, color = "Limited", "#EA580C"
    else: rating, color = "Insufficient", "#DC2626"
    return {"framework": framework, "avg_coverage": round(avg, 1),
            "active_controls": int(active), "total_controls": int(total),
            "rating": rating, "color": color}