"""Threat intelligence enrichment: MITRE ATT&CK, Kill Chain, geo-IP, threat scoring."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# ============================================================================
# MITRE ATT&CK MAPPING
# Each CICIDS class -> MITRE technique(s) per the official ATT&CK Enterprise Matrix
# https://attack.mitre.org/matrices/enterprise/
# ============================================================================
MITRE_MAPPING: Dict[str, dict] = {
    "BENIGN": {
        "technique_id": "—",
        "technique_name": "Normal traffic",
        "tactic": "—",
        "tactic_id": "—",
        "description": "No adversarial activity detected.",
    },
    "DDoS": {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "description": "Adversary degrades or blocks availability of targeted resources via flood traffic from many sources.",
    },
    "DoS": {
        "technique_id": "T1499",
        "technique_name": "Endpoint Denial of Service",
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "description": "Adversary degrades availability of an endpoint via resource exhaustion (SYN flood, slowloris, etc).",
    },
    "PortScan": {
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "tactic": "Discovery",
        "tactic_id": "TA0007",
        "description": "Adversary attempts to enumerate exposed services on target hosts to identify attack surface.",
    },
    "BruteForce": {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": "Credential Access",
        "tactic_id": "TA0006",
        "description": "Adversary systematically attempts authentication credentials to gain access (SSH, FTP, web).",
    },
    "Botnet": {
        "technique_id": "T1071",
        "technique_name": "Application Layer Protocol (C2)",
        "tactic": "Command and Control",
        "tactic_id": "TA0011",
        "description": "Compromised host beaconing to attacker C2 infrastructure for instructions.",
    },
    "WebAttack": {
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
        "tactic_id": "TA0001",
        "description": "Adversary exploits weakness in public-facing app (SQLi, XSS, command injection).",
    },
    "Infiltration": {
        "technique_id": "T1078",
        "technique_name": "Valid Accounts (Lateral Movement)",
        "tactic": "Lateral Movement",
        "tactic_id": "TA0008",
        "description": "Adversary uses stolen valid credentials to move laterally and persist in environment.",
    },
}

# ============================================================================
# LOCKHEED MARTIN CYBER KILL CHAIN MAPPING
# 7 stages: Recon -> Weaponize -> Deliver -> Exploit -> Install -> C2 -> Actions
# https://www.lockheedmartin.com/en-us/capabilities/cyber/cyber-kill-chain.html
# ============================================================================
KILL_CHAIN_STAGES = [
    "Reconnaissance",
    "Weaponization",
    "Delivery",
    "Exploitation",
    "Installation",
    "Command & Control",
    "Actions on Objectives",
]

KILL_CHAIN_MAPPING: Dict[str, str] = {
    "BENIGN": "—",
    "PortScan": "Reconnaissance",
    "BruteForce": "Delivery",
    "WebAttack": "Exploitation",
    "Infiltration": "Installation",
    "Botnet": "Command & Control",
    "DDoS": "Actions on Objectives",
    "DoS": "Actions on Objectives",
}

# ============================================================================
# DESTINATION PORT -> SERVICE MAPPING (top targeted services in real attack data)
# ============================================================================
SERVICE_BY_PORT: Dict[int, str] = {
    20: "FTP-Data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
    53: "DNS", 67: "DHCP", 80: "HTTP", 110: "POP3", 135: "RPC",
    139: "NetBIOS", 143: "IMAP", 161: "SNMP", 389: "LDAP",
    443: "HTTPS", 445: "SMB", 465: "SMTPS", 514: "Syslog",
    587: "SMTP-Submission", 636: "LDAPS", 993: "IMAPS", 995: "POP3S",
    1433: "MSSQL", 1521: "Oracle", 3306: "MySQL", 3389: "RDP",
    5432: "PostgreSQL", 5900: "VNC", 6379: "Redis", 8080: "HTTP-Alt",
    8443: "HTTPS-Alt", 9200: "Elasticsearch", 27017: "MongoDB",
}

def port_to_service(port) -> str:
    """Resolve dest port to service name; categorize ephemeral ranges."""
    try:
        p = int(port)
    except (ValueError, TypeError):
        return "Unknown"
    if p in SERVICE_BY_PORT:
        return SERVICE_BY_PORT[p]
    if 1024 <= p <= 49151:
        return f"Registered ({p})"
    if 49152 <= p <= 65535:
        return f"Ephemeral ({p})"
    return f"Reserved ({p})"

# ============================================================================
# SYNTHETIC GEOGRAPHIC ENRICHMENT
# CICIDS2017 strips real IPs. We derive deterministic synthetic origin from flow
# features so the threat map is reproducible and transparent. Distribution based
# on real 2024 threat intelligence reports (CrowdStrike Global Threat Report,
# Mandiant M-Trends): China, Russia, Iran, North Korea, USA most common origins.
# ============================================================================
THREAT_ORIGINS: List[Dict] = [
    {"country": "China",         "iso": "CHN", "lat": 35.86,  "lon": 104.20, "weight": 24},
    {"country": "Russia",        "iso": "RUS", "lat": 61.52,  "lon": 105.32, "weight": 18},
    {"country": "United States", "iso": "USA", "lat": 37.09,  "lon": -95.71, "weight": 14},
    {"country": "Iran",          "iso": "IRN", "lat": 32.43,  "lon": 53.69,  "weight": 9},
    {"country": "North Korea",   "iso": "PRK", "lat": 40.34,  "lon": 127.51, "weight": 7},
    {"country": "Brazil",        "iso": "BRA", "lat": -14.24, "lon": -51.93, "weight": 5},
    {"country": "India",         "iso": "IND", "lat": 20.59,  "lon": 78.96,  "weight": 5},
    {"country": "Vietnam",       "iso": "VNM", "lat": 14.06,  "lon": 108.28, "weight": 4},
    {"country": "Ukraine",       "iso": "UKR", "lat": 48.38,  "lon": 31.17,  "weight": 4},
    {"country": "Romania",       "iso": "ROU", "lat": 45.94,  "lon": 24.97,  "weight": 3},
    {"country": "Turkey",        "iso": "TUR", "lat": 38.96,  "lon": 35.24,  "weight": 3},
    {"country": "Pakistan",      "iso": "PAK", "lat": 30.38,  "lon": 69.35,  "weight": 2},
    {"country": "Indonesia",     "iso": "IDN", "lat": -0.79,  "lon": 113.92, "weight": 2},
]

# ASN simulation: top abusive ASNs from Spamhaus/AbuseIPDB
SIMULATED_ASNS = [
    "AS4134 (Chinanet)", "AS4837 (China Unicom)", "AS8359 (MTS Russia)",
    "AS9009 (M247)", "AS24940 (Hetzner)", "AS14061 (DigitalOcean)",
    "AS16509 (Amazon AWS)", "AS15169 (Google)", "AS39572 (DataWeb Romania)",
    "AS197207 (MCCI Iran)", "AS131279 (Star JV NK)",
]

def _hash_row_features(row) -> int:
    """Deterministic hash from flow features for reproducible synthetic geos."""
    keys = ["Destination Port", "Flow Duration", "Total Fwd Packets",
            "Total Length of Fwd Packets", "Flow IAT Mean"]
    parts = []
    for k in keys:
        v = row.get(k, 0)
        if pd.isna(v) or v in (np.inf, -np.inf):
            v = 0
        parts.append(str(int(float(v)) if isinstance(v, (int, float, np.number)) else hash(str(v))))
    h = hashlib.md5("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)

def geo_enrich(row, predicted_class: str) -> dict:
    """Return synthetic origin country/lat/lon/ASN for a flow row.
    Note: Geographic attribution is illustrative; CICIDS2017 strips real IP fields.
    """
    if predicted_class == "BENIGN":
        return {"country": "Internal", "iso": "—", "lat": 0.0, "lon": 0.0, "asn": "Internal LAN"}
    seed = _hash_row_features(row)
    weights = np.array([o["weight"] for o in THREAT_ORIGINS], dtype=float)
    cum = np.cumsum(weights / weights.sum())
    pick = (seed % 10000) / 10000.0
    idx = int(np.searchsorted(cum, pick))
    origin = THREAT_ORIGINS[min(idx, len(THREAT_ORIGINS) - 1)]
    asn_idx = (seed >> 8) % len(SIMULATED_ASNS)
    # Add small jitter so points don't all stack on one capital
    lat_jit = ((seed >> 16) % 1000 - 500) / 250.0
    lon_jit = ((seed >> 24) % 1000 - 500) / 250.0
    return {
        "country": origin["country"],
        "iso": origin["iso"],
        "lat": origin["lat"] + lat_jit,
        "lon": origin["lon"] + lon_jit,
        "asn": SIMULATED_ASNS[asn_idx],
    }

# ============================================================================
# DEFCON-STYLE THREAT SCORE (0-100) + alert level
# ============================================================================
DEFCON_LEVELS = {
    "DEFCON 5": (0, 20,   "#65A30D", "Normal peacetime posture"),
    "DEFCON 4": (20, 40,  "#3B82F6", "Increased intelligence watch"),
    "DEFCON 3": (40, 60,  "#F59E0B", "Increased force readiness"),
    "DEFCON 2": (60, 85,  "#EA580C", "Armed forces ready for combat"),
    "DEFCON 1": (85, 101, "#DC2626", "Maximum readiness; attack imminent"),
}

# Severity weights for proportional scoring (sum to 1.0 when normalised)
SEVERITY_WEIGHT = {"Critical": 1.0, "High": 0.55, "Medium": 0.20, "Low": 0.05, "Informational": 0.0}

def compute_threat_score(severity_counts: dict, total_flows: int = 0) -> dict:
    """Compute threat score 0-100 based on weighted attack ratio.

    Logic:
      - 0 alerts in any flows -> score 0 (DEFCON 5, peacetime)
      - 100% critical flows -> score 100 (DEFCON 1, attack imminent)
      - 50% critical -> score ~50 (DEFCON 3, increased readiness)
      - Mix of severities weighted: Critical=1.0, High=0.55, Medium=0.20, Low=0.05
    """
    if total_flows <= 0:
        total_flows = max(sum(severity_counts.values()), 1)
    if not severity_counts or all(c == 0 for c in severity_counts.values()):
        return {"score": 0.0, "level": "DEFCON 5", "color": "#65A30D", "message": "Normal peacetime posture"}
    weighted = sum(SEVERITY_WEIGHT.get(s, 0) * c for s, c in severity_counts.items())
    # Score = weighted attack ratio scaled to 0-100. Caps at 100 only when
    # 100% of flows are critical-severity attacks.
    score = min(100.0, max(0.0, (weighted / total_flows) * 100.0))
    level, color, msg = "DEFCON 5", "#65A30D", "Normal peacetime posture"
    for lvl, (lo, hi, c, m) in DEFCON_LEVELS.items():
        if lo <= score < hi:
            level, color, msg = lvl, c, m
            break
    return {"score": round(score, 1), "level": level, "color": color, "message": msg}

# ============================================================================
# ENRICHMENT PIPELINE: takes prediction list -> enriched DataFrame
# ============================================================================
def enrich_predictions(preds: list, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build enriched dataframe with MITRE / kill chain / geo / service for each prediction."""
    rows = []
    raw_df_reset = raw_df.reset_index(drop=True) if hasattr(raw_df, 'reset_index') else raw_df
    for i, p in enumerate(preds):
        cls = p.predicted_class
        mitre = MITRE_MAPPING.get(cls, MITRE_MAPPING["BENIGN"])
        kc = KILL_CHAIN_MAPPING.get(cls, "—")
        try:
            row = raw_df_reset.iloc[i] if i < len(raw_df_reset) else {}
        except Exception:
            row = {}
        geo = geo_enrich(row, cls)
        port = row.get("Destination Port", row.get("destination_port", 0)) if hasattr(row, 'get') else 0
        sev = p.advisories[0].severity if p.advisories else "Informational"
        rows.append({
            "flow_idx": i,
            "predicted_class": cls,
            "confidence": round(p.confidence, 3),
            "severity": sev,
            "advisory_title": p.advisories[0].title if p.advisories else "—",
            "mitre_id": mitre["technique_id"],
            "mitre_technique": mitre["technique_name"],
            "tactic": mitre["tactic"],
            "kill_chain_stage": kc,
            "country": geo["country"],
            "iso": geo["iso"],
            "lat": geo["lat"],
            "lon": geo["lon"],
            "asn": geo["asn"],
            "dest_port": int(port) if port and not pd.isna(port) else 0,
            "service": port_to_service(port),
        })
    return pd.DataFrame(rows)