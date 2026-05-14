"""SIEM-grade alert export formats: STIX 2.1 bundle, ArcSight CEF, RFC 5424 syslog."""
from __future__ import annotations
import json, uuid, time
from datetime import datetime, timezone
from typing import List, Dict
import pandas as pd

PRODUCER = {
    "name": "AIES NIDS",
    "version": "2.0",
    "vendor": "NED University CT-361 AIES CCP",
    "stix_identity_id": "identity--00000000-0000-4000-a000-aiesnids00001",
}

# CEF severity scale (0-10) from ArcSight ESM standard
CEF_SEVERITY = {"Critical": 10, "High": 8, "Medium": 5, "Low": 3, "Informational": 1}

# Syslog severity (RFC 5424) — lower = higher priority
SYSLOG_SEVERITY = {"Critical": 2, "High": 3, "Medium": 4, "Low": 5, "Informational": 6}


def to_stix_bundle(enriched_df: pd.DataFrame, exclude_benign: bool = True) -> dict:
    """Convert enriched alerts dataframe to STIX 2.1 bundle for SIEM ingestion.
    Each alert -> indicator + sighting + observed-data SDOs.
    """
    df = enriched_df.copy()
    if exclude_benign:
        df = df[df["predicted_class"] != "BENIGN"]

    objects = []
    # Producer identity
    objects.append({
        "type": "identity",
        "spec_version": "2.1",
        "id": PRODUCER["stix_identity_id"],
        "created": "2026-01-01T00:00:00.000Z",
        "modified": "2026-01-01T00:00:00.000Z",
        "name": PRODUCER["name"],
        "identity_class": "system",
        "description": f"{PRODUCER['vendor']} -- hybrid ML+expert NIDS",
    })

    now_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    for _, r in df.iterrows():
        ind_id = f"indicator--{uuid.uuid4()}"
        sight_id = f"sighting--{uuid.uuid4()}"
        obs_id = f"observed-data--{uuid.uuid4()}"

        # Indicator SDO with MITRE ATT&CK pattern reference
        objects.append({
            "type": "indicator",
            "spec_version": "2.1",
            "id": ind_id,
            "created": now_iso, "modified": now_iso,
            "created_by_ref": PRODUCER["stix_identity_id"],
            "name": f"{r['predicted_class']} -- {r['advisory_title']}",
            "description": f"ML-detected {r['predicted_class']} flow at confidence {r['confidence']}. "
                           f"MITRE {r['mitre_id']} {r['mitre_technique']} ({r['tactic']}).",
            "indicator_types": ["malicious-activity"],
            "pattern_type": "stix",
            "pattern": f"[network-traffic:dst_port = '{r['dest_port']}']",
            "valid_from": now_iso,
            "labels": [r["predicted_class"], r["mitre_id"], r["kill_chain_stage"]],
            "kill_chain_phases": [{
                "kill_chain_name": "lockheed-martin-cyber-kill-chain",
                "phase_name": str(r["kill_chain_stage"]).lower().replace(" ", "-").replace("&", "and"),
            }],
        })
        # Observed-data: source country, ASN, dest service
        objects.append({
            "type": "observed-data",
            "spec_version": "2.1",
            "id": obs_id,
            "created": now_iso, "modified": now_iso,
            "created_by_ref": PRODUCER["stix_identity_id"],
            "first_observed": now_iso, "last_observed": now_iso,
            "number_observed": 1,
            "object_refs": [ind_id],
            "x_aies_origin_country": r["country"],
            "x_aies_origin_asn": r["asn"],
            "x_aies_target_service": r["service"],
            "x_aies_severity": r["severity"],
        })
        # Sighting linking indicator to detection
        objects.append({
            "type": "sighting",
            "spec_version": "2.1",
            "id": sight_id,
            "created": now_iso, "modified": now_iso,
            "created_by_ref": PRODUCER["stix_identity_id"],
            "first_seen": now_iso, "last_seen": now_iso,
            "count": 1,
            "sighting_of_ref": ind_id,
            "where_sighted_refs": [PRODUCER["stix_identity_id"]],
            "observed_data_refs": [obs_id],
        })

    return {
        "type": "bundle",
        "id": f"bundle--{uuid.uuid4()}",
        "objects": objects,
    }


def to_cef(enriched_df: pd.DataFrame, exclude_benign: bool = True) -> str:
    """ArcSight Common Event Format (CEF v0). One alert per line for SIEM ingestion."""
    df = enriched_df.copy()
    if exclude_benign:
        df = df[df["predicted_class"] != "BENIGN"]

    lines = []
    for _, r in df.iterrows():
        sev = CEF_SEVERITY.get(r["severity"], 1)
        sig_id = r["mitre_id"] if r["mitre_id"] != "—" else r["predicted_class"]
        # CEF format: CEF:Version|Vendor|Product|Version|SignatureID|Name|Severity|Extension
        ext = (
            f"act={r['predicted_class']} "
            f"cs1Label=MITRETechnique cs1={r['mitre_technique']} "
            f"cs2Label=KillChain cs2={r['kill_chain_stage']} "
            f"cs3Label=Country cs3={r['country']} "
            f"cs4Label=ASN cs4={r['asn']} "
            f"cs5Label=Service cs5={r['service']} "
            f"cs6Label=Confidence cs6={r['confidence']} "
            f"dpt={r['dest_port']} "
            f"src=synthetic dst=internal"
        )
        line = (f"CEF:0|NEDU|AIES-NIDS|2.0|{sig_id}|"
                f"{r['advisory_title']}|{sev}|{ext}")
        lines.append(line)
    return "\n".join(lines)


def to_syslog(enriched_df: pd.DataFrame, exclude_benign: bool = True, hostname: str = "aies-nids-01") -> str:
    """RFC 5424 syslog format. Facility 13 = log audit."""
    df = enriched_df.copy()
    if exclude_benign:
        df = df[df["predicted_class"] != "BENIGN"]

    facility = 13
    lines = []
    for _, r in df.iterrows():
        sev = SYSLOG_SEVERITY.get(r["severity"], 6)
        pri = facility * 8 + sev
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        msg = (f"AIES-NIDS {r['predicted_class']} ({r['mitre_id']}) "
               f"sev={r['severity']} conf={r['confidence']} "
               f"origin={r['country']}/{r['asn']} target={r['service']}/{r['dest_port']} "
               f"chain={r['kill_chain_stage']}")
        lines.append(f"<{pri}>1 {ts} {hostname} aies-nids - - - {msg}")
    return "\n".join(lines)


def to_json_export(enriched_df: pd.DataFrame, exclude_benign: bool = True, pretty: bool = True) -> str:
    """JSON export for generic SIEM/SOAR ingestion (Splunk, Elastic, Sentinel)."""
    df = enriched_df.copy()
    if exclude_benign:
        df = df[df["predicted_class"] != "BENIGN"]
    records = df.to_dict(orient="records")
    payload = {
        "producer": PRODUCER,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "alert_count": len(records),
        "alerts": records,
    }
    return json.dumps(payload, indent=2 if pretty else None, default=str)