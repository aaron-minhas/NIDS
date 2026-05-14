"""
Forward-chaining rule-based expert system for NIDS advisory.

This is the "Expert System" half of the AIES project: turns a raw classifier
prediction into actionable security guidance via 10 rules with confidence
weights and explainability hooks.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Informational": 0}


@dataclass
class Fact:
    predicted_class: str
    confidence: float = 0.0
    features: dict = field(default_factory=dict)
    raw_label: str | None = None
    flow_id: str | None = None


@dataclass
class Advisory:
    rule_id: str
    title: str
    severity: str
    description: str
    recommended_actions: list
    confidence: float
    because: str
    related_features: list = field(default_factory=list)

    @property
    def severity_rank(self):
        return SEVERITY_RANK.get(self.severity, 0)


@dataclass
class Rule:
    rule_id: str
    description: str
    when: Callable
    then: Callable
    base_confidence: float = 1.0


def _feat(fact, name, default=0.0):
    v = fact.features.get(name)
    if v is None: return default
    try: return float(v)
    except (ValueError, TypeError): return default


def _build_rules():
    rules = []

    rules.append(Rule(
        rule_id="R-DDoS-001",
        description="Classifier flagged DDoS",
        when=lambda f: f.predicted_class == "DDoS",
        then=lambda f: Advisory(
            rule_id="R-DDoS-001",
            title="Distributed Denial-of-Service in progress",
            severity="Critical",
            description="Multiple sources are saturating the target with traffic. Service availability is at immediate risk.",
            recommended_actions=[
                "Engage upstream ISP / cloud scrubbing service (Cloudflare, AWS Shield).",
                "Activate rate-limiting on the perimeter (SYN cookies, connection limits).",
                "Switch DNS to a backup VIP if origin is unreachable.",
                "Capture pcap for post-incident attribution.",
            ],
            confidence=min(1.0, 0.6 + 0.4 * f.confidence),
            because=f"Predicted class = DDoS (model confidence {f.confidence:.2f}). Flow Packets/s = {_feat(f, 'Flow Packets/s'):.0f}, Flow Bytes/s = {_feat(f, 'Flow Bytes/s'):.0f}.",
            related_features=["Flow Packets/s", "Flow Bytes/s", "SYN Flag Count"],
        ),
    ))

    rules.append(Rule(
        rule_id="R-DoS-001",
        description="Classifier flagged DoS family",
        when=lambda f: f.predicted_class == "DoS",
        then=lambda f: Advisory(
            rule_id="R-DoS-001",
            title="Single-source Denial-of-Service",
            severity="High",
            description="A single source is exhausting server resources via abnormal request patterns (slow-read, half-open, or flood).",
            recommended_actions=[
                "Block the offending source IP at the edge firewall.",
                "Tune connection timeouts (Apache RequestReadTimeout; Nginx client_body_timeout).",
                "Enable mod_evasive / fail2ban with low thresholds.",
                "Patch known DoS-amplification CVEs (Heartbleed: CVE-2014-0160).",
            ],
            confidence=min(1.0, 0.6 + 0.4 * f.confidence),
            because=f"Predicted class = DoS (model confidence {f.confidence:.2f}). Active Mean = {_feat(f, 'Active Mean'):.0f} us, Idle Mean = {_feat(f, 'Idle Mean'):.0f} us.",
            related_features=["Flow Duration", "Active Mean", "Idle Mean", "PSH Flag Count"],
        ),
    ))

    rules.append(Rule(
        rule_id="R-Scan-001",
        description="Classifier flagged PortScan reconnaissance",
        when=lambda f: f.predicted_class == "PortScan",
        then=lambda f: Advisory(
            rule_id="R-Scan-001",
            title="Reconnaissance port scan detected",
            severity="Medium",
            description="An external host is enumerating open services. Usually precedes targeted exploitation within minutes-to-hours.",
            recommended_actions=[
                "Block the source IP in the perimeter firewall for 24h minimum.",
                "Cross-reference IP against threat-intel feeds (AbuseIPDB, AlienVault OTX).",
                "Audit which ports were probed; close anything unintentionally exposed.",
                "Increase IDS sensitivity for the source's /24 for 24 hours.",
            ],
            confidence=min(1.0, 0.55 + 0.45 * f.confidence),
            because=f"Predicted class = PortScan (confidence {f.confidence:.2f}). Total Fwd Packets = {_feat(f, 'Total Fwd Packets'):.0f}, SYN Flag Count = {_feat(f, 'SYN Flag Count'):.0f}.",
            related_features=["Total Fwd Packets", "SYN Flag Count", "Destination Port", "Flow Duration"],
        ),
    ))

    rules.append(Rule(
        rule_id="R-BF-001",
        description="Classifier flagged credential-guessing brute force",
        when=lambda f: f.predicted_class == "BruteForce",
        then=lambda f: Advisory(
            rule_id="R-BF-001",
            title="Authentication brute-force attempt",
            severity="High",
            description="Repeated authentication attempts against FTP/SSH/web logins. Without lockout, leads to account takeover.",
            recommended_actions=[
                "Enforce account lockout (5 attempts / 15-min window).",
                "Rotate SSH/FTP to key-based or certificate auth.",
                "Enable MFA on every admin and service account.",
                "Deploy fail2ban with bantime 24h after 3 fails.",
                "Audit successful logins from source over last 30 days.",
            ],
            confidence=min(1.0, 0.65 + 0.35 * f.confidence),
            because=f"Predicted class = BruteForce (confidence {f.confidence:.2f}). Destination Port = {int(_feat(f, 'Destination Port'))}, Total Fwd Packets = {_feat(f, 'Total Fwd Packets'):.0f}.",
            related_features=["Destination Port", "Total Fwd Packets", "Flow Duration"],
        ),
    ))

    rules.append(Rule(
        rule_id="R-Bot-001",
        description="Classifier flagged botnet C2 beaconing",
        when=lambda f: f.predicted_class == "Botnet",
        then=lambda f: Advisory(
            rule_id="R-Bot-001",
            title="Botnet command-and-control activity",
            severity="Critical",
            description="An internal host is beaconing to a suspected C2. Host is almost certainly compromised.",
            recommended_actions=[
                "Quarantine the source host from the network IMMEDIATELY.",
                "Pull memory + disk forensics before reimaging.",
                "Sinkhole the destination IP/domain at the DNS resolver.",
                "Hunt for lateral-movement indicators on adjacent hosts.",
                "Notify SIRT and start the IR playbook.",
            ],
            confidence=min(1.0, 0.7 + 0.3 * f.confidence),
            because=f"Predicted class = Botnet (confidence {f.confidence:.2f}). Idle Mean = {_feat(f, 'Idle Mean'):.0f} us.",
            related_features=["Flow Duration", "Idle Mean", "Bwd Packets/s"],
        ),
    ))

    rules.append(Rule(
        rule_id="R-Web-001",
        description="Classifier flagged web-application attack",
        when=lambda f: f.predicted_class == "WebAttack",
        then=lambda f: Advisory(
            rule_id="R-Web-001",
            title="Web application attack (XSS / SQLi / brute)",
            severity="High",
            description="Malicious payload patterns detected against a web service -- likely SQLi, XSS, or login brute-force.",
            recommended_actions=[
                "Enable / tune the WAF (ModSecurity OWASP CRS, Cloudflare WAF).",
                "Audit input validation on the targeted endpoint(s).",
                "Run sqlmap / OWASP ZAP against the same endpoints to confirm exploitability.",
                "Patch vulnerable libraries; re-run SAST scan (semgrep, snyk).",
            ],
            confidence=min(1.0, 0.55 + 0.45 * f.confidence),
            because=f"Predicted class = WebAttack (confidence {f.confidence:.2f}). Destination Port = {int(_feat(f, 'Destination Port'))}.",
            related_features=["Destination Port", "Total Length of Fwd Packets"],
        ),
    ))

    rules.append(Rule(
        rule_id="R-Inf-001",
        description="Classifier flagged insider / infiltration",
        when=lambda f: f.predicted_class == "Infiltration",
        then=lambda f: Advisory(
            rule_id="R-Inf-001",
            title="Insider infiltration / data exfil",
            severity="Critical",
            description="Anomalous internal-to-external transfer pattern consistent with data exfiltration or attacker pivot.",
            recommended_actions=[
                "Snapshot affected host and isolate from network.",
                "Check DLP logs for sensitive-data egress in last 7 days.",
                "Review privileged-account audit trails for unusual access.",
                "Trigger insider-threat response procedure.",
            ],
            confidence=min(1.0, 0.6 + 0.4 * f.confidence),
            because=f"Predicted class = Infiltration (confidence {f.confidence:.2f}).",
            related_features=["Flow Duration", "Total Length of Bwd Packets"],
        ),
    ))

    rules.append(Rule(
        rule_id="R-Benign-001",
        description="Traffic looks normal",
        when=lambda f: f.predicted_class == "BENIGN",
        then=lambda f: Advisory(
            rule_id="R-Benign-001",
            title="No threat indicators",
            severity="Informational",
            description="Flow features are consistent with normal traffic.",
            recommended_actions=["No action required. Continue monitoring."],
            confidence=f.confidence,
            because=f"Classifier confidence in BENIGN = {f.confidence:.2f}",
        ),
    ))

    rules.append(Rule(
        rule_id="R-Unsure-001",
        description="Low classifier confidence",
        when=lambda f: f.confidence > 0 and f.confidence < 0.55 and f.predicted_class != "BENIGN",
        then=lambda f: Advisory(
            rule_id="R-Unsure-001",
            title="Low-confidence detection -- manual review",
            severity="Medium",
            description="Classifier produced a non-benign label with low confidence. False-positive risk elevated.",
            recommended_actions=[
                "Pull the raw pcap for this 5-tuple and inspect manually.",
                "Cross-check with secondary IDS (Suricata / Zeek).",
                "Do not auto-block; tag for analyst review queue.",
            ],
            confidence=1.0 - f.confidence,
            because=f"Classifier confidence only {f.confidence:.2f} -- below 0.55 threshold.",
        ),
    ))

    # FIX (Codex Bug #5): R-Rate-001 must NEVER fire on BENIGN flows.
    # Conflating rate anomaly with severity High when classifier says BENIGN creates
    # misleading "max_severity=High" summaries on benign-dominated traffic.
    # If we don't trust the BENIGN call, that's what R-Unsure-001 is for.
    rules.append(Rule(
        rule_id="R-Rate-001",
        description="Extreme flow rate -- only fires on non-BENIGN predictions",
        when=lambda f: (
            _feat(f, "Flow Packets/s") > 100_000
            and f.predicted_class != "BENIGN"
        ),
        then=lambda f: Advisory(
            rule_id="R-Rate-001",
            title="Extreme packet rate -- investigate amplification",
            severity="High",
            description="Flow rate exceeds 100,000 pps -- far above any legitimate client. Likely amplification or volumetric attack.",
            recommended_actions=[
                "Check whether source is on a known reflector list.",
                "Apply BCP38 ingress filtering at upstream router.",
                "Rate-limit the offending 5-tuple at the edge.",
            ],
            confidence=0.9,
            because=f"Flow Packets/s = {_feat(f, 'Flow Packets/s'):.0f} (above 100k) on non-BENIGN or low-confidence flow.",
            related_features=["Flow Packets/s", "Flow Bytes/s"],
        ),
    ))

    return rules


class ExpertAdvisorySystem:
    def __init__(self):
        self.rules = _build_rules()

    def evaluate(self, fact):
        out = []
        for rule in self.rules:
            try:
                if rule.when(fact):
                    out.append(rule.then(fact))
            except Exception:
                continue
        out.sort(key=lambda a: (-a.severity_rank, -a.confidence))
        return out

    def evaluate_batch(self, facts):
        return [self.evaluate(f) for f in facts]

    def summarise(self, advisories):
        if not advisories:
            return {"top_severity": "Informational", "rules_fired": 0}
        return {
            "top_severity": advisories[0].severity,
            "top_title": advisories[0].title,
            "rules_fired": len(advisories),
            "rule_ids": [a.rule_id for a in advisories],
            "max_confidence": max(a.confidence for a in advisories),
        }
