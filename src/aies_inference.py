"""
AIES NIDS  -  Inference Engine and Educational Tools

This module implements eight core AI Expert Systems concepts taught in
CT-361 AIES (NED University BCIT). Each concept is a self-contained function
or class that can be invoked from the Streamlit UI to produce a live demo.

  1. ForwardChainTrace      -- match-resolve-act cycle, full trace capture
  2. BackwardChainExplainer -- goal-driven WHY proof tree
  3. CertaintyFactor        -- MYCIN-style CF combination (Shortliffe 1976)
  4. WhatIfSandbox          -- counterfactual feature override + re-classify
  5. KnowledgeBaseEditor    -- in-session editable rule store
  6. ModelComparator        -- ML-only vs Rules-only vs Hybrid eval
  7. AdversarialTester      -- Gaussian noise robustness curve
  8. ForensicReportBuilder  -- professional DOCX incident report

References:
  - Russell, Norvig: AI: A Modern Approach (Ch 9, Inference)
  - Shortliffe: Computer-Based Medical Consultations: MYCIN (1976)
  - Buchanan, Shortliffe: Rule-Based Expert Systems (1984)
"""
from __future__ import annotations
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# =====================================================================
# 1. FORWARD CHAINING TRACE
# =====================================================================
@dataclass
class TraceStep:
    cycle: int
    action: str          # "LOAD_WM", "MATCH", "FIRE", "HALT"
    rule_id: str = ""
    rule_desc: str = ""
    matched: bool = False
    severity: str = ""
    title: str = ""
    note: str = ""


def run_traced_inference(flow_features: dict, predicted_class: str, confidence: float) -> List[TraceStep]:
    """Execute every rule in src.expert_system, capturing match-resolve-act steps."""
    from src.expert_system import _build_rules
    rules = _build_rules()

    class _FakeFlow:
        pass
    ff = _FakeFlow()
    ff.flow_index = 0
    ff.predicted_class = predicted_class
    ff.confidence = confidence
    ff.features = flow_features

    trace: List[TraceStep] = []
    fired_count = 0

    trace.append(TraceStep(
        cycle=0, action="LOAD_WM",
        note=f"Loaded {len(flow_features)} features + ML output (class={predicted_class}, conf={confidence:.3f}) into working memory"
    ))

    cycle = 1
    for rule in rules:
        try:
            matched = bool(rule.when(ff))
        except Exception:
            matched = False
        # If matched, call rule.then(ff) to get the Advisory (severity/title come from there)
        adv = None
        if matched:
            try:
                adv = rule.then(ff)
            except Exception:
                adv = None
        sev = adv.severity if adv else ""
        title = adv.title if adv else ""
        trace.append(TraceStep(
            cycle=cycle, action="MATCH",
            rule_id=rule.rule_id, rule_desc=rule.description,
            matched=matched, severity=sev, title=title,
            note="Condition satisfied" if matched else "Condition NOT satisfied",
        ))
        if matched:
            fired_count += 1
            trace.append(TraceStep(
                cycle=cycle, action="FIRE",
                rule_id=rule.rule_id, severity=sev, title=title,
                note=f"Added advisory: {title}",
            ))

    trace.append(TraceStep(
        cycle=cycle + 1, action="HALT",
        note=f"Quiescence reached -- {fired_count} rule(s) fired total",
    ))
    return trace


# =====================================================================
# 2. BACKWARD CHAINING -- "Why?" Engine
# =====================================================================
@dataclass
class WhyNode:
    goal: str
    answer: str
    rule_id: Optional[str] = None
    explanation: str = ""
    is_leaf: bool = False
    children: List["WhyNode"] = field(default_factory=list)


def explain_why(predicted_class: str, confidence: float, fired_rules: list) -> WhyNode:
    """Build a WHY proof tree for a flow's classification."""
    root = WhyNode(
        goal=f"Why is this flow classified as {predicted_class}?",
        answer=predicted_class,
        explanation="Combined ML classifier output + expert system rules",
    )
    ml_node = WhyNode(
        goal="Sub-goal: ML classifier evidence",
        answer=predicted_class,
        explanation=f"RandomForest assigned class={predicted_class} with confidence {confidence:.3f} (top-1 probability)",
        is_leaf=True,
    )
    root.children.append(ml_node)

    for r in fired_rules:
        rule_node = WhyNode(
            goal=f"Sub-goal: rule {r.get('id', '?')} fired",
            answer=r.get("title", "?"),
            rule_id=r.get("id"),
            explanation=r.get("description", ""),
        )
        rule_node.children.append(WhyNode(
            goal="Why did this rule fire?",
            answer="all conditions matched",
            explanation=f"Predicate evaluated TRUE on the flow's working memory: {r.get('description', '')}",
            is_leaf=True,
        ))
        root.children.append(rule_node)

    return root


def render_why_tree_markdown(node: WhyNode, depth: int = 0) -> str:
    """Render a WhyNode tree as Streamlit-friendly markdown (indented bullets)."""
    indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * depth
    bullet = "└" if depth > 0 else "▶"
    lines = [f"{indent}{bullet} **{node.goal}**  →  `{node.answer}`"]
    if node.explanation:
        lines.append(f"{indent}&nbsp;&nbsp;&nbsp;&nbsp;_{node.explanation}_")
    for c in node.children:
        lines.append(render_why_tree_markdown(c, depth + 1))
    return "  \n".join(lines)


# =====================================================================
# 3. CERTAINTY FACTOR (MYCIN)
# =====================================================================
def cf_combine(cf1: float, cf2: float) -> float:
    """Combine two MYCIN certainty factors."""
    if cf1 >= 0 and cf2 >= 0:
        return cf1 + cf2 * (1 - cf1)
    if cf1 <= 0 and cf2 <= 0:
        return cf1 + cf2 * (1 + cf1)
    denom = 1 - min(abs(cf1), abs(cf2))
    if abs(denom) < 1e-9:
        return 1.0 if (cf1 + cf2) >= 0 else -1.0
    return (cf1 + cf2) / denom


def cf_combine_sequence(cfs: List[float]) -> List[Dict[str, Any]]:
    """Combine CFs left-to-right, returning each intermediate step."""
    steps: List[Dict[str, Any]] = []
    if not cfs:
        return steps
    cur = cfs[0]
    steps.append({"step": 0, "cf_in": cfs[0], "cf_running": cur, "formula": f"Initial: CF = {cur:+.3f}"})
    for i, c in enumerate(cfs[1:], start=1):
        prev = cur
        cur = cf_combine(cur, c)
        if prev >= 0 and c >= 0:
            formula = f"CF = {prev:+.3f} + {c:+.3f} * (1 - {prev:+.3f}) = {cur:+.3f}"
        elif prev <= 0 and c <= 0:
            formula = f"CF = {prev:+.3f} + {c:+.3f} * (1 + {prev:+.3f}) = {cur:+.3f}"
        else:
            formula = f"CF = ({prev:+.3f} + {c:+.3f}) / (1 - min(|{prev:.3f}|, |{c:.3f}|)) = {cur:+.3f}"
        steps.append({"step": i, "cf_in": c, "cf_running": cur, "formula": formula})
    return steps


def cf_strength_label(cf: float) -> str:
    a = abs(cf)
    if a >= 0.9: return "Very strong"
    if a >= 0.7: return "Strong"
    if a >= 0.5: return "Moderate"
    if a >= 0.2: return "Weak"
    return "Negligible"


def severity_to_cf(severity: str) -> float:
    return {
        "Critical": 0.95, "High": 0.80, "Medium": 0.55,
        "Low": 0.30, "Informational": 0.10,
    }.get(severity, 0.50)


# =====================================================================
# 4. WHAT-IF SANDBOX
# =====================================================================
def reclassify_with_overrides(detector, original_row: pd.Series, overrides: dict) -> dict:
    """Re-predict a flow after applying user-specified feature overrides."""
    df = pd.DataFrame([original_row.to_dict()])
    for col, val in overrides.items():
        if col in df.columns:
            df[col] = val
    try:
        preds = detector.predict_dataframe(df)
        if not preds:
            return {"class": "?", "confidence": 0.0, "severity": "?", "n_advisories": 0}
        p = preds[0]
        return {
            "class": p.predicted_class,
            "confidence": float(p.confidence),
            "severity": p.advisories[0].severity if p.advisories else "Informational",
            "n_advisories": len(p.advisories),
        }
    except Exception as e:
        return {"class": "ERROR", "confidence": 0.0, "severity": str(e)[:60], "n_advisories": 0}


# =====================================================================
# 5. KNOWLEDGE BASE EDITOR  (session-scoped user rules)
# =====================================================================
def evaluate_user_rule_on_df(rule: dict, df: pd.DataFrame) -> dict:
    """Evaluate an editor-defined rule on a DataFrame, return match stats."""
    feat, op, val = rule["feature"], rule["op"], float(rule["value"])
    if feat not in df.columns:
        return {"matches": 0, "total": len(df), "rate": 0.0, "error": f"Feature '{feat}' not in dataset"}
    series = pd.to_numeric(df[feat], errors="coerce")
    if op == ">":   m = series > val
    elif op == ">=": m = series >= val
    elif op == "<":  m = series < val
    elif op == "<=": m = series <= val
    elif op == "==": m = series == val
    elif op == "!=": m = series != val
    else: return {"matches": 0, "total": len(df), "rate": 0.0, "error": f"Unknown op: {op}"}
    matched = int(m.fillna(False).sum())
    total = len(df)
    return {"matches": matched, "total": total, "rate": matched / total if total else 0.0}


# =====================================================================
# 6. ML vs RULES vs HYBRID COMPARISON
# =====================================================================
def comparison_metrics(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-mode metrics treating ML-only as a ground-truth proxy.

    In a true evaluation we would use a held-out labelled set; here we
    illustrate the structural disagreement between approaches.
    """
    if comp_df is None or comp_df.empty:
        return pd.DataFrame()

    truth = comp_df["ml_class"]
    out = []
    for col, name in [("ml_class", "ML-only"), ("rule_class", "Rules-only"), ("hybrid_class", "Hybrid")]:
        pred = comp_df[col]
        agree = int((pred == truth).sum())
        attack_truth = (truth != "BENIGN")
        attack_pred = (pred != "BENIGN")
        tp = int((attack_pred & attack_truth).sum())
        fp = int((attack_pred & ~attack_truth).sum())
        fn = int((~attack_pred & attack_truth).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out.append({
            "mode": name,
            "agree_pct": round(agree / len(comp_df) * 100, 1),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "false_positives": fp,
            "false_negatives": fn,
        })
    return pd.DataFrame(out)


# =====================================================================
# 7. ADVERSARIAL ROBUSTNESS TEST
# =====================================================================
def adversarial_noise_test(detector, df_raw: pd.DataFrame, noise_levels=(0, 5, 10, 20, 30, 50)) -> pd.DataFrame:
    """Inject Gaussian noise (sigma-scaled) into numeric features and measure detection drop."""
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    try:
        baseline = detector.predict_dataframe(df_raw)
        baseline_cls = [p.predicted_class for p in baseline]
    except Exception as e:
        return pd.DataFrame([{"noise_pct": 0, "agreement_pct": 0.0, "attack_detection_pct": 0.0, "error": str(e)}])

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    rng = np.random.default_rng(42)
    rows = []
    for noise_pct in noise_levels:
        if noise_pct == 0:
            attack_det = sum(1 for c in baseline_cls if c != "BENIGN") / max(1, len(baseline_cls)) * 100
            rows.append({"noise_pct": 0, "agreement_pct": 100.0, "attack_detection_pct": round(attack_det, 1)})
            continue
        df_noisy = df_raw.copy()
        for c in numeric_cols:
            try:
                col = df_noisy[c].astype(float)
                sigma = float(col.std()) if col.std() > 0 else 1.0
                df_noisy[c] = col + rng.normal(0, sigma * noise_pct / 100, size=len(col))
            except Exception:
                pass
        try:
            np_preds = detector.predict_dataframe(df_noisy)
            np_cls = [p.predicted_class for p in np_preds]
            agree = sum(1 for a, b in zip(baseline_cls, np_cls) if a == b)
            attack_det = sum(1 for c in np_cls if c != "BENIGN")
            rows.append({
                "noise_pct": int(noise_pct),
                "agreement_pct": round(agree / len(baseline_cls) * 100, 1),
                "attack_detection_pct": round(attack_det / len(np_cls) * 100, 1),
            })
        except Exception:
            rows.append({"noise_pct": int(noise_pct), "agreement_pct": 0.0, "attack_detection_pct": 0.0})
    return pd.DataFrame(rows)


# =====================================================================
# 8. FORENSIC INCIDENT REPORT  (DOCX + Markdown fallback)
# =====================================================================
def build_forensic_report_docx(enriched_df: pd.DataFrame, summary: dict, audit_actions: Optional[list] = None) -> Optional[bytes]:
    """Generate a professional DOCX incident report.  Returns None if python-docx missing."""
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        return None

    doc = Document()
    title = doc.add_heading("AIES NIDS  -  Forensic Incident Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph()
    p.add_run(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n").italic = True
    p.add_run(f"Total flows analysed: {summary.get('total_flows', 0):,}\n")
    p.add_run(f"Attack ratio: {summary.get('attack_ratio', 0):.1%}\n")
    p.add_run(f"Maximum severity: {summary.get('max_severity', '-')}\n")
    p.add_run(f"Rules fired: {summary.get('rules_fired', 0):,}")

    doc.add_heading("1. Executive Summary", level=1)
    if enriched_df is not None and not enriched_df.empty:
        non_benign = enriched_df[enriched_df["predicted_class"] != "BENIGN"]
        critical = enriched_df[enriched_df.get("severity", pd.Series([""] * len(enriched_df))) == "Critical"]
        cls_counts = enriched_df["predicted_class"].value_counts()
        top_class = cls_counts.index[0] if len(cls_counts) > 0 else "BENIGN"
        doc.add_paragraph(
            f"This incident analysis covers {len(enriched_df):,} network flows captured during "
            f"the most recent monitoring window. Of these, {len(non_benign):,} flows "
            f"({len(non_benign) / max(1, len(enriched_df)):.1%}) were classified as malicious. "
            f"{len(critical):,} flow(s) were rated at Critical severity. "
            f"The dominant traffic class observed was {top_class} ({cls_counts.iloc[0]:,} flows)."
        )

    doc.add_heading("2. Threat Class Distribution", level=1)
    if enriched_df is not None and not enriched_df.empty:
        cls_counts = enriched_df["predicted_class"].value_counts()
        tbl = doc.add_table(rows=1, cols=3)
        tbl.style = "Light Grid Accent 1"
        h = tbl.rows[0].cells
        h[0].text = "Attack Class"; h[1].text = "Count"; h[2].text = "Percentage"
        for cls, n in cls_counts.items():
            r = tbl.add_row().cells
            r[0].text = str(cls); r[1].text = f"{n:,}"; r[2].text = f"{n / len(enriched_df):.1%}"

    doc.add_heading("3. MITRE ATT&CK Techniques Identified", level=1)
    if enriched_df is not None and "mitre_id" in enriched_df.columns:
        mc = enriched_df["mitre_id"].value_counts().head(10)
        tbl = doc.add_table(rows=1, cols=3)
        tbl.style = "Light Grid Accent 1"
        h = tbl.rows[0].cells
        h[0].text = "Technique ID"; h[1].text = "Tactic"; h[2].text = "Occurrences"
        for tid, n in mc.items():
            tac = "-"
            if "tactic" in enriched_df.columns:
                sub = enriched_df.loc[enriched_df["mitre_id"] == tid, "tactic"]
                if not sub.empty:
                    tac = str(sub.iloc[0])
            r = tbl.add_row().cells
            r[0].text = str(tid); r[1].text = tac; r[2].text = f"{n:,}"

    doc.add_heading("4. Geographic Threat Origins (Synthetic Attribution)", level=1)
    if enriched_df is not None and "country" in enriched_df.columns:
        co = enriched_df["country"].value_counts().head(10)
        tbl = doc.add_table(rows=1, cols=2)
        tbl.style = "Light Grid Accent 1"
        tbl.rows[0].cells[0].text = "Country"
        tbl.rows[0].cells[1].text = "Flow Count"
        for country, n in co.items():
            r = tbl.add_row().cells
            r[0].text = str(country); r[1].text = f"{n:,}"
    p = doc.add_paragraph()
    p.add_run("Note: ").bold = True
    p.add_run("Geographic attribution is synthesised using IP-reputation distribution from public threat-intelligence reports. Production deployments would integrate MaxMind GeoIP2 or similar.")

    doc.add_heading("5. Recommended Actions", level=1)
    recs = []
    if enriched_df is not None and not enriched_df.empty:
        seen = set(enriched_df["predicted_class"].unique())
        if "DDoS" in seen:        recs.append("Activate DDoS mitigation: rate-limiting, anycast scrubbing, upstream blackholing.")
        if "PortScan" in seen:    recs.append("Block source IPs at perimeter; enable port-knock on critical hosts.")
        if "BruteForce" in seen:  recs.append("Enforce account-lockout policy; mandate MFA on exposed services.")
        if "WebAttack" in seen:   recs.append("Review WAF rules; patch web application; sanitise inputs against XSS/SQLi.")
        if "Botnet" in seen:      recs.append("Quarantine infected hosts; rotate credentials; deep-scan endpoints with EDR.")
        if "Infiltration" in seen:recs.append("Initiate incident response playbook; preserve memory + disk for forensic analysis.")
        if "DoS" in seen:         recs.append("Tune connection-rate limits at load balancer; monitor for amplification vectors.")
    if not recs:
        recs.append("No critical attack class detected during this window. Continue routine monitoring.")
    for r in recs:
        doc.add_paragraph(r, style="List Bullet")

    if audit_actions:
        doc.add_heading("6. Operator Audit Trail", level=1)
        tbl = doc.add_table(rows=1, cols=3)
        tbl.style = "Light Grid Accent 1"
        h = tbl.rows[0].cells
        h[0].text = "Timestamp"; h[1].text = "Action"; h[2].text = "Subject"
        for a in audit_actions[:25]:
            r = tbl.add_row().cells
            r[0].text = str(a.get("ts_utc", "-"))[:19]
            r[1].text = str(a.get("action", "-"))
            r[2].text = str(a.get("subject", "-"))

    doc.add_paragraph()
    pf = doc.add_paragraph()
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.add_run("AIES NIDS  -  CT-361 AIES CCP  -  NED University BCIT").italic = True

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_forensic_report_markdown(enriched_df: pd.DataFrame, summary: dict, audit_actions: Optional[list] = None) -> str:
    """Plain Markdown fallback if python-docx is unavailable."""
    out = ["# AIES NIDS - Forensic Incident Report", ""]
    out.append(f"**Report generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    out.append(f"**Total flows:** {summary.get('total_flows', 0):,}  ")
    out.append(f"**Attack ratio:** {summary.get('attack_ratio', 0):.1%}  ")
    out.append(f"**Max severity:** {summary.get('max_severity', '-')}  ")
    out.append("")
    out.append("## Class Distribution")
    if enriched_df is not None and not enriched_df.empty:
        for cls, n in enriched_df["predicted_class"].value_counts().items():
            out.append(f"- **{cls}**: {n:,} ({n / len(enriched_df):.1%})")
    return "\n".join(out)
