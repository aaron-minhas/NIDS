"""
AIES NIDS  -  Streamlit UI rendering for the AIES Theory tab.

Provides render_theory_tab() which displays 8 interactive demos of core
AI Expert Systems concepts. Each demo includes:
  - A "What is this?" expander (so the operator can explain it to a teacher)
  - An interactive widget panel
  - A live result display

All demos are session-scoped and read from the shared analysis output.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.aies_inference import (
    run_traced_inference, explain_why, render_why_tree_markdown,
    cf_combine_sequence, cf_strength_label, severity_to_cf,
    reclassify_with_overrides, evaluate_user_rule_on_df,
    comparison_metrics, adversarial_noise_test,
    build_forensic_report_docx, build_forensic_report_markdown,
)


def _feature_dict_from_row(row: pd.Series, exclude=("flow_idx", "predicted_class", "confidence",
        "severity", "advisory_title", "mitre_id", "tactic", "kill_chain_stage",
        "country", "iso", "lat", "lon", "asn", "dest_port", "service")) -> dict:
    """Pull only numeric/feature columns out of an enriched row."""
    out = {}
    for c in row.index:
        if c in exclude:
            continue
        v = row[c]
        if pd.isna(v):
            continue
        try:
            out[c] = float(v)
        except Exception:
            pass
    if "dest_port" in row.index and pd.notna(row["dest_port"]):
        try:
            out["Destination Port"] = float(row["dest_port"])
        except Exception:
            pass
    return out


def render_theory_tab(detector, audit, enriched, summary, plotly_clean, kpi, section):
    section("AI Expert Systems  ·  live concept demonstrations")
    st.caption("Interactive demos of the inference techniques used inside the NIDS engine. "
               "Pick one, expand the explainer, then run it on real flow data.")

    demo = st.selectbox("Select demonstration", [
        "1. Forward Chaining  (Match-Resolve-Act trace)",
        "2. Backward Chaining  (Why?  proof tree)",
        "3. Certainty Factors  (MYCIN-style combination)",
        "4. What-If Counterfactual Sandbox",
        "5. Knowledge Base Editor  (live editable rules)",
        "6. ML  vs  Rules  vs  Hybrid  Comparison",
        "7. Adversarial Robustness Test",
        "8. Forensic Incident Report  (auto-generated DOCX)",
    ])
    st.markdown("---")

    if demo.startswith("1."):
        _render_fc(enriched, kpi)
    elif demo.startswith("2."):
        _render_bc(enriched)
    elif demo.startswith("3."):
        _render_cf(enriched, kpi)
    elif demo.startswith("4."):
        _render_whatif(detector, enriched, kpi)
    elif demo.startswith("5."):
        _render_kbe(enriched)
    elif demo.startswith("6."):
        _render_compare(enriched, plotly_clean)
    elif demo.startswith("7."):
        _render_adv(detector, plotly_clean)
    elif demo.startswith("8."):
        _render_report(audit, enriched, summary, kpi)


# ================================================================
# 1. FORWARD CHAINING
# ================================================================
def _render_fc(enriched, kpi):
    st.markdown("### 🔬  Forward Chaining Trace")
    with st.expander("📚  What is forward chaining?  (read this to your teacher)"):
        st.markdown("""
**Forward chaining** is **data-driven reasoning**. Start with the FACTS
(this flow's measured features and ML prediction), match them against
each rule's CONDITION, and FIRE every rule whose condition is satisfied.
The cycle repeats until no more rules fire (called **quiescence**).

The cycle:  **MATCH → RESOLVE → ACT**

This is the **opposite of backward chaining** (goal-driven, see Demo 2).

**Reference:** Russell & Norvig, *Artificial Intelligence: A Modern Approach*, Chapter 9.
        """)

    if enriched is None or len(enriched) == 0:
        st.info("Run an analysis first  (Run Analysis tab)  to enable this demo.")
        return

    nb = enriched[enriched["predicted_class"] != "BENIGN"]
    pool = nb.head(50) if len(nb) > 0 else enriched.head(50)
    flow_idx = st.selectbox(
        "Pick a flow to trace:",
        options=pool["flow_idx"].tolist(),
        format_func=lambda i: f"Flow #{int(i)}  -  {pool.loc[pool['flow_idx']==i,'predicted_class'].iloc[0]}  (conf={pool.loc[pool['flow_idx']==i,'confidence'].iloc[0]:.3f})",
        key="fc_flow",
    )
    row = enriched[enriched["flow_idx"] == flow_idx].iloc[0]
    feats = _feature_dict_from_row(row)
    trace = run_traced_inference(feats, str(row["predicted_class"]), float(row["confidence"]))

    st.markdown("**Working memory  (initial state):**")
    wm_rows = [
        {"Fact": "predicted_class", "Value": str(row["predicted_class"]), "Source": "ML output"},
        {"Fact": "confidence", "Value": f"{float(row['confidence']):.4f}", "Source": "ML output"},
        {"Fact": "severity", "Value": str(row.get("severity", "-")), "Source": "Rule output"},
        {"Fact": "mitre_id", "Value": str(row.get("mitre_id", "-")), "Source": "Mapping"},
        {"Fact": "n_features_loaded", "Value": str(len(feats)), "Source": "Working memory"},
    ]
    st.dataframe(pd.DataFrame(wm_rows), hide_index=True, width="stretch")

    st.markdown("**Inference trace  (step-by-step):**")
    rows = []
    for i, t in enumerate(trace):
        rows.append({
            "#": i,
            "Cycle": t.cycle,
            "Action": t.action,
            "Rule": t.rule_id or "-",
            "Matched?": ("✓" if t.matched else "✗") if t.action == "MATCH" else "-",
            "Severity": t.severity or "-",
            "Note": t.note,
        })
    trace_df = pd.DataFrame(rows)

    def _row_style(s):
        if s["Action"] == "FIRE":
            return ["background-color: rgba(204,120,92,0.15); font-weight: 600"] * len(s)
        if s["Action"] == "HALT":
            return ["background-color: rgba(91,124,153,0.15)"] * len(s)
        if s["Action"] == "MATCH" and s["Matched?"] == "✓":
            return ["background-color: rgba(212,160,76,0.10)"] * len(s)
        return [""] * len(s)

    st.dataframe(trace_df.style.apply(_row_style, axis=1), hide_index=True, width="stretch")

    n_match = sum(1 for t in trace if t.action == "MATCH" and t.matched)
    n_fire = sum(1 for t in trace if t.action == "FIRE")
    cs = st.columns(3)
    with cs[0]: kpi("Rules tested", str(len([t for t in trace if t.action == "MATCH"])))
    with cs[1]: kpi("Conditions matched", str(n_match), value_class="medium" if n_match else "")
    with cs[2]: kpi("Rules fired", str(n_fire), value_class="coral" if n_fire else "")
    st.success(f"Trace complete -- {n_match} match(es), {n_fire} rule(s) fired, halted at quiescence.")


# ================================================================
# 2. BACKWARD CHAINING
# ================================================================
def _render_bc(enriched):
    st.markdown("### 🔍  Backward Chaining  -  the “Why?” engine")
    with st.expander("📚  What is backward chaining?"):
        st.markdown("""
**Backward chaining** is **goal-driven reasoning**. Start with a GOAL
(e.g. "*why was this flow classified as DDoS?*") and work BACKWARDS to
find the chain of rules and observations that produced the goal.

This is the **opposite direction** of forward chaining. Used by MYCIN,
Prolog, and modern Explainable AI (XAI) systems for "right-to-explanation"
compliance under GDPR Article 22.

**Reference:** Buchanan & Shortliffe, *Rule-Based Expert Systems* (1984).
        """)

    if enriched is None or len(enriched) == 0:
        st.info("Run an analysis first.")
        return

    nb = enriched[enriched["predicted_class"] != "BENIGN"]
    pool = nb.head(50) if len(nb) > 0 else enriched.head(50)
    fidx = st.selectbox("Flow to explain:", options=pool["flow_idx"].tolist(),
                        format_func=lambda i: f"Flow #{int(i)}  -  {pool.loc[pool['flow_idx']==i,'predicted_class'].iloc[0]}",
                        key="bc_flow")
    row = enriched[enriched["flow_idx"] == fidx].iloc[0]
    feats = _feature_dict_from_row(row)

    trace = run_traced_inference(feats, str(row["predicted_class"]), float(row["confidence"]))
    fired = [{"id": t.rule_id, "title": t.title, "description": t.rule_desc}
             for t in trace if t.action == "FIRE"]

    tree = explain_why(str(row["predicted_class"]), float(row["confidence"]), fired)

    st.markdown("**Proof tree  (read top-to-bottom):**")
    st.markdown(render_why_tree_markdown(tree), unsafe_allow_html=True)

    st.caption("Each '└' bullet is a sub-goal that supports its parent. The leaves are direct observations from the ML classifier or rule conditions.")


# ================================================================
# 3. CERTAINTY FACTOR
# ================================================================
def _render_cf(enriched, kpi):
    st.markdown("### 📊  Certainty Factor combination  -  MYCIN style")
    with st.expander("📚  What is a Certainty Factor (CF)?"):
        st.markdown("""
A **Certainty Factor (CF)** is a number between **−1 and +1** that
represents how strongly evidence supports a conclusion.

| CF value | Meaning                |
|---------:|------------------------|
| +1.0     | Complete certainty FOR  |
|  0.0     | No evidence either way  |
| −1.0     | Complete certainty AGAINST |

When **multiple rules** support the same conclusion, their CFs combine
using the **MYCIN formula**:

```
Both positive:  CF = CF1 + CF2 × (1 − CF1)
Both negative:  CF = CF1 + CF2 × (1 + CF1)
Mixed signs:    CF = (CF1 + CF2) / (1 − min(|CF1|, |CF2|))
```

**Why MYCIN?**  MYCIN (1976, Stanford) was the first major medical
expert system. It pioneered CF reasoning before Bayesian networks
became practical. Modern AIES textbooks teach it as the canonical
example of uncertain reasoning under incomplete evidence.

**Reference:** Shortliffe, *Computer-Based Medical Consultations:  MYCIN* (1976).
        """)

    if enriched is None or len(enriched) == 0:
        st.info("Run an analysis first.")
        return

    nb = enriched[enriched["predicted_class"] != "BENIGN"]
    pool = nb.head(50) if len(nb) > 0 else enriched.head(50)
    fidx = st.selectbox("Flow:", options=pool["flow_idx"].tolist(),
                        format_func=lambda i: f"Flow #{int(i)}  -  {pool.loc[pool['flow_idx']==i,'predicted_class'].iloc[0]}",
                        key="cf_flow")
    row = enriched[enriched["flow_idx"] == fidx].iloc[0]
    feats = _feature_dict_from_row(row)
    trace = run_traced_inference(feats, str(row["predicted_class"]), float(row["confidence"]))
    fired = [t for t in trace if t.action == "FIRE"]

    if not fired:
        st.warning("No rules fired for this flow -- nothing to combine. Try a non-BENIGN flow.")
        return

    ml_cf = 2 * float(row["confidence"]) - 1   # [0, 1] -> [-1, +1]
    cf_inputs = [ml_cf] + [severity_to_cf(t.severity) for t in fired]

    st.markdown("**Evidence sources  &  individual CFs:**")
    inputs_rows = [{"Source": "ML classifier", "CF": f"{ml_cf:+.3f}",
                    "Note": f"confidence={row['confidence']:.3f}  →  CF = 2c − 1"}]
    for t in fired:
        inputs_rows.append({
            "Source": t.rule_id, "CF": f"{severity_to_cf(t.severity):+.3f}",
            "Note": f"{t.severity} severity  →  {t.title}"
        })
    st.dataframe(pd.DataFrame(inputs_rows), hide_index=True, width="stretch")

    steps = cf_combine_sequence(cf_inputs)

    st.markdown("**Step-by-step CF combination  (left-to-right):**")
    st.dataframe(pd.DataFrame([
        {"Step": s["step"], "Formula": s["formula"], "Running CF": f"{s['cf_running']:+.3f}"}
        for s in steps
    ]), hide_index=True, width="stretch")

    final = float(steps[-1]["cf_running"])
    cs = st.columns(3)
    with cs[0]:
        kpi("Final CF", f"{final:+.3f}", value_class="coral" if final >= 0.5 else "low")
    with cs[1]:
        kpi("Strength", cf_strength_label(final))
    with cs[2]:
        kpi("Conclusion", str(row["predicted_class"]),
            value_class="critical" if str(row["predicted_class"]) != "BENIGN" else "")


# ================================================================
# 4. WHAT-IF
# ================================================================
def _render_whatif(detector, enriched, kpi):
    st.markdown("### 🎚  What-If Counterfactual Sandbox")
    with st.expander("📚  What is a counterfactual?"):
        st.markdown("""
**Counterfactual reasoning** asks: *"What if X had been different?"*
This is a core technique in **Explainable AI (XAI)**.

In this demo you adjust a flow's key features with sliders and the
classifier re-decides in real time. This reveals the model's
**decision boundaries**. Practical use: incident responders can see
which features push a flow into a malicious classification, and by
how much.
        """)

    if enriched is None or len(enriched) == 0 or detector is None:
        st.info("Run an analysis first.")
        return

    pool = enriched.head(50)
    fidx = st.selectbox("Base flow:", options=pool["flow_idx"].tolist(),
                        format_func=lambda i: f"Flow #{int(i)}  -  {pool.loc[pool['flow_idx']==i,'predicted_class'].iloc[0]}",
                        key="wi_flow")
    row = enriched[enriched["flow_idx"] == fidx].iloc[0]

    candidates = [
        ("Flow Packets/s", 0.0, 200000.0, 1000.0),
        ("Total Length of Fwd Packets", 0.0, 100000.0, 500.0),
        ("SYN Flag Count", 0.0, 200.0, 1.0),
        ("Fwd Packets/s", 0.0, 100000.0, 500.0),
        ("Flow Duration", 0.0, 120000000.0, 100000.0),
    ]

    overrides = {}
    cols = st.columns(2)
    n = 0
    for feat, lo, hi, step in candidates:
        if feat in row.index and pd.notna(row[feat]):
            try:
                cur = float(row[feat])
            except Exception:
                continue
            with cols[n % 2]:
                top = max(float(hi), cur * 2.0)
                v = st.slider(feat, min_value=float(lo), max_value=float(top),
                              value=float(cur), step=float(step), key=f"wi_{feat}")
                overrides[feat] = v
            n += 1

    if not overrides:
        st.warning("No sliderable features found in this flow.")
        return

    if st.button("🔁  Re-classify with overrides", type="primary", key="wi_run"):
        result = reclassify_with_overrides(detector, row, overrides)
        cs = st.columns(3)
        with cs[0]:
            kpi("Original prediction", str(row["predicted_class"]),
                f"conf={float(row['confidence']):.3f}")
        with cs[1]:
            arrow = "→" if result["class"] != str(row["predicted_class"]) else "="
            kpi(f"Modified  {arrow}", str(result["class"]),
                f"conf={result['confidence']:.3f}",
                value_class="critical" if result["class"] != "BENIGN" else "")
        with cs[2]:
            changed = result["class"] != str(row["predicted_class"])
            kpi("Decision changed?", "YES" if changed else "NO",
                value_class="coral" if changed else "")
        if changed:
            st.success(f"Classification flipped:  {row['predicted_class']}  →  {result['class']}.  "
                       "The slider you moved crossed a decision boundary.")
        else:
            st.info("Classification unchanged -- modifications stayed inside the same decision region.")


# ================================================================
# 5. KNOWLEDGE BASE EDITOR
# ================================================================
def _render_kbe(enriched):
    st.markdown("### ✏️  Knowledge Base Editor  -  add live rules")
    with st.expander("📚  What is a Knowledge Base (KB)?"):
        st.markdown("""
A **Knowledge Base (KB)** is the explicit, editable store of expert
domain knowledge in a rule-based system. Unlike ML models (opaque,
require retraining), the KB is **human-readable, transparent, and
modifiable at runtime**.

This editor lets you add new IF-THEN rules in real time. Each rule
is immediately evaluated against the current dataset, showing how
many flows it would match.

**Why this matters:** *Knowledge engineering* is the human-in-the-loop
component of AIES. Your domain expert can encode a new threat
signature in seconds -- ML would take days of retraining.
        """)

    if "user_rules" not in st.session_state:
        st.session_state["user_rules"] = []

    st.markdown("#### Add a new rule")
    with st.form("kb_form", clear_on_submit=True):
        cs = st.columns([3, 1, 2, 3, 2])
        if enriched is not None and len(enriched) > 0:
            numeric_features = [c for c in enriched.columns
                                if pd.api.types.is_numeric_dtype(enriched[c])
                                and c not in ("flow_idx", "lat", "lon")]
            with cs[0]:
                feat = st.selectbox("Feature", numeric_features, key="kb_feat")
        else:
            with cs[0]:
                feat = st.text_input("Feature", value="Flow Packets/s")
        with cs[1]:
            op = st.selectbox("Op", [">", ">=", "<", "<=", "==", "!="], key="kb_op")
        with cs[2]:
            val = st.number_input("Value", value=100000.0, step=1000.0, key="kb_val")
        with cs[3]:
            conclusion = st.text_input("Conclusion", value="High flow rate -- likely DDoS", key="kb_conc")
        with cs[4]:
            sev = st.selectbox("Severity", ["Critical", "High", "Medium", "Low", "Informational"], key="kb_sev")
        submit = st.form_submit_button("➕  Add rule", type="primary")
        if submit:
            rule = {
                "id": f"R-USR-{len(st.session_state['user_rules']) + 1:03d}",
                "feature": feat, "op": op, "value": float(val),
                "conclusion": conclusion, "severity": sev,
                "added": datetime.now().isoformat(timespec="seconds"),
            }
            st.session_state["user_rules"].append(rule)
            st.success(f"Added {rule['id']}")

    rules = st.session_state.get("user_rules", [])
    if rules:
        st.markdown("#### Active user-defined rules")
        for r in rules:
            with st.container(border=True):
                cs = st.columns([1, 5, 1, 2])
                with cs[0]: st.markdown(f"**{r['id']}**")
                with cs[1]: st.markdown(f"`IF {r['feature']} {r['op']} {r['value']:g}  THEN  {r['conclusion']}`")
                with cs[2]: st.markdown(f"<span class='sev-pill sev-{r['severity']}'>{r['severity']}</span>", unsafe_allow_html=True)
                with cs[3]:
                    if enriched is not None and len(enriched) > 0:
                        stats = evaluate_user_rule_on_df(r, enriched)
                        if "error" in stats:
                            st.caption(stats["error"])
                        else:
                            st.markdown(f"**{stats['matches']:,}** match  ({stats['rate']:.1%})")
                    else:
                        st.caption("No data loaded")

        if st.button("🗑  Clear all user rules", key="kb_clear"):
            st.session_state["user_rules"] = []
            st.rerun()
    else:
        st.caption("No user rules yet. Add one using the form above.")


# ================================================================
# 6. ML vs RULES vs HYBRID
# ================================================================
def _render_compare(enriched, plotly_clean):
    st.markdown("### ⚖️  ML  vs  Rules  vs  Hybrid  Comparison")
    with st.expander("📚  Why hybrid?"):
        st.markdown("""
This demo runs the **same dataset** through three classification
strategies and compares them.

| Mode | What it uses |
|---|---|
| **ML-only** | Pure RandomForest probabilities |
| **Rules-only** | Hand-coded thresholds  (Flow Packets/s > 100k → DDoS, etc.) |
| **Hybrid** | RF picks the class; expert rules add severity & explanation |

The hybrid approach is the **central thesis of AIES** -- symbolic
reasoning gives statistical ML interpretability, auditability, and
the ability to encode domain expertise that the training data
lacked.
        """)

    if enriched is None or len(enriched) == 0:
        st.info("Run an analysis first.")
        return

    rows = []
    for _, row in enriched.iterrows():
        ml_class = str(row["predicted_class"])
        try:
            flow_pkts = float(row.get("Flow Packets/s", 0) or 0)
        except Exception:
            flow_pkts = 0.0
        try:
            syn = float(row.get("SYN Flag Count", 0) or 0)
        except Exception:
            syn = 0.0
        if flow_pkts > 100000:
            rule_class = "DDoS"
        elif syn > 100:
            rule_class = "PortScan"
        elif flow_pkts > 10000:
            rule_class = "DoS"
        else:
            rule_class = "BENIGN"
        rows.append({"ml_class": ml_class, "rule_class": rule_class, "hybrid_class": ml_class})
    comp = pd.DataFrame(rows)

    metrics = comparison_metrics(comp)
    st.markdown("**Comparison metrics**  (using ML-only as ground-truth proxy):")
    st.dataframe(metrics, hide_index=True, width="stretch")

    fig = go.Figure()
    for col, color in [("precision", "#5B7C99"), ("recall", "#D4A04C"), ("f1", "#CC785C")]:
        fig.add_trace(go.Bar(x=metrics["mode"], y=metrics[col], name=col.title(),
                              marker_color=color, text=metrics[col].apply(lambda v: f"{v:.2f}"),
                              textposition="outside"))
    fig.update_layout(**plotly_clean(height=340, barmode="group",
                                       margin=dict(t=30, l=10, r=10, b=10),
                                       yaxis=dict(range=[0, 1.05])))
    st.plotly_chart(fig, width="stretch")

    disagreement = int((comp["ml_class"] != comp["rule_class"]).sum())
    st.info(f"**ML and Rules-only disagree on {disagreement:,} of {len(comp):,} flows  "
            f"({disagreement / max(1, len(comp)):.1%}).**  "
            "The hybrid approach uses ML's class to resolve these disagreements while "
            "still using rule-derived severity for context.")


# ================================================================
# 7. ADVERSARIAL
# ================================================================
def _render_adv(detector, plotly_clean):
    st.markdown("### 🛡  Adversarial Robustness Test")
    with st.expander("📚  What is adversarial robustness?"):
        st.markdown("""
**Adversarial robustness** measures how well an AI system holds up
when its input is **perturbed**. Real attackers add small noise
("adversarial examples") to evade detection.

This test injects **Gaussian noise**, scaled to each feature's
standard deviation, then re-classifies. Result: a **degradation
curve** -- accuracy drops as noise rises. A robust model has a
shallow drop.

**Why this matters:** AIES is a *cybersecurity* application. Models
that fail under adversarial pressure cannot be trusted in
production.
        """)

    df_raw = st.session_state.get("last_df_raw")
    if df_raw is None or len(df_raw) == 0 or detector is None:
        st.info("Run an analysis first  (we need the original raw flows for noise injection).")
        return

    noise_levels = (0, 5, 10, 20, 30, 50)
    if st.button("Run adversarial test  (~10 seconds)", type="primary", key="adv_run"):
        sample = df_raw.head(300)
        with st.spinner(f"Injecting noise at 6 levels and re-classifying {len(sample)} flows..."):
            res = adversarial_noise_test(detector, sample, noise_levels)

        st.markdown("**Degradation curve:**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["noise_pct"], y=res["agreement_pct"],
                                   mode="lines+markers", name="Agreement with baseline",
                                   line=dict(color="#CC785C", width=3),
                                   marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=res["noise_pct"], y=res["attack_detection_pct"],
                                   mode="lines+markers", name="Attack detection rate",
                                   line=dict(color="#5B7C99", width=2, dash="dash"),
                                   marker=dict(size=8)))
        fig.update_layout(**plotly_clean(height=360,
                                           xaxis_title="Noise level  (% of feature stdev)",
                                           yaxis_title="Rate  (%)",
                                           margin=dict(t=30, l=10, r=10, b=10),
                                           yaxis=dict(range=[0, 105])))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(res, hide_index=True, width="stretch")

        if 30 in list(res["noise_pct"]):
            drop_at_30 = 100.0 - float(res.loc[res["noise_pct"] == 30, "agreement_pct"].iloc[0])
            st.info(f"At 30% noise the classifier disagrees with itself on **{drop_at_30:.1f}%** of flows. "
                    "Lower drop = more robust model.")


# ================================================================
# 8. FORENSIC REPORT
# ================================================================
def _render_report(audit, enriched, summary, kpi):
    st.markdown("### 📄  Forensic Incident Report  -  auto-generated DOCX")
    with st.expander("📚  Why an auto-generated report?"):
        st.markdown("""
Real SOC analysts spend hours writing incident reports. This module
auto-generates a **professional, structured DOCX** from the analysis
results, with six sections:

1. Executive Summary
2. Threat Class Distribution
3. MITRE ATT&CK Techniques Identified
4. Geographic Threat Origins (synthetic attribution disclosed)
5. Recommended Actions
6. Operator Audit Trail

**Practical artifact:** this is the deliverable you give to
management or compliance auditors -- not a screenshot, an actual
Word document.
        """)

    if enriched is None or summary is None:
        st.info("Run an analysis first.")
        return

    cs = st.columns(3)
    with cs[0]: kpi("Flows in report", f"{len(enriched):,}")
    with cs[1]: kpi("Attack ratio", f"{float(summary.get('attack_ratio', 0)):.1%}")
    with cs[2]: kpi("Max severity", str(summary.get("max_severity", "-")))

    audit_recent = audit.recent(limit=20)
    if hasattr(audit_recent, "to_dict"):
        audit_list = audit_recent.to_dict("records")
    elif isinstance(audit_recent, list):
        audit_list = audit_recent
    else:
        audit_list = []

    docx_bytes = build_forensic_report_docx(enriched, summary, audit_list)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if docx_bytes is None:
        st.warning("python-docx is not installed -- falling back to Markdown.")
        md = build_forensic_report_markdown(enriched, summary, audit_list)
        st.download_button(
            "⬇  Download incident report  (Markdown)",
            data=md, file_name=f"aies_nids_incident_report_{ts}.md",
            mime="text/markdown", type="primary",
        )
        st.markdown("**Preview:**")
        st.markdown(md)
    else:
        st.success(f"✓ Report ready  ({len(docx_bytes):,} bytes, 6 sections, professional formatting)")
        st.download_button(
            "⬇  Download incident report  (DOCX)",
            data=docx_bytes,
            file_name=f"aies_nids_incident_report_{ts}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
        )
        st.caption("Open in Microsoft Word or LibreOffice to view all six sections, three tables, and recommendations.")
