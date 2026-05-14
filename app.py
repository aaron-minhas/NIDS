"""
AIES NIDS  //  v4.0 
Hybrid AI + Expert System Network Intrusion Detection — clean, honest, zero-start.

Design philosophy:
- Empty by default. Every tab starts at zero state with a clear "Run Analysis" CTA.
- Truth-in-UI. Every simulated subsystem (replay mode, synthetic geo, demo compliance)
  is labelled honestly so a viewer cannot mistake the demo for production.
- Cyber SOC theme: dark graphite surface, terminal green telemetry, cyan AI accents.
  Kali-inspired energy without noisy gimmicks or fake hacker tropes.
- Session-scoped. The threat picture reflects ONLY current session by default;
  historical alerts are accessible via explicit toggle.
"""
from __future__ import annotations
import base64
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.prediction import NIDSDetector
from src.expert_system import Fact, _build_rules
from src.preprocessing import clean_columns, normalize_label
from src.alert_log import AlertLog
from src.live_capture import scapy_live_capture_stub, replay_csv
from src.threat_intel import (
    enrich_predictions, compute_threat_score, MITRE_MAPPING,
    KILL_CHAIN_STAGES, KILL_CHAIN_MAPPING, THREAT_ORIGINS, port_to_service,
)
from src.compliance import compute_coverage, overall_posture_score, FRAMEWORKS
from src.stix_export import to_stix_bundle, to_cef, to_syslog, to_json_export
from src.operator_audit import OperatorAudit, get_default_operator
from src.aies_render import render_theory_tab
from src.case_management import build_case_board, case_summary, cases_markdown
from src.mission_control import (
    build_attack_surface,
    build_brief_markdown,
    build_campaign_timeline,
    build_command_brief,
    build_data_quality_report,
    build_remediation_plan,
    event_feed_from_enriched,
    severity_counts_from_enriched,
)
from src.model_risk import class_risk_table, model_risk_summary, review_queue, uncertainty_rows
from src.readiness_report import build_readiness_report
from src.scenario_lab import scenario_description, scenario_names, scenario_targets
from src.system_health import health_summary, project_health

PROJECT_ROOT = Path(__file__).parent
HERO_ASSET = PROJECT_ROOT / "assets" / "aies-cyber-hero.png"

# ============================================================================
# PAGE CONFIG  +  THEME
# ============================================================================
st.set_page_config(
    page_title="AIES NIDS  -  Network Intrusion Detection",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def asset_data_uri(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


HERO_BG_URI = asset_data_uri(str(HERO_ASSET))

# Legacy CSS retained only as historical reference. It is not rendered.
_LEGACY_THEME_CSS = """
<style>
/* Legacy cream theme disabled. The production cyber theme below is the single
   source of truth for typography, controls, cards, and layout.
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:        #FAF9F5;
  --surface:   #FFFFFF;
  --surface-2: #F4F1EA;
  --border:    #E5E1D8;
  --text:      #191919;
  --text-2:    #5C5C5C;
  --text-3:    #8C8C8C;
  --accent:    #CC785C;   /* primary accent */
  --accent-2:  #B85F44;
  --critical:  #B91C1C;
  --high:      #CC785C;
  --medium:    #D4A04C;
  --low:       #3D5A80;
  --info:      #7B7B7B;
  --benign:    #5B7C99;
}

html, body, [class*="st-"], [data-testid="stAppViewContainer"], .main {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stHeader"] { background: var(--bg) !important; }
[data-testid="stSidebar"] { background: var(--surface-2) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3, h4 { color: var(--text) !important; font-weight: 600 !important; letter-spacing: -0.01em; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.25rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.35rem;
  margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 7px;
  color: var(--text-2);
  font-weight: 500;
  padding: 0.55rem 1.0rem;
  font-size: 0.88rem;
  border: 1px solid transparent;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--surface-2); color: var(--text); }
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: white !important;
  font-weight: 600;
}

/* Hero */
.hero {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.2rem;
}
.hero-title { font-size: 1.6rem; font-weight: 700; color: var(--text); margin: 0; letter-spacing: -0.02em; }
.hero-sub { font-size: 0.92rem; color: var(--text-2); margin-top: 0.2rem; font-family: 'JetBrains Mono', monospace; }
.hero-ts { font-size: 0.78rem; color: var(--text-3); margin-top: 0.5rem; font-family: 'JetBrains Mono', monospace; }

/* KPI tiles */
.kpi {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.0rem 1.1rem;
  height: 100%;
}
.kpi-label {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 0.4rem;
}
.kpi-value { font-size: 1.7rem; font-weight: 700; color: var(--text); line-height: 1.1; }
.kpi-value.coral { color: var(--accent); }
.kpi-value.critical { color: var(--critical); }
.kpi-value.high { color: var(--high); }
.kpi-value.medium { color: var(--medium); }
.kpi-value.low { color: var(--low); }
.kpi-delta { font-size: 0.78rem; color: var(--text-3); margin-top: 0.3rem; }

/* Section header */
.sect-h {
  font-size: 0.76rem;
  font-weight: 700;
  color: var(--text-2);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin: 1.4rem 0 0.7rem 0;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
}

/* Empty state cards */
.empty-state {
  background: var(--surface);
  border: 1px dashed var(--border);
  border-radius: 12px;
  padding: 3rem 2rem;
  text-align: center;
  margin: 1rem 0;
}
.empty-state-icon { font-size: 2.5rem; margin-bottom: 0.6rem; }
.empty-state-title { font-size: 1.1rem; font-weight: 600; color: var(--text); margin-bottom: 0.3rem; }
.empty-state-msg { font-size: 0.92rem; color: var(--text-2); max-width: 480px; margin: 0 auto 1rem auto; line-height: 1.55; }
.empty-state-cta {
  display: inline-block;
  background: var(--accent);
  color: white;
  padding: 0.55rem 1.2rem;
  border-radius: 7px;
  font-weight: 600;
  font-size: 0.88rem;
  text-decoration: none;
}

/* Severity pills */
.sev-pill {
  display: inline-block;
  padding: 0.18rem 0.55rem;
  border-radius: 4px;
  font-size: 0.72rem;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.04em;
}
.sev-Critical { background: rgba(185,28,28,0.10); color: var(--critical); border: 1px solid rgba(185,28,28,0.25); }
.sev-High { background: rgba(204,120,92,0.12); color: var(--accent); border: 1px solid rgba(204,120,92,0.30); }
.sev-Medium { background: rgba(212,160,76,0.12); color: var(--medium); border: 1px solid rgba(212,160,76,0.30); }
.sev-Low { background: rgba(61,90,128,0.10); color: var(--low); border: 1px solid rgba(61,90,128,0.25); }
.sev-Informational { background: rgba(123,123,123,0.10); color: var(--info); border: 1px solid rgba(123,123,123,0.25); }

/* Disclosure banners */
.disclose {
  background: #FFF8E7;
  border: 1px solid #E8D8A8;
  border-left: 3px solid #D4A04C;
  border-radius: 6px;
  padding: 0.7rem 1rem;
  font-size: 0.85rem;
  color: #5C4A1F;
  margin-bottom: 1rem;
  font-family: 'Inter', sans-serif;
}
.disclose b { color: #4A3A14; }

/* Data feed rows */
.feed {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  max-height: 420px;
  overflow-y: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
}
.feed-row {
  padding: 0.5rem 0.85rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  gap: 0.75rem;
  align-items: center;
}
.feed-row:last-child { border-bottom: none; }
.feed-ts { color: var(--text-3); flex-shrink: 0; }
.feed-cls { font-weight: 600; flex-shrink: 0; }

/* Dataframes */
[data-testid="stDataFrame"] { border-radius: 8px !important; border: 1px solid var(--border) !important; }

/* Buttons */
.stButton > button {
  background: var(--accent) !important;
  color: white !important;
  border: none !important;
  border-radius: 7px !important;
  font-weight: 600 !important;
}
.stButton > button:hover { background: var(--accent-2) !important; }
.stDownloadButton > button {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 7px !important;
  font-weight: 500 !important;
}
.stDownloadButton > button:hover { background: var(--surface-2) !important; border-color: var(--accent) !important; }

/* File uploader -- fix overlapping text/icon glitch */
[data-testid='stFileUploader'] {
  background: var(--surface) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 8px !important;
  padding: 0.75rem !important;
}
[data-testid='stFileUploader'] section {
  background: transparent !important;
  border: none !important;
  padding: 0.5rem !important;
}
[data-testid='stFileUploaderDropzone'] {
  background: var(--surface-2) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 6px !important;
  padding: 1.5rem 1rem !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  gap: 0.4rem !important;
}
[data-testid='stFileUploaderDropzone'] svg {
  width: 1.2rem !important;
  height: 1.2rem !important;
  color: var(--text-3) !important;
}
/* Hide Material Icons font glyphs that fail to render (the "uploa" text bug) */
[data-testid='stFileUploaderDropzone'] [data-testid='stIconMaterial'],
[data-testid='stFileUploaderDropzone'] span[class*='material-symbols'],
[data-testid='stFileUploaderDropzone'] span[class*='MuiIcon'],
[data-testid='stFileUploader'] .material-icons,
[data-testid='stFileUploader'] [translate='no'],
[data-testid='stFileUploaderDropzone'] button > div > span:first-child:not(:has(svg)),
button[kind='secondary'] > div > span[data-testid='stIconMaterial'] {
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  font-size: 0 !important;
}
/* Streamlit toolbar / Deploy button - hide */
[data-testid='stToolbar'], [data-testid='stDeployButton'], [data-testid='stStatusWidget'],
.stDeployButton {
  display: none !important;
}
footer, .stApp > footer, [data-testid='stFooter'] { display: none !important; }

/* === STRONG: Hide ALL Material Icons leaks (expander, checkbox, tooltip, etc.) === */
[data-testid='stIconMaterial'],
span[data-testid='stIconMaterial'],
span.material-symbols-rounded,
span.material-symbols-outlined,
span.material-symbols-sharp,
span.material-icons,
span.material-icons-outlined,
[data-testid='stExpander'] [data-testid='stIconMaterial'],
[data-testid='stCheckbox'] [data-testid='stIconMaterial'],
[data-testid='stTooltipIcon'] [data-testid='stIconMaterial'],
[data-testid='stExpanderIcon'],
details > summary > div > span[translate='no']:first-child {
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  height: 0 !important;
  font-size: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}
[data-testid='stExpander'] summary { gap: 0.4rem !important; padding-left: 0.6rem !important; }
[data-testid='stExpander'] summary > div { width: 100% !important; }
[data-testid='stFileUploaderDropzone'] small {
  font-size: 0.78rem !important;
  color: var(--text-3) !important;
  font-family: 'Inter', sans-serif !important;
}
[data-testid='stFileUploaderDropzoneInstructions'] {
  display: flex !important;
  flex-direction: column !important;
  gap: 0.3rem !important;
  align-items: center !important;
}
[data-testid='stFileUploaderDropzone'] button,
[data-testid='stBaseButton-secondary'] {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  font-weight: 500 !important;
  padding: 0.4rem 1rem !important;
  font-size: 0.85rem !important;
}
[data-testid='stFileUploaderDropzone'] button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
/* Plotly chart containers -- subtle border */
[data-testid='stPlotlyChart'] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.5rem;
}
/* Radio + selectbox cleanup */
.stRadio [role='radiogroup'] label { font-weight: 500 !important; }
[data-baseweb='select'] > div { border-color: var(--border) !important; }

/* Dataframe header */
[data-testid='stDataFrame'] thead { background: var(--surface-2) !important; }
[data-testid='stDataFrame'] thead th { color: var(--text-2) !important; font-weight: 600 !important; }
*/
</style>
"""

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* Cyber SOC theme override: dark graphite + terminal green + intelligence cyan. */
:root {
  --bg: #05080A;
  --surface: #0B1117;
  --surface-2: #0F171F;
  --surface-3: #121D26;
  --border: #23313C;
  --border-2: #334A56;
  --text: #E8F1F2;
  --text-2: #A8B7BE;
  --text-3: #6F858F;
  --accent: #00E676;
  --accent-2: #00B8D4;
  --critical: #FF4D4D;
  --high: #FF7A45;
  --medium: #F6C343;
  --low: #4EA8FF;
  --info: #8BA1AA;
  --benign: #00B8D4;
  --hero-image: url("__HERO_BG__");
}

html, body, [class*="st-"], [data-testid="stAppViewContainer"], .main {
  background:
    radial-gradient(circle at 18% 10%, rgba(0, 230, 118, 0.10), transparent 28rem),
    radial-gradient(circle at 82% 6%, rgba(0, 184, 212, 0.10), transparent 24rem),
    linear-gradient(180deg, #060A0D 0%, #05080A 46%, #070B0E 100%) !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Sans', Inter, -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(0, 230, 118, 0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 184, 212, 0.035) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: linear-gradient(to bottom, rgba(0,0,0,0.65), transparent 72%);
  z-index: 0;
}

[data-testid="stHeader"] {
  background: linear-gradient(180deg, rgba(5, 8, 10, 0.96), rgba(5, 8, 10, 0.72)) !important;
}
[data-testid="stToolbar"],
[data-testid="stDeployButton"],
[data-testid="stStatusWidget"],
.stDeployButton {
  display: none !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #070C10 0%, #0B1117 100%) !important;
  border-right: 1px solid rgba(0, 230, 118, 0.22) !important;
  box-shadow: 18px 0 60px rgba(0, 0, 0, 0.36);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] small {
  color: var(--text-3) !important;
}

.block-container {
  max-width: 1420px !important;
  padding-top: 4.4rem !important;
  padding-left: 2.4rem !important;
  padding-right: 2.4rem !important;
}

h1, h2, h3, h4 {
  color: var(--text) !important;
  font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif !important;
  letter-spacing: 0 !important;
}
p, li, label, span, div {
  letter-spacing: 0 !important;
}

code, pre, .hero-ts, .kpi-delta, .feed, [data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace !important;
}

.hero {
  position: relative;
  overflow: hidden;
  min-height: 228px;
  border: 1px solid rgba(0, 230, 118, 0.30);
  border-radius: 8px;
  padding: 1.45rem 1.55rem;
  margin-bottom: 1.15rem;
  background:
    linear-gradient(90deg, rgba(5, 8, 10, 0.92) 0%, rgba(6, 12, 16, 0.82) 46%, rgba(5, 8, 10, 0.42) 100%),
    var(--hero-image),
    linear-gradient(135deg, #071017 0%, #0B1117 100%);
  background-size: cover;
  background-position: center;
  box-shadow: 0 22px 70px rgba(0, 0, 0, 0.35), inset 0 0 0 1px rgba(255,255,255,0.03);
}
.hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(transparent 0%, rgba(0, 230, 118, 0.10) 48%, transparent 50%),
    radial-gradient(circle at 72% 28%, rgba(0,184,212,0.20), transparent 20rem);
  background-size: 100% 9px, auto;
  opacity: 0.45;
  pointer-events: none;
}
.hero > * { position: relative; z-index: 1; }
.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.25rem 0.55rem;
  margin-bottom: 0.78rem;
  border: 1px solid rgba(0, 230, 118, 0.38);
  border-radius: 999px;
  background: rgba(0, 230, 118, 0.08);
  color: var(--accent);
  font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
}
.hero-title {
  max-width: 760px;
  font-size: clamp(1.85rem, 3vw, 3.05rem);
  line-height: 1;
  font-weight: 800;
  font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif !important;
  color: #F4FFFF !important;
  margin: 0;
  text-shadow: 0 0 28px rgba(0, 230, 118, 0.15);
}
.hero-title span {
  color: var(--accent);
}
.hero-sub {
  max-width: 850px;
  color: #B6CBD0 !important;
  font-family: 'IBM Plex Sans', Inter, sans-serif !important;
  font-size: 0.95rem;
  line-height: 1.55;
  margin-top: 0.75rem;
}
.hero-ts {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  color: #8FA4AD !important;
  font-size: 0.74rem;
  margin-top: 1rem;
}
.hero-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.36rem 0.58rem;
  border: 1px solid rgba(0, 184, 212, 0.28);
  border-radius: 6px;
  background: rgba(5, 12, 16, 0.72);
}
.hero-chip strong { color: var(--accent); font-weight: 700; }

.stTabs [data-baseweb="tab-list"] {
  gap: 0.35rem;
  overflow-x: auto;
  flex-wrap: nowrap;
  background: rgba(11, 17, 23, 0.84);
  border: 1px solid rgba(0, 230, 118, 0.20);
  border-radius: 8px;
  padding: 0.38rem;
  margin-bottom: 1.35rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  scrollbar-width: thin;
}
.stTabs [data-baseweb="tab"] {
  min-height: 2.2rem;
  white-space: nowrap;
  background: rgba(255, 255, 255, 0.015);
  border-radius: 6px;
  color: var(--text-2) !important;
  font-weight: 650;
  padding: 0.52rem 0.9rem;
  font-size: 0.86rem;
  border: 1px solid transparent;
}
.stTabs [data-baseweb="tab"]:hover {
  background: rgba(0, 184, 212, 0.08);
  border-color: rgba(0, 184, 212, 0.22);
  color: var(--text) !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0, 230, 118, 0.18), rgba(0, 184, 212, 0.12)) !important;
  border-color: rgba(0, 230, 118, 0.56) !important;
  color: #F3FFFA !important;
  box-shadow: 0 0 24px rgba(0, 230, 118, 0.12);
}

.sect-h {
  color: var(--accent) !important;
  font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif !important;
  font-size: 0.82rem;
  letter-spacing: 0.095em !important;
  border-bottom: 1px solid rgba(0, 230, 118, 0.22);
}

.kpi, .empty-state, [data-testid='stPlotlyChart'], [data-testid='stDataFrame'] {
  background: linear-gradient(180deg, rgba(15, 23, 31, 0.96), rgba(9, 14, 19, 0.96)) !important;
  border: 1px solid rgba(0, 230, 118, 0.16) !important;
  border-radius: 8px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.035), 0 16px 50px rgba(0,0,0,0.18);
}
.kpi-label { color: var(--text-3) !important; }
.kpi-value { color: #F1FFFF !important; }
.kpi-value.coral, .kpi-value.high { color: var(--accent) !important; }
.kpi-value.critical { color: var(--critical) !important; }
.kpi-value.medium { color: var(--medium) !important; }
.kpi-value.low { color: var(--low) !important; }
.kpi-delta { color: var(--text-3) !important; }

.empty-state {
  border-style: dashed !important;
  padding: 2.4rem 1.6rem;
}
.empty-state-title { color: var(--text) !important; }
.empty-state-msg { color: var(--text-2) !important; }
.empty-state-cta {
  background: rgba(0, 230, 118, 0.12);
  border: 1px solid rgba(0, 230, 118, 0.34);
  color: var(--accent);
}

.disclose {
  background: rgba(246, 195, 67, 0.08);
  border: 1px solid rgba(246, 195, 67, 0.26);
  border-left: 3px solid var(--medium);
  border-radius: 6px;
  color: #D9C582;
}
.disclose b { color: #FFE28A; }

.feed {
  background: rgba(5, 9, 12, 0.92);
  border: 1px solid rgba(0, 230, 118, 0.18);
  border-radius: 8px;
}
.feed-row {
  border-bottom: 1px solid rgba(35, 49, 60, 0.85);
}
.feed-row span[style*="#191919"] { color: var(--text) !important; }
.feed-row span[style*="#5C5C5C"] { color: var(--text-2) !important; }
.feed-row span[style*="#8C8C8C"] { color: var(--text-3) !important; }

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button,
[data-testid='stFileUploaderDropzone'] button,
[data-testid='stBaseButton-primary'],
[data-testid='stBaseButton-secondary'] {
  min-height: 42px !important;
  padding: 0.56rem 1.05rem !important;
  border-radius: 8px !important;
  font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif !important;
  font-size: 0.92rem !important;
  font-weight: 700 !important;
  line-height: 1.1 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  transition: transform 140ms ease, border-color 140ms ease, background 140ms ease, box-shadow 140ms ease !important;
}
.stButton > button p,
.stFormSubmitButton > button p,
.stDownloadButton > button p,
[data-testid='stFileUploaderDropzone'] button p,
[data-testid='stBaseButton-primary'] p,
[data-testid='stBaseButton-secondary'] p {
  margin: 0 !important;
  font: inherit !important;
  line-height: 1.1 !important;
  letter-spacing: 0 !important;
}
.stButton > button[kind='primary'],
[data-testid='stBaseButton-primary'] {
  background: linear-gradient(135deg, #00E676 0%, #00B8D4 100%) !important;
  color: #03110C !important;
  border: 1px solid rgba(136, 255, 199, 0.70) !important;
  box-shadow: 0 10px 28px rgba(0, 230, 118, 0.18), inset 0 1px 0 rgba(255,255,255,0.34) !important;
}
.stButton > button:not([kind='primary']),
.stFormSubmitButton > button,
.stDownloadButton > button,
[data-testid='stFileUploaderDropzone'] button,
[data-testid='stBaseButton-secondary'] {
  background: linear-gradient(180deg, rgba(22, 34, 43, 0.98), rgba(10, 17, 23, 0.98)) !important;
  color: #E8F1F2 !important;
  border: 1px solid rgba(0, 184, 212, 0.34) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 8px 22px rgba(0,0,0,0.18) !important;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover,
[data-testid='stFileUploaderDropzone'] button:hover,
[data-testid='stBaseButton-primary']:hover,
[data-testid='stBaseButton-secondary']:hover {
  transform: translateY(-1px);
  border-color: rgba(0, 230, 118, 0.72) !important;
  box-shadow: 0 12px 30px rgba(0, 184, 212, 0.18), inset 0 1px 0 rgba(255,255,255,0.18) !important;
}
.stButton > button:active,
.stFormSubmitButton > button:active,
.stDownloadButton > button:active {
  transform: translateY(0);
}
.stButton > button:disabled,
.stButton > button[disabled],
[data-testid='stBaseButton-primary']:disabled,
[data-testid='stBaseButton-secondary']:disabled {
  background: rgba(18, 29, 38, 0.72) !important;
  color: rgba(168, 183, 190, 0.52) !important;
  border-color: rgba(51, 74, 86, 0.55) !important;
  box-shadow: none !important;
  transform: none !important;
}

[data-testid='stFileUploader'] {
  background: rgba(8, 14, 19, 0.72) !important;
  border: 1px dashed rgba(0, 230, 118, 0.26) !important;
  border-radius: 8px !important;
}
[data-testid='stFileUploaderDropzone'] {
  background: linear-gradient(180deg, rgba(10, 17, 23, 0.95), rgba(6, 10, 14, 0.95)) !important;
  border: 1px dashed rgba(0, 184, 212, 0.36) !important;
  min-height: 138px !important;
  justify-content: center !important;
}
[data-testid='stFileUploaderDropzone'] * {
  background: transparent !important;
}
[data-testid='stFileUploaderDropzone'] small,
[data-testid='stFileUploaderDropzone'] span,
[data-testid='stFileUploaderDropzone'] p {
  background: transparent !important;
  color: var(--text-2) !important;
  font-family: 'IBM Plex Sans', Inter, sans-serif !important;
  font-size: 0.88rem !important;
  line-height: 1.35 !important;
}
[data-testid='stFileUploaderDropzone'] button {
  min-width: 132px !important;
  margin: 0.35rem auto 0.45rem auto !important;
  background: linear-gradient(180deg, rgba(22, 34, 43, 0.98), rgba(10, 17, 23, 0.98)) !important;
  border: 1px solid rgba(0, 184, 212, 0.38) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 8px 22px rgba(0,0,0,0.16) !important;
  color: #E8F1F2 !important;
  font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif !important;
  font-size: 0.9rem !important;
  font-weight: 700 !important;
}
[data-testid='stFileUploaderDropzone'] button * {
  color: inherit !important;
  font-family: inherit !important;
  font-size: inherit !important;
  font-weight: inherit !important;
}
[data-testid='stFileUploaderDropzone'] button:not([aria-label]) {
  display: none !important;
}
[data-testid='stFileUploaderDropzone'] input[type='file'] {
  max-width: 100% !important;
  color: var(--text-2) !important;
  font-family: 'IBM Plex Sans', Inter, sans-serif !important;
  font-size: 0.88rem !important;
}
[data-testid='stFileUploaderDropzone'] input[type='file']::file-selector-button {
  margin-right: 0.75rem !important;
  padding: 0.48rem 0.9rem !important;
  border: 1px solid rgba(0, 184, 212, 0.38) !important;
  border-radius: 7px !important;
  background: linear-gradient(180deg, rgba(22, 34, 43, 0.98), rgba(10, 17, 23, 0.98)) !important;
  color: #E8F1F2 !important;
  font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif !important;
  font-size: 0.9rem !important;
  font-weight: 700 !important;
  cursor: pointer !important;
}

[data-baseweb='select'] > div,
[data-testid='stTextInput'] input,
[data-testid='stNumberInput'] input,
[data-testid='stSlider'] {
  background-color: rgba(10, 17, 23, 0.92) !important;
  border-color: rgba(0, 184, 212, 0.24) !important;
  color: var(--text) !important;
}

[data-testid='stWidgetLabel'] p,
.stRadio > label p,
[data-testid='stCheckbox'] > label p,
[data-testid='stSlider'] label p,
[data-testid='stFileUploader'] label p {
  color: #DCE8EA !important;
  font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif !important;
  font-size: 0.92rem !important;
  font-weight: 650 !important;
  letter-spacing: 0 !important;
}

.stRadio [role='radiogroup'] {
  gap: 0.55rem;
  align-items: center;
  flex-wrap: wrap;
}
.stRadio [role='radiogroup'] label {
  min-height: 40px;
  padding: 0.52rem 0.72rem !important;
  background: rgba(13, 21, 28, 0.86);
  border: 1px solid rgba(51, 74, 86, 0.78);
  border-radius: 8px;
  color: var(--text-2) !important;
  transition: border-color 140ms ease, background 140ms ease, color 140ms ease;
}
.stRadio [role='radiogroup'] label:hover {
  border-color: rgba(0, 184, 212, 0.48);
  background: rgba(0, 184, 212, 0.08);
}
.stRadio [role='radiogroup'] label:has(input:checked) {
  border-color: rgba(0, 230, 118, 0.68);
  background: rgba(0, 230, 118, 0.10);
  color: var(--text) !important;
}
.stRadio [role='radiogroup'] label p,
[data-testid='stCheckbox'] label p {
  color: inherit !important;
  font-family: 'IBM Plex Sans', Inter, sans-serif !important;
  font-size: 0.93rem !important;
  font-weight: 600 !important;
  margin: 0 !important;
}

[data-testid='stCheckbox'] label {
  position: relative;
  min-height: 38px;
  align-items: center;
  gap: 0.5rem;
  color: var(--text-2) !important;
}
[data-testid='stCheckbox'] label:has(input:checked)::after,
[data-testid='stCheckbox'] label:has(input[aria-checked='true'])::after {
  content: "✓";
  position: absolute;
  left: 0.17rem;
  top: 50%;
  transform: translateY(-53%);
  width: 1.05rem;
  text-align: center;
  color: #03110C;
  font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif;
  font-size: 0.86rem;
  font-weight: 800;
  pointer-events: none;
}
[data-testid='stCheckbox'] label:has(input:checked)::before,
[data-testid='stCheckbox'] label:has(input[aria-checked='true'])::before {
  content: "";
  position: absolute;
  left: 0.08rem;
  top: 50%;
  transform: translateY(-50%);
  width: 1.18rem;
  height: 1.18rem;
  border-radius: 4px;
  background: linear-gradient(135deg, #00E676, #00B8D4);
  box-shadow: 0 0 14px rgba(0, 230, 118, 0.22);
  pointer-events: none;
}
[data-testid='stCheckbox'] input,
.stRadio input {
  accent-color: #00E676 !important;
}
[data-testid='stCheckbox'] div[data-baseweb='checkbox'] > div,
.stRadio div[data-baseweb='radio'] > div {
  border-color: rgba(0, 230, 118, 0.56) !important;
  background-color: rgba(0, 230, 118, 0.08) !important;
}

[data-testid='stSlider'] [role='slider'] {
  box-shadow: 0 0 0 4px rgba(0, 230, 118, 0.10) !important;
}
[data-testid='stSlider'] p,
[data-testid='stSlider'] span {
  font-family: 'IBM Plex Sans', Inter, sans-serif !important;
}

[data-baseweb='select'] > div {
  min-height: 42px !important;
  background: linear-gradient(180deg, rgba(14, 24, 32, 0.98), rgba(8, 14, 19, 0.98)) !important;
  border: 1px solid rgba(0, 184, 212, 0.34) !important;
  border-radius: 8px !important;
}
[data-baseweb='select'] span,
[data-baseweb='select'] div {
  color: var(--text) !important;
  font-family: 'IBM Plex Sans', Inter, sans-serif !important;
}
[data-baseweb='popover'],
[data-baseweb='menu'],
[role='listbox'] {
  background: #0B1117 !important;
  border: 1px solid rgba(0, 184, 212, 0.34) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}
[role='option'] {
  background: #0B1117 !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Sans', Inter, sans-serif !important;
}
[role='option']:hover,
[aria-selected='true'][role='option'] {
  background: rgba(0, 230, 118, 0.12) !important;
  color: #F4FFFF !important;
}
[data-baseweb='tag'] {
  background: rgba(0, 230, 118, 0.12) !important;
  border: 1px solid rgba(0, 230, 118, 0.32) !important;
  color: var(--text) !important;
}

[data-testid='stAlert'] {
  background: rgba(13, 20, 27, 0.96) !important;
  color: var(--text) !important;
  border: 1px solid rgba(0, 184, 212, 0.22) !important;
  border-radius: 8px !important;
}

.incident-detail,
.evidence-panel,
.playbook-panel {
  background: linear-gradient(180deg, rgba(15, 23, 31, 0.98), rgba(7, 12, 16, 0.98));
  border: 1px solid rgba(0, 230, 118, 0.18);
  border-radius: 8px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 16px 45px rgba(0,0,0,0.20);
}
.incident-detail {
  padding: 1.05rem 1.12rem;
  margin: 0.45rem 0 0.8rem 0;
}
.incident-detail-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
}
.incident-eyebrow {
  color: var(--accent);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em !important;
  text-transform: uppercase;
}
.incident-title {
  margin-top: 0.3rem;
  color: #F4FFFF;
  font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif;
  font-size: 1.25rem;
  font-weight: 700;
  line-height: 1.18;
}
.incident-score {
  min-width: 5rem;
  text-align: right;
  color: var(--accent-2);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.1rem;
  font-weight: 700;
}
.incident-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.85rem;
}
.incident-meta span:not(.sev-pill) {
  padding: 0.28rem 0.5rem;
  border: 1px solid rgba(51, 74, 86, 0.82);
  border-radius: 999px;
  background: rgba(5, 10, 14, 0.72);
  color: var(--text-2);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.73rem;
}
.evidence-panel,
.playbook-panel {
  padding: 0.85rem;
  min-height: 12rem;
}
.evidence-row {
  display: grid;
  grid-template-columns: minmax(7rem, 0.46fr) 1fr;
  gap: 0.8rem;
  padding: 0.62rem 0;
  border-bottom: 1px solid rgba(35, 49, 60, 0.76);
}
.evidence-row:last-child { border-bottom: 0; }
.evidence-row span {
  color: var(--text-3);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.73rem;
  text-transform: uppercase;
}
.evidence-row strong {
  color: var(--text);
  font-family: 'IBM Plex Sans', Inter, sans-serif;
  font-size: 0.9rem;
  font-weight: 600;
}
.playbook-panel ol {
  margin: 0;
  padding-left: 1.25rem;
}
.playbook-panel li {
  color: var(--text-2);
  font-size: 0.92rem;
  line-height: 1.55;
  margin: 0.34rem 0;
}

.blueprint-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.72rem;
  margin: 0.6rem 0 1rem 0;
}
.blueprint-card {
  min-height: 9.6rem;
  padding: 0.9rem;
  background: linear-gradient(180deg, rgba(15, 23, 31, 0.98), rgba(7, 12, 16, 0.98));
  border: 1px solid rgba(0, 230, 118, 0.18);
  border-radius: 8px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 12px 36px rgba(0,0,0,0.18);
}
.blueprint-num {
  color: var(--accent);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  font-weight: 700;
}
.blueprint-title {
  margin-top: 0.35rem;
  color: #F4FFFF;
  font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif;
  font-size: 1.02rem;
  font-weight: 700;
}
.blueprint-copy {
  margin-top: 0.45rem;
  color: var(--text-2);
  font-size: 0.84rem;
  line-height: 1.45;
}
.blueprint-owner {
  margin-top: 0.7rem;
  color: var(--text-3);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
}

.brief-panel,
.quality-panel,
.action-panel {
  background: linear-gradient(180deg, rgba(15, 23, 31, 0.98), rgba(7, 12, 16, 0.98));
  border: 1px solid rgba(0, 230, 118, 0.18);
  border-radius: 8px;
  padding: 1rem 1.05rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 12px 36px rgba(0,0,0,0.18);
}
.brief-status {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.32rem 0.55rem;
  border-radius: 999px;
  border: 1px solid rgba(0, 230, 118, 0.35);
  background: rgba(0, 230, 118, 0.10);
  color: var(--accent);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
}
.brief-headline {
  margin-top: 0.75rem;
  color: #F4FFFF;
  font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif;
  font-size: 1.35rem;
  line-height: 1.2;
  font-weight: 700;
}
.brief-meta {
  margin-top: 0.55rem;
  color: var(--text-3);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.76rem;
}
.brief-list {
  margin: 0.6rem 0 0 0;
  padding-left: 1.05rem;
}
.brief-list li {
  margin: 0.42rem 0;
  color: var(--text-2);
  line-height: 1.45;
}
.quality-bar {
  width: 100%;
  height: 0.72rem;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(51, 74, 86, 0.55);
  border: 1px solid rgba(51, 74, 86, 0.8);
}
.quality-fill {
  height: 100%;
  background: linear-gradient(90deg, #00E676, #00B8D4);
  border-radius: inherit;
}
.micro-label {
  color: var(--text-3);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  text-transform: uppercase;
}

@media (max-width: 900px) {
  .block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }
  .hero {
    min-height: 250px;
    padding: 1.1rem;
    background-position: 62% center;
  }
  .hero-title { font-size: 1.75rem; }
  .hero-sub { font-size: 0.8rem; }
  .stTabs [data-baseweb="tab"] {
    padding: 0.46rem 0.7rem;
    font-size: 0.8rem;
  }
  .blueprint-grid {
    grid-template-columns: 1fr;
  }
}
</style>
""".replace("__HERO_BG__", HERO_BG_URI), unsafe_allow_html=True)


# ============================================================================
# RESOURCE LOADERS  (cached, mtime-keyed for hot reload)
# ============================================================================
def _mtime(p): return p.stat().st_mtime if p.exists() else 0

@st.cache_resource
def _load_detector_inner(path, mtime):
    return NIDSDetector(model_path=path)

def load_detector():
    p = PROJECT_ROOT / "models" / "nids_model.pkl"
    if not p.exists(): return None
    try: return _load_detector_inner(str(p), _mtime(p))
    except Exception as e:
        st.error(f"Failed to load CICIDS detector: {e}")
        return None

@st.cache_resource
def get_alert_log():
    return AlertLog(db_path=PROJECT_ROOT / "reports" / "alerts.db")

@st.cache_resource
def get_audit_log():
    return OperatorAudit(db_path=PROJECT_ROOT / "reports" / "audit.db")

@st.cache_data
def load_manifest(path):
    p = Path(path)
    if not p.exists(): return {}
    try: return json.loads(p.read_text())
    except Exception: return {}

@st.cache_data
def list_data_csvs():
    return sorted([str(p) for p in (PROJECT_ROOT / "data").glob("*.csv") if not p.name.startswith("_upload")])


def safe_upload_name(name: str) -> str:
    """Keep uploaded filenames local, short, and filesystem-safe."""
    base = Path(str(name or "flow.csv")).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return (base or "flow.csv")[:120]


def format_bytes(num_bytes: int | float | None) -> str:
    size = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def label_column(columns):
    for col in columns:
        if str(col).strip().lower() == "label":
            return col
    return None


def _read_csv_chunks(path, chunksize=120_000):
    last_error = None
    for enc in ("latin-1", "cp1252", "utf-8"):
        try:
            reader = pd.read_csv(path, chunksize=chunksize, encoding=enc, low_memory=False)
            for chunk in reader:
                yield clean_columns(chunk)
            return
        except UnicodeDecodeError as e:
            last_error = e
            continue
    if last_error:
        raise last_error


def _take_group_sample(group, n, seed):
    if len(group) <= n:
        return group
    return group.sample(n=n, random_state=seed)


def _finalize_balanced(frames, max_rows, seed=42):
    if not frames:
        return pd.DataFrame(), {}
    df = pd.concat(frames, ignore_index=True)
    if "__aies_label" not in df.columns:
        return df.head(max_rows), {}

    labels = [l for l in df["__aies_label"].dropna().unique().tolist()]
    if not labels:
        return df.drop(columns=["__aies_label"], errors="ignore").head(max_rows), {}

    per_label = max(1, max_rows // len(labels))
    balanced = []
    leftovers = []
    for label, group in df.groupby("__aies_label", sort=False):
        take = min(per_label, len(group))
        balanced.append(_take_group_sample(group, take, seed))
        if len(group) > take:
            leftovers.append(group.drop(balanced[-1].index, errors="ignore"))

    out = pd.concat(balanced, ignore_index=True) if balanced else pd.DataFrame()
    if len(out) < max_rows and leftovers:
        rest = pd.concat(leftovers, ignore_index=True)
        need = min(max_rows - len(out), len(rest))
        out = pd.concat([out, _take_group_sample(rest, need, seed + 7)], ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)

    profile = out["__aies_label"].value_counts().to_dict()
    return out.drop(columns=["__aies_label"], errors="ignore").reset_index(drop=True), profile


def _finalize_scenario_sample(frames, max_rows, targets, seed=42):
    if not frames:
        return pd.DataFrame(), {}
    df = pd.concat(frames, ignore_index=True)
    if "__aies_label" not in df.columns:
        return df.head(max_rows), {}

    selected = []
    leftovers = []
    for label, group in df.groupby("__aies_label", sort=False):
        take = min(int(targets.get(label, 0)), len(group))
        if take > 0:
            picked = _take_group_sample(group, take, seed + len(selected))
            selected.append(picked)
            if len(group) > take:
                leftovers.append(group.drop(picked.index, errors="ignore"))
        else:
            leftovers.append(group)

    out = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if len(out) < max_rows and leftovers:
        rest = pd.concat(leftovers, ignore_index=True)
        need = min(max_rows - len(out), len(rest))
        out = pd.concat([out, _take_group_sample(rest, need, seed + 17)], ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)

    profile = out["__aies_label"].value_counts().to_dict() if "__aies_label" in out else {}
    return out.drop(columns=["__aies_label"], errors="ignore").reset_index(drop=True), profile


@st.cache_data(show_spinner=False)
def load_smart_sample(path_str, max_rows, strategy="smart", seed=42):
    path = Path(path_str)
    if strategy == "head":
        df = pd.read_csv(path, nrows=max_rows, low_memory=False)
        df = clean_columns(df)
        lab = label_column(df.columns)
        profile = df[lab].map(normalize_label).value_counts().to_dict() if lab else {}
        return df, profile

    target_per_label = max(30, int(max_rows * 0.65))
    buckets = {}
    has_label = False
    for chunk in _read_csv_chunks(path):
        lab = label_column(chunk.columns)
        if not lab:
            return chunk.head(max_rows), {}
        has_label = True
        chunk["__aies_label"] = chunk[lab].map(normalize_label)
        for label, group in chunk.groupby("__aies_label", sort=False):
            current = sum(len(x) for x in buckets.get(label, []))
            if current >= target_per_label:
                continue
            need = target_per_label - current
            buckets.setdefault(label, []).append(_take_group_sample(group, min(need, len(group)), seed + len(buckets)))

    if not has_label:
        df = pd.read_csv(path, nrows=max_rows, low_memory=False)
        return clean_columns(df), {}
    return _finalize_balanced([x for parts in buckets.values() for x in parts], max_rows, seed)


@st.cache_data(show_spinner=False)
def load_uploaded_csv_bundle(path_strs, max_rows, strategy="smart", seed=42):
    """Load one or more uploaded CICIDS CSVs into a single smart-balanced batch."""
    paths = [Path(p) for p in path_strs if p]
    if not paths:
        return pd.DataFrame(), {}
    if len(paths) == 1:
        return load_smart_sample(str(paths[0]), max_rows, strategy=strategy, seed=seed)

    per_file_rows = max(50, int(np.ceil(max_rows / max(len(paths), 1))))
    frames = []
    for idx, path in enumerate(paths):
        df, _profile = load_smart_sample(str(path), per_file_rows, strategy=strategy, seed=seed + idx * 17)
        if df.empty:
            continue
        lab = label_column(df.columns)
        if lab:
            df = df.copy()
            df["__aies_label"] = df[lab].map(normalize_label)
        frames.append(df)

    if not frames:
        return pd.DataFrame(), {}
    if any("__aies_label" in f.columns for f in frames):
        return _finalize_balanced(frames, max_rows, seed)

    out = pd.concat(frames, ignore_index=True).head(max_rows)
    return out.reset_index(drop=True), {}


@st.cache_data(show_spinner=False)
def load_mixed_attack_lab(max_rows, scenario="Executive mixed", seed=42):
    paths = [Path(p) for p in list_data_csvs()]
    targets = scenario_targets(max_rows, scenario)
    target_classes = list(targets.keys())
    collect_caps = {label: max(25, count) for label, count in targets.items()}
    buckets = {c: [] for c in target_classes}

    for path in paths:
        for chunk in _read_csv_chunks(path):
            lab = label_column(chunk.columns)
            if not lab:
                continue
            chunk["__aies_label"] = chunk[lab].map(normalize_label)
            for label, group in chunk.groupby("__aies_label", sort=False):
                if label not in buckets:
                    continue
                current = sum(len(x) for x in buckets[label])
                collect_cap = collect_caps[label]
                if current >= collect_cap:
                    continue
                need = collect_cap - current
                buckets[label].append(_take_group_sample(group, min(need, len(group)), seed + current))
        if all(sum(len(x) for x in buckets[c]) >= collect_caps[c] or c == "Infiltration" for c in target_classes):
            # Infiltration has only 36 rows in CICIDS, so don't block on it.
            break

    frames = [x for parts in buckets.values() for x in parts]
    return _finalize_scenario_sample(frames, max_rows, targets, seed)


# ============================================================================
# UI HELPERS
# ============================================================================
def kpi(label, value, delta=None, value_class=""):
    delta_html = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value {value_class}'>{value}</div>{delta_html}</div>",
        unsafe_allow_html=True,
    )

def section(text):
    st.markdown(f"<div class='sect-h'>{text}</div>", unsafe_allow_html=True)

def severity_pill(sev):
    return f"<span class='sev-pill sev-{sev}'>{sev.upper()}</span>"

SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Informational": 0}

CLASS_ACTIONS = {
    "DDoS": [
        "Rate-limit edge traffic and enable upstream scrubbing.",
        "Check service saturation, SYN backlog, and CDN/WAF telemetry.",
        "Preserve flow sample and export STIX/CEF bundle for SIEM correlation.",
    ],
    "DoS": [
        "Throttle offending service path and validate server resource pressure.",
        "Review destination port exposure and firewall policy.",
        "Open an availability incident if repeated high-confidence flows continue.",
    ],
    "PortScan": [
        "Block or watchlist scanning origin and inspect exposed services.",
        "Compare scanned ports with approved attack surface inventory.",
        "Escalate if scan is followed by brute-force or exploitation-stage activity.",
    ],
    "BruteForce": [
        "Enforce account lockout and inspect authentication logs around the same window.",
        "Check SSH/FTP/VPN endpoints for repeated failed logins.",
        "Rotate affected credentials if successful login indicators are found.",
    ],
    "Botnet": [
        "Isolate suspected endpoint and inspect outbound C2 beacon patterns.",
        "Run EDR triage for persistence, scheduled tasks, and suspicious DNS.",
        "Block destination indicators and hunt for same pattern across the fleet.",
    ],
    "WebAttack": [
        "Review WAF/application logs for payloads, parameters, and response codes.",
        "Patch exposed application path and validate input filtering.",
        "Preserve HTTP evidence and open an application security ticket.",
    ],
    "Infiltration": [
        "Isolate host/network segment and begin containment immediately.",
        "Search for lateral movement, privilege escalation, and data staging.",
        "Escalate to incident response and preserve forensic image/logs.",
    ],
}


def incident_key(row):
    return f"INC-{int(row['flow_idx']):05d}"


def incident_priority(row):
    sev = str(row.get("severity", "Informational"))
    conf = float(row.get("confidence", 0) or 0)
    if sev == "Critical" or (sev == "High" and conf >= 0.85):
        return "P1"
    if sev == "High" or (sev == "Medium" and conf >= 0.80):
        return "P2"
    if sev == "Medium" or str(row.get("predicted_class", "")) != "BENIGN":
        return "P3"
    return "P4"


def incident_status(row):
    sev = str(row.get("severity", "Informational"))
    if sev in {"Critical", "High"}:
        return "Triage now"
    if sev == "Medium":
        return "Review"
    return "Monitor"


def prediction_for_flow(flow_idx):
    preds = st.session_state.get("last_predictions") or []
    try:
        idx = int(flow_idx)
    except Exception:
        return None
    if 0 <= idx < len(preds):
        return preds[idx]
    return None


def advisory_actions_for(row, prediction=None):
    if prediction and prediction.advisories:
        actions = prediction.advisories[0].recommended_actions
        if actions:
            return actions
    return CLASS_ACTIONS.get(str(row.get("predicted_class", "")), [
        "Keep monitoring the flow and correlate with surrounding telemetry.",
        "Export the evidence bundle if this event appears in a broader campaign.",
    ])


def render_incident_detail(row):
    prediction = prediction_for_flow(row.get("flow_idx"))
    advisory = prediction.advisories[0] if prediction and prediction.advisories else None
    key = incident_key(row)
    sev = str(row.get("severity", "Informational"))
    cls = str(row.get("predicted_class", "Unknown"))
    conf = float(row.get("confidence", 0) or 0)
    rule_id = advisory.rule_id if advisory else "Session rule output"
    because = advisory.because if advisory and advisory.because else "Evidence reconstructed from the enriched prediction row."
    actions = advisory_actions_for(row, prediction)

    st.markdown(
        f"""
        <div class="incident-detail">
          <div class="incident-detail-head">
            <div>
              <div class="incident-eyebrow">{key} · {incident_priority(row)} · {incident_status(row)}</div>
              <div class="incident-title">{cls} on {row.get('service', 'Unknown')} / port {row.get('dest_port', '—')}</div>
            </div>
            <div class="incident-score">{conf:.3f}</div>
          </div>
          <div class="incident-meta">
            <span class="sev-pill sev-{sev}">{sev.upper()}</span>
            <span>MITRE {row.get('mitre_id', '—')} · {row.get('mitre_technique', '—')}</span>
            <span>{row.get('tactic', '—')} / {row.get('kill_chain_stage', '—')}</span>
            <span>{row.get('country', '—')} · {row.get('asn', '—')}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns([1.05, 1])
    with d1:
        section("Why this matters")
        st.markdown(
            f"""
            <div class="evidence-panel">
              <div class="evidence-row"><span>Primary advisory</span><strong>{row.get('advisory_title', '—')}</strong></div>
              <div class="evidence-row"><span>Rule source</span><strong>{rule_id}</strong></div>
              <div class="evidence-row"><span>Reason</span><strong>{because}</strong></div>
              <div class="evidence-row"><span>Flow index</span><strong>{int(row.get('flow_idx', 0))}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d2:
        section("Response playbook")
        action_items = "".join(f"<li>{a}</li>" for a in actions)
        st.markdown(f"<div class='playbook-panel'><ol>{action_items}</ol></div>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.columns(4)
    with t1: kpi("Incident", key, incident_priority(row))
    with t2: kpi("Confidence", f"{conf:.3f}", "Model score")
    with t3: kpi("Technique", str(row.get("mitre_id", "—")), str(row.get("tactic", "—")))
    with t4: kpi("Stage", str(row.get("kill_chain_stage", "—")), str(row.get("service", "—")))


def rule_inventory_df():
    candidates = [
        Fact("DDoS", 0.93, {"Flow Packets/s": 150_000, "Flow Bytes/s": 900_000, "SYN Flag Count": 12}),
        Fact("DoS", 0.91, {"Active Mean": 8000, "Idle Mean": 1200, "Flow Duration": 200_000}),
        Fact("PortScan", 0.86, {"Total Fwd Packets": 80, "SYN Flag Count": 24, "Destination Port": 22}),
        Fact("BruteForce", 0.88, {"Destination Port": 22, "Total Fwd Packets": 120, "Flow Duration": 400_000}),
        Fact("Botnet", 0.90, {"Idle Mean": 70_000, "Flow Duration": 700_000, "Bwd Packets/s": 4}),
        Fact("WebAttack", 0.89, {"Destination Port": 80, "Total Length of Fwd Packets": 42_000}),
        Fact("Infiltration", 0.87, {"Flow Duration": 900_000, "Total Length of Bwd Packets": 90_000}),
        Fact("BENIGN", 0.97, {}),
        Fact("PortScan", 0.42, {"Total Fwd Packets": 20, "SYN Flag Count": 5, "Destination Port": 445}),
    ]
    rows = []
    for rule in _build_rules():
        matched_fact = None
        advisory = None
        for fact in candidates:
            try:
                if rule.when(fact):
                    matched_fact = fact
                    advisory = rule.then(fact)
                    break
            except Exception:
                continue
        rows.append({
            "Rule ID": rule.rule_id,
            "Fires on": matched_fact.predicted_class if matched_fact else "Feature condition",
            "Severity": advisory.severity if advisory else "—",
            "Advisory": advisory.title if advisory else "—",
            "Purpose": rule.description,
            "Actions": len(advisory.recommended_actions) if advisory else 0,
        })
    return pd.DataFrame(rows)


def disclose(text):
    st.markdown(f"<div class='disclose'>{text}</div>", unsafe_allow_html=True)

def empty_state(icon, title, message, cta_text="Go to Run Analysis →"):
    """Render a clean empty-state card. Used when shared_enriched is None."""
    st.markdown(
        f"<div class='empty-state'>"
        f"<div class='empty-state-icon'>{icon}</div>"
        f"<div class='empty-state-title'>{title}</div>"
        f"<div class='empty-state-msg'>{message}</div>"
        f"<div class='empty-state-cta'>{cta_text}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

def plotly_clean(**overrides):
    """Dark cyber Plotly base config. Keeps charts readable inside SOC panels."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0B1117",
        font=dict(family="Inter, sans-serif", color="#E8F1F2", size=12),
        xaxis=dict(gridcolor="#23313C", zerolinecolor="#334A56", linecolor="#334A56"),
        yaxis=dict(gridcolor="#23313C", zerolinecolor="#334A56", linecolor="#334A56"),
        margin=dict(t=40, l=20, r=20, b=20),
        colorway=["#00E676", "#00B8D4", "#F6C343", "#4EA8FF", "#FF7A45", "#8BA1AA", "#FF4D4D"],
    )
    base.update(overrides)
    return base


# Severity → color (for inline use)
SEV_COLOR = {
    "Critical": "#FF4D4D", "High": "#FF7A45", "Medium": "#F6C343",
    "Low": "#4EA8FF", "Informational": "#8BA1AA",
}


# ============================================================================
# RESOURCES
# ============================================================================
detector = load_detector()
alerts = get_alert_log()
audit = get_audit_log()
manifest = load_manifest(PROJECT_ROOT / "models" / "model_manifest.json")
nslkdd_manifest = load_manifest(PROJECT_ROOT / "models" / "nslkdd_manifest.json")
health_df = project_health(PROJECT_ROOT)
health_meta = health_summary(health_df)

# ============================================================================
# SESSION STATE
# ============================================================================
if "operator_id" not in st.session_state:
    st.session_state["operator_id"] = get_default_operator()
if "session_start" not in st.session_state:
    st.session_state["session_start"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    audit.log(st.session_state["operator_id"], "SESSION_START", "AIES-NIDS-v4", {"client": "streamlit"})
if "shared_enriched" not in st.session_state:
    st.session_state["shared_enriched"] = None
if "show_historical" not in st.session_state:
    st.session_state["show_historical"] = False  # By default, only show CURRENT-session data
if "last_summary" not in st.session_state:
    st.session_state["last_summary"] = None
if "last_predictions" not in st.session_state:
    st.session_state["last_predictions"] = None
if "last_sample_profile" not in st.session_state:
    st.session_state["last_sample_profile"] = None
if "last_sampling_mode" not in st.session_state:
    st.session_state["last_sampling_mode"] = None
if "last_quality" not in st.session_state:
    st.session_state["last_quality"] = None
if "last_brief" not in st.session_state:
    st.session_state["last_brief"] = None
if "last_timeline" not in st.session_state:
    st.session_state["last_timeline"] = None
if "last_attack_surface" not in st.session_state:
    st.session_state["last_attack_surface"] = None
if "last_remediation" not in st.session_state:
    st.session_state["last_remediation"] = None
if "last_model_risk" not in st.session_state:
    st.session_state["last_model_risk"] = None
if "last_case_board" not in st.session_state:
    st.session_state["last_case_board"] = None

session_start_iso = st.session_state["session_start"]
since_filter = None if st.session_state["show_historical"] else session_start_iso

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 🛡 AIES NIDS")
    st.caption("Hybrid AI + Expert System NIDS")
    st.caption(f"v4.0 · CT-361 AIES CCP · NEDU")

    st.markdown("---")
    st.markdown("**Operator**")
    st.markdown(f"`{st.session_state['operator_id']}`")
    st.caption(f"Session opened: {session_start_iso[11:19]}Z")

    st.markdown("---")
    st.markdown("**Sensors**")
    if detector:
        st.markdown(f"✓ CICIDS RandomForest")
        st.caption(f"Macro-F1: {detector.training_macro_f1:.3f}")
    else:
        st.error("CICIDS model not loaded")
    if nslkdd_manifest:
        st.markdown(f"✓ NSL-KDD {nslkdd_manifest.get('model_name', 'LogReg')}")
        st.caption(f"Macro-F1: {nslkdd_manifest.get('macro_f1', 0):.3f}")
    st.caption(
        f"System health: {health_meta['status']} · "
        f"{health_meta['pass']} pass / {health_meta['warn']} warn / {health_meta['fail']} fail"
    )

    st.markdown("---")
    st.markdown("**View Mode**")
    show_hist = st.checkbox(
        "Include historical alerts",
        value=st.session_state["show_historical"],
        help="By default, the dashboard shows only the current session. Enable this to also include alerts from previous runs.",
    )
    st.session_state["show_historical"] = show_hist

    st.markdown("---")
    st.markdown("**Maintenance**")
    if st.button("Clear Cache", width="stretch"):
        st.cache_data.clear(); st.cache_resource.clear()
        audit.log(st.session_state["operator_id"], "CACHE_CLEAR", "all", {})
        st.success("Caches cleared. Reloading…")
        st.rerun()
    if st.button("Reset Alert Log", width="stretch", help="Permanently deletes all alerts from alerts.db."):
        alerts.clear()
        audit.log(st.session_state["operator_id"], "ALERTS_CLEAR", "alerts.db", {})
        st.success("Alert log cleared.")
        st.rerun()
    if st.button("Reset Session View", width="stretch", help="Clear in-memory analysis results without touching the database."):
        st.session_state["shared_enriched"] = None
        st.session_state["last_summary"] = None
        st.session_state["last_predictions"] = None
        st.session_state["last_sample_profile"] = None
        st.session_state["last_sampling_mode"] = None
        st.session_state["last_quality"] = None
        st.session_state["last_brief"] = None
        st.session_state["last_timeline"] = None
        st.session_state["last_attack_surface"] = None
        st.session_state["last_remediation"] = None
        st.session_state["last_model_risk"] = None
        st.session_state["last_case_board"] = None
        st.success("Session view reset.")
        st.rerun()

# ============================================================================
# HERO
# ============================================================================
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(
    f"""<div class='hero'>
    <div class='hero-kicker'>AI + Cybersecurity Ops</div>
    <div class='hero-title'>AIES <span>NIDS</span> Command Center</div>
    <div class='hero-sub'>Hybrid AI + Forward-chaining Expert System  ·  CICIDS2017 trained  ·  NSL-KDD validated  ·  MITRE / Kill Chain / Forensics ready</div>
    <div class='hero-ts'>
      <span class='hero-chip'><strong>UTC</strong> {now_str}</span>
      <span class='hero-chip'><strong>OPERATOR</strong> {st.session_state['operator_id'].upper()}</span>
      <span class='hero-chip'><strong>SESSION</strong> {session_start_iso[11:19]}Z</span>
    </div>
    </div>""",
    unsafe_allow_html=True,
)

# ============================================================================
# TABS
# ============================================================================
tabs = st.tabs([
    "🔬  Run Analysis",
    "📊  Overview",
    "🌍  Geographic",
    "🎯  ATT&CK Matrix",
    "⛓  Kill Chain",
    "🔎  Forensics",
    "🤖  Model",
    "📋  Compliance",
    "🏗  Architecture",
    "🧠  AIES Theory",
    "🧭  SOC Brief",
])

# ============================================================================
# TAB 0  ·  RUN ANALYSIS  (primary entry point — always available)
# ============================================================================
with tabs[0]:
    section("Run an analysis batch")
    disclose(
        "<b>Replay Mode.</b> This dashboard analyses CICIDS-format flow CSV files. "
        "Real-time packet capture (Scapy / Zeek / CICFlowMeter) is provided as a stub for demo purposes only. "
        "All charts in other tabs populate AFTER you run an analysis below."
    )

    mode = st.radio(
        "Analysis mode",
        ["Batch (recommended)", "Streaming replay (for demo flair)"],
        horizontal=True,
        help="Batch is the standard workflow — predict + enrich + populate dashboard. Streaming replay just visualises the same data flow-by-flow at adjustable FPS.",
    )

    if mode == "Batch (recommended)":
        c1, c2 = st.columns([2, 1])
        with c1:
            src_choice = st.radio(
                "Source",
                ["Mixed Attack Lab", "Bundled sample CSV", "Upload CSV"],
                horizontal=True,
                help="Mixed Attack Lab builds a balanced demo batch from multiple CICIDS files so the dashboard shows different attack families instead of one flat class.",
            )
            src_path, src_label = None, None
            src_paths = []
            scenario_preset = "Executive mixed"
            if src_choice == "Mixed Attack Lab":
                src_path, src_label = "__mixed_attack_lab__", "Mixed Attack Lab"
                scenario_preset = st.selectbox(
                    "Scenario preset",
                    scenario_names(),
                    help="Choose the investigation story you want the analysis to produce from real local CICIDS rows.",
                )
                st.caption(scenario_description(scenario_preset))
                st.caption("First build scans the local CICIDS CSVs once; after that Streamlit cache makes repeat runs fast.")
            elif src_choice == "Upload CSV":
                uploaded_files = st.file_uploader(
                    "Drop one or more CICIDS-format flow CSV files",
                    type=["csv"],
                    accept_multiple_files=True,
                    help="You can upload multiple CICIDS-style CSV files. The app combines them and smart-balances visible labels for one analysis run. Limit: 300MB per file.",
                )
                if uploaded_files:
                    max_upload_bytes = 300 * 1024 * 1024
                    too_large = [u.name for u in uploaded_files if getattr(u, "size", 0) and u.size > max_upload_bytes]
                    upload_preview = pd.DataFrame([
                        {
                            "File": safe_upload_name(u.name),
                            "Size": format_bytes(getattr(u, "size", 0)),
                        }
                        for u in uploaded_files
                    ])
                    st.caption(
                        f"Selected {len(uploaded_files)} CSV file(s), "
                        f"{format_bytes(sum(getattr(u, 'size', 0) or 0 for u in uploaded_files))} total."
                    )
                    st.dataframe(upload_preview, width="stretch", hide_index=True, height=min(210, 38 + len(upload_preview) * 36))
                    if too_large:
                        st.error("These CSVs are larger than 300MB each: " + ", ".join(too_large))
                    else:
                        upload_dir = PROJECT_ROOT / "reports" / "uploads"
                        upload_dir.mkdir(parents=True, exist_ok=True)
                        stamp = int(time.time())
                        saved_names = []
                        for idx, uploaded in enumerate(uploaded_files, start=1):
                            clean_name = safe_upload_name(uploaded.name)
                            tmp = upload_dir / f"_upload_{stamp}_{idx}_{clean_name}"
                            tmp.write_bytes(bytes(uploaded.getbuffer()))
                            src_paths.append(str(tmp))
                            saved_names.append(clean_name)
                        if src_paths:
                            src_path = "__uploaded_csv_bundle__"
                            src_label = saved_names[0] if len(saved_names) == 1 else f"{len(saved_names)} uploaded CSVs"
                            st.caption("Uploaded CSV bundle: " + ", ".join(saved_names[:4]) + (" ..." if len(saved_names) > 4 else ""))
            else:
                csvs = list_data_csvs()
                if csvs:
                    sel = st.selectbox("Pick one", csvs, format_func=lambda p: Path(p).name)
                    src_path, src_label = sel, Path(sel).name
                else:
                    st.warning("No CSVs found in data/")
                    src_path, src_label = None, None

        with c2:
            max_rows = st.slider("Rows to analyse", 100, 10_000, 1000, 100,
                help="Larger batches give more representative dashboards but take longer. 1,000–2,000 is a sweet spot for live demo.")
            sampling_mode = st.radio(
                "Sampling",
                ["Smart balanced", "First rows"],
                horizontal=True,
                help="Smart balanced scans the CSV label column and samples across classes. First rows preserves raw file order and may be all BENIGN in CICIDS files.",
                disabled=(src_choice == "Mixed Attack Lab"),
            )
            log_to_db = st.checkbox("Log alerts to forensic DB", value=True,
                help="Each non-Informational advisory is written to reports/alerts.db with microsecond timestamps.")

        st.markdown("")
        run_now = st.button("Execute Analysis", type="primary", width="content", disabled=(detector is None or src_path is None))

        if detector is None:
            st.error("Detector not loaded — check models/nids_model.pkl exists.")
        elif run_now and src_path:
            t0 = time.time()
            with st.spinner(f"Reading {src_label}…"):
                if src_path == "__mixed_attack_lab__":
                    df_raw, sample_profile = load_mixed_attack_lab(max_rows, scenario=scenario_preset)
                    sampling_label = f"Mixed Attack Lab / {scenario_preset}"
                elif src_path == "__uploaded_csv_bundle__":
                    strategy = "smart" if sampling_mode == "Smart balanced" else "head"
                    df_raw, sample_profile = load_uploaded_csv_bundle(tuple(src_paths), max_rows, strategy=strategy)
                    sampling_label = f"{sampling_mode} / {len(src_paths)} uploaded CSVs"
                else:
                    strategy = "smart" if sampling_mode == "Smart balanced" else "head"
                    df_raw, sample_profile = load_smart_sample(src_path, max_rows, strategy=strategy)
                    sampling_label = sampling_mode
            with st.spinner(f"Running {len(df_raw):,} flows through hybrid AI + expert system…"):
                preds = detector.predict_dataframe(df_raw)
                summary = detector.summary(preds)
                enriched = enrich_predictions(preds, df_raw)
            elapsed = time.time() - t0
            quality = build_data_quality_report(df_raw, sample_profile)
            brief = build_command_brief(
                enriched,
                summary,
                quality,
                source_label=src_label,
                sampling_mode=sampling_label,
                elapsed_sec=elapsed,
            )
            timeline = build_campaign_timeline(enriched)
            attack_surface = build_attack_surface(enriched)
            remediation = build_remediation_plan(enriched)
            risk_summary = model_risk_summary(preds)
            case_board = build_case_board(enriched)

            logged = 0
            if log_to_db:
                logged = alerts.log_predictions(preds, skip_informational=True)

            # Persist to session for cross-tab consumption
            st.session_state["shared_enriched"] = enriched
            st.session_state["last_summary"] = summary
            st.session_state["last_predictions"] = preds
            st.session_state["last_elapsed"] = elapsed
            st.session_state["last_src_label"] = src_label
            st.session_state["last_sampling_mode"] = sampling_label
            st.session_state["last_sample_profile"] = sample_profile
            st.session_state["last_logged"] = logged
            st.session_state["last_df_raw"] = df_raw
            st.session_state["last_quality"] = quality
            st.session_state["last_brief"] = brief
            st.session_state["last_timeline"] = timeline
            st.session_state["last_attack_surface"] = attack_surface
            st.session_state["last_remediation"] = remediation
            st.session_state["last_model_risk"] = risk_summary
            st.session_state["last_case_board"] = case_board

            audit.log(
                st.session_state["operator_id"], "BATCH_ANALYSIS", src_label,
                {
                    "rows": int(len(df_raw)),
                    "sampling": sampling_label,
                    "source_profile": sample_profile,
                    "alerts_logged": int(logged),
                    "elapsed_sec": round(elapsed, 3),
                    "quality_score": quality.get("score"),
                    "brief_status": brief.get("status"),
                    "model_risk": risk_summary.get("status"),
                    "cases": int(len(case_board)),
                },
            )
            st.success(f"✓ Analysed {len(df_raw):,} flows in {elapsed:.2f}s · {logged:,} alerts logged · all dashboards refreshed.")
            if sample_profile:
                st.caption("Source-label mix used for this run: " + ", ".join(f"{k}: {v}" for k, v in sample_profile.items()))
                if len(sample_profile) <= 1 and src_choice != "Mixed Attack Lab":
                    st.info("This source contains one visible label in the selected sample. Use Mixed Attack Lab for a multi-attack viva/demo, or Smart balanced on a CSV that contains multiple labels.")

        # If we have results, show a compact summary card right here
        if st.session_state.get("last_summary"):
            s = st.session_state["last_summary"]
            q = st.session_state.get("last_quality") or {}
            section("Last run summary")
            cs = st.columns(6)
            with cs[0]: kpi("Total flows", f"{s.get('total_flows', 0):,}")
            with cs[1]: kpi("Attack ratio", f"{s.get('attack_ratio', 0):.1%}",
                            value_class="coral" if s.get('attack_ratio', 0) > 0 else "")
            with cs[2]: kpi("Max severity", s.get('max_severity', '—'),
                            value_class={"Critical":"critical","High":"high","Medium":"medium","Low":"low","Informational":""}.get(s.get('max_severity',''),''))
            with cs[3]: kpi("Rules fired", f"{s.get('rules_fired', 0):,}")
            with cs[4]: kpi("Alerts logged", f"{st.session_state.get('last_logged', 0):,}",
                            f"{st.session_state.get('last_elapsed', 0):.2f}s elapsed")
            with cs[5]: kpi("Data quality", f"{q.get('score', 0)}/100", q.get("grade", "—"))

            source_profile = st.session_state.get("last_sample_profile") or {}
            if source_profile:
                st.caption(
                    f"Sampling: {st.session_state.get('last_sampling_mode', '—')} · Source label mix: "
                    + ", ".join(f"{k}={v}" for k, v in source_profile.items())
                )

            section("Class distribution")
            cls_counts = pd.Series(s.get("class_counts", {})).sort_values(ascending=False)
            if not cls_counts.empty:
                fig = go.Figure(go.Bar(
                    x=cls_counts.values, y=cls_counts.index, orientation="h",
                    marker=dict(color=["#00B8D4" if c == "BENIGN" else "#00E676" for c in cls_counts.index],
                                line=dict(color="#23313C", width=1)),
                    text=cls_counts.values, textposition="outside",
                ))
                fig.update_layout(**plotly_clean(height=max(220, 40*len(cls_counts)),
                    margin=dict(t=10, l=10, r=40, b=10), yaxis=dict(autorange="reversed")))
                st.plotly_chart(fig, width="stretch")

            with st.expander("View per-flow predictions table"):
                tbl = detector.to_table([])  # signature OK with empty list
                # Rebuild from session predictions if needed; otherwise show enriched
                enr = st.session_state.get("shared_enriched")
                if enr is not None:
                    show_df = enr[["flow_idx", "predicted_class", "confidence", "severity",
                                   "advisory_title", "mitre_id", "tactic", "kill_chain_stage"]].head(200)
                    st.dataframe(show_df, width="stretch", hide_index=True)

    else:
        # Streaming replay
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            csvs = list_data_csvs()
            replay_choice = st.selectbox("Source CSV", csvs, format_func=lambda p: Path(p).name) if csvs else None
        with c2:
            fps = st.slider("Flows / second", 1, 200, 30)
        with c3:
            n_flows = st.slider("Total flows", 50, 2000, 300, 50)

        if st.button("Start Streaming", type="primary", width="content", disabled=(detector is None or not replay_choice)):
            audit.log(st.session_state["operator_id"], "LIVE_REPLAY_START", replay_choice, {"fps": fps, "n_flows": n_flows})
            df_full, replay_profile = load_smart_sample(replay_choice, n_flows, strategy="smart")
            if replay_profile:
                st.caption("Replay source-label mix: " + ", ".join(f"{k}: {v}" for k, v in replay_profile.items()))
            chart_ph = st.empty()
            log_ph = st.empty()
            counts = {}
            advisory_log = []
            t0 = time.time()
            for i, batch_idx in enumerate(range(0, len(df_full), max(1, fps // 4))):
                batch = df_full.iloc[batch_idx:batch_idx + max(1, fps // 4)].copy()
                if batch.empty: break
                preds = detector.predict_dataframe(batch)
                for p in preds:
                    counts[p.predicted_class] = counts.get(p.predicted_class, 0) + 1
                    if p.advisories and p.advisories[0].severity != "Informational":
                        advisory_log.append({
                            "flow": p.flow_index, "class": p.predicted_class,
                            "severity": p.advisories[0].severity, "title": p.advisories[0].title,
                        })
                # Update chart
                cs = pd.Series(counts).sort_values(ascending=False)
                bar = go.Figure(go.Bar(
                    x=cs.values, y=cs.index, orientation="h",
                    marker=dict(color=["#00B8D4" if c == "BENIGN" else "#00E676" for c in cs.index],
                                line=dict(color="#23313C", width=1)),
                    text=cs.values, textposition="outside",
                ))
                bar.update_layout(**plotly_clean(height=300, margin=dict(t=20, l=10, r=40, b=10),
                    yaxis=dict(autorange="reversed"), title=f"Live class distribution · {sum(counts.values())} flows"))
                chart_ph.plotly_chart(bar, width="stretch", key=f"live_{i}")
                if advisory_log:
                    log_ph.dataframe(pd.DataFrame(advisory_log).tail(15), width="stretch", hide_index=True)
                time.sleep(max(0.05, 1.0 / max(fps, 1)))
            audit.log(st.session_state["operator_id"], "LIVE_REPLAY_END", replay_choice,
                      {"flows_processed": sum(counts.values()), "advisories": len(advisory_log)})
            st.success(f"Replay finished · {sum(counts.values())} flows · {time.time() - t0:.1f}s · {len(advisory_log)} non-informational advisories")


# ============================================================================
# TAB 1  ·  OVERVIEW (Command Center)
# Uses current in-memory analysis first; falls back to alerts.db for history.
# ============================================================================
with tabs[1]:
    section("Threat picture · current session" + (" + history" if st.session_state["show_historical"] else ""))

    current_enr = st.session_state.get("shared_enriched")
    use_current_session = (
        current_enr is not None
        and not current_enr.empty
        and not st.session_state["show_historical"]
    )
    if use_current_session:
        sev_dict = severity_counts_from_enriched(current_enr, include_info=False)
        total_alerts = int(sum(sev_dict.values()))
        recent = event_feed_from_enriched(current_enr, limit=500)
        posture = compute_threat_score(sev_dict, total_flows=max(len(current_enr), 1))
        overview_source = "Current analysis result"
    else:
        sev = alerts.severity_breakdown(since=since_filter)
        sev_dict = dict(zip(sev["severity"], sev["n"])) if not sev.empty else {}
        total_alerts = int(sev["n"].sum()) if not sev.empty else 0
        recent = alerts.recent(limit=500, since=since_filter) if total_alerts > 0 else pd.DataFrame()
        posture = compute_threat_score(sev_dict, total_flows=max(total_alerts, 1))
        overview_source = "Forensic DB" + (" + history" if st.session_state["show_historical"] else "")

    if total_alerts == 0:
        empty_state(
            "🛡",
            "All quiet on the wire.",
            "No alerts have been recorded in the current session yet. Run a batch analysis on the Run Analysis tab to populate the threat picture, or enable Include historical alerts in the sidebar to see prior runs.",
            "Go to Run Analysis →",
        )
    else:
        st.caption(f"Threat picture source: {overview_source}")
        # Top row — DEFCON gauge + 4 KPIs
        g1, g2 = st.columns([1.2, 2.5])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=posture["score"],
                number=dict(font=dict(family="Inter", size=44, color=posture["color"]), suffix="<span style='font-size:0.5em;color:#8BA1AA'> /100</span>"),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#334A56", tickfont=dict(color="#A8B7BE", size=10)),
                    bar=dict(color=posture["color"], thickness=0.28),
                    bgcolor="#0B1117",
                    borderwidth=1,
                    bordercolor="#23313C",
                    steps=[
                        dict(range=[0, 20], color="rgba(0,184,212,0.12)"),
                        dict(range=[20, 40], color="rgba(78,168,255,0.13)"),
                        dict(range=[40, 60], color="rgba(246,195,67,0.18)"),
                        dict(range=[60, 85], color="rgba(255,122,69,0.22)"),
                        dict(range=[85, 100], color="rgba(255,77,77,0.28)"),
                    ],
                    threshold=dict(line=dict(color=posture["color"], width=3), thickness=0.85, value=posture["score"]),
                ),
                title=dict(text=f"<b>{posture['level']}</b><br><span style='font-size:0.55em;color:#8BA1AA'>{posture['message']}</span>",
                           font=dict(family="Inter", size=16, color=posture["color"])),
            ))
            gauge.update_layout(**plotly_clean(height=300, margin=dict(t=70, l=20, r=20, b=10)))
            st.plotly_chart(gauge, width="stretch")

        with g2:
            k1, k2, k3, k4 = st.columns(4)
            with k1: kpi("Total alerts", f"{total_alerts:,}", "Current session")
            with k2: kpi("Critical", f"{sev_dict.get('Critical', 0):,}", value_class="critical")
            with k3: kpi("High", f"{sev_dict.get('High', 0):,}", value_class="high")
            with k4: kpi("Medium", f"{sev_dict.get('Medium', 0):,}", value_class="medium")
            k5, k6, k7, k8 = st.columns(4)
            with k5: kpi("Sensor", "ONLINE" if detector else "OFFLINE",
                         f"F1 {detector.training_macro_f1:.3f}" if detector else "—",
                         value_class="" if detector else "critical")
            with k6: kpi("Models", "2", "CICIDS + NSL-KDD")
            with k7: kpi("Audit ops", f"{audit.summary()['total_actions']:,}")
            with k8: kpi("Uptime", session_start_iso[11:19] + "Z", "Since session start")

        # Threat feed + severity donut
        f1, f2 = st.columns([1.4, 1])
        with f1:
            section("Recent events · last 25")
            rows = []
            for _, r in recent.head(25).iterrows():
                ts_raw = str(r.get("ts_utc", "")) if r.get("ts_utc", "") else "?"
                ts = ts_raw[11:23] if len(ts_raw) >= 19 and ("T" in ts_raw or "-" in ts_raw) else ts_raw
                col = SEV_COLOR.get(r["severity"], "#8BA1AA")
                rows.append(
                    f"<div class='feed-row'>"
                    f"<span class='feed-ts'>{ts}</span>"
                    f"<span class='feed-cls' style='color:{col}'>[{r['severity'].upper()}]</span>"
                    f"<span style='color:#E8F1F2'>{r['pred_class']}</span>"
                    f"<span style='color:#A8B7BE'>{r['title']}</span>"
                    f"<span style='color:#8BA1AA;margin-left:auto'>conf {float(r['confidence']):.2f}</span>"
                    f"</div>"
                )
            st.markdown(f"<div class='feed'>{''.join(rows)}</div>", unsafe_allow_html=True)

        with f2:
            section("Severity breakdown")
            donut = go.Figure(data=[go.Pie(
                labels=list(sev_dict.keys()), values=list(sev_dict.values()), hole=0.62,
                marker=dict(colors=[SEV_COLOR.get(s, "#8BA1AA") for s in sev_dict.keys()],
                            line=dict(color="#0B1117", width=2)),
                textinfo="label+percent", textfont=dict(family="Inter", size=11, color="#E8F1F2"),
                textposition="outside",
            )])
            donut.update_layout(**plotly_clean(height=290, showlegend=False, margin=dict(t=10, l=10, r=10, b=10),
                annotations=[dict(text=f"<b>{total_alerts:,}</b><br><span style='font-size:0.55em;color:#8BA1AA'>events</span>",
                    x=0.5, y=0.5, font=dict(family="Inter", size=22, color="#00E676"), showarrow=False)]))
            st.plotly_chart(donut, width="stretch")

        # MITRE + Kill chain (from recent alerts)
        cls_counts = recent["pred_class"].value_counts()
        mitre_rows = []
        kc_counts = {s: 0 for s in KILL_CHAIN_STAGES}
        for cls, cnt in cls_counts.items():
            m = MITRE_MAPPING.get(cls, MITRE_MAPPING.get("BENIGN", {}))
            if m.get("technique_id") and m["technique_id"] != "—":
                mitre_rows.append({"mitre_id": m["technique_id"], "technique": m.get("technique_name", ""),
                                   "tactic": m.get("tactic", ""), "count": int(cnt)})
            stage = KILL_CHAIN_MAPPING.get(cls)
            if stage in kc_counts:
                kc_counts[stage] += int(cnt)
        mt_df = pd.DataFrame(mitre_rows)

        m1, m2 = st.columns(2)
        with m1:
            section("MITRE ATT&CK · technique hits")
            if not mt_df.empty:
                fig = go.Figure(go.Bar(
                    x=mt_df["count"], y=mt_df["mitre_id"] + " " + mt_df["technique"], orientation="h",
                    marker=dict(color="#00E676", line=dict(color="#23313C", width=1)),
                    text=mt_df["count"], textposition="outside",
                ))
                fig.update_layout(**plotly_clean(height=280, margin=dict(t=10, l=10, r=40, b=10),
                    yaxis=dict(autorange="reversed")))
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("Only BENIGN traffic — no ATT&CK techniques mapped.")

        with m2:
            section("Kill chain · stage progression")
            kc_df = pd.DataFrame([{"stage": s, "count": kc_counts[s]} for s in KILL_CHAIN_STAGES if kc_counts[s] > 0])
            if not kc_df.empty:
                funnel = go.Figure(go.Funnel(
                    y=kc_df["stage"], x=kc_df["count"],
                    marker=dict(color=["#00B8D4", "#4EA8FF", "#F6C343", "#00E676", "#FF7A45", "#FF5F45", "#FF4D4D"][:len(kc_df)],
                                line=dict(color="#23313C", width=1)),
                    textinfo="value+percent total", textfont=dict(family="Inter", color="#E8F1F2", size=11),
                ))
                funnel.update_layout(**plotly_clean(height=280, margin=dict(t=10, l=10, r=10, b=10)))
                st.plotly_chart(funnel, width="stretch")
            else:
                st.caption("No attack-stage events in current view.")


# ============================================================================
# TAB 2  ·  GEOGRAPHIC
# ============================================================================
with tabs[2]:
    section("Geographic threat origins")
    disclose(
        "<b>Synthetic geographic attribution.</b> CICIDS2017 strips source/destination IPs, so true geo-lookup is impossible. "
        "Origins below are produced by hashing flow features into a weighted distribution matching real threat-intel reports "
        "(CrowdStrike GTR / Mandiant M-Trends 2024). For production, replace <code>geo_enrich()</code> with MaxMind GeoLite2."
    )

    enr = st.session_state.get("shared_enriched")
    if enr is None or enr.empty:
        empty_state("🌍", "No analysis data yet.",
            "Geographic enrichment runs on the result of an analysis batch. Go to Run Analysis, execute a batch on a CICIDS CSV, then return here to see the world map and origin breakdown.")
    else:
        attacks = enr[enr["predicted_class"] != "BENIGN"]
        if attacks.empty:
            st.success("All flows in the last batch were classified BENIGN. No hostile origins to map.")
        else:
            country_counts = attacks["country"].value_counts()

            c1, c2 = st.columns([1.6, 1])
            with c1:
                section("World map · attack origins")
                geo_df = attacks.groupby(["country", "iso", "lat", "lon"]).size().reset_index(name="count")
                geo_df = geo_df[geo_df["country"] != "Internal LAN"]
                if not geo_df.empty:
                    fig = go.Figure(go.Scattergeo(
                        lon=geo_df["lon"], lat=geo_df["lat"], text=geo_df["country"] + " · " + geo_df["count"].astype(str) + " flows",
                        mode="markers",
                        marker=dict(size=geo_df["count"].clip(lower=8, upper=40), color=geo_df["count"],
                                    colorscale=[[0, "#F6C343"], [0.5, "#FF7A45"], [1, "#FF4D4D"]],
                                    line=dict(color="#0B1117", width=1), opacity=0.9),
                    ))
                    fig.update_geos(projection_type="natural earth", showland=True, landcolor="#101A22",
                                    showocean=True, oceancolor="#061016", showcountries=True, countrycolor="#334A56",
                                    showcoastlines=True, coastlinecolor="#334A56")
                    fig.update_layout(**plotly_clean(height=440, margin=dict(t=10, l=10, r=10, b=10)))
                    st.plotly_chart(fig, width="stretch")

            with c2:
                section("Top hostile origins")
                top = country_counts[country_counts.index != "Internal LAN"].head(10)
                if not top.empty:
                    fig = go.Figure(go.Bar(
                        x=top.values, y=top.index, orientation="h",
                        marker=dict(color=top.values, colorscale=[[0, "#F6C343"], [1, "#FF4D4D"]],
                                    line=dict(color="#23313C", width=1)),
                        text=top.values, textposition="outside",
                    ))
                    fig.update_layout(**plotly_clean(height=440, margin=dict(t=10, l=10, r=40, b=10),
                        yaxis=dict(autorange="reversed")))
                    st.plotly_chart(fig, width="stretch")

            section("Country → service → attack class")
            sankey_df = attacks.copy()
            sankey_df = sankey_df[sankey_df["country"] != "Internal LAN"]
            if not sankey_df.empty:
                # Build sankey nodes/links
                countries = sankey_df["country"].value_counts().head(8).index.tolist()
                services = sankey_df["service"].value_counts().head(8).index.tolist()
                classes = sankey_df["predicted_class"].value_counts().index.tolist()
                nodes = countries + services + classes
                node_idx = {n: i for i, n in enumerate(nodes)}
                links = {"source": [], "target": [], "value": []}
                # country -> service
                for (co, sv), n in sankey_df[sankey_df["country"].isin(countries) & sankey_df["service"].isin(services)].groupby(["country", "service"]).size().items():
                    links["source"].append(node_idx[co]); links["target"].append(node_idx[sv]); links["value"].append(int(n))
                # service -> class
                for (sv, cl), n in sankey_df[sankey_df["service"].isin(services)].groupby(["service", "predicted_class"]).size().items():
                    links["source"].append(node_idx[sv]); links["target"].append(node_idx[cl]); links["value"].append(int(n))
                if links["value"]:
                    sankey = go.Figure(go.Sankey(
                        node=dict(label=nodes, pad=15, thickness=18,
                                  color=["#00B8D4"] * len(countries) + ["#F6C343"] * len(services) + ["#00E676"] * len(classes),
                                  line=dict(color="#23313C", width=1),
                                  hovertemplate="<b>%{label}</b><extra></extra>"),
                        link=dict(source=links["source"], target=links["target"], value=links["value"],
                                  color="rgba(0,230,118,0.22)",
                                  hovertemplate="%{source.label} \u2192 %{target.label}: %{value}<extra></extra>"),
                    ))
                    sankey.update_layout(**plotly_clean(height=380, margin=dict(t=10, l=10, r=10, b=10),
                        hovermode="closest",
                        font=dict(family="Inter", size=11, color="#E8F1F2")))
                    st.plotly_chart(sankey, width="stretch")


# ============================================================================
# TAB 3  ·  ATT&CK MATRIX
# ============================================================================
with tabs[3]:
    section("MITRE ATT&CK technique inventory")

    enr = st.session_state.get("shared_enriched")
    if enr is None or enr.empty:
        empty_state("🎯", "No analysis data yet.",
            "ATT&CK matrix populates from the techniques mapped in the most recent analysis batch. Run an analysis to see which tactics and techniques the classifier triggered.")
    else:
        attacks = enr[enr["predicted_class"] != "BENIGN"]
        if attacks.empty:
            st.success("Last batch contained only BENIGN traffic — no ATT&CK techniques mapped.")
        else:
            # Tactic × class heatmap
            section("Tactic × predicted class")
            pivot = attacks.groupby(["tactic", "predicted_class"]).size().unstack(fill_value=0)
            if not pivot.empty:
                heat = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns, y=pivot.index,
                    colorscale=[[0, "#0B1117"], [0.3, "#F6C343"], [0.7, "#00E676"], [1, "#FF4D4D"]],
                    text=pivot.values, texttemplate="%{text}", textfont=dict(family="Inter", color="#E8F1F2", size=12),
                    showscale=True, colorbar=dict(title="hits", thickness=12, tickfont=dict(color="#A8B7BE")),
                ))
                heat.update_layout(**plotly_clean(height=300, margin=dict(t=10, l=10, r=10, b=40)))
                st.plotly_chart(heat, width="stretch")

            section("Technique inventory")
            tech_df = attacks.groupby(["mitre_id", "mitre_technique", "tactic", "predicted_class"]).size().reset_index(name="hits")
            tech_df = tech_df.sort_values("hits", ascending=False)
            st.dataframe(
                tech_df.rename(columns={"mitre_id": "MITRE ID", "mitre_technique": "Technique",
                                        "tactic": "Tactic", "predicted_class": "Detected as", "hits": "Hits"}),
                width="stretch", hide_index=True,
            )


# ============================================================================
# TAB 4  ·  KILL CHAIN
# ============================================================================
with tabs[4]:
    section("Lockheed Martin Cyber Kill Chain")

    enr = st.session_state.get("shared_enriched")
    if enr is None or enr.empty:
        empty_state("⛓", "No analysis data yet.",
            "Kill chain progression maps detected attack classes to the 7 stages of the Lockheed Martin model. Run an analysis to see where in the attacker progression the events sit.")
    else:
        attacks = enr[enr["predicted_class"] != "BENIGN"]
        if attacks.empty:
            st.success("Last batch contained only BENIGN traffic — kill chain is empty.")
        else:
            kc_counts = {s: 0 for s in KILL_CHAIN_STAGES}
            for cls in attacks["predicted_class"]:
                stage = KILL_CHAIN_MAPPING.get(cls)
                if stage in kc_counts:
                    kc_counts[stage] += 1

            # Funnel
            kc_df = pd.DataFrame([{"stage": s, "count": kc_counts[s]} for s in KILL_CHAIN_STAGES if kc_counts[s] > 0])
            if not kc_df.empty:
                stage_colors = ["#00B8D4", "#4EA8FF", "#F6C343", "#00E676", "#FF7A45", "#FF5F45", "#FF4D4D"]
                funnel = go.Figure(go.Funnel(
                    y=kc_df["stage"], x=kc_df["count"],
                    marker=dict(color=[stage_colors[KILL_CHAIN_STAGES.index(s) % len(stage_colors)] for s in kc_df["stage"]],
                                line=dict(color="#23313C", width=1.5)),
                    textinfo="value+percent total", textfont=dict(family="Inter", color="#E8F1F2", size=12),
                ))
                funnel.update_layout(**plotly_clean(height=380, margin=dict(t=10, l=10, r=10, b=10)))
                st.plotly_chart(funnel, width="stretch")

                # Intelligence assessment
                early_stages = sum(kc_counts[s] for s in KILL_CHAIN_STAGES[:3])
                late_stages = sum(kc_counts[s] for s in KILL_CHAIN_STAGES[3:])
                section("Intelligence assessment")
                if late_stages > early_stages * 2:
                    st.error("**Late-stage activity dominant.** Events concentrated at exploitation/installation/C2/objectives — possible APT footprint or successful breach in progress. Recommend immediate IR team activation.")
                elif early_stages > late_stages * 2:
                    st.warning("**Reconnaissance dominant.** Most events are early-stage (recon, weaponization, delivery). Attacker is likely still scouting. Time to harden defences before they pivot.")
                else:
                    st.info("**Mixed-stage activity.** Both early and late stages observed — could be parallel campaigns or one mature attacker active across multiple targets.")


# ============================================================================
# TAB 5  ·  FORENSICS
# ============================================================================
with tabs[5]:
    section("Evidence table & SIEM export")

    enr = st.session_state.get("shared_enriched")
    if enr is None or enr.empty:
        empty_state("🔎", "No analysis data yet.",
            "Forensics shows per-flow evidence + SIEM export bundles built from the most recent analysis. Run an analysis first, then come back to filter, inspect, and export.")
    else:
        attacks_only = st.checkbox("Show only non-BENIGN flows", value=True)
        evidence = enr[enr["predicted_class"] != "BENIGN"] if attacks_only else enr

        f1, f2, f3 = st.columns(3)
        with f1:
            sev_filter = st.multiselect("Severity", ["Critical", "High", "Medium", "Low", "Informational"],
                default=["Critical", "High", "Medium"])
        with f2:
            cls_filter = st.multiselect("Class", sorted(enr["predicted_class"].unique()),
                default=[c for c in enr["predicted_class"].unique() if c != "BENIGN"])
        with f3:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

        filt = evidence[
            evidence["severity"].isin(sev_filter)
            & evidence["predicted_class"].isin(cls_filter)
            & (evidence["confidence"] >= min_conf)
        ]

        # KPI strip
        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        with kc1: kpi("Total flows", f"{len(enr):,}")
        with kc2: kpi("Non-BENIGN", f"{(enr['predicted_class'] != 'BENIGN').sum():,}", value_class="coral")
        with kc3: kpi("Critical", f"{(enr['severity'] == 'Critical').sum():,}", value_class="critical")
        with kc4: kpi("High", f"{(enr['severity'] == 'High').sum():,}", value_class="high")
        with kc5: kpi("Filtered view", f"{len(filt):,}")

        case_board = st.session_state.get("last_case_board")
        if case_board is None or getattr(case_board, "empty", True):
            case_board = build_case_board(enr)
        cmeta = case_summary(case_board)
        section("Case board")
        cb1, cb2, cb3, cb4, cb5 = st.columns(5)
        with cb1: kpi("Cases", f"{cmeta['cases']:,}", cmeta["status"])
        with cb2: kpi("P1", f"{cmeta['p1']:,}", value_class="critical" if cmeta["p1"] else "")
        with cb3: kpi("P2", f"{cmeta['p2']:,}", value_class="high" if cmeta["p2"] else "")
        with cb4: kpi("Lead owner", cmeta["top_owner"])
        with cb5: kpi("Evidence", f"{len(filt):,}", "filtered flows")
        if case_board.empty:
            st.success("No non-BENIGN cases in the current analysis.")
        else:
            st.dataframe(case_board, width="stretch", hide_index=True, height=260)
            st.download_button(
                "Export Case Board",
                data=cases_markdown(case_board),
                file_name=f"aies_nids_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                width="content",
            )

        section("Incident queue")
        if filt.empty:
            empty_state(
                "⌁",
                "No incidents match the current filters.",
                "Relax the severity, class, or confidence filters to bring events back into the triage queue.",
                "Adjust filters above",
            )
        else:
            queue = filt.copy()
            queue["_sev_rank"] = queue["severity"].map(SEVERITY_RANK).fillna(0)
            queue["_priority"] = queue.apply(incident_priority, axis=1)
            queue["_incident"] = queue.apply(incident_key, axis=1)
            queue["_status"] = queue.apply(incident_status, axis=1)
            queue = queue.sort_values(["_sev_rank", "confidence"], ascending=[False, False]).head(100)

            q1, q2 = st.columns([0.78, 1.22])
            with q1:
                queue_view = queue[[
                    "_incident", "_priority", "_status", "severity", "predicted_class",
                    "confidence", "mitre_id", "kill_chain_stage", "service", "dest_port",
                ]].rename(columns={
                    "_incident": "Incident",
                    "_priority": "Priority",
                    "_status": "Status",
                    "severity": "Severity",
                    "predicted_class": "Class",
                    "confidence": "Confidence",
                    "mitre_id": "MITRE",
                    "kill_chain_stage": "Stage",
                    "service": "Service",
                    "dest_port": "Port",
                })
                st.dataframe(queue_view, width="stretch", hide_index=True, height=292)

                options = queue["flow_idx"].astype(int).tolist()
                selected_flow = st.selectbox(
                    "Open incident",
                    options=options,
                    format_func=lambda i: (
                        f"{incident_key(queue.loc[queue['flow_idx'] == i].iloc[0])} · "
                        f"{queue.loc[queue['flow_idx'] == i, 'severity'].iloc[0]} · "
                        f"{queue.loc[queue['flow_idx'] == i, 'predicted_class'].iloc[0]} · "
                        f"conf {float(queue.loc[queue['flow_idx'] == i, 'confidence'].iloc[0]):.3f}"
                    ),
                )
            with q2:
                selected_row = queue[queue["flow_idx"].astype(int) == int(selected_flow)].iloc[0]
                render_incident_detail(selected_row)

        section("Evidence")
        st.dataframe(
            filt[["flow_idx", "predicted_class", "confidence", "severity", "advisory_title",
                  "mitre_id", "tactic", "kill_chain_stage", "country", "service", "dest_port"]].head(500),
            width="stretch", hide_index=True,
        )

        # SIEM export
        section("SIEM export · STIX 2.1 / CEF / Syslog / JSON")
        st.caption("Each export action is logged to the operator audit trail with timestamp + filename + alert count.")
        non_benign = enr[enr["predicted_class"] != "BENIGN"]
        non_benign_count = len(non_benign)
        now_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            stix = json.dumps(to_stix_bundle(non_benign), indent=2) if non_benign_count > 0 else "{}"
            st.download_button(
                "📦  STIX 2.1 bundle", data=stix,
                file_name=f"aies_nids_stix_{now_ts}.json", mime="application/json",
                disabled=non_benign_count == 0, width="stretch",
                on_click=lambda: audit.log(st.session_state["operator_id"], "STIX_EXPORT", f"aies_nids_stix_{now_ts}.json", {"alerts": non_benign_count}),
            )
        with e2:
            cef = to_cef(non_benign) if non_benign_count > 0 else ""
            st.download_button(
                "📋  CEF (ArcSight)", data=cef,
                file_name=f"aies_nids_cef_{now_ts}.log", mime="text/plain",
                disabled=non_benign_count == 0, width="stretch",
                on_click=lambda: audit.log(st.session_state["operator_id"], "CEF_EXPORT", f"aies_nids_cef_{now_ts}.log", {"alerts": non_benign_count}),
            )
        with e3:
            sl = to_syslog(non_benign) if non_benign_count > 0 else ""
            st.download_button(
                "📡  RFC 5424 syslog", data=sl,
                file_name=f"aies_nids_syslog_{now_ts}.log", mime="text/plain",
                disabled=non_benign_count == 0, width="stretch",
                on_click=lambda: audit.log(st.session_state["operator_id"], "SYSLOG_EXPORT", f"aies_nids_syslog_{now_ts}.log", {"alerts": non_benign_count}),
            )
        with e4:
            js = to_json_export(non_benign) if non_benign_count > 0 else "{}"
            st.download_button(
                "📄  Generic JSON", data=js,
                file_name=f"aies_nids_alerts_{now_ts}.json", mime="application/json",
                disabled=non_benign_count == 0, width="stretch",
                on_click=lambda: audit.log(st.session_state["operator_id"], "JSON_EXPORT", f"aies_nids_alerts_{now_ts}.json", {"alerts": non_benign_count}),
            )

        # Operator audit trail
        section("Operator audit trail · last 50 actions")
        recent_audit = audit.recent(limit=50)
        has_data = (hasattr(recent_audit, "empty") and not recent_audit.empty) or (isinstance(recent_audit, list) and len(recent_audit) > 0)
        if has_data:
            df_audit = pd.DataFrame(recent_audit) if isinstance(recent_audit, list) else recent_audit
            st.dataframe(df_audit, width="stretch", hide_index=True)
        else:
            st.caption("No operator actions logged yet.")


# ============================================================================
# TAB 6  ·  MODEL  (always available — model perf doesn't need session data)
# ============================================================================
with tabs[6]:
    section("Detector performance · CICIDS2017 + NSL-KDD")

    if not detector:
        st.error("Detector not loaded.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi("CICIDS macro-F1", f"{detector.training_macro_f1:.3f}", "RandomForest")
        with k2: kpi("CICIDS accuracy", f"{detector.training_accuracy:.3f}")
        with k3: kpi("NSL-KDD macro-F1", f"{nslkdd_manifest.get('macro_f1', 0):.3f}",
                    nslkdd_manifest.get("model_name", "—"))
        with k4: kpi("Classes", f"{len(detector.label_classes)}", "CICIDS multiclass")

        # Honest weakness disclosure
        disclose(
            "<b>Honest weaknesses.</b> Botnet precision = 0.37 (severe class imbalance, only 0.07% of training rows). "
            "Infiltration test support = 7 rows (statistically thin). NSL-KDD U2R F1 = 0.12 (long-tail attack types absent from training). "
            "These are documented in <code>STUDY_GUIDE.md</code> and defended operationally via R-Unsure-001 (analyst-review routing for low-confidence non-BENIGN predictions)."
        )

        preds = st.session_state.get("last_predictions") or []
        enr = st.session_state.get("shared_enriched")
        if preds:
            risk = st.session_state.get("last_model_risk") or model_risk_summary(preds)
            section("Current batch confidence and uncertainty")
            r1, r2, r3, r4, r5, r6 = st.columns(6)
            with r1: kpi("Risk gate", risk.get("status", "—"))
            with r2: kpi("Median conf", f"{risk.get('median_confidence', 0):.3f}")
            with r3: kpi("P10 conf", f"{risk.get('p10_confidence', 0):.3f}")
            with r4: kpi("Low conf", f"{risk.get('low_confidence', 0):,}", value_class="medium" if risk.get("low_confidence", 0) else "")
            with r5: kpi("High entropy", f"{risk.get('high_entropy', 0):,}", value_class="medium" if risk.get("high_entropy", 0) else "")
            with r6: kpi("Review rate", f"{risk.get('review_rate', 0):.1%}", value_class="coral" if risk.get("review_rate", 0) > 0.08 else "")

            uncertainty = uncertainty_rows(preds, enr)
            u1, u2 = st.columns([1.05, 0.95])
            with u1:
                conf_fig = go.Figure(go.Histogram(
                    x=uncertainty["confidence"],
                    nbinsx=25,
                    marker=dict(color="#00E676", line=dict(color="#23313C", width=1)),
                    hovertemplate="Confidence %{x:.3f}<br>Flows %{y}<extra></extra>",
                ))
                conf_fig.update_layout(**plotly_clean(height=300, margin=dict(t=20, l=10, r=20, b=35)))
                st.plotly_chart(conf_fig, width="stretch")
            with u2:
                cls_risk = class_risk_table(preds)
                st.dataframe(cls_risk, width="stretch", hide_index=True, height=300)

            rq = review_queue(preds, enr, limit=50)
            section("Analyst review queue · uncertainty candidates")
            if rq.empty:
                st.success("No uncertainty candidates in the current batch. Confidence and class margins are stable.")
            else:
                st.dataframe(
                    rq[["flow_idx", "predicted_class", "severity", "confidence", "entropy", "margin", "review_reason", "service", "mitre_id"]],
                    width="stretch",
                    hide_index=True,
                    height=280,
                )
                st.caption("These are not necessarily wrong predictions; they are the flows an analyst should inspect before automation.")
        else:
            section("Current batch confidence and uncertainty")
            st.caption("Run an analysis to populate live confidence distribution, uncertainty candidates, and class-wise reliability.")

        # Per-class metrics
        per_class_csv = PROJECT_ROOT / "reports" / "per_class_metrics.csv"
        if per_class_csv.exists():
            section("CICIDS per-class metrics")
            pc = pd.read_csv(per_class_csv)
            st.dataframe(pc.round(3), width="stretch", hide_index=True)

            # Radar chart
            classes = pc["class"].tolist() if "class" in pc.columns else pc.iloc[:, 0].tolist()
            # Auto-detect f1 column (CSV uses 'f1', legacy code expected 'f1-score')
            f1_col = "f1-score" if "f1-score" in pc.columns else ("f1" if "f1" in pc.columns else None)
            metrics = [c for c in ["precision", "recall", f1_col] if c and c in pc.columns]
            if metrics:
                fig = go.Figure()
                colors = ["#00E676", "#00B8D4", "#F6C343"]
                for i, m in enumerate(metrics):
                    fig.add_trace(go.Scatterpolar(
                        r=pc[m].tolist() + [pc[m].iloc[0]],
                        theta=classes + [classes[0]],
                        fill="toself", name=m, line=dict(color=colors[i], width=2),
                    ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    polar=dict(bgcolor="#0B1117",
                               radialaxis=dict(visible=True, range=[0, 1], gridcolor="#23313C"),
                               angularaxis=dict(gridcolor="#23313C")),
                    font=dict(family="Inter", color="#E8F1F2"),
                    height=420, showlegend=True, margin=dict(t=20, l=40, r=40, b=20),
                )
                st.plotly_chart(fig, width="stretch")

        # NSL-KDD comparison
        nsl_csv = PROJECT_ROOT / "reports" / "nslkdd" / "per_class_metrics.csv"
        if nsl_csv.exists():
            section("NSL-KDD per-class metrics (KDDTest+ holdout)")
            nsl_pc = pd.read_csv(nsl_csv)
            st.dataframe(nsl_pc.round(3), width="stretch", hide_index=True)
            st.caption("KDDTest+ contains 17 attack types absent from training — tests true generalisation. LogReg wins here over RF because L2 regularisation handles unseen patterns better than tree ensembles.")


# ============================================================================
# TAB 7  ·  COMPLIANCE
# ============================================================================
with tabs[7]:
    section("Framework coverage posture")
    disclose(
        "<b>Demo compliance coverage.</b> Each framework control is marked active when at least one detected attack class "
        "in the current session contributes to its evidence. This is a demonstration of how detection capability maps to "
        "controls — not an audit. For a real audit, you'd cross-reference actual SIEM/SOAR logs, not a single batch run."
    )

    enr = st.session_state.get("shared_enriched")
    if enr is None or enr.empty:
        empty_state("📋", "No analysis data yet.",
            "Compliance posture is computed from the attack classes detected in the most recent analysis. Run an analysis to see how the detection capability maps to NIST CSF, ISO 27001, and CIS Controls v8.")
    else:
        detected_classes = set(enr[enr["predicted_class"] != "BENIGN"]["predicted_class"].unique())
        if not detected_classes:
            st.info("Last batch was all BENIGN — no attack classes detected, so no framework controls activated.")
        else:
            c1, c2, c3 = st.columns(3)
            for col, fw_key, fw_name in [
                (c1, "NIST_CSF", "NIST Cybersecurity Framework"),
                (c2, "ISO_27001", "ISO/IEC 27001:2022"),
                (c3, "CIS_CONTROLS", "CIS Controls v8"),
            ]:
                with col:
                    p = overall_posture_score(detected_classes, fw_key)
                    color = p["color"]
                    st.markdown(
                        f"<div class='kpi'>"
                        f"<div class='kpi-label'>{fw_name}</div>"
                        f"<div class='kpi-value' style='color:{color}'>{p['rating']}</div>"
                        f"<div class='kpi-delta'>{p['avg_coverage']:.1f}% avg coverage · {p['active_controls']}/{p['total_controls']} active</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            section("Drilldown")
            framework_choice = st.selectbox("Framework", ["NIST_CSF", "ISO_27001", "CIS_CONTROLS"],
                format_func=lambda k: {"NIST_CSF": "NIST Cybersecurity Framework",
                                       "ISO_27001": "ISO/IEC 27001:2022",
                                       "CIS_CONTROLS": "CIS Controls v8"}[k])
            cov_df = compute_coverage(detected_classes, framework_choice)
            if not cov_df.empty:
                st.dataframe(cov_df, width="stretch", hide_index=True)


# ============================================================================
# TAB 8  ·  ARCHITECTURE
# ============================================================================
with tabs[8]:
    section("System architecture")

    arch = """
```
                Network flow CSV  /  Replay stream
                              │
                              ▼
        ┌─────────────────────────────────────────────────────┐
        │  LAYER 1  ·  ML CLASSIFIER                          │
        │  RandomForest (CICIDS) + LogReg (NSL-KDD)           │
        │  78 → 70 features, 8 classes, class_weight=balanced │
        │  src/preprocessing.py + src/training.py             │
        └─────────────────────────────────────────────────────┘
                              │
              (predicted_class, confidence, features)
                              ▼
        ┌─────────────────────────────────────────────────────┐
        │  LAYER 2  ·  FORWARD-CHAINING EXPERT SYSTEM         │
        │  10 rules · severity ranking · "because" hooks      │
        │  R-Unsure-001 routes uncertain non-BENIGN to review │
        │  src/expert_system.py + src/prediction.py           │
        └─────────────────────────────────────────────────────┘
                              │
                  (advisory + severity)
                              ▼
        ┌─────────────────────────────────────────────────────┐
        │  LAYER 3  ·  THREAT-INTEL ENRICHMENT                │
        │  MITRE ATT&CK + Lockheed Kill Chain                 │
        │  Synthetic geographic attribution (transparent)     │
        │  Service identification by destination port         │
        │  DEFCON 1–5 threat scoring                          │
        │  src/threat_intel.py                                │
        └─────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────┐
        │  LAYER 4  ·  SIEM EXPORT + COMPLIANCE + AUDIT       │
        │  STIX 2.1 / CEF / RFC 5424 syslog / JSON            │
        │  NIST CSF + ISO 27001 + CIS v8 posture              │
        │  Operator audit trail (SQLite)                      │
        │  src/stix_export.py + compliance.py + audit.py      │
        └─────────────────────────────────────────────────────┘
                              │
                              ▼
                Forensic alert log + SIEM bundle download
```
"""
    st.code(arch, language="text")

    section("Operating model")
    st.markdown("""
<div class="blueprint-grid">
  <div class="blueprint-card">
    <div class="blueprint-num">01 / INGEST</div>
    <div class="blueprint-title">Flow Intake</div>
    <div class="blueprint-copy">Upload or replay CICIDS-format flow CSVs. The product boundary is network-flow telemetry, not raw packet capture.</div>
    <div class="blueprint-owner">data/ · live_capture.py</div>
  </div>
  <div class="blueprint-card">
    <div class="blueprint-num">02 / INFER</div>
    <div class="blueprint-title">ML Detection</div>
    <div class="blueprint-copy">Preprocess, align features, impute missing values, scale, then classify into 8 attack families with confidence.</div>
    <div class="blueprint-owner">preprocessing.py · prediction.py</div>
  </div>
  <div class="blueprint-card">
    <div class="blueprint-num">03 / REASON</div>
    <div class="blueprint-title">Expert Rules</div>
    <div class="blueprint-copy">Forward-chaining rules convert model output into severity, because text, and response actions.</div>
    <div class="blueprint-owner">expert_system.py</div>
  </div>
  <div class="blueprint-card">
    <div class="blueprint-num">04 / ENRICH</div>
    <div class="blueprint-title">Threat Context</div>
    <div class="blueprint-copy">Map detections to MITRE ATT&CK, kill chain, service, synthetic geo, and posture scoring.</div>
    <div class="blueprint-owner">threat_intel.py</div>
  </div>
  <div class="blueprint-card">
    <div class="blueprint-num">05 / ACT</div>
    <div class="blueprint-title">SOC Output</div>
    <div class="blueprint-copy">Triage incidents, inspect evidence, export STIX/CEF/syslog/JSON, and keep an operator audit trail.</div>
    <div class="blueprint-owner">alert_log.py · stix_export.py</div>
  </div>
</div>
""", unsafe_allow_html=True)

    section("System health")
    h1, h2, h3, h4 = st.columns(4)
    with h1: kpi("Readiness", health_meta["status"])
    with h2: kpi("Pass", f"{health_meta['pass']:,}")
    with h3: kpi("Warn", f"{health_meta['warn']:,}", value_class="medium" if health_meta["warn"] else "")
    with h4: kpi("Fail", f"{health_meta['fail']:,}", value_class="critical" if health_meta["fail"] else "")
    st.dataframe(health_df, width="stretch", hide_index=True)
    readiness_md = build_readiness_report(PROJECT_ROOT, health_df, health_meta)
    st.download_button(
        "Export Readiness Report",
        data=readiness_md,
        file_name=f"aies_nids_readiness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        width="content",
    )

    section("Expert rule book")
    st.dataframe(rule_inventory_df(), width="stretch", hide_index=True)

    section("Production roadmap")
    roadmap = [
        {"stage": "MVP", "capability": "Batch flow analysis", "definition of done": "CSV input produces predictions, advisories, dashboard state, and exports", "status": "Implemented"},
        {"stage": "SOC workflow", "capability": "Incident queue", "definition of done": "Analyst can prioritize, open, explain, and action individual alerts", "status": "Implemented"},
        {"stage": "Realtime", "capability": "Zeek/CICFlowMeter adapter", "definition of done": "Live flows stream into the same prediction + rule pipeline", "status": "Next"},
        {"stage": "Multi-user SaaS", "capability": "Auth + tenant isolation", "definition of done": "Separate operators, orgs, role-based access, and audit history", "status": "Next"},
        {"stage": "SOAR", "capability": "Playbook automation", "definition of done": "Optional integrations for firewall block, ticket creation, Slack/email escalation", "status": "Future"},
        {"stage": "Governance", "capability": "Model/rule registry", "definition of done": "Versioned model artifacts, rule approvals, drift metrics, reproducible reports", "status": "Future"},
    ]
    st.dataframe(pd.DataFrame(roadmap), width="stretch", hide_index=True)

    section("Module map")
    modules = [
        {"layer": "L1 · ML", "module": "src/preprocessing.py", "purpose": "CICIDS preprocessing, infinity replacement, median imputation"},
        {"layer": "L1 · ML", "module": "src/training.py", "purpose": "Multi-model race (RF + GB + LR), macro-F1 selection"},
        {"layer": "L1 · ML", "module": "src/evaluation.py", "purpose": "Confusion matrix, per-class metrics, feature importance"},
        {"layer": "L1 · ML", "module": "src/nslkdd_preprocessing.py", "purpose": "NSL-KDD pipeline (41 features, 3 categorical)"},
        {"layer": "L1 · ML", "module": "src/nslkdd_training.py", "purpose": "NSL-KDD trainer (LogReg winner)"},
        {"layer": "L1 · Ingest", "module": "src/scenario_lab.py", "purpose": "Mixed Attack Lab presets and per-class target weights"},
        {"layer": "L2 · ES", "module": "src/expert_system.py", "purpose": "10-rule forward-chaining engine — THE AIES CORE"},
        {"layer": "L2 · ES", "module": "src/prediction.py", "purpose": "Vectorised inference + advisory ranking"},
        {"layer": "L3 · Intel", "module": "src/threat_intel.py", "purpose": "MITRE/Kill Chain mapping, geo enrichment, DEFCON"},
        {"layer": "L4 · Export", "module": "src/stix_export.py", "purpose": "STIX 2.1 + CEF + syslog + JSON"},
        {"layer": "L4 · Export", "module": "src/compliance.py", "purpose": "NIST CSF + ISO 27001 + CIS v8 frameworks"},
        {"layer": "L4 · Export", "module": "src/operator_audit.py", "purpose": "Operator action audit trail (SQLite)"},
        {"layer": "L4 · Export", "module": "src/alert_log.py", "purpose": "Per-flow forensic alert log (SQLite, microsecond ts)"},
        {"layer": "L4 · SOC", "module": "src/mission_control.py", "purpose": "Command brief, data quality, timeline, attack surface, remediation plan"},
        {"layer": "L4 · SOC", "module": "src/case_management.py", "purpose": "Case roll-up, priority, owner, status, and first response action"},
        {"layer": "L4 · Assurance", "module": "src/model_risk.py", "purpose": "Confidence, entropy, margin, and analyst-review risk gate"},
        {"layer": "L4 · Ops", "module": "src/system_health.py", "purpose": "Local model/data/storage/docs readiness checks"},
        {"layer": "L4 · Ops", "module": "src/readiness_report.py", "purpose": "Pre-viva readiness report with health, models, scenarios, and scope"},
        {"layer": "L1 · ML", "module": "src/live_capture.py", "purpose": "Replay-mode flow generator + Scapy stub"},
        {"layer": "L5 · UI", "module": "src/aies_render.py", "purpose": "Interactive AIES theory demonstrations for viva/course defense"},
        {"layer": "L5 · UI", "module": "src/aies_inference.py", "purpose": "Forward trace, backward why-tree, CF, what-if, KB editor, report builders"},
    ]
    st.dataframe(pd.DataFrame(modules), width="stretch", hide_index=True)

    section("Honest scope")
    st.markdown("""
- **Replay Mode only.** Real packet capture via Scapy/Zeek/CICFlowMeter is provided as a stub and is not implemented in this build. Production deployment would require admin/npcap and is documented in the README.
- **Synthetic geographic attribution.** CICIDS2017 strips IPs. The `geo_enrich()` function uses deterministic-hashed origins from a weighted distribution matching real threat-intel reports. For production, swap to MaxMind GeoLite2.
- **Demo compliance coverage.** Framework controls are marked active when their evidence-providing attack classes are detected. This is a capability demonstration, not an audit.
- **Class imbalance defended operationally.** R-Unsure-001 routes low-confidence non-BENIGN predictions to analyst review rather than auto-blocking, defending Botnet's 0.37 precision and U2R's 0.12 F1.
- **R-Rate-001 gated.** Extreme-rate rule no longer fires on confident BENIGN flows (Codex bug fix v4) — prevents misleading "max_severity=High while attack_ratio=0%" reports.
""")

    section("Build")
    st.caption(f"AIES NIDS · CT-361 AIES CCP · NEDU BCIT · {datetime.now().strftime('%Y-%m-%d')}")


# ============================================================================
# TAB 9  ·  AIES THEORY  (forward chain, backward chain, CF, what-if, KB, etc.)
# ============================================================================
with tabs[9]:
    render_theory_tab(
        detector=detector,
        audit=audit,
        enriched=st.session_state.get("shared_enriched"),
        summary=st.session_state.get("last_summary"),
        plotly_clean=plotly_clean,
        kpi=kpi,
        section=section,
    )


# ============================================================================
# TAB 10  ·  SOC BRIEF  (executive brief, data quality, campaign, response)
# ============================================================================
with tabs[10]:
    section("SOC command brief")
    enr = st.session_state.get("shared_enriched")
    if enr is None or enr.empty:
        empty_state(
            "🧭",
            "No command brief yet.",
            "Run Mixed Attack Lab or a smart-balanced CSV analysis to generate an executive SOC brief, data quality score, attack surface, and remediation plan.",
            "Run analysis first",
        )
    else:
        brief = st.session_state.get("last_brief") or build_command_brief(
            enr,
            st.session_state.get("last_summary"),
            st.session_state.get("last_quality"),
            source_label=st.session_state.get("last_src_label", ""),
            sampling_mode=st.session_state.get("last_sampling_mode", ""),
            elapsed_sec=st.session_state.get("last_elapsed", 0.0),
        )
        quality = st.session_state.get("last_quality") or {}
        timeline = st.session_state.get("last_timeline")
        if timeline is None or getattr(timeline, "empty", True):
            timeline = build_campaign_timeline(enr)
        surface = st.session_state.get("last_attack_surface")
        if surface is None or getattr(surface, "empty", True):
            surface = build_attack_surface(enr)
        remediation = st.session_state.get("last_remediation")
        if remediation is None or getattr(remediation, "empty", True):
            remediation = build_remediation_plan(enr)
        case_board = st.session_state.get("last_case_board")
        if case_board is None or getattr(case_board, "empty", True):
            case_board = build_case_board(enr)

        st.markdown(
            f"""
            <div class="brief-panel">
              <div class="brief-status">{brief.get('status', 'UNKNOWN')}</div>
              <div class="brief-headline">{brief.get('headline', '')}</div>
              <div class="brief-meta">
                Source: {brief.get('source', '—')} · Sampling: {brief.get('sampling', '—')} ·
                Runtime: {brief.get('elapsed_sec', 0):.2f}s
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b1, b2, b3, b4, b5 = st.columns(5)
        with b1: kpi("Attack flows", f"{brief.get('attack_count', 0):,}", f"{brief.get('attack_ratio', 0):.1%}")
        with b2: kpi("Critical", f"{brief.get('critical', 0):,}", value_class="critical")
        with b3: kpi("Top class", brief.get("top_attack", "—"), f"{brief.get('top_attack_count', 0):,} flows")
        with b4: kpi("Top service", brief.get("top_service", "—"), "Most targeted")
        with b5: kpi("Quality", f"{quality.get('score', 0)}/100", quality.get("grade", "—"))

        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            section("Key findings")
            st.markdown(
                "<div class='brief-panel'><ul class='brief-list'>"
                + "".join(f"<li>{item}</li>" for item in brief.get("key_findings", []))
                + "</ul></div>",
                unsafe_allow_html=True,
            )
        with c2:
            section("Data quality")
            qscore = int(quality.get("score", 0))
            st.markdown(
                f"""
                <div class="quality-panel">
                  <div class="micro-label">{quality.get('grade', 'Unknown')}</div>
                  <div class="quality-bar"><div class="quality-fill" style="width:{qscore}%"></div></div>
                  <ul class="brief-list">
                    <li>{quality.get('rows', 0):,} rows · {quality.get('columns', 0):,} columns</li>
                    <li>{quality.get('missing_pct', 0):.2f}% missing · {quality.get('duplicate_pct', 0):.2f}% duplicate</li>
                    <li>{quality.get('distinct_labels', 0)} source labels visible</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for note in quality.get("notes", [])[:3]:
                st.caption(note)

        section("Campaign timeline")
        if not timeline.empty:
            timeline_fig = go.Figure()
            timeline_fig.add_trace(go.Bar(
                x=timeline["Window"],
                y=timeline["Events"],
                name="Events",
                marker=dict(color="#00E676", line=dict(color="#23313C", width=1)),
                text=timeline["Top class"],
                textposition="outside",
                hovertemplate="Window %{x}<br>Events %{y}<br>Top %{text}<extra></extra>",
            ))
            timeline_fig.update_layout(**plotly_clean(height=320, margin=dict(t=20, l=10, r=20, b=70)))
            st.plotly_chart(timeline_fig, width="stretch")
            st.dataframe(timeline, width="stretch", hide_index=True)

        s1, s2 = st.columns([1, 1])
        with s1:
            section("Attack surface")
            st.dataframe(surface, width="stretch", hide_index=True)
        with s2:
            section("Remediation plan")
            st.dataframe(remediation, width="stretch", hide_index=True)

        section("Case board")
        if case_board.empty:
            st.caption("No active non-BENIGN cases.")
        else:
            st.dataframe(case_board, width="stretch", hide_index=True, height=280)

        section("Analyst next actions")
        st.markdown(
            "<div class='action-panel'><ol class='brief-list'>"
            + "".join(f"<li>{action}</li>" for action in brief.get("next_actions", []))
            + "</ol></div>",
            unsafe_allow_html=True,
        )

        brief_md = build_brief_markdown(
            brief,
            quality,
            remediation,
            timeline=timeline,
            surface=surface,
            case_board=case_board,
            model_risk=st.session_state.get("last_model_risk"),
        )
        st.download_button(
            "Export Analyst Brief",
            data=brief_md,
            file_name=f"aies_nids_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            width="content",
        )
