"""Model confidence and uncertainty analytics for AIES NIDS."""
from __future__ import annotations

from math import log
from typing import Any

import numpy as np
import pandas as pd


def _prob_values(prediction) -> list[float]:
    probs = getattr(prediction, "class_probabilities", {}) or {}
    vals = [float(v) for v in probs.values() if v is not None]
    vals = [v for v in vals if np.isfinite(v)]
    return vals


def normalized_entropy(prediction) -> float:
    """0 means confident single-class distribution; 1 means fully uncertain."""
    vals = np.array(_prob_values(prediction), dtype=float)
    vals = vals[vals > 0]
    if len(vals) <= 1:
        return 0.0
    vals = vals / vals.sum()
    return float(-np.sum(vals * np.log(vals)) / log(len(vals)))


def probability_margin(prediction) -> float:
    """Difference between top-1 and top-2 probability."""
    vals = sorted(_prob_values(prediction), reverse=True)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    return float(vals[0] - vals[1])


def uncertainty_rows(predictions: list, enriched: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-flow uncertainty facts used by the UI and review queue."""
    enrich_by_idx = {}
    if enriched is not None and not enriched.empty and "flow_idx" in enriched.columns:
        enrich_by_idx = {
            int(row["flow_idx"]): row.to_dict()
            for _, row in enriched.iterrows()
        }

    rows = []
    for pred in predictions or []:
        idx = int(getattr(pred, "flow_index", len(rows)))
        extra = enrich_by_idx.get(idx, {})
        conf = float(getattr(pred, "confidence", 0.0))
        entropy = normalized_entropy(pred)
        margin = probability_margin(pred)
        rows.append({
            "flow_idx": idx,
            "predicted_class": str(getattr(pred, "predicted_class", "Unknown")),
            "severity": str(extra.get("severity", "Informational")),
            "confidence": round(conf, 4),
            "entropy": round(entropy, 4),
            "margin": round(margin, 4),
            "review_reason": review_reason(conf, entropy, margin),
            "service": str(extra.get("service", "Unknown")),
            "mitre_id": str(extra.get("mitre_id", "—")),
        })
    return pd.DataFrame(rows)


def review_reason(confidence: float, entropy: float, margin: float) -> str:
    reasons = []
    if confidence < 0.65:
        reasons.append("low confidence")
    if entropy > 0.45:
        reasons.append("high entropy")
    if margin < 0.20:
        reasons.append("near class tie")
    return ", ".join(reasons) if reasons else "clear"


def model_risk_summary(predictions: list) -> dict[str, Any]:
    if not predictions:
        return {
            "status": "NO DATA",
            "median_confidence": 0.0,
            "p10_confidence": 0.0,
            "low_confidence": 0,
            "high_entropy": 0,
            "near_tie": 0,
            "review_rate": 0.0,
        }

    conf = np.array([float(getattr(p, "confidence", 0.0)) for p in predictions], dtype=float)
    entropy = np.array([normalized_entropy(p) for p in predictions], dtype=float)
    margin = np.array([probability_margin(p) for p in predictions], dtype=float)
    low_conf = int((conf < 0.65).sum())
    high_entropy = int((entropy > 0.45).sum())
    near_tie = int((margin < 0.20).sum())
    review_mask = (conf < 0.65) | (entropy > 0.45) | (margin < 0.20)
    review_rate = float(review_mask.mean()) if len(review_mask) else 0.0

    if review_rate >= 0.25 or low_conf >= 50:
        status = "ANALYST REVIEW"
    elif review_rate >= 0.08:
        status = "WATCH"
    else:
        status = "STABLE"

    return {
        "status": status,
        "median_confidence": round(float(np.median(conf)), 4),
        "p10_confidence": round(float(np.percentile(conf, 10)), 4),
        "low_confidence": low_conf,
        "high_entropy": high_entropy,
        "near_tie": near_tie,
        "review_rate": round(review_rate, 4),
    }


def class_risk_table(predictions: list) -> pd.DataFrame:
    rows = uncertainty_rows(predictions)
    if rows.empty:
        return pd.DataFrame(columns=[
            "Class", "Flows", "Median confidence", "P10 confidence",
            "Median entropy", "Review candidates",
        ])

    out = []
    for cls, part in rows.groupby("predicted_class", sort=False):
        out.append({
            "Class": cls,
            "Flows": int(len(part)),
            "Median confidence": round(float(part["confidence"].median()), 3),
            "P10 confidence": round(float(part["confidence"].quantile(0.10)), 3),
            "Median entropy": round(float(part["entropy"].median()), 3),
            "Review candidates": int((part["review_reason"] != "clear").sum()),
        })
    return pd.DataFrame(out).sort_values(["Review candidates", "Flows"], ascending=False)


def review_queue(predictions: list, enriched: pd.DataFrame | None = None, limit: int = 50) -> pd.DataFrame:
    rows = uncertainty_rows(predictions, enriched)
    if rows.empty:
        return rows
    rows = rows[rows["review_reason"] != "clear"].copy()
    if rows.empty:
        return rows
    severity_rank = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Informational": 0}
    rows["_severity_rank"] = rows["severity"].map(severity_rank).fillna(0)
    rows["_risk"] = (
        (1.0 - rows["confidence"].astype(float))
        + rows["entropy"].astype(float)
        + (1.0 - rows["margin"].astype(float))
        + rows["_severity_rank"] * 0.15
    )
    return (
        rows.sort_values("_risk", ascending=False)
        .drop(columns=["_risk", "_severity_rank"])
        .head(limit)
        .reset_index(drop=True)
    )
