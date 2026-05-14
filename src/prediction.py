"""
Inference wrapper for the trained NIDS bundle.
Returns predicted class + confidence + advisories per flow.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.expert_system import Advisory, ExpertAdvisorySystem, Fact
from src.preprocessing import transform_inference


@dataclass
class FlowPrediction:
    flow_index: int
    predicted_class: str
    confidence: float
    class_probabilities: dict
    advisories: list


class NIDSDetector:
    def __init__(self, model_path="models/nids_model.pkl"):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.imputer = bundle["imputer"]
        self.scaler = bundle["scaler"]
        self.le = bundle["label_encoder"]
        self.feature_names = bundle["feature_names"]
        self.label_classes = bundle["label_classes"]
        self.model_name = bundle.get("model_name", "Unknown")
        self.training_macro_f1 = bundle.get("training_macro_f1", 0.0)
        self.training_accuracy = bundle.get("training_accuracy", 0.0)
        self._expert = ExpertAdvisorySystem()

    def predict_dataframe(self, df):
        X_scaled, aligned = transform_inference(df, self.feature_names, self.imputer, self.scaler)
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X_scaled)
            preds_idx = probas.argmax(axis=1)
            confs = probas.max(axis=1)
        else:
            preds_idx = self.model.predict(X_scaled)
            probas = np.zeros((len(preds_idx), len(self.label_classes)))
            confs = np.ones(len(preds_idx))

        labels = self.le.inverse_transform(preds_idx)

        # OPTIMIZATION: convert pandas to numpy ONCE (10-30x faster than per-row .iloc)
        feature_array = aligned.values
        feature_names = self.feature_names
        n_classes = len(self.label_classes)
        label_classes = self.label_classes
        n = len(preds_idx)

        # Pre-built static BENIGN advisory (avoids 10-rule eval for ~80% of rows)
        from src.expert_system import Advisory as _Adv
        results = []
        for i in range(n):
            label = str(labels[i])
            conf = float(confs[i])
            prob_row = probas[i]
            class_probs = {label_classes[j]: float(prob_row[j]) for j in range(n_classes)}

            if label == "BENIGN" and conf >= 0.55:
                # Fast path: skip features-dict construction + rule engine entirely
                advisories = [_Adv(
                    rule_id="R-Benign-001",
                    title="No threat indicators",
                    severity="Informational",
                    description="Flow features are consistent with normal traffic.",
                    recommended_actions=["No action required. Continue monitoring."],
                    confidence=conf,
                    because=f"Classifier confidence in BENIGN = {conf:.2f}",
                )]
            else:
                # Slow path (only for non-BENIGN or low-confidence): build feature dict from numpy row
                features_dict = dict(zip(feature_names, feature_array[i]))
                fact = Fact(predicted_class=label, confidence=conf, features=features_dict)
                advisories = self._expert.evaluate(fact)

            results.append(FlowPrediction(
                flow_index=i, predicted_class=label, confidence=conf,
                class_probabilities=class_probs, advisories=advisories,
            ))
        return results

    def summary(self, predictions):
        if not predictions:
            return {"total_flows": 0, "class_counts": {}, "attack_ratio": 0.0, "max_severity": "Informational"}
        counts = pd.Series([p.predicted_class for p in predictions]).value_counts().to_dict()
        attack_count = sum(c for k, c in counts.items() if k != "BENIGN")
        all_advisories = [a for p in predictions for a in p.advisories]
        max_sev_rank = max((a.severity_rank for a in all_advisories), default=0)
        max_sev = next((s for s, r in [("Critical",4),("High",3),("Medium",2),("Low",1),("Informational",0)] if r == max_sev_rank), "Informational")
        return {
            "total_flows": len(predictions),
            "class_counts": counts,
            "attack_ratio": attack_count / len(predictions),
            "max_severity": max_sev,
            "rules_fired": len(all_advisories),
        }

    def to_table(self, predictions):
        rows = []
        for p in predictions:
            top = p.advisories[0] if p.advisories else None
            rows.append({
                "flow": p.flow_index,
                "predicted_class": p.predicted_class,
                "confidence": round(p.confidence, 3),
                "top_advisory": top.title if top else "--",
                "severity": top.severity if top else "Informational",
                "rules_fired": len(p.advisories),
            })
        return pd.DataFrame(rows)
