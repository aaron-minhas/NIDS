"""
Multi-model training pipeline for NIDS.
Races RandomForest + GradientBoosting + LogisticBaseline; picks winner by macro-F1.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    prepare_data, fit_imputer_scaler, transform_with_pipeline,
    load_csv_robust, load_many,
)


@dataclass
class TrainingResult:
    model_name: str
    macro_f1: float
    accuracy: float
    classification_report: str
    confusion_matrix: np.ndarray
    per_class_metrics: pd.DataFrame
    label_classes: list
    feature_names: list
    feature_importance: pd.DataFrame | None
    train_time_sec: float


def build_candidate_models(random_state=42):
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=120, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.1, random_state=random_state,
        ),
        "LogisticBaseline": LogisticRegression(
            max_iter=300, class_weight="balanced", random_state=random_state,
        ),
    }


def train_and_evaluate(csv_paths, output_dir="models", random_state=42,
                       test_size=0.2, skip_models=(), max_rows=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading {len(csv_paths)} CSV(s)...")
    df = load_many(csv_paths) if len(csv_paths) > 1 else load_csv_robust(csv_paths[0])
    print(f"[train] raw shape: {df.shape}")

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        print(f"[train] subsampled to {len(df)} rows for speed")

    # === Codex Bug #3 fix: split BEFORE fitting imputer/scaler ===
    print("[train] preparing data (no leakage)...")
    X_df, y, le, feature_names = prepare_data(df)
    print(f"[train] processed shape: {X_df.shape}")
    print(f"[train] classes: {list(le.classes_)}")
    print(f"[train] class distribution: {dict(zip(le.classes_, np.bincount(y).tolist()))}")

    print("[train] train/test split...")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("[train] fitting imputer + scaler on TRAIN only (no leakage)...")
    imputer, scaler = fit_imputer_scaler(X_train_df)
    X_train = transform_with_pipeline(X_train_df, imputer, scaler)
    X_test = transform_with_pipeline(X_test_df, imputer, scaler)

    candidates = {n: m for n, m in build_candidate_models(random_state).items() if n not in skip_models}

    results = {}
    for name, model in candidates.items():
        print(f"[train] fitting {name}...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        y_pred = model.predict(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        accuracy = (y_pred == y_test).mean()
        cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
        report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

        prec, rec, f1, sup = precision_recall_fscore_support(
            y_test, y_pred, labels=range(len(le.classes_)), zero_division=0
        )
        pcm = pd.DataFrame({"class": le.classes_, "precision": prec, "recall": rec, "f1": f1, "support": sup})

        fi = None
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
            fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)

        print(f"[train]   {name}: macro-F1={macro_f1:.4f} acc={accuracy:.4f} ({elapsed:.1f}s)")

        results[name] = TrainingResult(
            model_name=name, macro_f1=macro_f1, accuracy=accuracy,
            classification_report=report, confusion_matrix=cm,
            per_class_metrics=pcm, label_classes=list(le.classes_),
            feature_names=feature_names, feature_importance=fi,
            train_time_sec=elapsed,
        )

    winner_name = max(results, key=lambda n: results[n].macro_f1)
    winner = results[winner_name]
    winning_model = candidates[winner_name]
    print(f"[train] winner: {winner_name} (macro-F1={winner.macro_f1:.4f})")

    bundle = {
        "model": winning_model, "model_name": winner_name,
        "imputer": imputer, "scaler": scaler, "label_encoder": le,
        "feature_names": feature_names,
        "training_macro_f1": winner.macro_f1,
        "training_accuracy": winner.accuracy,
        "label_classes": list(le.classes_),
    }
    pkl_path = output_dir / "nids_model.pkl"
    joblib.dump(bundle, pkl_path)
    print(f"[train] saved bundle to {pkl_path}")

    manifest = {
        "model_name": winner_name,
        "macro_f1": float(winner.macro_f1),
        "accuracy": float(winner.accuracy),
        "n_features": len(feature_names),
        "classes": list(le.classes_),
        "train_time_sec": winner.train_time_sec,
        "all_candidates": {n: {"macro_f1": float(r.macro_f1), "accuracy": float(r.accuracy)} for n, r in results.items()},
    }
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    return winner
