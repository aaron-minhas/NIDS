"""
End-to-end NSL-KDD trainer: load -> fit -> race RF/LR -> save bundle + eval artifacts.

Standalone trainer (not refactored to share with CICIDS pipeline because feature
schemas differ -- NSL-KDD has 41 features w/ 3 categorical, CICIDS has 78 numeric).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from src.nslkdd_preprocessing import (
    NSLKDD_COLUMNS, NSLKDD_CLASSES,
    load_nslkdd, fit_nslkdd_preprocessing,
)
from src.training import TrainingResult


def train_nslkdd(train_csv, test_csv=None, output_dir="models", random_state=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[nslkdd] loading train...")
    df_train = load_nslkdd(train_csv)
    print(f"[nslkdd] train shape: {df_train.shape}")

    print("[nslkdd] fitting preprocessing...")
    X_tr, y_tr, cat_encoders, scaler, le, feature_names = fit_nslkdd_preprocessing(df_train)
    print(f"[nslkdd] processed shape: {X_tr.shape}")
    print(f"[nslkdd] classes: {list(le.classes_)}")
    print(f"[nslkdd] class distribution: {dict(zip(le.classes_, np.bincount(y_tr).tolist()))}")

    # Use NSL-KDD official test set if provided (better generalisation evaluation)
    # else stratified 80/20 split.
    if test_csv is not None and Path(test_csv).exists():
        print("[nslkdd] using OFFICIAL KDDTest+ for evaluation")
        df_test = load_nslkdd(test_csv)
        # Apply same preprocessing
        from src.nslkdd_preprocessing import transform_nslkdd_inference, map_attack
        df_test = df_test.copy()
        if "level" in df_test.columns:
            df_test = df_test.drop(columns=["level"])
        y_test_raw = df_test["attack"].map(map_attack).astype(str).values
        # Filter test labels to only known classes; else map to Normal
        known = set(le.classes_)
        y_test_raw = np.array(["Normal" if y not in known else y for y in y_test_raw])
        y_test = le.transform(y_test_raw)
        X_test, _ = transform_nslkdd_inference(df_test.drop(columns=["attack"]), feature_names, cat_encoders, scaler, label_col="__nope__")
        X_train, y_train = X_tr, y_tr
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=random_state, stratify=y_tr
        )

    # Race RF + LR
    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=120, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        ),
        "LogisticBaseline": LogisticRegression(
            max_iter=300, class_weight="balanced", random_state=random_state,
        ),
    }

    results = {}
    for name, model in candidates.items():
        print(f"[nslkdd] fitting {name}...")
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
        print(f"[nslkdd]   {name}: macro-F1={macro_f1:.4f} acc={accuracy:.4f} ({elapsed:.1f}s)")
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
    print(f"[nslkdd] winner: {winner_name} (macro-F1={winner.macro_f1:.4f})")

    bundle = {
        "model": winning_model, "model_name": winner_name,
        "cat_encoders": cat_encoders, "scaler": scaler, "label_encoder": le,
        "feature_names": feature_names,
        "training_macro_f1": winner.macro_f1,
        "training_accuracy": winner.accuracy,
        "label_classes": list(le.classes_),
        "dataset": "NSL-KDD",
    }
    pkl_path = output_dir / "nslkdd_model.pkl"
    joblib.dump(bundle, pkl_path)
    print(f"[nslkdd] saved bundle to {pkl_path}")

    manifest = {
        "model_name": winner_name,
        "macro_f1": float(winner.macro_f1),
        "accuracy": float(winner.accuracy),
        "n_features": len(feature_names),
        "classes": list(le.classes_),
        "train_time_sec": winner.train_time_sec,
        "dataset": "NSL-KDD",
        "all_candidates": {n: {"macro_f1": float(r.macro_f1), "accuracy": float(r.accuracy)} for n, r in results.items()},
    }
    (output_dir / "nslkdd_manifest.json").write_text(json.dumps(manifest, indent=2))
    return winner
