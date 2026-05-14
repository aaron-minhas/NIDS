"""
Preprocessing pipeline for CICIDS2017 network flow data.

Fixes from v1:
- Replaces dropna() with median imputation (preserves rows with single-feature anomalies).
- Removes zero-variance and quasi-constant columns (noise filter).
- Handles inf/-inf properly before scaling.
- Builds a multi-class label encoder (not binary DDoS/not-DDoS).
- Maps fine-grained CICIDS labels to coarse attack families.
- Strict train-only fit for scaler & imputer to prevent leakage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# CICIDS2017 raw labels -> coarse attack families
LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Bot": "Botnet",
    "Web Attack \u0096 Brute Force": "WebAttack",
    "Web Attack \u0096 XSS": "WebAttack",
    "Web Attack \u0096 Sql Injection": "WebAttack",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Infiltration": "Infiltration",
}

CANONICAL_CLASSES = ["BENIGN","DoS","DDoS","PortScan","BruteForce","Botnet","WebAttack","Infiltration"]


def normalize_label(raw):
    if raw is None: return "BENIGN"
    s = str(raw).strip()
    if s in LABEL_MAP: return LABEL_MAP[s]
    sl = s.lower()
    if sl.startswith("web attack"): return "WebAttack"
    if "ddos" in sl: return "DDoS"
    if "dos" in sl: return "DoS"
    if "scan" in sl: return "PortScan"
    if "patator" in sl or "brute" in sl: return "BruteForce"
    if "bot" in sl: return "Botnet"
    if "infiltrat" in sl: return "Infiltration"
    if "heartbleed" in sl: return "DoS"
    return "BENIGN"


def clean_columns(df):
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def coerce_numeric(df, label_col="Label"):
    for col in df.columns:
        if col == label_col: continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def replace_infinities(df):
    return df.replace([np.inf, -np.inf], np.nan)


def fit_preprocessing(df, label_col="Label", drop_constant=True):
    df = clean_columns(df).copy()
    df = coerce_numeric(df, label_col=label_col)
    df = replace_infinities(df)
    df[label_col] = df[label_col].map(normalize_label)

    y_raw = df[label_col].astype(str).values
    X_df = df.drop(columns=[label_col])

    if drop_constant:
        keep = []
        for col in X_df.columns:
            non_null = X_df[col].dropna()
            if len(non_null) == 0: continue
            if non_null.nunique() <= 1: continue
            keep.append(col)
        X_df = X_df[keep]

    feature_names = list(X_df.columns)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    return X_scaled, y_encoded, imputer, scaler, le, feature_names


# ============================================================================
# SPLIT-SAFE PREPROCESSING  (Codex Bug #3 fix)
#
# Old fit_preprocessing() above fits imputer + scaler on the FULL dataset before
# train/test split, which leaks test-set statistics into the imputation median
# and the scaler standardisation. The functions below split the work:
#
#   1. prepare_data()       - non-leaky pre-split work (clean, drop constants,
#                              encode labels). Returns X_df + y, no fitted models.
#   2. fit_imputer_scaler()  - TRAIN-ONLY fit. Pass the train half of X_df only.
#   3. transform_with_pipeline() - apply fitted imputer + scaler to any X_df.
#
# Recommended usage in training.py:
#     X_df, y, le, feature_names = prepare_data(df)
#     X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, ...)
#     imputer, scaler = fit_imputer_scaler(X_train_df)
#     X_train = transform_with_pipeline(X_train_df, imputer, scaler)
#     X_test  = transform_with_pipeline(X_test_df,  imputer, scaler)
# ============================================================================
def prepare_data(df, label_col="Label", drop_constant=True):
    """Pre-split cleanup -- no fitted statistics yet. Safe to call on full data."""
    df = clean_columns(df).copy()
    df = coerce_numeric(df, label_col=label_col)
    df = replace_infinities(df)
    df[label_col] = df[label_col].map(normalize_label)

    y_raw = df[label_col].astype(str).values
    X_df = df.drop(columns=[label_col])

    if drop_constant:
        keep = []
        for col in X_df.columns:
            non_null = X_df[col].dropna()
            if len(non_null) == 0: continue
            if non_null.nunique() <= 1: continue
            keep.append(col)
        X_df = X_df[keep]

    feature_names = list(X_df.columns)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    return X_df, y_encoded, le, feature_names


def fit_imputer_scaler(X_train_df):
    """TRAIN-ONLY fit. Pass the train half of prepare_data() output."""
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train_df)
    scaler = StandardScaler()
    scaler.fit(X_imputed)
    return imputer, scaler


def transform_with_pipeline(X_df, imputer, scaler):
    """Apply the train-fitted imputer + scaler to any X_df (train or test)."""
    X_imputed = imputer.transform(X_df)
    return scaler.transform(X_imputed)


def transform_inference(df, feature_names, imputer, scaler, label_col="Label"):
    df = clean_columns(df).copy()
    if label_col and label_col in df.columns:
        df = df.drop(columns=[label_col])
    df = coerce_numeric(df, label_col="__nope__")
    df = replace_infinities(df)
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    aligned = df[list(feature_names)]
    X_imputed = imputer.transform(aligned)
    X_scaled = scaler.transform(X_imputed)
    return X_scaled, aligned


def load_csv_robust(path, label_col="Label"):
    path = Path(path)
    for enc in ("utf-8","latin-1","cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            return clean_columns(df)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


def load_many(paths, label_col="Label"):
    frames = [load_csv_robust(p, label_col=label_col) for p in paths]
    if len(frames) == 1: return frames[0]
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    common_cols = [c for c in frames[0].columns if c in common]
    return pd.concat([f[common_cols] for f in frames], ignore_index=True)
