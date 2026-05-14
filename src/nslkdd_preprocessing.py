"""
NSL-KDD preprocessing pipeline.

41 features + categorical encoding (protocol_type, service, flag) + 5-class folding:
Normal / DoS / Probe / R2L / U2R.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

NSLKDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","attack","level",
]

CATEGORICAL_FEATURES = ["protocol_type","service","flag"]

# Map fine-grained attack name -> coarse 5-class category
ATTACK_FAMILY = {
    "normal": "Normal",
    # DoS
    "back":"DoS","land":"DoS","neptune":"DoS","pod":"DoS","smurf":"DoS",
    "teardrop":"DoS","mailbomb":"DoS","apache2":"DoS","processtable":"DoS",
    "udpstorm":"DoS","worm":"DoS",
    # Probe
    "ipsweep":"Probe","nmap":"Probe","portsweep":"Probe","satan":"Probe",
    "mscan":"Probe","saint":"Probe",
    # R2L (Remote-to-Local)
    "ftp_write":"R2L","guess_passwd":"R2L","imap":"R2L","multihop":"R2L",
    "phf":"R2L","spy":"R2L","warezclient":"R2L","warezmaster":"R2L",
    "sendmail":"R2L","named":"R2L","snmpgetattack":"R2L","snmpguess":"R2L",
    "xlock":"R2L","xsnoop":"R2L",
    # U2R (User-to-Root)
    "buffer_overflow":"U2R","loadmodule":"U2R","perl":"U2R","rootkit":"U2R",
    "httptunnel":"U2R","ps":"U2R","sqlattack":"U2R","xterm":"U2R",
}

NSLKDD_CLASSES = ["Normal","DoS","Probe","R2L","U2R"]


def map_attack(label):
    if label is None: return "Normal"
    s = str(label).strip().lower()
    return ATTACK_FAMILY.get(s, "Normal")


def load_nslkdd(path):
    """Load a KDDTrain+.txt or KDDTest+.txt file."""
    df = pd.read_csv(path, names=NSLKDD_COLUMNS, header=None, low_memory=False)
    return df


def fit_nslkdd_preprocessing(df, label_col="attack"):
    """Fit categorical encoders + scaler + label encoder on training data."""
    df = df.copy()
    # Drop the difficulty level column - it's metadata, not a feature
    if "level" in df.columns:
        df = df.drop(columns=["level"])

    # Map attack types -> 5-class
    df[label_col] = df[label_col].map(map_attack)
    y_raw = df[label_col].astype(str).values
    X_df = df.drop(columns=[label_col])

    # Categorical encoders (one per categorical column, fitted on train)
    cat_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in X_df.columns:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            cat_encoders[col] = le

    feature_names = list(X_df.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    label_le = LabelEncoder()
    y_encoded = label_le.fit_transform(y_raw)

    return X_scaled, y_encoded, cat_encoders, scaler, label_le, feature_names


def transform_nslkdd_inference(df, feature_names, cat_encoders, scaler, label_col="attack"):
    """Apply saved preprocessing to new NSL-KDD data."""
    df = df.copy()
    if "level" in df.columns:
        df = df.drop(columns=["level"])
    if label_col in df.columns:
        df = df.drop(columns=[label_col])

    # Apply categorical encoders, mapping unknowns to most-common (-1 then clamp to 0)
    for col, le in cat_encoders.items():
        if col in df.columns:
            known = set(le.classes_.tolist())
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))

    # Reorder columns to match training
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0
    aligned = df[feature_names]
    X_scaled = scaler.transform(aligned.values)
    return X_scaled, aligned
