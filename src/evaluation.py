"""
Evaluation artifact generator: confusion matrices, per-class metrics, feature importances.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.training import TrainingResult


def save_confusion_matrix(result, out_path, normalize=True):
    out_path = Path(out_path)
    cm = result.confusion_matrix
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=result.label_classes,
                yticklabels=result.label_classes, ax=ax,
                cbar_kws={"label": "Proportion" if normalize else "Count"})
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    title = "Confusion Matrix" + (" (row-normalised)" if normalize else "")
    ax.set_title(f"{title} -- {result.model_name} (macro-F1={result.macro_f1:.3f})")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def save_feature_importance(result, out_path, top_n=20):
    out_path = Path(out_path)
    if result.feature_importance is None or result.feature_importance.empty:
        return None
    top = result.feature_importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.3)))
    ax.barh(top["feature"], top["importance"], color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} feature importances -- {result.model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def save_per_class_metrics_chart(result, out_path):
    out_path = Path(out_path)
    pcm = result.per_class_metrics.copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pcm))
    w = 0.27
    ax.bar(x - w, pcm["precision"], w, label="Precision", color="#2563eb")
    ax.bar(x,     pcm["recall"],    w, label="Recall",    color="#16a34a")
    ax.bar(x + w, pcm["f1"],        w, label="F1",        color="#db2777")
    ax.set_xticks(x)
    ax.set_xticklabels(pcm["class"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-class metrics -- {result.model_name}")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def save_text_report(result, out_path):
    out_path = Path(out_path)
    lines = [
        "NIDS Model Evaluation Report",
        "=" * 48,
        f"Model: {result.model_name}",
        f"Macro-F1: {result.macro_f1:.4f}",
        f"Accuracy: {result.accuracy:.4f}",
        f"Train time: {result.train_time_sec:.1f}s",
        f"Classes: {', '.join(result.label_classes)}",
        f"Feature count: {len(result.feature_names)}",
        "",
        "Per-class classification report:",
        result.classification_report,
    ]
    out_path.write_text("\n".join(lines))
    return out_path


def generate_all(result, reports_dir="reports"):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "confusion_matrix": save_confusion_matrix(result, reports_dir / "confusion_matrix.png"),
        "confusion_matrix_counts": save_confusion_matrix(result, reports_dir / "confusion_matrix_counts.png", normalize=False),
        "per_class_metrics_chart": save_per_class_metrics_chart(result, reports_dir / "per_class_metrics.png"),
        "feature_importance": save_feature_importance(result, reports_dir / "feature_importance.png"),
        "text_report": save_text_report(result, reports_dir / "classification_report.txt"),
    }
    result.per_class_metrics.to_csv(reports_dir / "per_class_metrics.csv", index=False)
    out["per_class_metrics_csv"] = reports_dir / "per_class_metrics.csv"
    if result.feature_importance is not None:
        result.feature_importance.to_csv(reports_dir / "feature_importance.csv", index=False)
        out["feature_importance_csv"] = reports_dir / "feature_importance.csv"
    return out
