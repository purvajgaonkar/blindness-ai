"""
utils.py
========
Shared utility functions for the DR Detection project:
  - Quadratic Weighted Kappa (QWK) — the APTOS competition metric
  - Confusion matrix plotting
  - Training curve plotting
  - ROC curve plotting
  - Classification report saving
  - Image-to-base64 encoding for Flask API
  - Logging helpers
"""

import os
import json
import io
import base64
import logging
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, cohen_kappa_score
)
from PIL import Image as PILImage

# ─── Logger setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DR_Detection")

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


# ─── Metrics ─────────────────────────────────────────────────────────────────

def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray,
                              num_classes: int = 5) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK) — the official APTOS 2019 metric.

    QWK penalises predictions proportionally to the squared distance between
    predicted and true class, making it sensitive to the ordinal nature of DR grades.

    Parameters
    ----------
    y_true      : np.ndarray  Ground-truth class indices
    y_pred      : np.ndarray  Predicted class indices
    num_classes : int         Number of classes (5 for APTOS)

    Returns
    -------
    float  QWK score in [-1, 1]; 1.0 = perfect agreement
    """
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    except Exception as exc:
        logger.warning(f"QWK computation failed: {exc}")
        return 0.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None) -> dict:
    """
    Compute all evaluation metrics in one call.

    Parameters
    ----------
    y_true : np.ndarray  Ground-truth labels
    y_pred : np.ndarray  Predicted labels
    y_prob : np.ndarray  Predicted class probabilities (N, C), optional

    Returns
    -------
    dict with keys: accuracy, qwk, per_class_accuracy, report_text
    """
    accuracy = float(np.mean(y_true == y_pred))
    qwk      = quadratic_weighted_kappa(y_true, y_pred)
    report   = classification_report(y_true, y_pred,
                                     target_names=CLASS_NAMES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)

    return {
        'accuracy':         accuracy,
        'qwk':              qwk,
        'per_class_accuracy': per_class_acc.tolist(),
        'report_text':      report,
        'confusion_matrix': cm.tolist(),
    }


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
):
    """
    Plot and save a styled confusion matrix heatmap.

    Parameters
    ----------
    y_true    : np.ndarray  Ground-truth labels
    y_pred    : np.ndarray  Predicted labels
    save_path : str         Output PNG path
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              facecolor='#0a0e1a')
    plt.rcParams.update({'text.color': 'white'})

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw Counts", "Normalised (Recall)"],
        ['d', '.2f']
    ):
        ax.set_facecolor('#0a0e1a')
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap='YlOrRd',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor='#1a1e2e',
            cbar_kws={'shrink': 0.8}, ax=ax
        )
        ax.set_title(f'Confusion Matrix — {title}', color='white',
                     fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted Class', color='#00d4ff', fontsize=11)
        ax.set_ylabel('True Class',      color='#00d4ff', fontsize=11)
        ax.tick_params(colors='white')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', color='white')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color='white')

    plt.suptitle('DR Classification — Confusion Matrix', color='white',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0e1a', edgecolor='none')
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def plot_training_curves(history: dict, save_path: str):
    """
    Plot training and validation loss / accuracy / QWK curves.

    Parameters
    ----------
    history   : dict  Keys: 'train_loss', 'val_loss', 'val_accuracy', 'val_qwk'
                      (each is a list of per-epoch values)
    save_path : str   Output PNG path
    """
    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0a0e1a')
    plt.rcParams.update({'text.color': 'white'})

    metrics = [
        ('train_loss', 'val_loss',     'Loss',         '#ff6b6b',  '#00d4ff'),
        ('val_accuracy', None,          'Val Accuracy', '#a78bfa',  None),
        ('val_qwk',      None,          'Val QWK',      '#34d399',  None),
    ]

    for ax, (m1, m2, title, c1, c2) in zip(axes, metrics):
        ax.set_facecolor('#0d1117')
        ax.grid(color='#1f2937', linestyle='--', linewidth=0.7)
        ax.spines[:].set_color('#1f2937')
        ax.tick_params(colors='white')

        if m1 in history:
            ax.plot(epochs, history[m1], color=c1, linewidth=2.2,
                    label=m1.replace('_', ' ').title(), marker='o', markersize=3)
        if m2 and m2 in history:
            ax.plot(epochs, history[m2], color=c2, linewidth=2.2,
                    label=m2.replace('_', ' ').title(), marker='s', markersize=3)

        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', color='#9ca3af')
        ax.set_ylabel(title, color='#9ca3af')
        legend = ax.legend(facecolor='#1f2937', edgecolor='#374151', labelcolor='white')

    plt.suptitle('Training Progress', color='white', fontsize=15, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0e1a', edgecolor='none')
    plt.close()
    logger.info(f"Training curves saved: {save_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    num_classes: int = 5,
):
    """
    Plot one-vs-rest ROC curves for each DR class.

    Parameters
    ----------
    y_true      : np.ndarray  (N,) integer ground-truth labels
    y_prob      : np.ndarray  (N, C) predicted probabilities
    save_path   : str         Output PNG path
    num_classes : int         Number of classes
    """
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0a0e1a')
    ax.set_facecolor('#0d1117')
    ax.grid(color='#1f2937', linestyle='--', linewidth=0.7)
    ax.spines[:].set_color('#1f2937')
    ax.tick_params(colors='white')

    colors_roc = ['#00d4ff', '#7c3aed', '#f59e0b', '#ef4444', '#10b981']

    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors_roc)):
        if y_bin.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc      = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], 'w--', linewidth=1, alpha=0.4, label='Random')
    ax.set_title('ROC Curves — One-vs-Rest per DR Grade',
                 color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('False Positive Rate', color='#9ca3af')
    ax.set_ylabel('True Positive Rate',  color='#9ca3af')
    ax.legend(facecolor='#1f2937', edgecolor='#374151', labelcolor='white')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0e1a', edgecolor='none')
    plt.close()
    logger.info(f"ROC curves saved: {save_path}")


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    extra_metrics: dict = None,
):
    """
    Save a textual classification report to disk.

    Parameters
    ----------
    y_true        : np.ndarray  Ground-truth labels
    y_pred        : np.ndarray  Predicted labels
    save_path     : str         Output .txt file path
    extra_metrics : dict        Additional metrics to append (e.g. QWK, accuracy)
    """
    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_NAMES, zero_division=0)
    lines  = [
        "=" * 65,
        " Diabetic Retinopathy Classification Report",
        "=" * 65,
        "",
        report,
        "",
    ]
    if extra_metrics:
        lines += ["─" * 65, " Additional Metrics", "─" * 65]
        for k, v in extra_metrics.items():
            lines.append(f"  {k:30s}: {v}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Classification report saved: {save_path}")


# ─── Image encoding ───────────────────────────────────────────────────────────

def numpy_to_base64(img: np.ndarray, fmt: str = 'PNG') -> str:
    """
    Encode a numpy uint8 RGB image to a base64 string for JSON transport.

    Parameters
    ----------
    img : np.ndarray  uint8 RGB image (H, W, 3) or greyscale (H, W)
    fmt : str         Image format ('PNG', 'JPEG', etc.)

    Returns
    -------
    str  Base64-encoded image string (no header)
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        pil_img = PILImage.fromarray(img, mode='L')
    else:
        pil_img = PILImage.fromarray(img, mode='RGB')
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def file_to_base64(path: str) -> str:
    """
    Read an image file from disk and encode it as base64.

    Parameters
    ----------
    path : str   Path to image file

    Returns
    -------
    str  Base64-encoded image string
    """
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ─── History persistence ──────────────────────────────────────────────────────

def save_history(history: dict, path: str):
    """Save training history dict to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved: {path}")


def load_history(path: str) -> dict:
    """Load training history from JSON."""
    with open(path, 'r') as f:
        return json.load(f)
