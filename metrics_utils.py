
from __future__ import annotations

from typing import List, Dict, Any
import numpy as np


def compute_confusion_metrics(cm: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    cm = np.asarray(cm)
    num_classes = cm.shape[0]
    per_class = []
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total else 0.0

    precisions = []
    recalls = []
    f1s = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class.append({
            "label": labels[i],
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        })

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


__all__ = ["compute_confusion_metrics"]
