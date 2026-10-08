from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


VERDICT_CLASSES = [
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "CONTRADICTED",
    "OVERSTATED",
    "INSUFFICIENT_EVIDENCE",
]


def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Computes standard multi-class evaluation metrics using scikit-learn.
    Returns accuracy, macro/weighted precision, recall, f1, per-class F1, and confusion matrix.
    """
    labels = VERDICT_CLASSES

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    # Per-class metrics
    per_class_p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1_score": float(per_class_f1[i]),
            "support": int(sum(1 for y in y_true if y == label)),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels,
    }


def compute_system_metrics(
    latencies: List[float],
    api_calls_count: int,
    tokens_per_claim: List[int],
    retrieval_precisions: List[float],
    json_parse_successes: int,
    total_json_attempts: int,
) -> Dict[str, Any]:
    """
    Computes system performance metrics specified in Section 27:
    - average processing time
    - number of API calls
    - average tokens per claim
    - retrieval latency / precision
    - JSON parsing success rate
    """
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    avg_tokens = float(np.mean(tokens_per_claim)) if tokens_per_claim else 0.0
    avg_retrieval_prec = float(np.mean(retrieval_precisions)) if retrieval_precisions else 0.0
    
    json_rate = (
        float(json_parse_successes / max(1, total_json_attempts))
        if total_json_attempts > 0
        else 1.0
    )

    return {
        "average_latency_sec": round(avg_latency, 3),
        "p95_latency_sec": round(p95_latency, 3),
        "total_api_calls": api_calls_count,
        "avg_tokens_per_claim": round(avg_tokens, 1),
        "retrieval_precision": round(avg_retrieval_prec, 3),
        "json_parsing_success_rate": round(json_rate, 4),
    }
