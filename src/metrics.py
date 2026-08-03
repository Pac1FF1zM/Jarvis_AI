"""Classification metrics shared by training and final evaluation."""
from __future__ import annotations

from typing import Any, Iterable

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.data.audit import EXPECTED_LABELS


def classification_metrics(target: Iterable[int], predicted: Iterable[int]) -> dict[str, Any]:
    target_list = list(target)
    predicted_list = list(predicted)
    if not target_list:
        raise ValueError("Cannot calculate metrics for an empty target")
    labels = list(range(len(EXPECTED_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        target_list,
        predicted_list,
        labels=labels,
        zero_division=0,
    )
    total = max(sum(support), 1)
    return {
        "accuracy": float(accuracy_score(target_list, predicted_list)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(sum(value * count for value, count in zip(f1, support)) / total),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(EXPECTED_LABELS)
        },
        "confusion_matrix": confusion_matrix(
            target_list, predicted_list, labels=labels
        ).tolist(),
    }
