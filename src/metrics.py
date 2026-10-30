"""Classification metrics. Macro-F1 is the main metric used by HASOC."""
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(labels, preds):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_hof": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall_hof": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "f1_hof": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        # rows = true label, columns = predicted label, order [NOT, HOF]
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }
