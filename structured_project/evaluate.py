from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

ARTIFACTS_DIR = Path("artifacts")


def evaluate(
    model: Pipeline,
    test_features,
    test_labels_encoded: np.ndarray,
    encoder: LabelEncoder,
    tag: str = "",
) -> dict:
    """
    Evaluate a single model on the test set.

    Returns a dict with metrics, report text, and confusion-matrix image path.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    preds = model.predict(test_features)

    metrics = {
        "test_accuracy": accuracy_score(test_labels_encoded, preds),
        "test_f1_macro": f1_score(test_labels_encoded, preds, average="macro"),
        "test_precision_macro": precision_score(test_labels_encoded, preds, average="macro"),
        "test_recall_macro": recall_score(test_labels_encoded, preds, average="macro"),
    }

    report = classification_report(
        test_labels_encoded, preds, target_names=encoder.classes_,
    )

    # Confusion matrix image
    fig, ax = plt.subplots(figsize=(12, 10))
    cm = confusion_matrix(test_labels_encoded, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=encoder.classes_)
    disp.plot(ax=ax)
    plt.xticks(rotation=45, fontsize=12)
    plt.title(f"Confusion Matrix — {tag}" if tag else "Confusion Matrix")
    plt.tight_layout()
    cm_filename = f"confusion_matrix_{tag}.png" if tag else "confusion_matrix.png"
    cm_path = ARTIFACTS_DIR / cm_filename
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # Classification report text file
    report_filename = f"classification_report_{tag}.txt" if tag else "classification_report.txt"
    report_path = ARTIFACTS_DIR / report_filename
    report_path.write_text(report)

    return {
        "metrics": metrics,
        "report": report,
        "cm_path": str(cm_path),
        "report_path": str(report_path),
    }
