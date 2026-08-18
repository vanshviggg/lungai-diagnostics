import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from lungai.config import BATCH_SIZE, DISEASE_LABELS, METRICS_DIR, PLOTS_DIR
from lungai.data.dataset import ChestXrayDataset
from lungai.data.transforms import validation_transform
from lungai.models.chest_xray_model import create_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_thresholds(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text())
    raw_thresholds = payload.get("thresholds")
    if not isinstance(raw_thresholds, dict):
        raise ValueError("Threshold JSON must contain a 'thresholds' object.")

    missing = set(DISEASE_LABELS).difference(raw_thresholds)
    if missing:
        raise ValueError(f"Missing thresholds for labels: {sorted(missing)}")

    thresholds = {label: float(raw_thresholds[label]) for label in DISEASE_LABELS}
    invalid = {
        label: value
        for label, value in thresholds.items()
        if not 0.0 <= value <= 1.0
    }
    if invalid:
        raise ValueError(f"Thresholds must be between 0 and 1: {invalid}")
    return thresholds


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict[str, float],
) -> dict:
    results = {"thresholds": thresholds, "labels": {}}

    for index, label in enumerate(DISEASE_LABELS):
        true_col = y_true[:, index]
        prob_col = y_prob[:, index]
        pred_col = (prob_col >= thresholds[label]).astype(int)

        results["labels"][label] = {
            "threshold": thresholds[label],
            "precision": float(
                precision_score(true_col, pred_col, zero_division=0)
            ),
            "recall": float(recall_score(true_col, pred_col, zero_division=0)),
            "f1": float(f1_score(true_col, pred_col, zero_division=0)),
            "auroc": (
                float(roc_auc_score(true_col, prob_col))
                if len(np.unique(true_col)) > 1
                else None
            ),
            "support": int(true_col.sum()),
        }

    return results


def save_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict[str, float],
) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))
    for index, label in enumerate(DISEASE_LABELS):
        true_col = y_true[:, index]
        if len(np.unique(true_col)) < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            true_col, y_prob[:, index]
        )
        score = auc(false_positive_rate, true_positive_rate)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{label} (AUC={score:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Test ROC curves",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "roc_curves.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    for index, label in enumerate(DISEASE_LABELS):
        precision, recall, _ = precision_recall_curve(
            y_true[:, index], y_prob[:, index]
        )
        axis.plot(recall, precision, label=label)
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Test precision-recall curves",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "precision_recall_curves.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(DISEASE_LABELS), figsize=(13, 4))
    for index, (axis, label) in enumerate(zip(axes, DISEASE_LABELS)):
        predictions = (y_prob[:, index] >= thresholds[label]).astype(int)
        matrix = confusion_matrix(y_true[:, index], predictions, labels=[0, 1])
        ConfusionMatrixDisplay(
            matrix,
            display_labels=["Negative", "Positive"],
        ).plot(ax=axis, colorbar=False)
        axis.set_title(label)
    figure.suptitle("Test confusion matrices")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=160)
    plt.close(figure)


def evaluate(test_csv: str, checkpoint: str, thresholds_json: str) -> dict:
    device = get_device()
    thresholds = load_thresholds(thresholds_json)
    print(f"device={device}")
    print(
        "thresholds="
        + ", ".join(
            f"{label}:{thresholds[label]:.2f}" for label in DISEASE_LABELS
        )
    )

    dataset = ChestXrayDataset(test_csv, transform=validation_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(len(DISEASE_LABELS), pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()

    all_targets = []
    all_probs = []

    with torch.inference_mode():
        for images, labels in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())

    if not all_targets:
        raise ValueError("Test CSV contains no samples.")

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    results = compute_metrics(y_true, y_prob, thresholds)
    results.update(
        {
            "device": str(device),
            "test_csv": test_csv,
            "checkpoint": checkpoint,
            "thresholds_json": thresholds_json,
        }
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / "final_test_evaluation.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    save_plots(y_true, y_prob, thresholds)

    print(json.dumps(results, indent=2))
    print(f"saved={output_path}")
    print(f"plots={PLOTS_DIR}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the untouched test set with locked validation thresholds."
    )
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds-json", required=True)
    args = parser.parse_args()
    evaluate(args.test_csv, args.checkpoint, args.thresholds_json)
