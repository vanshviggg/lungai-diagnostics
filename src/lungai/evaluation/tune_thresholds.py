import argparse
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from lungai.config import BATCH_SIZE, DISEASE_LABELS, METRICS_DIR
from lungai.data.dataset import ChestXrayDataset
from lungai.data.transforms import validation_transform
from lungai.models.chest_xray_model import create_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    scores = np.array(
        [
            f1_score(
                y_true,
                (y_prob >= threshold).astype(int),
                zero_division=0,
            )
            for threshold in thresholds
        ]
    )
    return float(thresholds[int(np.argmax(scores))])


def tune_thresholds(
    val_csv: str,
    checkpoint: str,
    grid_start: float = 0.05,
    grid_end: float = 0.95,
    grid_step: float = 0.01,
) -> dict:
    if not 0.0 <= grid_start <= grid_end <= 1.0:
        raise ValueError("Threshold grid must satisfy 0 <= start <= end <= 1.")
    if grid_step <= 0:
        raise ValueError("Threshold grid step must be positive.")

    device = get_device()
    print(f"device={device}")

    dataset = ChestXrayDataset(val_csv, transform=validation_transform)
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
        raise ValueError("Validation CSV contains no samples.")

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    thresholds = np.arange(
        grid_start,
        grid_end + grid_step / 2.0,
        grid_step,
        dtype=float,
    )
    thresholds = thresholds[thresholds <= grid_end + 1e-12]

    results = {
        "device": str(device),
        "validation_csv": val_csv,
        "checkpoint": checkpoint,
        "grid": {
            "start": grid_start,
            "end": grid_end,
            "step": grid_step,
        },
        "thresholds": {},
        "labels": {},
    }

    for index, label in enumerate(DISEASE_LABELS):
        true_col = y_true[:, index]
        prob_col = y_prob[:, index]

        best_threshold = select_best_threshold(true_col, prob_col, thresholds)
        pred_col = (prob_col >= best_threshold).astype(int)

        metrics = {
            "threshold": best_threshold,
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
        }
        results["thresholds"][label] = best_threshold
        results["labels"][label] = metrics

        auroc = "n/a" if metrics["auroc"] is None else f'{metrics["auroc"]:.4f}'
        print(
            f"{label}: threshold={best_threshold:.2f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} auroc={auroc}"
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / "thresholds.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved={output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune per-label classification thresholds on validation data."
    )
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grid-start", type=float, default=0.05)
    parser.add_argument("--grid-end", type=float, default=0.95)
    parser.add_argument("--grid-step", type=float, default=0.01)
    args = parser.parse_args()

    tune_thresholds(
        args.val_csv,
        args.checkpoint,
        grid_start=args.grid_start,
        grid_end=args.grid_end,
        grid_step=args.grid_step,
    )
