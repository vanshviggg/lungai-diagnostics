import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from lungai.config import BATCH_SIZE, DISEASE_LABELS, METRICS_DIR
from lungai.data.dataset import ChestXrayDataset
from lungai.data.transforms import validation_transform
from lungai.models.chest_xray_model import create_model


def evaluate(test_csv: str, checkpoint: str, threshold: float = 0.5) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChestXrayDataset(test_csv, transform=validation_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(len(DISEASE_LABELS), pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    all_targets = []
    all_probs = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    results = {"threshold": threshold, "labels": {}}

    for index, label in enumerate(DISEASE_LABELS):
        true_col = y_true[:, index]
        prob_col = y_prob[:, index]
        pred_col = y_pred[:, index]

        metrics = {
            "precision": float(precision_score(true_col, pred_col, zero_division=0)),
            "recall": float(recall_score(true_col, pred_col, zero_division=0)),
            "f1": float(f1_score(true_col, pred_col, zero_division=0)),
        }

        if len(np.unique(true_col)) > 1:
            metrics["auroc"] = float(roc_auc_score(true_col, prob_col))
        else:
            metrics["auroc"] = None

        results["labels"][label] = metrics

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / "evaluation.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"saved={output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.test_csv, args.checkpoint, threshold=args.threshold)
