import json

import numpy as np
import torch

from lungai.config import DISEASE_LABELS
from lungai.evaluation.evaluate import compute_metrics, get_device, load_thresholds
from lungai.evaluation.tune_thresholds import select_best_threshold


def test_get_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device() == torch.device("cpu")


def test_load_thresholds_and_result_structure(tmp_path) -> None:
    values = {label: 0.5 for label in DISEASE_LABELS}
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(json.dumps({"thresholds": values}))

    loaded = load_thresholds(threshold_path)
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_prob = np.array([[0.9, 0.2, 0.8], [0.1, 0.7, 0.3]])
    results = compute_metrics(y_true, y_prob, loaded)

    assert results["thresholds"] == values
    assert set(results["labels"]) == set(DISEASE_LABELS)
    for label in DISEASE_LABELS:
        assert set(results["labels"][label]) == {
            "threshold",
            "precision",
            "recall",
            "f1",
            "auroc",
            "support",
        }


def test_threshold_selection_maximizes_f1() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    grid = np.array([0.3, 0.5, 0.7])

    assert select_best_threshold(y_true, y_prob, grid) == 0.5
