# LungAI Diagnostics

LungAI Diagnostics is an explainable AI research prototype for thoracic disease classification from chest X-ray images. The project is designed as a reproducible machine-learning pipeline using PyTorch, transfer learning, measurable evaluation, and model explainability.

> **Research-use disclaimer:** This repository is for education, research, and software-engineering demonstration only. It is not a medical device, has not been clinically validated, and must not be used for diagnosis or patient-care decisions.

## Current scope

The first implementation focuses on a transparent baseline pipeline:

1. Load and preprocess chest X-ray images.
2. Fine-tune a pretrained DenseNet121 classifier.
3. Produce per-class probabilities using sigmoid outputs.
4. Evaluate the trained model with measurable metrics.
5. Add explainability and a web/API demonstration after the baseline is validated.

## Planned disease labels

The initial prototype is structured for three thoracic findings:

- Cardiomegaly
- Pneumonia
- Pneumothorax

The label set can be expanded after the baseline pipeline is validated on a documented public dataset.

## Tech stack

- Python
- PyTorch
- Torchvision
- Pandas / NumPy
- Scikit-learn
- Pillow
- Pytest

## Repository structure

```text
lungai-diagnostics/
├── src/lungai/
│   ├── config.py
│   ├── data/transforms.py
│   ├── models/chest_xray_model.py
│   └── inference/predictor.py
├── tests/
├── artifacts/
├── docs/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Current project status

The repository is under active development. Model training results, AUROC/F1/precision/recall metrics, Grad-CAM outputs, screenshots, and deployment evidence will only be added after they are generated from actual runs.

## Evidence policy

No clinical-performance claims are made without reproducible experiments. Any future metrics in this repository will include the dataset, split methodology, evaluation code, and generated artifacts needed to reproduce them.

## License

See `LICENSE`.
