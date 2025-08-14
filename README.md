# LungAI Diagnostics — AI-Powered Lung Disease Detection

**GitHub/Repo/Demo link:** _Add link here after you push_

## Description
LungAI Diagnostics is a machine learning–powered diagnostic tool designed to detect lung diseases from patient medical imaging and clinical data. It integrates a LAMS lung model with Large Language Models (LLMs) to analyze both image patterns and textual patient information, providing doctors with early, accurate diagnostic suggestions. The solution is deployed as a user-friendly web application built with Django, allowing healthcare professionals to input patient data and receive real-time insights.

## Problem it solves
Early detection of lung diseases is crucial for patient outcomes, but manual diagnosis can be slow and prone to human error. This tool automates the detection process using AI, significantly reducing diagnosis time while maintaining high accuracy, thus enabling faster treatment decisions.

## My role
- Led the end-to-end development — from data preparation and model training to deployment.
- Trained and fine-tuned the LAMS lung model and LLMs in Python, focusing on pattern recognition in medical imaging and clinical data.
- Designed and implemented the Django-based backend for real-time inference.
- Integrated AI models with a responsive web interface for ease of use in clinical environments.
- Conducted rigorous testing and optimization to ensure high precision and recall in predictions.

## Tech stack
Python, Django, TensorFlow/PyTorch, Hugging Face Transformers, LAMS Model, PostgreSQL, AWS EC2/S3, HTML/CSS/JS.

## Quickstart

```bash
# 1) Create virtual env & install deps
make setup

# 2) Prepare data (placeholders)
# Expected files:
#   data/train.csv (columns: path,label)
#   data/val.csv
#   data/<image files...>
# NOTE: data/ is gitignored by default

# 3) Train and export best model
make train

# 4) Inference (CLI)
python src/lungai/infer.py --image sample.jpg

# 5) Run API
python src/web/manage.py migrate
python src/web/manage.py runserver 0.0.0.0:8000
# POST an image:
# curl -F "file=@sample.jpg" http://localhost:8000/api/predict/
```

## Project layout
See the repo tree in the root of this project.

## Notes
- Replace the placeholder CNN with your LAMS pipeline or a stronger pretrained model.
- Put trained weights at `models/best_model.pt` (gitignored).
- Add experiment tracking (W&B / MLflow) and security for production.
