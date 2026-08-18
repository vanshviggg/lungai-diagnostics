.PHONY: setup test train eval

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest

train:
	. .venv/bin/activate && PYTHONPATH=src python -m lungai.training.train --train-csv data/train.csv --val-csv data/val.csv

eval:
	. .venv/bin/activate && PYTHONPATH=src python -m lungai.evaluation.evaluate --test-csv data/test.csv --checkpoint artifacts/models/best_model.pt
