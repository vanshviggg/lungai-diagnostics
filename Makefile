.PHONY: setup train eval infer runserver test docker-build docker-run

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train:
	. .venv/bin/activate && python src/lungai/train.py

eval:
	. .venv/bin/activate && python src/lungai/evaluate.py || true

infer:
	. .venv/bin/activate && python src/lungai/infer.py --image sample.jpg || true

runserver:
	. .venv/bin/activate && python src/web/manage.py migrate && python src/web/manage.py runserver 0.0.0.0:8000

test:
	. .venv/bin/activate && pytest -q

docker-build:
	docker build -t lungai:latest -f docker/Dockerfile .

docker-run:
	docker run --rm -p 8000:8000 lungai:latest
