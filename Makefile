SHELL := /bin/bash
PY ?= python

.PHONY: setup smoke-train tokenizer pretrain finetune up down logs build migrate test fmt lint

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

# End-to-end smoke training: tiny tokenizer + tiny model + 200 steps on the
# bundled mini-corpus. ~3 minutes on CPU. Produces runs/smoke/{tokenizer.json,final.pt}.
smoke-train:
	$(PY) -m data.scripts.prepare_smoke
	$(PY) -m tokenizer.bpe_trainer --config configs/tokenizer_smoke.yaml
	$(PY) -m training.scripts.train_pretrain --config configs/train_smoke.yaml
	$(PY) -m training.scripts.train_finetune --config configs/finetune_smoke.yaml

tokenizer:
	$(PY) -m tokenizer.bpe_trainer --config configs/tokenizer.yaml

pretrain:
	$(PY) -m training.scripts.train_pretrain --config configs/train_small.yaml

finetune:
	$(PY) -m training.scripts.train_finetune --config configs/finetune_small.yaml

up:
	docker compose -f infrastructure/docker-compose.yml up --build

down:
	docker compose -f infrastructure/docker-compose.yml down

logs:
	docker compose -f infrastructure/docker-compose.yml logs -f --tail=200

build:
	docker compose -f infrastructure/docker-compose.yml build

migrate:
	docker compose -f infrastructure/docker-compose.yml run --rm backend alembic upgrade head

test:
	$(PY) -m pytest -q

fmt:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

lint:
	$(PY) -m ruff check .
