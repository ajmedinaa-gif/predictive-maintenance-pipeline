# Todos los targets invocan `uv run ...`, nunca python/pip/pytest a secas
# (CLAUDE.md §14.1). Sintaxis compatible con GNU Make 3.81 (el de macOS):
# sin .ONESHELL ni funciones de Make 4.x.

.PHONY: install lint test eda validate train calibrate explain limits all

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff format --check .
	uv run ruff check .

test:
	uv run pytest

eda:
	uv run pdm-cli eda --dataset lab180

validate:
	uv run pdm-cli validate --dataset lab180

train:
	uv run pdm-cli train --dataset lab180

calibrate:
	uv run pdm-cli calibrate --dataset lab180

explain:
	uv run pdm-cli explain --dataset lab180

limits:
	uv run pdm-cli limits --dataset lab180

# `validate` NO entra en `all`: lab180 tiene una fila en cuarentena a
# propósito (CLAUDE.md §6.3), así que sale siempre con código 1. Es la señal
# correcta para un gate de CI que revisa cuarentena; no lo es para "build
# verde" de desarrollo local.
all: lint test eda train calibrate explain limits
