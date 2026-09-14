# Todos los targets invocan `uv run ...`, nunca python/pip/pytest a secas
# (CLAUDE.md §14.1). Sintaxis compatible con GNU Make 3.81 (el de macOS):
# sin .ONESHELL ni funciones de Make 4.x.

.PHONY: install lint test eda all

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

all: lint test eda
