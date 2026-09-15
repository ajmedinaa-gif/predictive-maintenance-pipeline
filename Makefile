# Todos los targets invocan `uv run ...`, nunca python/pip/pytest a secas
# (CLAUDE.md §14.1). Sintaxis compatible con GNU Make 3.81 (el de macOS):
# sin .ONESHELL ni funciones de Make 4.x.

.PHONY: install lint test eda validate train calibrate explain limits report \
        data run compare dashboard docker-build docker-run notebook clean all

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

report:
	uv run pdm-cli report --dataset lab180

# Descarga explícita del segundo dataset (CLAUDE.md §2.11): toca red, una
# sola vez. Nunca es una dependencia oculta de otro target salvo `run`.
data:
	uv run pdm-cli download --dataset ai4i2020

# El mismo pipeline sobre los dos datasets (CLAUDE.md §13.1): contrato +
# zoo de modelos + calibración + explicabilidad + límites + informe HTML,
# para `lab180` (sin red) y para `ai4i2020` (descarga primero vía `data`).
run: data
	uv run pdm-cli run --dataset lab180
	uv run pdm-cli calibrate --dataset lab180
	uv run pdm-cli explain --dataset lab180
	uv run pdm-cli limits --dataset lab180
	uv run pdm-cli report --dataset lab180
	uv run pdm-cli run --dataset ai4i2020
	uv run pdm-cli calibrate --dataset ai4i2020
	uv run pdm-cli explain --dataset ai4i2020
	uv run pdm-cli limits --dataset ai4i2020
	uv run pdm-cli report --dataset ai4i2020
	uv run pdm-cli compare

# Compara ambos datasets: requiere que sus reports/*.json ya existan
# (ejecuta `make run` antes si es la primera vez).
compare:
	uv run pdm-cli compare

dashboard:
	uv run streamlit run app/streamlit_app.py

# CLAUDE.md §14.7: esta máquina no tiene Docker. `docker-build`/`docker-run`
# existen para quien sí lo tenga (o para CI, que es quien de verdad los
# ejecuta y mide la imagen) -- no se invocan desde ningún otro target.
docker-build:
	docker build -t predictive-maintenance-pipeline:latest .

docker-run:
	docker run --rm predictive-maintenance-pipeline:latest version

# Re-ejecuta el anexo académico con un kernel Jupyter real (nbclient +
# ipykernel), NO ensambla el JSON a mano. NO entra en `all`: es lento
# (levanta un kernel) y no es parte del pipeline principal.
notebook:
	uv run python tools/run_notebook.py

clean:
	rm -rf .pytest_cache .ruff_cache .hypothesis .coverage coverage.xml htmlcov
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +

# `validate` NO entra en `all`: lab180 tiene una fila en cuarentena a
# propósito (CLAUDE.md §6.3), así que sale siempre con código 1. Es la señal
# correcta para un gate de CI que revisa cuarentena; no lo es para "build
# verde" de desarrollo local. `run` (con `ai4i2020`) tampoco entra: toca red.
all: lint test eda train calibrate explain limits report
