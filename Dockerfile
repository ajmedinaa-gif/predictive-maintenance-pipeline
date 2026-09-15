# syntax=docker/dockerfile:1
#
# Multi-stage: el stage `builder` compila el entorno con `uv` en /opt/venv;
# el stage `runtime` solo copia ese venv ya resuelto -- nada de toolchain de
# compilación, nada de caché de `uv`, nada de `.git` ni de tests en la imagen
# final. CI (ubuntu-latest, x86_64) es quien construye y mide esta imagen
# (CLAUDE.md §14.7): esta máquina de desarrollo no tiene Docker.
#
# Sin `--platform linux/amd64` en ningún sitio: el Air Intel de desarrollo y
# los runners de CI ya son x86_64 (CLAUDE.md §14.7).

FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv==0.12.13

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Capa de dependencias sola primero: cambia mucho menos que el código fuente,
# así que se cachea entre builds mientras `uv.lock` no cambie.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python 3.11 \
    && uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
# --no-editable: sin esto, `uv sync` instala el paquete en modo editable --
# el venv termina con un simple .pth apuntando a /build/src, que en el stage
# runtime no existe (el código vive en /app/src). Con --no-editable el
# paquete se instala de verdad dentro de site-packages y /opt/venv queda
# autocontenido: el runtime no necesita copiar src/ en absoluto.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="predictive-maintenance-pipeline" \
      org.opencontainers.image.description="Mantenimiento predictivo industrial: pipeline reproducible con validación honesta bajo desbalance de clases" \
      org.opencontainers.image.source="https://github.com/ajmedinaa-gif/predictive-maintenance-pipeline" \
      org.opencontainers.image.licenses="MIT"

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PDM_CONFIG_DIR=/app/config \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/config ./config
COPY app ./app
# `lab180` viene versionado en el repo (dataset simulado pequeño, CLAUDE.md
# §7): se hornea en la imagen. `ai4i2020` NUNCA -- es una descarga explícita
# y separada (`pdm-cli download --dataset ai4i2020`, CLAUDE.md §2.11), y el
# contenedor la hace en tiempo de ejecución, contra el volumen de `data/`.
COPY data/raw/lab180 ./data/raw/lab180

RUN mkdir -p data/interim data/quarantine reports/figures \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["pdm-cli", "version"]

ENTRYPOINT ["pdm-cli"]
CMD ["--help"]
