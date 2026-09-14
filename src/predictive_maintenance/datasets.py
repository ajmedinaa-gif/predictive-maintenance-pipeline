"""Registro de datasets y carga desde disco.

Regla dura (CLAUDE.md §2.11): **ninguna descarga externa ocurre como efecto
lateral de importar este módulo**. Aquí solo se lee lo que ya está en `data/raw`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class DatasetSpec:
    """Descripción de un dataset: dónde vive y cuál es su objetivo."""

    name: str
    path: Path
    target: str
    positive_label: str
    simulated: bool


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Lee `config/default.yaml`. La semilla global vive ahí y solo ahí."""
    with CONFIG_FILE.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def seed() -> int:
    """Semilla global del proyecto (CLAUDE.md §2.10)."""
    return int(load_config()["seed"])


def get_spec(name: str) -> DatasetSpec:
    """Devuelve la especificación del dataset `name`."""
    config = load_config()
    entries = config["datasets"]
    if name not in entries:
        disponibles = ", ".join(sorted(entries))
        raise KeyError(f"Dataset desconocido: {name!r}. Disponibles: {disponibles}")
    entry = entries[name]
    raw_root = PROJECT_ROOT / config["paths"]["data_raw"]
    return DatasetSpec(
        name=name,
        path=raw_root / entry["file"],
        target=entry["target"],
        positive_label=str(entry["positive_label"]),
        simulated=bool(entry.get("simulated", False)),
    )


def load(name: str) -> pd.DataFrame:
    """Carga el dataset `name` tal cual está en disco, sin transformar nada.

    No imputa, no escala y no descarta la fila anómala: eso ocurre dentro del
    `sklearn.Pipeline`, después del split (CLAUDE.md §2.6).
    """
    spec = get_spec(name)
    if not spec.path.exists():
        raise FileNotFoundError(
            f"No encuentro {spec.path}. El dataset {name!r} debe estar en data/raw/."
        )
    return pd.read_csv(spec.path)
