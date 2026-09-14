"""Configuración tipada del proyecto, leída de `config/default.yaml`.

`pydantic-settings` valida la FORMA de `default.yaml` al cargarlo: si falta una
clave o el tipo no cuadra, falla aquí, con un mensaje claro, y no en mitad del
entrenamiento. Regla dura (CLAUDE.md §2.10): la semilla se define una sola vez
en el YAML; ningún módulo debe llevar un literal de configuración propio. Las
rutas se resuelven absolutas contra la raíz del proyecto en `from_yaml`, nunca
en tiempo de importación (para no fijar una ruta antes de saber desde dónde se
ejecuta el CLI).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class PathsSettings(BaseModel):
    """Rutas del proyecto, ya resueltas como absolutas."""

    data_raw: Path
    data_interim: Path
    data_quarantine: Path
    reports: Path
    figures: Path


class DatasetEntry(BaseModel):
    """Metadatos de un dataset registrado en `datasets:` del YAML."""

    file: str
    target: str
    positive_label: str
    simulated: bool = False


class CrossValidationScheme(BaseModel):
    """Esquema de CV de un dataset (CLAUDE.md §8): nº de folds y repeticiones."""

    n_splits: int
    n_repeats: int = 1


class QualityThresholds(BaseModel):
    """Umbrales de los avisos de calidad del contrato de datos (CLAUDE.md §6.3)."""

    min_prevalence: float
    max_prevalence: float
    max_null_pct_per_column: float


class Settings(BaseSettings):
    """Vista tipada de `config/default.yaml`.

    Puede sobreescribirse con variables de entorno con prefijo `PDM_`, aunque
    el proyecto no depende de ese mecanismo hoy: el YAML es la fuente de
    verdad (CLAUDE.md, cabecera).
    """

    model_config = SettingsConfigDict(env_prefix="PDM_", extra="ignore")

    seed: int
    paths: PathsSettings
    datasets: dict[str, DatasetEntry]
    cross_validation: dict[str, CrossValidationScheme] = Field(default_factory=dict)
    quality_thresholds: QualityThresholds

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> Settings:
        """Construye `Settings` a partir de un fichero YAML, con rutas absolutas."""
        with path.open(encoding="utf-8") as fh:
            crudo = yaml.safe_load(fh)
        crudo = dict(crudo)
        crudo["paths"] = {clave: PROJECT_ROOT / valor for clave, valor in crudo["paths"].items()}
        return cls(**crudo)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instancia única y cacheada de `Settings`, leída de `config/default.yaml`."""
    return Settings.from_yaml()
