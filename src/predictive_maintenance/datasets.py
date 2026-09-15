"""Registro de datasets: el `DatasetAdapter` de cada uno y su carga desde disco.

Regla dura (CLAUDE.md §2.11): **ninguna descarga externa ocurre como efecto
lateral de importar este módulo**. Aquí solo se lee lo que ya está en
`data/raw`; la descarga es un método explícito (`adapter.download()`),
invocado únicamente por `pdm-cli download`.

`DatasetAdapter` es el límite de abstracción de CLAUDE.md §13.1: todo lo que
es específico de un dataset (dónde vive, qué columnas son features, cómo se
derivan las features de ingeniería) vive en su adaptador. Todo lo demás
—`pipeline.py`, `evaluate.py`, `calibration.py`...— es exactamente el mismo
código para `lab180` y `ai4i2020`, porque solo ven `X`/`y` ya numéricos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pandera.pandas as pa
import yaml

from predictive_maintenance import schema as schema_module
from predictive_maintenance.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT

logger = logging.getLogger(__name__)

CONFIG_FILE = DEFAULT_CONFIG_PATH


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Lee `config/default.yaml`. La semilla global vive ahí y solo ahí."""
    with CONFIG_FILE.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def seed() -> int:
    """Semilla global del proyecto (CLAUDE.md §2.10)."""
    return int(load_config()["seed"])


def _raw_path(name: str) -> Path:
    """Ruta absoluta del CSV crudo de `name`, resuelta desde `config/default.yaml`."""
    config = load_config()
    entries = config["datasets"]
    if name not in entries:
        disponibles = ", ".join(sorted(entries))
        raise KeyError(f"Dataset desconocido: {name!r}. Disponibles: {disponibles}")
    raw_root = PROJECT_ROOT / config["paths"]["data_raw"]
    return raw_root / entries[name]["file"]


# --------------------------------------------------------------------------- #
# Protocolo DatasetAdapter (CLAUDE.md §13)
# --------------------------------------------------------------------------- #


@runtime_checkable
class DatasetAdapter(Protocol):
    """Todo lo que un dataset necesita aportar para correr por el mismo pipeline.

    - `download()`: descarga (o verifica) el CSV crudo. Nunca se llama al
      importar el módulo (regla dura 11).
    - `schema`: el `pandera.DataFrameModel` que valida ese CSV crudo.
    - `feature_columns`: las columnas —ya con la ingeniería de features
      aplicada— que entran al `sklearn.Pipeline` como `X`. Nunca incluye el
      objetivo ni columnas que lo filtren (como los modos de fallo de
      `ai4i2020`).
    - `engineer_features(df)`: transformación determinista y sin ajuste
      (nada que se aprenda de los datos) de las columnas crudas a
      `feature_columns`. Para `lab180` es la identidad.
    """

    name: str
    target_column: str
    positive_label: str
    simulated: bool
    feature_columns: tuple[str, ...]
    schema: type[pa.DataFrameModel]

    def download(self) -> Path: ...

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame: ...


class Lab180Adapter:
    """`lab180`: datos SIMULADOS de laboratorio, versionados en el repo (CLAUDE.md §7)."""

    name = "lab180"
    target_column = "failure"
    positive_label = "yes"
    simulated = True
    feature_columns: tuple[str, ...] = (
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
        "hours_since_maintenance",
        "load_percent",
    )
    schema: type[pa.DataFrameModel] = schema_module.Lab180Schema

    def download(self) -> Path:
        """`lab180` no se descarga: ya viene versionado en `data/raw/` del repo.

        Solo verifica que el fichero exista, para que el protocolo sea
        conforme y `pdm-cli download --dataset lab180` dé un mensaje claro en
        vez de un `FileNotFoundError` más adelante en el pipeline.
        """
        path = _raw_path(self.name)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} no existe. 'lab180' viene versionado en el repositorio "
                "(no se descarga): revisa que el checkout esté completo."
            )
        logger.info("lab180 ya está en disco en %s; no hay descarga que hacer.", path)
        return path

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identidad: las cinco variables de `lab180` ya son las features de modelado."""
        return df.copy()


# Columnas de `ucimlrepo.fetch_ucirepo(id=601)` -> nombres de features físicas
# de CLAUDE.md §13 (snake_case, con la unidad en el nombre donde aplica).
_AI4I2020_COLUMN_MAP = {
    "Type": "type",
    "Air temperature": "air_temperature_k",
    "Process temperature": "process_temperature_k",
    "Rotational speed": "rotational_speed_rpm",
    "Torque": "torque_nm",
    "Tool wear": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


class AI4I2020Adapter:
    """`ai4i2020`: UCI id=601 (Matzka, 2020). Features físicamente acopladas (CLAUDE.md §13)."""

    name = "ai4i2020"
    target_column = "machine_failure"
    # "yes"/"no", NUNCA "1"/"0": con etiquetas que son dígitos puros,
    # `RandomForestClassifier(class_weight="balanced")` sobre datos con
    # bootstrap desencadena una recursión interna de scikit-learn que
    # reinterpreta las clases como enteros para el lookup de pesos y no
    # encuentra la clase en el diccionario que ella misma generó
    # (`ValueError: The classes, [0, 1], are not in class_weight`) --
    # reproducible con un `RandomForestClassifier` plano, sin este proyecto de
    # por medio. Usar el mismo vocabulario que `lab180` ("no"/"yes") evita el
    # problema de raíz y mantiene los informes de ambos datasets consistentes.
    positive_label = "yes"
    simulated = False
    # Post-ingeniería: los tres sensores derivados (justificación física en
    # `engineer_features`) más el one-hot de `type` sin columna de referencia
    # omitida (los modelos del zoo no llevan intercepto regularizado que la
    # necesite; con solo 3 categorías, mantener las tres es más legible que
    # ahorrar una columna). Los cinco modos de fallo NUNCA entran aquí.
    feature_columns: tuple[str, ...] = (
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "power_w",
        "temp_delta_k",
        "wear_x_torque",
        "type_L",
        "type_M",
        "type_H",
    )
    schema: type[pa.DataFrameModel] = schema_module.AI4I2020Schema

    def download(self) -> Path:
        """Descarga real vía `ucimlrepo.fetch_ucirepo(id=601)` (CLAUDE.md §2.11: paso explícito).

        Nunca se invoca desde otro sitio que no sea `pdm-cli download`: los
        demás comandos leen únicamente lo que ya está en `data/raw/ai4i2020/`.
        """
        from ucimlrepo import fetch_ucirepo

        logger.info("Descargando ai4i2020 (UCI id=601) vía ucimlrepo...")
        repo = fetch_ucirepo(id=601)
        df = pd.concat([repo.data.features, repo.data.targets], axis=1)
        df = df.rename(columns=_AI4I2020_COLUMN_MAP)
        columnas_esperadas = list(_AI4I2020_COLUMN_MAP.values())
        df = df[columnas_esperadas]

        path = _raw_path(self.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("ai4i2020 guardado en %s: %d filas.", path, len(df))
        return path

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade las tres features derivadas de CLAUDE.md §13 y codifica `type`.

        Transformación determinista de columnas ya observadas —no se ajusta
        nada a partir de los datos— así que es segura de aplicar antes del
        split de CV, igual que convertir unidades.

        - `power_w`: potencia mecánica = par x velocidad angular
          (`torque_nm * rotational_speed_rpm * 2*pi/60`, de rpm a rad/s).
        - `temp_delta_k`: disipación térmica = temperatura de proceso menos
          temperatura ambiente; un delta pequeño es la firma física de HDF
          (fallo por disipación de calor) en este dataset.
        - `wear_x_torque`: sobreesfuerzo acumulado = desgaste de herramienta
          por par aplicado.
        - `type_{L,M,H}`: one-hot de la categoría de calidad del producto — una
          recodificación de tres categorías fijas, no un ajuste estadístico.

        También remapea `machine_failure` de `{0, 1}` (el contrato valida
        estos enteros crudos) a `{"no", "yes"}`: mismo vocabulario que
        `lab180` y evita el problema de raíz descrito en el comentario de
        `positive_label`.
        """
        out = df.copy()
        out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2 * np.pi / 60
        out["temp_delta_k"] = out["process_temperature_k"] - out["air_temperature_k"]
        out["wear_x_torque"] = out["tool_wear_min"] * out["torque_nm"]
        for categoria in ("L", "M", "H"):
            out[f"type_{categoria}"] = (out["type"] == categoria).astype(int)
        if self.target_column in out.columns:
            out[self.target_column] = out[self.target_column].map({0: "no", 1: "yes"})
        return out


ADAPTERS: dict[str, DatasetAdapter] = {
    "lab180": Lab180Adapter(),
    "ai4i2020": AI4I2020Adapter(),
}


def get_adapter(name: str) -> DatasetAdapter:
    """Devuelve el `DatasetAdapter` registrado para `name`."""
    if name not in ADAPTERS:
        disponibles = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"Dataset desconocido: {name!r}. Disponibles: {disponibles}")
    return ADAPTERS[name]


# --------------------------------------------------------------------------- #
# `DatasetSpec`: vista ligera de un adaptador + su ruta en disco, la que
# consumen `figures.py` y `report.py` (Fases 1-4, sin cambios de firma).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetSpec:
    """Descripción de un dataset: dónde vive y cuál es su objetivo.

    Se deriva siempre de un `DatasetAdapter` (fuente de verdad única de
    `target`/`positive_label`/`simulated`/`feature_columns`); nunca se
    construye a mano con literales.
    """

    name: str
    path: Path
    target: str
    positive_label: str
    simulated: bool
    feature_columns: tuple[str, ...]


def get_spec(name: str) -> DatasetSpec:
    """Devuelve la especificación de `name`, derivada de su `DatasetAdapter`."""
    adapter = get_adapter(name)
    return DatasetSpec(
        name=adapter.name,
        path=_raw_path(name),
        target=adapter.target_column,
        positive_label=adapter.positive_label,
        simulated=adapter.simulated,
        feature_columns=tuple(adapter.feature_columns),
    )


def load(name: str) -> pd.DataFrame:
    """Carga el dataset `name` tal cual está en disco, sin transformar nada.

    No imputa, no escala, no aplica la ingeniería de features ni descarta
    ninguna fila: eso ocurre después del contrato de datos y, en el caso de
    la ingeniería de features, dentro de `adapter.engineer_features`
    (CLAUDE.md §2.6, §2.9).
    """
    spec = get_spec(name)
    if not spec.path.exists():
        raise FileNotFoundError(
            f"No encuentro {spec.path}. El dataset {name!r} debe estar en data/raw/ "
            f"(usa `pdm-cli download --dataset {name}` si aplica)."
        )
    return pd.read_csv(spec.path)
