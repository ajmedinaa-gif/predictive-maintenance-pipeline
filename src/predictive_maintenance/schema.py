"""Contrato de datos ejecutable para `lab180` (CLAUDE.md §6.3, §2.9).

Dos niveles, deliberadamente distintos:

- **Rangos físicos** (`Lab180Schema`, un `pandera.DataFrameModel`): lo que la
  física permite. Una fila que los viola no es un dato ruidoso, es un
  imposible físico — no se imputa, va a cuarentena (`data.py`).
- **Avisos de calidad** (`quality_warnings`): umbrales configurables sobre el
  dataframe COMPLETO (prevalencia, nulos por columna, duplicados) que no
  invalidan una fila concreta, así que nunca lanzan excepción — solo se
  registran como WARNING.

Los rangos físicos reutilizan `eda.PHYSICAL_RANGES` en vez de redeclarar los
límites: es la misma física, no dos fuentes de verdad.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa

from predictive_maintenance import eda
from predictive_maintenance.config import QualityThresholds

logger = logging.getLogger(__name__)

_TEMP_MIN, _TEMP_MAX = eda.PHYSICAL_RANGES["temperature_c"]
_VIB_MIN, _VIB_MAX = eda.PHYSICAL_RANGES["vibration_mm_s"]
_PRES_MIN, _PRES_MAX = eda.PHYSICAL_RANGES["pressure_bar"]
_HOURS_MIN, _HOURS_MAX = eda.PHYSICAL_RANGES["hours_since_maintenance"]
_LOAD_MIN, _LOAD_MAX = eda.PHYSICAL_RANGES["load_percent"]


class Lab180Schema(pa.DataFrameModel):
    """Contrato físico de `lab180`. `strict=True`: ninguna columna extra cuela.

    Los tres sensores con nulos conocidos (CLAUDE.md §6.3) son `nullable=True`:
    un nulo es un dato ausente, no una violación del contrato. Un imposible
    físico —como `vibration_mm_s = -0.34`— sí lo es: `ge=0.0` porque una
    amplitud RMS de vibración no puede ser negativa.
    """

    temperature_c: float = pa.Field(nullable=True, ge=_TEMP_MIN, le=_TEMP_MAX)
    vibration_mm_s: float = pa.Field(nullable=True, ge=_VIB_MIN, le=_VIB_MAX)
    pressure_bar: float = pa.Field(nullable=True, ge=_PRES_MIN, le=_PRES_MAX)
    hours_since_maintenance: int = pa.Field(ge=_HOURS_MIN, le=_HOURS_MAX)
    load_percent: float = pa.Field(ge=_LOAD_MIN, le=_LOAD_MAX)
    failure: str = pa.Field(isin=["yes", "no"])

    class Config:
        strict = True
        coerce = False


# Rangos físicos de `ai4i2020` (CLAUDE.md §13, Fase 5): temperaturas en
# Kelvin, rango plausible para un taller industrial (280-320 K, es decir
# 6.85-46.85 °C). El resto de sensores no puede ser negativo; no hay un
# máximo físico conocido más allá del que ya impone la resolución del sensor,
# así que se deja sin límite superior.
_AI4I_TEMP_MIN, _AI4I_TEMP_MAX = 280.0, 320.0


class AI4I2020Schema(pa.DataFrameModel):
    """Contrato físico de `ai4i2020` (UCI id=601, Matzka 2020). `strict=True`.

    Incluye las cinco columnas de modo de fallo (`twf`...`rnf`): son parte del
    dataset crudo y se validan igual que el resto, pero NUNCA entran como
    feature de modelado (`AI4I2020Adapter.feature_columns` no las incluye,
    CLAUDE.md §13) porque son sub-indicadores casi deterministas de
    `machine_failure` — incluirlas sería fuga del objetivo.
    """

    # `rotational_speed_rpm` y `tool_wear_min` llegan como enteros del CSV de
    # origen (UCI): rpm y minutos enteros, no una medición fraccionaria.
    type: str = pa.Field(isin=["L", "M", "H"])
    air_temperature_k: float = pa.Field(ge=_AI4I_TEMP_MIN, le=_AI4I_TEMP_MAX)
    process_temperature_k: float = pa.Field(ge=_AI4I_TEMP_MIN, le=_AI4I_TEMP_MAX)
    rotational_speed_rpm: int = pa.Field(gt=0)
    torque_nm: float = pa.Field(ge=0.0)
    tool_wear_min: int = pa.Field(ge=0)
    machine_failure: int = pa.Field(isin=[0, 1])
    twf: int = pa.Field(isin=[0, 1])
    hdf: int = pa.Field(isin=[0, 1])
    pwf: int = pa.Field(isin=[0, 1])
    osf: int = pa.Field(isin=[0, 1])
    rnf: int = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        coerce = False


SCHEMAS: dict[str, type[pa.DataFrameModel]] = {"lab180": Lab180Schema, "ai4i2020": AI4I2020Schema}
"""Registro dataset -> contrato. Espejo de `datasets.ADAPTERS[name].schema`."""


def quality_warnings(
    df: pd.DataFrame,
    thresholds: QualityThresholds,
    target: str = "failure",
    positive_label: str = "yes",
) -> list[str]:
    """Avisos de calidad sobre el dataframe completo. Nunca lanza excepción.

    Comprueba tres cosas que no son "inválidas" en el sentido del contrato de
    filas, pero sí indicios de que algo va mal con el dataset en su conjunto:
    prevalencia fuera de un rango razonable, una columna con demasiados nulos,
    o duplicados completos. Cada aviso se registra con `logging.warning` y se
    devuelve también como texto para incluirlo en el informe.
    """
    avisos: list[str] = []

    if target in df.columns and len(df):
        balance = eda.class_balance(df, target, positive_label)
        prevalencia = balance["prevalencia"]
        if not (thresholds.min_prevalence <= prevalencia <= thresholds.max_prevalence):
            aviso = (
                f"prevalencia de '{positive_label}' = {prevalencia:.4%}, fuera del rango "
                f"de aviso [{thresholds.min_prevalence:.0%}, {thresholds.max_prevalence:.0%}]"
            )
            avisos.append(aviso)

    n_filas = len(df)
    if n_filas:
        for col in eda.numeric_attributes(df, target):
            pct_nulos = df[col].isna().sum() / n_filas
            if pct_nulos > thresholds.max_null_pct_per_column:
                aviso = (
                    f"columna '{col}' tiene {pct_nulos:.2%} de nulos, por encima del umbral "
                    f"de aviso {thresholds.max_null_pct_per_column:.0%}"
                )
                avisos.append(aviso)

    n_duplicados = int(df.duplicated().sum())
    if n_duplicados:
        avisos.append(f"{n_duplicados} filas completamente duplicadas")

    for aviso in avisos:
        logger.warning(aviso)

    return avisos
