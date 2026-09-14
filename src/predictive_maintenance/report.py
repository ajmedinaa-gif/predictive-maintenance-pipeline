"""Construye el payload de resultados: `build_*_report` / `write_*_report` -> JSON.

Regla dura (CLAUDE.md §2.8): ningún número visible del repositorio se escribe a
mano. Se lee de `reports/*.json`, y este módulo es quien lo genera.

Este módulo NUNCA dibuja: si necesita rutas de figuras, importa `figures.py`.
En la Fase 5 crece con el renderizador HTML de jinja2, que lee este mismo JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from predictive_maintenance import datasets, eda


def _jsonable(obj):
    """Convierte tipos de numpy/pandas a tipos nativos serializables."""
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def build_eda_report(df: pd.DataFrame, spec: datasets.DatasetSpec) -> dict:
    """Reúne en un solo diccionario todo lo que el EDA sabe del dataset."""
    target = spec.target
    positive = spec.positive_label
    return {
        "dataset": spec.name,
        "fichero": str(spec.path.relative_to(datasets.PROJECT_ROOT)),
        "datos_simulados": spec.simulated,
        "n_filas": len(df),
        "n_columnas": int(df.shape[1]),
        "objetivo": target,
        "clase_positiva": positive,
        "balance_de_clases": _jsonable(eda.class_balance(df, target, positive)),
        "descriptiva": _jsonable(eda.descriptive_table(df, target)),
        "nulos": _jsonable(eda.missingness_report(df, target)),
        "anomalias_fisicas": _jsonable(eda.physical_anomalies(df)),
        "rangos_fisicos": {k: list(v) for k, v in eda.PHYSICAL_RANGES.items()},
        "asociacion_univariante": _jsonable(eda.univariate_auc(df, target, positive)),
        "correlaciones": {
            "matriz": _jsonable(eda.correlation_matrix(df, target).round(6)),
            "maxima_absoluta": _jsonable(eda.max_abs_correlation(df, target)),
        },
    }


def write_eda_report(report: dict, path: Path) -> Path:
    """Vuelca el informe a JSON. Es la única fuente de números del repositorio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    return path
