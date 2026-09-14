"""Construye el payload de resultados: `build_*_report` / `write_*_report` -> JSON.

Regla dura (CLAUDE.md §2.8): ningún número visible del repositorio se escribe a
mano. Se lee de `reports/*.json`, y este módulo es quien lo genera.

Este módulo NUNCA dibuja: si necesita rutas de figuras, importa `figures.py`.
En la Fase 5 crece con el renderizador HTML de jinja2, que lee este mismo JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from predictive_maintenance import datasets, eda, evaluate

if TYPE_CHECKING:
    from predictive_maintenance import plausibility
    from predictive_maintenance.data import ValidationResult

_COLUMNAS_METRICAS = ("fold", "n_test", "n_positivos_test")
_COLUMNAS_TABLA_RESULTADOS = (
    ("average_precision", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
    ("recall", "recall"),
    ("precision", "precision"),
    ("balanced_accuracy", "bal.acc"),
    ("accuracy", "accuracy"),
    ("brier_score_loss", "Brier"),
)


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


def write_json_report(payload: dict, path: Path) -> Path:
    """Vuelca cualquier payload de informe a JSON, indentado y legible.

    Es la única función que escribe un `reports/*.json` en todo el proyecto
    (CLAUDE.md §2.8): ningún número visible del repositorio se escribe a mano.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def write_eda_report(report: dict, path: Path) -> Path:
    """Vuelca el informe de EDA a JSON. Alias de `write_json_report`."""
    return write_json_report(report, path)


def build_validation_report(result: ValidationResult, spec: datasets.DatasetSpec) -> dict:
    """Payload del contrato de datos: qué pasó, qué se puso en cuarentena y por qué."""
    return {
        "dataset": spec.name,
        "fichero": str(spec.path.relative_to(datasets.PROJECT_ROOT)),
        **result.report,
        "cuarentena": _jsonable(result.quarantined),
    }


def write_validation_report(report: dict, path: Path) -> Path:
    """Vuelca el informe de validación del contrato a JSON."""
    return write_json_report(report, path)


def write_markdown_report(texto: str, path: Path) -> Path:
    """Vuelca un informe ya renderizado en markdown a disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texto, encoding="utf-8")
    return path


def build_metrics_report(
    *,
    dataset: str,
    n_filas: int,
    n_positivos: int,
    resultados_por_modelo: dict[str, dict],
    protocolo_principal: str,
    protocolo_matriz_confusion: str,
    seed: int,
    nested_cv: dict | None = None,
) -> dict:
    """Payload de la Fase 3: métricas por modelo (media + IC bootstrap) y matriz de confusión.

    `resultados_por_modelo[nombre]` trae `"folds"` (el `DataFrame` de
    `evaluate.cross_validate_model`, una fila por fold) y
    `"matriz_confusion"` (el dict de `evaluate.aggregate_confusion_matrix`).
    El orden de `resultados_por_modelo` se conserva tal cual en el payload:
    quien lo construye es responsable de poner los `dummy_*` primero
    (CLAUDE.md §2.1).
    """
    modelos = {}
    for nombre, resultado in resultados_por_modelo.items():
        folds = resultado["folds"]
        columnas_metrica = [c for c in folds.columns if c not in _COLUMNAS_METRICAS]
        metricas = {}
        for metrica in columnas_metrica:
            valores = folds[metrica].to_numpy()
            media = float(np.nanmean(valores))
            ic_lower, ic_upper = evaluate.bootstrap_ci(valores, seed=seed)
            metricas[metrica] = {"media": media, "ic_bootstrap_95": [ic_lower, ic_upper]}
        modelos[nombre] = {
            "metricas": metricas,
            "matriz_confusion": _jsonable(resultado["matriz_confusion"]),
        }

    payload = {
        "dataset": dataset,
        "n_filas_entrenamiento": n_filas,
        "n_positivos": n_positivos,
        "protocolo_principal": protocolo_principal,
        "protocolo_matriz_confusion": protocolo_matriz_confusion,
        "modelos": modelos,
    }
    if nested_cv is not None:
        payload["nested_cv"] = _jsonable(nested_cv)
    return _jsonable(payload)


def write_metrics_report(report: dict, path: Path) -> Path:
    """Vuelca el informe de métricas de la Fase 3 a JSON."""
    return write_json_report(report, path)


def render_results_markdown(payload: dict) -> str:
    """Tabla markdown de resultados a partir de `build_metrics_report`.

    Regla dura (CLAUDE.md §2.1): la accuracy nunca va sola, siempre junto a la
    fila del `DummyClassifier`. El orden de las filas es el orden de
    `payload["modelos"]`; construir ese diccionario con los `dummy_*` primero
    es responsabilidad de quien llama.
    """
    encabezado = ["modelo", *(etiqueta for _, etiqueta in _COLUMNAS_TABLA_RESULTADOS)]
    lineas = [
        "| " + " | ".join(encabezado) + " |",
        "|" + "|".join(["---"] * len(encabezado)) + "|",
    ]
    for nombre, resultado in payload["modelos"].items():
        fila = [nombre]
        for clave, _ in _COLUMNAS_TABLA_RESULTADOS:
            fila.append(f"{resultado['metricas'][clave]['media']:.4f}")
        lineas.append("| " + " | ".join(fila) + " |")
    return "\n".join(lineas) + "\n"


def build_plausibility_report(informe: plausibility.PlausibilityReport) -> dict:
    """Payload del auditor de plausibilidad, listo para volcar a JSON."""
    return _jsonable(informe.to_dict())


def write_plausibility_report(report: dict, path: Path) -> Path:
    """Vuelca el veredicto del auditor de plausibilidad a JSON."""
    return write_json_report(report, path)
