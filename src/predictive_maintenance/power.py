"""Presupuesto estadístico (CLAUDE.md §10.3, extra 2 de §12).

Traduce el ancho del intervalo de Wilson del recall (`evaluate.recall_wilson_ci`)
en la pregunta inversa, que es la que de verdad importa para planificar: para
un margen de error deseado en torno a un recall objetivo, ¿cuántos fallos
observados hacen falta? Y esos fallos, ¿a cuántos ciclos de máquina
corresponden a la prevalencia actual de `lab180` (CLAUDE.md §6.2, 5.5556 %)?

Con 10 fallos observados hoy, la respuesta (CLAUDE.md §10.3) es incómoda:
estimar un recall de 0.80 con ±10 puntos porcentuales de margen requeriría 62
fallos — unos 1 116 ciclos de máquina a la prevalencia actual. El repositorio
no tiene esos datos, y no los va a inventar.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, learning_curve

from predictive_maintenance import evaluate, pipeline
from predictive_maintenance.config import get_settings
from predictive_maintenance.evaluate import recall_wilson_ci  # reexportado (tarea 9 de la Fase 4)

__all__ = [
    "learning_curve_pr_auc",
    "recall_wilson_ci",
    "required_observations",
    "required_positives",
]


def _resolve_seed(seed: int | None) -> int:
    return seed if seed is not None else get_settings().seed


def required_positives(target_recall: float, margin: float, alpha: float = 0.05) -> int:
    """Nº de fallos observados para un margen `+/- margin` en torno a `target_recall`.

    Fórmula clásica de tamaño muestral para una proporción, por aproximación
    normal: `n = ceil(z^2 * p * (1-p) / margin^2)`, con `p = target_recall` y
    `z` el percentil normal para `1 - alpha/2`. Es la fórmula estándar para
    PLANIFICAR cuántas observaciones hacen falta — distinta del intervalo de
    Wilson (`recall_wilson_ci`), que se usa para REPORTAR el margen real de
    una muestra pequeña ya observada, más fiable que la normal en ese caso.
    Para `target_recall=0.80`: `margin=0.10` da 62; `margin=0.05` da 246
    (CLAUDE.md §10.3).
    """
    z = norm.ppf(1.0 - alpha / 2.0)
    p = target_recall
    n = (z**2) * p * (1.0 - p) / (margin**2)
    return math.ceil(n)


def required_observations(n_positives: int, prevalence: float) -> int:
    """Ciclos de máquina para acumular `n_positives` fallos, a `prevalence` dada.

    `round(n_positives / prevalence)`: con `n_positives=62` y la prevalencia
    real de `lab180` (5.5556 %), da 1116 (CLAUDE.md §10.3).
    """
    return round(n_positives / prevalence)


def learning_curve_pr_auc(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int | None = None,
    positive_label: str = "yes",
    model_name: str = "logistic_balanced",
    train_sizes: np.ndarray | None = None,
    cv_splits: int = 5,
) -> pd.DataFrame:
    """PR-AUC media y desviación (`StratifiedKFold`) para tamaños crecientes de entrenamiento.

    ADVERTENCIA (CLAUDE.md §10.2): con 10 positivos, esta curva NO sube con
    más datos: baja, con bandas de desviación de hasta ±0.30. No es un error
    de esta función — es la curva real, dominada por ruido de muestreo, no
    por aprendizaje. `figures.figure_learning_curve` la dibuja tal cual, con
    las bandas, y anota explícitamente que la señal está dominada por el
    ruido. No la suavices ni la presentes como creciente.
    """
    seed = _resolve_seed(seed)
    y_bin = evaluate.binarize(y, positive_label)
    pipe = pipeline.build_pipeline(model_name, seed=seed)
    tamanos = train_sizes if train_sizes is not None else np.linspace(0.25, 0.85, 6)
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    tamanos_abs, _, test_scores = learning_curve(
        pipe,
        X,
        y_bin,
        train_sizes=tamanos,
        cv=splitter,
        scoring="average_precision",
    )
    return pd.DataFrame(
        {
            "n_entrenamiento": tamanos_abs.astype(int),
            "pr_auc_media": test_scores.mean(axis=1),
            "pr_auc_desv": test_scores.std(axis=1),
        }
    )
