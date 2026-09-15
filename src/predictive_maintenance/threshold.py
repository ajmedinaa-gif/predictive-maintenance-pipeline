"""Umbral de decisión por coste (CLAUDE.md §9, extra 4 de §12).

Convierte una probabilidad en una decisión de mantenimiento: clasificar como
"fallo" tiene sentido económico solo si el coste de hacerlo es menor que el de
no hacerlo. El umbral por defecto de scikit-learn (0.5) no tiene ningún
significado económico aquí — es un accidente de que la función de pérdida
logística es simétrica, no una decisión sobre costes reales.

Regla dura (CLAUDE.md §2.5): el umbral por coste NUNCA se combina con
`class_weight="balanced"` — contarían el desbalance dos veces. Este módulo
asume probabilidades de un modelo ajustado SIN balanceo (`logistic_plain`),
idealmente ya calibradas (`calibration.py`).

CRÍTICO (CLAUDE.md §2.7): el umbral se optimiza DENTRO de cada fold de
validación cruzada, sobre las predicciones out-of-fold del fold de
ENTRENAMIENTO — nunca sobre el fold de test. Las funciones de este módulo son
puras y no saben nada de folds ni de particiones: es responsabilidad de quien
las llama (`calibration.evaluate_cost_variants`) pasarles solo predicciones
del entrenamiento al buscar el umbral, y aplicar ese umbral ya fijo al fold de
test únicamente para medir. Optimizar sobre el propio fold de test sería fuga
de datos: el umbral se ajustaría al ruido de esas 2-3 muestras positivas de
test en vez de a una señal generalizable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COST_KEYS: tuple[str, ...] = (
    "cost_true_negative",
    "cost_false_positive",
    "cost_false_negative",
    "cost_true_positive",
)


def _confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def realized_cost(y_true, y_pred, costs: dict) -> float:
    """Coste total en CLP de una decisión binaria YA tomada (`y_pred` en `{0, 1}`), contra `y_true`.

    A diferencia de `expected_cost`, no aplica ningún umbral: sirve para el
    caso en que el umbral usado varía por fold (CLAUDE.md §2.7,
    `calibration.evaluate_cost_variants`) y no hay un único `threshold` que
    pasarle a `expected_cost`.
    """
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    return (
        tn * costs["cost_true_negative"]
        + fp * costs["cost_false_positive"]
        + fn * costs["cost_false_negative"]
        + tp * costs["cost_true_positive"]
    )


def expected_cost(y_true, y_prob, threshold: float, costs: dict) -> float:
    """Coste total en CLP de clasificar `y_prob >= threshold` como fallo, contra `y_true`.

    No es "esperado" en sentido probabilístico (no promedia sobre la
    incertidumbre del modelo): es el coste REALIZADO (`realized_cost`) de
    aplicar `threshold` a las predicciones y compararlas con las etiquetas
    verdaderas, sumando la matriz de coste de CLAUDE.md §9.1
    (`config/costs.yaml`) sobre los cuatro resultados posibles.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return realized_cost(y_true, y_pred, costs)


def cost_curve(y_true, y_prob, costs: dict, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Coste total en función del umbral, base de `cost_vs_threshold.png`."""
    umbrales = np.linspace(0.0, 1.0, 1001) if thresholds is None else np.asarray(thresholds)
    costes = [expected_cost(y_true, y_prob, t, costs) for t in umbrales]
    return pd.DataFrame({"threshold": umbrales, "coste_total": costes})


def optimal_threshold(y_true, y_prob, costs: dict, thresholds: np.ndarray | None = None) -> float:
    """Umbral que minimiza el coste total, por barrido sobre `np.linspace(0, 1, 1001)`.

    Para un modelo PERFECTAMENTE calibrado y con datos suficientes, el óptimo
    empírico converge al valor teórico de CLAUDE.md §9.1 (`theoretical_threshold`):

        t* = (C_FP - C_TN) / ((C_FP - C_TN) + (C_FN - C_TP))

    En la práctica ningún modelo entrenado con 10 positivos está
    perfectamente calibrado (CLAUDE.md §9.3): el óptimo empírico y el teórico
    divergen, y esa distancia es en sí misma una medida de cuán mal calibrado
    está el modelo — no un error del barrido.

    En empate (varios umbrales consecutivos con el mismo coste mínimo — el
    coste es constante entre dos valores de `y_prob` consecutivos), se toma la
    mediana de los umbrales empatados: un desempate estable, ni el primero ni
    el último del barrido.
    """
    curva = cost_curve(y_true, y_prob, costs, thresholds)
    minimo = curva["coste_total"].min()
    empatados = curva.loc[curva["coste_total"] == minimo, "threshold"]
    return float(empatados.median())


def theoretical_threshold(costs: dict) -> float:
    """Umbral óptimo (t*) de CLAUDE.md §9.1: exacto solo si el modelo está PERFECTAMENTE calibrado.

    t* = (C_FP - C_TN) / ((C_FP - C_TN) + (C_FN - C_TP))

    Depende solo de la matriz de coste, nunca del dataset: con
    `config/costs.yaml` da 0.0556. La fórmula incluye `C_TP`, que NO es cero
    en este proyecto — implementarla como `C_FP / (C_FP + C_FN)` (ignorando
    `C_TP`) da 0.0476, y está mal (CLAUDE.md §9.1).
    """
    c_fp = costs["cost_false_positive"] - costs["cost_true_negative"]
    c_fn = costs["cost_false_negative"] - costs["cost_true_positive"]
    return c_fp / (c_fp + c_fn)
