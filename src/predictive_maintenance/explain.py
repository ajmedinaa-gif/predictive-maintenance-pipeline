"""Explicabilidad del modelo y límites de la evidencia (CLAUDE.md §12, extras 2 y 3; §10.1).

Dos análisis independientes:

- `shap_values_logistic`: valores SHAP de `logistic_plain` (CLAUDE.md §9.2,
  el modelo elegido) con `shap.LinearExplainer`, en el espacio ya imputado y
  escalado que ve el clasificador. La calibración de Platt (`calibration.py`)
  es una transformación monótona 1D (sigmoide) sobre la salida del modelo
  base: no cambia qué atributo empuja la predicción ni en qué dirección, solo
  reescala la probabilidad final — por eso se explica el modelo base, sin
  calibrar, y no el envoltorio `CalibratedClassifierCV`.
- `tree_root_stability`: el test de estabilidad de CLAUDE.md §10.1 —
  bootstraps ESTRATIFICADOS (cada clase se remuestrea por separado, así el
  desbalance 170/10 nunca cambia entre remuestreos) y, en cada uno, qué
  atributo queda como primer corte del árbol. Protege contra la conclusión
  errónea "el atributo que decide el fallo es la vibración": lo es en poco
  más de la mitad de los remuestreos, no siempre.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import shap

from predictive_maintenance import pipeline
from predictive_maintenance.config import get_settings


def _resolve_seed(seed: int | None) -> int:
    return seed if seed is not None else get_settings().seed


def shap_values_logistic(
    X: pd.DataFrame, y: pd.Series, seed: int | None = None, positive_label: str = "yes"
) -> dict:
    """Ajusta `logistic_plain` sobre TODO `X, y` y calcula valores SHAP con `LinearExplainer`.

    Es el único sitio del proyecto donde un modelo se ajusta sobre el dataset
    COMPLETO, a propósito: SHAP explica el modelo final que se desplegaría,
    no un modelo de un fold de CV — la capacidad de generalizar de ESE tipo de
    modelo ya se midió honestamente en `evaluate.py` (CLAUDE.md §8). Esta
    función responde una pregunta distinta: qué atributos usa el modelo final
    y en qué dirección, no si generaliza.
    """
    seed = _resolve_seed(seed)
    pipe = pipeline.build_pipeline("logistic_plain", seed=seed)
    pipe.fit(X, y)

    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    clasificador = pipe.named_steps["classifier"]
    X_transformado = scaler.transform(imputer.transform(X))

    explainer = shap.LinearExplainer(clasificador, X_transformado)
    valores = explainer.shap_values(X_transformado)
    valor_base = explainer.expected_value

    idx_pos = list(clasificador.classes_).index(positive_label)
    if isinstance(valores, list):
        valores = valores[idx_pos]
    if isinstance(valor_base, (list, np.ndarray)):
        valor_base = np.asarray(valor_base).reshape(-1)[idx_pos]

    return {
        "columnas": list(X.columns),
        "shap_values": np.asarray(valores),
        "valor_base": float(valor_base),
        "X_transformado": X_transformado,
        "positive_label": positive_label,
    }


def fit_tree_shallow_balanced(X: pd.DataFrame, y: pd.Series, seed: int | None = None):
    """Ajusta `tree_shallow_balanced` sobre TODO `X, y` (CLAUDE.md §10.1) para dibujarlo.

    Igual que `shap_values_logistic`: es el único árbol que se ajusta sobre el
    dataset completo, a propósito -- para renderizarlo, no para medir cómo
    generaliza (eso ya lo mide `evaluate.py` con CV). Devuelve el
    `DecisionTreeClassifier` ya ajustado, listo para `figures.figure_tree_render`.
    """
    seed = _resolve_seed(seed)
    pipe = pipeline.build_pipeline("tree_shallow_balanced", seed=seed)
    pipe.fit(X, y)
    return pipe.named_steps["classifier"]


def tree_root_stability(
    X: pd.DataFrame, y: pd.Series, n_boot: int = 300, seed: int | None = None
) -> dict:
    """Estabilidad de la raíz del árbol bajo `n_boot` bootstraps ESTRATIFICADOS (CLAUDE.md §10.1).

    Cada remuestreo se construye remuestreando CADA clase por separado, con
    reposición y al mismo tamaño que la clase original, y concatenando: así
    el desbalance 170/10 nunca varía entre remuestreos, solo varía QUÉ diez
    positivos (y qué 170 negativos, con repetición) entran en cada árbol. El
    árbol es siempre `tree_shallow_balanced` (`max_depth=3,
    min_samples_leaf=5, class_weight="balanced"`, CLAUDE.md §10.1).

    Devuelve el % de bootstraps en que cada atributo fue el primer corte y la
    profundidad efectiva media. Reproduce APROXIMADAMENTE los porcentajes de
    CLAUDE.md §10.1 (vibración ~51 %) — el valor exacto depende de la
    semilla, por diseño: el test `tests/test_tree_stability.py` protege que
    la vibración NO domine por encima de 0.70, no que valga un número exacto.
    """
    seed = _resolve_seed(seed)
    rng = np.random.default_rng(seed)
    y_str = y.astype(str).reset_index(drop=True)
    X = X.reset_index(drop=True)
    indices_por_clase = [np.flatnonzero((y_str == clase).to_numpy()) for clase in y_str.unique()]

    raiz_por_bootstrap: list[str] = []
    profundidades: list[int] = []
    for _ in range(n_boot):
        idx_muestra = np.concatenate(
            [rng.choice(idxs, size=len(idxs), replace=True) for idxs in indices_por_clase]
        )
        X_boot = X.iloc[idx_muestra]
        y_boot = y_str.iloc[idx_muestra]

        semilla_arbol = int(rng.integers(0, 2**31 - 1))
        pipe = pipeline.build_pipeline("tree_shallow_balanced", seed=semilla_arbol)
        pipe.fit(X_boot, y_boot)
        arbol = pipe.named_steps["classifier"]

        indice_raiz = int(arbol.tree_.feature[0])
        raiz_por_bootstrap.append(X.columns[indice_raiz])
        profundidades.append(int(arbol.get_depth()))

    conteo = Counter(raiz_por_bootstrap)
    porcentajes = {
        atributo: 100.0 * conteo.get(atributo, 0) / n_boot
        for atributo in sorted(X.columns, key=lambda c: -conteo.get(c, 0))
    }

    return {
        "n_boot": n_boot,
        "porcentaje_raiz_por_atributo": porcentajes,
        "atributo_raiz_mas_frecuente": max(porcentajes, key=porcentajes.get),
        "porcentaje_raiz_mas_frecuente": max(porcentajes.values()),
        "profundidad_efectiva_media": float(np.mean(profundidades)),
    }


def summarize_shap(shap_result: dict, y: pd.Series) -> tuple[dict, list]:
    """Resume `shap_values_logistic` para el informe JSON.

    Dos vistas: la importancia media de `|SHAP|` por atributo (para comparar
    magnitud de influencia, con signo perdido a propósito), y los tres
    atributos que más empujaron cada uno de los 10 casos `failure="yes"` —
    son pocos como para no poder mirarlos uno a uno (CLAUDE.md, tarea Fase 4
    D, pregunta 8 del anexo académico).
    """
    columnas = shap_result["columnas"]
    valores = shap_result["shap_values"]

    importancia = {col: float(np.mean(np.abs(valores[:, i]))) for i, col in enumerate(columnas)}
    importancia = dict(sorted(importancia.items(), key=lambda kv: -kv[1]))

    y_str = y.astype(str).reset_index(drop=True)
    idx_positivos = np.flatnonzero((y_str == "yes").to_numpy())
    casos = []
    for idx in idx_positivos:
        contribuciones = valores[idx]
        orden = np.argsort(-np.abs(contribuciones))[:3]
        casos.append(
            {
                "fila": int(idx),
                "top_contribuciones": [
                    {"atributo": columnas[i], "shap_value": float(contribuciones[i])} for i in orden
                ],
            }
        )
    return importancia, casos
