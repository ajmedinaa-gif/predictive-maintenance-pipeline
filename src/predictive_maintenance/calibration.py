"""Calibración de probabilidades y su cruce con el umbral por coste (CLAUDE.md §9).

`logistic_plain` (CLAUDE.md §9.2) es la base: es el único modelo del zoo sin
`class_weight`, así que sus probabilidades no están distorsionadas por
reescalar la función de pérdida — el punto de partida correcto para calibrar.
`class_weight="balanced"` y el umbral por coste NO se combinan nunca
(CLAUDE.md §2.5, regla dura): contarían el desbalance dos veces. Por eso
`logistic_balanced` entra en la comparación de `evaluate_cost_variants` solo
como CONTRAEJEMPLO de esa regla — para enseñar en números cuánto peor sale.

ADVERTENCIA (CLAUDE.md, tarea Fase 4 A.1): con 10 positivos, la calibración
isotónica es no paramétrica y sobreajusta — se espera que Platt (`sigmoid`,
paramétrica con dos parámetros) gane en Brier y en PR-AUC. Este módulo mide
las cuatro variantes con el mismo protocolo y reporta el resultado real, sea
cual sea: si algún día la isotónica ganara, sería un hallazgo que reportar, no
un motivo para ajustar el código hasta que Platt vuelva a ganar.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

from predictive_maintenance import evaluate, pipeline, threshold
from predictive_maintenance.config import get_settings

VARIANT_LABELS: dict[str, str] = {
    "logistic_balanced": "logistic balanced",
    "logistic_plain": "logistic plain",
    "logistic_plain_platt": "logistic plain + Platt",
    "logistic_plain_isotonic": "logistic plain + isotónica",
}


def _resolve_seed(seed: int | None) -> int:
    return seed if seed is not None else get_settings().seed


def _logistic_balanced(seed: int, cv: int):
    return pipeline.build_pipeline("logistic_balanced", seed=seed)


def _logistic_plain(seed: int, cv: int):
    return pipeline.build_pipeline("logistic_plain", seed=seed)


def _logistic_plain_platt(seed: int, cv: int):
    base = pipeline.build_pipeline("logistic_plain", seed=seed)
    return CalibratedClassifierCV(base, method="sigmoid", cv=cv)


def _logistic_plain_isotonic(seed: int, cv: int):
    base = pipeline.build_pipeline("logistic_plain", seed=seed)
    return CalibratedClassifierCV(base, method="isotonic", cv=cv)


VARIANT_BUILDERS: dict[str, Callable[[int, int], object]] = {
    "logistic_balanced": _logistic_balanced,
    "logistic_plain": _logistic_plain,
    "logistic_plain_platt": _logistic_plain_platt,
    "logistic_plain_isotonic": _logistic_plain_isotonic,
}


def reliability_curve(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Curva de fiabilidad: probabilidad media predicha frente a frecuencia observada, por bin.

    `n_bins` bins de ancho igual en `[0, 1]`. Un bin sin observaciones se
    conserva en la tabla con `n=0` y las medias en `NaN` — para que
    `figure_calibration_curve` no dibuje un punto donde no hay datos, pero la
    tabla documente que ese tramo de probabilidad no se observó en absoluto.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_prob, bordes[1:-1], right=False), 0, n_bins - 1)

    filas = []
    for b in range(n_bins):
        mascara = indices == b
        n = int(mascara.sum())
        filas.append(
            {
                "bin": b,
                "limite_inferior": float(bordes[b]),
                "limite_superior": float(bordes[b + 1]),
                "n": n,
                "prob_media_predicha": float(y_prob[mascara].mean()) if n else float("nan"),
                "frecuencia_observada": float(y_true[mascara].mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame(filas)


def brier_decomposition(y_true, y_prob, n_bins: int = 10) -> dict:
    """Descomposición de Murphy (1973) del Brier score: fiabilidad - resolución + incertidumbre.

    - **incertidumbre**: `obar * (1 - obar)`, la varianza de la prevalencia
      misma — el Brier del clasificador trivial que siempre predice la
      prevalencia observada.
    - **resolución**: cuánto se alejan las frecuencias observadas de cada bin
      de esa prevalencia — cuánta información aporta el modelo al separar los
      bins entre sí. Más alta es mejor.
    - **fiabilidad**: cuánto se alejan las probabilidades PREDICHAS de las
      observadas dentro de cada bin — el término que Platt/isotónica intentan
      llevar a cero. Más baja es mejor.

    Es una APROXIMACIÓN dependiente de `n_bins`: con más bins, `reconstruccion`
    converge al Brier score exacto (`sklearn.metrics.brier_score_loss`); con
    pocos bins se pierde resolución al promediar probabilidades distintas
    dentro del mismo bin.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    curva = reliability_curve(y_true, y_prob, n_bins)
    n_total = len(y_true)
    obar = float(y_true.mean())
    incertidumbre = obar * (1.0 - obar)

    validos = curva.dropna(subset=["prob_media_predicha"])
    fiabilidad = float(
        (
            validos["n"] * (validos["prob_media_predicha"] - validos["frecuencia_observada"]) ** 2
        ).sum()
        / n_total
    )
    resolucion = float(
        (validos["n"] * (validos["frecuencia_observada"] - obar) ** 2).sum() / n_total
    )
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "brier_score": brier,
        "fiabilidad": fiabilidad,
        "resolucion": resolucion,
        "incertidumbre": incertidumbre,
        "reconstruccion": fiabilidad - resolucion + incertidumbre,
        "n_bins": n_bins,
    }


def evaluate_cost_variants(
    X: pd.DataFrame,
    y: pd.Series,
    costs: dict,
    *,
    seed: int | None = None,
    outer_splits: int = 5,
    threshold_inner_splits: int = 3,
    calibration_cv: int = 3,
    positive_label: str = "yes",
) -> dict[str, dict]:
    """Reproduce la tabla de CLAUDE.md §9.2: Brier, PR-AUC, umbral empírico y ahorro, por variante.

    Protocolo, fiel a las reglas duras 5 y 7 de CLAUDE.md:

    - Reparto externo `StratifiedKFold(outer_splits, shuffle=True, seed)`, SIN
      repetir (igual que `evaluate.aggregate_confusion_matrix`, CLAUDE.md
      §8.2): cada fila recibe una predicción out-of-fold exactamente una vez.
    - Dentro de cada fold externo, el umbral de coste se optimiza (regla dura
      7) sobre predicciones out-of-fold de un `StratifiedKFold(
      threshold_inner_splits)` interno al fold de ENTRENAMIENTO — nunca se
      mira el fold de test para elegir el umbral. Ese umbral, ya fijo, es el
      que se aplica al fold de test para contar TP/FP/FN/TN y calcular coste.
    - `logistic_plain_platt` y `logistic_plain_isotonic` envuelven
      `logistic_plain` en un `CalibratedClassifierCV` con su propia CV interna
      (`calibration_cv`), así que hay hasta tres niveles de CV anidada. Con
      179 filas y 10 positivos esto sigue siendo rápido; `calibration_cv=3`
      (en vez de 5) es deliberado para dejar al menos 2-3 positivos en cada
      sub-fold interno de calibración.

    Devuelve, por variante, Brier y PR-AUC sobre las predicciones out-of-fold
    agregadas, el umbral empírico (mediana de los umbrales por fold), la
    matriz de confusión agregada a ESE umbral, y el coste total en CLP al
    umbral 0.5 frente al umbral empírico, con el ahorro resultante.
    """
    seed = _resolve_seed(seed)
    y_str = y.astype(str)
    y_true_bin_completo = evaluate.binarize(y_str, positive_label)
    n = len(y_str)

    splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    folds_externos = list(splitter.split(X, y_str))

    resultados: dict[str, dict] = {}
    for nombre, builder in VARIANT_BUILDERS.items():
        y_score_oof = np.empty(n, dtype=float)
        y_pred_oof = np.empty(n, dtype=int)
        umbrales_por_fold: list[float] = []

        for train_idx, test_idx in folds_externos:
            X_train, y_train = X.iloc[train_idx], y_str.iloc[train_idx]
            X_test = X.iloc[test_idx]

            modelo = builder(seed, calibration_cv)
            modelo.fit(X_train, y_train)
            idx_pos = evaluate.positive_column(modelo, positive_label)
            y_score_test = modelo.predict_proba(X_test)[:, idx_pos]

            # Umbral optimizado DENTRO del fold de entrenamiento (regla dura 7),
            # sobre predicciones out-of-fold de una CV interna a ese train.
            y_true_train_oof, y_score_train_oof, _ = evaluate.out_of_fold_predictions(
                builder(seed, calibration_cv),
                X_train,
                y_train,
                seed=seed,
                n_splits=threshold_inner_splits,
                positive_label=positive_label,
            )
            t_fold = threshold.optimal_threshold(y_true_train_oof, y_score_train_oof, costs)
            umbrales_por_fold.append(t_fold)

            y_score_oof[test_idx] = y_score_test
            y_pred_oof[test_idx] = (y_score_test >= t_fold).astype(int)

        brier = float(brier_score_loss(y_true_bin_completo, y_score_oof))
        pr_auc = float(average_precision_score(y_true_bin_completo, y_score_oof))
        t_empirico = float(np.median(umbrales_por_fold))

        tn, fp, fn, tp = _confusion(y_true_bin_completo, y_pred_oof)

        coste_t05 = threshold.expected_cost(y_true_bin_completo, y_score_oof, 0.5, costs)
        coste_t_empirico = threshold.realized_cost(y_true_bin_completo, y_pred_oof, costs)
        ahorro = coste_t05 - coste_t_empirico
        ahorro_pct = 100.0 * ahorro / coste_t05 if coste_t05 else float("nan")

        resultados[nombre] = {
            "etiqueta": VARIANT_LABELS[nombre],
            "brier_score": brier,
            "pr_auc": pr_auc,
            "umbral_empirico": t_empirico,
            "umbrales_por_fold": [float(t) for t in umbrales_por_fold],
            "matriz_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "coste_umbral_05": float(coste_t05),
            "coste_umbral_empirico": float(coste_t_empirico),
            "ahorro_clp": float(ahorro),
            "ahorro_pct": float(ahorro_pct),
        }

    return resultados


def _confusion(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> tuple[int, int, int, int]:
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    return tn, fp, fn, tp
