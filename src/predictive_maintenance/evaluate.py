"""Protocolo de validación honesto (CLAUDE.md §8): sin autoengaño posible.

Dos esquemas de validación cruzada, deliberadamente distintos y usados para
cosas distintas:

- `cross_validate_model` usa `RepeatedStratifiedKFold` (5 folds x 10
  repeticiones para `lab180`, CLAUDE.md §8): la tabla de métricas principal.
  Un único reparto en folds, con 10 positivos, es ruido — repetirlo 10 veces
  con particiones distintas es lo que hace que el intervalo de confianza
  signifique algo.
- `aggregate_confusion_matrix` usa un único `StratifiedKFold(5, shuffle=True)`
  (CLAUDE.md §8.2): cada fila del dataset recibe exactamente una predicción
  out-of-fold, y se agregan en una sola matriz de confusión. Repetir aquí no
  tiene sentido: agregaría la misma fila varias veces con predicciones de
  modelos distintos, mezclando conteos que no son comparables.

En ambos casos, el modelo se clona sin ajustar en cada fold: nunca se reutiliza
un `Pipeline` ya ajustado entre folds, así que el imputador (y el escalador, si
lo hay) siempre ven solo la partición de entrenamiento de ESE fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from statsmodels.stats.proportion import proportion_confint

from predictive_maintenance.config import Settings, get_settings

METRIC_NAMES: tuple[str, ...] = (
    "average_precision",
    "balanced_accuracy",
    "roc_auc",
    "recall",
    "precision",
    "f1",
    "accuracy",
    "brier_score_loss",
)


@dataclass(frozen=True)
class EvalConfig:
    """Esquema de CV para un dataset: nº de folds, repeticiones y semilla.

    Se construye a partir de `config/default.yaml` (`cross_validation:` +
    `seed:`), nunca con literales sueltos en el código que lo use.
    """

    n_splits: int
    n_repeats: int
    seed: int

    @classmethod
    def for_dataset(cls, dataset: str, settings: Settings | None = None) -> EvalConfig:
        settings = settings or get_settings()
        if dataset not in settings.cross_validation:
            disponibles = ", ".join(sorted(settings.cross_validation))
            raise KeyError(
                f"Sin esquema de CV para {dataset!r} en config/default.yaml. "
                f"Disponibles: {disponibles}"
            )
        esquema = settings.cross_validation[dataset]
        return cls(n_splits=esquema.n_splits, n_repeats=esquema.n_repeats, seed=settings.seed)


def _binarize(y, positive_label: str) -> np.ndarray:
    return (pd.Series(y).astype(str).to_numpy() == positive_label).astype(int)


def _positive_column(modelo, positive_label: str) -> int:
    clases = list(modelo.classes_)
    if positive_label not in clases:
        raise ValueError(f"La clase positiva {positive_label!r} no aparece en classes_={clases}")
    return clases.index(positive_label)


def _fold_metrics(y_test_bin: np.ndarray, y_pred_bin: np.ndarray, y_score: np.ndarray) -> dict:
    hay_dos_clases = len(np.unique(y_test_bin)) > 1
    return {
        "average_precision": float(average_precision_score(y_test_bin, y_score)),
        "roc_auc": float(roc_auc_score(y_test_bin, y_score)) if hay_dos_clases else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_test_bin, y_pred_bin)),
        "recall": float(recall_score(y_test_bin, y_pred_bin, zero_division=0)),
        "precision": float(precision_score(y_test_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_test_bin, y_pred_bin, zero_division=0)),
        "accuracy": float(accuracy_score(y_test_bin, y_pred_bin)),
        "brier_score_loss": float(brier_score_loss(y_test_bin, y_score)),
    }


def cross_validate_model(
    pipe, X: pd.DataFrame, y: pd.Series, cfg: EvalConfig, positive_label: str = "yes"
) -> pd.DataFrame:
    """Métricas de CLAUDE.md §8 por fold, sobre `RepeatedStratifiedKFold`.

    Devuelve un `DataFrame` con una fila por (repetición, fold) y una columna
    por métrica de `METRIC_NAMES` — la base tanto de la media reportada como
    de `bootstrap_ci`, que remuestrea estas filas, nunca el dataset original.
    `pipe` se clona sin ajustar en cada fold (ver docstring del módulo).
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.seed
    )
    filas = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        modelo = clone(pipe)
        modelo.fit(X.iloc[train_idx], y.iloc[train_idx])

        y_test_bin = _binarize(y.iloc[test_idx], positive_label)
        y_pred_bin = _binarize(modelo.predict(X.iloc[test_idx]), positive_label)
        idx_pos = _positive_column(modelo, positive_label)
        y_score = modelo.predict_proba(X.iloc[test_idx])[:, idx_pos]

        fila = {"fold": fold_id, "n_test": len(test_idx), "n_positivos_test": int(y_test_bin.sum())}
        fila.update(_fold_metrics(y_test_bin, y_pred_bin, y_score))
        filas.append(fila)

    return pd.DataFrame(filas)


def bootstrap_ci(
    values, n: int = 2000, alpha: float = 0.05, seed: int | None = None
) -> tuple[float, float]:
    """IC por bootstrap sobre los folds (CLAUDE.md §8): NO sobre las filas originales.

    Remuestrea `values` (los valores por fold de una métrica) con reposición,
    `n` veces, y toma los percentiles `[alpha/2, 1-alpha/2]` de la media de
    cada remuestra. Con `values` constante, toda remuestra tiene la misma
    media, así que el intervalo degenera al mismo valor en ambos extremos.
    """
    valores = np.asarray(values, dtype=float)
    valores = valores[~np.isnan(valores)]
    if len(valores) == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed if seed is not None else get_settings().seed)
    medias = np.empty(n)
    for i in range(n):
        remuestra = rng.choice(valores, size=len(valores), replace=True)
        medias[i] = remuestra.mean()

    lower = float(np.percentile(medias, 100 * alpha / 2))
    upper = float(np.percentile(medias, 100 * (1 - alpha / 2)))
    return (lower, upper)


def recall_wilson_ci(n_correct: int, n_positives: int, alpha: float = 0.05) -> tuple[float, float]:
    """IC de Wilson del recall (CLAUDE.md §8): sobre el CONTEO de positivos, no por bootstrap.

    Con recuentos tan pequeños como 10 positivos, el bootstrap sobre folds no
    tiene sentido para esta métrica en particular: es un conteo binomial
    (aciertos sobre positivos totales), y el intervalo de Wilson es el
    intervalo estándar para una proporción binomial con muestras pequeñas.
    """
    if n_positives == 0:
        return (float("nan"), float("nan"))
    lower, upper = proportion_confint(n_correct, n_positives, alpha=alpha, method="wilson")
    return (float(lower), float(upper))


def out_of_fold_predictions(
    pipe, X: pd.DataFrame, y: pd.Series, seed: int, n_splits: int = 5, positive_label: str = "yes"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicciones out-of-fold con un único `StratifiedKFold` (CLAUDE.md §8.2).

    Cada fila recibe una predicción de un modelo que nunca la vio, exactamente
    una vez. Es la base de `aggregate_confusion_matrix` y de las curvas PR/ROC
    (una curva por modelo, no 50 curvas superpuestas de la CV repetida).
    Devuelve `(y_true_bin, y_score, y_pred_bin)`.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_true_bin = _binarize(y, positive_label)
    y_score = np.empty(len(y_true_bin), dtype=float)
    y_pred_bin = np.empty(len(y_true_bin), dtype=int)

    for train_idx, test_idx in splitter.split(X, y):
        modelo = clone(pipe)
        modelo.fit(X.iloc[train_idx], y.iloc[train_idx])
        idx_pos = _positive_column(modelo, positive_label)
        y_score[test_idx] = modelo.predict_proba(X.iloc[test_idx])[:, idx_pos]
        y_pred_bin[test_idx] = _binarize(modelo.predict(X.iloc[test_idx]), positive_label)

    return y_true_bin, y_score, y_pred_bin


def aggregate_confusion_matrix(
    pipe, X: pd.DataFrame, y: pd.Series, seed: int, n_splits: int = 5, positive_label: str = "yes"
) -> dict:
    """Matriz de confusión agregada out-of-fold (CLAUDE.md §8.2), en conteos y normalizada.

    Un único `StratifiedKFold(n_splits, shuffle=True)`, NO la CV repetida de
    `cross_validate_model`: aquí cada fila cuenta una sola vez.
    """
    y_true_bin, _, y_pred_bin = out_of_fold_predictions(
        pipe, X, y, seed=seed, n_splits=n_splits, positive_label=positive_label
    )
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    n_negativos = tn + fp
    n_positivos = fn + tp

    def _tasa(numerador: int, denominador: int) -> float:
        return float(numerador / denominador) if denominador else float("nan")

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_negativos": int(n_negativos),
        "n_positivos": int(n_positivos),
        "normalizada_por_fila": {
            "no": {"no": _tasa(tn, n_negativos), "yes": _tasa(fp, n_negativos)},
            "yes": {"no": _tasa(fn, n_positivos), "yes": _tasa(tp, n_positivos)},
        },
        "recall": _tasa(tp, n_positivos),
        "accuracy": _tasa(tn + tp, n_negativos + n_positivos),
        "recall_wilson_ci": recall_wilson_ci(int(tp), int(n_positivos)),
    }


def nested_cv(
    pipe,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: EvalConfig,
    positive_label: str = "yes",
    scoring: str = "average_precision",
    inner_splits: int = 3,
) -> pd.DataFrame:
    """CV anidada: `GridSearchCV` interno de `inner_splits` folds, evaluado en un fold externo.

    Elegir hiperparámetros y reportar la métrica final sobre el mismo split
    produce una estimación optimista — el propio proceso de selección de
    modelo se ajusta al ruido de esa partición. La CV anidada separa las dos
    cosas: el `GridSearchCV` interno solo ve el fold de ENTRENAMIENTO externo;
    la métrica se calcula sobre el fold de test externo, que el interno nunca
    vio. Vabalas, A. et al. (2019). "Machine learning algorithm validation
    with a limited sample size." PLOS ONE 14(11): e0224365.

    `y` se binariza (1 = `positive_label`) antes de entrar a `GridSearchCV`:
    los scorers de scikit-learn como `"average_precision"` o `"roc_auc"`
    asumen `pos_label=1` y fallan con etiquetas de texto como `"yes"/"no"`.
    """
    y_bin_completo = _binarize(y, positive_label)
    splitter_externo = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.seed
    )
    filas = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter_externo.split(X, y_bin_completo)):
        splitter_interno = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=cfg.seed
        )
        buscador = GridSearchCV(
            clone(pipe), param_grid, scoring=scoring, cv=splitter_interno, refit=True
        )
        buscador.fit(X.iloc[train_idx], y_bin_completo[train_idx])
        mejor = buscador.best_estimator_

        y_test_bin = y_bin_completo[test_idx]
        y_pred_bin = mejor.predict(X.iloc[test_idx])
        # Con `y` ya binarizado a {0, 1}, `classes_` es siempre `[0, 1]`: la
        # columna 1 de `predict_proba` es la clase positiva sin ambigüedad.
        y_score = mejor.predict_proba(X.iloc[test_idx])[:, 1]

        fila = {"fold": fold_id, "mejores_hiperparametros": buscador.best_params_}
        fila.update(_fold_metrics(y_test_bin, y_pred_bin, y_score))
        filas.append(fila)

    return pd.DataFrame(filas)
