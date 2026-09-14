"""Tests del protocolo de validación (CLAUDE.md §8): que no se autoengañe.

Los tests de regresión contra los valores de CLAUDE.md §8.1 corren sobre las
filas que SÍ pasan el contrato de datos (`data.load_validated(...).valid`,
179 filas): entrenar sobre la fila en cuarentena violaría CLAUDE.md §2.9. Es
la razón por la que estos números no coinciden exactamente con la tabla de
§8.1 (calculada sobre las 180 filas crudas) — la tolerancia amplia de esa
sección ya anticipa que "el esquema de CV mueve el valor".

El test de la accuracy exacta del dummy mayoritario (0.944444) es la
excepción: usa deliberadamente el CSV crudo (`datasets.load`, 180 filas), NO
el dataset validado, porque valida la ARITMÉTICA de `cross_validate_model`
contra la constante documentada en CLAUDE.md §6.2 (170/180), no el
comportamiento del pipeline de entrenamiento real.
"""

from unittest import mock

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from predictive_maintenance import data, datasets, evaluate, pipeline, schema

TOL = 1e-3


@pytest.fixture(scope="module")
def lab180_valid_xy():
    """Las filas que pasan el contrato (179): base de todo entrenamiento real."""
    spec = datasets.get_spec("lab180")
    resultado = data.load_validated(spec.path, schema.Lab180Schema, dataset_name="lab180")
    df = resultado.valid
    X = df.drop(columns=[spec.target])
    y = df[spec.target].astype(str)
    return X, y


@pytest.fixture(scope="module")
def cfg_real():
    return evaluate.EvalConfig.for_dataset("lab180")


@pytest.fixture(scope="module")
def folds_por_modelo(lab180_valid_xy, cfg_real):
    """`cross_validate_model` para TODO el zoo, sobre las 179 filas validadas.

    Módulo-scoped: es la CV completa (5x10) de cada uno de los 8 modelos, cara
    de recalcular en cada test. Se comparte entre todas las regresiones de
    esta sección y el test estructural dummy-vs-real.
    """
    X, y = lab180_valid_xy
    resultado = {}
    for nombre in pipeline.MODEL_NAMES:
        pipe = pipeline.build_pipeline(nombre, seed=cfg_real.seed)
        resultado[nombre] = evaluate.cross_validate_model(
            pipe, X, y, cfg_real, positive_label="yes"
        )
    return resultado


@pytest.fixture(scope="module")
def logistic_balanced_folds(folds_por_modelo):
    return folds_por_modelo["logistic_balanced"]


@pytest.fixture(scope="module")
def rf_balanced_folds(folds_por_modelo):
    return folds_por_modelo["rf_balanced"]


# --------------------------------------------------------------------------- #
# Exactitud aritmética: dummy mayoritario sobre el CSV crudo (CLAUDE.md §6.2)
# --------------------------------------------------------------------------- #


def test_dummy_most_frequent_accuracy_is_exactly_the_documented_constant():
    df = datasets.load("lab180")  # 180 filas crudas, a propósito: ver docstring del módulo.
    X = df.drop(columns=["failure"])
    y = df["failure"].astype(str)
    cfg = evaluate.EvalConfig(n_splits=5, n_repeats=10, seed=42)
    pipe = pipeline.build_pipeline("dummy_most_frequent", seed=cfg.seed)

    folds = evaluate.cross_validate_model(pipe, X, y, cfg, positive_label="yes")

    # 170/5 y 10/5 son ambos enteros: cada uno de los 50 folds tiene EXACTAMENTE
    # 34 "no" y 2 "yes", así que la accuracy es 34/36 en los 50 folds, sin
    # varianza que promediar.
    assert folds["accuracy"].nunique() == 1
    assert folds["accuracy"].iloc[0] == pytest.approx(0.944444, abs=TOL)
    assert folds["accuracy"].mean() == pytest.approx(0.944444, abs=TOL)


# --------------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_on_a_constant_is_degenerate():
    valores = [0.6554] * 50
    lower, upper = evaluate.bootstrap_ci(valores, n=2000, alpha=0.05)
    assert lower == pytest.approx(0.6554, abs=TOL)
    assert upper == pytest.approx(0.6554, abs=TOL)


def test_bootstrap_ci_widens_with_more_variance():
    rng = np.random.default_rng(0)
    ajustado = evaluate.bootstrap_ci(rng.normal(0.5, 0.001, size=50), n=1000)
    disperso = evaluate.bootstrap_ci(rng.normal(0.5, 0.2, size=50), n=1000)
    assert (disperso[1] - disperso[0]) > (ajustado[1] - ajustado[0])


# --------------------------------------------------------------------------- #
# recall_wilson_ci
# --------------------------------------------------------------------------- #


def test_recall_wilson_ci_matches_claude_md():
    lower, upper = evaluate.recall_wilson_ci(8, 10)
    assert lower == pytest.approx(0.490, abs=TOL)
    assert upper == pytest.approx(0.943, abs=TOL)


def test_recall_wilson_ci_zero_positives_is_nan():
    lower, upper = evaluate.recall_wilson_ci(0, 0)
    assert np.isnan(lower)
    assert np.isnan(upper)


# --------------------------------------------------------------------------- #
# Estratificación: cada fold de test tiene al menos 1 positivo
# --------------------------------------------------------------------------- #


def test_folds_are_stratified_every_test_fold_has_a_positive(lab180_valid_xy, cfg_real):
    from sklearn.model_selection import RepeatedStratifiedKFold

    X, y = lab180_valid_xy
    splitter = RepeatedStratifiedKFold(
        n_splits=cfg_real.n_splits, n_repeats=cfg_real.n_repeats, random_state=cfg_real.seed
    )
    for _, test_idx in splitter.split(X, y):
        assert (y.iloc[test_idx] == "yes").sum() >= 1


# --------------------------------------------------------------------------- #
# Regresión: CLAUDE.md §8.1 (tolerancias anchas a propósito, ver docstring)
# --------------------------------------------------------------------------- #


def test_logistic_balanced_pr_auc_regression(logistic_balanced_folds):
    media = logistic_balanced_folds["average_precision"].mean()
    assert 0.55 <= media <= 0.80, (
        f"PR-AUC de logistic_balanced = {media:.4f}, fuera de [0.55, 0.80] "
        "(CLAUDE.md §8.1). Parar y reportar, no ajustar el rango."
    )


def test_rf_balanced_roc_auc_regression(rf_balanced_folds):
    media = rf_balanced_folds["roc_auc"].mean()
    assert 0.85 <= media <= 0.98, (
        f"ROC-AUC de rf_balanced = {media:.4f}, fuera de [0.85, 0.98] "
        "(CLAUDE.md §8.1). Parar y reportar, no ajustar el rango."
    )


def test_rf_balanced_recall_is_low(rf_balanced_folds):
    # El resultado titular del repo (CLAUDE.md §8.1): buen ROC-AUC, recall
    # bajo al umbral por defecto. El umbral 0.35 protege "el recall es malo
    # pese al buen ROC-AUC", no un valor puntual — con 10 positivos, el
    # recall salta de 0.1 en 0.1 por fold y un rango estrecho se rompe al
    # cambiar de versión de scikit-learn sin que nada esté mal.
    media = rf_balanced_folds["recall"].mean()
    assert media < 0.35, f"recall de rf_balanced = {media:.4f}, no es < 0.35 (CLAUDE.md §8.1)."


def test_every_real_model_beats_both_dummies_on_pr_auc(folds_por_modelo):
    # Test estructural, no un valor puntual: cualquier modelo que no ordene
    # mejor que los dos triviales no tiene nada que aportar a la tabla,
    # independientemente de la versión de scikit-learn o del reparto exacto
    # de folds (CLAUDE.md §8.1).
    techo_dummy = max(
        folds_por_modelo["dummy_most_frequent"]["average_precision"].mean(),
        folds_por_modelo["dummy_stratified"]["average_precision"].mean(),
    )
    modelos_reales = [n for n in pipeline.MODEL_NAMES if not n.startswith("dummy_")]
    for nombre in modelos_reales:
        media = folds_por_modelo[nombre]["average_precision"].mean()
        assert media > techo_dummy, (
            f"PR-AUC de {nombre} = {media:.4f} no supera el techo de los dummy ({techo_dummy:.4f})."
        )


# --------------------------------------------------------------------------- #
# Fuga de datos: el imputador solo ve el fold de entrenamiento
# --------------------------------------------------------------------------- #


def test_imputer_fits_once_per_fold_on_training_fold_size_only(lab180_valid_xy):
    X, y = lab180_valid_xy
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )
    cfg = evaluate.EvalConfig(n_splits=5, n_repeats=2, seed=42)

    with mock.patch.object(SimpleImputer, "fit", autospec=True, side_effect=SimpleImputer.fit) as m:
        evaluate.cross_validate_model(pipe, X, y, cfg, positive_label="yes")

    n_folds_esperados = cfg.n_splits * cfg.n_repeats
    assert m.call_count == n_folds_esperados

    n_total = len(X)
    for llamada in m.call_args_list:
        x_visto = llamada.args[1]
        n_visto = x_visto.shape[0]
        # Cada llamada ve un fold de ENTRENAMIENTO (4/5 del dataset), nunca
        # el dataset completo: fuga de datos si `n_visto == n_total`.
        assert n_visto < n_total
        assert n_visto == pytest.approx(n_total * (cfg.n_splits - 1) / cfg.n_splits, abs=1)
