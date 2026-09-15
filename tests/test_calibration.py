"""Tests de calibración (CLAUDE.md §9.2): fiabilidad, descomposición de Brier, cuatro variantes."""

import numpy as np
import pytest

from predictive_maintenance import calibration, data, schema


@pytest.fixture(scope="module")
def lab180_valid_xy(lab180_spec):
    resultado = data.load_validated(lab180_spec.path, schema.Lab180Schema, dataset_name="lab180")
    df = resultado.valid
    X = df.drop(columns=[lab180_spec.target])
    y = df[lab180_spec.target].astype(str)
    return X, y


COSTS = {
    "cost_true_negative": 0.0,
    "cost_false_positive": 500_000.0,
    "cost_false_negative": 10_000_000.0,
    "cost_true_positive": 1_500_000.0,
}


# --------------------------------------------------------------------------- #
# reliability_curve
# --------------------------------------------------------------------------- #


def test_reliability_curve_has_n_bins_rows_and_covers_zero_to_one():
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, size=200)
    y_true = rng.binomial(1, y_prob)
    curva = calibration.reliability_curve(y_true, y_prob, n_bins=10)
    assert len(curva) == 10
    assert curva["limite_inferior"].iloc[0] == pytest.approx(0.0)
    assert curva["limite_superior"].iloc[-1] == pytest.approx(1.0)
    assert curva["n"].sum() == 200


def test_reliability_curve_empty_bin_is_nan_not_dropped():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.06, 0.95, 0.96])  # nada en el medio del rango
    curva = calibration.reliability_curve(y_true, y_prob, n_bins=10)
    assert len(curva) == 10
    bins_vacios = curva[curva["n"] == 0]
    assert not bins_vacios.empty
    assert bins_vacios["prob_media_predicha"].isna().all()


def test_perfectly_calibrated_model_has_near_zero_reliability_term():
    y_true, y_prob = _perfectly_calibrated_sample(n=50_000, seed=1)
    descomposicion = calibration.brier_decomposition(y_true, y_prob, n_bins=10)
    assert descomposicion["fiabilidad"] < 0.001


def test_brier_decomposition_reconstructs_actual_brier_score():
    y_true, y_prob = _perfectly_calibrated_sample(n=20_000, seed=2)
    descomposicion = calibration.brier_decomposition(y_true, y_prob, n_bins=10)
    assert descomposicion["reconstruccion"] == pytest.approx(
        descomposicion["brier_score"], abs=0.01
    )


def _perfectly_calibrated_sample(n: int, seed: int):
    rng = np.random.default_rng(seed)
    y_prob = rng.uniform(0.0, 1.0, size=n)
    y_true = rng.binomial(1, y_prob)
    return y_true, y_prob


# --------------------------------------------------------------------------- #
# evaluate_cost_variants: las cuatro variantes de CLAUDE.md §9.2
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def variantes(lab180_valid_xy):
    X, y = lab180_valid_xy
    return calibration.evaluate_cost_variants(
        X,
        y,
        COSTS,
        seed=42,
        outer_splits=5,
        threshold_inner_splits=3,
        calibration_cv=3,
    )


def test_evaluate_cost_variants_returns_the_four_variants(variantes):
    assert set(variantes) == {
        "logistic_balanced",
        "logistic_plain",
        "logistic_plain_platt",
        "logistic_plain_isotonic",
    }


def test_confusion_matrices_sum_to_dataset_size(lab180_valid_xy, variantes):
    X, _y = lab180_valid_xy
    for resultado in variantes.values():
        matriz = resultado["matriz_confusion"]
        assert matriz["tn"] + matriz["fp"] + matriz["fn"] + matriz["tp"] == len(X)


def test_confusion_matrices_have_ten_positives(variantes):
    for resultado in variantes.values():
        matriz = resultado["matriz_confusion"]
        assert matriz["fn"] + matriz["tp"] == 10


def test_evaluate_cost_variants_is_reproducible(lab180_valid_xy):
    X, y = lab180_valid_xy
    r1 = calibration.evaluate_cost_variants(X, y, COSTS, seed=42)
    r2 = calibration.evaluate_cost_variants(X, y, COSTS, seed=42)
    for nombre in r1:
        assert r1[nombre]["brier_score"] == pytest.approx(r2[nombre]["brier_score"])
        assert r1[nombre]["umbral_empirico"] == pytest.approx(r2[nombre]["umbral_empirico"])
