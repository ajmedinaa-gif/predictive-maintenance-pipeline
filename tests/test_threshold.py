"""Tests del umbral de decisión por coste (CLAUDE.md §9.1, §9.3).

TEST CLAVE: `optimal_threshold` sobre un modelo simulado PERFECTAMENTE
calibrado debe converger al umbral teórico t* = 0.0556 de CLAUDE.md §9.1. La
fórmula incluye `C_TP`, que NO es cero en este proyecto: implementarla como
`C_FP / (C_FP + C_FN)` (ignorando `C_TP`) da 0.0476, un valor distinto y
equivocado — de ahí que el test compare explícitamente contra los dos.
"""

import numpy as np
import pytest

from predictive_maintenance import threshold

TOL = 1e-3

COSTS = {
    "cost_true_negative": 0.0,
    "cost_false_positive": 500_000.0,
    "cost_false_negative": 10_000_000.0,
    "cost_true_positive": 1_500_000.0,
}


def _perfectly_calibrated_sample(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Simula un modelo PERFECTAMENTE calibrado: `y_prob` es la probabilidad real.

    `y_prob` se muestrea uniforme en [0, 1] y `y_true` se genera como
    Bernoulli(y_prob): por construcción, entre los casos con `y_prob` próximo
    a cualquier valor `p`, la frecuencia observada de positivos es `p`. Es la
    definición de "perfectamente calibrado", sin pasar por ningún modelo real.
    """
    rng = np.random.default_rng(seed)
    y_prob = rng.uniform(0.0, 1.0, size=n)
    y_true = rng.binomial(1, y_prob)
    return y_true, y_prob


# --------------------------------------------------------------------------- #
# theoretical_threshold: la fórmula de CLAUDE.md §9.1, incluyendo C_TP
# --------------------------------------------------------------------------- #


def test_theoretical_threshold_matches_claude_md():
    t_star = threshold.theoretical_threshold(COSTS)
    assert t_star == pytest.approx(0.0556, abs=1e-3)


def test_theoretical_threshold_is_not_the_formula_without_c_tp():
    # C_FP / (C_FP + C_FN) = 0.0476: la fórmula incompleta que CLAUDE.md §9.1
    # advierte explícitamente que está mal. No deben coincidir.
    t_star = threshold.theoretical_threshold(COSTS)
    formula_incompleta = COSTS["cost_false_positive"] / (
        COSTS["cost_false_positive"] + COSTS["cost_false_negative"]
    )
    assert formula_incompleta == pytest.approx(0.0476, abs=1e-3)
    assert abs(t_star - formula_incompleta) > 1e-3


# --------------------------------------------------------------------------- #
# TEST CLAVE: optimal_threshold converge al teórico bajo calibración perfecta
# --------------------------------------------------------------------------- #


def test_optimal_threshold_converges_to_theoretical_under_perfect_calibration():
    y_true, y_prob = _perfectly_calibrated_sample(n=200_000, seed=42)
    t_hat = threshold.optimal_threshold(y_true, y_prob, COSTS)
    assert t_hat == pytest.approx(0.0556, abs=0.01)


# --------------------------------------------------------------------------- #
# El coste al óptimo nunca es peor que al umbral 0.5
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_optimal_threshold_cost_is_never_worse_than_default(seed):
    y_true, y_prob = _perfectly_calibrated_sample(n=500, seed=seed)
    t_hat = threshold.optimal_threshold(y_true, y_prob, COSTS)
    coste_optimo = threshold.expected_cost(y_true, y_prob, t_hat, COSTS)
    coste_default = threshold.expected_cost(y_true, y_prob, 0.5, COSTS)
    assert coste_optimo <= coste_default


# --------------------------------------------------------------------------- #
# Con C_FN == C_FP (y C_TP == C_TN == 0) el umbral óptimo es cercano a 0.5
# --------------------------------------------------------------------------- #


def test_symmetric_costs_give_threshold_near_half():
    costes_simetricos = {
        "cost_true_negative": 0.0,
        "cost_false_positive": 1_000_000.0,
        "cost_false_negative": 1_000_000.0,
        "cost_true_positive": 0.0,
    }
    assert threshold.theoretical_threshold(costes_simetricos) == pytest.approx(0.5, abs=1e-9)

    y_true, y_prob = _perfectly_calibrated_sample(n=200_000, seed=123)
    t_hat = threshold.optimal_threshold(y_true, y_prob, costes_simetricos)
    assert t_hat == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------------------------- #
# cost_curve y expected_cost: coherencia básica
# --------------------------------------------------------------------------- #


def test_cost_curve_matches_expected_cost_pointwise():
    y_true, y_prob = _perfectly_calibrated_sample(n=200, seed=5)
    umbrales = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    curva = threshold.cost_curve(y_true, y_prob, COSTS, thresholds=umbrales)
    for t in umbrales:
        esperado = threshold.expected_cost(y_true, y_prob, t, COSTS)
        obtenido = curva.loc[curva["threshold"] == t, "coste_total"].iloc[0]
        assert obtenido == pytest.approx(esperado)


def test_expected_cost_at_threshold_zero_is_all_positive_predictions():
    # threshold=0.0: todo se clasifica como fallo (y_prob >= 0.0 siempre).
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.6, 0.2])
    coste = threshold.expected_cost(y_true, y_prob, 0.0, COSTS)
    # 2 TP + 2 FP
    esperado = 2 * COSTS["cost_true_positive"] + 2 * COSTS["cost_false_positive"]
    assert coste == pytest.approx(esperado)


def test_expected_cost_at_threshold_above_one_is_all_negative_predictions():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.6, 0.2])
    coste = threshold.expected_cost(y_true, y_prob, 1.01, COSTS)
    # 2 FN + 2 TN
    esperado = 2 * COSTS["cost_false_negative"] + 2 * COSTS["cost_true_negative"]
    assert coste == pytest.approx(esperado)
