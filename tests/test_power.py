"""Tests del presupuesto estadístico (CLAUDE.md §10.3).

Los tres valores clave: para estimar un recall objetivo de 0.80, un margen de
±10 pp exige 62 fallos observados; ±5 pp exige 246; y 62 fallos, a la
prevalencia real de `lab180` (5.5556 %), corresponden a 1116 ciclos de
máquina.
"""

import pytest

from predictive_maintenance import power


def test_required_positives_ten_point_margin():
    assert power.required_positives(target_recall=0.80, margin=0.10) == 62


def test_required_positives_five_point_margin():
    assert power.required_positives(target_recall=0.80, margin=0.05) == 246


def test_required_positives_grows_as_margin_shrinks():
    assert power.required_positives(0.80, 0.05) > power.required_positives(0.80, 0.10)


def test_required_observations_matches_claude_md():
    assert power.required_observations(62, 5.55556 / 100) == 1116


def test_required_observations_uses_round_not_floor():
    # 62 / 0.0555556 = 1116.0004..., no 1115.
    assert power.required_observations(62, 0.0555556) == 1116


def test_recall_wilson_ci_is_reexported_from_evaluate():
    from predictive_maintenance import evaluate

    assert power.recall_wilson_ci is evaluate.recall_wilson_ci
    lower, upper = power.recall_wilson_ci(8, 10)
    assert lower == pytest.approx(0.490, abs=1e-3)
    assert upper == pytest.approx(0.943, abs=1e-3)


def test_learning_curve_pr_auc_decreases_and_has_wide_bands(lab180_df, lab180_spec):
    # CLAUDE.md §10.2: la curva real BAJA con más datos y tiene bandas anchas
    # (hasta +/-0.30). Este test protege la FORMA del hallazgo (no monótona
    # creciente, bandas amplias), no valores puntuales: se rompería con
    # cualquier cambio de versión de scikit-learn si fuera más estricto.
    from predictive_maintenance import data, schema

    resultado = data.load_validated(lab180_spec.path, schema.Lab180Schema, dataset_name="lab180")
    df = resultado.valid
    X = df.drop(columns=[lab180_spec.target])
    y = df[lab180_spec.target].astype(str)

    curva = power.learning_curve_pr_auc(X, y, seed=42)

    assert len(curva) >= 4
    assert (curva["pr_auc_desv"] > 0.15).any(), "se esperan bandas anchas, no una curva suave"
    # No es monótona creciente: en algún punto el PR-AUC medio baja al añadir
    # más datos de entrenamiento -- es el hallazgo de CLAUDE.md §10.2, no ruido
    # de la implementación.
    assert (curva["pr_auc_media"].diff().dropna() < 0).any()
