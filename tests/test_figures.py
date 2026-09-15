"""Tests del módulo de figuras: se genera un PNG no vacío, sin abrir ventana."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

from predictive_maintenance import calibration, data, explain, figures, schema, threshold


def test_backend_is_agg():
    # En macOS matplotlib intenta abrir ventanas y cuelga el CLI (CLAUDE.md
    # §14.4); importar `figures` debe haber fijado el backend sin interfaz.
    assert matplotlib.get_backend().lower() == "agg"


@pytest.mark.parametrize(
    "nombre_funcion",
    ["figure_histograms", "figure_boxplots", "figure_correlation", "figure_prevalence"],
)
def test_figure_functions_write_nonempty_png(lab180_df, lab180_spec, tmp_path, nombre_funcion):
    funcion = getattr(figures, nombre_funcion)
    ruta = funcion(lab180_df, lab180_spec, tmp_path / f"{nombre_funcion}.png")
    assert ruta.exists()
    assert ruta.stat().st_size > 0


def test_make_eda_figures_returns_four_existing_paths(lab180_df, lab180_spec, tmp_path):
    rutas = figures.make_eda_figures(lab180_df, lab180_spec, tmp_path / "figs")
    assert len(rutas) == 4
    nombres = {ruta.name for ruta in rutas}
    assert nombres == {
        "histogramas_por_clase.png",
        "boxplots_por_clase.png",
        "correlacion.png",
        "prevalencia.png",
    }
    for ruta in rutas:
        assert ruta.exists()
        assert ruta.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Fase 4: calibración, umbral por coste, explicabilidad y límites
# --------------------------------------------------------------------------- #


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


def test_figure_calibration_curve_writes_nonempty_png(lab180_valid_xy, lab180_spec, tmp_path):
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, size=179)
    y_true = rng.binomial(1, y_prob)
    curvas = {"logistic_plain": calibration.reliability_curve(y_true, y_prob, n_bins=10)}
    ruta = figures.figure_calibration_curve(curvas, lab180_spec, tmp_path / "calibration_curve.png")
    assert ruta.exists()
    assert ruta.stat().st_size > 0


def test_figure_cost_vs_threshold_writes_nonempty_png(lab180_spec, tmp_path):
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, size=179)
    y_true = rng.binomial(1, y_prob)
    curva = threshold.cost_curve(y_true, y_prob, COSTS)
    t_star = threshold.theoretical_threshold(COSTS)
    ruta = figures.figure_cost_vs_threshold(
        curva, 0.5, t_star, lab180_spec, tmp_path / "cost_vs_threshold.png"
    )
    assert ruta.exists()
    assert ruta.stat().st_size > 0


def test_figure_shap_beeswarm_writes_nonempty_png(lab180_valid_xy, lab180_spec, tmp_path):
    X, y = lab180_valid_xy
    shap_resultado = explain.shap_values_logistic(X, y, seed=42)
    ruta = figures.figure_shap_beeswarm(
        shap_resultado, X, lab180_spec, tmp_path / "shap_beeswarm.png"
    )
    assert ruta.exists()
    assert ruta.stat().st_size > 0


def test_figure_shap_waterfalls_positives_writes_nonempty_png(
    lab180_valid_xy, lab180_spec, tmp_path
):
    X, y = lab180_valid_xy
    shap_resultado = explain.shap_values_logistic(X, y, seed=42)
    ruta = figures.figure_shap_waterfalls_positives(
        shap_resultado, y, lab180_spec, tmp_path / "shap_waterfalls.png"
    )
    assert ruta.exists()
    assert ruta.stat().st_size > 0


def test_figure_tree_root_stability_writes_nonempty_png(lab180_valid_xy, lab180_spec, tmp_path):
    X, y = lab180_valid_xy
    stability = explain.tree_root_stability(X, y, n_boot=30, seed=42)
    ruta = figures.figure_tree_root_stability(
        stability, lab180_spec, tmp_path / "tree_root_stability.png"
    )
    assert ruta.exists()
    assert ruta.stat().st_size > 0


def test_figure_learning_curve_writes_nonempty_png(lab180_spec, tmp_path):
    curva = pd.DataFrame(
        {
            "n_entrenamiento": [40, 60, 80, 100],
            "pr_auc_media": [0.75, 0.74, 0.68, 0.63],
            "pr_auc_desv": [0.30, 0.31, 0.28, 0.27],
        }
    )
    ruta = figures.figure_learning_curve(curva, lab180_spec, tmp_path / "learning_curve.png")
    assert ruta.exists()
    assert ruta.stat().st_size > 0
