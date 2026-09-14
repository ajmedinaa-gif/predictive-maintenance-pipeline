"""Tests del payload de resultados contra CLAUDE.md §6 (tolerancia 1e-3)."""

import json

import pandas as pd
import pytest

from predictive_maintenance import report

TOL = 1e-3


@pytest.fixture(scope="module")
def informe(lab180_df, lab180_spec):
    return report.build_eda_report(lab180_df, lab180_spec)


def test_build_eda_report_top_level_keys(informe):
    esperadas = {
        "dataset",
        "fichero",
        "datos_simulados",
        "n_filas",
        "n_columnas",
        "objetivo",
        "clase_positiva",
        "balance_de_clases",
        "descriptiva",
        "nulos",
        "anomalias_fisicas",
        "rangos_fisicos",
        "asociacion_univariante",
        "correlaciones",
    }
    assert esperadas <= informe.keys()


def test_build_eda_report_identifica_el_dataset(informe):
    assert informe["dataset"] == "lab180"
    assert informe["datos_simulados"] is True
    assert informe["n_filas"] == 180
    assert informe["n_columnas"] == 6
    assert informe["objetivo"] == "failure"
    assert informe["clase_positiva"] == "yes"


def test_build_eda_report_balance_de_clases_matches_claude_md(informe):
    balance = informe["balance_de_clases"]
    assert balance["conteos"] == {"no": 170, "yes": 10}
    assert balance["prevalencia_pct"] == pytest.approx(5.5556, abs=TOL)
    assert balance["accuracy_trivial_mayoritario_pct"] == pytest.approx(94.4444, abs=TOL)


def test_build_eda_report_descriptiva_matches_claude_md(informe):
    filas = {fila["atributo"]: fila for fila in informe["descriptiva"]}
    assert filas["vibration_mm_s"]["min"] == pytest.approx(-0.34, abs=TOL)
    assert filas["vibration_mm_s"]["media"] == pytest.approx(4.2438, abs=TOL)
    assert filas["vibration_mm_s"]["desv_tipica"] == pytest.approx(1.3349, abs=TOL)
    assert filas["temperature_c"]["media"] == pytest.approx(67.8102, abs=TOL)
    assert filas["hours_since_maintenance"]["max"] == pytest.approx(896, abs=TOL)
    assert filas["load_percent"]["mediana"] == pytest.approx(71.400, abs=TOL)


def test_build_eda_report_anomalia_fila_82(informe):
    anomalias = informe["anomalias_fisicas"]
    assert len(anomalias) == 1
    assert anomalias[0]["fila"] == 82
    assert anomalias[0]["atributo"] == "vibration_mm_s"
    assert anomalias[0]["valor"] == pytest.approx(-0.34, abs=TOL)


def test_build_eda_report_univariate_auc_matches_claude_md(informe):
    filas = {fila["atributo"]: fila for fila in informe["asociacion_univariante"]}
    assert filas["vibration_mm_s"]["auc"] == pytest.approx(0.8503, abs=TOL)
    assert filas["pressure_bar"]["auc"] == pytest.approx(0.2515, abs=TOL)
    assert filas["pressure_bar"]["direccion"] == "inversa"


def test_build_eda_report_max_correlation_matches_claude_md(informe):
    maximo = informe["correlaciones"]["maxima_absoluta"]
    assert maximo["max_abs_r"] == pytest.approx(0.097, abs=1e-2)
    assert set(maximo["par"]) == {"load_percent", "vibration_mm_s"}


def test_write_eda_report_produces_valid_readable_json(informe, tmp_path):
    ruta = report.write_eda_report(informe, tmp_path / "sub" / "eda_lab180.json")
    assert ruta.exists()

    releido = json.loads(ruta.read_text(encoding="utf-8"))
    assert releido["dataset"] == "lab180"
    assert releido["balance_de_clases"]["conteos"] == {"no": 170, "yes": 10}
    assert releido == informe


# --------------------------------------------------------------------------- #
# Fase 3: build_metrics_report / render_results_markdown
# --------------------------------------------------------------------------- #


def _folds_sinteticos(valores_ap: list[float]) -> pd.DataFrame:
    n = len(valores_ap)
    return pd.DataFrame(
        {
            "fold": range(n),
            "n_test": [36] * n,
            "n_positivos_test": [2] * n,
            "average_precision": valores_ap,
            "roc_auc": [0.9] * n,
            "balanced_accuracy": [0.8] * n,
            "recall": [0.7] * n,
            "precision": [0.5] * n,
            "f1": [0.6] * n,
            "accuracy": [0.9] * n,
            "brier_score_loss": [0.05] * n,
        }
    )


@pytest.fixture
def payload_sintetico():
    resultados = {
        "dummy_most_frequent": {
            "folds": _folds_sinteticos([0.0556] * 5),
            "matriz_confusion": {"tn": 170, "fp": 0, "fn": 10, "tp": 0},
        },
        "logistic_balanced": {
            "folds": _folds_sinteticos([0.60, 0.65, 0.70, 0.68, 0.62]),
            "matriz_confusion": {"tn": 156, "fp": 14, "fn": 2, "tp": 8},
        },
    }
    return report.build_metrics_report(
        dataset="lab180",
        n_filas=179,
        n_positivos=10,
        resultados_por_modelo=resultados,
        protocolo_principal="RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)",
        protocolo_matriz_confusion="StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        seed=42,
    )


def test_build_metrics_report_preserves_model_order(payload_sintetico):
    # `dummy_most_frequent` primero: CLAUDE.md §2.1. El orden lo decide quien
    # construye `resultados_por_modelo`; este test confirma que se conserva.
    assert list(payload_sintetico["modelos"]) == ["dummy_most_frequent", "logistic_balanced"]


def test_build_metrics_report_computes_mean_and_bootstrap_ci(payload_sintetico):
    metricas = payload_sintetico["modelos"]["logistic_balanced"]["metricas"]
    assert metricas["average_precision"]["media"] == pytest.approx(0.65, abs=1e-6)
    lower, upper = metricas["average_precision"]["ic_bootstrap_95"]
    assert lower <= 0.65 <= upper


def test_build_metrics_report_includes_confusion_matrix(payload_sintetico):
    matriz = payload_sintetico["modelos"]["logistic_balanced"]["matriz_confusion"]
    assert matriz == {"tn": 156, "fp": 14, "fn": 2, "tp": 8}


def test_render_results_markdown_lists_dummy_first(payload_sintetico):
    tabla = report.render_results_markdown(payload_sintetico)
    lineas = tabla.strip().splitlines()
    assert lineas[0].startswith("| modelo | PR-AUC | ROC-AUC")
    assert "dummy_most_frequent" in lineas[2]
    assert "logistic_balanced" in lineas[3]


def test_write_metrics_report_round_trips(payload_sintetico, tmp_path):
    ruta = report.write_metrics_report(payload_sintetico, tmp_path / "metrics_lab180.json")
    releido = json.loads(ruta.read_text(encoding="utf-8"))
    assert releido == payload_sintetico
