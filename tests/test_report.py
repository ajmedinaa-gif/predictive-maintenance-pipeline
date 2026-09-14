"""Tests del payload de resultados contra CLAUDE.md §6 (tolerancia 1e-3)."""

import json

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
