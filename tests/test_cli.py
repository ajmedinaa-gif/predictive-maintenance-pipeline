"""Tests del CLI."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from predictive_maintenance import __version__, cli, data, datasets
from predictive_maintenance.cli import app
from predictive_maintenance.config import get_settings

runner = CliRunner()


def test_version_command_prints_installed_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_command_prints_package_name():
    result = runner.invoke(app, ["version"])
    assert "predictive-maintenance-pipeline" in result.stdout


def test_cli_help_lists_commands():
    # `no_args_is_help=True` hace que Click salga con código 2 al invocar sin
    # argumentos (comportamiento estándar de Click, aunque igual imprime la
    # ayuda). Se invoca `--help` explícitamente para probar el camino feliz.
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.stdout
    assert "eda" in result.stdout


def test_eda_command_writes_report_and_figures(tmp_path, monkeypatch):
    """`pdm-cli eda --dataset lab180` debe salir en 0 y volcar el JSON y las figuras.

    Se redirige la salida a `tmp_path` mediante un `load_config` con las rutas
    de reports/figures absolutas; `data_raw` se deja apuntando al CSV real.
    """
    config_real = datasets.load_config()
    config_prueba = copy.deepcopy(config_real)
    raw_dir_real = datasets.PROJECT_ROOT / config_real["paths"]["data_raw"]
    config_prueba["paths"]["data_raw"] = str(raw_dir_real)
    config_prueba["paths"]["reports"] = str(tmp_path / "reports")
    config_prueba["paths"]["figures"] = str(tmp_path / "reports" / "figures")
    monkeypatch.setattr(datasets, "load_config", lambda: config_prueba)

    result = runner.invoke(app, ["eda", "--dataset", "lab180"])
    assert result.exit_code == 0, result.output

    ruta_json = tmp_path / "reports" / "eda_lab180.json"
    assert ruta_json.exists()
    informe = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert informe["dataset"] == "lab180"

    figuras_dir = tmp_path / "reports" / "figures" / "lab180"
    pngs = list(figuras_dir.glob("*.png"))
    assert len(pngs) == 4
    for png in pngs:
        assert png.stat().st_size > 0


def test_validate_command_quarantines_row_82_and_exits_1(tmp_path, monkeypatch):
    """`pdm-cli validate --dataset lab180` debe salir en 1: lab180 SIEMPRE
    tiene una fila en cuarentena a propósito (CLAUDE.md §6.3)."""
    config_real = datasets.load_config()
    config_prueba = copy.deepcopy(config_real)
    raw_dir_real = datasets.PROJECT_ROOT / config_real["paths"]["data_raw"]
    config_prueba["paths"]["data_raw"] = str(raw_dir_real)
    config_prueba["paths"]["reports"] = str(tmp_path / "reports")
    monkeypatch.setattr(datasets, "load_config", lambda: config_prueba)

    settings_prueba = get_settings().model_copy(deep=True)
    settings_prueba.paths.data_quarantine = tmp_path / "quarantine"
    monkeypatch.setattr(data, "get_settings", lambda: settings_prueba)

    result = runner.invoke(app, ["validate", "--dataset", "lab180"])
    assert result.exit_code == 1, result.output
    assert "cuarentena" in result.output.lower()
    assert "probablemente sintético" in result.output

    ruta_validacion = tmp_path / "reports" / "validation_lab180.json"
    assert ruta_validacion.exists()
    informe_validacion = json.loads(ruta_validacion.read_text(encoding="utf-8"))
    assert informe_validacion["n_cuarentena"] == 1

    ruta_plausibilidad = tmp_path / "reports" / "plausibility_lab180.json"
    assert ruta_plausibilidad.exists()
    informe_plausibilidad = json.loads(ruta_plausibilidad.read_text(encoding="utf-8"))
    assert informe_plausibilidad["veredicto"] == "probablemente sintético"

    assert list((tmp_path / "quarantine").glob("lab180_*.csv"))


def test_train_command_writes_metrics_report_results_table_and_figure(tmp_path, monkeypatch):
    """`pdm-cli train --dataset lab180` con un esquema de CV reducido, solo para
    probar el cableado del comando: la tabla de resultados real (5x10) la
    ejercen los tests de `test_evaluate.py` contra CLAUDE.md §8.1."""
    config_real = datasets.load_config()
    config_prueba = copy.deepcopy(config_real)
    raw_dir_real = datasets.PROJECT_ROOT / config_real["paths"]["data_raw"]
    config_prueba["paths"]["data_raw"] = str(raw_dir_real)
    config_prueba["paths"]["reports"] = str(tmp_path / "reports")
    config_prueba["paths"]["figures"] = str(tmp_path / "reports" / "figures")
    monkeypatch.setattr(datasets, "load_config", lambda: config_prueba)

    settings_prueba = get_settings().model_copy(deep=True)
    settings_prueba.paths.data_quarantine = tmp_path / "quarantine"
    settings_prueba.cross_validation["lab180"] = settings_prueba.cross_validation[
        "lab180"
    ].model_copy(update={"n_splits": 2, "n_repeats": 1})
    monkeypatch.setattr(data, "get_settings", lambda: settings_prueba)
    monkeypatch.setattr(cli, "get_settings", lambda: settings_prueba)

    result = runner.invoke(app, ["train", "--dataset", "lab180"])
    assert result.exit_code == 0, result.output
    assert "dummy_most_frequent" in result.output

    ruta_json = tmp_path / "reports" / "metrics_lab180.json"
    assert ruta_json.exists()
    informe = json.loads(ruta_json.read_text(encoding="utf-8"))
    nombres_modelo = list(informe["modelos"])
    assert nombres_modelo[0] == "dummy_most_frequent"
    assert nombres_modelo[1] == "dummy_stratified"
    assert "nested_cv" in informe

    ruta_md = tmp_path / "reports" / "results_lab180.md"
    assert ruta_md.exists()
    tabla = ruta_md.read_text(encoding="utf-8")
    assert tabla.strip().splitlines()[0].startswith("| modelo | PR-AUC")

    ruta_figura = tmp_path / "reports" / "figures" / "lab180" / "curvas_pr_roc.png"
    assert ruta_figura.exists()
    assert ruta_figura.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Fase 4: calibrate / explain / limits
# --------------------------------------------------------------------------- #


def _config_de_prueba(tmp_path):
    # Calcula `config_prueba` UNA VEZ, ANTES de parchear `load_config`: si el
    # lambda de `monkeypatch.setattr` llamara aquí dentro a
    # `datasets.load_config()` (ya parcheado a ese mismo lambda), sería
    # recursión infinita -- el bug que tenía la primera versión de este
    # helper.
    config_real = datasets.load_config()
    config_prueba = copy.deepcopy(config_real)
    raw_dir_real = datasets.PROJECT_ROOT / config_real["paths"]["data_raw"]
    config_prueba["paths"]["data_raw"] = str(raw_dir_real)
    config_prueba["paths"]["reports"] = str(tmp_path / "reports")
    config_prueba["paths"]["figures"] = str(tmp_path / "reports" / "figures")
    return config_prueba


def test_calibrate_command_writes_report_and_figures(tmp_path, monkeypatch):
    config_prueba = _config_de_prueba(tmp_path)
    monkeypatch.setattr(datasets, "load_config", lambda: config_prueba)

    result = runner.invoke(app, ["calibrate", "--dataset", "lab180"])
    assert result.exit_code == 0, result.output
    assert "logistic plain + Platt" in result.output

    ruta_json = tmp_path / "reports" / "calibration_lab180.json"
    assert ruta_json.exists()
    informe = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert informe["dataset"] == "lab180"
    assert set(informe["variantes"]) == {
        "logistic_balanced",
        "logistic_plain",
        "logistic_plain_platt",
        "logistic_plain_isotonic",
    }
    assert informe["umbral_teorico"] == pytest.approx(0.0556, abs=1e-3)

    figuras_dir = tmp_path / "reports" / "figures" / "lab180"
    for nombre in ("calibration_curve.png", "cost_vs_threshold.png"):
        ruta = figuras_dir / nombre
        assert ruta.exists()
        assert ruta.stat().st_size > 0


def test_explain_command_writes_report_and_figures(tmp_path, monkeypatch):
    config_prueba = _config_de_prueba(tmp_path)
    monkeypatch.setattr(datasets, "load_config", lambda: config_prueba)

    result = runner.invoke(app, ["explain", "--dataset", "lab180"])
    assert result.exit_code == 0, result.output
    assert "vibration_mm_s" in result.output

    ruta_json = tmp_path / "reports" / "explainability_lab180.json"
    assert ruta_json.exists()
    informe = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert len(informe["shap_casos_positivos"]) == 10
    assert informe["estabilidad_raiz_arbol"]["n_boot"] == 300

    figuras_dir = tmp_path / "reports" / "figures" / "lab180"
    for nombre in ("shap_beeswarm.png", "shap_waterfalls_positivos.png", "tree_root_stability.png"):
        ruta = figuras_dir / nombre
        assert ruta.exists()
        assert ruta.stat().st_size > 0


def test_limits_command_writes_report_and_figure(tmp_path, monkeypatch):
    config_prueba = _config_de_prueba(tmp_path)
    monkeypatch.setattr(datasets, "load_config", lambda: config_prueba)

    result = runner.invoke(app, ["limits", "--dataset", "lab180"])
    assert result.exit_code == 0, result.output
    assert "0.490" in result.output

    ruta_json = tmp_path / "reports" / "limits_lab180.json"
    assert ruta_json.exists()
    informe = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert informe["presupuesto_estadistico"]["margen_0.10"]["fallos_necesarios"] == 62
    assert informe["presupuesto_estadistico"]["margen_0.05"]["fallos_necesarios"] == 246

    ruta_figura = tmp_path / "reports" / "figures" / "lab180" / "learning_curve.png"
    assert ruta_figura.exists()
    assert ruta_figura.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Fase 5: download / run / compare
# --------------------------------------------------------------------------- #


def test_download_command_calls_adapter_download_lab180(monkeypatch):
    """`lab180` no toca la red: solo verifica que el fichero exista."""
    llamadas = []
    monkeypatch.setattr(
        datasets.Lab180Adapter, "download", lambda self: llamadas.append(1) or Path("x")
    )
    result = runner.invoke(app, ["download", "--dataset", "lab180"])
    assert result.exit_code == 0, result.output
    assert llamadas == [1]


def test_download_command_never_touches_network_for_ai4i2020(monkeypatch):
    """CLAUDE.md §2.11: la descarga es un paso explícito -- aquí se mockea para
    que el test no dependa de la red, no para probar que la red funciona."""
    llamadas = []

    def _fake_download(self):
        llamadas.append(1)
        return Path("data/raw/ai4i2020/ai4i2020.csv")

    monkeypatch.setattr(datasets.AI4I2020Adapter, "download", _fake_download)
    result = runner.invoke(app, ["download", "--dataset", "ai4i2020"])
    assert result.exit_code == 0, result.output
    assert llamadas == [1]
    assert "ai4i2020" in result.output


def test_download_command_unknown_dataset_fails():
    result = runner.invoke(app, ["download", "--dataset", "no_existe"])
    assert result.exit_code != 0


def test_run_command_validates_and_trains_lab180(tmp_path, monkeypatch):
    """`run` nunca sale con código 1 por cuarentena -- a diferencia de `validate`
    solo -- porque `lab180` cuarentena la fila 82 en cada ejecución por diseño."""
    config_prueba = _config_de_prueba(tmp_path)
    monkeypatch.setattr(datasets, "load_config", lambda: config_prueba)

    settings_prueba = get_settings().model_copy(deep=True)
    settings_prueba.paths.data_quarantine = tmp_path / "quarantine"
    settings_prueba.cross_validation["lab180"] = settings_prueba.cross_validation[
        "lab180"
    ].model_copy(update={"n_splits": 2, "n_repeats": 1})
    monkeypatch.setattr(data, "get_settings", lambda: settings_prueba)
    monkeypatch.setattr(cli, "get_settings", lambda: settings_prueba)

    result = runner.invoke(app, ["run", "--dataset", "lab180"])
    assert result.exit_code == 0, result.output
    assert "probablemente sintético" in result.output
    assert "dummy_most_frequent" in result.output

    assert (tmp_path / "reports" / "validation_lab180.json").exists()
    assert (tmp_path / "reports" / "plausibility_lab180.json").exists()
    assert (tmp_path / "reports" / "metrics_lab180.json").exists()
    assert (tmp_path / "reports" / "results_lab180.md").exists()


def _informe_metricas_sintetico(n_positivos: int, n_filas: int) -> dict:
    """El "mejor modelo" es a propósito `dummy_most_frequent`: el test no
    necesita un modelo real, solo que `compare` sepa recalcular su curva
    out-of-fold -- `DummyClassifier` es el ajuste más rápido posible."""
    return {
        "dataset": "x",
        "n_filas_entrenamiento": n_filas,
        "n_positivos": n_positivos,
        "modelos": {
            "logistic_plain": {
                "metricas": {
                    "average_precision": {"media": 0.05},
                    "accuracy": {"media": 1 - n_positivos / n_filas},
                },
                "matriz_confusion": {"tp": 0, "recall_wilson_ci": [0.0, 0.3]},
            },
            "dummy_most_frequent": {
                "metricas": {
                    "average_precision": {"media": 0.9},
                    "accuracy": {"media": 0.95},
                },
                "matriz_confusion": {"tp": n_positivos, "recall_wilson_ci": [0.5, 0.95]},
            },
        },
    }


def _informe_calibracion_sintetico() -> dict:
    return {
        "variantes": {
            "logistic_plain_platt": {"umbral_empirico": 0.1, "ahorro_pct": 50.0},
        }
    }


def test_compare_command_requires_existing_reports(tmp_path, monkeypatch):
    config_prueba = _config_de_prueba(tmp_path)
    monkeypatch.setattr(datasets, "load_config", lambda: config_prueba)

    result = runner.invoke(app, ["compare"])
    assert result.exit_code == 2, result.output
    assert "lab180" in result.output


def test_compare_command_writes_comparison_report_and_figure(tmp_path, monkeypatch):
    config_prueba = _config_de_prueba(tmp_path)
    monkeypatch.setattr(datasets, "load_config", lambda: config_prueba)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for nombre, n_pos, n_filas in (("lab180", 10, 179), ("ai4i2020", 339, 10000)):
        (reports_dir / f"metrics_{nombre}.json").write_text(
            json.dumps(_informe_metricas_sintetico(n_pos, n_filas)),
            encoding="utf-8",
        )
        (reports_dir / f"calibration_{nombre}.json").write_text(
            json.dumps(_informe_calibracion_sintetico()), encoding="utf-8"
        )

    # `compare` recalcula las curvas out-of-fold del "mejor modelo" sobre datos
    # reales: se fuerza a `dummy_most_frequent` (rápido, sin ajuste real) para
    # que el test no dependa de descargar `ai4i2020`.
    def _xy_sintetico(dataset):
        spec = datasets.get_spec(dataset)
        n = 60
        X = pd.DataFrame({col: range(n) for col in spec.feature_columns})
        y = pd.Series(["no"] * (n - 6) + ["yes"] * 6)
        return spec, None, X, y

    monkeypatch.setattr(cli, "_cargar_xy_validado", _xy_sintetico)

    result = runner.invoke(app, ["compare"])
    assert result.exit_code == 0, result.output

    ruta_json = reports_dir / "comparison.json"
    assert ruta_json.exists()
    informe = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert informe["datasets"]["lab180"]["mejor_modelo"] == "dummy_most_frequent"
    assert informe["datasets"]["ai4i2020"]["n_positivos"] == 339

    ruta_figura = tmp_path / "reports" / "figures" / "comparacion_pr.png"
    assert ruta_figura.exists()
    assert ruta_figura.stat().st_size > 0
