"""Tests del CLI."""

import copy
import json

from typer.testing import CliRunner

from predictive_maintenance import __version__, data, datasets
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
