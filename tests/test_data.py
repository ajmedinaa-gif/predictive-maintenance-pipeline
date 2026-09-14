"""Tests de `load_validated` y la cuarentena (CLAUDE.md §2.9, §6.3)."""

import pandas as pd
import pandera.pandas as pa
import pytest

from predictive_maintenance import data, schema
from predictive_maintenance.config import get_settings


def _settings_con_cuarentena_en(tmp_path):
    """Copia de la configuración real, redirigiendo `data_quarantine` a `tmp_path`.

    Así los tests nunca escriben en `data/quarantine/` del repositorio.
    """
    settings = get_settings().model_copy(deep=True)
    settings.paths.data_quarantine = tmp_path
    return settings


def test_row_82_is_quarantined_with_correct_reason_and_is_the_only_one(lab180_spec, tmp_path):
    settings = _settings_con_cuarentena_en(tmp_path)

    resultado = data.load_validated(
        lab180_spec.path, schema.Lab180Schema, dataset_name="lab180", settings=settings
    )

    assert len(resultado.quarantined) == 1
    assert len(resultado.valid) == 179
    assert resultado.report["n_filas_leidas"] == 180
    assert resultado.report["n_validas"] == 179
    assert resultado.report["n_cuarentena"] == 1

    fila = resultado.quarantined.iloc[0]
    assert fila.name == 82
    assert "vibration_mm_s" in fila["quarantine_reason"]
    assert "-0.34" in fila["quarantine_reason"]

    # Se escribió exactamente un CSV de cuarentena, con la columna del motivo.
    ficheros = list(tmp_path.glob("lab180_*.csv"))
    assert len(ficheros) == 1
    cuarentena_en_disco = pd.read_csv(ficheros[0])
    assert len(cuarentena_en_disco) == 1
    assert "quarantine_reason" in cuarentena_en_disco.columns


def test_valid_dataframe_has_zero_quarantined_rows(lab180_df, tmp_path):
    settings = _settings_con_cuarentena_en(tmp_path)
    # El único motivo de rechazo en lab180 es la fila 82: al quitarla, el CSV
    # que se lee de disco debe pasar el contrato entero sin cuarentena.
    ruta_limpia = tmp_path / "lab180_limpio.csv"
    lab180_df.drop(index=82).to_csv(ruta_limpia, index=False)

    resultado = data.load_validated(
        ruta_limpia, schema.Lab180Schema, dataset_name="lab180_limpio", settings=settings
    )

    assert len(resultado.quarantined) == 0
    assert len(resultado.valid) == 179
    assert resultado.report["n_cuarentena"] == 0
    assert resultado.report["ruta_cuarentena"] is None
    assert list(tmp_path.glob("lab180_limpio_*.csv")) == []


def test_missing_column_raises_and_writes_no_quarantine_file(tmp_path):
    settings = _settings_con_cuarentena_en(tmp_path)
    ruta = tmp_path / "incompleto.csv"
    pd.DataFrame({"temperature_c": [70.0]}).to_csv(ruta, index=False)

    with pytest.raises(pa.errors.SchemaErrors):
        data.load_validated(ruta, schema.Lab180Schema, dataset_name="incompleto", settings=settings)

    # Un fallo estructural no genera cuarentena: no hay fila que poner en ella.
    assert list(tmp_path.glob("incompleto_*.csv")) == []
