"""Tests del contrato de datos de `lab180` (CLAUDE.md §6.3, §2.9)."""

import pandas as pd
import pandera.pandas as pa
import pytest
from config.config import QualityThresholds
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from predictive_maintenance import schema

# --------------------------------------------------------------------------- #
# La fila 82 de lab180 real: única anomalía física conocida (CLAUDE.md §6.3)
# --------------------------------------------------------------------------- #


def test_lab180_schema_flags_only_the_known_anomaly(lab180_df):
    with pytest.raises(pa.errors.SchemaErrors) as excinfo:
        schema.Lab180Schema.validate(lab180_df, lazy=True)

    casos = excinfo.value.failure_cases
    assert len(casos) == 1
    fila = casos.iloc[0]
    assert fila["column"] == "vibration_mm_s"
    assert fila["index"] == 82
    assert fila["failure_case"] == pytest.approx(-0.34, abs=1e-6)


def test_lab180_schema_passes_once_the_anomaly_is_removed(lab180_df):
    df_limpio = lab180_df.drop(index=82)
    # No debe lanzar.
    schema.Lab180Schema.validate(df_limpio, lazy=True)


# --------------------------------------------------------------------------- #
# El contrato falla si falta una columna o el tipo es incorrecto
# --------------------------------------------------------------------------- #


def test_schema_fails_if_a_required_column_is_missing():
    df = pd.DataFrame({"temperature_c": [70.0]})
    with pytest.raises(pa.errors.SchemaErrors):
        schema.Lab180Schema.validate(df, lazy=True)


def test_schema_fails_if_a_column_has_the_wrong_type():
    df = pd.DataFrame(
        {
            "temperature_c": ["no es un número"],
            "vibration_mm_s": [1.0],
            "pressure_bar": [6.0],
            "hours_since_maintenance": [10],
            "load_percent": [50.0],
            "failure": ["no"],
        }
    )
    with pytest.raises(pa.errors.SchemaErrors):
        schema.Lab180Schema.validate(df, lazy=True)


def test_failure_only_accepts_yes_no():
    df = pd.DataFrame(
        {
            "temperature_c": [70.0],
            "vibration_mm_s": [1.0],
            "pressure_bar": [6.0],
            "hours_since_maintenance": [10],
            "load_percent": [50.0],
            "failure": ["tal_vez"],
        }
    )
    with pytest.raises(pa.errors.SchemaErrors):
        schema.Lab180Schema.validate(df, lazy=True)


# --------------------------------------------------------------------------- #
# Cualquier dataframe dentro de los rangos válidos pasa el contrato
# --------------------------------------------------------------------------- #

_CAMPOS_VALIDOS = st.fixed_dictionaries(
    {
        "temperature_c": st.floats(
            min_value=-273.15, max_value=300.0, allow_nan=False, allow_infinity=False
        ),
        "vibration_mm_s": st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        "pressure_bar": st.floats(
            min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        "hours_since_maintenance": st.integers(min_value=0, max_value=100_000),
        "load_percent": st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        "failure": st.sampled_from(["yes", "no"]),
    }
)


@st.composite
def _dataframes_validos(draw):
    filas = draw(st.lists(_CAMPOS_VALIDOS, min_size=1, max_size=20))
    return pd.DataFrame(filas)


@given(df=_dataframes_validos())
@hypothesis_settings(max_examples=50, deadline=None)
def test_any_dataframe_within_valid_ranges_passes_the_contract(df):
    # No debe lanzar para ningún dataframe generado dentro de los rangos físicos.
    schema.Lab180Schema.validate(df, lazy=True)


# --------------------------------------------------------------------------- #
# Avisos de calidad: nunca excepción, siempre WARNING
# --------------------------------------------------------------------------- #

UMBRALES = QualityThresholds(min_prevalence=0.01, max_prevalence=0.50, max_null_pct_per_column=0.10)


def test_quality_warnings_empty_for_lab180(lab180_df):
    # lab180 (sin la fila en cuarentena) no dispara ningún aviso de calidad.
    avisos = schema.quality_warnings(lab180_df.drop(index=82), UMBRALES)
    assert avisos == []


def test_quality_warnings_flags_low_prevalence():
    df = pd.DataFrame({"failure": ["no"] * 199 + ["yes"]})
    avisos = schema.quality_warnings(df, UMBRALES, target="failure", positive_label="yes")
    assert any("prevalencia" in aviso for aviso in avisos)


def test_quality_warnings_flags_high_null_percentage():
    df = pd.DataFrame(
        {
            "temperature_c": [70.0] * 5 + [None] * 5,
            "failure": ["no"] * 9 + ["yes"],
        }
    )
    avisos = schema.quality_warnings(df, UMBRALES, target="failure", positive_label="yes")
    assert any("temperature_c" in aviso and "nulos" in aviso for aviso in avisos)


def test_quality_warnings_flags_complete_duplicates():
    df = pd.DataFrame(
        {
            "temperature_c": [70.0, 70.0, 65.0],
            "failure": ["no", "no", "yes"],
        }
    )
    avisos = schema.quality_warnings(df, UMBRALES, target="failure", positive_label="yes")
    assert any("duplicadas" in aviso for aviso in avisos)
