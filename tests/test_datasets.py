"""Tests del protocolo `DatasetAdapter` (CLAUDE.md §13, Fase 5)."""

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from predictive_maintenance import datasets

ADAPTER_NAMES = sorted(datasets.ADAPTERS)


# --------------------------------------------------------------------------- #
# Ambos adaptadores cumplen el protocolo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("nombre", ADAPTER_NAMES)
def test_adapter_conforms_to_protocol(nombre):
    adapter = datasets.get_adapter(nombre)
    assert isinstance(adapter, datasets.DatasetAdapter)


@pytest.mark.parametrize("nombre", ADAPTER_NAMES)
def test_adapter_has_required_attributes(nombre):
    adapter = datasets.get_adapter(nombre)
    assert isinstance(adapter.name, str) and adapter.name == nombre
    assert isinstance(adapter.target_column, str) and adapter.target_column
    assert isinstance(adapter.positive_label, str) and adapter.positive_label
    assert isinstance(adapter.simulated, bool)
    assert len(adapter.feature_columns) > 0
    assert adapter.target_column not in adapter.feature_columns
    assert issubclass(adapter.schema, pa.DataFrameModel)


@pytest.mark.parametrize("nombre", ADAPTER_NAMES)
def test_get_spec_matches_adapter(nombre):
    spec = datasets.get_spec(nombre)
    adapter = datasets.get_adapter(nombre)
    assert spec.name == adapter.name
    assert spec.target == adapter.target_column
    assert spec.positive_label == adapter.positive_label
    assert spec.simulated == adapter.simulated
    assert spec.feature_columns == tuple(adapter.feature_columns)


def test_unknown_dataset_raises_key_error():
    with pytest.raises(KeyError):
        datasets.get_adapter("no_existe")


# --------------------------------------------------------------------------- #
# lab180: la identidad
# --------------------------------------------------------------------------- #


def test_lab180_engineer_features_is_identity(lab180_df):
    adapter = datasets.get_adapter("lab180")
    resultado = adapter.engineer_features(lab180_df)
    pd.testing.assert_frame_equal(resultado, lab180_df)


def test_lab180_is_flagged_as_simulated():
    assert datasets.get_adapter("lab180").simulated is True


# --------------------------------------------------------------------------- #
# ai4i2020: ingeniería de features física y dimensionalmente correcta
# --------------------------------------------------------------------------- #


@pytest.fixture
def ai4i2020_raw_sample():
    """Una muestra sintética con la FORMA del contrato de `ai4i2020`, sin red.

    No descarga nada (regla dura CLAUDE.md §2.11): construye a mano un puñado
    de filas con valores plausibles para probar `engineer_features` de forma
    aislada y determinista.
    """
    return pd.DataFrame(
        {
            "type": ["L", "M", "H", "L"],
            "air_temperature_k": [298.1, 298.2, 300.0, 295.3],
            "process_temperature_k": [308.6, 308.7, 310.0, 305.7],
            "rotational_speed_rpm": [1551, 1408, 1200, 2886],
            "torque_nm": [42.8, 46.3, 60.0, 3.8],
            "tool_wear_min": [0, 3, 200, 253],
            "machine_failure": [0, 0, 1, 0],
        }
    )


def test_ai4i2020_is_not_flagged_as_simulated():
    assert datasets.get_adapter("ai4i2020").simulated is False


def test_ai4i2020_engineer_features_adds_expected_columns(ai4i2020_raw_sample):
    adapter = datasets.get_adapter("ai4i2020")
    salida = adapter.engineer_features(ai4i2020_raw_sample)
    for columna in adapter.feature_columns:
        assert columna in salida.columns


def test_ai4i2020_power_w_is_torque_times_angular_speed(ai4i2020_raw_sample):
    adapter = datasets.get_adapter("ai4i2020")
    salida = adapter.engineer_features(ai4i2020_raw_sample)
    esperado = (
        ai4i2020_raw_sample["torque_nm"]
        * ai4i2020_raw_sample["rotational_speed_rpm"]
        * 2
        * np.pi
        / 60
    )
    np.testing.assert_allclose(salida["power_w"].to_numpy(), esperado.to_numpy())


def test_ai4i2020_power_w_is_plausible_for_industrial_machinery(ai4i2020_raw_sample):
    """Watts, no kilovatios ni milivatios: el orden de magnitud correcto para
    un motor pequeño-mediano de taller es de cientos a pocos miles de vatios."""
    adapter = datasets.get_adapter("ai4i2020")
    salida = adapter.engineer_features(ai4i2020_raw_sample)
    assert (salida["power_w"] > 100).all()
    assert (salida["power_w"] < 50_000).all()


def test_ai4i2020_temp_delta_k_is_process_minus_air(ai4i2020_raw_sample):
    adapter = datasets.get_adapter("ai4i2020")
    salida = adapter.engineer_features(ai4i2020_raw_sample)
    esperado = (
        ai4i2020_raw_sample["process_temperature_k"] - ai4i2020_raw_sample["air_temperature_k"]
    )
    np.testing.assert_allclose(salida["temp_delta_k"].to_numpy(), esperado.to_numpy())


def test_ai4i2020_wear_x_torque_is_wear_times_torque(ai4i2020_raw_sample):
    adapter = datasets.get_adapter("ai4i2020")
    salida = adapter.engineer_features(ai4i2020_raw_sample)
    esperado = ai4i2020_raw_sample["tool_wear_min"] * ai4i2020_raw_sample["torque_nm"]
    np.testing.assert_allclose(salida["wear_x_torque"].to_numpy(), esperado.to_numpy())


def test_ai4i2020_type_one_hot_is_exclusive_and_exhaustive(ai4i2020_raw_sample):
    adapter = datasets.get_adapter("ai4i2020")
    salida = adapter.engineer_features(ai4i2020_raw_sample)
    suma = salida["type_L"] + salida["type_M"] + salida["type_H"]
    assert (suma == 1).all()


def test_ai4i2020_target_is_remapped_to_no_yes(ai4i2020_raw_sample):
    """Nunca "0"/"1": ver el comentario de `AI4I2020Adapter.positive_label`."""
    adapter = datasets.get_adapter("ai4i2020")
    salida = adapter.engineer_features(ai4i2020_raw_sample)
    assert set(salida["machine_failure"].unique()) <= {"no", "yes"}
    assert salida["machine_failure"].tolist() == ["no", "no", "yes", "no"]


def test_ai4i2020_engineered_features_are_all_numeric(ai4i2020_raw_sample):
    adapter = datasets.get_adapter("ai4i2020")
    salida = adapter.engineer_features(ai4i2020_raw_sample)
    X = salida[list(adapter.feature_columns)]
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)


def test_ai4i2020_schema_rejects_out_of_range_temperature(ai4i2020_raw_sample):
    """280-320 K es el rango plausible declarado (CLAUDE.md §13); 400 K no lo es."""
    df = ai4i2020_raw_sample.copy()
    df["air_temperature_k"] = 400.0
    for col in ("twf", "hdf", "pwf", "osf", "rnf"):
        df[col] = 0
    adapter = datasets.get_adapter("ai4i2020")
    with pytest.raises(pa.errors.SchemaErrors):
        adapter.schema.validate(df, lazy=True)


def test_ai4i2020_schema_rejects_unknown_type_category(ai4i2020_raw_sample):
    df = ai4i2020_raw_sample.copy()
    df["type"] = "X"
    for col in ("twf", "hdf", "pwf", "osf", "rnf"):
        df[col] = 0
    adapter = datasets.get_adapter("ai4i2020")
    with pytest.raises(pa.errors.SchemaErrors):
        adapter.schema.validate(df, lazy=True)
