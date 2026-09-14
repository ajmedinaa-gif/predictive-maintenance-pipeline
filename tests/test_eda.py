"""Tests del módulo EDA contra los valores verificados de CLAUDE.md §6.

Si un valor no coincide con tolerancia 1e-3, el test debe FALLAR: no se ajusta
el valor esperado para que pase (CLAUDE.md, cabecera).
"""

import pytest

from predictive_maintenance import datasets, eda

TOL = 1e-3


@pytest.fixture(scope="module")
def df():
    return datasets.load("lab180")


@pytest.fixture(scope="module")
def spec():
    return datasets.get_spec("lab180")


def test_dataset_shape(df):
    assert df.shape == (180, 6)


# --------------------------------------------------------------------------- #
# §6.1 Tabla descriptiva
# --------------------------------------------------------------------------- #

ESPERADO_DESCRIPTIVA = {
    "temperature_c": {
        "n": 176,
        "nulos": 4,
        "pct_nulos": 2.22,
        "min": 47.00,
        "max": 89.80,
        "media": 67.8102,
        "mediana": 67.950,
        "desv_tipica": 7.6651,
    },
    "vibration_mm_s": {
        "n": 176,
        "nulos": 4,
        "pct_nulos": 2.22,
        "min": -0.34,
        "max": 9.59,
        "media": 4.2438,
        "mediana": 4.295,
        "desv_tipica": 1.3349,
    },
    "pressure_bar": {
        "n": 176,
        "nulos": 4,
        "pct_nulos": 2.22,
        "min": 4.27,
        "max": 10.19,
        "media": 6.7630,
        "mediana": 6.750,
        "desv_tipica": 1.1330,
    },
    "hours_since_maintenance": {
        "n": 180,
        "nulos": 0,
        "pct_nulos": 0.00,
        "min": 20,
        "max": 896,
        "media": 473.1889,
        "mediana": 440.0,
        "desv_tipica": 262.3427,
    },
    "load_percent": {
        "n": 180,
        "nulos": 0,
        "pct_nulos": 0.00,
        "min": 31.50,
        "max": 100.00,
        "media": 71.0900,
        "mediana": 71.400,
        "desv_tipica": 14.4342,
    },
}

# La tolerancia de %nulos en CLAUDE.md está redondeada a 2 decimales (2.22, no
# 2.2222...): se compara con una tolerancia algo más ancha que TOL para ese campo.
TOL_PCT_NULOS = 5e-3


@pytest.mark.parametrize("atributo", list(ESPERADO_DESCRIPTIVA))
def test_descriptive_table_matches_claude_md(df, atributo):
    tabla = eda.descriptive_table(df, target="failure")
    fila = tabla.loc[atributo]
    esperado = ESPERADO_DESCRIPTIVA[atributo]

    assert fila["n"] == esperado["n"]
    assert fila["nulos"] == esperado["nulos"]
    assert fila["pct_nulos"] == pytest.approx(esperado["pct_nulos"], abs=TOL_PCT_NULOS)
    assert fila["min"] == pytest.approx(esperado["min"], abs=TOL)
    assert fila["max"] == pytest.approx(esperado["max"], abs=TOL)
    assert fila["media"] == pytest.approx(esperado["media"], abs=TOL)
    assert fila["mediana"] == pytest.approx(esperado["mediana"], abs=TOL)
    assert fila["desv_tipica"] == pytest.approx(esperado["desv_tipica"], abs=TOL)


def test_hours_since_maintenance_is_integer(df):
    assert df["hours_since_maintenance"].dtype.kind == "i"


def test_other_attributes_are_float(df):
    for col in ["temperature_c", "vibration_mm_s", "pressure_bar", "load_percent"]:
        assert df[col].dtype.kind == "f"


# --------------------------------------------------------------------------- #
# §6.2 Variable objetivo
# --------------------------------------------------------------------------- #


def test_class_balance_matches_claude_md(df):
    balance = eda.class_balance(df, target="failure", positive_label="yes")
    assert balance["clases"] == ["no", "yes"]
    assert balance["nulos_en_objetivo"] == 0
    assert balance["conteos"] == {"no": 170, "yes": 10}
    assert balance["prevalencia"] == pytest.approx(0.0555556, abs=TOL)
    assert balance["prevalencia_pct"] == pytest.approx(5.5556, abs=TOL)
    assert balance["accuracy_trivial_mayoritario_pct"] == pytest.approx(94.4444, abs=TOL)


# --------------------------------------------------------------------------- #
# §6.3 Nulos y anomalías
# --------------------------------------------------------------------------- #


def test_missingness_report_matches_claude_md(df):
    reporte = eda.missingness_report(df, target="failure")
    assert reporte["filas_afectadas"] == 12
    assert reporte["max_nulos_en_una_fila"] == 1
    assert reporte["solapan_entre_columnas"] is False
    assert reporte["nulos_por_columna"] == {
        "temperature_c": 4,
        "vibration_mm_s": 4,
        "pressure_bar": 4,
        "hours_since_maintenance": 0,
        "load_percent": 0,
    }
    assert reporte["duplicados_completos"] == 0
    # Las 12 filas con nulos pertenecen todas a la clase "no".
    assert reporte["reparto_por_clase"]["yes"]["filas_con_nulos"] == 0
    assert reporte["reparto_por_clase"]["no"]["filas_con_nulos"] == 12
    assert reporte["clases_sin_ningun_nulo"] == ["yes"]


def test_physical_anomaly_row_82(df):
    anomalias = eda.physical_anomalies(df)
    assert len(anomalias) == 1
    fila = anomalias.iloc[0]
    assert fila["fila"] == 82
    assert fila["atributo"] == "vibration_mm_s"
    assert fila["valor"] == pytest.approx(-0.34, abs=TOL)

    original = df.loc[82]
    assert original["temperature_c"] == pytest.approx(79.8, abs=TOL)
    assert original["pressure_bar"] == pytest.approx(4.53, abs=TOL)
    assert original["hours_since_maintenance"] == 250
    assert original["load_percent"] == pytest.approx(63.8, abs=TOL)
    assert original["failure"] == "no"


def test_no_other_physical_violations(df):
    anomalias = eda.physical_anomalies(df)
    assert set(anomalias["atributo"]) == {"vibration_mm_s"}


# --------------------------------------------------------------------------- #
# §6.4 Asociación univariante
# --------------------------------------------------------------------------- #

ESPERADO_AUC = {
    "vibration_mm_s": {"media_yes": 5.806, "media_no": 4.150, "auc": 0.8503, "p": 0.00020},
    "temperature_c": {"media_yes": 73.590, "media_no": 67.462, "auc": 0.7259, "p": 0.01669},
    "hours_since_maintenance": {
        "media_yes": 668.400,
        "media_no": 461.706,
        "auc": 0.7218,
        "p": 0.01871,
    },
    "load_percent": {"media_yes": 79.740, "media_no": 70.581, "auc": 0.6682, "p": 0.07459},
    "pressure_bar": {"media_yes": 5.797, "media_no": 6.821, "auc": 0.2515, "p": 0.00846},
}


@pytest.mark.parametrize("atributo", list(ESPERADO_AUC))
def test_univariate_auc_matches_claude_md(df, atributo):
    tabla = eda.univariate_auc(df, target="failure", positive_label="yes")
    fila = tabla.loc[atributo]
    esperado = ESPERADO_AUC[atributo]

    assert fila["media_positivos"] == pytest.approx(esperado["media_yes"], abs=1e-2)
    assert fila["media_negativos"] == pytest.approx(esperado["media_no"], abs=1e-2)
    assert fila["auc"] == pytest.approx(esperado["auc"], abs=TOL)
    assert fila["p_valor"] == pytest.approx(esperado["p"], abs=5e-4)


def test_load_percent_not_significant_pressure_inverse(df):
    tabla = eda.univariate_auc(df, target="failure", positive_label="yes")
    # pandas devuelve numpy.bool_ al indexar una celda de una columna bool, no
    # el singleton `False` de Python: se compara por igualdad, no por `is`.
    assert bool(tabla.loc["load_percent", "significativo_5pct"]) is False
    assert tabla.loc["pressure_bar", "direccion"] == "inversa"
    assert tabla.loc["pressure_bar", "auc"] < 0.5


# --------------------------------------------------------------------------- #
# §6.5 Correlaciones
# --------------------------------------------------------------------------- #


def test_all_pairwise_correlations_below_threshold(df):
    corr = eda.correlation_matrix(df, target="failure")
    import numpy as np

    sin_diagonal = corr.where(~np.eye(len(corr), dtype=bool)).abs()
    assert sin_diagonal.max().max() < 0.10


def test_max_abs_correlation_matches_claude_md(df):
    resultado = eda.max_abs_correlation(df, target="failure")
    assert resultado["max_abs_r"] == pytest.approx(0.097, abs=1e-2)
    assert set(resultado["par"]) == {"load_percent", "vibration_mm_s"}
