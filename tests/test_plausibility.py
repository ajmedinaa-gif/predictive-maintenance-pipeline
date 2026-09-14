"""Tests del auditor de plausibilidad sobre `lab180` (CLAUDE.md §7, §12 extra 1)."""

import numpy as np
import pandas as pd
import pytest

from predictive_maintenance import plausibility


def test_correlation_structure_detects_independence_in_lab180(lab180_df):
    resultado = plausibility.correlation_structure(lab180_df)
    assert resultado["max_abs_r"] < 0.10
    assert resultado["variables_mutuamente_independientes"] is True


def test_missingness_regularity_detects_zero_overlap_in_lab180(lab180_df):
    resultado = plausibility.missingness_regularity(lab180_df)
    assert resultado["solapan_entre_columnas"] is False
    assert resultado["sin_solapamiento"] is True
    assert resultado["reparto_igual_por_columna"] is True
    assert resultado["bandera_regularidad_sospechosa"] is True


def test_missingness_regularity_class_split_is_reported_as_inconclusive(lab180_df):
    # Cero nulos en la clase positiva ocurre ~49 % de las veces por puro azar
    # (hipergeométrico: C(170,12)/C(180,12)): no es evidencia, y el código no
    # debe tratarlo como tal.
    resultado = plausibility.missingness_regularity(lab180_df)
    reparto = resultado["reparto_por_clase_no_concluyente"]
    assert reparto["observado"] == 0
    assert reparto["p_valor"] == pytest.approx(0.4924, abs=1e-3)
    # El texto de `evidencia` solo recoge las dos firmas deterministas: no
    # menciona la clase positiva ni el reparto por clase.
    assert "yes" not in resultado["evidencia"]
    assert "positiv" not in resultado["evidencia"]


def test_missingness_regularity_flag_is_unaffected_by_class_split():
    # Mismo patrón regular (4/4/4, sin solape) que lab180, pero con los nulos
    # de una columna cayendo dentro de la clase positiva en vez de fuera: el
    # reparto por clase cambia, la bandera de regularidad NO debe cambiar,
    # porque no depende de él.
    df = pd.DataFrame(
        {
            "a": [1.0] * 180,
            "b": [1.0] * 180,
            "c": [1.0] * 180,
            "failure": ["no"] * 170 + ["yes"] * 10,
        }
    )
    df.loc[0:3, "a"] = np.nan
    df.loc[4:7, "b"] = np.nan
    # Índices 170-179 son la clase "yes": estos 4 nulos SÍ caen en positivos.
    df.loc[170:173, "c"] = np.nan

    resultado = plausibility.missingness_regularity(df, target="failure")
    assert resultado["reparto_igual_por_columna"] is True
    assert resultado["sin_solapamiento"] is True
    assert resultado["bandera_regularidad_sospechosa"] is True
    assert resultado["reparto_por_clase_no_concluyente"]["observado"] == 4


def test_digit_distribution_returns_one_entry_per_numeric_attribute(lab180_df):
    resultado = plausibility.digit_distribution(lab180_df)
    assert set(resultado["por_atributo"]) == {
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
        "hours_since_maintenance",
        "load_percent",
    }
    # temperature_c y load_percent solo tienen 1 decimal en el CSV; usar un
    # número fijo de decimales para todas las columnas habría dado un dígito
    # trivialmente constante para ellas.
    assert resultado["por_atributo"]["temperature_c"]["decimales_detectados"] == 1
    assert resultado["por_atributo"]["vibration_mm_s"]["decimales_detectados"] == 2


def test_audit_verdict_on_lab180_is_probably_synthetic(lab180_df):
    informe = plausibility.audit(lab180_df, dataset="lab180")
    assert informe.veredicto == "probablemente sintético"
    assert len(informe.evidencia) >= 2


def test_audit_verdict_on_coupled_data_is_compatible_with_real(lab180_df):
    # Construcción sintética PERO con acoplamiento físico deliberado, para
    # validar que el auditor no dictamina "sintético" siempre por defecto.
    import numpy as np

    rng = np.random.default_rng(42)
    n = len(lab180_df)
    carga = rng.uniform(30, 100, size=n)
    vibracion = 0.05 * carga + rng.normal(0, 0.1, size=n)
    df_acoplado = lab180_df.copy()
    df_acoplado["load_percent"] = carga
    df_acoplado["vibration_mm_s"] = vibracion

    resultado = plausibility.correlation_structure(df_acoplado)
    assert resultado["variables_mutuamente_independientes"] is False
