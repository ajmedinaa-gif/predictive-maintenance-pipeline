"""Tests del auditor de plausibilidad sobre `lab180` (CLAUDE.md §7, §12 extra 1)."""

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
