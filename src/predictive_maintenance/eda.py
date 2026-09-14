"""Análisis exploratorio reproducible.

Funciones puras: entran DataFrames, salen DataFrames o diccionarios. Ninguna
escribe en disco ni imprime nada; de eso se encarga `cli.py`. Así cada número
del repositorio es testeable contra los valores verificados de CLAUDE.md §6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Rangos físicamente admisibles por sensor. No son rangos "esperados" ni
# percentiles: son lo que la física permite. Un valor fuera de aquí no se imputa,
# va a cuarentena (CLAUDE.md §6.3).
PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    # Amplitud RMS de vibración: una amplitud no puede ser negativa.
    "vibration_mm_s": (0.0, 100.0),
    # Presión absoluta de línea: no puede ser negativa.
    "pressure_bar": (0.0, 1000.0),
    # Temperatura de un equipo en servicio, en °C.
    "temperature_c": (-273.15, 300.0),
    # Horas transcurridas desde la última intervención: no puede ser negativo.
    "hours_since_maintenance": (0.0, 100_000.0),
    # Carga expresada en porcentaje de la nominal.
    "load_percent": (0.0, 100.0),
}


def numeric_attributes(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """Columnas numéricas del DataFrame, excluyendo el objetivo si es numérico."""
    cols = df.select_dtypes(include="number").columns.tolist()
    if target is not None and target in cols:
        cols.remove(target)
    return cols


def descriptive_table(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Tabla descriptiva por atributo numérico.

    Devuelve n, nulos, %nulos, mínimo, máximo, media, mediana, desviación típica
    (ddof=1) y asimetría. El índice son los nombres de atributo.
    """
    n_filas = len(df)
    filas = []
    for col in numeric_attributes(df, target):
        s = df[col]
        n_nulos = int(s.isna().sum())
        filas.append(
            {
                "atributo": col,
                "n": int(s.notna().sum()),
                "nulos": n_nulos,
                "pct_nulos": 100.0 * n_nulos / n_filas if n_filas else np.nan,
                "min": float(s.min()),
                "max": float(s.max()),
                "media": float(s.mean()),
                "mediana": float(s.median()),
                "desv_tipica": float(s.std()),
                "asimetria": float(s.skew()),
            }
        )
    return pd.DataFrame(filas).set_index("atributo")


def missingness_report(df: pd.DataFrame, target: str = "failure") -> dict:
    """Radiografía de los nulos: dónde están, si se solapan y a qué clase pertenecen.

    Responde con datos a tres preguntas: ¿cuántos faltan y dónde?, ¿se concentran
    en las mismas filas?, ¿están repartidos igual entre las clases?
    """
    attrs = numeric_attributes(df, target)
    nulos_por_columna = {col: int(df[col].isna().sum()) for col in attrs}
    nulos_por_fila = df[attrs].isna().sum(axis=1)
    conteo_por_fila = nulos_por_fila[nulos_por_fila > 0].value_counts().sort_index()
    filas_por_numero_de_nulos = {int(k): int(v) for k, v in conteo_por_fila.items()}
    max_por_fila = int(nulos_por_fila.max()) if len(df) else 0
    filas_afectadas = int((nulos_por_fila > 0).sum())

    reparto_por_clase: dict[str, dict[str, float]] = {}
    if target in df.columns:
        afectadas = nulos_por_fila > 0
        for clase, sub in df.groupby(target, observed=True):
            n_clase = len(sub)
            n_afectadas = int(afectadas.loc[sub.index].sum())
            reparto_por_clase[str(clase)] = {
                "n_clase": n_clase,
                "filas_con_nulos": n_afectadas,
                "pct_de_la_clase": 100.0 * n_afectadas / n_clase if n_clase else np.nan,
            }

    clases_sin_nulos = sorted(
        clase for clase, v in reparto_por_clase.items() if v["filas_con_nulos"] == 0
    )
    solapan = (
        "Hay filas con nulos en varias columnas."
        if max_por_fila > 1
        else ("Los nulos NO se solapan entre columnas.")
    )
    lectura = (
        f"{int(sum(nulos_por_columna.values()))} nulos repartidos en "
        f"{filas_afectadas} filas; ninguna fila acumula más de {max_por_fila}. {solapan}"
    )
    if clases_sin_nulos:
        lectura += (
            f" Ninguna fila de la clase {', '.join(repr(c) for c in clases_sin_nulos)} "
            f"tiene nulos: el patrón de ausencia no es aleatorio respecto al objetivo."
        )

    return {
        "n_filas": len(df),
        "nulos_por_columna": nulos_por_columna,
        "total_nulos": int(sum(nulos_por_columna.values())),
        "filas_afectadas": filas_afectadas,
        "filas_por_numero_de_nulos": filas_por_numero_de_nulos,
        "max_nulos_en_una_fila": max_por_fila,
        "solapan_entre_columnas": bool(max_por_fila > 1),
        "reparto_por_clase": reparto_por_clase,
        "clases_sin_ningun_nulo": clases_sin_nulos,
        "duplicados_completos": int(df.duplicated().sum()),
        "lectura": lectura,
    }


def physical_anomalies(
    df: pd.DataFrame, ranges: dict[str, tuple[float, float]] | None = None
) -> pd.DataFrame:
    """Valores fuera del rango físicamente posible.

    Una fila listada aquí va a CUARENTENA, no se imputa: un sensor que devuelve
    un imposible físico no es un dato que falte, es un dato que miente.
    """
    ranges = PHYSICAL_RANGES if ranges is None else ranges
    filas = []
    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue
        s = df[col]
        fuera = s.notna() & ((s < low) | (s > high))
        for idx in df.index[fuera]:
            valor = float(s.loc[idx])
            filas.append(
                {
                    "fila": idx,
                    "atributo": col,
                    "valor": valor,
                    "limite_inferior": low,
                    "limite_superior": high,
                    "violacion": "por debajo del mínimo"
                    if valor < low
                    else "por encima del máximo",
                }
            )
    columnas = ["fila", "atributo", "valor", "limite_inferior", "limite_superior", "violacion"]
    if not filas:
        return pd.DataFrame(columns=columnas)
    return pd.DataFrame(filas, columns=columnas).sort_values(
        ["fila", "atributo"], ignore_index=True
    )


def univariate_auc(
    df: pd.DataFrame, target: str = "failure", positive_label: str = "yes"
) -> pd.DataFrame:
    """Asociación univariante por atributo: AUC de Mann-Whitney y p-valor a dos colas.

    AUC = U / (n_pos * n_neg), es decir la probabilidad de que un positivo tomado
    al azar tenga un valor mayor que un negativo tomado al azar. AUC < 0.5 no es
    "ausencia de señal": es señal en sentido **inverso**.
    """
    y = df[target].astype(str)
    es_positivo = y == str(positive_label)
    filas = []
    for col in numeric_attributes(df, target):
        pos = df.loc[es_positivo, col].dropna()
        neg = df.loc[~es_positivo, col].dropna()
        u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        auc = float(u) / (len(pos) * len(neg))
        filas.append(
            {
                "atributo": col,
                "n_positivos": len(pos),
                "n_negativos": len(neg),
                "media_positivos": float(pos.mean()),
                "media_negativos": float(neg.mean()),
                "auc": auc,
                "p_valor": float(p),
                "direccion": "directa" if auc >= 0.5 else "inversa",
                "significativo_5pct": bool(p < 0.05),
            }
        )
    tabla = pd.DataFrame(filas).set_index("atributo")
    # Ordenar por distancia a 0.5: lo informativo es alejarse del azar, en
    # cualquiera de los dos sentidos.
    return tabla.reindex((tabla["auc"] - 0.5).abs().sort_values(ascending=False).index)


def class_balance(df: pd.DataFrame, target: str = "failure", positive_label: str = "yes") -> dict:
    """Conteos, prevalencia y accuracy del clasificador trivial mayoritario.

    El último número es el que hay que poner delante de cualquier accuracy que
    reporte el repositorio (CLAUDE.md §6.2).
    """
    y = df[target].astype(str)
    conteos = {str(k): int(v) for k, v in y.value_counts().sort_index().items()}
    n = len(y)
    n_positivos = conteos.get(str(positive_label), 0)
    clase_mayoritaria = max(conteos, key=lambda k: conteos[k])
    return {
        "n": n,
        "clases": sorted(conteos),
        "conteos": conteos,
        "nulos_en_objetivo": int(df[target].isna().sum()),
        "clase_positiva": str(positive_label),
        "n_positivos": n_positivos,
        "prevalencia": n_positivos / n if n else np.nan,
        "prevalencia_pct": 100.0 * n_positivos / n if n else np.nan,
        "clase_mayoritaria": clase_mayoritaria,
        "accuracy_trivial_mayoritario": conteos[clase_mayoritaria] / n if n else np.nan,
        "accuracy_trivial_mayoritario_pct": 100.0 * conteos[clase_mayoritaria] / n if n else np.nan,
    }


def correlation_matrix(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Matriz de correlación de Pearson entre atributos numéricos."""
    return df[numeric_attributes(df, target)].corr(method="pearson")


def max_abs_correlation(df: pd.DataFrame, target: str | None = None) -> dict:
    """Mayor |r| fuera de la diagonal, con el par que lo produce.

    En maquinaria real, carga -> temperatura -> viscosidad -> vibración es una
    cadena causal acoplada. Un máximo cercano a cero es la primera firma de que
    el dataset es sintético (CLAUDE.md §7).
    """
    corr = correlation_matrix(df, target)
    sin_diagonal = corr.where(~np.eye(len(corr), dtype=bool))
    apilada = sin_diagonal.abs().stack()
    if apilada.empty:
        return {"max_abs_r": np.nan, "par": None}
    par = apilada.idxmax()
    return {
        "max_abs_r": float(apilada.max()),
        "par": [str(par[0]), str(par[1])],
        "r_con_signo": float(corr.loc[par[0], par[1]]),
    }
