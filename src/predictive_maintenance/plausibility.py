"""Auditor de plausibilidad física: ¿son estos datos reales o simulados?

Tres diagnósticos independientes (CLAUDE.md §7, §12 extra 1), cada uno con su
propio veredicto y evidencia numérica. Ninguno por separado demuestra nada —
`audit()` los combina, y es la CONCURRENCIA de firmas lo que hace defendible
la conclusión "probablemente sintético".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from predictive_maintenance import eda

# Cadena causal física esperada en una máquina real (CLAUDE.md §7.1): la carga
# eleva la temperatura por fricción; la temperatura reduce la viscosidad del
# lubricante; una viscosidad menor deja más juego mecánico, que se traduce en
# más vibración. La viscosidad no se mide, pero su rastro debería verse como
# correlación de punta a punta entre `load_percent` y `vibration_mm_s`, y en
# menor medida entre `temperature_c` y las otras dos.
CADENA_FISICA_ESPERADA = "carga -> temperatura -> viscosidad del lubricante -> vibración"

# Por debajo de este |r|, ningún par de atributos está acoplado. En un equipo
# físico real esto no ocurre: alguna cadena causal (aunque sea débil) deja
# huella en al menos un par de sensores.
UMBRAL_CORRELACION_MINIMA = 0.15

# Umbral de significancia para el test de uniformidad del último dígito.
ALPHA_UNIFORMIDAD = 0.05


def correlation_structure(
    df: pd.DataFrame,
    target: str | None = "failure",
    threshold: float = UMBRAL_CORRELACION_MINIMA,
) -> dict:
    """Primera firma: independencia mutua total entre atributos.

    Cadena física esperada: carga -> temperatura -> viscosidad del lubricante
    -> vibración (`CADENA_FISICA_ESPERADA`). Si el dataset viene de un equipo
    real, se espera |r| >= `threshold` en al menos un par de atributos
    numéricos. Una correlación máxima por debajo del umbral, en TODO el
    dataframe, es la primera firma de generación sintética.
    """
    resultado = eda.max_abs_correlation(df, target)
    max_abs_r = resultado["max_abs_r"]
    independientes = bool(max_abs_r < threshold) if not np.isnan(max_abs_r) else False

    if independientes:
        veredicto = "variables mutuamente independientes: compatible con generación sintética"
    else:
        veredicto = "hay al menos un par de atributos acoplados: compatible con un equipo real"

    return {
        "max_abs_r": max_abs_r,
        "par": resultado["par"],
        "umbral": threshold,
        "cadena_fisica_esperada": CADENA_FISICA_ESPERADA,
        "variables_mutuamente_independientes": independientes,
        "veredicto": veredicto,
        "evidencia": (
            f"correlación máxima |r| = {max_abs_r:.4f} entre {resultado['par']} "
            f"({'por debajo' if independientes else 'por encima'} del umbral {threshold:.2f})"
        ),
    }


def missingness_regularity(df: pd.DataFrame, target: str = "failure") -> dict:
    """Segunda firma: un patrón de nulos demasiado regular para ser real.

    La bandera depende de dos preguntas deterministas sobre el mismo
    `eda.missingness_report`: ¿el número de nulos es idéntico en cada columna
    afectada?, ¿se solapan alguna vez en la misma fila? Un sensor real falla
    por causas independientes (corte de energía, saturación puntual, error de
    transmisión): que el recuento de nulos coincida exacto entre columnas y
    jamás se solape es un patrón de generación, no de fallo de
    instrumentación.

    Una tercera pregunta —¿se reparten entre las clases en proporción a su
    tamaño?— se reporta, pero NO participa en la bandera: bajo un modelo
    puramente al azar (hipergeométrico, sin reposición), la probabilidad de
    que ninguno de los nulos caiga en una clase minoritaria de 10 sobre 180 ya
    es considerable por sí sola. Tratar ese resultado como evidencia sería
    sobreinterpretar ruido de muestra pequeña.
    """
    informe = eda.missingness_report(df, target)
    conteos_afectados = [n for n in informe["nulos_por_columna"].values() if n > 0]
    reparto_igual_por_columna = len(conteos_afectados) > 0 and len(set(conteos_afectados)) == 1
    sin_solapamiento = not informe["solapan_entre_columnas"]
    bandera_regularidad_sospechosa = reparto_igual_por_columna and sin_solapamiento

    if bandera_regularidad_sospechosa:
        veredicto = "patrón de nulos demasiado regular: compatible con generación sintética"
    else:
        veredicto = "patrón de nulos irregular: compatible con fallos de instrumentación reales"

    total_nulos = informe["total_nulos"]
    balance = eda.class_balance(df, target)
    observado = informe["reparto_por_clase"].get(balance["clase_positiva"], {"filas_con_nulos": 0})[
        "filas_con_nulos"
    ]
    esperado_bajo_azar = total_nulos * balance["prevalencia"]
    # P(observar `observado` nulos o menos en la clase positiva por puro azar),
    # bajo un modelo hipergeométrico: repartir `total_nulos` filas al azar,
    # sin reposición, entre las `balance["n"]` filas del dataset. NO es parte
    # de la bandera de regularidad; se reporta como evidencia NO concluyente.
    p_valor = (
        float(stats.hypergeom.cdf(observado, balance["n"], balance["n_positivos"], total_nulos))
        if total_nulos and balance["n_positivos"]
        else None
    )

    return {
        "nulos_por_columna": informe["nulos_por_columna"],
        "reparto_igual_por_columna": reparto_igual_por_columna,
        "solapan_entre_columnas": informe["solapan_entre_columnas"],
        "sin_solapamiento": sin_solapamiento,
        "bandera_regularidad_sospechosa": bandera_regularidad_sospechosa,
        "veredicto": veredicto,
        "reparto_por_clase_no_concluyente": {
            "observado": observado,
            "esperado_bajo_azar": esperado_bajo_azar,
            "p_valor": p_valor,
            "interpretacion": (
                f"no concluyente: un reparto así de extremo, o más, ocurre por puro azar "
                f"con probabilidad {p_valor:.4f}"
                if p_valor is not None
                else "no concluyente: sin nulos o sin positivos, no hay nada que probar"
            ),
        },
        "evidencia": (
            f"{total_nulos} nulos repartidos en exactamente "
            f"{conteos_afectados} por columna, sin solapar jamás entre columnas"
        ),
    }


def _decimales_necesarios(serie: pd.Series, maximo: int = 4) -> int:
    """Nº de decimales con los que está realmente registrada la columna.

    No todos los sensores se registran con la misma resolución (en `lab180`,
    `temperature_c` trae 1 decimal y `vibration_mm_s` trae 2): usar un número
    fijo de decimales para todas las columnas haría que el "último dígito" de
    las columnas de menor resolución fuera trivialmente 0 siempre, lo que no
    es una firma de nada — es no haber leído la resolución real del dato.
    """
    valores = serie.dropna().to_numpy(dtype=float)
    if len(valores) == 0:
        return 0
    for d in range(maximo + 1):
        if np.allclose(np.round(valores, d), valores, atol=1e-9):
            return d
    return maximo


def _ultimo_digito(serie: pd.Series) -> tuple[np.ndarray, int]:
    valores = serie.dropna()
    if pd.api.types.is_integer_dtype(serie):
        return valores.to_numpy() % 10, 0
    decimales = _decimales_necesarios(valores)
    escalado = np.round(valores.to_numpy() * (10**decimales)).astype(np.int64)
    return escalado % 10, decimales


def digit_distribution(df: pd.DataFrame, target: str | None = "failure") -> dict:
    """Tercera firma, de apoyo: ¿cómo se distribuye el último dígito decimal?

    Un sensor físico de bajo coste arrastra sesgos de cuantización del
    conversor analógico-digital: el último dígito rara vez es perfectamente
    uniforme. Datos generados por muestreo continuo (`numpy.random.uniform` +
    redondeo) sí tienden a la uniformidad exacta. Un chi-cuadrado de bondad de
    ajuste que NUNCA rechaza la uniformidad en ninguna columna es evidencia
    débil y de apoyo — no decide por sí sola, se combina con las otras dos. La
    resolución (nº de decimales) se detecta por columna: ver
    `_decimales_necesarios`.
    """
    columnas = {}
    for col in eda.numeric_attributes(df, target):
        digitos, decimales = _ultimo_digito(df[col])
        conteos = np.bincount(digitos, minlength=10)
        chi2, p_valor = stats.chisquare(conteos)
        columnas[col] = {
            "decimales_detectados": decimales,
            "conteos_por_digito": conteos.tolist(),
            "chi2": float(chi2),
            "p_valor_uniformidad": float(p_valor),
            "compatible_con_uniforme": bool(p_valor > ALPHA_UNIFORMIDAD),
        }

    compatibles = [c for c, v in columnas.items() if v["compatible_con_uniforme"]]
    todas_uniformes = len(compatibles) == len(columnas) and len(columnas) > 0

    return {
        "alpha": ALPHA_UNIFORMIDAD,
        "por_atributo": columnas,
        "atributos_compatibles_con_uniforme": compatibles,
        "todas_las_columnas_compatibles_con_uniforme": todas_uniformes,
        "veredicto": (
            "distribución del último dígito compatible con muestreo uniforme (evidencia débil "
            "de apoyo a generación sintética)"
            if todas_uniformes
            else "al menos un atributo se aparta de la uniformidad en el último dígito"
        ),
    }


@dataclass
class PlausibilityReport:
    """Veredicto combinado de las tres firmas, con su evidencia numérica."""

    dataset: str
    correlacion: dict
    nulos: dict
    digitos: dict
    veredicto: str
    evidencia: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "correlacion": self.correlacion,
            "nulos": self.nulos,
            "digitos": self.digitos,
            "veredicto": self.veredicto,
            "evidencia": self.evidencia,
        }


def audit(df: pd.DataFrame, dataset: str, target: str = "failure") -> PlausibilityReport:
    """Combina las tres firmas en un veredicto único.

    Las dos firmas fuertes y deterministas —independencia de correlación y
    regularidad del patrón de nulos— deciden el veredicto; la distribución del
    último dígito se reporta siempre como evidencia adicional, nunca como
    criterio único, porque por sí sola es compatible con datos reales de baja
    resolución.
    """
    correlacion = correlation_structure(df, target)
    nulos = missingness_regularity(df, target)
    digitos = digit_distribution(df, target)

    firmas_fuertes = [
        correlacion["variables_mutuamente_independientes"],
        nulos["bandera_regularidad_sospechosa"],
    ]
    probablemente_sintetico = all(firmas_fuertes)

    evidencia = [correlacion["evidencia"], nulos["evidencia"]]
    if digitos["todas_las_columnas_compatibles_con_uniforme"]:
        evidencia.append(
            "el último dígito decimal es compatible con muestreo uniforme en todos los "
            "atributos (evidencia débil, de apoyo)"
        )

    veredicto = (
        "probablemente sintético" if probablemente_sintetico else "compatible con datos reales"
    )

    return PlausibilityReport(
        dataset=dataset,
        correlacion=correlacion,
        nulos=nulos,
        digitos=digitos,
        veredicto=veredicto,
        evidencia=evidencia,
    )
