"""Carga con contrato: ningún dato entra al pipeline sin pasar `schema.py`.

Regla dura (CLAUDE.md §2.9): las filas que violan el contrato van a
`data/quarantine/`, con el motivo por el que se rechazaron. No se borran ni se
imputan en silencio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from config.config import PROJECT_ROOT, Settings, get_settings

from predictive_maintenance import schema as schema_module

logger = logging.getLogger(__name__)

# Checks cuyo `check` describe una incompatibilidad de TIPO o una columna
# ausente, no un valor concreto fuera de rango. Un fallo de este tipo dice que
# la fuente de datos no tiene la forma que el contrato espera — es un fallo
# estructural, no un dato sucio — así que nunca se convierte en cuarentena: se
# relanza para que quien llame lo vea de inmediato.
_STRUCTURAL_SCHEMA_CONTEXTS = {"DataFrameSchema"}


def _es_fallo_estructural(schema_context: str, check: str) -> bool:
    return schema_context in _STRUCTURAL_SCHEMA_CONTEXTS or str(check).startswith("dtype(")


@dataclass
class ValidationResult:
    """Resultado de validar un dataset contra su contrato.

    `report` es el mismo diccionario que se vuelca a `reports/*.json`
    (CLAUDE.md §2.8): nada de lo que diga este objeto se recalcula a mano en
    otro sitio.
    """

    valid: pd.DataFrame
    quarantined: pd.DataFrame
    report: dict = field(default_factory=dict)


def _motivo_de_fila(grupo: pd.DataFrame) -> str:
    partes = []
    for fila in grupo.itertuples(index=False):
        valor = fila.failure_case
        if hasattr(valor, "item"):
            valor = valor.item()
        partes.append(f"{fila.column}={valor!r} incumple {fila.check}")
    return "; ".join(partes)


def _separar_validas_y_cuarentena(
    df: pd.DataFrame, schema: type[pa.DataFrameModel]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el contrato en modo `lazy` y separa filas válidas de inválidas.

    Un fallo con índice de fila (rango físico, `isin`, nulo inesperado) es un
    dato sucio: esa fila va a cuarentena. Un fallo sin índice de fila (columna
    ausente) o de tipo (`dtype(...)`) es estructural: se relanza tal cual,
    porque no hay fila que poner en cuarentena — el fichero entero no tiene la
    forma que el contrato exige.
    """
    try:
        schema.validate(df, lazy=True)
        vacio = df.iloc[0:0].copy()
        vacio["quarantine_reason"] = pd.Series(dtype=str)
        return df.copy(), vacio
    except pa.errors.SchemaErrors as exc:
        casos = exc.failure_cases
        estructurales = casos[
            casos.apply(lambda r: _es_fallo_estructural(r["schema_context"], r["check"]), axis=1)
        ]
        if not estructurales.empty:
            raise

        filas_malas = casos.dropna(subset=["index"]).copy()
        filas_malas["index"] = filas_malas["index"].astype(int)
        motivos = filas_malas.groupby("index").apply(_motivo_de_fila, include_groups=False)

        indices_malos = motivos.index
        cuarentena = df.loc[indices_malos].copy()
        cuarentena["quarantine_reason"] = motivos.to_numpy()
        validas = df.drop(index=indices_malos).copy()
        return validas, cuarentena


def load_validated(
    path: Path,
    schema: type[pa.DataFrameModel],
    *,
    dataset_name: str | None = None,
    settings: Settings | None = None,
) -> ValidationResult:
    """Lee `path`, valida contra `schema` y separa filas válidas de cuarentena.

    Las filas en cuarentena se vuelcan a
    `data/quarantine/<dataset_name>_<timestamp>.csv`, con la columna
    `quarantine_reason`. No se borran ni se imputan: quedan en disco para que
    alguien las revise.
    """
    settings = settings or get_settings()
    dataset_name = dataset_name or Path(path).stem

    df = pd.read_csv(path).reset_index(drop=True)
    validas, cuarentena = _separar_validas_y_cuarentena(df, schema)
    avisos = schema_module.quality_warnings(validas, settings.quality_thresholds)

    ruta_cuarentena: Path | None = None
    if len(cuarentena):
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        directorio = settings.paths.data_quarantine
        directorio.mkdir(parents=True, exist_ok=True)
        ruta_cuarentena = directorio / f"{dataset_name}_{timestamp}.csv"
        cuarentena.to_csv(ruta_cuarentena, index=False)

    ruta_cuarentena_relativa = None
    if ruta_cuarentena is not None:
        try:
            ruta_cuarentena_relativa = str(ruta_cuarentena.relative_to(PROJECT_ROOT))
        except ValueError:
            ruta_cuarentena_relativa = str(ruta_cuarentena)

    resumen = {
        "n_filas_leidas": len(df),
        "n_validas": len(validas),
        "n_cuarentena": len(cuarentena),
        "ruta_cuarentena": ruta_cuarentena_relativa,
        "avisos_calidad": avisos,
        "motivos_cuarentena": (
            cuarentena[["quarantine_reason"]].reset_index().to_dict(orient="records")
            if len(cuarentena)
            else []
        ),
    }
    logger.info(
        "validate(%s): %d filas leídas, %d válidas, %d en cuarentena, %d avisos",
        dataset_name,
        resumen["n_filas_leidas"],
        resumen["n_validas"],
        resumen["n_cuarentena"],
        len(avisos),
    )

    return ValidationResult(valid=validas, quarantined=cuarentena, report=resumen)
