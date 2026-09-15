"""Verifica que el anexo académico está EJECUTADO de verdad (CLAUDE.md, Fase 4 D).

Un notebook con celdas de código sin `execution_count` (o sin salidas) es un
notebook sin ejecutar. Este test NO re-ejecuta el notebook — eso es
`make notebook`, con un kernel real vía `nbclient` (`tools/run_notebook.py`)
— solo comprueba que el que está versionado en el repositorio sí se
ejecutó: en este repositorio, una salida que no se puede reproducir no vale.
"""

from pathlib import Path

import nbformat

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "00_lab_original.ipynb"


def _code_cells() -> list:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    return [c for c in nb.cells if c.cell_type == "code"]


def test_notebook_file_exists_and_is_under_two_megabytes():
    assert NOTEBOOK_PATH.exists()
    tamano_mb = NOTEBOOK_PATH.stat().st_size / (1024 * 1024)
    assert tamano_mb < 2.0, f"{NOTEBOOK_PATH.name} pesa {tamano_mb:.2f} MB, CLAUDE.md exige < 2 MB"


def test_notebook_has_code_cells():
    assert len(_code_cells()) > 0


def test_every_code_cell_has_a_nonnull_execution_count():
    for i, celda in enumerate(_code_cells()):
        assert celda.execution_count is not None, (
            f"celda de código #{i} sin ejecutar (execution_count=None)"
        )


def test_execution_counts_are_sequential_and_increasing():
    # Un notebook ejecutado de arriba a abajo, de una sola vez, tiene
    # execution_count estrictamente creciente y consecutivo desde 1 -- la
    # firma de una ejecución real, no de celdas pegadas de ejecuciones sueltas.
    conteos = [c.execution_count for c in _code_cells()]
    assert conteos == list(range(1, len(conteos) + 1))


def test_every_code_cell_produced_at_least_one_output():
    for i, celda in enumerate(_code_cells()):
        assert celda.outputs, f"celda de código #{i} no produjo ninguna salida"


def test_no_code_cell_has_an_error_output():
    for celda in _code_cells():
        for salida in celda.outputs:
            assert salida.get("output_type") != "error", (
                f"la celda tiene una salida de error: {salida}"
            )
