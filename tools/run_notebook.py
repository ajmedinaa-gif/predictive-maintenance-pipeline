"""Re-ejecuta notebooks/00_lab_original.ipynb con un kernel Jupyter REAL.

Uso: `make notebook`, o `uv run python tools/run_notebook.py`.

Deliberadamente NO usa `nbconvert` (no está en las dependencias, CLAUDE.md
§5): lee y escribe el notebook directamente con `nbformat` + `nbclient`,
ejecutando cada celda de código en un kernel `ipykernel` real. El kernelspec
se (re)instala dentro de `.venv/share/jupyter/kernels/` en cada ejecución —
idempotente, y así un `uv sync` limpio en otra máquina no necesita un paso
manual aparte.

Es lo que exige el CHECKPOINT de la Fase 4: un notebook cuya salida se
transcribiera a mano no vale, y la única forma de que valga es que este
script pueda volver a producir el mismo fichero desde cero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from ipykernel.kernelspec import install as install_kernelspec
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "00_lab_original.ipynb"
KERNEL_NAME = "pdm-python3"


def _ensure_kernelspec() -> None:
    """Registra el kernel `pdm-python3` dentro de `.venv`, si no existe ya."""
    install_kernelspec(
        user=False,
        kernel_name=KERNEL_NAME,
        display_name="predictive-maintenance-pipeline",
        prefix=sys.prefix,
    )


def main() -> None:
    _ensure_kernelspec()
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name=KERNEL_NAME,
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
    nb["metadata"]["kernelspec"] = {
        "display_name": "predictive-maintenance-pipeline",
        "language": "python",
        "name": KERNEL_NAME,
    }
    nbformat.write(nb, NOTEBOOK_PATH)
    tamano_kb = NOTEBOOK_PATH.stat().st_size / 1024
    print(f"Notebook re-ejecutado: {NOTEBOOK_PATH} ({tamano_kb:.1f} KB)")


if __name__ == "__main__":
    main()
