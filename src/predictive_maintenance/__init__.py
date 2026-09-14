"""Pipeline de mantenimiento predictivo industrial.

El repositorio no vende un modelo: vende un protocolo de validación honesto
bajo desbalance extremo de clases. Ver CLAUDE.md §1.
"""

import sys
from pathlib import Path

__version__ = "0.1.0"

# `config/config.py` vive fuera de `src/` a propósito (no es parte del paquete
# instalable: es configuración del repositorio). Para que `import config` sea
# fiable desde cualquier punto de entrada -- `pdm-cli` instalado, pytest,
# `uv run python` -- sin importar el directorio de trabajo actual, se añade la
# raíz del proyecto a `sys.path` una única vez, al importar este paquete.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
