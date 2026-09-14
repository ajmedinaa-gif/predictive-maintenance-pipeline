#!/usr/bin/env python3
"""Extrae el prompt de una fase de FASES.md.

Uso:
    python3 tools/fase.py 1 | pbcopy     # copia la Fase 1 al portapapeles
    python3 tools/fase.py 1              # la imprime en pantalla
    python3 tools/fase.py                # lista las fases disponibles

Funciona con el Python del sistema (3.9+); no necesita el entorno del proyecto.
"""
import pathlib
import re
import sys

FASES = pathlib.Path(__file__).resolve().parent.parent / "FASES.md"


def main() -> int:
    if not FASES.exists():
        print(f"No encuentro {FASES}", file=sys.stderr)
        return 1
    texto = FASES.read_text(encoding="utf-8")
    titulos = re.findall(r"^# FASE (\d+) — (.+)$", texto, flags=re.MULTILINE)

    if len(sys.argv) < 2:
        print("Fases disponibles:\n")
        for num, titulo in titulos:
            print(f"  {num}. {titulo}")
        print("\nUso: python3 tools/fase.py <n> | pbcopy")
        return 0

    n = sys.argv[1].strip()
    patron = re.compile(rf"^# FASE {re.escape(n)} — .+$", flags=re.MULTILINE)
    m = patron.search(texto)
    if not m:
        print(f"No existe la FASE {n}. Disponibles: "
              f"{', '.join(num for num, _ in titulos)}", file=sys.stderr)
        return 1

    resto = texto[m.end():]
    bloques = re.findall(r"^```\n(.*?)^```", resto, flags=re.MULTILINE | re.DOTALL)
    if not bloques:
        print(f"La FASE {n} no tiene bloque de prompt", file=sys.stderr)
        return 1

    print(bloques[0].strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
