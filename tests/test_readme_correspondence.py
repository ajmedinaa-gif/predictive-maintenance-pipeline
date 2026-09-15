"""Verifica que los números clave del README coinciden con `reports/*.json`
(CLAUDE.md regla dura 8: ningún número se escribe a mano).

Se salta con un mensaje claro cuando el `reports/*.json` correspondiente
todavía no existe -- es la situación normal en un checkout limpio, o en el
job `test` de CI (que corre rápido, en matriz de dos versiones de Python, y
deliberadamente NO ejecuta el pipeline primero). Los jobs `pipeline` y
`ai4i2020` de `ci.yml` sí generan esos ficheros antes, así que ahí este test
se ejecuta de verdad -- no es un test decorativo, es la comprobación real
cuando hay algo que comprobar.
"""

from __future__ import annotations

import json

import pytest

from predictive_maintenance import datasets

REPORTS_DIR = datasets.PROJECT_ROOT / "reports"
README = datasets.PROJECT_ROOT / "README.md"


def _leer_json(nombre: str) -> dict | None:
    ruta = REPORTS_DIR / f"{nombre}.json"
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def _requerir(informe: dict | None, mensaje: str) -> None:
    if informe is None:
        pytest.skip(mensaje)


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_exists():
    assert README.exists()


def test_lab180_dummy_accuracy_matches_metrics_json(readme_text):
    metrics = _leer_json("metrics_lab180")
    _requerir(
        metrics, "reports/metrics_lab180.json no existe -- ejecuta `pdm-cli run --dataset lab180`."
    )
    accuracy_pct = metrics["modelos"]["dummy_most_frequent"]["metricas"]["accuracy"]["media"] * 100
    assert f"{accuracy_pct:.2f}" in readme_text


def test_lab180_positive_count_matches_eda_json(readme_text):
    eda = _leer_json("eda_lab180")
    _requerir(eda, "reports/eda_lab180.json no existe -- ejecuta `pdm-cli eda --dataset lab180`.")
    n_positivos = eda["balance_de_clases"]["n_positivos"]
    assert str(n_positivos) in readme_text


def test_ai4i2020_positive_count_matches_eda_json(readme_text):
    eda = _leer_json("eda_ai4i2020")
    _requerir(
        eda, "reports/eda_ai4i2020.json no existe -- ejecuta `pdm-cli eda --dataset ai4i2020`."
    )
    n_positivos = eda["balance_de_clases"]["n_positivos"]
    assert str(n_positivos) in readme_text


def test_comparison_table_numbers_match_comparison_json(readme_text):
    """La tabla "Resultado principal" del README: cada celda numérica clave
    debe rastrearse hasta `reports/comparison.json`, no a un cálculo aparte."""
    comparacion = _leer_json("comparison")
    _requerir(comparacion, "reports/comparison.json no existe -- ejecuta `pdm-cli compare`.")

    for info in comparacion["datasets"].values():
        assert str(info["n_positivos"]) in readme_text

        recall_str = f"{info['recall_platt_tp']}/{info['recall_platt_n_positivos']}"
        assert recall_str in readme_text

        ci_inferior, ci_superior = info["recall_wilson_ci_platt"]
        assert f"{ci_inferior:.3f}" in readme_text
        assert f"{ci_superior:.3f}" in readme_text

        assert f"{info['umbral_optimo_platt']:.3f}" in readme_text


def test_ai4i2020_failure_mode_counts_match_raw_csv(readme_text):
    """Los cinco modos de fallo de ai4i2020 (TWF/HDF/PWF/OSF/RNF): el README
    debe declarar la suma real (373), no un supuesto de que coincide con 339."""
    spec = datasets.get_spec("ai4i2020")
    if not spec.path.exists():
        pytest.skip(
            "data/raw/ai4i2020/ai4i2020.csv no existe -- ejecuta "
            "`pdm-cli download --dataset ai4i2020`."
        )
    import pandas as pd

    df = pd.read_csv(spec.path)
    modos = ["twf", "hdf", "pwf", "osf", "rnf"]
    suma_modos = int(df[modos].sum().sum())
    assert str(suma_modos) in readme_text
