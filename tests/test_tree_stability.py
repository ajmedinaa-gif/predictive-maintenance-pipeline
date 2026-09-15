"""Test de estabilidad de la raíz del árbol (CLAUDE.md §10.1).

TEST CLAVE E INTENCIONADO: `root_stability_of("vibration_mm_s") < 0.70`. Es
deliberado — protege contra la conclusión errónea "el atributo que decide el
fallo es la vibración". Con 300 bootstraps estratificados, la vibración es el
primer corte del árbol solo un poco más de la mitad de las veces (CLAUDE.md
§10.1: ~51.3 %; en este repo, con la implementación real, ronda el 50-60 %).
NO conviertas este test en una igualdad con un valor exacto: el porcentaje
depende de la semilla y de la versión de scikit-learn, y fijarlo destruiría el
propósito del test — que la vibración NO domine por encima de 0.70.
"""

import pytest

from predictive_maintenance import data, explain, schema


@pytest.fixture(scope="module")
def lab180_valid_xy(lab180_spec):
    resultado = data.load_validated(lab180_spec.path, schema.Lab180Schema, dataset_name="lab180")
    df = resultado.valid
    X = df.drop(columns=[lab180_spec.target])
    y = df[lab180_spec.target].astype(str)
    return X, y


@pytest.fixture(scope="module")
def stability_report(lab180_valid_xy):
    X, y = lab180_valid_xy
    return explain.tree_root_stability(X, y, n_boot=300, seed=42)


def _root_stability_of(report: dict, attribute: str) -> float:
    return report["porcentaje_raiz_por_atributo"].get(attribute, 0.0) / 100.0


def test_vibration_is_not_the_dominant_root_split(stability_report):
    # El test clave: deliberado, no un accidente de una mala semilla.
    assert _root_stability_of(stability_report, "vibration_mm_s") < 0.70


def test_vibration_is_still_the_most_common_root_split(stability_report):
    # Es el atributo con más señal univariante (CLAUDE.md §6.4, AUC 0.85), así
    # que se espera que gane la raíz más a menudo que cualquier otro atributo
    # individual -- solo que no de forma dominante (test anterior).
    assert stability_report["atributo_raiz_mas_frecuente"] == "vibration_mm_s"


def test_root_percentages_sum_to_one_hundred(stability_report):
    total = sum(stability_report["porcentaje_raiz_por_atributo"].values())
    assert total == pytest.approx(100.0, abs=1e-6)


def test_effective_depth_is_close_to_but_not_exceeding_max_depth(stability_report):
    # max_depth=3 (CLAUDE.md §10.1): la profundidad efectiva media no puede
    # superarlo, y con min_samples_leaf=5 se espera algo por debajo de 3.
    assert 2.0 <= stability_report["profundidad_efectiva_media"] <= 3.0


def test_all_five_attributes_appear_as_root_at_least_once_in_300_boots(stability_report):
    # Con 300 remuestreos y cinco atributos, es señal de que el procedimiento
    # de bootstrap realmente varía la muestra -- no un árbol degenerado que
    # siempre corta por el mismo atributo.
    assert len(stability_report["porcentaje_raiz_por_atributo"]) == 5


def test_reproducible_with_fixed_seed(lab180_valid_xy):
    X, y = lab180_valid_xy
    r1 = explain.tree_root_stability(X, y, n_boot=50, seed=42)
    r2 = explain.tree_root_stability(X, y, n_boot=50, seed=42)
    assert r1["porcentaje_raiz_por_atributo"] == r2["porcentaje_raiz_por_atributo"]
