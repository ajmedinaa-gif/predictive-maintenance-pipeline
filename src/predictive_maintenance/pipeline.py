"""Fábrica de `sklearn.Pipeline` completos: imputación -> escalado -> clasificador.

Regla dura (CLAUDE.md §2.6): la imputación (y el escalado) van SIEMPRE dentro
del `Pipeline`, nunca antes del split. Si se imputara antes de dividir en
folds, la mediana usada para rellenar los nulos del fold de test se habría
calculado viendo también el fold de test — fuga de datos, por pequeña que
parezca. Con todo dentro del `Pipeline`, `cross_validate_model` (`evaluate.py`)
clona un `Pipeline` sin ajustar en cada fold, así que el imputador (y el
escalador, donde lo hay) se ajustan una sola vez por fold, solo sobre su
partición de entrenamiento.

El escalado (`StandardScaler`) solo se añade para los modelos sensibles a la
escala de las variables (regresión logística): los basados en árboles
(`DecisionTreeClassifier`, `RandomForestClassifier`,
`GradientBoostingClassifier`) parten el espacio por umbrales univariantes y
son invariantes a escalar o no.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from predictive_maintenance.config import get_settings

# Orden deliberado: los dos `dummy_*` van primero. Es el mismo orden en el que
# CLAUDE.md §2.1 exige presentarlos en cualquier tabla de resultados.
MODEL_NAMES: tuple[str, ...] = (
    "dummy_most_frequent",
    "dummy_stratified",
    "tree_default",
    "tree_shallow_balanced",
    "logistic_balanced",
    "logistic_plain",
    "rf_balanced",
    "gradient_boosting",
)

# Modelos cuyo clasificador es sensible a la escala de las variables.
_NECESITA_ESCALADO = frozenset({"logistic_balanced", "logistic_plain"})


def _resolve_seed(seed: int | None) -> int:
    return seed if seed is not None else get_settings().seed


def _build_classifier(name: str, seed: int):
    if name == "dummy_most_frequent":
        # Sin aleatoriedad: siempre predice la clase mayoritaria. No necesita
        # `random_state`.
        return DummyClassifier(strategy="most_frequent")
    if name == "dummy_stratified":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "tree_default":
        # Sin restricciones de profundidad ni balanceo: el árbol que
        # sobreajusta y falla los 10 positivos (CLAUDE.md §8.2).
        return DecisionTreeClassifier(random_state=seed)
    if name == "tree_shallow_balanced":
        return DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "logistic_balanced":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    if name == "logistic_plain":
        # SIN `class_weight`: es la base de la Fase 4 (calibración de Platt +
        # umbral por coste). CLAUDE.md §2.5: `class_weight="balanced"` y el
        # umbral por coste no se usan a la vez porque cuentan el desbalance
        # dos veces.
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "rf_balanced":
        return RandomForestClassifier(class_weight="balanced", random_state=seed)
    if name == "gradient_boosting":
        # `GradientBoostingClassifier` no admite `class_weight` en
        # scikit-learn: por eso el zoo no tiene una variante
        # `gradient_boosting_balanced`.
        return GradientBoostingClassifier(random_state=seed)
    disponibles = ", ".join(MODEL_NAMES)
    raise KeyError(f"Modelo desconocido: {name!r}. Disponibles: {disponibles}")


def build_pipeline(name: str, seed: int | None = None) -> Pipeline:
    """Construye el `Pipeline` completo para el modelo `name`.

    `seed` por defecto viene de `config/default.yaml` (CLAUDE.md §2.10): nunca
    se fija un literal de semilla dentro de este módulo.
    """
    seed_resuelto = _resolve_seed(seed)
    pasos = [("imputer", SimpleImputer(strategy="median"))]
    if name in _NECESITA_ESCALADO:
        pasos.append(("scaler", StandardScaler()))
    pasos.append(("classifier", _build_classifier(name, seed_resuelto)))
    return Pipeline(pasos)


def build_all_pipelines(seed: int | None = None) -> dict[str, Pipeline]:
    """Un `Pipeline` sin ajustar por cada modelo del zoo, en el orden de `MODEL_NAMES`."""
    return {name: build_pipeline(name, seed=seed) for name in MODEL_NAMES}
