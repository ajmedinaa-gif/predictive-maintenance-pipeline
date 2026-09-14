"""Tests de la fábrica de pipelines (CLAUDE.md §2.6)."""

import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from predictive_maintenance import pipeline


@pytest.mark.parametrize("nombre", pipeline.MODEL_NAMES)
def test_build_pipeline_starts_with_imputer(nombre):
    pipe = pipeline.build_pipeline(nombre, seed=42)
    assert isinstance(pipe.steps[0][1], SimpleImputer)


@pytest.mark.parametrize("nombre", ["logistic_balanced", "logistic_plain"])
def test_scale_sensitive_models_get_a_scaler(nombre):
    pipe = pipeline.build_pipeline(nombre, seed=42)
    assert isinstance(pipe.named_steps.get("scaler"), StandardScaler)


@pytest.mark.parametrize(
    "nombre",
    [
        "dummy_most_frequent",
        "dummy_stratified",
        "tree_default",
        "tree_shallow_balanced",
        "rf_balanced",
        "gradient_boosting",
    ],
)
def test_scale_invariant_models_get_no_scaler(nombre):
    pipe = pipeline.build_pipeline(nombre, seed=42)
    assert "scaler" not in pipe.named_steps


def test_logistic_plain_has_no_class_weight():
    pipe = pipeline.build_pipeline("logistic_plain", seed=42)
    assert pipe.named_steps["classifier"].class_weight is None


def test_logistic_balanced_has_class_weight():
    pipe = pipeline.build_pipeline("logistic_balanced", seed=42)
    assert pipe.named_steps["classifier"].class_weight == "balanced"


def test_tree_shallow_balanced_has_the_documented_hyperparameters():
    clasificador = pipeline.build_pipeline("tree_shallow_balanced", seed=42).named_steps[
        "classifier"
    ]
    assert clasificador.max_depth == 3
    assert clasificador.min_samples_leaf == 5
    assert clasificador.class_weight == "balanced"


def test_gradient_boosting_has_no_class_weight_option():
    # GradientBoostingClassifier no admite `class_weight` en scikit-learn.
    clasificador = pipeline.build_pipeline("gradient_boosting", seed=42).named_steps["classifier"]
    assert not hasattr(clasificador, "class_weight")


def test_unknown_model_raises_key_error():
    with pytest.raises(KeyError):
        pipeline.build_pipeline("no_existe", seed=42)


def test_seed_propagates_to_the_classifier():
    pipe = pipeline.build_pipeline("tree_default", seed=7)
    assert pipe.named_steps["classifier"].random_state == 7


def test_seed_defaults_to_global_config_seed():
    pipe = pipeline.build_pipeline("tree_default")
    assert pipe.named_steps["classifier"].random_state == 42


def test_build_all_pipelines_returns_one_per_model_name():
    todas = pipeline.build_all_pipelines(seed=42)
    assert set(todas) == set(pipeline.MODEL_NAMES)


def test_dummy_models_are_first_in_model_names():
    assert pipeline.MODEL_NAMES[0] == "dummy_most_frequent"
    assert pipeline.MODEL_NAMES[1] == "dummy_stratified"


@pytest.mark.parametrize("nombre", pipeline.MODEL_NAMES)
def test_every_pipeline_fits_and_predicts_on_lab180(nombre, lab180_df):
    df = lab180_df.drop(index=82)  # sin la anomalía física (CLAUDE.md §6.3)
    X = df.drop(columns=["failure"])
    y = df["failure"].astype(str)
    pipe = pipeline.build_pipeline(nombre, seed=42)
    pipe.fit(X, y)
    predicciones = pipe.predict(X)
    assert len(predicciones) == len(y)
    assert set(predicciones) <= {"no", "yes"}
