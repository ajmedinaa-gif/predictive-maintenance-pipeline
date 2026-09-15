"""Construye el payload de resultados: `build_*_report` / `write_*_report` -> JSON.

Regla dura (CLAUDE.md §2.8): ningún número visible del repositorio se escribe a
mano. Se lee de `reports/*.json`, y este módulo es quien lo genera.

Este módulo NUNCA dibuja: si necesita rutas de figuras, importa `figures.py`.
En la Fase 5 crece con el renderizador HTML de jinja2, que lee este mismo JSON.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from predictive_maintenance import datasets, eda, evaluate

if TYPE_CHECKING:
    from predictive_maintenance import plausibility
    from predictive_maintenance.data import ValidationResult

TEMPLATES_DIR = Path(__file__).parent / "templates"

_COLUMNAS_METRICAS = ("fold", "n_test", "n_positivos_test")
_COLUMNAS_TABLA_RESULTADOS = (
    ("average_precision", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
    ("recall", "recall"),
    ("precision", "precision"),
    ("balanced_accuracy", "bal.acc"),
    ("accuracy", "accuracy"),
    ("brier_score_loss", "Brier"),
)


def _jsonable(obj):
    """Convierte tipos de numpy/pandas a tipos nativos serializables."""
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def build_eda_report(df: pd.DataFrame, spec: datasets.DatasetSpec) -> dict:
    """Reúne en un solo diccionario todo lo que el EDA sabe del dataset."""
    target = spec.target
    positive = spec.positive_label
    return {
        "dataset": spec.name,
        "fichero": str(spec.path.relative_to(datasets.PROJECT_ROOT)),
        "datos_simulados": spec.simulated,
        "n_filas": len(df),
        "n_columnas": int(df.shape[1]),
        "objetivo": target,
        "clase_positiva": positive,
        "balance_de_clases": _jsonable(eda.class_balance(df, target, positive)),
        "descriptiva": _jsonable(eda.descriptive_table(df, target)),
        "nulos": _jsonable(eda.missingness_report(df, target)),
        "anomalias_fisicas": _jsonable(eda.physical_anomalies(df)),
        "rangos_fisicos": {k: list(v) for k, v in eda.PHYSICAL_RANGES.items()},
        "asociacion_univariante": _jsonable(eda.univariate_auc(df, target, positive)),
        "correlaciones": {
            "matriz": _jsonable(eda.correlation_matrix(df, target).round(6)),
            "maxima_absoluta": _jsonable(eda.max_abs_correlation(df, target)),
        },
    }


def write_json_report(payload: dict, path: Path) -> Path:
    """Vuelca cualquier payload de informe a JSON, indentado y legible.

    Es la única función que escribe un `reports/*.json` en todo el proyecto
    (CLAUDE.md §2.8): ningún número visible del repositorio se escribe a mano.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def write_eda_report(report: dict, path: Path) -> Path:
    """Vuelca el informe de EDA a JSON. Alias de `write_json_report`."""
    return write_json_report(report, path)


def build_validation_report(result: ValidationResult, spec: datasets.DatasetSpec) -> dict:
    """Payload del contrato de datos: qué pasó, qué se puso en cuarentena y por qué."""
    return {
        "dataset": spec.name,
        "fichero": str(spec.path.relative_to(datasets.PROJECT_ROOT)),
        **result.report,
        "cuarentena": _jsonable(result.quarantined),
    }


def write_validation_report(report: dict, path: Path) -> Path:
    """Vuelca el informe de validación del contrato a JSON."""
    return write_json_report(report, path)


def write_markdown_report(texto: str, path: Path) -> Path:
    """Vuelca un informe ya renderizado en markdown a disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texto, encoding="utf-8")
    return path


def build_metrics_report(
    *,
    dataset: str,
    n_filas: int,
    n_positivos: int,
    resultados_por_modelo: dict[str, dict],
    protocolo_principal: str,
    protocolo_matriz_confusion: str,
    seed: int,
    nested_cv: dict | None = None,
) -> dict:
    """Payload de la Fase 3: métricas por modelo (media + IC bootstrap) y matriz de confusión.

    `resultados_por_modelo[nombre]` trae `"folds"` (el `DataFrame` de
    `evaluate.cross_validate_model`, una fila por fold) y
    `"matriz_confusion"` (el dict de `evaluate.aggregate_confusion_matrix`).
    El orden de `resultados_por_modelo` se conserva tal cual en el payload:
    quien lo construye es responsable de poner los `dummy_*` primero
    (CLAUDE.md §2.1).
    """
    modelos = {}
    for nombre, resultado in resultados_por_modelo.items():
        folds = resultado["folds"]
        columnas_metrica = [c for c in folds.columns if c not in _COLUMNAS_METRICAS]
        metricas = {}
        for metrica in columnas_metrica:
            valores = folds[metrica].to_numpy()
            media = float(np.nanmean(valores))
            ic_lower, ic_upper = evaluate.bootstrap_ci(valores, seed=seed)
            metricas[metrica] = {"media": media, "ic_bootstrap_95": [ic_lower, ic_upper]}
        modelos[nombre] = {
            "metricas": metricas,
            "matriz_confusion": _jsonable(resultado["matriz_confusion"]),
        }

    payload = {
        "dataset": dataset,
        "n_filas_entrenamiento": n_filas,
        "n_positivos": n_positivos,
        "protocolo_principal": protocolo_principal,
        "protocolo_matriz_confusion": protocolo_matriz_confusion,
        "modelos": modelos,
    }
    if nested_cv is not None:
        payload["nested_cv"] = _jsonable(nested_cv)
    return _jsonable(payload)


def write_metrics_report(report: dict, path: Path) -> Path:
    """Vuelca el informe de métricas de la Fase 3 a JSON."""
    return write_json_report(report, path)


def render_results_markdown(payload: dict) -> str:
    """Tabla markdown de resultados a partir de `build_metrics_report`.

    Regla dura (CLAUDE.md §2.1): la accuracy nunca va sola, siempre junto a la
    fila del `DummyClassifier`. El orden de las filas es el orden de
    `payload["modelos"]`; construir ese diccionario con los `dummy_*` primero
    es responsabilidad de quien llama.
    """
    encabezado = ["modelo", *(etiqueta for _, etiqueta in _COLUMNAS_TABLA_RESULTADOS)]
    lineas = [
        "| " + " | ".join(encabezado) + " |",
        "|" + "|".join(["---"] * len(encabezado)) + "|",
    ]
    for nombre, resultado in payload["modelos"].items():
        fila = [nombre]
        for clave, _ in _COLUMNAS_TABLA_RESULTADOS:
            fila.append(f"{resultado['metricas'][clave]['media']:.4f}")
        lineas.append("| " + " | ".join(fila) + " |")
    return "\n".join(lineas) + "\n"


def build_plausibility_report(informe: plausibility.PlausibilityReport) -> dict:
    """Payload del auditor de plausibilidad, listo para volcar a JSON."""
    return _jsonable(informe.to_dict())


def write_plausibility_report(report: dict, path: Path) -> Path:
    """Vuelca el veredicto del auditor de plausibilidad a JSON."""
    return write_json_report(report, path)


def build_calibration_report(
    *,
    dataset: str,
    n_filas: int,
    n_positivos: int,
    costs: dict,
    theoretical_threshold: float,
    protocolo: str,
    variantes: dict,
    reliability_curves: dict,
    brier_decompositions: dict,
    oof_y_true: list | None = None,
    oof_y_score: list | None = None,
) -> dict:
    """Payload de la Fase 4, tarea A: la tabla de CLAUDE.md §9.2, medida sobre `n_filas` filas.

    `variantes` es la salida de `calibration.evaluate_cost_variants`;
    `reliability_curves[nombre]` un `DataFrame` de `calibration.reliability_curve`;
    `brier_decompositions[nombre]` un `dict` de `calibration.brier_decomposition`.

    `oof_y_true`/`oof_y_score` son las predicciones out-of-fold del modelo de
    la decisión (`logistic_plain` + Platt, un único `StratifiedKFold(5)`) --
    las mismas que dibujan `cost_vs_threshold.png`. Se guardan en el JSON
    (Fase 5, Bloque B) para que la pestaña "Decisión" del dashboard pueda
    recalcular umbral, matriz de confusión y coste EN VIVO con
    `threshold.py` según los sliders de coste, sin reentrenar nada.
    """
    payload = {
        "dataset": dataset,
        "n_filas": n_filas,
        "n_positivos": n_positivos,
        "costs": costs,
        "umbral_teorico": theoretical_threshold,
        "protocolo": protocolo,
        "variantes": variantes,
        "curvas_fiabilidad": reliability_curves,
        "descomposicion_brier": brier_decompositions,
    }
    if oof_y_true is not None and oof_y_score is not None:
        payload["oof_decision_model"] = {"y_true": oof_y_true, "y_score": oof_y_score}
    return _jsonable(payload)


def write_calibration_report(report: dict, path: Path) -> Path:
    """Vuelca el informe de calibración y umbral por coste a JSON."""
    return write_json_report(report, path)


def build_explainability_report(
    *,
    dataset: str,
    n_filas: int,
    shap_importancia: dict,
    shap_casos_positivos: list,
    tree_stability: dict,
) -> dict:
    """Payload de la Fase 4 C: SHAP de `logistic_plain` y estabilidad de la raíz del árbol."""
    return _jsonable(
        {
            "dataset": dataset,
            "n_filas": n_filas,
            "shap_importancia_media": shap_importancia,
            "shap_casos_positivos": shap_casos_positivos,
            "estabilidad_raiz_arbol": tree_stability,
        }
    )


def write_explainability_report(report: dict, path: Path) -> Path:
    """Vuelca el informe de explicabilidad (SHAP + estabilidad del árbol) a JSON."""
    return write_json_report(report, path)


def build_limits_report(
    *,
    dataset: str,
    n_filas: int,
    n_positivos: int,
    prevalencia: float,
    recall_wilson_ci_ilustrativo: tuple,
    recall_wilson_ci_ilustrativo_aciertos: int,
    presupuesto: dict,
    learning_curve: pd.DataFrame,
) -> dict:
    """Payload de la Fase 4 C: presupuesto estadístico (CLAUDE.md §10.3) y curva de aprendizaje.

    `recall_wilson_ci_ilustrativo` es el IC de Wilson para un recall del 80 %
    ilustrativo, anclado a `n_positivos` REALES del dataset (no un ejemplo fijo
    de "8/10"): con `lab180` (10 positivos) reproduce el IC exacto de CLAUDE.md
    §10.3; con `ai4i2020` (339 positivos) muestra cuánto se estrecha con más
    datos (CLAUDE.md §13, Fase 5).

    IMPORTANTE: `recall_wilson_ci_ilustrativo_aciertos` (`round(0.8 *
    n_positivos)`) es un SUPUESTO de diseño muestral, no una medición -- quien
    lo muestre (CLI, dashboard, informe HTML) nunca debe llamarlo "aciertos"
    a secas, porque sugiere un conteo real. El recall MEDIDO del modelo de la
    decisión vive en `reports/comparison.json`
    (`datasets.<nombre>.recall_platt_tp` / `recall_wilson_ci_platt`).
    """
    return _jsonable(
        {
            "dataset": dataset,
            "n_filas": n_filas,
            "n_positivos": n_positivos,
            "prevalencia": prevalencia,
            "recall_wilson_ci_ilustrativo": list(recall_wilson_ci_ilustrativo),
            "recall_wilson_ci_ilustrativo_aciertos": recall_wilson_ci_ilustrativo_aciertos,
            "recall_wilson_ci_ilustrativo_n": n_positivos,
            "presupuesto_estadistico": presupuesto,
            "curva_aprendizaje_pr_auc": learning_curve,
        }
    )


def write_limits_report(report: dict, path: Path) -> Path:
    """Vuelca el informe de límites estadísticos a JSON."""
    return write_json_report(report, path)


def build_comparison_report(*, datasets_info: dict[str, dict]) -> dict:
    """Payload de la Fase 5, Bloque A: la tabla "un pipeline, dos datasets" (CLAUDE.md §13).

    `datasets_info[nombre]` trae, por dataset: `n_filas`, `n_positivos`,
    `prevalencia`, `dummy_accuracy`; `mejor_modelo_pr_auc_nombre` y
    `mejor_modelo_pr_auc_valor` son SOLO informativos (el modelo de mejor
    PR-AUC medio, el que dibuja `figure_pr_curves_two_datasets`) y nunca la
    base del recall ni del IC. El recall (`recall_platt_a_umbral_optimo`,
    `recall_platt_tp`/`recall_platt_n_positivos`) y su IC de Wilson
    (`recall_wilson_ci_platt`), el umbral (`umbral_optimo_platt`) y el ahorro
    (`ahorro_pct_platt`) son SIEMPRE del mismo modelo -- `logistic_plain` +
    Platt, la decisión fija del proyecto (CLAUDE.md §9.2) -- evaluado a SU
    umbral óptimo por coste, nunca a 0.5. Todo leído de
    `reports/metrics_*.json` y `reports/calibration_*.json` ya generados --
    ningún número se escribe a mano aquí.
    """
    return _jsonable({"datasets": datasets_info})


def write_comparison_report(report: dict, path: Path) -> Path:
    """Vuelca la comparación entre datasets a JSON."""
    return write_json_report(report, path)


# --------------------------------------------------------------------------- #
# Fase 5, Bloque B: informe HTML autocontenido (CLAUDE.md §15)
#
# Lee los `reports/*.json` ya generados por los demás comandos y las figuras
# PNG ya dibujadas por `figures.py` (embebidas en base64): este módulo sigue
# sin dibujar nada. Cero dependencias de red -- CSS y las imágenes van
# incrustadas en el propio HTML.
# --------------------------------------------------------------------------- #


def _leer_json(reports_dir: Path, nombre: str) -> dict | None:
    """Lee `reports/<nombre>.json` si existe; `None` si ese informe no se generó.

    Algunas secciones (p.ej. explicabilidad) pueden no existir para todos los
    datasets todavía -- el HTML se degrada mostrando "no generado", nunca
    lanza una excepción por un fichero ausente.
    """
    ruta = reports_dir / f"{nombre}.json"
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def _imagen_base64(path: Path) -> str | None:
    """`data:` URI de un PNG ya generado, o `None` si no existe todavía."""
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_html_report_context(dataset: str, reports_dir: Path, figures_dir: Path) -> dict:
    """Reúne todos los `reports/*.json` y figuras PNG de `dataset` en un único contexto.

    Ninguno de los números que acaba mostrando el HTML se calcula aquí: todos
    vienen ya calculados de los JSON que escriben `eda`, `validate`, `train`,
    `calibrate`, `explain` y `limits`. Si un JSON no existe, la sección
    correspondiente del contexto queda en `None` y la plantilla lo muestra
    como "no generado todavía" en vez de fallar.
    """
    fig_dir = figures_dir / dataset

    eda_r = _leer_json(reports_dir, f"eda_{dataset}")
    validation_r = _leer_json(reports_dir, f"validation_{dataset}")
    plausibility_r = _leer_json(reports_dir, f"plausibility_{dataset}")
    metrics_r = _leer_json(reports_dir, f"metrics_{dataset}")
    calibration_r = _leer_json(reports_dir, f"calibration_{dataset}")
    explainability_r = _leer_json(reports_dir, f"explainability_{dataset}")
    limits_r = _leer_json(reports_dir, f"limits_{dataset}")
    comparison_r = _leer_json(reports_dir, "comparison")
    comparison_dataset = comparison_r["datasets"].get(dataset) if comparison_r else None

    modelos_tabla = None
    if metrics_r is not None:
        modelos_tabla = []
        for nombre, resultado in metrics_r["modelos"].items():
            fila = {"nombre": nombre}
            for clave, etiqueta in _COLUMNAS_TABLA_RESULTADOS:
                fila[clave] = resultado["metricas"][clave]["media"]
                fila[f"{clave}_etiqueta"] = etiqueta
            fila["matriz_confusion"] = resultado["matriz_confusion"]
            modelos_tabla.append(fila)

    mejor_modelo = None
    if metrics_r is not None:
        nombre_mejor = max(
            metrics_r["modelos"],
            key=lambda n: metrics_r["modelos"][n]["metricas"]["average_precision"]["media"],
        )
        mejor_modelo = {
            "nombre": nombre_mejor,
            "pr_auc": metrics_r["modelos"][nombre_mejor]["metricas"]["average_precision"]["media"],
        }

    variantes_calibracion = None
    if calibration_r is not None:
        variantes_calibracion = calibration_r["variantes"]

    figuras = {
        "histogramas": _imagen_base64(fig_dir / "histogramas_por_clase.png"),
        "boxplots": _imagen_base64(fig_dir / "boxplots_por_clase.png"),
        "correlacion": _imagen_base64(fig_dir / "correlacion.png"),
        "prevalencia": _imagen_base64(fig_dir / "prevalencia.png"),
        "curvas_pr_roc": _imagen_base64(fig_dir / "curvas_pr_roc.png"),
        "calibration_curve": _imagen_base64(fig_dir / "calibration_curve.png"),
        "cost_vs_threshold": _imagen_base64(fig_dir / "cost_vs_threshold.png"),
        "shap_beeswarm": _imagen_base64(fig_dir / "shap_beeswarm.png"),
        "shap_waterfalls": _imagen_base64(fig_dir / "shap_waterfalls_positivos.png"),
        "tree_root_stability": _imagen_base64(fig_dir / "tree_root_stability.png"),
        "tree_render": _imagen_base64(fig_dir / "tree_render.png"),
        "learning_curve": _imagen_base64(fig_dir / "learning_curve.png"),
    }

    return {
        "dataset": dataset,
        "eda": eda_r,
        "validation": validation_r,
        "plausibility": plausibility_r,
        "metrics": metrics_r,
        "modelos_tabla": modelos_tabla,
        "mejor_modelo": mejor_modelo,
        "calibration": calibration_r,
        "variantes_calibracion": variantes_calibracion,
        "explainability": explainability_r,
        "limits": limits_r,
        "comparison_dataset": comparison_dataset,
        "figuras": figuras,
    }


def render_html_report(context: dict) -> str:
    """Renderiza `templates/report.html.jinja2` con `context` (Jinja2, `autoescape=True`)."""
    entorno = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),
    )
    plantilla = entorno.get_template("report.html.jinja2")
    return plantilla.render(**context)


def write_html_report(html: str, path: Path) -> Path:
    """Vuelca el informe HTML ya renderizado a disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
