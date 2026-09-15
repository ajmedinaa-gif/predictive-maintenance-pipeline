"""Interfaz de línea de comandos del proyecto.

Regla dura (CLAUDE.md §2.11): las descargas externas son un paso explícito del
CLI (`download`), nunca un efecto lateral de importar un módulo o de correr
otro comando. El resto de comandos son deliberadamente los mismos para
cualquier dataset registrado en `datasets.ADAPTERS` (CLAUDE.md §13.1).
"""

from __future__ import annotations

import json
import logging

import typer

from predictive_maintenance import (
    __version__,
    calibration,
    data,
    datasets,
    eda,
    evaluate,
    explain,
    figures,
    pipeline,
    plausibility,
    power,
    report,
    threshold,
)
from predictive_maintenance.config import get_costs, get_settings

app = typer.Typer(
    name="pdm-cli",
    help="Mantenimiento predictivo industrial: pipeline reproducible con "
    "validación honesta bajo desbalance de clases.",
    no_args_is_help=True,
)


def _mostrar_ruta(ruta) -> str:
    """Ruta relativa a la raíz del proyecto para el mensaje en pantalla.

    Si `ruta` no cuelga de `PROJECT_ROOT` (por ejemplo, en un test que redirige
    la salida a un directorio temporal), se muestra la ruta absoluta.
    """
    try:
        return str(ruta.relative_to(datasets.PROJECT_ROOT))
    except ValueError:
        return str(ruta)


@app.command()
def version() -> None:
    """Muestra la versión instalada del paquete."""
    typer.echo(f"predictive-maintenance-pipeline {__version__}")


@app.command(name="download")
def download_command(
    dataset: str = typer.Option(..., "--dataset", help="Dataset a descargar (ej. ai4i2020)."),
) -> None:
    """Descarga explícita de un dataset externo (CLAUDE.md §2.11).

    Nunca ocurre como efecto lateral de otro comando: ningún módulo del
    pipeline llama a esto por su cuenta. `lab180` no necesita red -- viene
    versionado en el repo -- pero el comando también acepta ese nombre y solo
    verifica que el fichero exista.
    """
    _configurar_logging()
    adapter = datasets.get_adapter(dataset)
    ruta = adapter.download()
    typer.echo(f"Dataset {dataset!r} listo en: {_mostrar_ruta(ruta)}")


@app.command(name="eda")
def eda_command(
    dataset: str = typer.Option("lab180", "--dataset", help="Nombre del dataset a analizar."),
) -> None:
    """Ejecuta el análisis exploratorio de `dataset` y lo vuelca a reports/."""
    df = datasets.load(dataset)
    spec = datasets.get_spec(dataset)

    balance = eda.class_balance(df, spec.target, spec.positive_label)
    descriptiva = eda.descriptive_table(df, spec.target)
    nulos = eda.missingness_report(df, spec.target)
    anomalias = eda.physical_anomalies(df)
    univariante = eda.univariate_auc(df, spec.target, spec.positive_label)
    max_corr = eda.max_abs_correlation(df, spec.target)

    typer.echo(f"\n=== {dataset}: {spec.path.name} ({balance['n']} filas) ===\n")
    typer.echo("--- Balance de clases ---")
    typer.echo(
        f"{spec.target}: {balance['conteos']}  ->  prevalencia "
        f"{balance['prevalencia_pct']:.4f} %  |  accuracy trivial "
        f"{balance['accuracy_trivial_mayoritario_pct']:.4f} %"
    )
    typer.echo("\n--- Tabla descriptiva ---")
    typer.echo(descriptiva.round(4).to_string())
    typer.echo("\n--- Nulos ---")
    typer.echo(nulos["lectura"])
    typer.echo("\n--- Anomalías físicas (van a cuarentena) ---")
    typer.echo(anomalias.to_string() if not anomalias.empty else "(ninguna)")
    typer.echo("\n--- Asociación univariante (AUC de Mann-Whitney) ---")
    typer.echo(univariante.round(4).to_string())
    typer.echo(
        f"\n--- Correlación máxima |r| = {max_corr['max_abs_r']:.4f} entre {max_corr['par']} ---"
    )

    config = datasets.load_config()
    reports_dir = datasets.PROJECT_ROOT / config["paths"]["reports"]
    figures_dir = datasets.PROJECT_ROOT / config["paths"]["figures"] / dataset

    informe = report.build_eda_report(df, spec)
    ruta_json = report.write_eda_report(informe, reports_dir / f"eda_{dataset}.json")
    rutas_figuras = figures.make_eda_figures(df, spec, figures_dir)

    typer.echo(f"\nInforme JSON: {_mostrar_ruta(ruta_json)}")
    for ruta in rutas_figuras:
        typer.echo(f"Figura: {_mostrar_ruta(ruta)}")


def _configurar_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )


def _validar_y_auditar(dataset: str) -> dict:
    """Cuerpo compartido de `validate` y `run`: contrato de datos + plausibilidad.

    Imprime y vuelca los dos informes a `reports/`; devuelve el resumen de
    validación para que quien llame decida qué hacer con el código de salida.
    """
    spec = datasets.get_spec(dataset)
    adapter = datasets.get_adapter(dataset)

    resultado = data.load_validated(spec.path, adapter.schema, dataset_name=dataset)

    typer.echo(f"\n=== Contrato de datos: {dataset} ({spec.path.name}) ===\n")
    typer.echo(
        f"{resultado.report['n_filas_leidas']} filas leídas -> "
        f"{resultado.report['n_validas']} válidas, "
        f"{resultado.report['n_cuarentena']} en cuarentena"
    )
    if resultado.report["n_cuarentena"]:
        typer.echo(f"Cuarentena escrita en: {resultado.report['ruta_cuarentena']}")
        for motivo in resultado.report["motivos_cuarentena"]:
            typer.echo(f"  fila {motivo['index']}: {motivo['quarantine_reason']}")
    if resultado.report["avisos_calidad"]:
        typer.echo("\n--- Avisos de calidad ---")
        for aviso in resultado.report["avisos_calidad"]:
            typer.echo(f"  ! {aviso}")
    else:
        typer.echo("Sin avisos de calidad.")

    config = datasets.load_config()
    reports_dir = datasets.PROJECT_ROOT / config["paths"]["reports"]

    informe_validacion = report.build_validation_report(resultado, spec)
    ruta_validacion = report.write_validation_report(
        informe_validacion, reports_dir / f"validation_{dataset}.json"
    )
    typer.echo(f"\nInforme de validación: {_mostrar_ruta(ruta_validacion)}")

    df_completo = datasets.load(dataset)
    auditoria = plausibility.audit(df_completo, dataset=dataset, target=spec.target)
    informe_plausibilidad = report.build_plausibility_report(auditoria)
    ruta_plausibilidad = report.write_plausibility_report(
        informe_plausibilidad, reports_dir / f"plausibility_{dataset}.json"
    )
    typer.echo("\n--- Auditoría de plausibilidad ---")
    typer.echo(f"Veredicto: {auditoria.veredicto}")
    for evidencia in auditoria.evidencia:
        typer.echo(f"  - {evidencia}")
    typer.echo(f"Informe de plausibilidad: {_mostrar_ruta(ruta_plausibilidad)}")

    return resultado.report


@app.command(name="validate")
def validate_command(
    dataset: str = typer.Option("lab180", "--dataset", help="Nombre del dataset a validar."),
) -> None:
    """Valida `dataset` contra su contrato y audita su plausibilidad física.

    Sale con código 1 si alguna fila terminó en cuarentena, para que un CI
    pueda usarlo como puerta de calidad (CLAUDE.md §2.9).
    """
    _configurar_logging()
    resumen = _validar_y_auditar(dataset)
    if resumen["n_cuarentena"]:
        raise typer.Exit(code=1)


def _entrenar_y_reportar(dataset: str) -> None:
    """Cuerpo compartido de `train` y `run`: compara el zoo de modelos (CLAUDE.md §8).

    Solo entrena sobre las filas que pasan el contrato de datos y con las
    features ya derivadas por el `DatasetAdapter` (CLAUDE.md §2.9, §13): la
    fila en cuarentena de `lab180` nunca entra al split de CV, y `ai4i2020`
    entra con `power_w`/`temp_delta_k`/`wear_x_torque`/`type_*` ya calculados.
    """
    settings = get_settings()
    spec, df, X, y = _cargar_xy_validado(dataset)
    n_positivos = int((y == spec.positive_label).sum())

    cfg = evaluate.EvalConfig.for_dataset(dataset, settings)
    protocolo_principal = (
        f"RepeatedStratifiedKFold(n_splits={cfg.n_splits}, n_repeats={cfg.n_repeats}, "
        f"random_state={cfg.seed})"
    )
    protocolo_matriz = f"StratifiedKFold(n_splits=5, shuffle=True, random_state={cfg.seed})"

    typer.echo(f"\n=== Entrenando {dataset}: {len(df)} filas válidas, {n_positivos} positivos ===")
    typer.echo(f"Protocolo principal: {protocolo_principal}\n")

    resultados_por_modelo: dict[str, dict] = {}
    curvas: dict[str, tuple] = {}
    for nombre in pipeline.MODEL_NAMES:
        typer.echo(f"  - {nombre}")
        pipe = pipeline.build_pipeline(nombre, seed=cfg.seed)
        folds = evaluate.cross_validate_model(pipe, X, y, cfg, positive_label=spec.positive_label)
        matriz = evaluate.aggregate_confusion_matrix(
            pipe, X, y, seed=cfg.seed, n_splits=5, positive_label=spec.positive_label
        )
        resultados_por_modelo[nombre] = {"folds": folds, "matriz_confusion": matriz}

        y_true_bin, y_score, _ = evaluate.out_of_fold_predictions(
            pipe, X, y, seed=cfg.seed, n_splits=5, positive_label=spec.positive_label
        )
        curvas[nombre] = (y_true_bin, y_score)

    typer.echo(
        "\nCV anidada de demostración (Vabalas et al. 2019): logistic_balanced, "
        "C en {0.01, 0.1, 1, 10}"
    )
    pipe_logistic = pipeline.build_pipeline("logistic_balanced", seed=cfg.seed)
    grid_logistic = {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
    tabla_nested = evaluate.nested_cv(
        pipe_logistic, grid_logistic, X, y, cfg, positive_label=spec.positive_label
    )
    conteo_c = tabla_nested["mejores_hiperparametros"].apply(lambda d: d["classifier__C"])
    nested_cv_payload = {
        "modelo": "logistic_balanced",
        "param_grid": grid_logistic,
        "average_precision_media": float(tabla_nested["average_precision"].mean()),
        "roc_auc_media": float(tabla_nested["roc_auc"].mean()),
        "conteo_mejor_c": {str(k): int(v) for k, v in conteo_c.value_counts().items()},
    }

    config_yaml = datasets.load_config()
    reports_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["reports"]
    figures_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["figures"] / dataset
    figures_dir.mkdir(parents=True, exist_ok=True)

    informe = report.build_metrics_report(
        dataset=dataset,
        n_filas=len(df),
        n_positivos=n_positivos,
        resultados_por_modelo=resultados_por_modelo,
        protocolo_principal=protocolo_principal,
        protocolo_matriz_confusion=protocolo_matriz,
        seed=cfg.seed,
        nested_cv=nested_cv_payload,
    )
    ruta_json = report.write_metrics_report(informe, reports_dir / f"metrics_{dataset}.json")

    tabla_md = report.render_results_markdown(informe)
    ruta_md = report.write_markdown_report(tabla_md, reports_dir / f"results_{dataset}.md")

    prevalencia = n_positivos / len(df)
    ruta_figura = figures.figure_pr_roc_curves(
        curvas, prevalencia, spec, figures_dir / "curvas_pr_roc.png"
    )

    typer.echo("\n--- Resultados (media sobre folds) ---")
    typer.echo(tabla_md)
    typer.echo(f"Informe JSON: {_mostrar_ruta(ruta_json)}")
    typer.echo(f"Tabla de resultados: {_mostrar_ruta(ruta_md)}")
    typer.echo(f"Figura: {_mostrar_ruta(ruta_figura)}")


@app.command(name="train")
def train_command(
    dataset: str = typer.Option("lab180", "--dataset", help="Nombre del dataset a entrenar."),
) -> None:
    """Compara el zoo de modelos de `pipeline.py` con el protocolo de CLAUDE.md §8."""
    _configurar_logging()
    _entrenar_y_reportar(dataset)


@app.command(name="run")
def run_command(
    dataset: str = typer.Option("lab180", "--dataset", help="Nombre del dataset a ejecutar."),
) -> None:
    """Pipeline completo sobre `dataset`: contrato de datos + comparación de modelos.

    Es la prueba de que el código es una abstracción y no un script atado a
    un CSV (CLAUDE.md §13.1): el mismo comando, cambiando solo `--dataset`,
    corre sobre `lab180` (5 x 10 repeticiones, 10 positivos) y sobre
    `ai4i2020` (5 folds sin repetir, ~339 positivos). A diferencia de
    `validate`, `run` nunca sale con código 1 por cuarentena -- reporta el
    conteo y sigue al entrenamiento -- porque `lab180` cuarentena la fila 82
    por diseño en cada ejecución.
    """
    _configurar_logging()
    typer.echo(f"\n########## pdm-cli run --dataset {dataset} ##########")
    _validar_y_auditar(dataset)
    _entrenar_y_reportar(dataset)


def _cargar_xy_validado(dataset: str) -> tuple:
    """Contrato de datos + ingeniería de features del `DatasetAdapter` (CLAUDE.md §13).

    Devuelve `(spec, df, X, y)` con `X` ya restringido y ordenado según
    `spec.feature_columns` -- nunca incluye el objetivo ni, para `ai4i2020`,
    los cinco modos de fallo (fuga de objetivo).
    """
    spec = datasets.get_spec(dataset)
    adapter = datasets.get_adapter(dataset)
    resultado = data.load_validated(spec.path, adapter.schema, dataset_name=dataset)
    df = adapter.engineer_features(resultado.valid)
    X = df[list(spec.feature_columns)]
    y = df[spec.target].astype(str)
    return spec, df, X, y


@app.command(name="calibrate")
def calibrate_command(
    dataset: str = typer.Option("lab180", "--dataset", help="Nombre del dataset a calibrar."),
) -> None:
    """Calibra `logistic_plain`, reproduce CLAUDE.md §9.2 y optimiza el umbral por coste.

    Regla dura (CLAUDE.md §2.5): NUNCA combina `class_weight="balanced"` con
    el umbral por coste — `logistic_balanced` entra en la comparación solo
    como contraejemplo de esa regla.
    """
    _configurar_logging()
    settings = get_settings()
    spec, df, X, y = _cargar_xy_validado(dataset)
    n_positivos = int((y == spec.positive_label).sum())

    costs = get_costs().as_dict()
    t_teorico = threshold.theoretical_threshold(costs)

    typer.echo(f"\n=== Calibrando {dataset}: {len(df)} filas, {n_positivos} positivos ===")
    typer.echo(f"Umbral teórico t* = {t_teorico:.4f} (CLAUDE.md §9.1)\n")

    variantes = calibration.evaluate_cost_variants(
        X, y, costs, seed=settings.seed, positive_label=spec.positive_label
    )

    reliability_curves = {}
    brier_decomps = {}
    for nombre, builder in calibration.VARIANT_BUILDERS.items():
        y_true_oof, y_score_oof, _ = evaluate.out_of_fold_predictions(
            builder(settings.seed, 3),
            X,
            y,
            seed=settings.seed,
            n_splits=5,
            positive_label=spec.positive_label,
        )
        reliability_curves[nombre] = calibration.reliability_curve(
            y_true_oof, y_score_oof, n_bins=10
        )
        brier_decomps[nombre] = calibration.brier_decomposition(y_true_oof, y_score_oof, n_bins=10)

    for nombre, resultado in variantes.items():
        typer.echo(
            f"  {calibration.VARIANT_LABELS[nombre]:28s} Brier={resultado['brier_score']:.4f}  "
            f"PR-AUC={resultado['pr_auc']:.4f}  t_empirico={resultado['umbral_empirico']:.4f}  "
            f"ahorro={resultado['ahorro_pct']:.1f}%"
        )

    config_yaml = datasets.load_config()
    reports_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["reports"]
    figures_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["figures"] / dataset
    figures_dir.mkdir(parents=True, exist_ok=True)

    # El modelo que decide el proyecto (CLAUDE.md §9.2): logistic_plain + Platt.
    # Sus predicciones out-of-fold alimentan el gráfico de coste Y se guardan
    # en el JSON (Fase 5, Bloque B) para que la pestaña "Decisión" del
    # dashboard recalcule umbral/matriz/coste en vivo sin reentrenar nada.
    y_true_oof_mejor, y_score_oof_mejor, _ = evaluate.out_of_fold_predictions(
        calibration.VARIANT_BUILDERS["logistic_plain_platt"](settings.seed, 3),
        X,
        y,
        seed=settings.seed,
        n_splits=5,
        positive_label=spec.positive_label,
    )

    informe = report.build_calibration_report(
        dataset=dataset,
        n_filas=len(df),
        n_positivos=n_positivos,
        costs=costs,
        theoretical_threshold=t_teorico,
        protocolo=(
            f"StratifiedKFold(n_splits=5, shuffle=True, random_state={settings.seed}); "
            "umbral optimizado dentro de cada fold de entrenamiento (CLAUDE.md §2.7)"
        ),
        variantes=variantes,
        reliability_curves=reliability_curves,
        brier_decompositions=brier_decomps,
        oof_y_true=y_true_oof_mejor.tolist(),
        oof_y_score=y_score_oof_mejor.tolist(),
    )
    ruta_json = report.write_calibration_report(
        informe, reports_dir / f"calibration_{dataset}.json"
    )

    ruta_calibracion = figures.figure_calibration_curve(
        {
            nombre: reliability_curves[nombre]
            for nombre in ("logistic_plain", "logistic_plain_platt", "logistic_plain_isotonic")
        },
        spec,
        figures_dir / "calibration_curve.png",
    )

    curva_coste = threshold.cost_curve(y_true_oof_mejor, y_score_oof_mejor, costs)
    t_empirico_mejor = variantes["logistic_plain_platt"]["umbral_empirico"]
    ruta_coste = figures.figure_cost_vs_threshold(
        curva_coste, 0.5, t_empirico_mejor, spec, figures_dir / "cost_vs_threshold.png"
    )

    typer.echo(f"\nInforme JSON: {_mostrar_ruta(ruta_json)}")
    typer.echo(f"Figura: {_mostrar_ruta(ruta_calibracion)}")
    typer.echo(f"Figura: {_mostrar_ruta(ruta_coste)}")


@app.command(name="explain")
def explain_command(
    dataset: str = typer.Option("lab180", "--dataset", help="Nombre del dataset a explicar."),
) -> None:
    """SHAP de `logistic_plain` y estabilidad de la raíz del árbol (CLAUDE.md §10.1, §12)."""
    _configurar_logging()
    settings = get_settings()
    spec, df, X, y = _cargar_xy_validado(dataset)

    typer.echo(f"\n=== Explicabilidad de {dataset}: {len(df)} filas ===\n")

    shap_resultado = explain.shap_values_logistic(
        X, y, seed=settings.seed, positive_label=spec.positive_label
    )
    importancia, casos_positivos = explain.summarize_shap(shap_resultado, y)
    typer.echo("--- Importancia media |SHAP| ---")
    for atributo, valor in importancia.items():
        typer.echo(f"  {atributo:28s} {valor:.4f}")

    typer.echo("\n--- Estabilidad de la raíz del árbol (300 bootstraps estratificados) ---")
    stability = explain.tree_root_stability(X, y, n_boot=300, seed=settings.seed)
    for atributo, pct in stability["porcentaje_raiz_por_atributo"].items():
        typer.echo(f"  {atributo:28s} {pct:5.1f} %")
    typer.echo(f"  profundidad efectiva media: {stability['profundidad_efectiva_media']:.2f}")

    config_yaml = datasets.load_config()
    reports_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["reports"]
    figures_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["figures"] / dataset
    figures_dir.mkdir(parents=True, exist_ok=True)

    informe = report.build_explainability_report(
        dataset=dataset,
        n_filas=len(df),
        shap_importancia=importancia,
        shap_casos_positivos=casos_positivos,
        tree_stability=stability,
    )
    ruta_json = report.write_explainability_report(
        informe, reports_dir / f"explainability_{dataset}.json"
    )

    ruta_beeswarm = figures.figure_shap_beeswarm(
        shap_resultado, X, spec, figures_dir / "shap_beeswarm.png"
    )
    ruta_waterfalls = figures.figure_shap_waterfalls_positives(
        shap_resultado,
        y,
        spec,
        figures_dir / "shap_waterfalls_positivos.png",
        max_cases=20,
    )
    ruta_estabilidad = figures.figure_tree_root_stability(
        stability, spec, figures_dir / "tree_root_stability.png"
    )
    arbol = explain.fit_tree_shallow_balanced(X, y, seed=settings.seed)
    ruta_arbol = figures.figure_tree_render(
        arbol, list(X.columns), spec, figures_dir / "tree_render.png"
    )

    typer.echo(f"\nInforme JSON: {_mostrar_ruta(ruta_json)}")
    for ruta in (ruta_beeswarm, ruta_waterfalls, ruta_estabilidad, ruta_arbol):
        typer.echo(f"Figura: {_mostrar_ruta(ruta)}")


@app.command(name="limits")
def limits_command(
    dataset: str = typer.Option("lab180", "--dataset", help="Nombre del dataset a analizar."),
) -> None:
    """Presupuesto estadístico y curva de aprendizaje (CLAUDE.md §10.2, §10.3)."""
    _configurar_logging()
    settings = get_settings()
    spec, df, X, y = _cargar_xy_validado(dataset)
    n_positivos = int((y == spec.positive_label).sum())
    prevalencia = n_positivos / len(df)

    typer.echo(
        f"\n=== Límites estadísticos de {dataset}: {len(df)} filas, {n_positivos} positivos ===\n"
    )

    # Ilustración del IC de Wilson anclada a la escala REAL de `dataset`, no a
    # un ejemplo fijo: con `lab180` (10 positivos) esto reproduce exactamente
    # el "8/10" de CLAUDE.md §10.3; con `ai4i2020` (339 positivos) ilustra en
    # cambio cuánto se estrecha el IC al tener más de 30 veces más positivos.
    aciertos_ilustrativos = round(0.8 * n_positivos)
    ic_ilustrativo = evaluate.recall_wilson_ci(aciertos_ilustrativos, n_positivos)
    typer.echo(
        f"IC de Wilson del recall ({aciertos_ilustrativos}/{n_positivos} aciertos, "
        f"80% ilustrativo): [{ic_ilustrativo[0]:.3f}, {ic_ilustrativo[1]:.3f}]"
    )

    presupuesto = {}
    for margen in (0.10, 0.05):
        n_necesarios = power.required_positives(target_recall=0.80, margin=margen)
        n_obs = power.required_observations(n_necesarios, prevalencia)
        presupuesto[f"margen_{margen:.2f}"] = {
            "margen_pp": margen * 100,
            "fallos_necesarios": n_necesarios,
            "ciclos_de_maquina_necesarios": n_obs,
        }
        typer.echo(
            f"  margen ±{margen * 100:.0f} pp -> {n_necesarios} fallos "
            f"-> {n_obs} ciclos de máquina a la prevalencia actual"
        )

    curva_aprendizaje = power.learning_curve_pr_auc(X, y, seed=settings.seed)
    typer.echo(
        "\n--- Curva de aprendizaje PR-AUC (NO es creciente, bandas anchas: CLAUDE.md §10.2) ---"
    )
    typer.echo(curva_aprendizaje.round(4).to_string(index=False))

    config_yaml = datasets.load_config()
    reports_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["reports"]
    figures_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["figures"] / dataset
    figures_dir.mkdir(parents=True, exist_ok=True)

    informe = report.build_limits_report(
        dataset=dataset,
        n_filas=len(df),
        n_positivos=n_positivos,
        prevalencia=prevalencia,
        recall_wilson_ci_ilustrativo=ic_ilustrativo,
        recall_wilson_ci_ilustrativo_aciertos=aciertos_ilustrativos,
        presupuesto=presupuesto,
        learning_curve=curva_aprendizaje,
    )
    ruta_json = report.write_limits_report(informe, reports_dir / f"limits_{dataset}.json")
    ruta_figura = figures.figure_learning_curve(
        curva_aprendizaje, spec, figures_dir / "learning_curve.png"
    )

    typer.echo(f"\nInforme JSON: {_mostrar_ruta(ruta_json)}")
    typer.echo(f"Figura: {_mostrar_ruta(ruta_figura)}")


@app.command(name="report")
def report_command(
    dataset: str = typer.Option("lab180", "--dataset", help="Nombre del dataset a informar."),
) -> None:
    """Informe HTML autocontenido (CLAUDE.md §15): `reports/report_<dataset>.html`.

    Lee los `reports/*.json` y las figuras PNG ya generados por `eda`,
    `validate`, `run`, `calibrate`, `explain` y `limits` -- no reentrena ni
    dibuja nada. Las secciones cuyo informe aún no existe se muestran como
    "no generado todavía" en vez de fallar. Cero dependencias de red: CSS y
    figuras van incrustados en el propio HTML (figuras en base64).
    """
    _configurar_logging()
    config_yaml = datasets.load_config()
    reports_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["reports"]
    figures_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["figures"]

    contexto = report.build_html_report_context(dataset, reports_dir, figures_dir)
    html = report.render_html_report(contexto)
    ruta_html = report.write_html_report(html, reports_dir / f"report_{dataset}.html")

    secciones_ausentes = [
        clave
        for clave in ("eda", "validation", "plausibility", "metrics", "calibration", "limits")
        if contexto[clave] is None
    ]
    typer.echo(f"Informe HTML: {_mostrar_ruta(ruta_html)}")
    if secciones_ausentes:
        typer.echo(f"Secciones sin generar todavía: {', '.join(secciones_ausentes)}")


@app.command(name="compare")
def compare_command() -> None:
    """Tabla y figura comparativas de "un pipeline, dos datasets" (CLAUDE.md §13).

    Requiere que `train`, `calibrate` y `limits` ya se hayan ejecutado sobre
    `lab180` y `ai4i2020` -- lee sus `reports/*.json`, nunca reentrana nada.
    Escribe `reports/comparison.json` y
    `reports/figures/comparacion_pr.png`.
    """
    _configurar_logging()
    config_yaml = datasets.load_config()
    reports_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["reports"]
    figures_dir = datasets.PROJECT_ROOT / config_yaml["paths"]["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    datasets_info: dict[str, dict] = {}
    curvas: dict[str, tuple] = {}
    prevalencias: dict[str, float] = {}
    etiquetas_modelo: dict[str, str] = {}

    for nombre_dataset in ("lab180", "ai4i2020"):
        metrics_path = reports_dir / f"metrics_{nombre_dataset}.json"
        calibration_path = reports_dir / f"calibration_{nombre_dataset}.json"
        if not metrics_path.exists() or not calibration_path.exists():
            typer.echo(
                f"Faltan informes de {nombre_dataset!r}: ejecuta antes "
                f"`pdm-cli run --dataset {nombre_dataset}` y "
                f"`pdm-cli calibrate --dataset {nombre_dataset}`."
            )
            raise typer.Exit(code=2)

        metricas = json.loads(metrics_path.read_text(encoding="utf-8"))
        calibracion = json.loads(calibration_path.read_text(encoding="utf-8"))

        modelos = metricas["modelos"]
        # "Mejor modelo por PR-AUC": SOLO informativo (para la figura de curvas
        # PR y su propia fila en la tabla). NUNCA es la base del recall ni del
        # IC de la tabla comparativa -- eso sería reportar el recall al umbral
        # 0.5 (`aggregate_confusion_matrix` lo mide ahí) justo después de
        # argumentar que 0.5 no tiene sentido económico en este problema.
        mejor_pr_auc_nombre = max(
            modelos, key=lambda nombre: modelos[nombre]["metricas"]["average_precision"]["media"]
        )
        mejor_pr_auc = modelos[mejor_pr_auc_nombre]
        n_positivos = metricas["n_positivos"]
        n_filas = metricas["n_filas_entrenamiento"]

        # El modelo de la DECISIÓN del proyecto (CLAUDE.md §9.2): siempre
        # `logistic_plain` + Platt, en los dos datasets, evaluado a SU umbral
        # óptimo por coste -- nunca a 0.5. `evaluate_cost_variants` ya deja la
        # matriz de confusión calculada exactamente a ese umbral.
        platt = calibracion["variantes"]["logistic_plain_platt"]
        matriz_platt = platt["matriz_confusion"]
        tp_platt, fn_platt = matriz_platt["tp"], matriz_platt["fn"]
        n_positivos_platt = tp_platt + fn_platt
        recall_platt = tp_platt / n_positivos_platt if n_positivos_platt else float("nan")
        recall_ci_platt = evaluate.recall_wilson_ci(tp_platt, n_positivos_platt)

        datasets_info[nombre_dataset] = {
            "n_filas": n_filas,
            "n_positivos": n_positivos,
            "prevalencia": n_positivos / n_filas,
            "dummy_accuracy": modelos["dummy_most_frequent"]["metricas"]["accuracy"]["media"],
            "mejor_modelo_pr_auc_nombre": mejor_pr_auc_nombre,
            "mejor_modelo_pr_auc_valor": mejor_pr_auc["metricas"]["average_precision"]["media"],
            "modelo_decision": "logistic_plain_platt",
            "umbral_optimo_platt": platt["umbral_empirico"],
            "recall_platt_a_umbral_optimo": recall_platt,
            "recall_platt_tp": tp_platt,
            "recall_platt_n_positivos": n_positivos_platt,
            "recall_wilson_ci_platt": list(recall_ci_platt),
            "ahorro_pct_platt": platt["ahorro_pct"],
        }

        spec, _df, X, y = _cargar_xy_validado(nombre_dataset)
        cfg = evaluate.EvalConfig.for_dataset(nombre_dataset, get_settings())
        pipe = pipeline.build_pipeline(mejor_pr_auc_nombre, seed=cfg.seed)
        y_true_bin, y_score, _ = evaluate.out_of_fold_predictions(
            pipe, X, y, seed=cfg.seed, n_splits=5, positive_label=spec.positive_label
        )
        curvas[nombre_dataset] = (y_true_bin, y_score)
        prevalencias[nombre_dataset] = n_positivos / n_filas
        etiquetas_modelo[nombre_dataset] = mejor_pr_auc_nombre

        typer.echo(
            f"{nombre_dataset}: mejor PR-AUC {mejor_pr_auc_nombre} "
            f"({datasets_info[nombre_dataset]['mejor_modelo_pr_auc_valor']:.4f}) -- "
            f"logistic_plain+Platt @ t={platt['umbral_empirico']:.3f}: "
            f"recall {tp_platt}/{n_positivos_platt}, "
            f"IC Wilson [{recall_ci_platt[0]:.3f}, {recall_ci_platt[1]:.3f}]"
        )

    informe = report.build_comparison_report(datasets_info=datasets_info)
    ruta_json = report.write_comparison_report(informe, reports_dir / "comparison.json")

    ruta_figura = figures.figure_pr_curves_two_datasets(
        curvas, prevalencias, etiquetas_modelo, figures_dir / "comparacion_pr.png"
    )

    typer.echo(f"\nInforme JSON: {_mostrar_ruta(ruta_json)}")
    typer.echo(f"Figura: {_mostrar_ruta(ruta_figura)}")


if __name__ == "__main__":
    app()
