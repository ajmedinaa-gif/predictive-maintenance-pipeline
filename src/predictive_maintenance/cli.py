"""Interfaz de línea de comandos del proyecto.

Regla dura (CLAUDE.md §2.11): las descargas externas son un paso explícito del
CLI, nunca un efecto lateral de importar un módulo. Ningún comando de esta
Fase 1 toca la red.
"""

from __future__ import annotations

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
from predictive_maintenance import schema as schema_module
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


@app.command(name="validate")
def validate_command(
    dataset: str = typer.Option("lab180", "--dataset", help="Nombre del dataset a validar."),
) -> None:
    """Valida `dataset` contra su contrato y audita su plausibilidad física.

    Sale con código 1 si alguna fila terminó en cuarentena, para que un CI
    pueda usarlo como puerta de calidad (CLAUDE.md §2.9).
    """
    _configurar_logging()

    spec = datasets.get_spec(dataset)
    if dataset not in schema_module.SCHEMAS:
        disponibles = ", ".join(sorted(schema_module.SCHEMAS))
        typer.echo(f"No hay contrato registrado para {dataset!r}. Disponibles: {disponibles}")
        raise typer.Exit(code=2)
    esquema = schema_module.SCHEMAS[dataset]

    resultado = data.load_validated(spec.path, esquema, dataset_name=dataset)

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

    if resultado.report["n_cuarentena"]:
        raise typer.Exit(code=1)


@app.command(name="train")
def train_command(
    dataset: str = typer.Option("lab180", "--dataset", help="Nombre del dataset a entrenar."),
) -> None:
    """Compara el zoo de modelos de `pipeline.py` con el protocolo de CLAUDE.md §8.

    Solo entrena sobre las filas que pasan el contrato de datos (CLAUDE.md
    §2.9): la fila en cuarentena de `lab180` nunca entra al split de CV.
    """
    _configurar_logging()

    settings = get_settings()
    spec = datasets.get_spec(dataset)
    if dataset not in schema_module.SCHEMAS:
        disponibles = ", ".join(sorted(schema_module.SCHEMAS))
        typer.echo(f"No hay contrato registrado para {dataset!r}. Disponibles: {disponibles}")
        raise typer.Exit(code=2)

    resultado_validacion = data.load_validated(
        spec.path, schema_module.SCHEMAS[dataset], dataset_name=dataset
    )
    df = resultado_validacion.valid
    X = df.drop(columns=[spec.target])
    y = df[spec.target].astype(str)
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


def _cargar_xy_validado(dataset: str) -> tuple:
    spec = datasets.get_spec(dataset)
    if dataset not in schema_module.SCHEMAS:
        disponibles = ", ".join(sorted(schema_module.SCHEMAS))
        typer.echo(f"No hay contrato registrado para {dataset!r}. Disponibles: {disponibles}")
        raise typer.Exit(code=2)
    resultado = data.load_validated(spec.path, schema_module.SCHEMAS[dataset], dataset_name=dataset)
    df = resultado.valid
    X = df.drop(columns=[spec.target])
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

    # El gráfico de coste usa el modelo que decide el proyecto (CLAUDE.md §9.2):
    # logistic_plain + Platt.
    y_true_oof_mejor, y_score_oof_mejor, _ = evaluate.out_of_fold_predictions(
        calibration.VARIANT_BUILDERS["logistic_plain_platt"](settings.seed, 3),
        X,
        y,
        seed=settings.seed,
        n_splits=5,
        positive_label=spec.positive_label,
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
        shap_resultado, y, spec, figures_dir / "shap_waterfalls_positivos.png"
    )
    ruta_estabilidad = figures.figure_tree_root_stability(
        stability, spec, figures_dir / "tree_root_stability.png"
    )

    typer.echo(f"\nInforme JSON: {_mostrar_ruta(ruta_json)}")
    for ruta in (ruta_beeswarm, ruta_waterfalls, ruta_estabilidad):
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

    ic_8_10 = evaluate.recall_wilson_ci(8, 10)
    typer.echo(f"IC de Wilson del recall (8/10 aciertos): [{ic_8_10[0]:.3f}, {ic_8_10[1]:.3f}]")

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
        recall_wilson_ci_8_10=ic_8_10,
        presupuesto=presupuesto,
        learning_curve=curva_aprendizaje,
    )
    ruta_json = report.write_limits_report(informe, reports_dir / f"limits_{dataset}.json")
    ruta_figura = figures.figure_learning_curve(
        curva_aprendizaje, spec, figures_dir / "learning_curve.png"
    )

    typer.echo(f"\nInforme JSON: {_mostrar_ruta(ruta_json)}")
    typer.echo(f"Figura: {_mostrar_ruta(ruta_figura)}")


if __name__ == "__main__":
    app()
