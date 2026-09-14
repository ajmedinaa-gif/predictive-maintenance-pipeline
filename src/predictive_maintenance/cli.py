"""Interfaz de línea de comandos del proyecto.

Regla dura (CLAUDE.md §2.11): las descargas externas son un paso explícito del
CLI, nunca un efecto lateral de importar un módulo. Ningún comando de esta
Fase 1 toca la red.
"""

from __future__ import annotations

import typer

from predictive_maintenance import __version__, datasets, eda, figures, report

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


if __name__ == "__main__":
    app()
