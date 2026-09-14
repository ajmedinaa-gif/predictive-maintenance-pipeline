"""Dibuja: todas las funciones `figure_*` y `make_eda_figures`.

Nada de lógica de negocio aquí; eso vive en `eda.py`. Este módulo solo traduce
resultados ya calculados a PNG. Backend `Agg` fijado antes de importar pyplot
(CLAUDE.md §14.4) porque en macOS matplotlib intenta abrir ventanas y cuelga el
CLI. Diagnóstico interno, no un dashboard: dos colores fijos y consistentes
para las clases `no`/`yes`, títulos y ejes en español. Todas las figuras
declaran que los datos son SIMULADOS (CLAUDE.md §2.12).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# En macOS matplotlib intenta abrir ventanas y cuelga el CLI: el backend sin
# interfaz se fija ANTES de importar pyplot (CLAUDE.md §14.4).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predictive_maintenance import datasets, eda

# Dos colores fijos, siempre en el mismo orden: negativa (no) y positiva (yes).
# Se usan igual en las cuatro figuras para que la clase se reconozca de un
# vistazo entre gráficos distintos.
COLOR_NEGATIVA = "#2a78d6"  # azul
COLOR_POSITIVA = "#eb6834"  # naranja

_ETIQUETA_ATRIBUTO = {
    "temperature_c": "Temperatura (°C)",
    "vibration_mm_s": "Vibración RMS (mm/s)",
    "pressure_bar": "Presión (bar)",
    "hours_since_maintenance": "Horas desde mantenimiento",
    "load_percent": "Carga (% de la nominal)",
}


def _etiqueta(col: str) -> str:
    return _ETIQUETA_ATRIBUTO.get(col, col)


def _pie_de_figura(fig, spec: datasets.DatasetSpec, extra: str = "") -> None:
    nota = f"{spec.name} — datos SIMULADOS de laboratorio"
    if extra:
        nota = f"{extra}  ·  {nota}"
    fig.text(0.01, 0.01, nota, fontsize=7, color="gray", ha="left", va="bottom")


def _colores_de_clase(clases: list[str], positive_label: str) -> dict[str, str]:
    return {c: (COLOR_POSITIVA if c == positive_label else COLOR_NEGATIVA) for c in clases}


def _rejilla(n: int):
    ncols = 3 if n > 4 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    ejes = np.atleast_1d(axes).ravel()
    for ax in ejes[n:]:
        ax.set_visible(False)
    return fig, ejes


def figure_histograms(df: pd.DataFrame, spec: datasets.DatasetSpec, path: Path) -> Path:
    """Histograma por atributo con las dos clases superpuestas.

    Normalizado dentro de cada clase: con 10 positivos frente a 170 negativos,
    un histograma de conteos deja la clase minoritaria invisible.
    """
    target, positive = spec.target, spec.positive_label
    atributos = eda.numeric_attributes(df, target)
    clases = sorted(df[target].astype(str).unique())
    colores = _colores_de_clase(clases, positive)
    conteos = df[target].astype(str).value_counts()

    fig, ejes = _rejilla(len(atributos))
    for ax, col in zip(ejes, atributos, strict=False):
        serie = df[col].dropna()
        bordes = np.histogram_bin_edges(serie, bins=18)
        for clase in clases:
            valores = df.loc[df[target].astype(str) == clase, col].dropna()
            ax.hist(
                valores,
                bins=bordes,
                weights=np.full(len(valores), 1.0 / len(valores)),
                color=colores[clase],
                alpha=0.55,
                edgecolor=colores[clase],
                linewidth=1.0,
                label=f"{target}={clase} (n={conteos[clase]})",
            )
        ax.set_xlabel(_etiqueta(col), fontsize=8)
        ax.set_ylabel("proporción de la clase", fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle("Distribución de cada sensor, por clase", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _pie_de_figura(fig, spec, "histogramas normalizados dentro de cada clase")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_boxplots(df: pd.DataFrame, spec: datasets.DatasetSpec, path: Path) -> Path:
    """Boxplot por clase, con los puntos individuales encima.

    Con 10 positivos, una caja sin los puntos sugiere una precisión que la
    muestra no tiene.
    """
    target, positive = spec.target, spec.positive_label
    atributos = eda.numeric_attributes(df, target)
    clases = sorted(df[target].astype(str).unique())
    colores = _colores_de_clase(clases, positive)
    rng = np.random.default_rng(datasets.seed())

    fig, ejes = _rejilla(len(atributos))
    for ax, col in zip(ejes, atributos, strict=False):
        grupos = [df.loc[df[target].astype(str) == c, col].dropna().to_numpy() for c in clases]
        bp = ax.boxplot(
            grupos,
            patch_artist=True,
            widths=0.45,
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": "", "linestyle": "none"},
        )
        for parche, clase in zip(bp["boxes"], clases, strict=False):
            parche.set_facecolor(colores[clase])
            parche.set_alpha(0.3)
            parche.set_edgecolor(colores[clase])
        for i, (valores, clase) in enumerate(zip(grupos, clases, strict=False), start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(valores))
            ax.scatter(
                np.full(len(valores), i) + jitter,
                valores,
                s=14,
                color=colores[clase],
                alpha=0.85,
            )
        ax.set_xticks(range(1, len(clases) + 1))
        ax.set_xticklabels([f"{c}\n(n={len(g)})" for c, g in zip(clases, grupos, strict=False)])
        ax.set_ylabel(_etiqueta(col), fontsize=8)

    fig.suptitle("Cada sensor frente a la clase", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _pie_de_figura(fig, spec, "un punto = una observación")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_correlation(df: pd.DataFrame, spec: datasets.DatasetSpec, path: Path) -> Path:
    """Mapa de calor de correlación, con la escala fija en [-1, 1].

    Fijar la escala es el punto: autoescalar a [-0.10, 0.10] convertiría ruido
    en un patrón aparente. Debe verse plano.
    """
    corr = eda.correlation_matrix(df, spec.target)
    maximo = eda.max_abs_correlation(df, spec.target)
    etiquetas = [_etiqueta(c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(etiquetas)))
    ax.set_yticklabels(etiquetas, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    barra = fig.colorbar(im, ax=ax, shrink=0.75)
    barra.set_label("r de Pearson (escala fija en [-1, 1])", fontsize=8)

    ax.set_title(
        f"Correlación entre sensores — máxima |r| = {maximo['max_abs_r']:.3f}",
        fontsize=12,
        loc="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _pie_de_figura(fig, spec, "plano por construcción: las cinco variables son independientes")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_prevalence(df: pd.DataFrame, spec: datasets.DatasetSpec, path: Path) -> Path:
    """Barras de clase con la línea del clasificador trivial mayoritario.

    La línea cae exactamente sobre la barra de la clase mayoritaria: el
    accuracy del modelo trivial ya es esa barra (CLAUDE.md §6.2).
    """
    balance = eda.class_balance(df, spec.target, spec.positive_label)
    clases = sorted(balance["conteos"])
    colores = _colores_de_clase(clases, spec.positive_label)
    pcts = [100.0 * balance["conteos"][c] / balance["n"] for c in clases]
    trivial = balance["accuracy_trivial_mayoritario_pct"]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    barras = ax.bar(clases, pcts, color=[colores[c] for c in clases], width=0.5)
    for barra, clase, pct in zip(barras, clases, pcts, strict=False):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            pct + 1.5,
            f"{pct:.4f} %\n(n={balance['conteos'][clase]})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axhline(trivial, color="black", linewidth=1.5, linestyle="--")
    ax.text(
        len(clases) - 0.5,
        trivial - 4,
        f"accuracy del clasificador trivial mayoritario = {trivial:.4f} %",
        ha="right",
        va="top",
        fontsize=9,
    )
    ax.set_ylim(0, 108)
    ax.set_ylabel("% de las observaciones", fontsize=9)
    ax.set_xlabel(spec.target, fontsize=9)
    ax.set_title(
        "Cualquier accuracy por debajo de la línea es peor que no hacer nada",
        fontsize=12,
        loc="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _pie_de_figura(fig, spec)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_eda_figures(df: pd.DataFrame, spec: datasets.DatasetSpec, outdir: Path) -> list[Path]:
    """Genera las cuatro figuras del EDA y devuelve sus rutas."""
    outdir.mkdir(parents=True, exist_ok=True)
    return [
        figure_histograms(df, spec, outdir / "histogramas_por_clase.png"),
        figure_boxplots(df, spec, outdir / "boxplots_por_clase.png"),
        figure_correlation(df, spec, outdir / "correlacion.png"),
        figure_prevalence(df, spec, outdir / "prevalencia.png"),
    ]


def figure_pr_roc_curves(
    curvas: dict[str, tuple[np.ndarray, np.ndarray]],
    prevalencia: float,
    spec: datasets.DatasetSpec,
    path: Path,
) -> Path:
    """Curvas PR y ROC de todos los modelos, cada tipo en su propio eje.

    `curvas[nombre] = (y_true_bin, y_score)`: las predicciones out-of-fold de
    `evaluate.out_of_fold_predictions` (CLAUDE.md §8.2, un único
    `StratifiedKFold`), no la CV repetida — una curva por modelo, no 50
    superpuestas. La línea de azar en el eje PR se fija en la prevalencia real
    (CLAUDE.md §6.2): por debajo de esa línea, el modelo ordena peor que
    puntuar al azar.
    """
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12.0, 5.2))
    colores = plt.get_cmap("tab10").colors

    for i, (nombre, (y_true, y_score)) in enumerate(curvas.items()):
        color = colores[i % len(colores)]
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax_pr.plot(recall, precision, label=nombre, color=color, linewidth=1.6)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax_roc.plot(fpr, tpr, label=nombre, color=color, linewidth=1.6)

    ax_pr.axhline(prevalencia, color="black", linewidth=1.0, linestyle="--")
    ax_pr.text(
        0.02,
        prevalencia + 0.02,
        f"azar (prevalencia = {prevalencia:.4f})",
        fontsize=7,
        va="bottom",
    )
    ax_pr.set_xlabel("recall", fontsize=9)
    ax_pr.set_ylabel("precision", fontsize=9)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_title("Curvas precisión-recall", fontsize=11, loc="left")

    ax_roc.plot([0, 1], [0, 1], color="black", linewidth=1.0, linestyle="--")
    ax_roc.set_xlabel("tasa de falsos positivos", fontsize=9)
    ax_roc.set_ylabel("tasa de verdaderos positivos", fontsize=9)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.set_title("Curvas ROC", fontsize=11, loc="left")
    ax_roc.legend(fontsize=6.5, loc="lower right")

    fig.suptitle(
        "Comparación de modelos: predicciones out-of-fold (StratifiedKFold, sin repetir)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    _pie_de_figura(fig, spec)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
