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
    "air_temperature_k": "Temperatura ambiente (K)",
    "process_temperature_k": "Temperatura de proceso (K)",
    "rotational_speed_rpm": "Velocidad de rotación (rpm)",
    "torque_nm": "Par (Nm)",
    "tool_wear_min": "Desgaste de herramienta (min)",
    "power_w": "Potencia mecánica (W)",
    "temp_delta_k": "Delta de temperatura (K)",
    "wear_x_torque": "Desgaste x par",
    "machine_failure": "fallo de máquina",
}


def _etiqueta(col: str) -> str:
    return _ETIQUETA_ATRIBUTO.get(col, col)


def _pie_de_figura(fig, spec: datasets.DatasetSpec, extra: str = "") -> None:
    procedencia = (
        "datos SIMULADOS de laboratorio" if spec.simulated else "datos reales (UCI id=601)"
    )
    nota = f"{spec.name} — {procedencia}"
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


def figure_pr_curves_two_datasets(
    curvas: dict[str, tuple[np.ndarray, np.ndarray]],
    prevalencias: dict[str, float],
    etiquetas: dict[str, str],
    path: Path,
) -> Path:
    """Curvas PR del mejor modelo de `lab180` y de `ai4i2020`, en el mismo eje (CLAUDE.md §13).

    Es la prueba visual de "un pipeline, dos datasets": el mismo tipo de curva,
    calculada con el mismo código (`evaluate.out_of_fold_predictions`), para
    dos datasets de tamaño radicalmente distinto. `curvas[dataset] =
    (y_true_bin, y_score)`; `etiquetas[dataset]` es el nombre del modelo
    graficado (el de mejor PR-AUC medio de cada uno, no necesariamente el
    mismo modelo en los dos).
    """
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    colores = {"lab180": COLOR_POSITIVA, "ai4i2020": COLOR_NEGATIVA}

    for nombre_dataset, (y_true, y_score) in curvas.items():
        color = colores.get(nombre_dataset, "gray")
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        modelo = etiquetas.get(nombre_dataset, "?")
        ax.plot(
            recall,
            precision,
            label=f"{nombre_dataset} ({modelo}, n={len(y_true)})",
            color=color,
            linewidth=2.0,
        )
        prevalencia = prevalencias[nombre_dataset]
        ax.axhline(prevalencia, color=color, linewidth=1.0, linestyle=":", alpha=0.7)

    ax.set_xlabel("recall", fontsize=9)
    ax.set_ylabel("precision", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "El mismo pipeline, dos datasets: la curva PR de cada uno",
        fontsize=12,
        loc="left",
    )
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(
        0.01,
        0.01,
        "líneas punteadas = azar (prevalencia real de cada dataset) · "
        "predicciones out-of-fold, StratifiedKFold(5) sin repetir",
        fontsize=7,
        color="gray",
        ha="left",
        va="bottom",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Fase 4: calibración, umbral por coste, explicabilidad y límites
# --------------------------------------------------------------------------- #


def figure_calibration_curve(
    curvas: dict[str, pd.DataFrame], spec: datasets.DatasetSpec, path: Path
) -> Path:
    """Curva de fiabilidad de varias variantes, con la diagonal de calibración ideal (CLAUDE.md §9).

    `curvas[nombre]` es un `DataFrame` de `calibration.reliability_curve`
    (columnas `prob_media_predicha`, `frecuencia_observada`); los bins vacíos
    (`NaN`) no se dibujan.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, label="calibración ideal")
    colores = plt.get_cmap("tab10").colors
    for i, (nombre, curva) in enumerate(curvas.items()):
        validos = curva.dropna(subset=["prob_media_predicha"])
        ax.plot(
            validos["prob_media_predicha"],
            validos["frecuencia_observada"],
            marker="o",
            linewidth=1.6,
            color=colores[i % len(colores)],
            label=nombre,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("probabilidad media predicha (por bin)", fontsize=9)
    ax.set_ylabel("frecuencia observada de 'yes' (por bin)", fontsize=9)
    ax.set_title(
        "Curva de fiabilidad: ¿la probabilidad predicha significa lo que dice?",
        fontsize=11,
        loc="left",
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _pie_de_figura(fig, spec, "10 bins de ancho igual en [0, 1]")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_cost_vs_threshold(
    curva: pd.DataFrame,
    t_default: float,
    t_star: float,
    spec: datasets.DatasetSpec,
    path: Path,
) -> Path:
    """Coste total (MM CLP) en función del umbral, con marcadores en t=0.5 y en t* (CLAUDE.md §9).

    `curva` es un `threshold.cost_curve(...)`: columnas `threshold` y
    `coste_total` en CLP. Se grafica en millones para que el eje sea legible.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    coste_mm = curva["coste_total"] / 1e6
    ax.plot(curva["threshold"], coste_mm, color=COLOR_NEGATIVA, linewidth=1.8)

    def _coste_en(t: float) -> float:
        return float(np.interp(t, curva["threshold"], coste_mm))

    ax.axvline(t_default, color="black", linestyle="--", linewidth=1.2)
    ax.scatter([t_default], [_coste_en(t_default)], color="black", zorder=5)
    ax.annotate(
        f"t=0.5\n{_coste_en(t_default):.1f} MM CLP",
        (t_default, _coste_en(t_default)),
        textcoords="offset points",
        xytext=(8, 10),
        fontsize=8,
    )

    ax.axvline(t_star, color=COLOR_POSITIVA, linestyle="--", linewidth=1.2)
    ax.scatter([t_star], [_coste_en(t_star)], color=COLOR_POSITIVA, zorder=5)
    ax.annotate(
        f"t*={t_star:.4f}\n{_coste_en(t_star):.1f} MM CLP",
        (t_star, _coste_en(t_star)),
        textcoords="offset points",
        xytext=(8, -22),
        fontsize=8,
        color=COLOR_POSITIVA,
    )

    ax.set_xlabel("umbral de decisión", fontsize=9)
    ax.set_ylabel("coste total (millones de CLP)", fontsize=9)
    ax.set_title("El umbral 0.5 es una decisión, no un valor por defecto", fontsize=11, loc="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _pie_de_figura(fig, spec, "costes de config/costs.yaml: SUPUESTOS, no medidos en campo")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_shap_beeswarm(
    shap_result: dict, X: pd.DataFrame, spec: datasets.DatasetSpec, path: Path
) -> Path:
    """Beeswarm de valores SHAP de `logistic_plain`, coloreado por el valor ORIGINAL del atributo.

    Los valores SHAP se calcularon sobre el espacio imputado y escalado que ve
    el clasificador (`explain.shap_values_logistic`); aquí se colorea con el
    valor crudo de `X` para que el eje de color sea interpretable (°C, mm/s,
    bar...) en vez de un z-score sin unidades.
    """
    import shap

    explicacion = shap.Explanation(
        values=shap_result["shap_values"],
        base_values=np.full(len(X), shap_result["valor_base"]),
        data=X.to_numpy(),
        feature_names=[_etiqueta(c) for c in shap_result["columnas"]],
    )
    shap.plots.beeswarm(explicacion, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"Contribución SHAP por atributo — logistic_plain "
        f"(log-odds de failure='{shap_result['positive_label']}')",
        fontsize=10,
    )
    _pie_de_figura(fig, spec, "SHAP en el espacio escalado; color = valor original del atributo")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_shap_waterfalls_positives(
    shap_result: dict,
    y: pd.Series,
    spec: datasets.DatasetSpec,
    path: Path,
    max_cases: int | None = None,
) -> Path:
    """Un mini-waterfall por cada caso positivo: qué atributo empujó esa predicción y cuánto.

    Barras horizontales con la contribución SHAP (log-odds) de cada atributo,
    ordenadas por magnitud. Pensado para `lab180` (10 positivos, CLAUDE.md
    §6.2): son pocos como para no poder mirarlos uno a uno. Con `max_cases`
    se limita a los primeros `max_cases` (por índice de fila, determinista) --
    necesario para `ai4i2020` (339 positivos): sin límite, la figura tendría
    68 filas de subplots. `None` (por defecto) no limita nada.
    """
    y_str = y.astype(str).reset_index(drop=True)
    idx_positivos = np.flatnonzero((y_str == spec.positive_label).to_numpy())
    n_total = len(idx_positivos)
    limitado = max_cases is not None and n_total > max_cases
    if limitado:
        idx_positivos = idx_positivos[:max_cases]
    columnas = [_etiqueta(c) for c in shap_result["columnas"]]
    valores = shap_result["shap_values"]

    n = len(idx_positivos)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows))
    ejes = np.atleast_1d(axes).ravel()
    for ax in ejes[n:]:
        ax.set_visible(False)

    for ax, idx in zip(ejes, idx_positivos, strict=False):
        contribuciones = valores[idx]
        orden = np.argsort(np.abs(contribuciones))
        colores = [COLOR_POSITIVA if v > 0 else COLOR_NEGATIVA for v in contribuciones[orden]]
        ax.barh(range(len(orden)), contribuciones[orden], color=colores)
        ax.set_yticks(range(len(orden)))
        ax.set_yticklabels([columnas[i] for i in orden], fontsize=7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"fila {idx}", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)

    titulo_casos = f"los primeros {n} de {n_total}" if limitado else f"los {n}"
    fig.suptitle(
        f"Contribución SHAP por atributo en {titulo_casos} casos "
        f"{spec.target}='{spec.positive_label}'",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    nota = "naranja empuja hacia la clase positiva, azul hacia la negativa"
    if limitado:
        nota += f" · muestra de {max_cases} sobre {n_total} positivos totales"
    _pie_de_figura(fig, spec, nota)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_learning_curves_comparison(
    curvas: dict[str, tuple[pd.DataFrame, int]], path: Path
) -> Path:
    """Curvas de aprendizaje de ambos datasets en el mismo eje, normalizado por tamaño.

    `curvas[dataset] = (df, n_filas)`: `df` es la salida de
    `power.learning_curve_pr_auc` (columnas `n_entrenamiento`, `pr_auc_media`,
    `pr_auc_desv`, ya en `reports/limits_<dataset>.json`); `n_filas` el
    tamaño total de ESE dataset. El eje x se normaliza a
    `n_entrenamiento / n_filas` porque 179 y 10 000 no son comparables en
    valor absoluto -- la fracción del dataset sí lo es. Es la prueba visual
    más directa del proyecto de que el tamaño muestral es el problema: la
    curva de `ai4i2020` sube con bandas estrechas; la de `lab180` baja con
    bandas de hasta ±0.30 -- no es una curva de aprendizaje, es ruido de
    muestreo (CLAUDE.md §10.2).
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colores = {"lab180": COLOR_POSITIVA, "ai4i2020": COLOR_NEGATIVA}

    for nombre_dataset, (df, n_filas) in curvas.items():
        color = colores.get(nombre_dataset, "gray")
        x = df["n_entrenamiento"] / n_filas
        media = df["pr_auc_media"]
        desv = df["pr_auc_desv"]
        ax.plot(
            x,
            media,
            marker="o",
            color=color,
            linewidth=1.8,
            label=f"{nombre_dataset} (n={n_filas})",
        )
        ax.fill_between(x, media - desv, media + desv, color=color, alpha=0.2)

    ax.set_xlabel("fracción del dataset usada para entrenar", fontsize=9)
    ax.set_ylabel("PR-AUC (StratifiedKFold)", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Con 10 000 filas la curva es una curva; con 180, es ruido",
        fontsize=12,
        loc="left",
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(
        0.01,
        0.01,
        "bandas = ±1 desviación · StratifiedKFold(5), sin suavizar",
        fontsize=7,
        color="gray",
        ha="left",
        va="bottom",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_tree_render(arbol, columnas: list[str], spec: datasets.DatasetSpec, path: Path) -> Path:
    """Dibuja `tree_shallow_balanced` ya ajustado sobre todo el dataset (pestaña Explicabilidad).

    Un solo árbol, no los 300 bootstraps de la estabilidad: este es el árbol
    concreto que se entrenaría hoy sobre todos los datos, para leerlo entero.
    La estabilidad de su raíz bajo remuestreo es una pregunta aparte
    (`figure_tree_root_stability`, CLAUDE.md §10.1).
    """
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(max(10.0, 3.0 * 2 ** arbol.get_depth()), 6.0))
    plot_tree(
        arbol,
        feature_names=[_etiqueta(c) for c in columnas],
        class_names=list(arbol.classes_),
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(
        f"tree_shallow_balanced (max_depth=3, min_samples_leaf=5) — "
        f"profundidad real {arbol.get_depth()}",
        fontsize=11,
        loc="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _pie_de_figura(fig, spec, "ajustado sobre TODO el dataset, no un fold de CV")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_tree_root_stability(stability: dict, spec: datasets.DatasetSpec, path: Path) -> Path:
    """Barras del % de bootstraps en que cada atributo fue el primer corte del árbol (§10.1).

    Línea horizontal en 70 %: el test `tests/test_tree_stability.py` protege
    que ninguna barra la cruce — si la cruzara, "la vibración decide el
    fallo" dejaría de ser una conclusión errónea.
    """
    porcentajes = stability["porcentaje_raiz_por_atributo"]
    atributos = list(porcentajes)
    valores = [porcentajes[a] for a in atributos]
    etiquetas = [_etiqueta(a) for a in atributos]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    barras = ax.bar(etiquetas, valores, color=COLOR_POSITIVA, alpha=0.8)
    for barra, v in zip(barras, valores, strict=False):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            v + 1.5,
            f"{v:.1f} %",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axhline(70.0, color="black", linewidth=1.2, linestyle="--")
    ax.text(len(atributos) - 0.5, 72.0, "umbral del test: 70 %", ha="right", fontsize=8)
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"% de {stability['n_boot']} bootstraps en que fue la raíz", fontsize=9)
    ax.set_title("Ningún atributo domina la raíz del árbol", fontsize=12, loc="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _pie_de_figura(
        fig, spec, f"profundidad efectiva media: {stability['profundidad_efectiva_media']:.2f}"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_learning_curve(curva: pd.DataFrame, spec: datasets.DatasetSpec, path: Path) -> Path:
    """PR-AUC frente a tamaño de entrenamiento, con banda de ±1 desviación (CLAUDE.md §10.2).

    ADVERTENCIA: esta curva NO sube con más datos — baja, con bandas de hasta
    ±0.30. Se dibuja tal cual, sin suavizar: es uno de los gráficos más
    honestos del repositorio.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = curva["n_entrenamiento"]
    media = curva["pr_auc_media"]
    desv = curva["pr_auc_desv"]
    ax.plot(x, media, marker="o", color=COLOR_NEGATIVA, linewidth=1.8)
    ax.fill_between(
        x, media - desv, media + desv, color=COLOR_NEGATIVA, alpha=0.2, label="±1 desviación"
    )
    ax.set_xlabel("nº de observaciones de entrenamiento", fontsize=9)
    ax.set_ylabel("PR-AUC (StratifiedKFold)", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "La curva de aprendizaje NO sube: señal dominada por ruido de muestreo",
        fontsize=11,
        loc="left",
    )
    ax.legend(fontsize=8)
    ax.text(
        0.02,
        0.02,
        "No suavizado: con 10 positivos, esto es ruido, no una tendencia.",
        transform=ax.transAxes,
        fontsize=8,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _pie_de_figura(fig, spec)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
