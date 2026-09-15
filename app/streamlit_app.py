"""Dashboard de mantenimiento predictivo (CLAUDE.md §12, extra 4).

Lee `reports/*.json` y `reports/figures/*/*.png` ya generados por el CLI
(`pdm-cli eda/run/calibrate/explain/limits`) -- NUNCA reentrena ni recalcula
nada pesado al arrancar. La única excepción es la pestaña "Decisión": ahí sí
se recalcula en vivo, pero es aritmética pura (`threshold.py`) sobre las
predicciones out-of-fold ya guardadas en `calibration_<dataset>.json`
(`oof_decision_model`), no un reentrenamiento del modelo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from predictive_maintenance import datasets, threshold

st.set_page_config(
    page_title="Mantenimiento predictivo",
    page_icon="🔧",
    layout="wide",
)

REPORTS_DIR = datasets.PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATASET_NAMES = tuple(sorted(datasets.ADAPTERS))


@st.cache_data(show_spinner=False)
def _load_json(nombre: str) -> dict | None:
    ruta = REPORTS_DIR / f"{nombre}.json"
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def _figura(dataset: str, nombre_archivo: str) -> Path | None:
    ruta = FIGURES_DIR / dataset / nombre_archivo
    return ruta if ruta.exists() else None


def _mostrar_figura(dataset: str, nombre_archivo: str, **kwargs) -> None:
    ruta = _figura(dataset, nombre_archivo)
    if ruta is not None:
        st.image(str(ruta), use_container_width=True, **kwargs)


st.title("🔧 Mantenimiento predictivo industrial")
st.caption(
    "Pipeline reproducible con validación honesta bajo desbalance de clases — "
    "[@ajmedinaa-gif](https://github.com/ajmedinaa-gif/predictive-maintenance-pipeline)"
)

dataset = st.sidebar.selectbox("Dataset", DATASET_NAMES, index=0)
adapter = datasets.get_adapter(dataset)
if adapter.simulated:
    st.sidebar.warning("⚠️ Datos SIMULADOS de laboratorio (CLAUDE.md §7).")
else:
    st.sidebar.success("✅ Datos reales (UCI id=601, Matzka 2020).")
st.sidebar.caption(
    "Este selector cambia las cinco pestañas a la vez: es el mismo dashboard "
    "leyendo los `reports/*.json` de uno u otro dataset (CLAUDE.md §13.1)."
)

eda = _load_json(f"eda_{dataset}")
validation = _load_json(f"validation_{dataset}")
plausibility = _load_json(f"plausibility_{dataset}")
metrics = _load_json(f"metrics_{dataset}")
calibration_r = _load_json(f"calibration_{dataset}")
explainability = _load_json(f"explainability_{dataset}")
limits = _load_json(f"limits_{dataset}")

tab_datos, tab_modelos, tab_decision, tab_explicabilidad, tab_limites = st.tabs(
    ["📊 Datos", "🧪 Modelos", "💰 Decisión", "🔍 Explicabilidad", "📐 Límites"]
)

# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
with tab_datos:
    if eda is None:
        st.info(f"EDA no generado todavía. Ejecuta `pdm-cli eda --dataset {dataset}`.")
    else:
        balance = eda["balance_de_clases"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas", f"{eda['n_filas']:,}")
        c2.metric("Positivos", f"{balance['n_positivos']:,}", f"{balance['prevalencia_pct']:.2f} %")
        c3.metric("Accuracy del dummy", f"{balance['accuracy_trivial_mayoritario_pct']:.2f} %")

        if plausibility is not None:
            veredicto = plausibility["veredicto"]
            if veredicto == "probablemente sintético":
                st.error(f"Auditor de plausibilidad: **{veredicto}**")
            else:
                st.success(f"Auditor de plausibilidad: **{veredicto}**")
            with st.expander("Evidencia del auditor"):
                for e in plausibility["evidencia"]:
                    st.markdown(f"- {e}")

        if validation is not None and validation["n_cuarentena"]:
            st.warning(
                f"{validation['n_cuarentena']} fila(s) en cuarentena "
                f"(de {validation['n_filas_leidas']} leídas) -- no entran al modelo."
            )

        st.subheader("Tabla descriptiva")
        st.dataframe(
            pd.DataFrame(eda["descriptiva"]).set_index("atributo").round(4),
            use_container_width=True,
        )
        st.caption(eda["nulos"]["lectura"])

        if eda["anomalias_fisicas"]:
            st.subheader("Anomalías físicas (van a cuarentena)")
            st.dataframe(pd.DataFrame(eda["anomalias_fisicas"]), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            _mostrar_figura(dataset, "prevalencia.png")
            _mostrar_figura(dataset, "histogramas_por_clase.png")
        with col_b:
            _mostrar_figura(dataset, "correlacion.png")
            _mostrar_figura(dataset, "boxplots_por_clase.png")

# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
with tab_modelos:
    if metrics is None:
        st.info(f"Modelos no entrenados todavía. Ejecuta `pdm-cli run --dataset {dataset}`.")
    else:
        st.write(
            f"{metrics['n_filas_entrenamiento']:,} filas de entrenamiento, "
            f"{metrics['n_positivos']} positivos."
        )
        st.caption(f"Protocolo: `{metrics['protocolo_principal']}`")

        filas = []
        for nombre, r in metrics["modelos"].items():
            m = r["metricas"]
            filas.append(
                {
                    "modelo": nombre,
                    "PR-AUC": m["average_precision"]["media"],
                    "ROC-AUC": m["roc_auc"]["media"],
                    "recall": m["recall"]["media"],
                    "precision": m["precision"]["media"],
                    "bal.acc": m["balanced_accuracy"]["media"],
                    "accuracy": m["accuracy"]["media"],
                    "Brier": m["brier_score_loss"]["media"],
                }
            )
        # `dummy_*` primero: el orden de `metrics["modelos"]` ya lo respeta
        # (CLAUDE.md regla dura 1), no se reordena aquí.
        tabla = pd.DataFrame(filas).set_index("modelo")
        st.dataframe(tabla.round(4), use_container_width=True)
        st.caption(
            "Las filas `dummy_*` van primero: cualquier accuracy por debajo de "
            "`dummy_most_frequent` es peor que no hacer nada."
        )
        _mostrar_figura(dataset, "curvas_pr_roc.png")

# --------------------------------------------------------------------------- #
# Decisión -- la pestaña que recalcula en vivo
# --------------------------------------------------------------------------- #
with tab_decision:
    if calibration_r is None or "oof_decision_model" not in calibration_r:
        st.info(
            f"Calibración no ejecutada todavía. Ejecuta `pdm-cli calibrate --dataset {dataset}`."
        )
    else:
        st.write(
            "Modelo de la decisión: **`logistic_plain` + Platt** (CLAUDE.md §9.2), "
            "predicciones out-of-fold de un único `StratifiedKFold(5)`. Mueve los "
            "sliders: el umbral óptimo, la matriz de confusión y el coste se "
            "recalculan al instante -- son los mismos números que "
            "`pdm-cli calibrate` mediría con estos costes, sin reentrenar nada."
        )
        costs_defecto = calibration_r["costs"]

        col_a, col_b, col_c = st.columns(3)
        c_fn = col_a.slider(
            "Coste de un fallo NO detectado — C_FN (CLP)",
            min_value=0,
            max_value=20_000_000,
            value=int(costs_defecto["cost_false_negative"]),
            step=100_000,
        )
        c_fp = col_b.slider(
            "Coste de una inspección innecesaria — C_FP (CLP)",
            min_value=0,
            max_value=3_000_000,
            value=int(costs_defecto["cost_false_positive"]),
            step=50_000,
        )
        c_tp = col_c.slider(
            "Coste de un mantenimiento planificado — C_TP (CLP)",
            min_value=0,
            max_value=3_000_000,
            value=int(costs_defecto["cost_true_positive"]),
            step=50_000,
        )
        costs_vivo = {
            "cost_true_negative": 0.0,
            "cost_false_positive": float(c_fp),
            "cost_false_negative": float(c_fn),
            "cost_true_positive": float(c_tp),
        }

        y_true = np.array(calibration_r["oof_decision_model"]["y_true"])
        y_score = np.array(calibration_r["oof_decision_model"]["y_score"])

        t_optimo = threshold.optimal_threshold(y_true, y_score, costs_vivo)
        t_teorico_vivo = threshold.theoretical_threshold(costs_vivo)

        def _matriz_y_coste(t: float):
            y_pred = (y_score >= t).astype(int)
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            coste = threshold.realized_cost(y_true, y_pred, costs_vivo)
            return tn, fp, fn, tp, coste

        tn0, fp0, fn0, tp0, coste0 = _matriz_y_coste(0.5)
        tn1, fp1, fn1, tp1, coste1 = _matriz_y_coste(t_optimo)

        col_izq, col_der = st.columns(2)
        with col_izq:
            st.markdown("#### Umbral por defecto (t = 0.5)")
            st.metric("Coste total", f"{coste0:,.0f} CLP")
            st.table(
                pd.DataFrame(
                    [[tn0, fp0], [fn0, tp0]],
                    index=["real: no", "real: yes"],
                    columns=["pred: no", "pred: yes"],
                )
            )
        with col_der:
            st.markdown(f"#### Umbral óptimo por coste (t = {t_optimo:.4f})")
            st.metric(
                "Coste total",
                f"{coste1:,.0f} CLP",
                delta=f"{coste1 - coste0:,.0f} CLP frente a t=0.5",
                delta_color="inverse",
            )
            st.table(
                pd.DataFrame(
                    [[tn1, fp1], [fn1, tp1]],
                    index=["real: no", "real: yes"],
                    columns=["pred: no", "pred: yes"],
                )
            )

        st.caption(
            f"Umbral teórico t* con estos costes: `{t_teorico_vivo:.4f}` "
            "(exacto solo si el modelo está perfectamente calibrado, CLAUDE.md §9.1) -- "
            f"frente al `{t_optimo:.4f}` empírico de arriba."
        )

        st.subheader("Coste total frente al umbral")
        curva = threshold.cost_curve(y_true, y_score, costs_vivo)
        st.line_chart(curva.set_index("threshold")["coste_total"])

# --------------------------------------------------------------------------- #
# Explicabilidad
# --------------------------------------------------------------------------- #
with tab_explicabilidad:
    if explainability is None:
        st.info(
            f"Explicabilidad no generada todavía. Ejecuta `pdm-cli explain --dataset {dataset}`."
        )
    else:
        st.subheader("Importancia media |SHAP| (logistic_plain)")
        importancia = pd.Series(
            explainability["shap_importancia_media"], name="importancia"
        ).sort_values(ascending=False)
        st.bar_chart(importancia)
        _mostrar_figura(dataset, "shap_beeswarm.png")
        _mostrar_figura(dataset, "shap_waterfalls_positivos.png")

        st.subheader("Árbol renderizado (tree_shallow_balanced, sobre todo el dataset)")
        _mostrar_figura(dataset, "tree_render.png")

        estabilidad = explainability["estabilidad_raiz_arbol"]
        st.subheader(f"Estabilidad de la raíz del árbol ({estabilidad['n_boot']} bootstraps)")
        tabla_estabilidad = pd.Series(
            estabilidad["porcentaje_raiz_por_atributo"], name="% de bootstraps como raíz"
        ).to_frame()
        st.dataframe(tabla_estabilidad.round(1), use_container_width=True)
        st.caption(
            f"Profundidad efectiva media: {estabilidad['profundidad_efectiva_media']:.2f}. "
            "Ningún atributo debería dominar por encima del 70 % (`tests/test_tree_stability.py`)."
        )
        _mostrar_figura(dataset, "tree_root_stability.png")

# --------------------------------------------------------------------------- #
# Límites
# --------------------------------------------------------------------------- #
with tab_limites:
    if limits is None:
        st.info(f"Límites no calculados todavía. Ejecuta `pdm-cli limits --dataset {dataset}`.")
    else:
        ic = limits["recall_wilson_ci_ilustrativo"]
        st.metric(
            f"IC de Wilson del recall "
            f"({limits['recall_wilson_ci_ilustrativo_aciertos']}/"
            f"{limits['recall_wilson_ci_ilustrativo_n']} aciertos, 80 % ilustrativo)",
            f"[{ic[0]:.3f}, {ic[1]:.3f}]",
            help="Ancho del intervalo = cuánta confianza da esta muestra. Con pocos "
            "positivos, un IC ancho no es un error de cálculo: es la muestra "
            "diciendo que no alcanza para decidir nada (CLAUDE.md §10.3).",
        )

        st.subheader("Presupuesto estadístico")
        st.dataframe(
            pd.DataFrame(limits["presupuesto_estadistico"]).T.rename(
                columns={
                    "margen_pp": "margen (pp)",
                    "fallos_necesarios": "fallos necesarios",
                    "ciclos_de_maquina_necesarios": "ciclos de máquina",
                }
            ),
            use_container_width=True,
        )

        st.subheader("Curva de aprendizaje (PR-AUC, no suavizada)")
        _mostrar_figura(dataset, "learning_curve.png")
