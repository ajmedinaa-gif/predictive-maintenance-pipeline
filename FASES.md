# Plan de construcción — 5 fases

Pega **un prompt por vez** en Claude Code. Cuando presente el CHECKPOINT,
ejecuta tú misma los comandos que dice haber ejecutado, y solo entonces pega el
siguiente. Si algo no cuadra, responde con la corrección en vez de avanzar.

## Antes de empezar (una sola vez, en tu Terminal de macOS)

Entorno verificado: MacBook Air **Intel (x86_64)**, zsh, `uv` instalado, sin conda
activo, Python del sistema 3.9.6 (no se usa).

> **NO uses Homebrew para nada en este proyecto.** Desde septiembre de 2026
> Homebrew dejó de publicar binarios precompilados para Intel x86_64: cada
> `brew install` compila desde código fuente. Instalar `gh` por brew arrastra
> compilar el toolchain completo de Go (30-60 min en un Air Intel).
>
> Esto NO afecta a Python: `uv` descarga un Python precompilado y todas las
> dependencias del proyecto (numpy, scipy, scikit-learn, shap, streamlit) tienen
> rueda precompilada para macOS x86_64 en PyPI. `uv sync` tarda un par de minutos.

### 1. GitHub CLI — binario oficial, sin compilar

Descarga e instala el paquete firmado por GitHub (doble clic, siguiente,
siguiente):

<https://github.com/cli/cli/releases/latest> → asset
**`gh_<version>_macOS_universal.pkg`**

Alternativa por terminal, sin instalador:

```bash
cd ~/Downloads
curl -LO https://github.com/cli/cli/releases/download/v2.100.0/gh_2.100.0_macOS_amd64.zip
unzip -q gh_2.100.0_macOS_amd64.zip
mkdir -p ~/.local/bin && mv gh_2.100.0_macOS_amd64/bin/gh ~/.local/bin/
gh --version
```

`~/.local/bin` ya está en tu PATH (es donde vive `uv`).

### 2. Python del proyecto

```bash
uv python install 3.11
```

### 3. Autenticar GitHub

```bash
gh auth login
```

Elegir: GitHub.com → HTTPS → Yes (autenticar git) → Login with a web browser.
Copia el código `XXXX-XXXX` que muestra y pégalo en el navegador. Luego:

```bash
gh auth status
```

Debe decir: `Logged in to github.com account ajmedinaa-gif`.

> **Pega los comandos DE UNO EN UNO.** Varios de estos abren un prompt
> interactivo; si pegas un bloque entero, el shell entrega las líneas siguientes
> como respuesta al prompt y se pierden sin ejecutarse.

### 4. Arrancar

```bash
cd ~/Documents/predictive-maintenance-pipeline
claude
```

> **Regla del proyecto:** nunca escribas `python`, `pip` ni `pytest` a secas — el
> Python del sistema es 3.9.6 y el proyecto necesita 3.11. Siempre
> `uv run python`, `uv run pytest`, `uv run pdm-cli`, `uv add <paquete>`.

Docker Desktop no hace falta: la imagen la construye y la prueba GitHub Actions.

Dentro de Claude Code: `/clear` entre fases si el contexto se satura (CLAUDE.md
se recarga solo), `Esc` para interrumpir si se desvía, `#` al inicio de un
mensaje para añadir una regla permanente a CLAUDE.md.

---

# FASE 1 — Andamiaje, EDA riguroso y primer push

```
Lee CLAUDE.md completo antes de hacer nada. Vamos por fases; esta es la Fase 1 y
solo harás lo que se pide aquí.

OBJETIVO: repositorio profesional con barreras de calidad activas, análisis
exploratorio reproducible, y publicado en GitHub.

TAREAS

A. Andamiaje
1. git init y estructura de §15 de CLAUDE.md.
2. pyproject.toml con uv: nombre predictive-maintenance-pipeline,
   requires-python ">=3.11", dependencias de §5, script de consola
   pdm-cli = "predictive_maintenance.cli:app". Ruff line-length 100, reglas
   E,F,I,N,UP,B,SIM,RUF. pytest con --cov=src --cov-fail-under=80. SIN mypy.
3. `uv python pin 3.11`, luego `uv sync`. Verifica el entorno mostrando la
   salida de `uv run python -c "import sys; print(sys.executable)"`: debe
   apuntar a `.venv/bin/python` DENTRO del proyecto. Si la ruta contiene
   `anaconda` o `miniconda`, PARA y repórtalo — ver CLAUDE.md §14.1.
4. .pre-commit-config.yaml con ruff-format, ruff y check-added-large-files.
   pre-commit install.
5. .gitignore para Python + data/interim + data/quarantine + reports/* (pero
   versionando reports/.gitkeep) + .venv + macOS (.DS_Store, ._*,
   .Spotlight-V100, .Trashes, __MACOSX) — ver CLAUDE.md §14.6.
6. LICENSE MIT a nombre de ajmedinaa-gif.
7. cli.py con typer y un comando `version` funcionando + tests/test_cli.py.
8. Makefile: install, lint, test, eda, all. **Todos los targets invocan
   `uv run ...`**, nunca `python`/`pip`/`pytest` a secas (CLAUDE.md §14.1).
   Sintaxis compatible con GNU Make 3.81, que es el que trae macOS: sin
   `.ONESHELL` ni funciones de Make 4.x.

B. EDA
9. src/predictive_maintenance/eda.py con funciones puras y testeables:
   - descriptive_table(df) -> DataFrame: n, nulos, %nulos, min, max, media,
     mediana, desviación típica, asimetría, por atributo
   - missingness_report(df) -> dict: nulos por columna, filas afectadas, si los
     nulos se solapan entre columnas, y reparto por clase. Responder con datos.
   - physical_anomalies(df) -> DataFrame: valores fuera de rango físicamente
     posible
   - univariate_auc(df, target) -> DataFrame: AUC de Mann-Whitney, p-valor y
     dirección, por atributo
   - class_balance(df, target) -> dict: conteos, prevalencia, accuracy del
     clasificador trivial mayoritario
10. tests/test_eda.py verificando contra los valores de CLAUDE.md §6 con
    tolerancia 1e-3. Si algún valor no coincide, PARA y repórtalo. No ajustes el
    test.
11. CLI `pdm-cli eda --dataset lab180`: imprime las tablas y vuelca
    reports/eda_lab180.json.
12. Figuras en reports/figures/lab180/. **Antes de importar pyplot, fijar el
    backend: `matplotlib.use("Agg")`** — en macOS matplotlib intenta abrir
    ventanas y cuelga el CLI (CLAUDE.md §14.4). Guardar con `fig.savefig` +
    `plt.close(fig)`, nunca `plt.show()`. Las figuras son: histogramas por atributo con la clase
    superpuesta, boxplots por clase, mapa de calor de correlación (debe verse
    plano: todas |r| < 0.10), y barras de prevalencia con la línea del 94.4444 %
    de accuracy trivial marcada.

C. Publicación
13. README mínimo: título, una frase del problema, tabla descriptiva real, la
    anomalía de vibración negativa, la prevalencia del 5.5556 % y el bloque
    "En construcción". NO inventes resultados de modelos: aún no existen.
14. Verifica que make lint y make test pasan en verde.
15. Commits con Conventional Commits en la rama `main`. ANTES del primer commit,
    verifica que `git status` no lista `.DS_Store`, `.venv/`, `data/interim/`,
    `data/quarantine/` ni `reports/*` (salvo los `.gitkeep`). Muéstrame la salida
    de `git status --short` en el checkpoint.
16. Publicar en GitHub. Comprueba primero si `gh` está disponible:
    `command -v gh`.
    - **Si `gh` existe**, ejecuta:
      gh repo create ajmedinaa-gif/predictive-maintenance-pipeline --public \
        --description "Mantenimiento predictivo industrial: pipeline reproducible con validación honesta bajo desbalance de clases" \
        --source=. --remote=origin --push
    - **Si `gh` NO existe**, no intentes instalarlo (CLAUDE.md §14.2: nada de
      Homebrew en esta máquina). Deja el repo local listo y PARA: dame en el
      checkpoint las instrucciones exactas para crearlo yo desde
      github.com/new (nombre, descripción, público, SIN README ni .gitignore ni
      licencia porque ya existen en el repo local) más los dos comandos que
      tendré que pegar después:
        git remote add origin https://github.com/ajmedinaa-gif/predictive-maintenance-pipeline.git
        git push -u origin main

NO hagas: modelos, contrato pandera, Docker, CI, dashboard, notebook.

Al terminar, para y presenta el CHECKPOINT en el formato de CLAUDE.md §3.
```

---

# FASE 2 — Contrato de datos, cuarentena y auditor de plausibilidad

```
Fase 2. Lee CLAUDE.md, en particular §6.3 y §7.

OBJETIVO: ningún dato entra al pipeline sin pasar un contrato ejecutable, y el
repositorio sabe diagnosticar si un dataset es sintético.

TAREAS

A. Contrato
1. schema.py: pandera.DataFrameModel Lab180Schema con rangos físicos.
   vibration_mm_s con ge=0 — la amplitud RMS no puede ser negativa.
   failure solo acepta {"yes","no"}.
2. Checks a nivel de dataframe que emiten WARNING, no excepción: prevalencia de
   la clase positiva en [0.01, 0.50], nulos por columna <= 0.10, sin duplicados
   completos. Son avisos de calidad, no de validez.
3. data.py:
   - load_validated(path, schema) -> ValidationResult, dataclass con
     valid: DataFrame, quarantined: DataFrame, report: dict
   - las filas que violan el contrato van a
     data/quarantine/<dataset>_<timestamp>.csv con columna quarantine_reason
   - logging estructurado del resumen
4. config/default.yaml + config/config.py con pydantic-settings: rutas, semilla
   42, esquema de CV, umbrales de aviso. Cero constantes mágicas en el código.
5. CLI `pdm-cli validate --dataset lab180`: imprime el informe y sale con código
   1 si hay filas en cuarentena.

B. Auditor de plausibilidad (extra 1 de §12)
6. plausibility.py con tres diagnósticos independientes, cada uno devolviendo
   un veredicto y la evidencia numérica:
   - correlation_structure(df): correlación máxima absoluta entre atributos
     numéricos. Bandera si es < 0.15 en un dominio donde se esperan variables
     acopladas. Documenta en el docstring la cadena física esperada:
     carga -> temperatura -> viscosidad del lubricante -> vibración.
   - missingness_regularity(df): ¿los nulos están repartidos en partes iguales?
     ¿se solapan alguna vez? ¿se reparten proporcionalmente entre clases?
   - digit_distribution(df): distribución del último decimal por columna.
   - audit(df) -> PlausibilityReport que combina los tres en un veredicto
     "probablemente sintético" / "compatible con datos reales", con la evidencia.
7. Sobre lab180 el veredicto debe ser "probablemente sintético". Escribe el
   resultado en reports/plausibility_lab180.json.

C. Tests
8. tests/test_schema.py y tests/test_data.py:
   - la fila con vibration_mm_s = -0.34 termina en cuarentena con el reason
     correcto, y es la ÚNICA fila en cuarentena
   - un dataframe válido pasa sin filas en cuarentena
   - test con hypothesis: cualquier dataframe generado dentro de los rangos
     válidos pasa el contrato
   - el contrato falla si falta una columna o si el tipo es incorrecto
9. tests/test_plausibility.py: sobre lab180, correlation_structure detecta
   máximo |r| < 0.10 y missingness_regularity detecta cero solapamientos.

D. README
10. Sección "Contrato de datos": por qué la vibración negativa se pone en
    cuarentena en vez de imputarse, y qué habría pasado si no la hubiéramos
    visto. Sección "¿Son reales estos datos?" con el veredicto del auditor y sus
    tres evidencias.

Al terminar, para y presenta el CHECKPOINT.
```

---

# FASE 3 — Modelado y evaluación honesta

```
Fase 3. Lee CLAUDE.md. Repasa las reglas duras 1-7 y toda la §8 antes de
escribir código. Esta es la fase donde se juega la credibilidad del repo.

OBJETIVO: comparar modelos con un protocolo de validación que no se autoengañe.

TAREAS
1. pipeline.py: fábrica que devuelve sklearn.Pipeline completos
   (imputación -> escalado si procede -> clasificador). Zoo de modelos:
   dummy_most_frequent, dummy_stratified, tree_default, tree_shallow_balanced
   (depth=3, min_samples_leaf=5, class_weight=balanced), logistic_balanced,
   logistic_plain (SIN class_weight — es la base de la Fase 4, ver §9.2),
   rf_balanced, gradient_boosting.
   La imputación SIEMPRE dentro del Pipeline, nunca antes del split. Explícalo
   en el docstring.
2. evaluate.py:
   - cross_validate_model(pipe, X, y, cfg) con
     RepeatedStratifiedKFold(5, n_repeats=10, random_state=42)
   - métricas de §8
   - bootstrap_ci(values, n=2000, alpha=0.05) para cada métrica
   - recall_wilson_ci(n_correct, n_positives) — con 8/10 debe dar [0.490, 0.943]
   - nested_cv(pipe, param_grid, X, y, cfg) con GridSearchCV interno 3-fold.
     Docstring citando Vabalas et al. 2019.
   - aggregate_confusion_matrix(...): matriz agregada sobre todos los folds, en
     conteos absolutos y normalizada por fila
3. CLI `pdm-cli train --dataset lab180`: ejecuta todo, escribe
   reports/metrics_lab180.json y una tabla markdown en
   reports/results_lab180.md. La tabla incluye SIEMPRE las dos filas de dummy,
   en primer lugar.
4. Figuras: curvas PR y ROC de todos los modelos en el mismo eje, con la línea
   de azar de PR en 0.0556.
5. tests/test_evaluate.py:
   - el dummy mayoritario obtiene exactamente 0.944444 de accuracy
   - bootstrap_ci sobre una constante devuelve un intervalo degenerado
   - recall_wilson_ci(8, 10) == (0.490, 0.943) con tolerancia 1e-3
   - los folds son estratificados: cada fold de test contiene al menos 1 positivo
   - regresión: PR-AUC de logistic_balanced en [0.55, 0.75] y ROC-AUC de
     rf_balanced en [0.88, 0.96]. Si sale fuera, PARA y repórtalo.
   - regresión: recall de rf_balanced < 0.15 (el resultado titular del repo)
   - fuga de datos: mockear el imputer y verificar que fit se llama una vez por
     fold, con el tamaño del fold de entrenamiento y no del dataset completo
6. README, sección "Resultados" con la tabla generada, y sección
   **"Por qué la accuracy miente en este problema"**: el 94.4444 % del dummy en
   negrita, y el contraste RandomForest ROC-AUC 0.929 / recall 0.03 junto a
   logistic PR-AUC 0.655 / recall 0.77.
7. README, sección "Decisiones de diseño": por qué NO SMOTE, por qué NO holdout
   único, por qué PR-AUC y no ROC-AUC, por qué CV repetida. Con las referencias
   de §8.

Al terminar, para y presenta el CHECKPOINT con la tabla de métricas real y
márcame cualquier celda que se desvíe de CLAUDE.md §8.1.
```

---

# FASE 4 — Calibración, umbral por coste, límites y anexo académico

```
Fase 4. Lee CLAUDE.md, especialmente §9 y §10. La §9.2 describe un error que este
proyecto NO va a cometer: no uses class_weight y umbral por coste a la vez.

OBJETIVO: convertir una probabilidad en una decisión de mantenimiento con
criterio económico, y cuantificar cuánto NO podemos afirmar.

TAREAS

A. Calibración
1. calibration.py:
   - curva de fiabilidad con 10 bins
   - Brier score y su descomposición
   - CalibratedClassifierCV con isotónica y con Platt sobre logistic_plain.
     ADVERTENCIA en el docstring: con 10 positivos la isotónica sobreajusta;
     se espera que Platt (sigmoid) gane. Reporta el resultado real sea cual sea.
   - reproduce la tabla de §9.2 y escríbela en reports/calibration_lab180.json

B. Umbral por coste (extra 4 de §12)
2. config/costs.yaml con los valores de §9.1 en CLP, con comentarios dejando
   claro que son supuestos configurables, NO medidos en campo.
3. threshold.py:
   - expected_cost(y_true, y_prob, threshold, costs) -> float
   - optimal_threshold(y_true, y_prob, costs) por barrido sobre
     np.linspace(0,1,1001)
   - CRÍTICO: el umbral se optimiza DENTRO de cada fold de CV, sobre las
     predicciones out-of-fold del fold de entrenamiento. Nunca sobre el fold de
     test. Documéntalo en el docstring.
   - cost_curve(...) para graficar coste vs umbral
4. tests/test_threshold.py:
   - TEST CLAVE: para un modelo perfectamente calibrado simulado,
     optimal_threshold converge al valor teorico de CLAUDE.md §9.1,
     t* = (C_FP-C_TN)/((C_FP-C_TN)+(C_FN-C_TP)) = 0.0556, tolerancia 0.01.
     OJO: la formula incluye C_TP, que NO es cero. Si implementas
     C_FP/(C_FP+C_FN) obtendras 0.0476 y estara mal.
   - el coste esperado al umbral óptimo es <= al coste al umbral 0.5, siempre
   - con C_FN == C_FP el umbral resultante es cercano a 0.5
5. Figuras: cost_vs_threshold.png con marcadores en t=0.5 y en t*, y
   calibration_curve.png con la diagonal ideal.

C. Explicabilidad y límites (extras 2 y 3 de §12)
6. explain.py: valores SHAP del mejor modelo calibrado (LinearExplainer o
   TreeExplainer según corresponda), beeswarm global y waterfall de los 10 casos
   positivos.
7. explain.py: tree_root_stability(X, y, n_boot=300) — bootstraps
   estratificados, registrar atributo raíz y profundidad efectiva. Debe
   reproducir aproximadamente los porcentajes de §10.1.
8. tests/test_tree_stability.py — TEST CLAVE E INTENCIONADO:
   assert root_stability_of("vibration_mm_s") < 0.70
   Comentario explicando que es deliberado: el test protege contra la conclusión
   errónea de que "el primer corte es la vibración". No lo conviertas en un test
   de igualdad con 0.513: el valor depende de la semilla.
9. power.py:
   - recall_wilson_ci reutilizado de evaluate.py
   - required_positives(target_recall, margin) -> int. Para recall 0.80 y ±10 pp
     debe dar 62; para ±5 pp, 246.
   - required_observations(n_positives, prevalence). 62 / 0.0555556 = 1116.
   - curva de aprendizaje PR-AUC vs fracción de muestra, con bandas de IC.
     ADVERTENCIA: los valores reales de §10.2 DECRECEN con más datos y tienen
     desviación ±0.30. No lo suavices ni lo presentes como una curva creciente.
     Graficarla tal cual, con las bandas, y anotar en el gráfico que la señal
     está dominada por el ruido.
10. tests/test_power.py verifica los tres valores: 62, 246, 1116.

D. Anexo académico
11. notebooks/00_lab_original.ipynb, ejecutado y con salidas guardadas (NO uses
    nbstripout: queremos que el reclutador las vea en GitHub sin ejecutar nada;
    verifica que pesa < 2 MB). Responde LITERALMENTE, en español, las 9
    preguntas del enunciado original:
    nº de instancias, nº de atributos, min/max/media/desviación/ausentes por
    atributo, verificación de que failure es nominal con dos clases,
    % de instancias correctamente clasificadas, lectura de la matriz de
    confusión, si se equivoca más en fallos o no fallos, atributos que aparecen
    en el árbol, sentido ingenieril de esos atributos, primer atributo de
    división, combinación de condiciones que lleva a failure=yes, facilidad de
    interpretación, ventaja frente a modelos opacos.
    REQUISITOS:
    - cada respuesta numérica se calcula en una celda, nunca se escribe a mano
    - la respuesta al "% correctamente clasificadas" va SIEMPRE acompañada del
      94.4444 % del dummy, en la misma celda
    - la respuesta al "primer atributo de división" incluye la advertencia de
      §10.1: solo es la vibración en el 51 % de los bootstraps
    - la respuesta al "sentido ingenieril" menciona que pressure_bar se asocia
      INVERSAMENTE al fallo (AUC 0.25), compatible con fuga o pérdida de
      lubricación, y que load_percent NO es significativo (p = 0.075)
    - celda final de markdown: "Estas preguntas asumen clases equilibradas. El
      pipeline principal de este repositorio explica por qué la accuracy es la
      métrica equivocada para este problema.", con enlace al README

E. README
12. Sección **"El umbral 0.5 es una decisión, no un valor por defecto"** con la
    fórmula completa de §9.1 (incluyendo C_TP), t* = 0.0556, la tabla de §9.2, el
    gráfico de coste y el ahorro real en CLP. Deja claro que los costes son
    supuestos declarados, no medidos en campo.
12b. Sección corta **"La fórmula no gana al barrido, y eso es un hallazgo"**
    reproduciendo §9.3: el umbral teórico 0.0556 cuesta 51.0 MM CLP y el empírico
    0.114 cuesta 30.0 MM. La fórmula es el ancla teórica que demuestra que 0.5 es
    absurdo; el barrido dentro de cada fold es el procedimiento operativo; y la
    distancia entre ambos mide cuán mal calibrado está un modelo entrenado con 10
    positivos. Preséntalo como hallazgo, no como problema, y NO escondas la
    discrepancia.
13. Sección **"Qué NO podemos afirmar"**: el IC del recall [0.490, 0.943] —
    45 puntos de anchura —, la curva de aprendizaje plana y ruidosa, y la
    recomendación: "para estimar el recall con ±10 pp habría que observar 62
    fallos, aproximadamente 1.116 ciclos de máquina a la prevalencia actual".
14. MODEL_CARD.md completo: uso previsto, uso fuera de alcance, datos de
    entrenamiento (declarar SIMULADOS, con las tres evidencias del auditor de la
    Fase 2), métricas con IC, análisis de subgrupos (omitido y por qué), y la
    sección **"Por qué este modelo no debe desplegarse"**: 10 positivos,
    datos sintéticos, variables mutuamente independientes (|r| < 0.10, irreal en
    maquinaria física), y ausencia de eje temporal — por tanto imposibilidad de
    validación temporal, que es la única válida en mantenimiento predictivo.

Al terminar, para y presenta el CHECKPOINT con el umbral óptimo real, el ahorro
calculado en CLP, y los porcentajes reales de estabilidad de la raíz.
```

---

# FASE 5 — Segundo dataset, contenedores, CI y dashboard

```
Fase 5. Lee CLAUDE.md, sección §13. Última fase. Es larga: preséntame un
CHECKPOINT intermedio al terminar el bloque A antes de seguir con el B.

OBJETIVO: demostrar que el pipeline es una abstracción y no un script atado a un
CSV, y que el resultado es visible en 3 segundos sin instalar nada.

BLOQUE A — Segundo dataset
1. Añade ucimlrepo a las dependencias.
2. datasets.py: protocolo DatasetAdapter con name, download(), schema,
   target_column, positive_label, feature_columns, engineer_features(df).
   Implementaciones Lab180Adapter y AI4I2020Adapter. Refactoriza las fases
   anteriores para usar el adaptador; los tests existentes deben seguir pasando
   sin cambios de comportamiento.
3. CLI `pdm-cli download --dataset ai4i2020` vía fetch_ucirepo(id=601), guarda
   en data/raw/ai4i2020/. Paso separado y explícito (regla dura 8).
4. PRIMERO: verifica las cifras reales del dataset descargado — nº de filas,
   prevalencia de Machine failure, conteo de cada modo TWF/HDF/PWF/OSF/RNF — y
   ACTUALIZA CLAUDE.md §13 con los valores medidos. Si difieren de lo anotado,
   dilo en el checkpoint.
5. AI4I2020Schema en pandera: temperaturas en Kelvin (rango plausible 280-320 K),
   velocidad de rotación > 0, par >= 0, desgaste >= 0, Type en {L, M, H}.
6. engineer_features con las tres variables de §13 y la justificación física de
   cada una en el docstring. Verifica que power_w correlaciona con el modo PWF y
   temp_delta_k con HDF. Si no, repórtalo, no lo fuerces.
7. Ejecuta plausibility.audit() sobre ai4i2020: el veredicto debe ser distinto
   al de lab180. Esa comparación valida el auditor.
8. `pdm-cli run --dataset ai4i2020` con StratifiedKFold(5) sin repeticiones.
9. tests/test_datasets.py: ambos adaptadores cumplen el protocolo, y las
   features derivadas son dimensionalmente correctas (potencia en watts con
   valores plausibles para maquinaria industrial).
10. README, sección **"Un pipeline, dos datasets"**: tabla comparativa lado a
    lado con n, positivos, prevalencia, accuracy del dummy, PR-AUC del mejor
    modelo, IC del recall y umbral óptimo. La conclusión debe saltar a la vista:
    con 180 filas el IC del recall es inútil; con 10.000 es accionable.
    Añade una figura con las curvas PR de ambos datasets en el mismo eje.

--- CHECKPOINT INTERMEDIO AQUÍ ---

BLOQUE B — Contenedores, CI, dashboard y cierre

IMPORTANTE (CLAUDE.md §14.7): esta máquina NO tiene Docker y no se va a instalar.
Escribe el Dockerfile y el docker-compose.yml, y deja que el job `docker` de CI
sea lo que prueba que funcionan. NO pidas ejecutar `docker build` ni
`docker compose up` localmente. El dashboard se prueba con
`uv run streamlit run app/streamlit_app.py`.
11. Dockerfile multi-stage: builder python:3.11-slim + uv instalando en
    /opt/venv; runtime python:3.11-slim, copia el venv, usuario no-root appuser,
    WORKDIR /app, HEALTHCHECK, ENTRYPOINT en pdm-cli. Objetivo < 400 MB, mide y
    repórtalo. .dockerignore agresivo.
12. docker-compose.yml con dos servicios (sin claves `platform`: la máquina y
    CI son ambas x86_64): `pipeline` (ejecuta lab180 y
    ai4i2020, volumen para reports/) y `dashboard` (Streamlit en 8501).
13. .github/workflows/ci.yml: push a main y pull_request; matriz Python 3.11 y
    3.12; checkout -> setup-uv con caché -> uv sync -> ruff check ->
    ruff format --check -> pytest con cobertura -> subir cobertura como
    artefacto. Job separado `docker`: build y `docker run ... pdm-cli version`.
    Job separado `pipeline`: ejecuta lab180 y verifica que
    reports/metrics_lab180.json contiene las claves esperadas.
    NO ejecutar ai4i2020 en CI (descarga externa): márcalo como
    workflow_dispatch manual.
14. report.py (que ya existe desde la Fase 1 con el payload JSON): añádele el
    informe HTML estático autocontenido con jinja2. NO metas código de dibujo
    aquí: las figuras viven en figures.py (CLAUDE.md §15). Informe con jinja2, CSS embebido,
    figuras en base64 o SVG inline, cero dependencias de red. Secciones: resumen
    ejecutivo, los datos, plausibilidad, contrato y cuarentena, resultados con
    baselines, umbral por coste, calibración, explicabilidad, límites
    estadísticos. Se genera en reports/report_<dataset>.html.
15. app/streamlit_app.py con cinco pestañas. Lee de reports/*.json, NO reentrena
    nada al arrancar.
    - Datos: tabla descriptiva, nulos, anomalías, veredicto del auditor de
      plausibilidad, selector de dataset
    - Modelos: tabla comparativa con los dummy fijados arriba, curvas PR y ROC
    - Decisión: sliders de coste de fallo no detectado y de inspección, en CLP,
      que recalculan EN VIVO el umbral óptimo, la matriz de confusión resultante
      y el coste esperado. Esta pestaña es la que impresiona: cuídala. Muestra
      siempre el coste al umbral 0.5 al lado, para que se vea la diferencia.
    - Explicabilidad: SHAP beeswarm, árbol renderizado, tabla de estabilidad de
      la raíz
    - Límites: IC del recall, curva de aprendizaje con sus bandas, presupuesto
      estadístico en ciclos de máquina
16. Prepara el repo para Streamlit Community Cloud (.streamlit/config.toml y
    requirements.txt generado con uv export si la plataforma lo requiere). Dame
    las instrucciones exactas para conectarlo desde share.streamlit.io — el
    deploy lo hago yo.
17. .github/workflows/pages.yml que publique los informes HTML en GitHub Pages
    en cada push a main.
18. Makefile completo: install, lint, test, eda, validate, train, data, run,
    docker-build, docker-run, clean, all.
19. README FINAL, reescrito entero:
    1. Título + badges (CI, Python, licencia) + una frase del problema
    2. Enlaces arriba del todo: dashboard en vivo, informe HTML, model card
    3. Resultado principal en la primera pantalla: la tabla comparativa de los
       dos datasets, con la fila del dummy primero
    4. "Por qué la accuracy miente en este problema"
    5. Diagrama del pipeline en mermaid
    6. Quickstart: docker compose up, máximo 3 líneas
    7. Estructura del repo, árbol resumido
    8. Decisiones de diseño y alternativas descartadas
    9. Limitaciones -> enlace a MODEL_CARD.md
    10. Referencias: Vabalas 2019, Saito & Rehmsmeier 2015, Varoquaux 2018,
        Matzka 2020
    11. Autor: @ajmedinaa-gif
    REGLA: ningún número del README escrito a mano. Todos vienen de
    reports/*.json renderizados por plantilla. Añade un test que verifique esa
    correspondencia.
20. CONTRIBUTING.md documentando las 5 fases y cómo se construyó el repo.
21. Revisión final: `make all` en verde localmente, CI en verde **incluido el
    job docker** (ahí se verifica la imagen, no en local), cobertura >= 80 %,
    dashboard arrancando con `uv run streamlit run`, README sin TODOs, y
    git log -p | grep -iE "token|secret|password|ghp_" debe salir vacío.

Al terminar, presenta el CHECKPOINT final: enlace al repo, enlace a CI, enlace a
Pages, tamaño de la imagen Docker **según el log del job docker de CI**,
cobertura, y una lista honesta de lo que quedó pendiente.
```

---

## Sobre ampliar los datos más adelante

No intentes mejorar `lab180`. Diez positivos son diez positivos; ninguna técnica
de limpieza, imputación o aumento cambia eso. Su valor en el repositorio es
servir de contraejemplo.

Si después de la Fase 5 quieres ir más allá de AI4I:

| Dataset | n | Por qué | Dificultad |
|---|---|---|---|
| **MetroPT-3** (UCI id=791) | ~1,5 M | Compresor de aire real del metro de Oporto, con serie temporal y fallos documentados. Permite validación temporal, que es la única válida en mantenimiento predictivo. | Media-alta |
| **NASA C-MAPSS** | 4 conjuntos | Benchmark clásico. Cambia el problema a regresión de vida útil remanente. Muy reconocible. | Media |
| **CWRU Bearing** | señales crudas | Vibración a 12/48 kHz. Obliga a procesamiento de señal (FFT, envolvente). El más impresionante si el puesto es de ingeniería. | Alta |

Recomendación: **MetroPT-3**, porque es el salto de "datos simulados" a "equipo
industrial real en operación".
