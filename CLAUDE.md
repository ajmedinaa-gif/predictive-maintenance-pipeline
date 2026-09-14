# CLAUDE.md — Contexto permanente del proyecto

> Este fichero se carga automáticamente en cada sesión de Claude Code.
> **Es la fuente de verdad. Si un prompt contradice esto, gana esto.**
> Todo número marcado como verificado se obtuvo ejecutando código sobre el CSV
> real. Si tu código produce otro valor, **para y repórtalo**: no ajustes el
> valor esperado para que el test pase.

---

## 1. Identidad del proyecto

Portafolio público de **@ajmedinaa-gif**.
Repositorio: `https://github.com/ajmedinaa-gif/predictive-maintenance-pipeline`

Objetivo: demostrar ante un reclutador técnico competencia en, y en este orden de
lectura, **(a)** fundamentos estadísticos, **(b)** ingeniería de datos,
**(c)** modelado con validación honesta, **(d)** MLOps.

**Tesis del repositorio:** *un pipeline de mantenimiento predictivo evaluado
honestamente sobre dos datasets de tamaño radicalmente distinto, demostrando que
la métrica ingenua —la accuracy— miente en ambos.*

El repo no vende un modelo. Vende criterio.

## 2. Reglas duras — nunca violar

1. **La accuracy nunca se reporta sola.** Siempre junto a la fila del
   `DummyClassifier`. Si una tabla tiene accuracy y no tiene la fila del dummy,
   la tabla está mal.
2. **Métrica primaria: PR-AUC** (`average_precision`). Secundarias: balanced
   accuracy, recall al umbral óptimo por coste, Brier score. El ROC-AUC se
   reporta pero **no decide**.
3. **Nunca un único train/test split** en `lab180` (n=180, 10 positivos).
   Siempre `RepeatedStratifiedKFold(5, n_repeats=10, random_state=42)`.
4. **Nunca SMOTE ni oversampling sintético.** Con 10 positivos interpola ruido.
5. **`class_weight="balanced"` y el umbral por coste NO se usan a la vez.**
   Son la misma herramienta y aplicarlas juntas cuenta el desbalance dos veces.
   La ruta del proyecto para decisiones económicas es: **sin balanceo →
   calibrar con Platt → optimizar umbral**. Ver §7, que lo demuestra con números.
6. **La imputación, el escalado y la calibración van SIEMPRE dentro del
   `sklearn.Pipeline`**, nunca antes del split. Hay un test de fuga de datos.
7. **El umbral se optimiza dentro de cada fold**, sobre predicciones out-of-fold
   del fold de entrenamiento. Nunca sobre el fold de test.
8. **Ningún número del README se escribe a mano.** Todo sale de una ejecución
   real volcada a `reports/*.json` y renderizada por plantilla. Hay un test que
   verifica esa correspondencia.
9. **Los datos que no pasan el contrato van a `data/quarantine/`.** No se borran
   ni se imputan en silencio.
10. **Semillas fijadas.** Dos ejecuciones consecutivas producen métricas
    idénticas. Semilla global 42, definida una sola vez en
    `config/default.yaml`.
11. **Sin llamadas de red en tiempo de ejecución del pipeline.** La descarga de
    datos es un paso explícito y separado (`pdm-cli download`, `make data`).
12. **Datos declarados como SIMULADOS** donde se presenten resultados de
    `lab180`. §5 explica cómo lo sabemos.
13. **Un commit por unidad conceptual.** Nunca un commit gigante
    "add everything". **Jamás `git push --force` sobre `main`.**
14. **Al final de cada fase: parar y presentar el CHECKPOINT.** No continuar sin
    aprobación explícita de la usuaria.
15. Español en documentación y comentarios; inglés en nombres de código y en los
    mensajes de commit.
16. **Las figuras PNG de `reports/figures/` SÍ se versionan, a propósito: el
    destinatario del repo no ejecuta el código.** El resto de salidas
    generadas (`reports/*.json`, `*.md`, `*.html`), no.

## 3. Formato del CHECKPOINT (obligatorio al cerrar cada fase)

```
## CHECKPOINT Fase N — <nombre>

### Qué se construyó
<3-6 bullets>

### Ficheros creados/modificados
<lista con nº de líneas>

### Comandos ejecutados y su salida real
<pegar la salida, no parafrasear>

### Números medidos en esta fase
<tabla; marcar los que NO coinciden con este fichero; si no se midió nada,
decir "ninguno">

### Tests: X pasan / Y fallan / Z omitidos — cobertura N%

### Commit
<hash corto + mensaje>

### Decisiones que tomé y que podrías querer revertir
<bullets, o "ninguna">

### Bloqueos o supuestos
<bullets, o "ninguno">

### Qué deberías verificar tú antes de seguir
<2-4 comandos concretos que la usuaria pueda copiar y ejecutar>
```

## 4. Convención de commits

Conventional Commits, en inglés, cuerpo opcional en inglés:

```
feat(schema): add pandera data contract with physical range checks
fix(data): route negative vibration readings to quarantine
test(stability): assert decision tree root split is unstable under bootstrap
docs(readme): add dummy baseline to results table
chore(ci): add ruff gate to GitHub Actions
```

Un commit por fase como mínimo, varios si la fase tiene partes separables.

## 5. Stack fijado

`uv` · `pandas` · `numpy` · `scikit-learn>=1.5` · `scipy` · `statsmodels` ·
`pandera` · `pydantic-settings` · `typer` · `jinja2` · `matplotlib` · `shap` ·
`streamlit` · `ucimlrepo` · `pytest` + `pytest-cov` + `hypothesis` · `ruff` ·
`pre-commit` · Docker multi-stage no-root · GitHub Actions.

No añadir dependencias fuera de esta lista sin preguntar primero.

**Deliberadamente FUERA:**
- **`mypy --strict`** — ceremonia desproporcionada para un proyecto de análisis.
- **`mlflow`** — no hay espacio de búsqueda de hiperparámetros que lo justifique.
  Los resultados van a `reports/*.json`, que es versionable y legible en GitHub,
  que es donde mira el reclutador.
- **Homebrew** — ver §14.2: no sirve en esta máquina.

Cobertura objetivo: **80 %**.

## 6. Números verificados de `lab180`

Fichero: `data/raw/lab180/Lab1_engineering_failures.csv`. 180 filas, 6 columnas.

### 6.1 Tabla descriptiva (tolerancia 1e-3)

| atributo | n | nulos | %nulos | min | max | media | mediana | desv. típica |
|---|---|---|---|---|---|---|---|---|
| temperature_c | 176 | 4 | 2.22 | 47.00 | 89.80 | 67.8102 | 67.950 | 7.6651 |
| vibration_mm_s | 176 | 4 | 2.22 | **-0.34** | 9.59 | 4.2438 | 4.295 | 1.3349 |
| pressure_bar | 176 | 4 | 2.22 | 4.27 | 10.19 | 6.7630 | 6.750 | 1.1330 |
| hours_since_maintenance | 180 | 0 | 0.00 | 20 | 896 | 473.1889 | 440.0 | 262.3427 |
| load_percent | 180 | 0 | 0.00 | 31.50 | 100.00 | 71.0900 | 71.400 | 14.4342 |

`hours_since_maintenance` es entero; el resto float.

### 6.2 Variable objetivo

- `failure` es nominal con exactamente dos clases `{"no", "yes"}`. Cero nulos.
- `no` = 170, `yes` = 10.
- Prevalencia = 0.0555556 → **5.5556 %**
- **Accuracy del clasificador trivial mayoritario = 94.4444 %** ← el número que
  va delante de cualquier accuracy que reporte el repo.

### 6.3 Nulos y anomalías

- 12 filas tienen exactamente 1 nulo. **Ninguna fila tiene 2 o más.** Los nulos
  están repartidos 4/4/4 entre las tres primeras columnas sin solaparse jamás.
- **Las 12 filas con nulos pertenecen todas a la clase `no`.** Ninguna a `yes`.
- Duplicados completos: 0.
- **Anomalía física: fila índice 82, `vibration_mm_s = -0.34`.** La amplitud RMS
  de vibración no puede ser negativa. Esa fila va a CUARENTENA, no se imputa.
  Resto de la fila: temp 79.8, presión 4.53, horas 250, carga 63.8, failure=no.
- Sin otras violaciones: temperatura, presión, horas y carga están en rango.

### 6.4 Asociación univariante (AUC de Mann-Whitney, dos colas)

| atributo | media `yes` | media `no` | AUC | p-valor | lectura |
|---|---|---|---|---|---|
| vibration_mm_s | 5.806 | 4.150 | 0.8503 | 0.00020 | el más fuerte |
| temperature_c | 73.590 | 67.462 | 0.7259 | 0.01669 | significativo |
| hours_since_maintenance | 668.400 | 461.706 | 0.7218 | 0.01871 | significativo |
| load_percent | 79.740 | 70.581 | 0.6682 | 0.07459 | **NO significativo** |
| pressure_bar | 5.797 | 6.821 | 0.2515 | 0.00846 | **INVERSO** |

### 6.5 Correlaciones entre atributos

**Todas las correlaciones de Pearson tienen |r| < 0.10** (máximo 0.097 entre
`load_percent` y `vibration_mm_s`). Las cinco variables son mutuamente
independientes.

## 7. Por qué sabemos que `lab180` es sintético

Tres firmas, todas verificadas:

1. **Independencia mutua total.** En maquinaria física, carga → temperatura →
   viscosidad del lubricante → vibración forman una cadena causal acoplada.
   Correlaciones de 0.00 entre las cinco variables no ocurren en un equipo real.
2. **Patrón de nulos demasiado regular.** Exactamente 4 nulos en cada una de
   tres columnas, jamás dos en la misma fila, y ninguno en la clase minoritaria.
3. **Rangos limpios.** Una sola anomalía física, inyectada deliberadamente.

Esto no descalifica el ejercicio: lo define. `lab180` sirve de contraejemplo. El
módulo `plausibility.py` (§12) convierte estas tres firmas en un diagnóstico
ejecutable y reutilizable.

## 8. Protocolo de validación

- `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)` para
  `lab180`. Holdout único descartado: dejaría 2 positivos en test, donde un solo
  error mueve el recall 50 puntos.
- `StratifiedKFold(5)` sin repeticiones para `ai4i2020` (~339 positivos).
- Métricas: `average_precision` (PRIMARIA), `balanced_accuracy`, `roc_auc`,
  `recall`, `precision`, `f1`, `accuracy`, `brier_score_loss`.
- IC por bootstrap sobre los folds, n=2000, alpha=0.05.
- **IC del recall por Wilson** sobre el conteo de positivos, no por bootstrap.
- Nested CV (`GridSearchCV` interno 3-fold) para modelos con hiperparámetros.

### 8.1 Resultados medidos — CV repetida (5 × 10, semilla 42)

**IMPORTANTE: estas cifras se miden sobre 179 filas, no 180.** El pipeline
entrena sobre el dataset que sale del contrato (§2.9): la fila 82 va a
cuarentena y no entra al modelo. Por eso la prevalencia operativa es
10/179 = 0.0559 y la accuracy del dummy es 169/179 = 0.9441, mientras que §6
—que describe el CSV **crudo**— dice 0.0556 y 0.9444. Las dos cosas son
correctas y describen datasets distintos. No las cuadres a la fuerza.

Medido con `pdm-cli train --dataset lab180`, `RandomForestClassifier` con
`n_estimators` por defecto (100):

| modelo | PR-AUC | ROC-AUC | recall | precision | bal.acc | accuracy | Brier |
|---|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.0559 | 0.5000 | 0.00 | 0.000 | 0.5000 | 0.9441 | 0.0559 |
| dummy_stratified | 0.0714 | 0.4832 | 0.08 | 0.040 | 0.4832 | 0.8413 | 0.1587 |
| tree_default | 0.1214 | 0.5684 | 0.19 | 0.129 | 0.5684 | 0.9045 | 0.0955 |
| tree_shallow_balanced | 0.1781 | 0.6438 | 0.40 | 0.165 | 0.6363 | 0.8461 | 0.1191 |
| logistic_balanced | 0.6748 | 0.9189 | **0.78** | 0.419 | 0.8524 | 0.9168 | 0.0667 |
| logistic_plain | **0.7153** | **0.9292** | 0.35 | 0.450 | 0.6700 | 0.9542 | **0.0329** |
| rf_balanced | 0.6117 | 0.9254 | **0.14** | 0.220 | 0.5659 | 0.9441 | 0.0411 |
| gradient_boosting | 0.4387 | 0.8496 | 0.14 | 0.170 | 0.5576 | 0.9285 | 0.0657 |

**El resultado titular del repositorio:** `rf_balanced` tiene ROC-AUC 0.925 —de
los mejores de la tabla— y detecta el **14 %** de los fallos. `gradient_boosting`
hace lo mismo con ROC-AUC 0.850. Mientras tanto `logistic_balanced`, con un
ROC-AUC *peor* (0.919), detecta el **78 %**.

**Cómo se formula ese argumento correctamente** (importa, porque es el titular):
el ROC-AUC mide **calidad de ordenación** y es independiente del umbral. El
RandomForest ordena bien, pero comprime sus probabilidades por debajo de 0.5, así
que al umbral por defecto casi no clasifica a nadie como fallo. El ROC-AUC mide
lo primero y lo premia; el recall mide lo segundo y lo castiga. **No digas que
"el umbral explica el ROC-AUC de 0.925": el umbral explica el recall de 0.14.**

**Nota para la Fase 4:** `logistic_plain` ya es el mejor ordenador (PR-AUC 0.7153)
y el mejor calibrado con diferencia (Brier 0.0329, la mitad que los demás). Es
exactamente el modelo que §9.2 designa para calibración + umbral. La tabla lo
confirma antes de empezar esa fase.

**Tolerancias para tests de regresión — anchas a propósito.** Con 10 positivos el
recall se mueve en saltos de 0.1 por fold; un test estrecho se rompe al cambiar
de versión de scikit-learn sin que nada esté mal:
- PR-AUC de `logistic_balanced` en [0.55, 0.80]
- ROC-AUC de `rf_balanced` en [0.85, 0.98]
- recall de `rf_balanced` **< 0.35** (lo que se protege es "el recall es malo pese
  al buen ROC-AUC", no que valga exactamente 0.14)
- PR-AUC de cualquier modelo real > PR-AUC de ambos dummy

### 8.2 Matrices de confusión agregadas out-of-fold, `StratifiedKFold(5, shuffle, seed 42)`

| modelo | TN | FP | FN | TP | recall | accuracy | IC Wilson del recall |
|---|---|---|---|---|---|---|---|
| tree_default | 162 | 8 | 10 | **0** | 0.00 | 0.9000 | [0.000, 0.278] |
| tree_shallow_balanced | 152 | 18 | 7 | 3 | 0.30 | 0.8611 | [0.108, 0.603] |
| rf_balanced | 170 | 0 | 10 | **0** | 0.00 | 0.9444 | [0.000, 0.278] |
| logistic_balanced | 156 | 14 | 2 | **8** | 0.80 | 0.9111 | **[0.490, 0.943]** |

Respuestas al enunciado académico que salen de aquí:
- El modelo **se equivoca mucho más prediciendo fallos** (falsos negativos) que
  no-fallos. El árbol por defecto falla los 10 y encima produce 8 falsas alarmas.
- El IC del recall del mejor modelo abarca **45 puntos porcentuales**. Con 10
  positivos, "recall 0.80" y "recall 0.50" son indistinguibles con estos datos.

## 9. Coste, calibración y umbral — LEER ANTES DE LA FASE 3

### 9.1 Matriz de coste (SUPUESTOS declarados, configurables)

`config/costs.yaml`, con comentarios dejando claro que son supuestos y **no
medidos en campo**. Moneda: **CLP**, no euros.

```yaml
cost_false_negative: 10000000   # CLP  parada no planificada, daño al equipo
cost_false_positive:   500000   # CLP  inspección innecesaria
cost_true_positive:   1500000   # CLP  mantenimiento planificado (también cuesta)
cost_true_negative:         0
```

Como `C_TP` no es cero, la fórmula del umbral óptimo para un modelo **calibrado**
NO es `C_FP/(C_FP+C_FN)`, sino:

```
t* = (C_FP − C_TN) / ((C_FP − C_TN) + (C_FN − C_TP))
   = 500 000 / (500 000 + 10 000 000 − 1 500 000)
   = 500 000 / 9 000 000
   = 0.0556
```

**t\* = 0.0556.** Si alguna vez ves 0.048 escrito en este proyecto, viene de
ignorar `C_TP` y está mal.

Test clave: para un modelo perfectamente calibrado **simulado**,
`optimal_threshold` debe converger a 0.0556 con tolerancia 0.01.

### 9.2 Verificado: qué variante usar, y por qué

`StratifiedKFold(5, shuffle, seed 42)`, costes de §9.1, cifras en millones de CLP:

| variante | Brier | PR-AUC | t\* empírico | coste t=0.5 | coste t\* | ahorro | conf. en t\* |
|---|---|---|---|---|---|---|---|
| logistic **balanced** | 0.0726 | 0.5696 | **0.569** | 39.0 | 38.5 | 0.5 (**1.3 %**) | FN=2 TP=8 FP=13 |
| logistic plain | 0.0343 | 0.5572 | 0.065 | 67.5 | 34.5 | 33.0 (48.9 %) | FN=1 TP=9 FP=22 |
| logistic plain + **Platt** | 0.0361 | **0.6402** | 0.114 | 91.5 | **30.0** | 61.5 (**67.2 %**) | FN=1 TP=9 FP=13 |
| logistic plain + isotónica | 0.0363 | 0.4903 | 0.093 | 59.0 | 38.0 | 21.0 (35.6 %) | FN=2 TP=8 FP=12 |

**Decisión del proyecto: `logistic_plain` + calibración Platt (`sigmoid`) +
umbral optimizado.** Platt gana a la isotónica, como se esperaba: con 10
positivos la isotónica sobreajusta y su PR-AUC cae a 0.49.

### 9.3 Verificado y contraintuitivo: la fórmula NO gana al barrido

Aplicando el umbral **teórico** 0.0556 al mejor modelo calibrado, el coste es
**51.0 MM CLP**. El umbral **empírico** 0.114 da **30.0 MM CLP**. El barrido
empírico gana por 21 MM.

Razón: `t* = 0.0556` es óptimo solo si el modelo está **perfectamente**
calibrado. Con 10 positivos la calibración de Platt es aproximada (Brier 0.036),
y al umbral teórico el modelo dispara 55 falsas alarmas en lugar de 13.

**Cómo presentarlo en el README, sin contradicción:** la fórmula es el **ancla
teórica** que demuestra por qué 0.5 es absurdo aquí — el umbral correcto está un
orden de magnitud más abajo. El barrido empírico dentro de cada fold (regla dura
7) es el **procedimiento operativo**. Ambas cosas apuntan en la misma dirección y
la diferencia entre ellas es, ella misma, una medida de cuán mal calibrado está
un modelo entrenado con 10 positivos. Es un hallazgo, no un problema.

## 10. Límites estadísticos verificados

### 10.1 Estabilidad de la raíz del árbol (300 bootstraps estratificados, seed 42)

Árbol `max_depth=3, min_samples_leaf=5, class_weight="balanced"`:

| atributo raíz | % de bootstraps |
|---|---|
| vibration_mm_s | **51.3 %** |
| pressure_bar | 19.3 % |
| load_percent | 14.3 % |
| temperature_c | 7.7 % |
| hours_since_maintenance | 7.3 % |

Profundidad efectiva media: 2.99.

**El primer corte es la vibración solo en la mitad de los remuestreos.** El test
`assert root_stability < 0.70` es deliberado y protege contra la conclusión
errónea "el atributo que decide el fallo es la vibración". Documentarlo con
comentario para que nadie lo "arregle". **No lo conviertas en una igualdad con
0.513**: el valor depende de la semilla.

### 10.2 Curva de aprendizaje (PR-AUC, `StratifiedKFold(5)`)

| n entrenamiento | PR-AUC test | desv. |
|---|---|---|
| 43 | 0.778 | ±0.333 |
| 63 | 0.741 | ±0.348 |
| 83 | 0.745 | ±0.317 |
| 103 | 0.734 | ±0.318 |
| 123 | 0.611 | ±0.319 |
| 144 | 0.633 | ±0.301 |

**La curva baja en vez de subir, con bandas de ±0.30.** No es una curva de
aprendizaje: es ruido. Publicarla tal cual, con las bandas, y decir exactamente
eso. Es uno de los gráficos más honestos del repositorio. **No la suavices ni la
presentes como creciente.**

### 10.3 Presupuesto estadístico

Para estimar un recall objetivo de 0.80 por aproximación normal:

| margen deseado | fallos necesarios | ciclos de máquina a prevalencia 5.5556 % |
|---|---|---|
| ±10 pp | **62** | **1.116** |
| ±5 pp | **246** | **4.428** |

Hoy: 10 fallos, 180 ciclos. Recomendación del repo: *"para estimar el recall con
±10 pp habría que observar 62 fallos, aproximadamente 1.116 ciclos de máquina a
la prevalencia actual"*.

## 11. Interpretación de ingeniería (para README y notebooks)

- **Vibración**: firma canónica de degradación de rodamientos y desalineación de
  ejes. El aumento de amplitud RMS precede al fallo catastrófico.
- **Presión baja → fallo** (contraintuitivo, AUC 0.25): pérdida de estanqueidad,
  fuga, desgaste de sellos, o bomba perdiendo eficiencia volumétrica. **NO** es
  "más presión = más estrés".
- **Horas desde mantenimiento**: riesgo acumulado, rama derecha de la curva de
  bañera.
- **Temperatura**: fricción, refrigeración deficiente, degradación del
  lubricante — la viscosidad cae con la temperatura, lo que realimenta el
  desgaste.
- **Carga**: dirección esperada pero **no significativa** (p=0.075) en `lab180`.
  No inventar una historia sobre ella: decir que no se puede concluir.

## 12. Los cuatro extras que diferencian este repo

1. **`plausibility.py` — auditor de plausibilidad física.** Dictamina si un
   dataset es sintético: correlación entre variables que deberían estar
   acopladas, regularidad del patrón de nulos, distribución de dígitos finales.
   Sobre `lab180` debe concluir "simulado"; sobre `ai4i2020`, no.
2. **`power.py` — presupuesto estadístico.** Curva de margen de error vs número
   de fallos observados, traducida a ciclos de máquina.
3. **`tests/test_tree_stability.py` — el test que afirma la inestabilidad.**
4. **Pestaña "Decisión" del dashboard** — sliders de `C_FN`, `C_FP` y `C_TP` que
   recalculan en vivo umbral óptimo, matriz de confusión y coste esperado, con el
   coste al umbral 0.5 siempre al lado para que se vea la diferencia.

## 13. Los dos datasets

| | lab180 | ai4i2020 |
|---|---|---|
| origen | laboratorio, **simulado** | UCI id=601, Matzka (2020) |
| n | 180 | ~10 000 |
| positivos | 10 (5.5556 %) | ~339 (~3.4 %) — **verificar al descargar** |
| variables | 5, mutuamente independientes | 6 + `Type`, físicamente acopladas |
| modos de fallo | no etiquetados | 5: TWF, HDF, PWF, OSF, RNF |
| papel en el repo | **contraejemplo**: qué NO se puede concluir con pocos datos | demostración de que el pipeline es una abstracción |

`lab180` **no se presenta como resultado principal.**

`ai4i2020`: descarga con `ucimlrepo.fetch_ucirepo(id=601)`. Features `Type`
(L/M/H), `Air temperature [K]`, `Process temperature [K]`,
`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`; target
`Machine failure`. **Verificar las cifras al descargar y corregir esta tabla si
difieren; decirlo en el checkpoint.**

Ventaja decisiva sobre `lab180`: las features están **físicamente acopladas**, lo
que permite feature engineering con sentido ingenieril real, y ~339 positivos
permiten estimar el recall con intervalos de confianza útiles.

Features derivadas, con justificación física en el docstring:
- `power_w = torque_nm * rotational_speed_rpm * 2*pi/60` (potencia mecánica)
- `temp_delta_k = process_temperature_k - air_temperature_k` (disipación térmica)
- `wear_x_torque = tool_wear_min * torque_nm` (sobreesfuerzo acumulado)

### 13.1 El mismo pipeline debe correr sobre ambos

Es la prueba de que el código es una abstracción y no un script. Un
`DatasetAdapter` por dataset, todo lo demás compartido:

```
pdm-cli run --dataset lab180
pdm-cli run --dataset ai4i2020
```

## 14. Entorno de desarrollo: macOS Intel

**Entorno verificado:** MacBook Air **Intel (x86_64)**, zsh, `uv` 0.12.13 en
`~/.local/bin`, Python 3.11.16 instalado por uv, Python del sistema 3.9.6 (no
usar), `gh` 2.100.0 autenticado como `ajmedinaa-gif` con scopes
`gist, read:org, repo, workflow`, **sin Docker Desktop**, **sin conda en el
PATH** (Anaconda existe en el home pero no interfiere).

### 14.1 Nunca `python`, `pip` ni `pytest` a secas

El Python del sistema es 3.9.6 y el proyecto necesita 3.11. Usa siempre
`uv run python`, `uv run pytest`, `uv run pdm-cli`, `uv add <paquete>`. En el
`Makefile`, **todos** los targets llaman a `uv run ...`.

Tras `uv sync`, verifica y muestra la salida de
`uv run python -c "import sys; print(sys.executable)"`. Debe apuntar a
`.venv/bin/python` dentro del proyecto. Si la ruta contiene `anaconda` o
`miniconda`, PARA y repórtalo.

### 14.2 Homebrew no sirve en esta máquina — regla crítica

Desde septiembre de 2026 **Homebrew dejó de publicar bottles (binarios
precompilados) para macOS Intel x86_64**. Cada `brew install` compila desde
código fuente (instalar `gh` arrastró compilar el toolchain completo de Go).

**Nunca propongas `brew install <algo>`.** Alternativas, en orden:
1. Un paquete de PyPI vía `uv add`. Casi siempre existe.
2. Un binario oficial precompilado del proyecto (release de GitHub, .pkg firmado).
3. Nada: replantear la fase para no necesitar la herramienta.

**Esto NO afecta a las dependencias Python.** numpy, scipy, scikit-learn,
statsmodels, shap y streamlit publican rueda `macosx_x86_64` en PyPI; `uv sync`
no compila nada. Si `uv sync` empieza a compilar C, es que una dependencia nueva
no tiene rueda: repórtalo y busca alternativa.

Apple retira Intel en macOS 27 y GitHub Actions retira los runners macOS Intel
en 2027. **Ninguna de las dos cosas afecta a este proyecto**: el CI corre en
`ubuntu-latest`.

### 14.3 Pegar comandos de uno en uno

Varios comandos abren prompts interactivos. Si se pega un bloque de varias
líneas, el shell entrega las siguientes como respuesta al prompt y se pierden sin
ejecutarse. **Un comando por bloque si puede pedir confirmación**, y di qué
respuesta se espera.

### 14.4 matplotlib sin interfaz gráfica

Fijar el backend **antes** de importar pyplot:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Guardar con `fig.savefig(...)` + `plt.close(fig)`. Nunca `plt.show()`.

### 14.5 Utilidades BSD, no GNU

- `sed -i` requiere argumento: `sed -i '' 's/a/b/' fichero`. **Mejor: no uses
  `sed -i`.** Edita con Python o con las herramientas de edición nativas.
- `date`, `stat`, `readlink` y `find` tienen banderas distintas a las de GNU. Si
  necesitas una, usa Python: es portable y CI corre en Linux.
- `make` de macOS es GNU Make **3.81**. Sin `.ONESHELL` ni sintaxis de Make 4.x.

**Regla general: todo lo que el pipeline necesite hacer, lo hace Python.** El
shell solo orquesta.

### 14.6 `.gitignore` específico de macOS

Obligatorio: `.DS_Store`, `._*`, `.Spotlight-V100`, `.Trashes`, `__MACOSX`. Un
`.DS_Store` versionado en un repo de portafolio es una señal descuidada.

### 14.7 Docker: lo construye CI, no esta máquina

x86_64 coincide con `ubuntu-latest`, así que **NO** añadas `--platform
linux/amd64` en ningún sitio: sobra.

El `Dockerfile` y el `docker-compose.yml` se escriben y se versionan igual — son
parte del entregable. Pero **el job `docker` de CI es lo que prueba que
funcionan**, y ese job no puede ser opcional ni `continue-on-error`. Un Air Intel
construyendo scipy + sklearn + shap + streamlit tarda 15-30 minutos y se
estrangula térmicamente.

- **No pidas a la usuaria ejecutar `docker build` ni `docker compose up`.** No
  tiene Docker instalado y no es un bloqueo: el checkpoint se cierra con CI verde.
- El dashboard se desarrolla y se prueba **sin Docker**:
  `uv run streamlit run app/streamlit_app.py`.
- En el README, `docker compose up` se documenta como quickstart para quien clone
  el repo. CI demuestra que funciona.

### 14.8 Sistema de ficheros insensible a mayúsculas

APFS no distingue `Data/` de `data/`; Linux y GitHub sí. Nombres en minúsculas y
consistentes desde el principio.

## 15. Estructura del repositorio

```
src/predictive_maintenance/   eda, schema, data, datasets, pipeline, evaluate,
                              threshold, calibration, explain, power,
                              plausibility, figures, report, cli

  figures.py  dibuja: todas las funciones figure_* y make_*_figures.
              matplotlib con backend Agg (§14.4). NADA de logica de negocio.
  report.py   construye el payload: build_*_report / write_*_report -> JSON.
              En la Fase 5 crece con el renderizador HTML de jinja2 que lee
              ese mismo JSON. NUNCA dibuja: importa figures si necesita rutas.
tests/                        cobertura >= 80 %
config/                       default.yaml, costs.yaml
data/raw|interim|quarantine/
notebooks/00_lab_original.ipynb   anexo académico, con salidas versionadas
app/streamlit_app.py
reports/                      json, md, figures/, html
tools/fase.py                 extrae prompts de FASES.md
.github/workflows/ci.yml
```

## 16. Referencias a citar en el README

- **Vabalas, A. et al. (2019).** *Machine learning algorithm validation with a
  limited sample size.* PLOS ONE 14(11): e0224365. — nested CV.
- **Saito, T., Rehmsmeier, M. (2015).** *The Precision-Recall Plot Is More
  Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced
  Datasets.* PLOS ONE 10(3): e0118432. — PR-AUC sobre ROC-AUC.
- **Varoquaux, G. (2018).** *Cross-validation failure: small sample sizes lead to
  large error bars.* NeuroImage 180: 68-77. — anchura de los IC.
- **Matzka, S. (2020).** *Explainable Artificial Intelligence for Predictive
  Maintenance Applications.* — dataset AI4I 2020.
