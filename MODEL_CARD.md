# Model Card — `predictive-maintenance-pipeline` / `lab180`

> Formato inspirado en Mitchell et al. (2019), *Model Cards for Model
> Reporting*. Todos los números de esta ficha salen de `reports/*.json`,
> generados por `pdm-cli train|calibrate|explain|limits --dataset lab180`
> (CLAUDE.md §2.8): ninguno está escrito a mano.

## Resumen

Un clasificador binario (`failure` ∈ {`no`, `yes`}) sobre cinco lecturas de
sensor de una máquina de laboratorio: `logistic_plain` calibrado con Platt
(`sigmoid`) y un umbral de decisión optimizado por coste (CLAUDE.md §9). Es el
modelo del **contraejemplo** del repositorio — ver [README](README.md), sección
"¿Son reales estos datos?" — no el resultado principal.

## Uso previsto

- Demostrar, con números reales y código auditable, un pipeline honesto de
  mantenimiento predictivo bajo desbalance de clases severo: contrato de
  datos, validación cruzada repetida, calibración, umbral por coste,
  explicabilidad y cuantificación explícita de la incertidumbre.
- Servir de material de portafolio técnico y de comparación metodológica
  frente al segundo dataset del repositorio (`ai4i2020`, más grande y
  físicamente acoplado).
- Enseñar, con evidencia, por qué la accuracy y el ROC-AUC pueden engañar bajo
  desbalance severo, y por qué un intervalo de confianza importa tanto como
  la media.

## Uso FUERA de alcance

- **Cualquier decisión de mantenimiento real sobre equipos físicos.** Ver "Por
  qué este modelo no debe desplegarse" más abajo — no es una advertencia
  genérica, son cuatro razones concretas y verificadas.
- Extrapolar los coeficientes o las importancias SHAP de este modelo a otra
  maquinaria, otro fabricante de sensores u otro régimen de carga: el dataset
  es de una única máquina de laboratorio, simulada.
- Usar los costes de `config/costs.yaml` como una tarifa real: son SUPUESTOS
  declarados para poder demostrar el procedimiento (CLAUDE.md §9.1), no una
  medición de campo.
- Cualquier comparación de subgrupos (por tipo de máquina, turno, operario...):
  el dataset no tiene esas columnas — ver "Análisis de subgrupos" más abajo.

## Datos de entrenamiento

⚠️ **Datos SIMULADOS de laboratorio, no mediciones de campo.**
`data/raw/lab180/Lab1_engineering_failures.csv` — 180 filas, 5 sensores +
`failure`. 179 filas entrenan realmente (la fila 82, `vibration_mm_s=-0.34`,
un imposible físico, va a `data/quarantine/` — CLAUDE.md §2.9). 10 positivos
(5.56 % de prevalencia).

El auditor de plausibilidad física (`src/predictive_maintenance/plausibility.py`,
CLAUDE.md §7 y §12) dictamina **"probablemente sintético"**, con tres
evidencias independientes, todas en `reports/plausibility_lab180.json`:

1. **Independencia mutua total.** La correlación de Pearson máxima entre
   cualquier par de los cinco sensores es **0.0972** (`vibration_mm_s` /
   `load_percent`), muy por debajo del umbral de 0.15 al que se esperaría un
   acoplamiento físico real (carga → temperatura → viscosidad del lubricante →
   vibración es una cadena causal acoplada en maquinaria real).
2. **Patrón de nulos demasiado regular.** 12 nulos, repartidos en exactamente
   **[4, 4, 4]** entre `temperature_c`, `vibration_mm_s` y `pressure_bar`, que
   **jamás se solapan** en la misma fila.
3. **Última cifra decimal compatible con muestreo uniforme** en los cinco
   atributos (evidencia de apoyo, no decisiva por sí sola).

## Métricas, con intervalo de confianza

Protocolo: `RepeatedStratifiedKFold(5, n_repeats=10, seed=42)` para la media;
un único `StratifiedKFold(5, shuffle=True, seed=42)` para la matriz de
confusión agregada (CLAUDE.md §8). Fuente: `reports/metrics_lab180.json`,
`reports/calibration_lab180.json`.

| métrica | valor | IC 95 % | método |
|---|---|---|---|
| PR-AUC (`logistic_plain`) | 0.7153 | [0.634, 0.790] | bootstrap sobre folds, n=2000 |
| Brier (`logistic_plain`, sin calibrar) | 0.0329 | [0.029, 0.037] | bootstrap sobre folds, n=2000 |
| Brier (`logistic_plain` + Platt, out-of-fold anidado) | 0.0370 | — | `calibration.evaluate_cost_variants` |
| recall a umbral óptimo por coste (`logistic_plain`+Platt) | 8/10 = 0.80 | **[0.490, 0.943]** | Wilson, sobre el conteo de positivos |
| accuracy del `DummyClassifier` mayoritario | 94.41 % | — | referencia obligatoria (CLAUDE.md §2.1) |

**El IC del recall, [0.490, 0.943], tiene 45 puntos porcentuales de ancho.**
No es un intervalo de confianza operativo: "0.80" y "0.50" son estadísticamente
indistinguibles con 10 positivos. Ver README, sección "Qué NO podemos
afirmar".

## Análisis de subgrupos

**Omitido, deliberadamente.** `lab180` no tiene ninguna columna de subgrupo
protegido ni operativo (turno, tipo de máquina, ubicación, operario, fecha):
solo cinco lecturas de sensor y la etiqueta. Publicar un análisis de
subgrupos inventando una partición arbitraria del dataset (por ejemplo, por
cuartil de una variable continua) sería fabricar una conclusión sobre datos
que no la sostienen — con 10 positivos en el TOTAL del dataset, cualquier
subgrupo tendría, en el mejor de los casos, un puñado de positivos: menos
que el propio dataset completo, cuyo IC de recall ya es demasiado ancho para
ser útil.

## Explicabilidad

`shap.LinearExplainer` sobre `logistic_plain` (el modelo base, sin calibrar —
la calibración de Platt es una transformación monótona 1D que no cambia qué
atributo empuja la predicción). Importancia media `|SHAP|`
(`reports/explainability_lab180.json`):

| atributo | importancia media `|SHAP|` |
|---|---|
| hours_since_maintenance | 1.0713 |
| vibration_mm_s | 1.0223 |
| pressure_bar | 0.9247 |
| temperature_c | 0.7739 |
| load_percent | 0.6236 |

Estabilidad de la raíz del árbol (`tree_shallow_balanced`, 300 bootstraps
estratificados, CLAUDE.md §10.1): `vibration_mm_s` es el primer corte en el
**56.0 %** de los remuestreos — el más frecuente, pero lejos de ser
determinista. `tests/test_tree_stability.py` protege explícitamente que este
valor se mantenga por debajo del 70 %, precisamente para que nadie concluya
"el atributo que decide el fallo es la vibración" a partir de un único árbol.

## Por qué este modelo no debe desplegarse

Cuatro razones, cada una verificada en este repositorio, no supuestas:

1. **10 positivos totales.** Cualquier métrica de detección (recall,
   precision, F1) calculada sobre 10 eventos tiene una incertidumbre enorme
   — el IC de Wilson del recall abarca 45 puntos porcentuales. Estimar un
   recall de 0.80 con un margen razonable (±10 pp) requeriría observar **62
   fallos**, unos **1116 ciclos de máquina** a la prevalencia actual — más de
   seis veces los datos disponibles hoy (`src/predictive_maintenance/power.py`,
   CLAUDE.md §10.3).
2. **Los datos son sintéticos**, con probabilidad alta según tres evidencias
   independientes (arriba). Un modelo entrenado sobre un generador no
   necesariamente reproduce el comportamiento de sensores reales, con su
   ruido, sus fallos de instrumentación y sus correlaciones físicas.
3. **Las cinco variables son mutuamente independientes** (|r| < 0.10 entre
   cualquier par). En maquinaria física esto es irreal: carga, temperatura,
   viscosidad del lubricante y vibración están acopladas por la física del
   sistema. Un modelo entrenado sobre variables artificialmente
   independientes no ha visto nunca la estructura de correlación que vería en
   una máquina real.
4. **No hay eje temporal.** `lab180` no tiene columna de fecha ni de orden de
   observación: cada fila es una instantánea sin memoria del historial de la
   máquina. La única validación con sentido en mantenimiento predictivo es
   la **validación temporal** — entrenar con el pasado, validar con el
   futuro, porque el objetivo real es predecir un fallo ANTES de que ocurra,
   no interpolar entre observaciones ya mezcladas al azar. Sin eje temporal,
   esa validación es, literalmente, imposible de hacer sobre este dataset.

Estas cuatro razones son la explicación de por qué `lab180` es, en este
repositorio, un **contraejemplo** deliberado — no el resultado principal (ver
CLAUDE.md §13). El segundo dataset del repositorio, `ai4i2020`, existe
precisamente para mostrar el mismo pipeline sobre datos con más orden de
magnitud de positivos y variables físicamente acopladas.

## Referencias

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAT* '19.
- Vabalas, A. et al. (2019). *Machine learning algorithm validation with a
  limited sample size.* PLOS ONE 14(11): e0224365.
- Saito, T., Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More
  Informative than the ROC Plot When Evaluating Binary Classifiers on
  Imbalanced Datasets.* PLOS ONE 10(3): e0118432.
- Varoquaux, G. (2018). *Cross-validation failure: small sample sizes lead to
  large error bars.* NeuroImage 180: 68-77.
