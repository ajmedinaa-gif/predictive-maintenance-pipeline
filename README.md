# predictive-maintenance-pipeline

Clasificación de riesgo de fallo en maquinaria industrial a partir de sensores.

**Tesis del repositorio:** un pipeline de mantenimiento predictivo evaluado
honestamente sobre dos datasets de tamaño radicalmente distinto, demostrando
que la métrica ingenua —la accuracy— miente en ambos. El repo no vende un
modelo. Vende criterio.

Todos los números de este README se leen de `reports/eda_lab180.json`, generado
por `pdm-cli eda --dataset lab180`. Ningún número está escrito a mano.

## Dataset: `lab180`

⚠️ **Datos SIMULADOS de laboratorio**, no mediciones de campo. `data/raw/lab180/Lab1_engineering_failures.csv`
— 180 filas, 6 columnas (5 sensores + variable objetivo `failure`).

### Tabla descriptiva

| atributo | n | nulos | %nulos | min | max | media | mediana | desv. típica |
|---|---|---|---|---|---|---|---|---|
| temperature_c | 176 | 4 | 2.2222 | 47.00 | 89.80 | 67.8102 | 67.950 | 7.6651 |
| vibration_mm_s | 176 | 4 | 2.2222 | **-0.34** | 9.59 | 4.2438 | 4.295 | 1.3349 |
| pressure_bar | 176 | 4 | 2.2222 | 4.27 | 10.19 | 6.7630 | 6.750 | 1.1330 |
| hours_since_maintenance | 180 | 0 | 0.0000 | 20.00 | 896.00 | 473.1889 | 440.000 | 262.3427 |
| load_percent | 180 | 0 | 0.0000 | 31.50 | 100.00 | 71.0900 | 71.400 | 14.4342 |

### La anomalía: vibración negativa

La fila **82** registra `vibration_mm_s = -0.34`. Una amplitud RMS de vibración
no puede ser negativa: es un imposible físico, no un valor extremo. Esa fila va
a **cuarentena**, no se imputa — imputar un sensor que miente sería fabricar un
dato, no rescatar uno perdido.

### Balance de clases

- `failure`: `no` = 170, `yes` = 10.
- **Prevalencia = 5.5556 %.**
- **Accuracy del clasificador trivial mayoritario = 94.4444 %** — el número que
  debe ir delante de cualquier accuracy que este repositorio reporte más
  adelante. Un modelo con menos de 94.4444 % de accuracy es peor que no hacer
  nada.

### Asociación univariante (AUC de Mann-Whitney)

| atributo | AUC | p-valor | dirección |
|---|---|---|---|
| vibration_mm_s | 0.8503 | 0.0002 | directa (la más fuerte) |
| temperature_c | 0.7259 | 0.0167 | directa |
| hours_since_maintenance | 0.7218 | 0.0187 | directa |
| load_percent | 0.6682 | 0.0746 | **no significativa** |
| pressure_bar | 0.2515 | 0.0085 | **inversa** |

`pressure_bar` con AUC < 0.5 significa que los fallos ocurren a presión
**baja**, no alta: compatible con fuga o pérdida de lubricación/fluido
hidráulico.

### Correlaciones

Todas las correlaciones de Pearson entre atributos tienen **|r| < 0.10**
(máximo 0.0972, entre `vibration_mm_s` y `load_percent`). Las cinco variables
son mutuamente independientes — una señal de que el dataset es simulado: en
maquinaria física, carga → temperatura → viscosidad del lubricante → vibración
forman una cadena causal acoplada.

### Figuras

![Barras de prevalencia con la línea de accuracy trivial](reports/figures/lab180/prevalencia.png)
*La barra `yes` es 5.5556 % del total; la línea discontinua marca 94.4444 %,
el accuracy que logra no hacer nada.*

![Mapa de correlación entre sensores, escala fija en [-1, 1]](reports/figures/lab180/correlacion.png)
*Plano por construcción: la celda más oscura fuera de la diagonal es 0.097 —
las cinco variables son mutuamente independientes.*

![Histogramas por sensor, clase superpuesta](reports/figures/lab180/histogramas_por_clase.png)
*`vibration_mm_s` es donde la clase `yes` (naranja) se separa más de la `no`
(azul); en `pressure_bar` el naranja cae hacia valores más bajos.*

![Boxplots por sensor con los puntos individuales](reports/figures/lab180/boxplots_por_clase.png)
*Cada punto es una observación: con 10 positivos, la caja `yes` no promedia
más precisión de la que hay puntos para sostenerla — y se ve el -0.34 de
`vibration_mm_s` por debajo de cero.*

## Contrato de datos

Ningún dato entra al pipeline sin pasar antes por un contrato ejecutable
(`src/predictive_maintenance/schema.py`, un `pandera.DataFrameModel`): rangos
físicamente posibles por sensor y las dos categorías válidas de `failure`. La
fila que lo incumple no se corrige ni se descarta en silencio — va a
[`data/quarantine/`](data/quarantine/), con el motivo exacto por el que se
rechazó, para que alguien la revise.

`pdm-cli validate --dataset lab180` ejecuta el contrato y vuelca el resultado
a `reports/validation_lab180.json`:

- **180 filas leídas → 179 válidas, 1 en cuarentena.**
- Motivo: `vibration_mm_s=-0.34 incumple greater_than_or_equal_to(0.0)`
  (fila 82).

### Por qué cuarentena y no imputación

`vibration_mm_s = -0.34` es un imposible físico: una amplitud RMS de
vibración no puede ser negativa, así que no es un valor extremo que suavizar,
es un sensor que miente. Imputar ese valor —con la media, la mediana o un
modelo— habría fabricado un dato a partir de una lectura que sabemos que está
mal, y ese dato fabricado habría entrado a entrenar un modelo como si fuera
una medición real.

Si no la hubiéramos visto, la fila habría entrado tal cual al split de
entrenamiento/test: un valor negativo en una columna que después se escala y
se pasa a una regresión logística o un árbol no lanza ningún error, así que el
error no se detecta comparando un `Pipeline` contra un contrato — se detecta
prácticamente porque alguien mira el mínimo de una tabla descriptiva a mano
(CLAUDE.md §6.1). Con un contrato ejecutable, esa comparación deja de depender
de que alguien se acuerde de mirar.

Ningún otro atributo de `lab180` viola su rango físico: la única fila en
cuarentena es la 82, y **avisos de calidad: ninguno** — la prevalencia
(5.5556 %), los nulos por columna (máximo 2.22 %) y los duplicados (0) están
todos dentro de los umbrales de aviso de `config/default.yaml`.

## ¿Son reales estos datos?

**No. El auditor de plausibilidad (`src/predictive_maintenance/plausibility.py`)
dictamina: `probablemente sintético`.**

El veredicto combina dos firmas fuertes, deterministas, escritas a
`reports/plausibility_lab180.json`:

1. **Independencia mutua total.** En una máquina real, carga → temperatura →
   viscosidad del lubricante → vibración forman una cadena causal acoplada:
   se espera |r| ≥ 0.15 entre algún par de sensores. La correlación máxima
   observada entre `vibration_mm_s` y `load_percent` es **0.0972**, muy por
   debajo del umbral. Ningún par de atributos está acoplado.
2. **Patrón de nulos demasiado regular.** 12 nulos, repartidos en **exactamente
   4, 4 y 4** entre `temperature_c`, `vibration_mm_s` y `pressure_bar`, que
   **jamás se solapan** en la misma fila (ninguna fila tiene 2 nulos o más). Un
   sensor real falla por causas independientes entre sí; que el recuento
   coincida exacto entre tres columnas distintas es la firma de un generador,
   no de un fallo de instrumentación.

Como evidencia adicional, de apoyo —no decisiva por sí sola—, el auditor
también comprueba la distribución del último dígito decimal de cada sensor
(`digit_distribution`): en los cinco atributos es compatible con muestreo
uniforme (`chi²`, p > 0.05 en todos), consistente con datos generados por
`numpy.random.uniform` y redondeo, no con la cuantización de un sensor físico
de bajo coste.

Esto no descalifica `lab180` como ejercicio: lo define. Sirve de
**contraejemplo** en este repositorio — qué NO se puede concluir con pocos
datos simulados — y nunca se presenta como resultado principal (CLAUDE.md
§13).

## En construcción

Esto cubre las Fases 1 y 2 (andamiaje + EDA + contrato de datos + auditor de
plausibilidad). Todavía **no existe ningún modelo entrenado ni resultado de
validación**: eso llega en fases posteriores, con `RepeatedStratifiedKFold`
sobre `lab180`, calibración de probabilidades, umbral por coste y el segundo
dataset (`ai4i2020`). Nada de lo que sigue está escrito todavía a propósito —
no hay número que reportar sin haberlo medido.

## Desarrollo

```bash
uv sync
make lint
make test
make eda
make validate
```

Ver `CLAUDE.md` para las reglas del proyecto.

## Licencia

MIT. Ver [`LICENSE`](LICENSE).
