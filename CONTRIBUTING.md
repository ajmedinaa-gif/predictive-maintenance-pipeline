# Contribuir a `predictive-maintenance-pipeline`

Este repositorio se construyó en 5 fases, cada una con su propio prompt en
[`FASES.md`](FASES.md) y su propio CHECKPOINT revisado a mano antes de seguir
a la siguiente. Este documento explica cómo se construyó y cómo seguir
trabajando sobre él con las mismas reglas.

## Cómo se construyó

| Fase | Contenido |
|---|---|
| 1 | Andamiaje del proyecto: `uv`, estructura de `src/`, EDA reproducible sobre `lab180`, CLI (`typer`) con los primeros comandos. |
| 2 | Contrato de datos ejecutable (`pandera`): rangos físicos por sensor, cuarentena de filas inválidas, auditor de plausibilidad (`plausibility.py`) que dictamina si un dataset es simulado. |
| 3 | Modelado con validación honesta: `RepeatedStratifiedKFold`, zoo de modelos con los `DummyClassifier` siempre presentes, CV anidada para selección de hiperparámetros, matrices de confusión out-of-fold. |
| 4 | Calibración de probabilidades (Platt/isotónica), umbral de decisión por coste, explicabilidad (SHAP, estabilidad de la raíz del árbol), límites estadísticos (IC de Wilson, presupuesto de muestra, curva de aprendizaje). |
| 5 | Segundo dataset (`ai4i2020`, real, físicamente acoplado) vía el protocolo `DatasetAdapter`; empaquetado final: Docker, CI, dashboard de Streamlit, informe HTML autocontenido. |

**[`CLAUDE.md`](CLAUDE.md) es la fuente de verdad del proyecto** -- convenciones,
reglas duras (qué nunca se hace) y todos los números verificados con su
procedencia. Si algo en el código contradice `CLAUDE.md`, hay un error en uno
de los dos y hay que pararse a averiguar cuál, no ajustar el que sea más fácil
de cambiar.

## Regla de oro: ningún número se escribe a mano

Todo número que aparece en el README, en `CLAUDE.md` o en el dashboard sale de
`reports/*.json`, generado por una ejecución real de `pdm-cli`. Si vas a
cambiar una cifra en la documentación, la pregunta correcta no es "¿qué número
pongo?" sino "¿qué comando genera ese número?" -- ejecútalo y copia lo que
salga. Hay un test (`tests/test_report.py`) que compara el HTML generado
contra el JSON del que sale.

## Entorno de desarrollo

Ver CLAUDE.md §14 para el detalle completo (macOS Intel, sin Homebrew, sin
Docker local). Resumen:

```bash
uv sync                    # instala el entorno; nunca `pip install` a secas
uv run python -c "import sys; print(sys.executable)"  # debe apuntar a .venv/bin/python
make lint                  # ruff check + ruff format --check
make test                  # pytest con cobertura (objetivo 80 %, CLAUDE.md §5)
```

Nunca `python`, `pip` ni `pytest` a secas -- siempre `uv run ...` (CLAUDE.md
§14.1). Nunca `brew install` nada (CLAUDE.md §14.2). Esta máquina no tiene
Docker: `docker build`/`docker compose up` los prueba el job `docker` de CI,
no el desarrollo local (CLAUDE.md §14.7).

## Reglas duras (resumen; el detalle completo está en CLAUDE.md §2)

1. La accuracy nunca se reporta sola: siempre con la fila del `DummyClassifier` al lado.
2. Métrica primaria: PR-AUC. El ROC-AUC se reporta pero no decide.
3. `lab180` (10 positivos) siempre con `RepeatedStratifiedKFold(5, n_repeats=10)`, nunca un único split.
4. Nunca SMOTE ni oversampling sintético.
5. `class_weight="balanced"` y el umbral por coste nunca se combinan.
6. Imputación, escalado y calibración van siempre dentro del `sklearn.Pipeline`, nunca antes del split.
7. El umbral por coste se optimiza dentro de cada fold de entrenamiento, nunca sobre el fold de test.
8. Ningún número del README o el dashboard se escribe a mano.
9. Los datos que no pasan el contrato van a `data/quarantine/`, nunca se borran ni se imputan en silencio.
10. Semilla global única, en `config/default.yaml`.
11. Sin llamadas de red en tiempo de ejecución del pipeline: la descarga de un dataset es un paso explícito y separado (`pdm-cli download`).
13. Un commit por unidad conceptual. Nunca `git push --force` sobre `main`.

## Flujo de trabajo para un cambio

1. Lee la sección relevante de `CLAUDE.md` antes de tocar código: probablemente
   ya documenta por qué algo se hizo de una manera concreta.
2. Escribe o actualiza el test antes (o junto con) el cambio -- cobertura
   objetivo 80 %, `pytest --cov-fail-under=80` es parte de `make test` y de CI.
3. `make lint && make test` en verde antes de commitear.
4. Si el cambio afecta a un número visible (README, dashboard, `CLAUDE.md`),
   ejecuta el comando de `pdm-cli` real que lo produce y usa esa salida --
   nunca lo escribas de memoria.
5. Commits en [Conventional Commits](https://www.conventionalcommits.org/),
   en inglés, un commit por unidad conceptual (CLAUDE.md §4):

   ```
   feat(datasets): add DatasetAdapter protocol and ai4i2020 support
   fix(compare): base the recall/CI row on logistic_plain+Platt, not PR-AUC
   docs(readme): add the two-dataset comparison table
   ```

6. Si el cambio añade un dataset, un modelo o una figura nueva: revisa si
   `CLAUDE.md` necesita actualizarse en la misma unidad de trabajo -- un
   número nuevo sin su procedencia documentada es, para este proyecto, un
   número a medias.

## Extender a un tercer dataset

El punto de extensión es `src/predictive_maintenance/datasets.py`: implementa
el protocolo `DatasetAdapter` (`name`, `download()`, `schema`, `target_column`,
`positive_label`, `feature_columns`, `engineer_features(df)`), regístralo en
`ADAPTERS`, añade su contrato en `schema.py` y su entrada de
`cross_validation:` en `config/default.yaml`. Ni `pipeline.py`, ni
`evaluate.py`, ni `calibration.py` deberían necesitar ningún cambio -- si lo
necesitan, probablemente `engineer_features` no está devolviendo una matriz
completamente numérica.

## Reportar un problema

Este es un repositorio de portafolio personal sin un proceso formal de
issues/PRs externos, pero si encuentras un número que no cuadra con su
`reports/*.json`, o una regla dura de `CLAUDE.md` que el código no respeta en
la práctica, es bienvenido abrir un issue en GitHub describiendo exactamente
el comando ejecutado y la discrepancia observada.
