# Comentarios y explicación — `02_end_to_end_machine_learning_project.ipynb`

Este documento resume el notebook **02_end_to_end_machine_learning_project** de forma práctica, con foco en el “por qué” de cada bloque.

## 1) Objetivo del proyecto

El notebook plantea un caso realista: predecir `median_house_value` (precio medio de vivienda) en distritos de California a partir de variables socioeconómicas y geográficas.

Idea central: no es solo “entrenar un modelo”, sino recorrer un flujo **end-to-end**:

1. Obtención de datos.
2. Análisis exploratorio.
3. Separación train/test correcta.
4. Preprocesamiento reproducible.
5. Entrenamiento y comparación de modelos.
6. Ajuste de hiperparámetros.
7. Evaluación final con intervalo de confianza.
8. Persistencia del modelo para producción.

## 2) Carga y primera inspección

Se descarga/carga el dataset de housing y se inspecciona con:

- `head()` para ver columnas.
- `info()` para tipos y nulos.
- `value_counts()` en categorías.
- `describe()` para estadísticos básicos.

**Qué debes mirar aquí:**

- Columnas con faltantes (ej. `total_bedrooms`).
- Variables categóricas (ej. `ocean_proximity`).
- Escalas muy distintas entre atributos (te obliga a escalar más adelante).

## 3) Separación de test set (punto crítico)

El notebook enfatiza un error típico: hacer EDA/feature engineering con todo el dataset antes de separar test.

Se muestran varias estrategias:

- Partición aleatoria simple.
- Partición estable por identificador (`crc32`) para evitar que el test “cambie” cuando se actualizan datos.
- **Muestreo estratificado por `income_cat`** para mantener distribución de ingresos en train/test.

**Mensaje importante:** usar estratificación cuando una variable influye mucho en la target y no quieres sesgos de muestreo.

## 4) EDA (análisis exploratorio)

Se exploran relaciones con:

- Scatter geográfico (`longitude`, `latitude`).
- Color/tamaño para añadir información de valor y población.
- Matriz de correlación.
- `scatter_matrix` entre variables prometedoras.

**Lecturas clave del EDA:**

- `median_income` correlaciona fuertemente con el precio.
- La localización impacta mucho (efectos costa/interior).
- Hay señales de no linealidad y posibles topes/censura en el target.

## 5) Feature engineering básico

Se prueban atributos derivados como:

- `rooms_per_house`.
- `bedrooms_ratio`.
- `people_per_house`.

Esto enseña que ratios suelen capturar mejor patrones que valores absolutos.

## 6) Preparación de datos

### 6.1 Limpieza de nulos

Se comparan opciones y se usa `SimpleImputer(strategy="median")` para numéricas.

### 6.2 Categóricas

Se codifica `ocean_proximity` con `OneHotEncoder`.

### 6.3 Outliers

Se muestra detección con `IsolationForest` (y se comenta cuándo eliminar o no).

### 6.4 Pipelines

Se encapsula el preprocesado para:

- Evitar fugas de información.
- Repetibilidad en validación y producción.
- Aplicar exactamente la misma transformación a train/test.

## 7) Modelado

Se entrenan y comparan modelos de regresión (p. ej. baseline lineal y `RandomForestRegressor`).

Evaluación con:

- RMSE en train.
- Validación cruzada (`cross_val_score`) para estimar generalización real.

**Aprendizaje clave:** no confiar en un único split ni en métrica de entrenamiento.

## 8) Ajuste de hiperparámetros

Se usa:

- `GridSearchCV` (rejilla finita).
- `RandomizedSearchCV` (más eficiente cuando hay muchos parámetros/rangos amplios).

Tras elegir mejor estimador, se analizan:

- Mejores hiperparámetros.
- Importancias de atributos.

## 9) Evaluación final en test

Con el mejor pipeline/modelo:

1. Se evalúa en `strat_test_set` una sola vez.
2. Se calcula intervalo de confianza del RMSE (bootstrap), para comunicar incertidumbre.

Esto aproxima prácticas de reporting real (no solo “un número”).

## 10) Persistencia y producción

Se guarda con `joblib` y se muestra cómo cargar y predecir.

Idea importante: guardar **el pipeline completo**, no solo el modelo, para conservar transformaciones idénticas en inferencia.

## 11) Ejercicios del notebook

La parte final profundiza buenas prácticas:

- Probar `SVR` con grid/random search.
- Selección de características con `SelectFromModel`.
- Transformador personalizado compatible con API Scikit-Learn.
- Exploración automática de variantes del pipeline.

Aquí el mensaje es de ingeniería ML: diseño modular + experimentación sistemática.

---

## Resumen conceptual rápido

Si te quedas con 5 ideas del notebook, que sean estas:

1. **Separa test temprano** para evitar leakage.
2. **Usa estratificación** cuando la distribución importa.
3. **Construye pipelines** para reproducibilidad y consistencia.
4. **Compara modelos con validación cruzada** y luego ajusta hiperparámetros.
5. **Evalúa una sola vez en test** y comunica incertidumbre (IC del RMSE).

## Errores comunes que este notebook ayuda a evitar

- Tocar el test durante el desarrollo.
- Imputar/escalar fuera de pipeline.
- Concluir con una sola métrica en train.
- No versionar/guardar preprocesamiento.
- Hacer tuning “a mano” sin procedimiento reproducible.

## Cómo estudiarlo mejor (recomendación)

1. Ejecuta el notebook una vez completo para mapa general.
2. Repite solo bloques de split + pipeline + CV hasta entenderlos al detalle.
3. Cambia 1–2 decisiones (p. ej. modelo, estrategia de imputación) y compara RMSE/CV.
4. Documenta cada experimento (semilla, parámetros, resultado).

Así pasas de “seguir pasos” a “pensar como ML engineer”.
