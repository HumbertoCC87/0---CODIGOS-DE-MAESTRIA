# Proyecto Integrador PLN - Clasificación de géneros cinematográficos

Este proyecto desarrolla un flujo completo de Procesamiento de Lenguaje Natural (PLN) para analizar reseñas de películas y clasificar su género cinematográfico utilizando técnicas de representación vectorial, análisis de sentimiento, resúmenes automáticos y sistema de recuperación semántica.

## Objetivo

Construir un pipeline end-to-end para:

- limpiar y normalizar reseñas textuales,
- analizar sentimiento por reseña y por género,
- generar resúmenes de textos,
- crear representaciones numéricas (BOW, TF, TF-IDF),
- evaluar modelos de clasificación,
- comparar métricas de recuperación semántica,
- generar gráficas descriptivas y analíticas.

## Archivos principales

- `proyecto_integrador_3.py`: script principal del proyecto.
- `film_reviews_result.csv`: dataset de reseñas de películas.
- `salidas/`: carpeta con resultados y gráficos generados.

## Flujo del proyecto

1. Carga del dataset.
2. Preprocesamiento del texto.
3. Análisis de ambigüedad.
4. Análisis de sentimiento.
5. Generación de resúmenes textualizados.
6. Sistema de consulta basado en similitud semántica.
7. Generación de representaciones numéricas.
8. Entrenamiento y comparación de modelos:
   - Naive Bayes
   - SVM Lineal
   - Regresión Logística
9. Evaluación de métricas:
   - Accuracy
   - Precision@K
   - Recall@K
   - MRR
10. Exportación de gráficas y archivos CSV en la carpeta `salidas`.

## Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Ejecución

Desde la carpeta del proyecto:

```bash
python proyecto_integrador_3.py
```

## Dependencias principales

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- textblob

## Salidas esperadas

La carpeta `salidas/` incluye:

- archivos CSV de datos procesados,
- matrices de representación vectorial,
- modelos comparativos,
- métricas de recuperación semántica,
- gráficas de análisis y comparación.

## Notas

El proyecto utiliza recursos de NLTK como `punkt` y `stopwords`. Si no están presentes, el script los descargará automáticamente durante la ejecución.

## Autores

- Humberto Cruz Cruz
- Eric Pérez Sepulveda
