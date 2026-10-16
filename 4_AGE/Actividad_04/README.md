# Actividad 4: Algoritmos genéticos para el VRPTW

Este trabajo presenta tablas y visualizaciones relacionadas con el **Vehicle Routing Problem with Time Windows (VRPTW)**, usando como referencia el benchmark de Solomon, instancia C101, citado como procedente de Kaggle.

## Requisitos

- Python 3.10 o posterior.
- `numpy` para generar las series numéricas de convergencia.
- `matplotlib` para guardar las figuras.

La escritura de CSV utiliza el módulo `csv` incluido en Python. No se requiere `python-docx` ni un archivo de datos externo para ejecutar la versión actual.

## Instalación y ejecución

Desde esta carpeta, instala las dependencias y ejecuta el script:

```bash
python -m pip install -r requirements.txt
python Actividad04.py
```

En Windows también se puede ejecutar desde la terminal de VS Code. Los archivos se guardan en la misma carpeta que `Actividad04.py`, independientemente del directorio actual de la terminal.

## Archivos generados

### Figuras PNG

- `figura_1_restricciones_vrptw.png`: esquema de rutas, cargas de vehículos frente a capacidad y ventanas de tiempo.
- `fig_convergencia_ga_vrptw.png`: curvas de convergencia mostradas por operador de cruzamiento y mecanismo de selección durante 120 generaciones.
- `fig_rutas_optimas_vrptw.png`: distribución espacial de las tres rutas.
- `fig_comparativa_barras_3x3.png`: comparación de fitness para las nueve combinaciones de selección y cruzamiento.

### Tablas CSV

- `convergencia_ga_vrptw.csv`: una fila por generación, con las seis series de convergencia.
- `resultados_factoriales_ga_vrptw.csv`: nueve combinaciones de selección y cruzamiento con su fitness registrado.
- `rutas_optimas_vrptw.csv`: secuencia de paradas y atributos de cada nodo para cada vehículo.

Los CSV se escriben con codificación UTF-8 con BOM para facilitar su apertura en Excel.

## Alcance de la versión actual

Las rutas, puntuaciones y parámetros de las curvas están definidos directamente en `Actividad04.py`. Las curvas se calculan con expresiones exponenciales parametrizadas para las figuras; no son el registro de una ejecución en vivo del algoritmo genético. El script tampoco descarga automáticamente datos desde Kaggle. Por ello, las salidas representan los datos y resultados configurados en el archivo, no una optimización recalculada desde el benchmark al ejecutar el programa.

Para usar otros resultados, actualiza los datos de clientes, rutas y puntuaciones definidos en el script. Para reproducir experimentos de optimización, será necesario incorporar la carga del dataset y la implementación del algoritmo genético.