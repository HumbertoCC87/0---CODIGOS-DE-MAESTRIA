# Green-VRPTW Multiobjetivo

Aplicación industrial local con NSGA-II desde cero para optimizar costo, emisiones y nivel de servicio.

## Instalación y ejecución local
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

En Windows también puede usarse el intérprete del entorno Conda:
```bash
"/c/Users/no-lo/.conda/envs/pln_311/python.exe" -m pip install -r requirements.txt
"/c/Users/no-lo/.conda/envs/pln_311/python.exe" -m streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`. El tema local está definido en
`.streamlit/config.toml`; el CSS compartido y las tarjetas ejecutivas se encuentran
en `ui/styles.py`.

## Formulación
Se minimiza $(f_1,f_2,f_3)$:
- Costo operativo: costo fijo por vehículo, combustible por arco, desgaste y espera.
- Emisiones: litros × factor $CO_2$ × factor topográfico.
- SLA: $3\max(0,t-b)+0.35\max(0,a-t)$.

El cromosoma es `[perm, alpha_q, max_late_split]`; la decodificación divide el Giant Tour por capacidad estricta y tolerancia temporal.

Los escenarios base, congestión y flota verde permiten comparar frentes bajo parámetros configurables. El despacho inteligente programa la salida del depósito según la primera ventana de servicio; las rutas, distancias, emisiones y retornos se contabilizan como ciclos cerrados.

El CSV Solomon cargado se procesa en memoria. Al exportar evidencias, los CSV y las figuras Pareto, convergencia de hipervolumen y Gantt a 300 DPI se guardan localmente en `outputs/`. El historial de convergencia conserva límites de normalización fijos durante cada corrida para comparar generaciones.
