import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ==========================================
# OBJaETIVO 1: Abrir el archivo, eliminar user_id y observar rangos
# ==========================================
print("--- Objetivo 1 ---")
# Cargar el dataset
df = pd.read_csv('tripadvisor_reviews.csv')

# Eliminar la columna 'user_id'
df = df.drop('user_id', axis=1)

# Observar el rango de cada valoración (mínimo y máximo)
print("Rangos de cada valoración:")
print(df.describe().loc[['min', 'max']])


# ==========================================
# OBJETIVO 2: Visualizar los datos mediante un gráfico de pares
# ==========================================
print("\n--- Objetivo 2 ---")
# Usamos Seaborn para crear un pairplot con los datos originales
sns.pairplot(df)
plt.suptitle("Gráfico de pares - Datos Originales", y=1.02)
# plt.show() # Descomentar para ver el gráfico en tu entorno local
plt.savefig('pairplot_obj2.png')
plt.close()
print("Se han generado los gráficos pairplot_obj2.png y pairplot_obj4.png.")


# ==========================================
# OBJETIVO 3: Aplicar Isolation Forest con contaminación = 0.01
# ==========================================
print("\n--- Objetivo 3 ---")
# Inicializamos el modelo indicando que el 1% de los datos son posibles anomalías
iso_forest_01 = IsolationForest(contamination=0.01, random_state=42)

# Entrenamos el modelo y predecimos. 
# Devuelve 1 para datos normales y -1 para anomalías.
df['anomaly_01'] = iso_forest_01.fit_predict(df)
print("Anomalías detectadas con contaminación 0.01:")
print(df['anomaly_01'].value_counts())


# ==========================================
# OBJETIVO 4: Visualizar las anomalías en el gráfico de pares
# ==========================================
print("\n--- Objetivo 4 ---")
# Definimos las columnas a graficar para no incluir la columna 'anomaly_01' como eje
features = ['avg_museum_rating', 'avg_park_rating', 'avg_restaurant_rating', 'avg_nightlife_rating']

# Visualizamos coloreando según sean anomalías (-1, rojo) o normales (1, azul)
sns.pairplot(df, vars=features, hue='anomaly_01', palette={1: 'blue', -1: 'red'})
plt.suptitle("Anomalías - Contaminación 0.01", y=1.02)
# plt.show()
plt.savefig('pairplot_obj4.png')
plt.close()
print("Se han generado los gráficos pairplot_obj2.png y pairplot_obj4.png.")


# ==========================================
# OBJETIVO 5: Fíjate dónde hay anomalías en el gráfico de pares
# ==========================================
print("--- Objetivo 5 ---")
print("Interpretación en el docuemnto de texto")
# Interpretacion:
# Al observar el gráfico del Objetivo 4, los puntos rojos (anomalías) tienden a 
# localizarse en los extremos o bordes de las distribuciones. Es decir, usuarios 
# que han dado calificaciones inusualmente altas o inusualmente bajas en varias
# categorías simultáneamente (puntos alejados de la "nube" central azul).


# ==========================================
# OBJETIVO 6: Modificar contaminación a 0.005, visualizar y observar diferencias
# ==========================================
print("\n--- Objetivo 6 ---")
# Cambiamos la contaminación a la mitad (0.5%)
iso_forest_005 = IsolationForest(contamination=0.005, random_state=42)

# Predecimos usando solo las columnas de valoraciones
df['anomaly_005'] = iso_forest_005.fit_predict(df[features])

print("Anomalías detectadas con contaminación 0.005:")
print(df['anomaly_005'].value_counts())

# Visualizamos la nueva clasificación
sns.pairplot(df, vars=features, hue='anomaly_005', palette={1: 'blue', -1: 'red'})
plt.suptitle("Anomalías - Contaminación 0.005", y=1.02)
# plt.show()
plt.savefig('pairplot_obj6.png')
plt.close()

# Observación final:
# Al reducir el nivel de contaminación a 0.005, el algoritmo es mucho más estricto. 
# Ahora solo detecta a los turistas con las valoraciones más "extremas" absolutas,
# reduciendo la cantidad de puntos rojos en el gráfico.