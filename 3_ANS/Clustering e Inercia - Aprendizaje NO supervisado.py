import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# PREPARACIÓN GENERAL DE DATOS (Aplica para ambas tareas)
# ==========================================
# 1. Leer el archivo
df = pd.read_csv('cereal.csv')

# 2. Preparar los datos eliminando columnas categóricas
X = df.drop(['Cereal Name', 'Manufacturer'], axis=1)

# Estandarizamos los datos numéricos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================================================================
# ================================= TAREA 1 ====================================
# ==============================================================================
print("\n" + "="*40)
print(" INICIANDO TAREA 1: MODELO DE 2 CLUSTERS")
print("="*40)

# --- EDA (Análisis Exploratorio de Datos) ---
print("\n--- Primeras filas del dataset ---")
print(df.head())

# Visualización: Matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('TAREA 1: Matriz de Correlación de Nutrientes')
plt.tight_layout()
plt.show()

# --- Modelo K-Means (2 Clusters) ---
kmeans_t1 = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster_T1'] = kmeans_t1.fit_predict(X_scaled)

# --- Evaluación e Interpretación Tarea 1 ---
cluster_means_t1 = df.groupby('Cluster_T1')[X.columns].mean()
print("\n--- Promedios de Nutrientes por Cluster (TAREA 1) ---")
print(cluster_means_t1)

# Visualizar los clusters de la Tarea 1
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Sugars', y='Calories', hue='Cluster_T1', palette='Set1', s=100)
plt.title('TAREA 1: Segmentación K-Means (2 Clusters) - Azúcares vs Calorías')
plt.xlabel('Azúcares (g)')
plt.ylabel('Calorías')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()


# ==============================================================================
# ================================= TAREA 2 ====================================
# ==============================================================================
print("\n" + "="*40)
print(" INICIANDO TAREA 2: MÉTODO DEL CODO Y MAPA DE CALOR")
print("="*40)

# --- 1. Bucle para ajustar modelos entre 2 y 15 clusters ---
inercia = []
rango_clusters = range(2, 16)

for k in rango_clusters:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inercia.append(kmeans_temp.inertia_)

# --- 2 y 3. Gráfico del Codo y su identificación ---
plt.figure(figsize=(8, 5))
plt.plot(rango_clusters, inercia, marker='o', linestyle='--', color='b')
plt.title('TAREA 2: Método del Codo para K-Means (2 a 15 Clusters)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia (Suma de distancias al cuadrado)')
plt.xticks(rango_clusters)
plt.grid(True)
plt.show()

# Visualmente, la inercia dejao de caer een k=4. 
k_optimo = 4

# --- 4. Ajustar modelo al número específico del codo (4 clusters) ---
kmeans_t2 = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df['Cluster_T2'] = kmeans_t2.fit_predict(X_scaled)

# --- 5. Visualizar los centros utilizando un mapa de calor ---

centros_escalados = kmeans_t2.cluster_centers_
df_centros_escalados = pd.DataFrame(centros_escalados, columns=X.columns)
# Nombres propuestos para los clusters (se asignan como índice)
nombres_clusters = [
    'Cluster 0: Ligeros/Dietéticos', 
    'Cluster 1: Súper Fortificados', 
    'Cluster 2: Infantiles/Dulces', 
    'Cluster 3: Densos/Energéticos'
]
df_centros_escalados.index = nombres_clusters

plt.figure(figsize=(10, 6))
sns.heatmap(df_centros_escalados, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title('TAREA 2: Mapa de Calor de Centros (Desviaciones vs Promedio Global)')
plt.ylabel('Segmentos Propuestos')
plt.tight_layout()
plt.show()

# Imprimir los valores reales para tener el dato exacto en el reporte
centros_originales = scaler.inverse_transform(centros_escalados)
df_centros_reales = pd.DataFrame(centros_originales, columns=X.columns, index=nombres_clusters)
print("\n--- Valores Nutricionales Promedio Reales por Cluster (TAREA 2) ---")
print(df_centros_reales.round(2))