import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# --- Nuevas librerías para la Tarea 3 ---
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

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
print("" + "="*40)
print(" INICIANDO TAREA 1: MODELO DE 2 CLUSTERS")
print("="*40)

# --- EDA (Análisis Exploratorio de Datos) ---
print("--- Primeras filas del dataset ---")
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
print("--- Promedios de Nutrientes por Cluster (TAREA 1) ---")
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
print("" + "="*40)
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

# Visualmente, la inercia deja de caer en k=4.
k_optimo = 4

# --- 4. Ajustar modelo al número específico del codo (4 clusters) ---
kmeans_t2 = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df['Cluster_T2'] = kmeans_t2.fit_predict(X_scaled)

# --- 5. Visualizar los centros utilizando un mapa de calor ---
centros_escalados = kmeans_t2.cluster_centers_
df_centros_escalados = pd.DataFrame(centros_escalados, columns=X.columns)

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
print("--- Valores Nutricionales Promedio Reales por Cluster (TAREA 2) ---")
print(df_centros_reales.round(2))


# ==============================================================================
# ================================= TAREA 3 ====================================
# ==============================================================================
print("" + "="*40)
print(" INICIANDO TAREA 3: DENDROGRAMAS Y MAPA DE CLUSTERS (JERÁRQUICO)")
print("="*40)

# --- OBJETIVO 1 y 2: Dendrograma con datos originales (5 campos) ---
print("Generando Dendrograma con datos originales...")
plt.figure(figsize=(12, 7))
plt.title('TAREA 3: Dendrograma (Datos Originales - 5 Campos)')
# Método Ward para minimizar la varianza dentro de los clusters
Z_original = sch.linkage(X, method='ward')
# El umbral (color_threshold) se ajusta visualmente (ej. 50) para detectar los cortes
sch.dendrogram(Z_original, color_threshold=50, leaf_font_size=8)
plt.xlabel('Índice del Cereal')
plt.ylabel('Distancia Euclidiana')
plt.axhline(y=50, color='r', linestyle='--') # Línea visual de corte
plt.show()


# --- OBJETIVO 3 y 4: Dendrograma con datos estandarizados (sin 'Grasa') ---
print("Generando Dendrograma con datos estandarizados (sin 'Fat')...")
# Excluimos 'Fat'
X_sin_grasa = X.drop('Fat', axis=1)

# Estandarizamos los 4 campos restantes
scaler_t3 = StandardScaler()
X_scaled_sin_grasa = scaler_t3.fit_transform(X_sin_grasa)

plt.figure(figsize=(12, 7))
plt.title('TAREA 3: Dendrograma (Datos Estandarizados - Sin Grasa)')
Z_std = sch.linkage(X_scaled_sin_grasa, method='ward')
# Ajustamos el umbral (color_threshold) para obtener ~4 clusters, alineado al K-Means previo
sch.dendrogram(Z_std, color_threshold=6, leaf_font_size=8)
plt.xlabel('Índice del Cereal')
plt.ylabel('Distancia Euclidiana')
plt.axhline(y=6, color='r', linestyle='--') # Línea visual de corte
plt.show()


# --- OBJETIVO 5: Aplicar modelo jerárquico a los "mejores" resultados ---
# Con base en el dendrograma estandarizado y el análisis del codo previo, 4 clusters es lo ideal
n_clusters_jerarquico = 4
hc = AgglomerativeClustering(n_clusters=n_clusters_jerarquico, metric='euclidean', linkage='ward')
df['Cluster_T3_Jerarquico'] = hc.fit_predict(X_scaled_sin_grasa)

print(f"Se ha ajustado el Modelo Jerárquico con {n_clusters_jerarquico} clusters.")
print("Distribución de cereales por cluster jerárquico:")
print(df['Cluster_T3_Jerarquico'].value_counts().sort_index())


# --- OBJETIVO 6: Crear un mapa de clusters e interpretarlo ---
print("Generando Mapa de Clusters (Clustermap)...")
# Preparamos un DataFrame con los nombres de los cereales en el índice para mejor visualización
df_mapa = pd.DataFrame(X_scaled_sin_grasa, columns=X_sin_grasa.columns, index=df['Cereal Name'])

# Generamos el clustermap
# Nota: clustermap internamente vuelve a hacer el linkage jerárquico para ordenar filas y columnas
clustermap_fig = sns.clustermap(df_mapa, method='ward', cmap='coolwarm', figsize=(10, 10), standard_scale=None, center=0)
plt.suptitle('TAREA 3: Mapa de Clusters Jerárquico (Estandarizado, Sin Grasa)', y=1.02)
plt.show()


# ==============================================================================
# ================================= TAREA 4 ====================================
# ==============================================================================
print("" + "="*40)
print(" INICIANDO TAREA 4: AJUSTE DE HIPERPARÁMETROS DBSCAN")
print("="*40)

# --- Nuevas librerías para la Tarea 4 ---
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --- OBJETIVO 1: Recorrer varios valores de «eps» y «min_samples» ---
eps_range = np.arange(0.1, 2.1, 0.1)
min_samples_range = range(2, 11)

def find_best_dbscan_params(data, eps_range, min_samples_range):
    """
    Recorre los hiperparámetros de DBSCAN para encontrar la mejor puntuación de silueta.
    """
    best_score = -1
    best_eps = -1
    best_min_samples = -1
    results = []

    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(data)
            
            # El coeficiente de silueta solo se puede calcular si hay más de 1 cluster
            # y menos clusters que muestras.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1 and len(set(labels)) < len(data):
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
                results.append((eps, min_samples, score, n_clusters))
            else:
                # Se asigna -1 para combinaciones que no generan clusters válidos
                results.append((eps, min_samples, -1, n_clusters))

    if not results:
        return -1, -1, -1, pd.DataFrame()

    results_df = pd.DataFrame(results, columns=['eps', 'min_samples', 'silhouette_score', 'n_clusters'])
    return best_eps, best_min_samples, best_score, results_df

# --- OBJETIVO 2 : Aplicar la función al conjunto de datos original ---
print("--- Búsqueda en datos originales (no estandarizados) ---")
best_eps_orig, best_ms_orig, best_score_orig, results_orig = find_best_dbscan_params(X, eps_range, min_samples_range)
print(f"Mejor Puntuación de Silueta (Original): {best_score_orig:.4f}")
print(f"Mejor eps: {best_eps_orig:.1f}")
print(f"Mejor min_samples: {best_ms_orig}")

# Gráfico de respaldo: Mapa de calor de la puntuación de silueta
results_orig['eps'] = results_orig['eps'].round(1)
pivot_orig = results_orig.pivot(index='eps', columns='min_samples', values='silhouette_score')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_orig, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Puntuación de Silueta'})
plt.title('TAREA 4: Puntuación de Silueta DBSCAN (Datos Originales)')
plt.xlabel('min_samples')
plt.ylabel('eps')
plt.show()

# Aplicar la función al conjunto de datos estandarizado ---
print("--- Búsqueda en datos estandarizados ---")
best_eps_std, best_ms_std, best_score_std, results_std = find_best_dbscan_params(X_scaled, eps_range, min_samples_range)
print(f"Mejor Puntuación de Silueta (Estandarizado): {best_score_std:.4f}")
print(f"Mejor eps: {best_eps_std:.1f}")
print(f"Mejor min_samples: {best_ms_std}")

# Gráfico de respaldo: Mapa de calor de la puntuación de silueta
results_std['eps'] = results_std['eps'].round(1)
pivot_std = results_std.pivot(index='eps', columns='min_samples', values='silhouette_score')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_std, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Puntuación de Silueta'})
plt.title('TAREA 4: Puntuación de Silueta DBSCAN (Datos Estandarizados)')
plt.xlabel('min_samples')
plt.ylabel('eps')
plt.show()


# --- OBJETIVO 3: Buscar la puntuación más alta y anotar los valores ---
if best_score_orig > best_score_std:
    print("El mejor modelo se obtuvo con los datos ORIGINALES.")
    final_eps = best_eps_orig
    final_min_samples = best_ms_orig
    final_data = X
    dataset_name = "Original"
else:
    print(" El mejor modelo se obtuvo con los datos ESTANDARIZADOS.")
    final_eps = best_eps_std
    final_min_samples = best_ms_std
    final_data = X_scaled
    dataset_name = "Estandarizado"

print(f"Parámetros finales elegidos: eps={final_eps:.1f}, min_samples={final_min_samples}, sobre datos {dataset_name}")

# --- OBJETIVO 4: Ajustar un modelo DBSCAN definitivo y revisar etiquetas ---
dbscan_final = DBSCAN(eps=final_eps, min_samples=final_min_samples)
labels = dbscan_final.fit_predict(final_data)
df['Cluster_T4_DBSCAN'] = labels

n_clusters_final = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Modelo final ajustado. Resultados:")
print(f"Número de clusters encontrados: {n_clusters_final}")
print(f"Número de puntos de ruido: {n_noise}")
print("Distribución de cereales por cluster DBSCAN:")
print(df['Cluster_T4_DBSCAN'].value_counts())

# Gráfico de respaldo: Visualización de los clusters finales con PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(final_data)
df['pca1'] = data_pca[:, 0]
df['pca2'] = data_pca[:, 1]

plt.figure(figsize=(10, 8))
# Puntos de cluster
if n_clusters_final > 0:
    sns.scatterplot(data=df[df['Cluster_T4_DBSCAN'] != -1], x='pca1', y='pca2', hue='Cluster_T4_DBSCAN', 
                    palette=sns.color_palette("hsv", n_colors=n_clusters_final), s=100)

# Puntos de ruido
sns.scatterplot(data=df[df['Cluster_T4_DBSCAN'] == -1], x='pca1', y='pca2',
                color='black', marker='x', s=50, label='Ruido (-1)')

plt.title(f'TAREA 4: Clusters DBSCAN Finales (eps={final_eps:.1f}, min_samples={final_min_samples}) en datos {dataset_name}')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Cluster DBSCAN')
plt.show()


# ==============================================================================
# ========================== VALIDACION ADICIONAL ============================
# ==============================================================================
print("\n" + "="*50)
print(" ANÁLISIS CUANTITATIVO DE LOS CLUSTERS DBSCAN FINALES")
print("="*50)

# Filtrar solo los puntos que pertenecen a un cluster (excluir ruido)
df_clusters_finales = df[df['Cluster_T4_DBSCAN'] != -1].copy()

if not df_clusters_finales.empty:
    print("\n--- Características promedio de los clusters (en escala original) ---")
    # Agrupamos el DataFrame original (X) usando las etiquetas obtenidas
    X_con_clusters = X.copy()
    X_con_clusters['Cluster'] = df['Cluster_T4_DBSCAN']
    
    cluster_analysis = X_con_clusters[X_con_clusters['Cluster'] != -1].groupby('Cluster').mean()
    print(cluster_analysis.round(2))

    print("\n" + "-"*50)
    print("Interpretación:")
    print("- La tabla de arriba muestra los valores nutricionales promedio para cada cluster encontrado.")
    print("- Compara las filas (clusters) para ver en qué nutrientes se diferencian.")
    print("- Este análisis en el espacio original de características es la mejor forma de validar la separación,")
    print("  ya que el gráfico 2D con PCA es solo una proyección y puede causar que clusters distintos parezcan solapados.")
else:
    print("\nNo se encontraron clusters para analizar (todos los puntos fueron clasificados como ruido).")


