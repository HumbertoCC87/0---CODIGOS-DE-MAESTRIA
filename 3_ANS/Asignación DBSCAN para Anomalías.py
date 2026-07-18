import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Cargar los datos
# Excluimos 'user_id' porque es una variable categórica que no sirve para el cálculo de distancias
df = pd.read_csv('tripadvisor_reviews.csv')
X = df.drop(columns=['user_id'])

# ==========================================
# OBJETIVO 1: Función DBSCAN para evaluación
# ==========================================
def evaluar_dbscan(X, eps_rango, min_samples_rango):
    mejor_silueta = -1
    mejores_parametros = {'eps': None, 'min_samples': None}
    
    for eps in eps_rango:
        for min_samples in min_samples_rango:
            modelo = DBSCAN(eps=eps, min_samples=min_samples)
            etiquetas = modelo.fit_predict(X)
            
            # La métrica de silueta requiere al menos 2 clusters y no considerar todo como ruido
            if len(set(etiquetas)) > 1 and len(set(etiquetas)) < len(X):
                score = silhouette_score(X, etiquetas)
                if score > mejor_silueta:
                    mejor_silueta = score
                    mejores_parametros = {'eps': eps, 'min_samples': min_samples}
                    
    return mejor_silueta, mejores_parametros

# ==========================================
# OBJETIVO 2: Aplicar la función al conjunto de datos
# ==========================================
# Definimos los hiperparámetros a explorar
valores_eps = np.arange(0.1, 1.0, 0.1)
valores_min_samples = range(2, 10)

mejor_score, mejores_params = evaluar_dbscan(X, valores_eps, valores_min_samples)

# ==========================================
# OBJETIVO 3: Buscar la puntuación de silueta más alta y anotar valores
# ==========================================
print("--- RESULTADOS DE LA EVALUACIÓN ---")
print(f"Mejor puntuación de silueta: {mejor_score:.4f}")
print(f"Mejores parámetros: eps={mejores_params['eps']:.1f}, min_samples={mejores_params['min_samples']}")

# ==========================================
# OBJETIVO 4: Aplicar un único modelo DBSCAN utilizando esos valores
# ==========================================
modelo_final = DBSCAN(eps=mejores_params['eps'], min_samples=mejores_params['min_samples'])
df['cluster'] = modelo_final.fit_predict(X)

# ==========================================
# OBJETIVO 5: Anotar las anomalías (-1) y visualizarlas
# ==========================================
anomalias = df[df['cluster'] == -1]

print("\n--- DETECCIÓN DE ANOMALÍAS ---")
print(f"Total de anomalías detectadas: {len(anomalias)}")
print("\nRegistros anómalos (Primeros 5):")
print(anomalias.head())

# Visualización en un gráfico de pares
plt.figure(figsize=(10, 8))
sns.pairplot(df.drop(columns=['user_id']), hue='cluster', palette='tab10', diag_kind='kde')
plt.suptitle('DBSCAN: Detección de Anomalías en Valoraciones Turísticas', y=1.02)
plt.show()