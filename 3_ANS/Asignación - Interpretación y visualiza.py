#Asignación - Interpretación y visualización PCA
#Aprendizaje NO supervisado
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Carga de datos
df = pd.read_excel('student_grades.xlsx')
X = df.drop(columns=['student_id'])

# Centrado de datos
scaler = StandardScaler(with_std=False)
X_centered = scaler.fit_transform(X)

# Modelo PCA (2 componentes)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_centered)

# Dataframe de componentes (Loadings)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=X.columns
)

print("--- CARGAS VECTORIALES (LOADINGS) ---")
print(loadings.round(3))

# 2. Representación Gráfica de Dispersión (PC1 vs PC2)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c='purple', alpha=0.7, edgecolors='k', s=60)

# Ejes de referencia (origen 0,0)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

# Etiquetas e información del gráfico
plt.title('Mapa de Perfiles Académicos de los Alumnos (PC1 vs PC2)', fontsize=13, fontweight='bold')
plt.xlabel('Eje X: PC1 (81.8% Varianza - Nivel Académico Global)', fontsize=11)
plt.ylabel('Eje Y: PC2 (9.8% Varianza - Perfil STEM vs Humanidades)', fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()