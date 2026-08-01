#Asignación - PCA
#Aprendizaje no supervisado - Análisis de Componentes Principales (PCA)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Leer el conjunto de datos de las notas de los alumnos
archivo = 'student_grades.xlsx'
df = pd.read_excel(archivo)

# 2. Eliminar la primera columna ('student_id')
X = df.drop(columns=['student_id'])

# 3. Centrar de datos
scaler = StandardScaler(with_std=False)
X_centered = scaler.fit_transform(X)

# 4. Ajustar un modelo de PCA con 2 componentes
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_centered)

# 5. Visualizar e interpretar los índices de varianza explicada
var_explicada = pca_2d.explained_variance_ratio_
var_acumulada = np.sum(var_explicada)

print("--- RESULTADOS DEL PCA ---")
print(f"Varianza explicada por Componente 1 (PC1): {var_explicada[0]*100:.2f}%")
print(f"Varianza explicada por Componente 2 (PC2): {var_explicada[1]*100:.2f}%")
print(f"Varianza explicada acumulada (2D):        {var_acumulada*100:.2f}%\n")

# Graficar la varianza explicada y la proyección 2D
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Varianza Explicada por todas las componentes
pca_full = PCA().fit(X_centered)
componentes = [f'PC{i+1}' for i in range(len(pca_full.explained_variance_ratio_))]

ax[0].bar(componentes, pca_full.explained_variance_ratio_ * 100, color='skyblue', label='Varianza Individual')
ax[0].plot(componentes, np.cumsum(pca_full.explained_variance_ratio_) * 100, color='red', marker='o', label='Varianza Acumulada')
ax[0].axhline(y=var_acumulada*100, color='green', linestyle='--', label=f'2 Componentes ({var_acumulada*100:.1f}%)')
ax[0].set_title('Varianza Explicada por Componente Principal')
ax[0].set_xlabel('Componentes Principales')
ax[0].set_ylabel('Porcentaje de Varianza Explicada (%)')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# Gráfico 2: Dispersión de los alumnos en 2D
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c='purple', alpha=0.7, edgecolors='k')
ax[1].set_title('Proyección 2D de Perfiles de Alumnos')
ax[1].set_xlabel(f'PC1 ({var_explicada[0]*100:.1f}% varianza)')
ax[1].set_ylabel(f'PC2 ({var_explicada[1]*100:.1f}% varianza)')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()