'''El propósito principal de este script es tomar un conjunto de datos de calificaciones de estudiantes, que  tiene múltiples columnas (una por cada materia), y reducir toda esa información a solo dos dimensiones. Esto permite visualizar a cada estudiante como un punto en un gráfico 2D.

La técnica que utiliza para lograr esto es t-SNE (t-distributed Stochastic Neighbor Embedding). Es un método de aprendizaje no supervisado excelente para visualizar datos de alta dimensionalidad, ya que agrupa los puntos que son "similares" en el espacio original (estudiantes con patrones de notas parecidos) y separa los que son diferentes.'''


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1. Ajustar un modelo t-SNE con dos componentes
# Carga y preprocesamiento de datos
df = pd.read_excel('student_grades.xlsx')
X = df.drop(columns=['student_id'])

# Centrado/Estandarización de datos
scaler = StandardScaler(with_std=False)
X_centered = scaler.fit_transform(X)

# Inicialización y ajuste del modelo t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init='pca',
    learning_rate='auto'
)
X_tsne = tsne.fit_transform(X_centered)

# 2. Representar a los alumnos en un gráfico de dispersión (Componente 1 vs Componente 2)
plt.figure(figsize=(9, 6))
plt.scatter(
    X_tsne[:, 0], 
    X_tsne[:, 1], 
    c='coral', 
    alpha=0.8, 
    edgecolors='k', 
    s=70
)

# Configuración de ejes y títulos
plt.title('Representación 2D de Alumnos mediante t-SNE', fontsize=13, fontweight='bold')
plt.xlabel('Componente 1 de t-SNE (Dimensión 1)', fontsize=11)
plt.ylabel('Componente 2 de t-SNE (Dimensión 2)', fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()