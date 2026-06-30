#Codigo PLN y HPC_Tarea 3
#Autor Humberto Cruz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

# Importar librerías para GPU (cuML)
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml.linear_model import LogisticRegression as cuLR
from cuml.svm import SVC as cuSVC
from cuml.preprocessing import StandardScaler as cuStandardScaler

# Cargar el dataset
try:
    df = pd.read_csv('ecommerceDataset.csv', header=None)
except FileNotFoundError:
    print("Error: 'ecommerceDataset.csv' no encontrado. Asegúrate de que el archivo esté en el mismo directorio.")
    exit()

# Asignar nombres a las columnas para mayor claridad (opcional, basado en la estructura común de estos datasets)
# Suponemos que la primera columna es la etiqueta y el resto son características.
df.columns = ['label'] + [f'feature_{i}' for i in range(df.shape[1] - 1)]

# Convertir la columna 'label' a tipo numérico si es necesario (ej. si son strings)
# Esto es crucial para cuML, que espera etiquetas numéricas.
# Si las etiquetas son strings, necesitamos codificarlas.
if df['label'].dtype == 'object':
    df['label'] = df['label'].astype('category').cat.codes

# Separar características (X) y etiquetas (y)
X = df.drop('label', axis=1)
y = df['label']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir a arrays de cuML (GPU)
# cuML trabaja mejor con sus propios tipos de datos o arrays de NumPy que luego convierte.
# Para grandes datasets, es preferible usar cuDF para la carga y preprocesamiento.
# Aquí, para simplificar, asumimos que X_train, X_test, y_train, y_test ya son pandas DataFrames/Series.
# cuML puede manejar directamente DataFrames de Pandas, pero la conversión explícita a#Codigo PLN y HPC_Tarea 3