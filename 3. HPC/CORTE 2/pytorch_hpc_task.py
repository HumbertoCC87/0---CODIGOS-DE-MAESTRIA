# Adaptación de PLN y HPC Tarea 3 a PyTorch


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import numpy as np
import sys

# --- 1. Configuración del Dispositivo (GPU o CPU) ---
# Comprobar si la GPU (CUDA) está disponible. Si no, detener el programa.
if not torch.cuda.is_available():
    print("ERROR: No se encontró una GPU compatible con CUDA. El programa se detendrá.")
    sys.exit()  # Detener la ejecución si no hay GPU

device = torch.device("cuda")
print(f"Usando dispositivo: {device}")

# --- 2. Carga y Preparación de Datos ---
try:
    df = pd.read_csv('3. HPC/CORTE 2/ecommerceDataset.csv', header=None)
except FileNotFoundError:
    print("Error: 'ecommerceDataset.csv' no encontrado. Asegúrate de que la ruta sea correcta.")
    exit()

# Asignar nombres a las columnas
num_features = df.shape[1] - 1
df.columns = ['label'] + [f'feature_{i}' for i in range(num_features)]

# Codificar etiquetas si son de tipo 'object'
if df['label'].dtype == 'object':
    df['label'] = df['label'].astype('category').cat.codes

# Separar características (X) y etiquetas (y)
X = df.drop('label', axis=1).values
y = df['label'].values

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar características: Es una buena práctica para redes neuronales
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir datos a tensores de PyTorch y moverlos a la GPU
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)


# --- 3. Definición de los Modelos de Red Neuronal ---

# Modelo 1: Regresión Logística (una capa lineal)
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Modelo 2: Red Neuronal Sencilla (MLP - Multi-Layer Perceptron)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Modelo 3: Red Neuronal más Compleja (MLP con más capas y Dropout)
class ComplexMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ComplexMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

# --- 4. Función de Entrenamiento y Evaluación ---

def train_and_evaluate(model, model_name, epochs=100, learning_rate=0.01):
    print(f"---Entrenando y Evaluando: {model_name} ---")
    
    # Mover el modelo a la GPU
    model.to(device)
    
    # Pérdida y optimizador
    # CrossEntropyLoss es ideal para clasificación multi-clase.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Iniciar el cronómetro
    start_time = time.time()
    
    # Bucle de entrenamiento
    for epoch in range(epochs):
        model.train() # Poner el modelo en modo de entrenamiento
        
        optimizer.zero_grad() # Limpiar gradientes
        outputs = model(X_train_tensor) # Forward pass
        loss = criterion(outputs, y_train_tensor) # Calcular pérdida
        loss.backward() # Backward pass
        optimizer.step() # Actualizar pesos
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    # Detener el cronómetro de entrenamiento
    training_time = time.time() - start_time
    
    # Fase de evaluación
    model.eval() # Poner el modelo en modo de evaluación
    with torch.no_grad(): # No necesitamos calcular gradientes en evaluación
        test_outputs = model(X_test_tensor)
        _, predicted_indices = torch.max(test_outputs, 1)
        
        # Mover las predicciones y etiquetas a la CPU para usar sklearn
        y_pred = predicted_indices.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        
        # Calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support_array = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        support = np.sum(support_array) if support_array is not None else 0

    # Imprimir resultados
    print(f"Modelo: {model_name}")
    print(f"  Tiempo de entrenamiento: {training_time:.4f} segundos")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (ponderada): {precision:.4f}")
    print(f"  Recall (ponderado): {recall:.4f}")
    print(f"  Support: {support}")
    print("-" * 30)

# --- 5. Ejecución Secuencial de los Modelos ---

# Obtener dimensiones para los modelos
input_dim = X_train_tensor.shape[1]
output_dim = len(np.unique(y)) # Número de clases

# Instanciar modelos
model1 = LogisticRegressionModel(input_dim, output_dim)
model2 = SimpleMLP(input_dim, hidden_dim=100, output_dim=output_dim)
model3 = ComplexMLP(input_dim, hidden_dim1=128, hidden_dim2=64, output_dim=output_dim)

# Ejecutar el proceso para cada modelo
train_and_evaluate(model1, "PyTorch Regresión Logística")
train_and_evaluate(model2, "PyTorch MLP Sencillo")
train_and_evaluate(model3, "PyTorch MLP Complejo")
