# ---
# Tarea HPC con RAPIDS (cuML)
# Modelos: Regresión Logística, SVM, Random Forest en GPU
# ---

import cudf
import cuml
from cuml.model_selection import train_test_split
from cuml.preprocessing import StandardScaler
from cuml.metrics import accuracy_score, precision_recall_fscore_support
import time
import numpy as np
import sys

def main():
    # --- 1. Configuración y Carga de Datos en GPU ---
    print("--- Iniciando Tarea HPC con RAPIDS cuML ---")
    
    # Comprobar si hay una GPU disponible a través de RAPIDS
    try:
        cuml.common.device_memory_resource.get_default_resource()
        print("GPU detectada por RAPIDS.")
    except Exception as e:
        print(f"ERROR: No se pudo inicializar RAPIDS. ¿Está la GPU disponible y los drivers de NVIDIA instalados? Error: {e}")
        sys.exit()

    # Cargar datos directamente en la GPU con cuDF
    try:
        df = cudf.read_csv('ecommerceDataset.csv', header=None)
        print("Dataset 'ecommerceDataset.csv' cargado en la GPU.")
    except FileNotFoundError:
        print("ERROR: 'ecommerceDataset.csv' no encontrado. Asegúrate de que esté en el mismo directorio que el script dentro del contenedor.")
        sys.exit()

    # --- 2. Preparación de Datos ---
    # Asignar nombres a las columnas
    num_features = df.shape[1] - 1
    df.columns = ['label'] + [f'feature_{i}' for i in range(num_features)]

    # Codificar etiquetas a formato numérico
    if df['label'].dtype == 'object':
        df['label'] = df['label'].astype('category').cat.codes

    # Separar características (X) y etiquetas (y)
    X = df.drop('label', axis=1)
    y = df['label']

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Escalar características para Regresión Logística y SVM
    # No es estrictamente necesario para Random Forest, pero lo hacemos para consistencia
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Datos preparados: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} muestras de prueba.")
    print("-" * 40)


    # --- 3. Definición y Ejecución de Modelos ---

    def train_and_evaluate(model, model_name, X_train_data, y_train_data, X_test_data, y_test_data):
        """Función para entrenar, evaluar y medir el rendimiento de un modelo cuML."""
        print(f"--- Entrenando y Evaluando: {model_name} ---")
        
        # Iniciar cronómetro
        start_time = time.time()
        
        # Entrenar el modelo
        model.fit(X_train_data, y_train_data)
        
        # Detener cronómetro
        training_time = time.time() - start_time
        
        # Realizar predicciones
        y_pred = model.predict(X_test_data)
        
        # Calcular métricas (en GPU)
        # Nota: debemos mover los datos a CPU para imprimir/usar con algunas librerías
        y_test_cpu = y_test_data.to_numpy()
        y_pred_cpu = y_pred.to_numpy()

        accuracy = accuracy_score(y_test_data, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_data, y_pred, average='weighted')

        # Imprimir resultados
        print(f"  Modelo: {model_name}")
        print(f"  Tiempo de entrenamiento: {training_time:.4f} segundos")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision (ponderada): {precision:.4f}")
        print(f"  Recall (ponderado): {recall:.4f}")
        print("-" * 40)

    # --- 4. Ejecución Secuencial ---
    
    # Modelo 1: Regresión Logística
    # Utiliza los datos escalados
    lr_model = cuml.LogisticRegression(penalty='l2', C=1.0, tol=1e-4, fit_intercept=True, max_iter=1000)
    train_and_evaluate(lr_model, "cuML Regresión Logística", X_train_scaled, y_train, X_test_scaled, y_test)

    # Modelo 2: Support Vector Machine (SVM)
    # Utiliza los datos escalados
    svm_model = cuml.svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, max_iter=-1)
    train_and_evaluate(svm_model, "cuML Support Vector Machine (SVC)", X_train_scaled, y_train, X_test_scaled, y_test)

    # Modelo 3: Random Forest
    # Puede usar los datos originales o escalados, usamos los originales
    rf_model = cuml.RandomForestClassifier(n_estimators=100, max_depth=16, split_criterion='gini', n_bins=128, random_state=42)
    train_and_evaluate(rf_model, "cuML Random Forest", X_train, y_train, X_test, y_test)
    
    print("--- Tarea HPC con RAPIDS cuML completada ---")

if __name__ == "__main__":
    main()
