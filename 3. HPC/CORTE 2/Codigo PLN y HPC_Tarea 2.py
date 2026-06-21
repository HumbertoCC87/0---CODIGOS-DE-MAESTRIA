#Codigo de Entrenamiento y Evaluación de Modelos ML con Paralelismo de Tareas (HPC) - Parte 2
import pandas as pd
import time
import logging
import multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.sparse import hstack, csr_matrix

# Librerías de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ==========================================
# CONFIGURACIÓN DEL SISTEMA DE LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("HPC_ML_Training")

# ==========================================
# FUNCIONES AISLADAS PARA ENTRENAMIENTO (1 NÚCLEO CADA UNA)
# ==========================================

def entrenar_evaluar_rf(X_train, X_test, y_train, y_test):
    start = time.time()
    modelo = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    tiempo = time.time() - start
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return 'Random Forest', acc, rec, f1, tiempo

def entrenar_evaluar_lr(X_train, X_test, y_train, y_test):
    start = time.time()
    modelo = LogisticRegression(max_iter=1000, n_jobs=1, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    tiempo = time.time() - start
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return 'Regresión Logística', acc, rec, f1, tiempo

def entrenar_evaluar_svm(X_train, X_test, y_train, y_test):
    start = time.time()
    modelo = SVC(kernel='linear', random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    tiempo = time.time() - start
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return 'SVM', acc, rec, f1, tiempo


if __name__ == '__main__':
    # 1. Configuración de Rutas
    INPUT_PATH = r"E:\Carpeta compartidad Workgroup\INTELIGENCIA ARTIFICIAL\COMPUTACION DE ALTO DESEMPEÑO\CODIGO HPC\ecommerce_procesado_task_parallel.csv"
    
    logger.info("Iniciando fase de Entrenamiento y Evaluación ML HPC...")
    logger.info(f"Cargando dataset preprocesado: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=['label', 'text'])

# =========================================================================
    # BLOQUE DE INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
    # Integrando Texto (TF-IDF), Polaridad de Sentimiento y Conteo de Entidades
    # =========================================================================
    logger.info("Iniciando Ingeniería de Características con protección de tipos...")

    # A. Característica Textual: Matriz TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_text = vectorizer.fit_transform(df['text'])

    # B. Característica Numérica 1: Sentimiento (Con Coerción Numérica Estricta)
    # pd.to_numeric con errors='coerce' transforma cualquier texto infiltrado en NaN
    sentiment_clean = pd.to_numeric(df['Sentiment'], errors='coerce').fillna(0.0)
    sentiment_feature = sentiment_clean.values.reshape(-1, 1)
    X_sentiment = csr_matrix(sentiment_feature)

    # C. Característica Numérica 2: Densidad de Entidades
    def contar_entidades(x):
        if pd.isna(x) or str(x).strip() == "":
            return 0
        return len(str(x).split(' | '))

    # Aplicamos el conteo y forzamos el tipo a float para compatibilidad con Scipy
    entities_clean = df['Entities'].apply(contar_entidades)
    entities_count = pd.to_numeric(entities_clean, errors='coerce').fillna(0.0).values.reshape(-1, 1)
    X_entities = csr_matrix(entities_count)

    # D. Fusión de Características
    logger.info("Fusionando características de forma segura (TF-IDF + Sentimiento + Entidades)...")
    X_final = hstack([X_text, X_sentiment, X_entities])
    y_final = df['label']
    # =========================================================================
    # FIN DEL BLOQUE DE INGENIERÍA DE CARACTERÍSTICAS
    # =========================================================================
    
    logger.info("Realizando partición Train/Test (80% - 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
    
    # 3. Ejecución Simultánea (Task Parallelism)
    logger.info("Lanzando RF, LR y SVM simultáneamente (1 Núcleo por modelo)...")
    start_total_hpc = time.time()
    
    resultados_metricas = []
    
    with ProcessPoolExecutor(max_workers=3) as executor:
        futuro_rf = executor.submit(entrenar_evaluar_rf, X_train, X_test, y_train, y_test)
        futuro_lr = executor.submit(entrenar_evaluar_lr, X_train, X_test, y_train, y_test)
        futuro_svm = executor.submit(entrenar_evaluar_svm, X_train, X_test, y_train, y_test)
        
        futuros = [futuro_rf, futuro_lr, futuro_svm]
        
        for futuro in as_completed(futuros):
            nombre, acc, rec, f1, tiempo = futuro.result()
            resultados_metricas.append({
                'Modelo': nombre,
                'Tiempo (s)': round(tiempo, 2),
                'Accuracy': round(acc, 4),
                'Recall': round(rec, 4),
                'F1-Score': round(f1, 4)
            })
            logger.info(f"✓ Modelo {nombre} entrenado y evaluado en {tiempo:.2f} segundos.")

    tiempo_total_hpc = time.time() - start_total_hpc

    # 4. Presentación de Resultados Tabulados
    df_resultados = pd.DataFrame(resultados_metricas)
    df_resultados = df_resultados.sort_values(by='Modelo').reset_index(drop=True)
    
    print("\n" + "=" * 70)
    print("TABLA DE RESULTADOS Y MÉTRICAS (Feature Engineering + HPC)")
    print("=" * 70)
    print(df_resultados.to_string(index=False))
    print("-" * 70)
    print(f"TIEMPO TOTAL DEL FLUJO: {tiempo_total_hpc:.2f} s")
    print("=" * 70)
    logger.info("Proceso ML HPC completado con éxito.")