import pandas as pd
import time
import spacy
import multiprocessing as mp
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os

# ==========================================
# FUNCIONES AISLADAS PARA PARALELISMO DE TAREAS
# Cada función correrá en un núcleo distinto
# ==========================================

def ejecutar_tfidf(textos):
    """Ejecuta el modelo TF-IDF completo en un solo núcleo"""
    start = time.time()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(textos)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    keywords_list = []
    # Procesamiento secuencial dentro de este núcleo
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(i)
        if len(row.indices) == 0:
            keywords_list.append("")
            continue
        sorted_indices = row.indices[np.argsort(-row.data)][:5]
        top_words = [feature_names[idx] for idx in sorted_indices]
        keywords_list.append(", ".join(top_words))
        
    tiempo = time.time() - start
    return 'Keywords', keywords_list, tiempo

def ejecutar_sentiment(textos):
    """Ejecuta el análisis de TextBlob completo en un solo núcleo"""
    start = time.time()
    sentimientos = []
    # Procesamiento secuencial dentro de este núcleo
    for text in textos:
        try:
            sentimientos.append(TextBlob(str(text)).sentiment.polarity)
        except:
            sentimientos.append(0.0)
            
    tiempo = time.time() - start
    return 'Sentiment', sentimientos, tiempo

def ejecutar_ner(textos):
    """Ejecuta la extracción de Spacy completa en un solo núcleo"""
    start = time.time()
    # Cargamos el modelo dentro de la función para evitar conflictos de memoria entre procesos
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    
    entidades_list = []
    # Usamos n_process=1 para asegurar que Spacy no intente sub-paralelizar
    for doc in nlp.pipe(textos, n_process=1, batch_size=500):
        entidades = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
        entidades_list.append(" | ".join(entidades))
        
    tiempo = time.time() - start
    return 'Entities', entidades_list, tiempo


if __name__ == '__main__':
    # 1. Configuración de Rutas
    FILE_PATH = r"E:\Carpeta compartidad Workgroup\INTELIGENCIA ARTIFICIAL\COMPUTACION DE ALTO DESEMPEÑO\CODIGO HPC\ecommerceDataset.csv"
    OUTPUT_PATH = r"E:\Carpeta compartidad Workgroup\INTELIGENCIA ARTIFICIAL\COMPUTACION DE ALTO DESEMPEÑO\CODIGO HPC\ecommerce_procesado_task_parallel.csv"
    
    print("Iniciando evaluación HPC: Paralelismo de Tareas (Task Parallelism)...")
    print("Se asignará 1 núcleo dedicado a cada modelo (3 núcleos en uso simultáneo).\n")

    # 2. Carga del Dataset
    print(f"Cargando dataset desde: {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH, header=None, names=['label', 'text'])
    df['text'] = df['text'].fillna("").astype(str)
    textos = df['text'].tolist()

    # 3. Ejecución Simultánea (Task Parallelism)
    start_total_hpc = time.time()
    
    resultados_datos = {}
    tiempos_ejecucion = {}
    
    # max_workers=3 fuerza a que solo haya 3 procesos independientes (uno por modelo)
    with ProcessPoolExecutor(max_workers=3) as executor:
        print("Lanzando TF-IDF, TextBlob y Spacy de forma concurrente...")
        
        # submit() envía las tareas al pool y las ejecuta de inmediato
        futuro_tfidf = executor.submit(ejecutar_tfidf, textos)
        futuro_sentiment = executor.submit(ejecutar_sentiment, textos)
        futuro_ner = executor.submit(ejecutar_ner, textos)
        
        diccionario_futuros = {
            futuro_tfidf: "TF-IDF Keywords",
            futuro_sentiment: "TextBlob Sentiment",
            futuro_ner: "Spacy NER"
        }
        
        # as_completed nos avisa conforme cada núcleo va terminando su tarea
        for futuro in as_completed(diccionario_futuros):
            nombre_modelo = diccionario_futuros[futuro]
            # Obtenemos los resultados de la función
            col_name, datos, tiempo = futuro.result()
            
            resultados_datos[col_name] = datos
            tiempos_ejecucion[nombre_modelo] = tiempo
            print(f"✓ {nombre_modelo} finalizó su tarea en: {tiempo:.2f} segundos.")

    tiempo_total_hpc = time.time() - start_total_hpc

    # 4. Integración de resultados al DataFrame
    print("\nEnsamblando resultados en el dataset original...")
    df['Keywords'] = resultados_datos['Keywords']
    df['Sentiment'] = resultados_datos['Sentiment']
    df['Entities'] = resultados_datos['Entities']
    
    # 5. Guardado y Resumen
    df_final = df[['label', 'text', 'Keywords', 'Sentiment', 'Entities']]
    df_final.to_csv(OUTPUT_PATH, index=False)
    
    print("-" * 50)
    print("RESUMEN HPC - PARALELISMO DE TAREAS (1 Núcleo por Modelo):")
    print("-" * 50)
    print(f"1. TF-IDF Keywords:    {tiempos_ejecucion.get('TF-IDF Keywords', 0):.2f} s")
    print(f"2. TextBlob Sentiment: {tiempos_ejecucion.get('TextBlob Sentiment', 0):.2f} s")
    print(f"3. Spacy NER:          {tiempos_ejecucion.get('Spacy NER', 0):.2f} s")
    print("-" * 50)
    print(f"TIEMPO TOTAL DEL FLUJO: {tiempo_total_hpc:.2f} s")
    print(f"(Nota: El tiempo total está dictado por el modelo más lento, ya que corrieron al mismo tiempo)")
    print(f"Archivo guardado en:\n{OUTPUT_PATH}")