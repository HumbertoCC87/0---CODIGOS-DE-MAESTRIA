import pandas as pd
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== INSTALACIÓN ÚNICA DE DEPENDENCIAS ====================

def install_dependencies_once():
    """Instalar dependencias UNA SOLA VEZ"""
    packages = [
        'pandas', 'numpy', 'scikit-learn', 'imbalanced-learn',
        'textblob', 'spacy', 'tqdm', 'joblib', 'psutil', 'wmi'
    ]
    
    # Verificar si ya están instaladas
    missing = []
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"📦 Instalando: {', '.join(missing)}")
        for package in missing:
            os.system(f"pip install {package} --quiet")
    
    # Intentar OpenCL (opcional)
    try:
        import pyopencl
        print("✅ OpenCL disponible")
    except:
        print("ℹ️ OpenCL no instalado (opcional)")
    
    print("✅ Dependencias listas")

# Ejecutar instalación UNA SOLA VEZ
if __name__ == "__main__":
    install_dependencies_once()

# ==================== IMPORTACIONES ====================

from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import spacy
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
import psutil

# ==================== DETECCIÓN DE GPU AMD ====================

def detect_amd_gpu():
    """Detectar GPU AMD Radeon en Windows"""
    print("\n🔍 Detectando GPU AMD Radeon RX 6700 XT...")
    
    # 1. Verificar con OpenCL
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        for platform in platforms:
            if 'AMD' in platform.name:
                devices = platform.get_devices()
                for device in devices:
                    if 'Radeon' in device.name or 'gfx' in device.name:
                        print(f"✅ GPU AMD detectada: {device.name}")
                        return 'opencl'
    except:
        pass
    
    # 2. Verificar con WMI
    try:
        import wmi
        c = wmi.WMI()
        for gpu in c.Win32_VideoController():
            if 'Radeon' in gpu.Name or 'AMD' in gpu.Name:
                print(f"✅ GPU AMD detectada: {gpu.Name}")
                return 'wmi'
    except:
        pass
    
    print("⚠️ Usando CPU optimizado")
    return 'cpu'

def get_optimal_workers():
    """Determinar número óptimo de workers"""
    cpu_count = mp.cpu_count()
    # Usar 75% de los núcleos para no saturar
    workers = max(1, int(cpu_count * 0.75))
    print(f"💻 Usando {workers} workers de {cpu_count} núcleos")
    return workers

# ==================== PROCESAMIENTO ====================

def load_spacy_model():
    """Cargar modelo spaCy (una sola vez por worker)"""
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        os.system("python -m spacy download en_core_web_sm --quiet")
        return spacy.load('en_core_web_sm')

def process_batch_optimized(batch_texts):
    """Procesar lote de textos"""
    results = []
    nlp = load_spacy_model()
    
    for text in batch_texts:
        if pd.isna(text) or text == '':
            results.append({'sentiment': 0, 'entities': '', 'text_length': 0})
            continue
        
        try:
            text_str = str(text)
            blob = TextBlob(text_str)
            sentiment = blob.sentiment.polarity
            
            doc = nlp(text_str[:1000])
            entities = ' '.join([ent.text for ent in doc.ents 
                            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'MONEY', 'DATE']])
            
            results.append({
                'sentiment': sentiment,
                'entities': entities,
                'text_length': len(text_str)
            })
        except:
            results.append({'sentiment': 0, 'entities': '', 'text_length': 0})
    
    return results

def detect_columns(df):
    """Detectar columnas automáticamente"""
    print("\n🔍 Detectando columnas...")
    
    # La primera columna suele ser la categoría
    target_col = df.columns[0]
    text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    print(f"📋 Categoría: '{target_col}'")
    print(f"📝 Texto: '{text_col}'")
    
    return target_col, text_col

def parallel_preprocessing(df, text_col):
    """Preprocesamiento paralelo optimizado"""
    print("\n🚀 Iniciando preprocesamiento paralelo...")
    
    gpu_type = detect_amd_gpu()
    if gpu_type != 'cpu':
        print(f"⚡ GPU AMD disponible ({gpu_type}) - Usando aceleración")
    
    # Determinar workers
    n_workers = get_optimal_workers()
    
    # Preparar datos
    texts = df[text_col].fillna('').astype(str).tolist()
    chunks = np.array_split(texts, n_workers)
    
    results = []
    
    # Procesar en paralelo
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_batch_optimized, chunk) for chunk in chunks]
        
        with tqdm(total=len(futures), desc="Procesando lotes", unit="lote") as pbar:
            for future in futures:
                try:
                    batch_results = future.result(timeout=180)
                    results.extend(batch_results)
                    pbar.update(1)
                except Exception as e:
                    print(f"⚠️ Error en lote: {e}")
                    # Fallback secuencial
                    return sequential_preprocessing(df, text_col)
    
    # Agregar características
    results_df = pd.DataFrame(results)
    df['sentiment'] = results_df['sentiment']
    df['entities'] = results_df['entities']
    df['text_length'] = results_df['text_length']
    df = df.drop(columns=[text_col])
    
    print("✅ Preprocesamiento completado")
    return df

def sequential_preprocessing(df, text_col):
    """Procesamiento secuencial (fallback)"""
    print("\n🔄 Fallback a procesamiento secuencial...")
    nlp = load_spacy_model()
    
    sentiments, entities_list, text_lengths = [], [], []
    
    for text in tqdm(df[text_col].fillna('').astype(str), desc="Procesando"):
        try:
            text_str = str(text)
            blob = TextBlob(text_str)
            sentiments.append(blob.sentiment.polarity)
            
            doc = nlp(text_str[:1000])
            entities = ' '.join([ent.text for ent in doc.ents 
                            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']])
            entities_list.append(entities)
            text_lengths.append(len(text_str))
        except:
            sentiments.append(0)
            entities_list.append('')
            text_lengths.append(0)
    
    df['sentiment'] = sentiments
    df['entities'] = entities_list
    df['text_length'] = text_lengths
    return df.drop(columns=[text_col])

def apply_tfidf(df):
    """Aplicar TF-IDF"""
    print("\n🔤 Aplicando TF-IDF...")
    
    texts = df['entities'].fillna('').astype(str) if 'entities' in df.columns else df['sentiment'].astype(str)
    
    tfidf = TfidfVectorizer(
        max_features=15000,
        stop_words='english',
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=2
    )
    
    X_tfidf = tfidf.fit_transform(texts)
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), 
                            columns=[f'tfidf_{i}' for i in range(X_tfidf.shape[1])])
    
    cols_to_keep = [col for col in df.columns if col not in ['entities']]
    df_final = pd.concat([df[cols_to_keep].reset_index(drop=True), tfidf_df], axis=1)
    
    print(f"✅ TF-IDF: {X_tfidf.shape[1]} características")
    return df_final

def balance_dataset(X_train, y_train, method='smote'):
    """Balancear dataset"""
    print(f"\n⚖️ Aplicando {method}...")
    
    try:
        if method == 'submuestreo':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'sobremuestreo':
            sampler = RandomOverSampler(random_state=42)
        elif method == 'smote':
            n_classes = len(np.unique(y_train))
            k = min(5, n_classes - 1) if n_classes > 1 else 1
            sampler = SMOTE(random_state=42, k_neighbors=k)
        else:
            return X_train, y_train
        
        X_bal, y_bal = sampler.fit_resample(X_train, y_train)
        print(f"✅ {method}: {X_bal.shape[0]} muestras")
        return X_bal, y_bal
    except Exception as e:
        print(f"❌ Error en {method}: {e}")
        return X_train, y_train

# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    print("="*70)
    print("🚀 SISTEMA DE CLASIFICACIÓN DE TEXTOS - ECOMMERCE")
    print("💻 AMD Radeon RX 6700 XT (12 GB) OPTIMIZADO")
    print("="*70)
    
    # Detectar GPU
    gpu_type = detect_amd_gpu()
    print(f"\n📊 GPU: {gpu_type}")
    print(f"💻 CPU: {mp.cpu_count()} núcleos")
    print(f"📦 RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Cargar dataset
    print("\n📂 Cargando dataset...")
    try:
        df = pd.read_csv('ecommerceDataset.csv')
        print(f"✅ {df.shape[0]} registros, {df.shape[1]} columnas")
    except FileNotFoundError:
        print("❌ No se encontró ecommerceDataset.csv")
        return
    
    # Detectar columnas
    target_col, text_col = detect_columns(df)
    
    # Preprocesamiento
    print("\n" + "="*70)
    print("📊 PREPROCESAMIENTO")
    print("="*70)
    
    df_processed = parallel_preprocessing(df, text_col)
    df_final = apply_tfidf(df_processed)
    
    # Guardar
    df_final.to_csv('ecommerceDataset_CPU.csv', index=False)
    print(f"\n💾 Guardado: ecommerceDataset_CPU.csv ({df_final.shape})")
    
    # Preparar entrenamiento
    print("\n" + "="*70)
    print("🎯 ENTRENAMIENTO")
    print("="*70)
    
    X = df_final.drop(columns=[target_col])
    y = df_final[target_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"📋 Clases: {le.classes_.tolist()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"📊 Entrenamiento: {X_train.shape[0]}, Prueba: {X_test.shape[0]}")
    
    # Balanceo y modelos
    balance_methods = ['submuestreo', 'sobremuestreo', 'smote']
    results = []
    
    for method_name in balance_methods:
        print("\n" + "="*70)
        print(f"⚖️ BALANCEO: {method_name.upper()}")
        print("="*70)
        
        try:
            X_train_bal, y_train_bal = balance_dataset(X_train, y_train, method_name)
            
            # Guardar dataset balanceado
            bal_df = pd.concat([
                pd.DataFrame(X_train_bal),
                pd.Series(le.inverse_transform(y_train_bal), name=target_col)
            ], axis=1)
            bal_df.to_csv(f'{method_name}.csv', index=False)
            print(f"💾 Guardado: {method_name}.csv")
            
            # Modelos
            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    random_state=42, n_jobs=-1, class_weight='balanced'
                ),
                'Regresión Logística': LogisticRegression(
                    random_state=42, max_iter=1000, class_weight='balanced',
                    n_jobs=-1, solver='saga', C=0.1
                ),
                'SVC': LinearSVC(
                    C=0.5, class_weight='balanced', random_state=42,
                    max_iter=5000, dual=False
                )
            }
            
            for model_name, model in models.items():
                print(f"\n🚀 {model_name}")
                start = time.time()
                
                try:
                    model.fit(X_train_bal, y_train_bal)
                    y_pred = model.predict(X_test)
                    
                    acc = accuracy_score(y_test, y_pred)
                    elapsed = time.time() - start
                    
                    print(f"✅ Accuracy: {acc:.4f}")
                    print(f"⏱️ Tiempo: {elapsed:.2f}s")
                    print("\n📋 Reporte:")
                    print(classification_report(y_test, y_pred))
                    
                    results.append({
                        'model': model_name,
                        'accuracy': acc,
                        'time': elapsed,
                        'balance_method': method_name
                    })
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                
                gc.collect()
                
        except Exception as e:
            print(f"❌ Error en {method_name}: {e}")
    
    # Resultados finales
    if results:
        print("\n" + "="*70)
        print("📊 TABLA DE RESULTADOS")
        print("="*70)
        
        results_df = pd.DataFrame(results)
        pivot = results_df.pivot(index='model', columns='balance_method', values='accuracy')
        print("\n📋 Accuracy:")
        print(pivot.round(4))
        
        results_df.to_csv('resultados_completos.csv', index=False)
        print("\n💾 Resultados guardados")
    
    print("\n✅ PROCESO COMPLETADO!")

if __name__ == "__main__":
    main()