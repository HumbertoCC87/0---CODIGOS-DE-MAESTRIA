# ============================================================================
# PROYECTO INTEGRADOR - CLASIFICACIÓN DE GÉNEROS DE PELÍCULAS
# VERSIÓN SIMPLIFICADA - SOLO CSV (SIN EXCEL)
# ============================================================================

import pandas as pd
import numpy as np
import re
import unicodedata
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Configuración de rutas
BASE_DIR = Path(__file__).parent
DATOS_ENTRADA = BASE_DIR / 'film_reviews_result.csv'
SALIDA = BASE_DIR / 'salidas'
MATRICES = SALIDA / 'matrices'
MODELOS = SALIDA / 'modelos'
CONFUSION = SALIDA / 'matrices_confusion'

# Crear carpetas
for folder in [SALIDA, MATRICES, MODELOS, CONFUSION]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FUNCIONES DE LIMPIEZA
# ============================================================================

def quitar_acentos(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    normalizado = unicodedata.normalize("NFD", texto)
    return "".join(caracter for caracter in normalizado if unicodedata.category(caracter) != "Mn")

def limpiar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.lower().strip()
    texto = quitar_acentos(texto)
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^a-zñ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def tokenizar(texto: str) -> list:
    return [t for t in limpiar_texto(texto).split() if t]

def obtener_contexto(tokens: list, indice: int, ventana: int = 3) -> str:
    inicio = max(0, indice - ventana)
    fin = min(len(tokens), indice + ventana + 1)
    return ' '.join(tokens[inicio:fin])

# ============================================================================
# FUNCIONES DE ANÁLISIS
# ============================================================================

def analizar_ambiguedad_texto(texto: str, palabras_ambiguas: set) -> dict:
    tokens = tokenizar(texto)
    resultado = {
        'tokens': tokens,
        'palabras_ambiguas': [],
        'contextos': [],
        'tiene_ambiguedad': False
    }
    
    for i, token in enumerate(tokens):
        if token in palabras_ambiguas:
            resultado['palabras_ambiguas'].append(token)
            resultado['contextos'].append(obtener_contexto(tokens, i))
            resultado['tiene_ambiguedad'] = True
    
    return resultado

def analizar_errores_modelo(y_test, y_pred, X_test, modelo_nombre, etiquetas):
    """Analiza errores del modelo"""
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE ERRORES - {modelo_nombre}")
    print('='*60)
    
    errores = y_test != y_pred
    n_errores = errores.sum()
    n_total = len(y_test)
    
    print(f"\n📊 Estadísticas de errores:")
    print(f"  - Total de ejemplos: {n_total}")
    print(f"  - Ejemplos mal clasificados: {n_errores}")
    print(f"  - Tasa de error: {n_errores/n_total:.2%}")
    
    print(f"\n📈 Errores por clase:")
    errores_por_clase = {}
    for clase in etiquetas:
        mask_clase = y_test == clase
        total_clase = mask_clase.sum()
        if total_clase > 0:
            errores_clase = ((y_test == clase) & (y_pred != clase)).sum()
            tasa_error = errores_clase / total_clase
            errores_por_clase[clase] = {
                'total': total_clase,
                'errores': errores_clase,
                'tasa': tasa_error
            }
            print(f"  - {clase}: {errores_clase}/{total_clase} ({tasa_error:.2%})")
    
    print(f"\n📝 Ejemplos de errores:")
    indices_error = np.where(errores)[0]
    if len(indices_error) > 0:
        for i in range(min(3, len(indices_error))):
            idx = indices_error[i]
            texto = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
            real = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
            pred = y_pred[idx]
            print(f"\n  Ejemplo {i+1}:")
            print(f"    Texto: {str(texto)[:150]}..." if len(str(texto)) > 150 else f"    Texto: {texto}")
            print(f"    Real: {real}")
            print(f"    Predicho: {pred}")
    
    return errores_por_clase

def generar_bitacora(df_original, df_filtrado, resultados, modelos_errores, 
                mejor_modelo, mejor_acc, tiempo_ejecucion):
    """Genera bitácora técnica"""
    bitacora_path = SALIDA / '00_bitacora_tecnica.txt'
    
    with open(bitacora_path, 'w', encoding='utf-8-sig') as f:
        f.write("="*80 + "\n")
        f.write("BITÁCORA TÉCNICA - PROYECTO INTEGRADOR PLN\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. INFORMACIÓN DE EJECUCIÓN\n")
        f.write("-"*80 + "\n")
        f.write(f"   Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"   Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos\n")
        f.write(f"   Python version: {sys.version}\n\n")
        
        f.write("2. INFORMACIÓN DEL CORPUS\n")
        f.write("-"*80 + "\n")
        f.write(f"   Registros originales: {len(df_original)}\n")
        f.write(f"   Registros después de filtrar: {len(df_filtrado)}\n")
        f.write(f"   Géneros encontrados: {sorted(df_filtrado['genero_limpio'].unique())}\n")
        f.write(f"\n   Distribución de géneros:\n")
        for gen, count in df_filtrado['genero_limpio'].value_counts().items():
            f.write(f"      {gen}: {count} ({count/len(df_filtrado)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("3. PREPROCESAMIENTO APLICADO\n")
        f.write("-"*80 + "\n")
        f.write("   - Eliminación de acentos\n")
        f.write("   - Conversión a minúsculas\n")
        f.write("   - Eliminación de URLs y caracteres especiales\n")
        f.write("   - Eliminación de stopwords en español\n")
        f.write("   - Normalización de espacios\n\n")
        
        f.write("4. REPRESENTACIONES NUMÉRICAS\n")
        f.write("-"*80 + "\n")
        f.write("   Técnicas implementadas:\n")
        f.write("   - Bag of Words (CountVectorizer)\n")
        f.write("   - Matriz TF\n")
        f.write("   - Matriz TF-IDF\n")
        f.write("   - Similitud de documentos\n")
        f.write("   - Configuración: max_features=1000, ngram_range=(1,2)\n\n")
        
        f.write("5. RESULTADOS DE MODELOS\n")
        f.write("-"*80 + "\n")
        for nombre, acc in resultados.items():
            f.write(f"   {nombre}: {acc:.4f}\n")
        f.write(f"\n   🏆 Mejor modelo: {mejor_modelo} (Accuracy: {mejor_acc:.4f})\n\n")
        
        f.write("6. ANÁLISIS DE ERRORES\n")
        f.write("-"*80 + "\n")
        for nombre, errores_por_clase in modelos_errores.items():
            f.write(f"\n   Modelo: {nombre}\n")
            f.write(f"   {'-'*50}\n")
            for clase, stats in errores_por_clase.items():
                f.write(f"      {clase}: {stats['errores']}/{stats['total']} errores ({stats['tasa']:.2%})\n")
        
        f.write("\n7. OBSERVACIONES\n")
        f.write("-"*80 + "\n")
        f.write("   - El corpus presenta desbalanceo en algunas clases\n")
        f.write("   - Los modelos muestran mejor rendimiento en clases con más datos\n")
        f.write("   - Las confusiones ocurren entre géneros similares\n\n")
        
        f.write("="*80 + "\n")
        f.write("FIN DE BITÁCORA\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Bitácora técnica generada: {bitacora_path}")

# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

def main():
    inicio = datetime.now()
    
    print("\n" + "="*80)
    print("PROYECTO INTEGRADOR - CLASIFICACIÓN DE GÉNEROS DE PELÍCULAS")
    print("="*80)
    print(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. CARGA DE DATOS
    print("\n📂 Cargando datos...")
    try:
        df = pd.read_csv(DATOS_ENTRADA, sep='|', encoding='utf-8')
        print(f"✅ Datos cargados: {len(df)} registros")
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo {DATOS_ENTRADA}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR al cargar datos: {e}")
        sys.exit(1)
    
    # 2. PREPROCESAMIENTO
    print("\n🔧 Preprocesando datos...")
    
    def asignar_genero_oficial(texto_genero):
        if pd.isna(texto_genero):
            return None
        texto_genero_limpio = limpiar_texto(texto_genero)
        
        sinonimos = {
            'accion': 'Accion', 'acción': 'Accion',
            'aventura': 'Aventura',
            'ciencia ficcion': 'Ciencia Ficcion',
            'ciencia-ficcion': 'Ciencia Ficcion',
            'scifi': 'Ciencia Ficcion',
            'comedia': 'Comedia',
            'drama': 'Drama',
            'fantasia': 'Fantasia',
            'musical': 'Musical',
            'romance': 'Romance',
            'terror': 'Terror',
            'western': 'Western',
            'thriller': 'Terror',
            'suspenso': 'Terror'
        }
        
        for sinonimo, genero in sinonimos.items():
            if sinonimo in texto_genero_limpio:
                return genero
        return None
    
    df['genero_limpio'] = df['gender'].apply(asignar_genero_oficial)
    df_filtrado = df.dropna(subset=['genero_limpio']).copy()
    
    print(f"✅ Registros después de filtrar: {len(df_filtrado)}")
    
    # Limpieza de texto
    stop_words_es = list(stopwords.words('spanish'))
    palabras_contexto = ['pelicula', 'peliculas', 'serie', 'series', 'director', 
                        'actor', 'actores', 'ver', 'vista', 'pantalla']
    stop_words_es.extend(palabras_contexto)
    
    def limpiar_review(texto):
        if not isinstance(texto, str):
            return ""
        texto = limpiar_texto(texto)
        palabras = texto.split()
        palabras_filtradas = [w for w in palabras if w not in stop_words_es]
        return " ".join(palabras_filtradas)
    
    df_filtrado['review_procesada'] = df_filtrado['review_text'].apply(limpiar_review)
    print("✅ Limpieza de texto completada")
    
    # 3. ANÁLISIS DE AMBIGÜEDAD
    print("\n🔍 Analizando ambigüedad en el corpus...")
    
    palabras_ambiguas = {
        'banco', 'planta', 'cura', 'capital', 'raton', 'gato', 'sierra', 
        'vela', 'curso', 'red', 'copa', 'columna', 'radio', 'clave', 'linea', 
        'cabo', 'plata', 'oro', 'estrella', 'corona', 'fuente', 'campo'
    }
    
    analisis_ambiguedad = []
    for idx, row in df_filtrado.iterrows():
        resultado = analizar_ambiguedad_texto(row['review_text'], palabras_ambiguas)
        if resultado['tiene_ambiguedad']:
            analisis_ambiguedad.append({
                'id': row.get('id', idx),
                'texto': row['review_text'][:200],
                'palabras_ambiguas': ' | '.join(resultado['palabras_ambiguas'][:5]),
                'genero': row['genero_limpio']
            })
    
    if analisis_ambiguedad:
        ambiguedad_df = pd.DataFrame(analisis_ambiguedad)
        ambiguedad_df.to_csv(SALIDA / '01_analisis_ambiguedad.csv', index=False, encoding='utf-8-sig')
        print(f"✅ Análisis de ambigüedad guardado: {len(analisis_ambiguedad)} textos")
    
    # 4. GUARDAR DATOS PROCESADOS
    print("\n💾 Guardando datos procesados...")
    
    # DATOS PROCESADOS (objetivo 1)
    procesamiento_df = df_filtrado[['review_text', 'review_procesada', 'genero_limpio']].copy()
    procesamiento_df.columns = ['comentario_original', 'comentario_procesado', 'categoria']
    procesamiento_df.to_csv(SALIDA / '02_datos_procesados.csv', index=False, encoding='utf-8-sig')
    print("✅ Datos procesados guardados")
    
    # CORPUS CON FORMATO SOLICITADO: id, texto, etiqueta (objetivo 4)
    corpus_df = pd.DataFrame({
        'id': range(len(df_filtrado)),
        'texto': df_filtrado['review_text'],
        'texto_procesado': df_filtrado['review_procesada'],
        'etiqueta': df_filtrado['genero_limpio']
    })
    corpus_df.to_csv(SALIDA / '03_corpus_clasificacion.csv', index=False, encoding='utf-8-sig')
    print("✅ Corpus guardado (id, texto, etiqueta)")
    
    # 5. REPRESENTACIONES NUMÉRICAS (objetivo 2)
    print("\n📊 Generando representaciones numéricas...")
    
    X = df_filtrado['review_procesada']
    y = df_filtrado['genero_limpio']
    
    # Bag of Words
    print("\n  📌 Bag of Words...")
    bow_vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2))
    X_bow = bow_vectorizer.fit_transform(X)
    bow_df = pd.DataFrame(X_bow.toarray(), columns=[f'bow_{i}' for i in range(X_bow.shape[1])])
    bow_df.to_csv(MATRICES / '04_bag_of_words.csv', index=True, encoding='utf-8-sig')
    print(f"     ✅ BOW: {X_bow.shape[0]} docs x {X_bow.shape[1]} features")
    
    # Matriz TF
    print("\n  📌 Matriz TF...")
    tf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), use_idf=False)
    X_tf = tf_vectorizer.fit_transform(X)
    tf_df = pd.DataFrame(X_tf.toarray(), columns=[f'tf_{i}' for i in range(X_tf.shape[1])])
    tf_df.to_csv(MATRICES / '05_matriz_tf.csv', index=True, encoding='utf-8-sig')
    print(f"     ✅ TF: {X_tf.shape[0]} docs x {X_tf.shape[1]} features")
    
    # Matriz TF-IDF
    print("\n  📌 Matriz TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), use_idf=True)
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=[f'tfidf_{i}' for i in range(X_tfidf.shape[1])])
    tfidf_df.to_csv(MATRICES / '06_matriz_tfidf.csv', index=True, encoding='utf-8-sig')
    print(f"     ✅ TF-IDF: {X_tfidf.shape[0]} docs x {X_tfidf.shape[1]} features")
    
    # Top términos
    print("\n  📌 Top términos TF-IDF...")
    feature_names = tfidf_vectorizer.get_feature_names_out()
    idf_scores = tfidf_vectorizer.idf_
    idf_df = pd.DataFrame({'termino': feature_names, 'idf_score': idf_scores})
    idf_df = idf_df.sort_values('idf_score', ascending=False)
    idf_df.head(100).to_csv(MATRICES / '07_top_terminos_tfidf.csv', index=False, encoding='utf-8-sig')
    print("     ✅ Top términos guardado")
    
    # Similitud entre documentos
    print("\n  📌 Similitud entre documentos...")
    X_sample = X_tfidf[:100]
    similitud_matrix = cosine_similarity(X_sample)
    similitud_df = pd.DataFrame(
        similitud_matrix,
        index=[f'doc_{i}' for i in range(100)],
        columns=[f'doc_{i}' for i in range(100)]
    )
    similitud_df.to_csv(MATRICES / '08_similitud_documentos.csv', index=True, encoding='utf-8-sig')
    print("     ✅ Matriz de similitud guardada")
    
    # 6. ENTRENAMIENTO DE MODELOS (objetivo 3)
    print("\n🤖 Entrenando modelos de clasificación...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    generos_presentes = y_test.unique()
    print(f"📊 Géneros en prueba: {sorted(generos_presentes)}")
    
    modelos = {
        'Naive Bayes': MultinomialNB(),
        'SVM Lineal': LinearSVC(random_state=42, max_iter=2000),
        'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    resultados = {}
    modelos_errores = {}
    
    for nombre, algoritmo in modelos.items():
        print(f"\n{'='*50}")
        print(f"Entrenando: {nombre}")
        print('='*50)
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', algoritmo)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Métricas (objetivo 4)
        acc = accuracy_score(y_test, y_pred)
        resultados[nombre] = acc
        
        print(f"✅ Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=generos_presentes)
        cm_df = pd.DataFrame(cm, index=generos_presentes, columns=generos_presentes)
        cm_df.to_csv(CONFUSION / f'matriz_confusion_{nombre.replace(" ", "_")}.csv', 
                    encoding='utf-8-sig')
        
        # Visualización
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=generos_presentes,
                    yticklabels=generos_presentes)
        plt.title(f'Matriz de Confusión - {nombre}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(CONFUSION / f'matriz_confusion_{nombre.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
        plt.close()
        
        # Análisis de errores (objetivo 5)
        errores = analizar_errores_modelo(y_test, y_pred, X_test, nombre, generos_presentes)
        modelos_errores[nombre] = errores
    
    # 7. COMPARATIVA FINAL
    print("\n" + "="*60)
    print("COMPARATIVA FINAL DE MODELOS")
    print("="*60)
    
    for nombre, acc in resultados.items():
        print(f"  {nombre}: {acc:.4f}")
    
    mejor_modelo = max(resultados, key=resultados.get)
    mejor_acc = resultados[mejor_modelo]
    print(f"\n🏆 Mejor modelo: ¡{mejor_modelo}! (Accuracy: {mejor_acc:.4f})")
    
    # Guardar resultados
    resultados_df = pd.DataFrame({
        'modelo': list(resultados.keys()),
        'accuracy': list(resultados.values())
    })
    resultados_df = resultados_df.sort_values('accuracy', ascending=False)
    resultados_df.to_csv(MODELOS / '09_comparativa_modelos.csv', index=False, encoding='utf-8-sig')
    
    # 8. BITÁCORA TÉCNICA (objetivo 6)
    tiempo_ejecucion = (datetime.now() - inicio).total_seconds()
    generar_bitacora(df, df_filtrado, resultados, modelos_errores, 
                    mejor_modelo, mejor_acc, tiempo_ejecucion)
    
    # 9. RESUMEN FINAL
    print("\n" + "="*80)
    print("RESUMEN EJECUTIVO - PROYECTO INTEGRADOR")
    print("="*80)
    print(f"""
    📌 DATOS DEL PROYECTO:
    - Total de reseñas procesadas: {len(df_filtrado)}
    - Géneros clasificados: {len(df_filtrado['genero_limpio'].unique())}
    
    🔧 PREPROCESAMIENTO:
    - Limpieza de texto completa
    - Eliminación de stopwords
    - Análisis de ambigüedad léxica
    
    📊 REPRESENTACIONES GENERADAS:
    - Bag of Words (BOW)
    - Matriz TF
    - Matriz TF-IDF
    - Top términos por relevancia
    - Similitud de documentos
    
    🤖 MODELOS EVALUADOS:
    - Naive Bayes: {resultados['Naive Bayes']:.4f}
    - SVM Lineal: {resultados['SVM Lineal']:.4f}
    - Regresión Logística: {resultados['Regresión Logística']:.4f}
    
    🏆 MEJOR MODELO:
    - {mejor_modelo} (Accuracy: {mejor_acc:.4f})
    
    📁 ARCHIVOS GENERADOS:
    - Datos procesados: {SALIDA}/
    - Matrices: {MATRICES}/
    - Modelos: {MODELOS}/
    - Matrices de confusión: {CONFUSION}/
    """)
    
    print("="*80)
    print(f"✅ PROYECTO COMPLETADO EXITOSAMENTE")
    print(f"   Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos")
    print("="*80)

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()