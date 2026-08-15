# ============================================================================
# PROYECTO INTEGRADOR - CLASIFICACIÓN DE GÉNEROS DE PELÍCULAS
# VERSIÓN MEJORADA CON SENTIMIENTO, RESUMEN Y CONSULTA
# ============================================================================

import pandas as pd
import numpy as np
import re
import unicodedata
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import json
from pathlib import Path
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Descargar recursos necesarios de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Descargando recursos NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("📥 Descargando punkt_tab...")
    nltk.download('punkt_tab')

# Configuración de rutas
BASE_DIR = Path(__file__).parent
DATOS_ENTRADA = BASE_DIR / 'film_reviews_result.csv'
SALIDA = BASE_DIR / 'salidas'
MATRICES = SALIDA / 'matrices'
MODELOS = SALIDA / 'modelos'
CONFUSION = SALIDA / 'matrices_confusion'
SENTIMIENTOS = SALIDA / 'sentimientos'
RESUMENES = SALIDA / 'resumenes'

# Crear carpetas
for folder in [SALIDA, MATRICES, MODELOS, CONFUSION, SENTIMIENTOS, RESUMENES]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FUNCIONES DE LIMPIEZA (MANTENIDAS)
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
# NUEVAS FUNCIONES: ANÁLISIS DE SENTIMIENTO
# ============================================================================

def analizar_sentimiento(texto: str) -> dict:
    """
    Analiza el sentimiento de un texto usando TextBlob
    Retorna: polaridad (-1 a 1), subjetividad (0 a 1) y categoría
    """
    if not isinstance(texto, str) or not texto.strip():
        return {
            'polaridad': 0.0,
            'subjetividad': 0.0,
            'categoria': 'Neutro',
            'texto_limpio': ''
        }
    
    try:
        blob = TextBlob(texto)
        polaridad = blob.sentiment.polarity
        subjetividad = blob.sentiment.subjectivity
        
        # Clasificación en categorías
        if polaridad > 0.3:
            categoria = 'Positivo'
        elif polaridad < -0.3:
            categoria = 'Negativo'
        else:
            categoria = 'Neutro'
        
        return {
            'polaridad': round(polaridad, 4),
            'subjetividad': round(subjetividad, 4),
            'categoria': categoria,
            'texto_limpio': limpiar_texto(texto)
        }
    except Exception as e:
        return {
            'polaridad': 0.0,
            'subjetividad': 0.0,
            'categoria': 'Neutro',
            'texto_limpio': limpiar_texto(texto)
        }

def analizar_sentimiento_por_genero(df, columna_texto='review_text', columna_genero='genero_limpio'):
    """Analiza sentimiento agrupado por género"""
    resultados = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.get(columna_texto)):
            sentimiento = analizar_sentimiento(row[columna_texto])
            sentimiento['genero'] = row.get(columna_genero, 'Desconocido')
            sentimiento['id'] = row.get('id', idx)
            resultados.append(sentimiento)
    
    df_sentimiento = pd.DataFrame(resultados)
    
    # Estadísticas por género
    if not df_sentimiento.empty:
        stats_genero = df_sentimiento.groupby('genero').agg({
            'polaridad': ['mean', 'std', 'count'],
            'categoria': lambda x: x.value_counts().to_dict()
        }).round(4)
        return df_sentimiento, stats_genero
    else:
        return df_sentimiento, pd.DataFrame()

# ============================================================================
# NUEVAS FUNCIONES: RESUMEN DE TEXTOS (CORREGIDAS)
# ============================================================================

def sent_tokenize_seguro(texto: str, idioma: str = 'spanish') -> list:
    """
    Versión segura de sent_tokenize que maneja errores de recursos
    """
    if not texto or not texto.strip():
        return []
    
    try:
        # Intentar con el idioma especificado
        return sent_tokenize(texto, language=idioma)
    except LookupError:
        try:
            # Intentar con inglés como fallback
            return sent_tokenize(texto, language='english')
        except LookupError:
            # Si falla, dividir por puntos como último recurso
            # Dividir por . ! ? y mantener el separador
            import re
            oraciones = re.split(r'(?<=[.!?])\s+', texto)
            return [o for o in oraciones if o.strip()]

def resumir_texto(texto: str, num_oraciones: int = 5) -> dict:
    """
    Genera un resumen del texto usando frecuencia de palabras
    Retorna: resumen, oraciones originales, palabras clave
    """
    if not isinstance(texto, str) or not texto.strip():
        return {
            'resumen': '',
            'oraciones_originales': [],
            'palabras_clave': [],
            'num_oraciones': 0
        }
    
    # Tokenizar en oraciones usando la versión segura
    oraciones = sent_tokenize_seguro(texto, 'spanish')
    
    if not oraciones:
        return {
            'resumen': texto[:500] + '...' if len(texto) > 500 else texto,
            'oraciones_originales': [],
            'palabras_clave': [],
            'num_oraciones': 0
        }
    
    if len(oraciones) <= num_oraciones:
        return {
            'resumen': texto,
            'oraciones_originales': oraciones,
            'palabras_clave': [],
            'num_oraciones': len(oraciones)
        }
    
    # Limpiar y tokenizar todas las palabras
    texto_limpio = limpiar_texto(texto)
    palabras = tokenizar(texto_limpio)
    
    # Eliminar stopwords
    try:
        stop_words = set(stopwords.words('spanish'))
    except:
        stop_words = set()
    
    palabras_filtradas = [p for p in palabras if p not in stop_words and len(p) > 2]
    
    # Frecuencia de palabras
    if palabras_filtradas:
        fdist = FreqDist(palabras_filtradas)
        palabras_clave = [palabra for palabra, _ in fdist.most_common(10)]
    else:
        fdist = FreqDist(palabras)
        palabras_clave = [palabra for palabra, _ in fdist.most_common(10)]
    
    # Calcular importancia de cada oración
    importancia_oraciones = []
    for oracion in oraciones:
        oracion_limpia = limpiar_texto(oracion)
        palabras_oracion = tokenizar(oracion_limpia)
        # Puntaje basado en palabras clave
        puntaje = sum(fdist.get(palabra, 0) for palabra in palabras_oracion)
        importancia_oraciones.append((oracion, puntaje))
    
    # Ordenar por importancia y seleccionar las mejores
    importancia_oraciones.sort(key=lambda x: x[1], reverse=True)
    oraciones_seleccionadas = [oracion for oracion, _ in importancia_oraciones[:num_oraciones]]
    
    # Reordenar según aparición original
    oraciones_seleccionadas = sorted(
        oraciones_seleccionadas,
        key=lambda x: oraciones.index(x) if x in oraciones else 0
    )
    
    resumen = ' '.join(oraciones_seleccionadas)
    
    return {
        'resumen': resumen,
        'oraciones_originales': oraciones,
        'palabras_clave': palabras_clave[:10],
        'num_oraciones': len(oraciones_seleccionadas)
    }

def generar_resumen_corpus(df, columna_texto='review_text', num_oraciones=4):
    """Genera resúmenes para el corpus"""
    resumenes = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.get(columna_texto)):
            resumen = resumir_texto(row[columna_texto], num_oraciones)
            resumen['id'] = row.get('id', idx)
            resumen['genero'] = row.get('genero_limpio', 'Desconocido')
            texto_original = str(row[columna_texto])
            resumen['texto_original'] = texto_original[:200] + '...' if len(texto_original) > 200 else texto_original
            resumenes.append(resumen)
    
    return pd.DataFrame(resumenes)

# ============================================================================
# NUEVAS FUNCIONES: CONSULTA BÁSICA
# ============================================================================

class ConsultaPLN:
    """Sistema de consulta básica para el corpus"""
    
    def __init__(self, df, columna_texto='review_text', columna_etiqueta='genero_limpio'):
        self.df = df
        self.columna_texto = columna_texto
        self.columna_etiqueta = columna_etiqueta
        self.vectorizer = None
        self.tfidf_matrix = None
        self._indexar_documentos()
    
    def _indexar_documentos(self):
        """Indexa los documentos usando TF-IDF"""
        textos = self.df[self.columna_texto].fillna('').astype(str)
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(textos)
    
    def buscar(self, consulta: str, top_k: int = 5) -> pd.DataFrame:
        """
        Busca documentos similares a la consulta
        Retorna: DataFrame con los resultados
        """
        if not consulta.strip():
            return pd.DataFrame()
        
        consulta_vector = self.vectorizer.transform([consulta])
        similitudes = cosine_similarity(consulta_vector, self.tfidf_matrix).flatten()
        
        # Obtener índices de los más similares
        indices_top = similitudes.argsort()[-top_k:][::-1]
        
        resultados = []
        for idx in indices_top:
            row = self.df.iloc[idx]
            texto = str(row[self.columna_texto])
            resultados.append({
                'id': row.get('id', idx),
                'texto': texto[:300] + '...' if len(texto) > 300 else texto,
                'genero': row.get(self.columna_etiqueta, 'Desconocido'),
                'similitud': round(similitudes[idx], 4)
            })
        
        return pd.DataFrame(resultados)
    
    def filtrar_por_genero(self, genero: str, top_k: int = 10) -> pd.DataFrame:
        """Filtra documentos por género"""
        df_filtrado = self.df[self.df[self.columna_etiqueta].str.lower() == genero.lower()]
        if len(df_filtrado) == 0:
            return pd.DataFrame()
        
        return df_filtrado.head(top_k)[['id', self.columna_texto, self.columna_etiqueta]]
    
    def estadisticas_genero(self, genero: str = None) -> dict:
        """Estadísticas de un género o de todo el corpus"""
        if genero:
            df_filtrado = self.df[self.df[self.columna_etiqueta].str.lower() == genero.lower()]
        else:
            df_filtrado = self.df
        
        if len(df_filtrado) == 0:
            return {'error': 'Género no encontrado'}
        
        # Análisis de sentimiento
        df_sent, _ = analizar_sentimiento_por_genero(df_filtrado)
        
        estadisticas = {
            'total_documentos': len(df_filtrado),
            'generos_presentes': df_filtrado[self.columna_etiqueta].value_counts().to_dict(),
            'promedio_longitud': df_filtrado[self.columna_texto].str.len().mean(),
            'sentimiento_promedio': df_sent['polaridad'].mean() if not df_sent.empty else 0,
            'distribucion_sentimiento': df_sent['categoria'].value_counts().to_dict() if not df_sent.empty else {}
        }
        
        return estadisticas

# ============================================================================
# FUNCIONES DE ANÁLISIS (MANTENIDAS)
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
    
    return errores_por_clase

def generar_bitacora(df_original, df_filtrado, resultados, modelos_errores, 
                     mejor_modelo, mejor_acc, tiempo_ejecucion, 
                     df_sentimiento, stats_genero, df_resumenes):
    """Genera bitácora técnica mejorada"""
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
        
        f.write("3. ANÁLISIS DE SENTIMIENTO\n")
        f.write("-"*80 + "\n")
        if not df_sentimiento.empty:
            f.write(f"   Total de textos analizados: {len(df_sentimiento)}\n")
            f.write(f"   Polaridad promedio: {df_sentimiento['polaridad'].mean():.4f}\n")
            f.write(f"   Subjetividad promedio: {df_sentimiento['subjetividad'].mean():.4f}\n")
            f.write("\n   Distribución de sentimientos:\n")
            sent_dist = df_sentimiento['categoria'].value_counts()
            for cat, count in sent_dist.items():
                f.write(f"      {cat}: {count} ({count/len(df_sentimiento)*100:.1f}%)\n")
            f.write("\n   Sentimiento por género:\n")
            if not stats_genero.empty:
                for gen, stats in stats_genero.iterrows():
                    f.write(f"      {gen}: polaridad={stats[('polaridad','mean')]:.4f}, n={stats[('polaridad','count')]}\n")
        f.write("\n")
        
        f.write("4. RESUMEN DE TEXTOS\n")
        f.write("-"*80 + "\n")
        if not df_resumenes.empty:
            f.write(f"   Total de resúmenes generados: {len(df_resumenes)}\n")
            f.write(f"   Promedio de oraciones por resumen: {df_resumenes['num_oraciones'].mean():.1f}\n")
            f.write("\n   Ejemplo de palabras clave:\n")
            for i, row in df_resumenes.head(3).iterrows():
                palabras_clave = row.get('palabras_clave', [])
                if isinstance(palabras_clave, list) and palabras_clave:
                    f.write(f"      Doc {row['id']}: {', '.join(palabras_clave[:5])}\n")
        f.write("\n")
        
        f.write("5. PREPROCESAMIENTO APLICADO\n")
        f.write("-"*80 + "\n")
        f.write("   - Eliminación de acentos\n")
        f.write("   - Conversión a minúsculas\n")
        f.write("   - Eliminación de URLs y caracteres especiales\n")
        f.write("   - Eliminación de stopwords en español\n")
        f.write("   - Normalización de espacios\n\n")
        
        f.write("6. REPRESENTACIONES NUMÉRICAS\n")
        f.write("-"*80 + "\n")
        f.write("   Técnicas implementadas:\n")
        f.write("   - Bag of Words (CountVectorizer)\n")
        f.write("   - Matriz TF\n")
        f.write("   - Matriz TF-IDF\n")
        f.write("   - Similitud de documentos\n")
        f.write("   - Configuración: max_features=1000, ngram_range=(1,2)\n\n")
        
        f.write("7. RESULTADOS DE MODELOS\n")
        f.write("-"*80 + "\n")
        for nombre, acc in resultados.items():
            f.write(f"   {nombre}: {acc:.4f}\n")
        f.write(f"\n   🏆 Mejor modelo: {mejor_modelo} (Accuracy: {mejor_acc:.4f})\n\n")
        
        f.write("8. ANÁLISIS DE ERRORES\n")
        f.write("-"*80 + "\n")
        for nombre, errores_por_clase in modelos_errores.items():
            f.write(f"\n   Modelo: {nombre}\n")
            f.write(f"   {'-'*50}\n")
            for clase, stats in errores_por_clase.items():
                f.write(f"      {clase}: {stats['errores']}/{stats['total']} errores ({stats['tasa']:.2%})\n")
        
        f.write("\n9. OBSERVACIONES\n")
        f.write("-"*80 + "\n")
        f.write("   - El corpus presenta desbalanceo en algunas clases\n")
        f.write("   - Los modelos muestran mejor rendimiento en clases con más datos\n")
        f.write("   - Las confusiones ocurren entre géneros similares\n")
        f.write("   - El análisis de sentimiento muestra tendencias por género\n\n")
        
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
    print("VERSIÓN MEJORADA CON SENTIMIENTO, RESUMEN Y CONSULTA")
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
    try:
        stop_words_es = list(stopwords.words('spanish'))
    except:
        stop_words_es = []
    
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
    
    # 4. NUEVO: ANÁLISIS DE SENTIMIENTO
    print("\n💖 Analizando sentimiento del corpus...")
    df_sentimiento, stats_sentimiento = analizar_sentimiento_por_genero(df_filtrado)
    
    if not df_sentimiento.empty:
        df_sentimiento.to_csv(SENTIMIENTOS / '10_analisis_sentimiento.csv', index=False, encoding='utf-8-sig')
        
        # Visualización de sentimiento
        plt.figure(figsize=(12, 6))
        
        # Distribución de sentimientos
        plt.subplot(1, 2, 1)
        sent_counts = df_sentimiento['categoria'].value_counts()
        colors = {'Positivo': 'green', 'Neutro': 'gray', 'Negativo': 'red'}
        plt.pie(sent_counts.values, labels=sent_counts.index, autopct='%1.1f%%', 
                colors=[colors.get(c, 'blue') for c in sent_counts.index])
        plt.title('Distribución de Sentimientos')
        
        # Polaridad por género
        plt.subplot(1, 2, 2)
        polaridad_por_genero = df_sentimiento.groupby('genero')['polaridad'].mean().sort_values()
        polaridad_por_genero.plot(kind='barh', color='skyblue')
        plt.title('Polaridad Promedio por Género')
        plt.xlabel('Polaridad')
        plt.tight_layout()
        plt.savefig(SENTIMIENTOS / '11_grafico_sentimiento.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Análisis de sentimiento completado: {len(df_sentimiento)} textos")
        print(f"   Polaridad promedio: {df_sentimiento['polaridad'].mean():.4f}")
    
    # 5. NUEVO: RESUMEN DE TEXTOS
    print("\n📝 Generando resúmenes del corpus...")
    df_resumenes = generar_resumen_corpus(df_filtrado, num_oraciones=4)
    
    if not df_resumenes.empty:
        df_resumenes.to_csv(RESUMENES / '12_resumenes_textos.csv', index=False, encoding='utf-8-sig')
        
        # Guardar resúmenes por género
        for genero in df_filtrado['genero_limpio'].unique():
            resumenes_genero = df_resumenes[df_resumenes['genero'] == genero]
            if len(resumenes_genero) > 0:
                genero_limpio = genero.replace(' ', '_')
                resumenes_genero.to_csv(
                    RESUMENES / f'13_resumenes_{genero_limpio}.csv', 
                    index=False, encoding='utf-8-sig'
                )
        
        print(f"✅ Resúmenes generados: {len(df_resumenes)} textos")
        print(f"   Promedio de oraciones por resumen: {df_resumenes['num_oraciones'].mean():.1f}")
    
    # 6. NUEVO: SISTEMA DE CONSULTA
    print("\n🔎 Inicializando sistema de consulta...")
    consulta_sistema = ConsultaPLN(df_filtrado)
    
    # Ejemplos de consulta automáticos
    consultas_ejemplo = [
        "pelicula de acción emocionante",
        "buen drama con final triste",
        "comedia divertida para reir",
        "ciencia ficción con extraterrestres"
    ]
    
    resultados_consulta = []
    for consulta in consultas_ejemplo:
        resultados = consulta_sistema.buscar(consulta, top_k=3)
        if not resultados.empty:
            resultados['consulta'] = consulta
            resultados_consulta.append(resultados)
    
    if resultados_consulta:
        df_consultas = pd.concat(resultados_consulta, ignore_index=True)
        df_consultas.to_csv(SALIDA / '14_ejemplos_consulta.csv', index=False, encoding='utf-8-sig')
        print(f"✅ Sistema de consulta inicializado")
        print(f"   Ejemplos de consulta ejecutados: {len(consultas_ejemplo)}")
    
    # 7. GUARDAR DATOS PROCESADOS
    print("\n💾 Guardando datos procesados...")
    
    # DATOS PROCESADOS
    procesamiento_df = df_filtrado[['review_text', 'review_procesada', 'genero_limpio']].copy()
    procesamiento_df.columns = ['comentario_original', 'comentario_procesado', 'categoria']
    procesamiento_df.to_csv(SALIDA / '02_datos_procesados.csv', index=False, encoding='utf-8-sig')
    print("✅ Datos procesados guardados")
    
    # CORPUS CON FORMATO SOLICITADO
    corpus_df = pd.DataFrame({
        'id': range(len(df_filtrado)),
        'texto': df_filtrado['review_text'],
        'texto_procesado': df_filtrado['review_procesada'],
        'etiqueta': df_filtrado['genero_limpio']
    })
    corpus_df.to_csv(SALIDA / '03_corpus_clasificacion.csv', index=False, encoding='utf-8-sig')
    print("✅ Corpus guardado (id, texto, etiqueta)")
    
    # 8. REPRESENTACIONES NUMÉRICAS
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
    
    # 9. ENTRENAMIENTO DE MODELOS
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
        
        # Métricas
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
        
        # Análisis de errores
        errores = analizar_errores_modelo(y_test, y_pred, X_test, nombre, generos_presentes)
        modelos_errores[nombre] = errores
    
    # 10. COMPARATIVA FINAL
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
    
    # 11. BITÁCORA TÉCNICA
    tiempo_ejecucion = (datetime.now() - inicio).total_seconds()
    generar_bitacora(df, df_filtrado, resultados, modelos_errores, 
                    mejor_modelo, mejor_acc, tiempo_ejecucion,
                    df_sentimiento, stats_sentimiento, df_resumenes)
    
    # 12. RESUMEN FINAL (CORREGIDO)
    print("\n" + "="*80)
    print("RESUMEN EJECUTIVO - PROYECTO INTEGRADOR")
    print("="*80)
    
    # Calcular valores para evitar errores en el f-string
    total_sentimientos = len(df_sentimiento) if not df_sentimiento.empty else 0
    polaridad_promedio = df_sentimiento['polaridad'].mean() if not df_sentimiento.empty else 0
    distribucion_sentimiento = dict(df_sentimiento['categoria'].value_counts()) if not df_sentimiento.empty else 'N/A'
    total_resumenes = len(df_resumenes) if not df_resumenes.empty else 0
    promedio_oraciones = df_resumenes['num_oraciones'].mean() if not df_resumenes.empty else 0
    
    print(f"""
📌 DATOS DEL PROYECTO:
  - Total de reseñas procesadas: {len(df_filtrado)}
  - Géneros clasificados: {len(df_filtrado['genero_limpio'].unique())}

💖 ANÁLISIS DE SENTIMIENTO:
  - Textos analizados: {total_sentimientos}
  - Polaridad promedio: {polaridad_promedio:.4f}
  - Distribución: {distribucion_sentimiento}

📝 RESUMEN DE TEXTOS:
  - Resúmenes generados: {total_resumenes}
  - Promedio de oraciones por resumen: {promedio_oraciones:.1f}

🔎 SISTEMA DE CONSULTA:
  - Documentos indexados: {len(df_filtrado)}
  - Características: TF-IDF, búsqueda por similitud, filtrado por género

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
  - Sentimientos: {SENTIMIENTOS}/
  - Resúmenes: {RESUMENES}/
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