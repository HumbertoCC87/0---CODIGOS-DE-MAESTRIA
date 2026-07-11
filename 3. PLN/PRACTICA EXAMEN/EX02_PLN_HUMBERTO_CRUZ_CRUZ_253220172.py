### EXAMEN PRÁCTICO PLN · CORTE 2
### Materia: Procesamiento de Lenguaje Natural
### Alumno: Humberto Cruz Cruz

from pathlib import Path
import os
import pandas as pd
import spacy

# Configuración de rutas [3]
RUTA_DATASET = Path("Dataset_Examen_Practico_PLN_Corte2.csv")
CARPETA_SALIDA = Path("salidas_examen_practico")
CARPETA_SALIDA.mkdir(exist_ok=True)

def verificar_conda():
    """Regresa el nombre del ambiente Conda si está activo.""" [1]
    return os.environ.get("CONDA_DEFAULT_ENV", "NO_DETECTADO")

def cargar_dataset(ruta):
    # TODO 1: cargar el CSV con pandas y validar columnas [1]
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo en {ruta}")
    
    df = pd.read_csv(ruta)
    columnas_requeridas = ['id', 'texto', 'tipo_esperado']
    
    for col in columnas_requeridas:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria: {col}")
            
    print(f"Dataset cargado exitosamente con {len(df)} oraciones.")
    return df

def preparar_modelo():
    # TODO 2: cargar el recurso o modelo lingüístico [1]
    try:
        nlp = spacy.load("es_core_news_sm")
        return nlp
    except OSError:
        print("Descargando modelo es_core_news_sm...")
        os.system("python -m spacy download es_core_news_sm")
        return spacy.load("es_core_news_sm")

def analizar_oracion(texto, modelo):
    # TODO 3: analizar una oración y extraer métricas clave [2]
    doc = modelo(texto)
    
    # Extracción de componentes sintácticos
    raiz = [token.text for token in doc if token.head == token]
    sujeto = [token.text for token in doc if "subj" in token.dep_]
    objeto = [token.text for token in doc if "obj" in token.dep_]
    
    # Formatear dependencias principales (Relación(Cabeza, Dependiente))
    deps = [f"{token.dep_}({token.head.text}, {token.text})" for token in doc]
    
    # Lógica de clasificación de dificultad [4]
    dificultad = "Baja"
    obs = "Estructura analizada sin anomalías."
    
    if len(doc) > 15:
        dificultad = "Alta"
        obs = "Oración extensa; posible pérdida de precisión en dependencias largas."
    elif any(token.dep_ == "relcl" for token in doc):
        dificultad = "Media-Alta"
        obs = "Presencia de oración subordinada relativa."

    return {
        "raiz_detectada": raiz if raiz else "No detectada",
        "sujeto_detectado": ", ".join(sujeto) if sujeto else "No detectado",
        "objeto_detectado": ", ".join(objeto) if objeto else "No detectado",
        "dependencias_principales": " | ".join(deps[:5]), # Top 5 para el CSV
        "posible_dificultad": dificultad,
        "observacion_automatica": obs
    }

def analizar_dataset(df, modelo):
    # TODO 4: recorrer TODAS las oraciones y generar resultados [2]
    resultados = []
    for _, fila in df.iterrows():
        analisis = analizar_oracion(fila['texto'], modelo)
        # Combinar datos originales con análisis
        res_fila = {**fila.to_dict(), **analisis}
        resultados.append(res_fila)
    
    df_res = pd.DataFrame(resultados)
    # E04: Generar el CSV de resultados completos [4, 5]
    df_res.to_csv(CARPETA_SALIDA / "resultados_completos.csv", index=False)
    return df_res

def seleccionar_muestra(df_resultados):
    # TODO 5: seleccionar 12 oraciones (3 de cada tipo) [6]
    tipos = ['simple', 'sintagma_preposicional', 'ambigua', 'dificil_para_reglas']
    muestra = []
    
    for t in tipos:
        sub_df = df_resultados[df_resultados['tipo_esperado'] == t].head(3)
        muestra.append(sub_df)
    
    df_muestra = pd.concat(muestra)
    # E05: Generar el CSV de la muestra de 12 [5, 7]
    df_muestra.to_csv(CARPETA_SALIDA / "muestra_12_oraciones.csv", index=False)
    return df_muestra

def main():
    # E01: Verificar ambiente Conda [1, 8]
    ambiente = verificar_conda()
    print(f"Ambiente Conda detectado: {ambiente}")
    
    # Preparación
    nlp = preparar_modelo()
    df = cargar_dataset(RUTA_DATASET)
    
    # Procesamiento masivo (E04)
    df_resultados = analizar_dataset(df, nlp)
    
    # Generación de muestras (E05)
    df_muestra = seleccionar_muestra(df_resultados)
    
    # --- BLOQUE PARA CUMPLIR E10 (Archivos adicionales) --- [5]
    
    # 1. Generar archivo de casos de ambigüedad (ID 51-75 aproximadamente)
    df_ambiguas = df_resultados[df_resultados['tipo_esperado'] == 'ambigua']
    df_ambiguas.to_csv(CARPETA_SALIDA / "casos_ambiguedad.csv", index=False)
    
    # 2. Generar archivo de limitaciones detectadas (Basado en dificultad)
    df_limitaciones = df_resultados[df_resultados['posible_dificultad'].isin(['Alta', 'Media-Alta'])]
    df_limitaciones.to_csv(CARPETA_SALIDA / "limitaciones_detectadas.csv", index=False)
    
    print(f"Proceso finalizado. Archivos generados en: {CARPETA_SALIDA}")

if __name__ == "__main__":
    main()