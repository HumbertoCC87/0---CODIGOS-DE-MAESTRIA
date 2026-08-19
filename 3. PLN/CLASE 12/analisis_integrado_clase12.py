# Clase 12 PLN · Integración de sistema de intención, entidades y respuesta
# Autoría académica: M.C. Pablo Ricardo Sánchez Gómez
# Objetivo: analizar intención, entidades, confianza y errores sobre un dataset amplio.
# Irregularidad intencional: el dataset incluye etiquetas no canónicas y el catálogo de entidades está incompleto.
# La actividad consiste en detectar, ajustar y comparar resultados antes/después.

from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RUTA = Path("dataset_intenciones_clase12.csv")
SALIDA = Path("salidas_clase12")
SALIDA.mkdir(exist_ok=True)

# Ajuste esperado 1: completar y corregir estas variantes con base en las advertencias del sistema.
NORMALIZACION_INTENCIONES = {
    "soporte técnico": "soporte_tecnico",
    "soporte_tecncio": "soporte_tecnico",
    "entrega actividad": "entrega_actividad",
    "consulta-proyecto": "consulta_proyecto",
    "tramite admin": "tramite_administrativo",
    "tramite administrativo": "tramite_administrativo",
    "consulta proyecto": "consulta_proyecto",
    "consulta de proyecto": "consulta_proyecto",
    "entrega de actividad": "entrega_actividad",
}

# Ajuste esperado 2: ampliar este catálogo con entidades reales observadas en el dataset.
CATALOGO_ENTIDADES = {
    "archivo": ["pdf", "csv", "docx", "zip", "archivo", "documento", "adjunto"],
    "plataforma": ["classroom", "teams", "moodle", "canvas", "zoom", "google classroom"],
    "concepto": ["intencion", "entidades", "ambiguedad", "pipeline", "matriz de confusion", "normalizacion", "entrenamiento", "modelo", "clasificador"],
    "clase": ["clase 9", "clase 10", "clase 11", "clase 12", "practica 8", "practica 9", "clase12"],
    "proyecto": ["proyecto", "actividad", "tarea", "trabajo", "entrega"],
    "soporte": ["soporte", "ayuda", "problema", "error", "fallo", "bug"],
}


def limpiar(texto: str) -> str:
    texto = str(texto).lower().strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def cargar_dataset():
    df = pd.read_csv(RUTA)
    requeridas = {"id", "texto", "intencion"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {faltantes}")
    df["texto_limpio"] = df["texto"].map(limpiar)
    return df


def normalizar_intenciones(df):
    df = df.copy()
    df["intencion_original"] = df["intencion"]
    df["intencion_normalizada"] = df["intencion"].replace(NORMALIZACION_INTENCIONES)
    advertencias = df[df["intencion_original"] != df["intencion_normalizada"]][["id", "texto", "intencion_original", "intencion_normalizada"]]
    etiquetas_no_canonicas = sorted(set(df["intencion_normalizada"]) - {
        "consulta_horario", "entrega_actividad", "soporte_tecnico", "aclaracion_calificacion",
        "consulta_concepto", "saludo_cierre", "tramite_administrativo", "consulta_proyecto"
    })
    return df, advertencias, etiquetas_no_canonicas


def extraer_entidades(texto):
    texto_limpio = limpiar(texto)
    encontradas = []
    for tipo, patrones in CATALOGO_ENTIDADES.items():
        for patron in patrones:
            if patron in texto_limpio:
                encontradas.append(f"{tipo}:{patron}")
    return " | ".join(encontradas) if encontradas else "SIN_ENTIDAD"


def entrenar_y_evaluar(df, columna_etiqueta, prefijo):
    conteos = df[columna_etiqueta].value_counts()
    stratify = df[columna_etiqueta] if (conteos.min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto_limpio"], df[columna_etiqueta],
        test_size=0.30, random_state=42, stratify=stratify
    )
    modelo = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, pred)
    reporte = classification_report(y_test, pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, pred, labels=sorted(df[columna_etiqueta].unique()))
    pd.DataFrame(reporte).transpose().to_csv(SALIDA / f"{prefijo}_reporte_metricas.csv", encoding="utf-8-sig")
    pd.DataFrame(matriz, index=sorted(df[columna_etiqueta].unique()), columns=sorted(df[columna_etiqueta].unique())).to_csv(SALIDA / f"{prefijo}_matriz_confusion.csv", encoding="utf-8-sig")
    resultados = pd.DataFrame({"texto": X_test, "real": y_test, "prediccion": pred})
    resultados["correcto"] = resultados["real"] == resultados["prediccion"]
    resultados.to_csv(SALIDA / f"{prefijo}_predicciones.csv", index=False, encoding="utf-8-sig")
    return acc, resultados


def main():
    df = cargar_dataset()
    df["entidades_detectadas"] = df["texto"].map(extraer_entidades)
    df.to_csv(SALIDA / "01_dataset_validado_con_entidades.csv", index=False, encoding="utf-8-sig")

    df_norm, advertencias, etiquetas_no_canonicas = normalizar_intenciones(df)
    advertencias.to_csv(SALIDA / "02_advertencias_etiquetas_normalizadas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"etiqueta_no_canonica": etiquetas_no_canonicas}).to_csv(SALIDA / "03_etiquetas_no_canonicas.csv", index=False, encoding="utf-8-sig")

    acc_original, pred_original = entrenar_y_evaluar(df, "intencion", "04_antes_ajuste")
    acc_normalizada, pred_normalizada = entrenar_y_evaluar(df_norm, "intencion_normalizada", "05_despues_ajuste")

    resumen = pd.DataFrame([
        {"escenario": "antes_ajuste", "accuracy": acc_original, "clases": df["intencion"].nunique()},
        {"escenario": "despues_ajuste", "accuracy": acc_normalizada, "clases": df_norm["intencion_normalizada"].nunique()},
    ])
    resumen.to_csv(SALIDA / "06_comparacion_antes_despues.csv", index=False, encoding="utf-8-sig")

    sin_entidad = df[df["entidades_detectadas"] == "SIN_ENTIDAD"][["id", "texto", "intencion", "entidades_detectadas"]]
    sin_entidad.to_csv(SALIDA / "07_casos_sin_entidad_detectada.csv", index=False, encoding="utf-8-sig")

    print("Proceso terminado")
    print(resumen)
    print("Etiquetas no canónicas detectadas:", etiquetas_no_canonicas)
    print("Casos sin entidad detectada:", len(sin_entidad))

if __name__ == "__main__":
    main()
