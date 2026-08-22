"""
EXAMEN PRACTICO CORTE 3 - PROCESAMIENTO DE LENGUAJE NATURAL
Esqueleto de trabajo. Complete las secciones marcadas.
M.C. Pablo Ricardo Sanchez Gomez
"""
from pathlib import Path
import json
import re
import unicodedata
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent
DATASET = BASE / "Dataset_Examen_Practico_PLN_Corte3.csv"
CONSULTAS = BASE / "Consultas_Examen_Practico_PLN_Corte3.csv"
DICCIONARIO = BASE / "Diccionario_Entidades_Inicial_Corte3.json"
SALIDAS = BASE / "salidas_examen_corte3"
SALIDAS.mkdir(exist_ok=True)

RANDOM_STATE = 31
TEST_SIZE = 0.25
UMBRAL_CONFIANZA = 0.60


def normalizar_texto(texto):
    texto = str(texto).strip().lower()
    texto = "".join(c for c in unicodedata.normalize("NFD", texto)
                    if unicodedata.category(c) != "Mn")
    texto = re.sub(r"\s+", " ", texto)
    texto = texto.replace("-", " ")
    return texto


def cargar_recursos():
    """Carga dataset, consultas y diccionario inicial."""
    df = pd.read_csv(DATASET)
    consultas = pd.read_csv(CONSULTAS)
    diccionario = json.loads(DICCIONARIO.read_text(encoding="utf-8"))
    return df, consultas, diccionario


def auditar_dataset(df):
    """
    Debe producir un resumen verificable con al menos:
    - numero de registros y columnas;
    - nulos;
    - duplicados de texto;
    - distribucion por dominio;
    - catalogo y frecuencia de intenciones.
    Guarde el resultado en CSV.
    """
    resumen = []
    resumen.append({"metrica": "numero_registros", "detalle": "total", "valor": int(len(df))})
    resumen.append({"metrica": "numero_columnas", "detalle": "total", "valor": int(len(df.columns))})
    resumen.append({"metrica": "nulos", "detalle": "total", "valor": int(df.isna().sum().sum())})
    for columna in df.columns:
        resumen.append({"metrica": "nulos_por_columna", "detalle": columna, "valor": int(df[columna].isna().sum())})
    resumen.append({"metrica": "duplicados_texto", "detalle": "texto", "valor": int(df["texto"].duplicated().sum())})
    for dominio, frecuencia in df["dominio"].value_counts().sort_index().items():
        resumen.append({"metrica": "distribucion_dominio", "detalle": dominio, "valor": int(frecuencia)})
    for etiqueta, frecuencia in df["etiqueta"].value_counts().sort_index().items():
        resumen.append({"metrica": "catalogo_etiquetas", "detalle": etiqueta, "valor": int(frecuencia)})

    auditoria_df = pd.DataFrame(resumen)
    auditoria_df.to_csv(SALIDAS / "01_auditoria_dataset.csv", index=False)
    return auditoria_df


def ajustar_datos(df):
    """
    Revise los resultados de la auditoria y aplique solamente ajustes
    tecnicamente demostrables. Cualquier cambio debe conservar:
    id, valor_original, valor_final y justificacion.
    No elimine clases o registros validos para mejorar metricas.
    """
    trazabilidad = []
    df_corr = df.copy()

    for idx, fila in df_corr.iterrows():
        valor_original = str(fila["etiqueta"])
        valor_final = valor_original
        if "-" in valor_original:
            valor_final = valor_original.replace("-", "_")
            df_corr.at[idx, "etiqueta"] = valor_final
            trazabilidad.append({
                "id": int(fila["id"]),
                "valor_original": valor_original,
                "valor_final": valor_final,
                "justificacion": "Normalización de variantes de la misma intención: se unifican etiquetas con guion medio y subrayado para evitar inconsistencia en la clasificación supervisada."
            })

    trazabilidad_df = pd.DataFrame(trazabilidad, columns=["id", "valor_original", "valor_final", "justificacion"])
    if not trazabilidad_df.empty:
        trazabilidad_df.to_csv(SALIDAS / "05_trazabilidad_ajustes.csv", index=False)
    return df_corr, trazabilidad_df


def entrenar_clasificador(df, etiqueta_salida):
    """
    Construya un pipeline reproducible:
    1) train_test_split con stratify y RANDOM_STATE;
    2) TF-IDF dentro del Pipeline;
    3) LogisticRegression o un clasificador lineal equivalente;
    4) predict sobre prueba;
    5) reporte, matriz y predicciones con confianza.
    Guarde archivos antes/despues segun etiqueta_salida.
    """
    X = df["texto"].astype(str).fillna("")
    y = df["etiqueta"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    clases = sorted(y.unique())
    matriz = confusion_matrix(y_test, y_pred, labels=clases)
    matriz_df = pd.DataFrame(matriz, index=clases, columns=clases)

    if etiqueta_salida == "antes":
        report_path = SALIDAS / "02_reporte_clasificacion_antes.csv"
        matriz_path = SALIDAS / "03_matriz_confusion_antes.csv"
        pred_path = SALIDAS / "04_predicciones_antes.csv"
    elif etiqueta_salida == "despues":
        report_path = SALIDAS / "06_reporte_clasificacion_despues.csv"
        matriz_path = SALIDAS / "07_matriz_confusion_despues.csv"
        pred_path = SALIDAS / "08_predicciones_despues.csv"
    else:
        report_path = SALIDAS / f"{etiqueta_salida}_reporte_clasificacion.csv"
        matriz_path = SALIDAS / f"{etiqueta_salida}_matriz_confusion.csv"
        pred_path = SALIDAS / f"{etiqueta_salida}_predicciones.csv"

    report_df.to_csv(report_path)
    matriz_df.to_csv(matriz_path)

    predicciones = []
    for idx, (texto, etiqueta_real, pred, probs) in enumerate(zip(X_test, y_test, y_pred, probabilities)):
        predicciones.append({
            "indice_prueba": idx,
            "texto": texto,
            "etiqueta_real": etiqueta_real,
            "prediccion": pred,
            "correcto": etiqueta_real == pred,
            "confianza": round(float(max(probs)), 6)
        })
    pd.DataFrame(predicciones).to_csv(pred_path, index=False)

    return {
        "accuracy": accuracy,
        "report": report_df,
        "matriz": matriz_df,
        "predicciones": pd.DataFrame(predicciones),
        "pipeline": pipeline
    }


def extraer_entidades(texto, diccionario):
    """
    Extraiga entidades mediante el diccionario.
    La funcion debe devolver categoria, valor detectado y evidencia textual.
    """
    texto_norm = normalizar_texto(texto)
    resultados = []
    for categoria, valores in diccionario.items():
        for valor in valores:
            valor_norm = normalizar_texto(valor)
            if valor_norm and valor_norm in texto_norm:
                resultados.append({
                    "categoria": categoria,
                    "valor": valor,
                    "evidencia": valor
                })
    return resultados


def decidir_respuesta(intencion, entidades, confianza):
    """
    Implemente una regla operativa:
    - confianza por debajo del umbral -> revision/pedir aclaracion;
    - intencion clara pero entidad necesaria ausente -> solicitar dato;
    - intencion y entidades suficientes -> accion/respuesta coherente.
    """
    if confianza < UMBRAL_CONFIANZA:
        return "Revisión requerida: la intención no tiene suficiente confianza para actuar. Solicite aclaración al usuario."

    requeridas = {
        "consulta_informacion": ["tema", "apoyo", "programa"],
        "tramite": ["materia", "periodo", "registro"],
        "cancelacion": ["servicio", "fecha"],
        "pago": ["monto", "concepto"],
        "urgencia": ["persona", "riesgo"],
        "soporte_tecnico": ["cuenta", "acceso"],
    }

    entidades_detectadas = {item["categoria"] for item in entidades}
    if intencion in requeridas:
        faltantes = [e for e in requeridas[intencion] if e not in entidades_detectadas]
        if faltantes:
            return f"Falta información clave para atender la intención '{intencion}': {', '.join(faltantes)}. Solicite ese dato."

    if intencion == "consulta_informacion":
        return "Se puede responder con información institucional, requisitos y fechas del programa solicitado."
    if intencion == "tramite":
        return "Procede a orientar el trámite o registro relacionado con las materias o inscripciones indicadas."
    if intencion == "cancelacion":
        return "Se recomienda revisar la cancelación del servicio y verificar la fecha de cobro o renovación."
    if intencion == "pago":
        return "Se puede verificar el pago, comprobante y estado de la operación indicada."
    if intencion == "urgencia":
        return "Se recomienda activar la atención médica inmediata y coordinar con personal de salud."
    if intencion == "soporte_tecnico":
        return "Se puede orientar la recuperación de acceso o soporte técnico del sistema."
    return "La intención es clara y se dispone del contexto suficiente para responder con un procedimiento operativo."


def recuperar_topk(df, consultas, k=3, equivalencias=None, etiqueta_salida="base"):
    """
    Debe:
    - representar documentos y consultas con TF-IDF;
    - calcular similitud coseno;
    - generar top-k por consulta;
    - guardar consulta_id, posicion, id_documento, score,
      intencion_relevante, intencion_documento y es_relevante;
    - calcular Precision@k por consulta.
    equivalencias puede ampliar la consulta con relaciones lexicales justificadas.
    """
    if equivalencias is None:
        equivalencias = {}

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98
    )
    documentos = df["texto"].astype(str).fillna("").tolist()
    matriz_documentos = vectorizer.fit_transform(documentos)

    ranking_total = []
    precision_total = []

    for _, fila in consultas.iterrows():
        consulta = str(fila["consulta"])
        consulta_normalizada = normalizar_texto(consulta)
        if equivalencias:
            extras = []
            for clave, valor in equivalencias.items():
                if normalizar_texto(clave) in consulta_normalizada or any(palabra in consulta_normalizada for palabra in normalizar_texto(clave).split()):
                    extras.extend(str(valor).split())
            consulta_enriquecida = consulta_normalizada + " " + " ".join(extras)
        else:
            consulta_enriquecida = consulta_normalizada

        consulta_vec = vectorizer.transform([consulta_enriquecida])
        similitudes = cosine_similarity(consulta_vec, matriz_documentos).ravel()
        indices_top = similitudes.argsort()[-k:][::-1]

        relevantes_topk = 0
        for posicion, idx in enumerate(indices_top, start=1):
            doc = df.iloc[idx]
            grupo_esperado = fila["grupo_relevancia_esperado"]
            grupo_documento = doc["grupo_relevancia"]
            es_relevante = grupo_documento == grupo_esperado
            if es_relevante:
                relevantes_topk += 1
            ranking_total.append({
                "consulta_id": fila["consulta_id"],
                "posicion": posicion,
                "id_documento": int(doc["id"]),
                "score": round(float(similitudes[idx]), 6),
                "intencion_relevante": grupo_esperado,
                "intencion_documento": grupo_documento,
                "es_relevante": bool(es_relevante)
            })

        precision = relevantes_topk / k
        precision_total.append({
            "consulta_id": fila["consulta_id"],
            "precision@3": round(float(precision), 3),
            "relevantes_en_top3": relevantes_topk,
            "grupo_esperado": grupo_esperado
        })

    ranking_df = pd.DataFrame(ranking_total)
    precision_df = pd.DataFrame(precision_total)

    if etiqueta_salida == "base":
        ranking_path = SALIDAS / "09_ranking_recuperacion_base.csv"
        precision_path = SALIDAS / "10_precision3_base.csv"
    elif etiqueta_salida == "ajustado":
        ranking_path = SALIDAS / "11_ranking_recuperacion_ajustado.csv"
        precision_path = SALIDAS / "12_precision3_ajustado.csv"
    else:
        ranking_path = SALIDAS / f"{etiqueta_salida}_ranking.csv"
        precision_path = SALIDAS / f"{etiqueta_salida}_precision3.csv"

    ranking_df.to_csv(ranking_path, index=False)
    precision_df.to_csv(precision_path, index=False)
    return ranking_df, precision_df


def analizar_casos(df, consultas, ranking_base, ranking_ajustado):
    """Analiza los 15 casos del examen, sin alterar el dataset."""
    casos = pd.read_csv(BASE / "Casos_Analisis_Examen_Practico_PLN_Corte3.csv")
    salida = []
    for _, caso in casos.iterrows():
        caso_id = caso["caso_id"]
        tipo = caso["tipo_analisis"]
        if tipo == "clasificacion":
            ref = int(caso["referencia"])
            fila = df[df["id"] == ref].iloc[0]
            salida.append({
                "caso_id": caso_id,
                "tipo": tipo,
                "id_documento": ref,
                "etiqueta_real": fila["etiqueta"],
                "dominio": fila["dominio"],
                "texto": fila["texto"]
            })
        elif tipo == "recuperacion":
            qid = caso["referencia"]
            base_q = ranking_base[ranking_base["consulta_id"] == qid].head(3).to_dict("records")
            ajuste_q = ranking_ajustado[ranking_ajustado["consulta_id"] == qid].head(3).to_dict("records")
            salida.append({
                "caso_id": caso_id,
                "tipo": tipo,
                "consulta_id": qid,
                "top3_base": base_q,
                "top3_ajustado": ajuste_q
            })
        else:
            salida.append({
                "caso_id": caso_id,
                "tipo": tipo,
                "referencia": caso["referencia"],
                "instruccion": caso["instruccion"],
                "termino_1": caso["termino_1"],
                "termino_2": caso["termino_2"]
            })
    return salida


def main():
    df, consultas, diccionario = cargar_recursos()

    print("[1/7] Auditoría inicial del dataset...")
    auditoria = auditar_dataset(df)

    print("[2/7] Ajuste de inconsistencias detectadas...")
    df_ajustado, trazabilidad = ajustar_datos(df)

    print("[3/7] Clasificación antes del ajuste...")
    resultado_antes = entrenar_clasificador(df, "antes")

    print("[4/7] Clasificación después del ajuste...")
    resultado_despues = entrenar_clasificador(df_ajustado, "despues")

    print("[5/7] Recuperación semántica base...")
    ranking_base, precision_base = recuperar_topk(df, consultas, k=3, etiqueta_salida="base")

    equivalencias = {
        "apoyo economico": "beca apoyo financiero",
        "apoyo financiero": "beca apoyo economico",
        "beca": "apoyo economico financiamiento",
        "cancelar": "cancelacion baja servicio",
        "servicio": "suscripcion cobro",
        "cuenta": "acceso login portal",
        "perfil": "datos personales",
    }
    print("[6/7] Recuperación semántica ajustada con equivalencias justificadas...")
    ranking_ajustado, precision_ajustado = recuperar_topk(df_ajustado, consultas, k=3, equivalencias=equivalencias, etiqueta_salida="ajustado")

    print("[7/7] Análisis de casos y validación de entidades...")
    analisis = analizar_casos(df_ajustado, consultas, ranking_base, ranking_ajustado)
    for caso in analisis[:3]:
        print(caso)

    print("\n=== Resumen ejecutivo ===")
    print(f"Accuracy antes: {resultado_antes['accuracy']:.4f}")
    print(f"Accuracy después: {resultado_despues['accuracy']:.4f}")
    print(f"Precision@3 base promedio: {precision_base['precision@3'].mean():.3f}")
    print(f"Precision@3 ajustado promedio: {precision_ajustado['precision@3'].mean():.3f}")
    print("\nProceso terminado. Revise:", SALIDAS)


if __name__ == "__main__":
    main()
