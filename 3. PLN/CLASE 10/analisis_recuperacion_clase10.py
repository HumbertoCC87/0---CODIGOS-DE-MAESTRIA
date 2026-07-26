# Clase 10 PLN · Recuperación semántica con vectores y ajuste de equivalencias
# Autor: M.C. Pablo Ricardo Sánchez Gómez
# Requisito: ejecutar en ambiente Conda con Python 3.11.

from pathlib import Path
import re
import unicodedata
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos"
SALIDAS = BASE / "salidas_clase10"
SALIDAS.mkdir(exist_ok=True)

DOCUMENTOS = DATOS / "documentos_clase10.csv"
CONSULTAS = DATOS / "consultas_clase10.csv"
K = 5

# Caso técnico integrado:
# El diccionario inicia incompleto de forma deliberada.
# El ajuste consiste en ampliar equivalencias justificadas por dominio y comparar antes/después.
EQUIVALENCIAS_INICIALES = {
    "alumno": ["estudiante"],
    "estudiantes": ["alumnos"],
    "ia": ["inteligencia artificial"],
    "seguridad informatica": ["ciberseguridad"],
    "cloud": ["nube"],
}

# Complete o amplíe este diccionario después de revisar 03_casos_baja_recuperacion.csv.
EQUIVALENCIAS_AJUSTADAS = {
    **EQUIVALENCIAS_INICIALES,
    "apoyo economico": ["beca", "becas"],
    "prestamo": ["credito"],
    "negocios": ["empresas"],
    "bolsa de trabajo": ["vacantes", "empleo"],
    "inmunizacion": ["vacunacion", "vacuna"],
    "historial clinico": ["expediente clinico"],
    "servicio web": ["api"],
    "delito": ["incidente"],
    "vulnerabilidades": ["riesgo"],
    "camara": ["vigilancia"],
    "competencia": ["torneo"],
    "resultado del encuentro": ["marcador"],
    "pruebas fisicas": ["rendimiento deportivo"],
}


def normalizar(texto: str) -> str:
    texto = str(texto).lower().strip()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z0-9ñ\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


def expandir_texto(texto: str, equivalencias: dict) -> str:
    base = normalizar(texto)
    agregado = []
    for termino, equivalentes in equivalencias.items():
        termino_norm = normalizar(termino)
        if re.search(r'\b' + re.escape(termino_norm) + r'\b', base):
            agregado.extend(normalizar(eq) for eq in equivalentes)
    return (base + " " + " ".join(agregado)).strip()


def cargar_datos():
    docs = pd.read_csv(DOCUMENTOS)
    qs = pd.read_csv(CONSULTAS)
    requeridos_docs = {"id", "texto", "dominio", "tema", "relevancia_grupo"}
    requeridos_q = {"id_consulta", "consulta", "dominio_esperado", "relevancia_esperada"}
    if not requeridos_docs.issubset(docs.columns):
        raise ValueError(f"Faltan columnas en documentos: {requeridos_docs - set(docs.columns)}")
    if not requeridos_q.issubset(qs.columns):
        raise ValueError(f"Faltan columnas en consultas: {requeridos_q - set(qs.columns)}")
    docs["texto_limpio"] = docs["texto"].map(normalizar)
    qs["consulta_limpia"] = qs["consulta"].map(normalizar)
    return docs, qs


def generar_ranking(docs: pd.DataFrame, consultas: pd.DataFrame, equivalencias: dict, etiqueta: str):
    docs = docs.copy()
    consultas = consultas.copy()
    docs["texto_ajustado"] = docs["texto"].map(lambda t: expandir_texto(t, equivalencias))
    consultas["consulta_ajustada"] = consultas["consulta"].map(lambda t: expandir_texto(t, equivalencias))

    vectorizador = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_docs = vectorizador.fit_transform(docs["texto_ajustado"])
    X_q = vectorizador.transform(consultas["consulta_ajustada"])

    filas = []
    resumen = []
    for qi, q in consultas.iterrows():
        scores = cosine_similarity(X_q[qi], X_docs).ravel()
        orden = np.argsort(scores)[::-1]
        top = orden[:K]
        relevante_total = int((docs["relevancia_grupo"] == q["relevancia_esperada"]).sum())
        relevantes_top = 0
        primer_relevante = None
        for rank, idx in enumerate(top, start=1):
            d = docs.iloc[idx]
            es_rel = d["relevancia_grupo"] == q["relevancia_esperada"]
            if es_rel:
                relevantes_top += 1
                if primer_relevante is None:
                    primer_relevante = rank
            filas.append({
                "escenario": etiqueta,
                "id_consulta": q["id_consulta"],
                "consulta": q["consulta"],
                "consulta_ajustada": q["consulta_ajustada"],
                "dominio_esperado": q["dominio_esperado"],
                "relevancia_esperada": q["relevancia_esperada"],
                "rank": rank,
                "id_documento": d["id"],
                "texto_documento": d["texto"],
                "dominio_documento": d["dominio"],
                "relevancia_documento": d["relevancia_grupo"],
                "score_coseno": round(float(scores[idx]), 4),
                "es_relevante": bool(es_rel),
            })
        precision = relevantes_top / K
        recall = relevantes_top / relevante_total if relevante_total else 0
        mrr = 1 / primer_relevante if primer_relevante else 0
        resumen.append({
            "escenario": etiqueta,
            "id_consulta": q["id_consulta"],
            "consulta": q["consulta"],
            "relevancia_esperada": q["relevancia_esperada"],
            "precision_at_5": round(precision, 4),
            "recall_at_5": round(recall, 4),
            "mrr": round(mrr, 4),
            "primer_relevante_en_top5": primer_relevante if primer_relevante else "NO_TOP5",
        })
    return pd.DataFrame(filas), pd.DataFrame(resumen)


def detectar_baja_recuperacion(resumen_sin: pd.DataFrame):
    casos = resumen_sin[(resumen_sin["primer_relevante_en_top5"] == "NO_TOP5") | (resumen_sin["precision_at_5"] < 0.2)].copy()
    if casos.empty:
        casos = resumen_sin.sort_values(["precision_at_5", "mrr"]).head(8).copy()
    casos["diagnostico"] = "Revisar vocabulario de consulta, dominio y equivalencias no registradas."
    casos["accion_sugerida"] = "Agregar equivalencias justificadas y comparar antes/despues."
    return casos


def comparar(res_sin: pd.DataFrame, res_con: pd.DataFrame):
    cols = ["id_consulta", "consulta", "relevancia_esperada", "precision_at_5", "recall_at_5", "mrr", "primer_relevante_en_top5"]
    a = res_sin[cols].rename(columns={c: f"sin_{c}" for c in cols if c not in ["id_consulta", "consulta", "relevancia_esperada"]})
    b = res_con[cols].rename(columns={c: f"con_{c}" for c in cols if c not in ["id_consulta", "consulta", "relevancia_esperada"]})
    df = a.merge(b, on=["id_consulta", "consulta", "relevancia_esperada"])
    df["delta_precision"] = (df["con_precision_at_5"] - df["sin_precision_at_5"]).round(4)
    df["delta_recall"] = (df["con_recall_at_5"] - df["sin_recall_at_5"]).round(4)
    df["delta_mrr"] = (df["con_mrr"] - df["sin_mrr"]).round(4)
    df["observacion"] = df.apply(lambda r: "mejora" if (r["delta_mrr"] > 0 or r["delta_precision"] > 0) else "sin_mejora_o_revisar_distractores", axis=1)
    return df


def main():
    print("Clase 10 PLN · Recuperación semántica")
    print("Ambiente Conda:", __import__('os').environ.get('CONDA_DEFAULT_ENV', 'NO_DETECTADO'))
    docs, consultas = cargar_datos()
    docs.to_csv(SALIDAS / "01_dataset_validado.csv", index=False, encoding="utf-8-sig")

    ranking_sin, resumen_sin = generar_ranking(docs, consultas, EQUIVALENCIAS_INICIALES, "sin_ajuste")
    ranking_sin.to_csv(SALIDAS / "02_resultados_sin_ajuste.csv", index=False, encoding="utf-8-sig")
    baja = detectar_baja_recuperacion(resumen_sin)
    baja.to_csv(SALIDAS / "03_casos_baja_recuperacion.csv", index=False, encoding="utf-8-sig")

    ranking_con, resumen_con = generar_ranking(docs, consultas, EQUIVALENCIAS_AJUSTADAS, "con_ajuste")
    ranking_con.to_csv(SALIDAS / "04_resultados_con_ajuste.csv", index=False, encoding="utf-8-sig")
    comp = comparar(resumen_sin, resumen_con)
    comp.to_csv(SALIDAS / "05_comparacion_antes_despues.csv", index=False, encoding="utf-8-sig")

    resumen = pd.concat([resumen_sin, resumen_con], ignore_index=True)
    resumen.to_csv(SALIDAS / "06_resumen_metricas.csv", index=False, encoding="utf-8-sig")

    print("Documentos procesados:", len(docs))
    print("Consultas procesadas:", len(consultas))
    print("Archivos generados en:", SALIDAS)
    print("Revise 03_casos_baja_recuperacion.csv y 05_comparacion_antes_despues.csv")

if __name__ == "__main__":
    main()
