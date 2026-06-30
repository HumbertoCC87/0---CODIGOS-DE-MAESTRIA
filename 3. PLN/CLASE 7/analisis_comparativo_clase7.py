# Clase 7 - Procesamiento de Lenguaje Natural
# M.C. Pablo Ricardo Sánchez Gómez
# Análisis comparativo: reglas, PCFG y parser moderno

from pathlib import Path
import csv
import re

try:
    import nltk
    from nltk import CFG, PCFG
    from nltk.parse import ChartParser, ViterbiParser
except Exception as exc:
    raise SystemExit("Falta NLTK. Instale con: python -m pip install nltk") from exc

try:
    import spacy
except Exception:
    spacy = None

ENTRADA = Path("datos/oraciones_clase7.csv")
SALIDA = Path("salidas_clase7")
SALIDA.mkdir(exist_ok=True)

GRAMATICA_CFG = CFG.fromstring(r"""
O -> SN SV
SN -> Det N | Det N SP | NProp | SN SP
SV -> V SN | V SN SP | V SP | V OSub
SP -> P SN
OSub -> Conj O
Det -> 'el' | 'la' | 'los' | 'las' | 'un' | 'una'
N -> 'estudiante' | 'texto' | 'profesora' | 'evidencia' | 'parser' | 'arbol' | 'programa' | 'plataforma' | 'modelo' | 'noticias' | 'tecnologia' | 'errores' | 'telescopio' | 'profesor' | 'proyecto' | 'gobierno' | 'inversion' | 'hospitales' | 'alumnos' | 'analisis' | 'clase' | 'sistema' | 'inconsistencias' | 'archivo' | 'columnas' | 'interpretacion' | 'contexto'
NProp -> 'tfidf' | 'spacy'
V -> 'analiza' | 'revisa' | 'genera' | 'clasifico' | 'vi' | 'califico' | 'anuncio' | 'entregaron' | 'detecto' | 'tenia' | 'fallo' | 'era'
P -> 'con' | 'en' | 'de' | 'despues' | 'porque' | 'cuando'
Conj -> 'que'
""")

GRAMATICA_PCFG = PCFG.fromstring(r"""
O -> SN SV [1.0]
SN -> Det N [0.55] | Det N SP [0.15] | NProp [0.05] | SN SP [0.25]
SV -> V SN [0.45] | V SN SP [0.25] | V SP [0.15] | V OSub [0.15]
SP -> P SN [1.0]
OSub -> Conj O [1.0]
Det -> 'el' [0.25] | 'la' [0.25] | 'los' [0.15] | 'las' [0.10] | 'un' [0.15] | 'una' [0.10]
N -> 'estudiante' [0.05] | 'texto' [0.05] | 'profesora' [0.05] | 'evidencia' [0.05] | 'parser' [0.05] | 'arbol' [0.05] | 'programa' [0.05] | 'plataforma' [0.05] | 'modelo' [0.05] | 'noticias' [0.05] | 'tecnologia' [0.05] | 'errores' [0.05] | 'telescopio' [0.05] | 'profesor' [0.05] | 'proyecto' [0.05] | 'gobierno' [0.05] | 'inversion' [0.05] | 'hospitales' [0.05] | 'alumnos' [0.05] | 'analisis' [0.05]
NProp -> 'tfidf' [0.5] | 'spacy' [0.5]
V -> 'analiza' [0.11] | 'revisa' [0.11] | 'genera' [0.11] | 'clasifico' [0.11] | 'vi' [0.11] | 'califico' [0.11] | 'anuncio' [0.11] | 'entregaron' [0.11] | 'detecto' [0.06] | 'fallo' [0.06]
P -> 'con' [0.35] | 'en' [0.30] | 'de' [0.20] | 'despues' [0.10] | 'porque' [0.05]
Conj -> 'que' [1.0]
""")

def limpiar(texto: str) -> str:
    texto = texto.lower()
    reemplazos = str.maketrans("áéíóúüñ", "aeiouun")
    texto = texto.translate(reemplazos)
    texto = re.sub(r"[^a-z0-9áéíóúüñ\s-]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def leer_dataset(ruta: Path):
    with ruta.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            yield {"id": fila.get("id", ""), "texto": fila.get("texto", ""), "tipo": fila.get("tipo", "")}

def parse_cfg(tokens):
    parser = ChartParser(GRAMATICA_CFG)
    try:
        return list(parser.parse(tokens))
    except ValueError:
        return []

def parse_pcfg(tokens):
    parser = ViterbiParser(GRAMATICA_PCFG)
    try:
        return list(parser.parse(tokens))
    except ValueError:
        return []

def cargar_spacy():
    if spacy is None:
        return None
    try:
        return spacy.load("es_core_news_sm")
    except Exception:
        return None

def resumen_spacy(nlp, texto):
    if nlp is None:
        return {"raiz":"NO_DISPONIBLE", "sujeto":"", "objeto":"", "dependencias":"modelo spaCy no disponible"}
    doc = nlp(texto)
    raiz = next((t.text for t in doc if t.dep_ == "ROOT"), "")
    sujetos = [t.text for t in doc if t.dep_ in {"nsubj", "nsubj:pass"}]
    objetos = [t.text for t in doc if t.dep_ in {"obj", "iobj"}]
    deps = "; ".join([f"{t.head.text}->{t.text}:{t.dep_}" for t in doc])
    return {"raiz":raiz, "sujeto":"|".join(sujetos), "objeto":"|".join(objetos), "dependencias":deps}

def escribir_csv(nombre, filas, campos):
    with (SALIDA/nombre).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader(); w.writerows(filas)

nlp = cargar_spacy()
registros = list(leer_dataset(ENTRADA))
validado, cfg_rows, spacy_rows, comp_rows, amb_rows = [], [], [], [], []

for r in registros:
    limpio = limpiar(r["texto"])
    tokens = limpio.split()
    arboles_cfg = parse_cfg(tokens)
    arboles_pcfg = parse_pcfg(tokens)
    sp = resumen_spacy(nlp, r["texto"])
    valida = len(arboles_cfg) > 0
    posible_amb = len(arboles_cfg) > 1 or r["tipo"] == "ambigua" or any(p in tokens for p in ["con", "en", "de"])
    obs = "aceptada por reglas" if valida else "no aceptada por la CFG; revisar vocabulario o reglas"
    if len(arboles_cfg) > 1:
        obs = "genera varios árboles; posible ambigüedad estructural"

    validado.append({"id":r["id"],"texto_original":r["texto"],"texto_limpio":limpio,"tokens":" | ".join(tokens),"tipo":r["tipo"]})
    cfg_rows.append({"id":r["id"],"valida_cfg":valida,"num_parseos_cfg":len(arboles_cfg),"num_parseos_pcfg":len(arboles_pcfg),"observacion_cfg":obs})
    spacy_rows.append({"id":r["id"],"raiz_spacy":sp["raiz"],"sujeto_spacy":sp["sujeto"],"objeto_spacy":sp["objeto"],"dependencias_principales":sp["dependencias"]})
    enfoque = "comparar reglas y parser moderno" if posible_amb or not valida else "reglas suficientes para estructura básica"
    comp_rows.append({"id":r["id"],"tipo":r["tipo"],"valida_cfg":valida,"num_parseos_cfg":len(arboles_cfg),"raiz_spacy":sp["raiz"],"enfoque_recomendado":enfoque,"observacion":"revisar manualmente si hay preposición, múltiples complementos o rechazo por reglas"})
    if posible_amb:
        amb_rows.append({"id":r["id"],"texto":r["texto"],"motivo":"posible ambigüedad, sintagma preposicional o varios análisis","revision_sugerida":"comparar interpretación por reglas, spaCy y criterio humano"})

escribir_csv("01_dataset_validado.csv", validado, ["id","texto_original","texto_limpio","tokens","tipo"])
escribir_csv("02_resultado_cfg.csv", cfg_rows, ["id","valida_cfg","num_parseos_cfg","num_parseos_pcfg","observacion_cfg"])
escribir_csv("03_resultado_spacy.csv", spacy_rows, ["id","raiz_spacy","sujeto_spacy","objeto_spacy","dependencias_principales"])
escribir_csv("04_comparacion_reglas_vs_parser.csv", comp_rows, ["id","tipo","valida_cfg","num_parseos_cfg","raiz_spacy","enfoque_recomendado","observacion"])
escribir_csv("05_casos_ambiguedad.csv", amb_rows, ["id","texto","motivo","revision_sugerida"])

print("Proceso terminado. Revise la carpeta salidas_clase7.")
