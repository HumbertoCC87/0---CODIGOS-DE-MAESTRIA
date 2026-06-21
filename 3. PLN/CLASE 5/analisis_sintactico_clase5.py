"""
UPMH · Procesamiento de Lenguaje Natural · Clase 5
Análisis sintáctico con spaCy: POS tagging y dependencias
Autoría académica: M.C. Pablo Ricardo Sánchez Gómez

Objetivo:
Leer un archivo CSV con oraciones en español, procesarlas con spaCy y generar
salidas verificables sobre tokens, lemas, categorías gramaticales, dependencias,
cabezas sintácticas y posibles casos de ambigüedad.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


def cargar_modelo(nombre_modelo: str = "es_core_news_sm"):
    """Carga el modelo de spaCy indicado.

    Si el modelo no está instalado, muestra un mensaje claro con el comando
    necesario para instalarlo. Esto evita que el error parezca un problema
    del código cuando en realidad falta una dependencia lingüística.
    """
    try:
        import spacy
        return spacy.load(nombre_modelo)
    except OSError:
        print("No se encontró el modelo de español de spaCy:", nombre_modelo)
        print("Instálalo con el siguiente comando:")
        print(f"python -m spacy download {nombre_modelo}")
        sys.exit(1)
    except ImportError:
        print("No se encontró la librería spaCy.")
        print("Instálala con: python -m pip install spacy")
        sys.exit(1)


def limpiar_espacios(texto: str) -> str:
    """Normaliza espacios y saltos de línea sin alterar el contenido lingüístico."""
    texto = str(texto).replace("\n", " ").replace("\t", " ")
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def cargar_oraciones(ruta_csv: Path) -> pd.DataFrame:
    """Lee el CSV de oraciones y valida columnas mínimas."""
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {ruta_csv}")

    datos = pd.read_csv(ruta_csv, encoding="utf-8-sig")
    columnas_requeridas = {"id", "texto"}
    faltantes = columnas_requeridas - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {faltantes}")

    datos = datos.dropna(subset=["texto"]).copy()
    datos["texto"] = datos["texto"].apply(limpiar_espacios)
    datos = datos[datos["texto"].str.len() > 0].copy()
    return datos


def analizar_oraciones(datos: pd.DataFrame, nlp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Procesa cada oración y construye cuatro salidas tabulares.

    Salidas:
    1. tokens_df: análisis por token.
    2. resumen_df: resumen sintáctico por oración.
    3. dependencias_df: relaciones cabeza-dependiente.
    4. ambiguedad_df: candidatos de ambigüedad o revisión manual.
    """
    registros_tokens = []
    registros_resumen = []
    registros_dependencias = []
    registros_ambiguedad = []

    for _, fila in datos.iterrows():
        id_oracion = fila["id"]
        texto = fila["texto"]
        dominio = fila["dominio"] if "dominio" in fila else "sin_dominio"

        doc = nlp(texto)

        sujetos = []
        objetos = []
        complementos = []
        raices = []

        for token in doc:
            registros_tokens.append({
                "id_oracion": id_oracion,
                "dominio": dominio,
                "texto_oracion": texto,
                "indice_token": token.i,
                "token": token.text,
                "lema": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dependencia": token.dep_,
                "cabeza": token.head.text,
                "indice_cabeza": token.head.i,
                "es_stopword": token.is_stop,
                "es_alfabetico": token.is_alpha,
            })

            registros_dependencias.append({
                "id_oracion": id_oracion,
                "token_dependiente": token.text,
                "dependencia": token.dep_,
                "token_cabeza": token.head.text,
                "pos_dependiente": token.pos_,
                "pos_cabeza": token.head.pos_,
            })

            if token.dep_ == "ROOT":
                raices.append(token.text)
            if token.dep_ in {"nsubj", "nsubj:pass"}:
                sujetos.append(token.text)
            if token.dep_ in {"obj", "iobj"}:
                objetos.append(token.text)
            if token.dep_ in {"obl", "advmod", "amod", "nmod"}:
                complementos.append(token.text)

        registros_resumen.append({
            "id_oracion": id_oracion,
            "dominio": dominio,
            "texto": texto,
            "raiz": ", ".join(raices) if raices else "sin_raiz_detectada",
            "sujetos": ", ".join(sujetos) if sujetos else "sin_sujeto_detectado",
            "objetos": ", ".join(objetos) if objetos else "sin_objeto_detectado",
            "complementos": ", ".join(complementos) if complementos else "sin_complementos_detectados",
            "total_tokens": len(doc),
        })

        texto_lower = texto.lower()
        tiene_preposicion_ambigua = any(palabra in texto_lower for palabra in [" con ", " en ", " de ", " para "])
        tiene_varios_complementos = len(complementos) >= 2
        if tiene_preposicion_ambigua or tiene_varios_complementos:
            registros_ambiguedad.append({
                "id_oracion": id_oracion,
                "texto": texto,
                "criterio_detectado": "preposicion_o_complementos",
                "motivo": "La oración puede requerir revisión manual por posible adjunción ambigua o múltiples complementos.",
                "pregunta_guia": "¿A qué palabra modifica el complemento y qué contexto ayudaría a decidirlo?",
            })

    tokens_df = pd.DataFrame(registros_tokens)
    resumen_df = pd.DataFrame(registros_resumen)
    dependencias_df = pd.DataFrame(registros_dependencias)
    ambiguedad_df = pd.DataFrame(registros_ambiguedad)
    return tokens_df, resumen_df, dependencias_df, ambiguedad_df


def guardar_salidas(output_dir: Path, tokens_df: pd.DataFrame, resumen_df: pd.DataFrame, dependencias_df: pd.DataFrame, ambiguedad_df: pd.DataFrame) -> None:
    """Guarda las tablas generadas en archivos CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_df.to_csv(output_dir / "01_tokens_analizados.csv", index=False, encoding="utf-8-sig")
    resumen_df.to_csv(output_dir / "02_resumen_oraciones.csv", index=False, encoding="utf-8-sig")
    dependencias_df.to_csv(output_dir / "03_dependencias.csv", index=False, encoding="utf-8-sig")
    ambiguedad_df.to_csv(output_dir / "04_candidatos_ambiguedad.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis sintáctico de oraciones en español con spaCy.")
    parser.add_argument("--input", default="datos/oraciones_clase5.csv", help="Ruta del CSV de entrada.")
    parser.add_argument("--output-dir", default="salidas_clase5", help="Carpeta donde se guardarán las salidas.")
    parser.add_argument("--modelo", default="es_core_news_sm", help="Modelo de spaCy para español.")
    args = parser.parse_args()

    ruta_csv = Path(args.input)
    output_dir = Path(args.output_dir)

    nlp = cargar_modelo(args.modelo)
    datos = cargar_oraciones(ruta_csv)
    tokens_df, resumen_df, dependencias_df, ambiguedad_df = analizar_oraciones(datos, nlp)
    guardar_salidas(output_dir, tokens_df, resumen_df, dependencias_df, ambiguedad_df)

    print("Análisis sintáctico finalizado.")
    print(f"Oraciones procesadas: {len(datos)}")
    print(f"Tokens analizados: {len(tokens_df)}")
    print(f"Candidatos de ambigüedad: {len(ambiguedad_df)}")
    print(f"Salidas generadas en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
