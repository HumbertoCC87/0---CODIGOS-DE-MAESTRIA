#Parser_cfg_clase6 modificado con indicaciones de clase 6
#Se modifica el código original para agregar una nueva regla sintáctica y dos nuevas palabras al léxico de la gramática, con el fin de ampliar la capacidad de parseo y permitir estructuras oracionales más complejas. Además, se mantiene la estructura general del programa para asegurar la compatibilidad con los datos de entrada y las salidas esperadas.
# UPMH - Procesamiento de Lenguaje Natural
# Clase 6: gramáticas libres de contexto, reglas y parsing básico
# Autoría: M.C. Pablo Ricardo Sánchez Gómez

# ==============================================================================
# ENCABEZADO DE MODIFICACIÓN:
# Se ha modificado la gramática original agregando lo siguiente:
# 1. Una nueva regla sintáctica: SV -> V SN Adv (Permite estructurar oraciones como "analiza el texto rapidamente")
# 2. Dos nuevas palabras (terminales): 'computadora' (Sustantivo) e 'inteligente' (Adjetivo)
# ==============================================================================

from pathlib import Path
import csv
import re
import unicodedata
import pandas as pd
import nltk

ARCHIVO_ENTRADA = Path("datos/oraciones_clase6.csv")
CARPETA_SALIDAS = Path("salidas_clase6")
CARPETA_SALIDAS.mkdir(exist_ok=True)


def quitar_acentos(texto: str) -> str:
    """Elimina marcas diacríticas para que 'práctica' y 'practica' sean equivalentes."""
    normalizado = unicodedata.normalize("NFD", texto)
    return "".join(c for c in normalizado if unicodedata.category(c) != "Mn")


def normalizar_texto(texto: str) -> str:
    """Convierte texto a minúsculas, elimina acentos, signos y espacios extra."""
    texto = quitar_acentos(str(texto).lower())
    texto = texto.replace(" al ", " a el ")
    texto = re.sub(r"[^a-zñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto_normalizado: str) -> list[str]:
    """Divide la oración normalizada en tokens compatibles con la gramática."""
    return texto_normalizado.split()


def crear_gramatica() -> nltk.CFG:
    """Define una gramática libre de contexto pequeña para oraciones simples en español."""
    gramatica_texto = r"""
        S -> SN SV

        SN -> Det N
        SN -> Det Adj N
        SN -> Det N Adj
        SN -> NProp
        SN -> SN SP

        SV -> V
        SV -> V SN
        SV -> V SP
        SV -> V SN SP
        SV -> V Adv
        # REGLA AGREGADA ---
        SV -> V SN Adv
        # --- FIN BLOQUE DE LAS REGLAS ---

        SP -> Prep SN

        Det -> 'el' | 'la' | 'los' | 'las' | 'un' | 'una'
        N -> 'profesor' | 'tema' | 'plataforma' | 'archivo' | 'estudiantes' | 'proyecto'
        N -> 'clase' | 'sistema' | 'errores' | 'reglas' | 'alumna' | 'estudiante'
        N -> 'telescopio' | 'modelo' | 'noticias' | 'docente' | 'evidencia' | 'equipo'
        N -> 'datos' | 'chatbot' | 'preguntas' | 'universidad' | 'algoritmo' | 'texto'
        N -> 'español' | 'habilidades' | 'modelos' | 'metricas' | 'docentes'
        N -> 'conceptos' | 'aplicacion' | 'reportes' | 'patrones' | 'alumno'
        # PALABRA AGREGADA ---
        N -> 'computadora'
        
        Adj -> 'dificil' | 'frecuentes' | 'practica' | 'complejos' | 'linguisticos' | 'nuevas'
        # PALABRA AGREGADA ---
        Adj -> 'inteligente'
        # --- FIN BLOQUE DE LAS PALABRAS ---
        
        V -> 'explica' | 'rechaza' | 'revisan' | 'detecta' | 'observa' | 'clasifica'
        V -> 'revisa' | 'entrega' | 'analiza' | 'responde' | 'publica' | 'procesa'
        V -> 'mejora' | 'compara' | 'explican' | 'genera' | 'aprende' | 'depura'
        Adv -> 'rapidamente' | 'automaticamente'
        Prep -> 'en' | 'con' | 'a'
        NProp -> 'pandas'
    """
    return nltk.CFG.fromstring(gramatica_texto)


def obtener_lexico(gramatica: nltk.CFG) -> set[str]:
    """Extrae terminales definidos en la gramática."""
    lexico = set()
    for produccion in gramatica.productions():
        for simbolo in produccion.rhs():
            if isinstance(simbolo, str):
                lexico.add(simbolo)
    return lexico


def detectar_fuera_gramatica(tokens: list[str], lexico: set[str]) -> list[str]:
    """Detecta tokens que no están definidos como terminales de la gramática."""
    return [token for token in tokens if token not in lexico]


def parsear_oracion(tokens: list[str], parser: nltk.ChartParser) -> list[nltk.Tree]:
    """Aplica el parser y devuelve todos los árboles encontrados."""
    return list(parser.parse(tokens))


def cargar_oraciones(ruta: Path) -> pd.DataFrame:
    """Carga el CSV y valida columnas mínimas."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {ruta}")
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    columnas = set(df.columns)
    requeridas = {"id", "texto", "tipo_esperado"}
    faltantes = requeridas - columnas
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {faltantes}")
    return df


def main() -> None:
    gramatica = crear_gramatica()
    parser = nltk.ChartParser(gramatica)
    lexico = obtener_lexico(gramatica)
    df = cargar_oraciones(ARCHIVO_ENTRADA)

    resumen = []
    fuera_gramatica = []
    ambiguas = []
    arboles_txt = []

    for _, fila in df.iterrows():
        id_oracion = fila["id"]
        texto_original = fila["texto"]
        tipo_esperado = fila["tipo_esperado"]
        texto_limpio = normalizar_texto(texto_original)
        tokens = tokenizar(texto_limpio)
        tokens_fuera = detectar_fuera_gramatica(tokens, lexico)

        if tokens_fuera:
            arboles = []
            observacion = "fuera_de_gramatica_por_lexico"
            for token in tokens_fuera:
                fuera_gramatica.append({
                    "id": id_oracion,
                    "texto": texto_original,
                    "token_fuera": token,
                    "observacion": "token_no_definido_en_lexico"
                })
        else:
            arboles = parsear_oracion(tokens, parser)
            if len(arboles) == 0:
                observacion = "sin_parseo_por_regla_faltante"
            elif len(arboles) == 1:
                observacion = "parseo_unico"
            else:
                observacion = "ambigua_multiples_parseos"
                ambiguas.append({
                    "id": id_oracion,
                    "texto": texto_original,
                    "tokens": " ".join(tokens),
                    "num_parseos": len(arboles),
                    "observacion": "la_gramatica_genero_mas_de_un_arbol"
                })

        resumen.append({
            "id": id_oracion,
            "texto_original": texto_original,
            "texto_limpio": texto_limpio,
            "tokens": " ".join(tokens),
            "tipo_esperado": tipo_esperado,
            "valida": "si" if len(arboles) > 0 else "no",
            "num_parseos": len(arboles),
            "observacion": observacion
        })

        arboles_txt.append("=" * 90)
        arboles_txt.append(f"ID: {id_oracion}")
        arboles_txt.append(f"Texto: {texto_original}")
        arboles_txt.append(f"Tokens: {' '.join(tokens)}")
        arboles_txt.append(f"Parseos encontrados: {len(arboles)}")
        for i, arbol in enumerate(arboles[:3], start=1):
            arboles_txt.append(f"\nÁrbol {i}:")
            arboles_txt.append(str(arbol))
        if len(arboles) > 3:
            arboles_txt.append(f"\nSe omitieron {len(arboles) - 3} árboles adicionales para mantener legible la salida.")

    pd.DataFrame(resumen).to_csv(CARPETA_SALIDAS / "01_parseos_resumen.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fuera_gramatica).to_csv(CARPETA_SALIDAS / "03_tokens_fuera_gramatica.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ambiguas).to_csv(CARPETA_SALIDAS / "04_oraciones_ambiguas.csv", index=False, encoding="utf-8-sig")
    (CARPETA_SALIDAS / "02_arboles_parseo.txt").write_text("\n".join(arboles_txt), encoding="utf-8")

    print("Proceso terminado.")
    print(f"Resumen: {CARPETA_SALIDAS / '01_parseos_resumen.csv'}")
    print(f"Árboles: {CARPETA_SALIDAS / '02_arboles_parseo.txt'}")
    print(f"Tokens fuera de gramática: {CARPETA_SALIDAS / '03_tokens_fuera_gramatica.csv'}")
    print(f"Ambigüedades: {CARPETA_SALIDAS / '04_oraciones_ambiguas.csv'}")


if __name__ == "__main__":
    main()