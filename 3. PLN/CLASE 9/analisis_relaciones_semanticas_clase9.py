# Clase 9 PLN · Relaciones semánticas y similitud léxica
# UPMH · M.C. Pablo Ricardo Sánchez Gómez
# Requisito: ejecutar en ambiente Conda.
# Caso realista integrado: el recurso léxico inicial está incompleto.
# Tarea técnica: detectar casos sin cobertura y ampliar el diccionario de relaciones.

from pathlib import Path
import csv
import math
import re
import unicodedata
from collections import Counter, defaultdict

RUTA_DATASET = Path("dataset_relaciones_semanticas_clase9.csv")
CARPETA_SALIDA = Path("salidas_clase9")
CARPETA_SALIDA.mkdir(exist_ok=True)

# Recurso léxico inicial: intencionalmente incompleto para simular un caso real.
# La ausencia de términos de dominio provoca falsos "sin_relacion_clara".
LEXICO_RELACIONES = {
    "sinonimia": {
        ("alumno", "estudiante"), ("docente", "profesor"), ("coche", "automovil"),
        ("computadora", "ordenador"), ("evidencia", "prueba"), ("error", "fallo"),
        ("rapido", "veloz"), ("analizar", "examinar"), ("resultado", "salida"),
        ("archivo", "documento"), ("oracion", "frase"), ("sentido", "significado")
    },
    "hiperonimia": {
        ("animal", "perro"), ("vehiculo", "automovil"), ("institucion", "universidad"),
        ("herramienta", "parser"), ("modelo", "clasificador"), ("lenguaje", "espanol"),
        ("archivo", "csv"), ("documento", "pdf"), ("persona", "alumno"),
        ("tecnica", "tokenizacion"), ("metrica", "precision"), ("algoritmo", "svm")
    },
    "meronimia": {
        ("automovil", "motor"), ("computadora", "teclado"), ("oracion", "sujeto"),
        ("oracion", "predicado"), ("documento", "pagina"), ("texto", "parrafo"),
        ("parrafo", "oracion"), ("dataset", "registro"), ("tabla", "columna"),
        ("codigo", "funcion"), ("pipeline", "etapa"), ("corpus", "documento")
    },
    "antonimia": {
        ("correcto", "incorrecto"), ("valido", "invalido"), ("simple", "complejo"),
        ("claro", "ambiguo"), ("aceptado", "rechazado"), ("presente", "ausente"),
        ("positivo", "negativo"), ("alto", "bajo"), ("rapido", "lento"),
        ("completo", "incompleto"), ("manual", "automatico"), ("verdadero", "falso")
    },
    "relacion_contextual": {
        ("medico", "hospital"), ("profesor", "aula"), ("modelo", "dataset"),
        ("parser", "oracion"), ("algoritmo", "codigo"), ("banco", "credito"),
        ("planta", "fabrica"), ("juez", "sentencia"), ("equipo", "partido"),
        ("paciente", "diagnostico"), ("estudiante", "tarea"), ("programa", "terminal")
    }
}

# Ajuste esperado: agregar pares faltantes al recurso léxico cuando el análisis lo justifique.
# Ejemplos de huecos reales del dataset: (token, unidad), (diccionario, lexico),
# (universidad, aula), (examen, pregunta), (texto, contexto), (archivo, ruta), etc.


def normalizar(texto):
    texto = texto.strip().lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-zñ0-9\s_-]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


# Estructuras para el léxico pre-procesado
LEXICO_PREPROCESADO = defaultdict(set)
SYMMETRIC_RELATIONS = {"sinonimia", "antonimia", "relacion_contextual"}

def preprocesar_lexico():
    """Normaliza y estructura el léxico para búsquedas eficientes."""
    for tipo, pares in LEXICO_RELACIONES.items():
        for t1, t2 in pares:
            norm_t1 = normalizar(t1)
            norm_t2 = normalizar(t2)
            if tipo in SYMMETRIC_RELATIONS:
                # Para relaciones simétricas, guardar en orden canónico (ordenado)
                par = tuple(sorted((norm_t1, norm_t2)))
            else:
                # Para asimétricas, guardar en el orden original
                par = (norm_t1, norm_t2)
            LEXICO_PREPROCESADO[tipo].add(par)


def jaccard_caracteres(a, b):
    ca = set(normalizar(a))
    cb = set(normalizar(b))
    if not ca and not cb:
        return 0.0
    return len(ca & cb) / len(ca | cb)


def detectar_por_lexico(t1, t2):
    norm_t1 = normalizar(t1)
    norm_t2 = normalizar(t2)
    par_asimetrico = (norm_t1, norm_t2)
    par_simetrico = tuple(sorted(par_asimetrico))

    # Buscar primero en relaciones asimétricas con el orden original
    for tipo, pares_norm in LEXICO_PREPROCESADO.items():
        if tipo not in SYMMETRIC_RELATIONS:
            if par_asimetrico in pares_norm:
                return tipo
    
    # Si no se encuentra, buscar en relaciones simétricas con orden canónico
    for tipo, pares_norm in LEXICO_PREPROCESADO.items():
        if tipo in SYMMETRIC_RELATIONS:
            if par_simetrico in pares_norm:
                return tipo

    return "sin_cobertura_lexica"


def sugerir_observacion(t1, t2, pred, score):
    if pred == "sin_cobertura_lexica":
        return "El par no existe en el recurso léxico inicial; requiere revisión y posible ampliación del diccionario."
    if score > 0.65 and pred not in {"sinonimia", "antonimia"}:
        return "La similitud superficial es alta; revisar si existe relación semántica no cubierta por el tipo detectado."
    return "Relación detectada por recurso léxico inicial."


def leer_dataset(ruta):
    with ruta.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    requeridas = {'id','termino_1','termino_2','tipo_esperado','dominio'}
    if not rows:
        raise ValueError('El dataset está vacío.')
    if not requeridas.issubset(rows[0].keys()):
        raise ValueError(f'Columnas requeridas: {sorted(requeridas)}')
    return rows


def escribir_csv(nombre, filas, campos):
    with (CARPETA_SALIDA / nombre).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def analizar(rows):
    resultados=[]
    for r in rows:
        t1, t2 = r['termino_1'], r['termino_2']
        pred = detectar_por_lexico(t1, t2)
        score = round(jaccard_caracteres(t1, t2), 3)
        correcto = pred == r['tipo_esperado']
        resultados.append({
            'id': r['id'],
            'termino_1': t1,
            'termino_2': t2,
            'dominio': r['dominio'],
            'tipo_esperado': r['tipo_esperado'],
            'tipo_detectado': pred,
            'similitud_jaccard_caracteres': score,
            'coincide_con_esperado': 'si' if correcto else 'no',
            'observacion': sugerir_observacion(t1, t2, pred, score)
        })
    return resultados


def matriz_confusion(resultados):
    tipos = sorted({r['tipo_esperado'] for r in resultados} | {r['tipo_detectado'] for r in resultados})
    conteo = defaultdict(Counter)
    for r in resultados:
        conteo[r['tipo_esperado']][r['tipo_detectado']] += 1
    filas=[]
    for real in tipos:
        fila={'tipo_esperado': real}
        for pred in tipos:
            fila[pred] = conteo[real][pred]
        filas.append(fila)
    return filas, ['tipo_esperado'] + tipos


def resumen(resultados):
    total=len(resultados)
    correctos=sum(1 for r in resultados if r['coincide_con_esperado']=='si')
    por_tipo=defaultdict(lambda: {'total':0,'correctos':0,'sin_cobertura':0})
    for r in resultados:
        d=por_tipo[r['tipo_esperado']]
        d['total'] += 1
        d['correctos'] += 1 if r['coincide_con_esperado']=='si' else 0
        d['sin_cobertura'] += 1 if r['tipo_detectado']=='sin_cobertura_lexica' else 0
    filas=[]
    for tipo, d in sorted(por_tipo.items()):
        filas.append({
            'tipo': tipo,
            'total': d['total'],
            'correctos': d['correctos'],
            'sin_cobertura': d['sin_cobertura'],
            'accuracy_tipo': round(d['correctos']/d['total'],3) if d['total'] else 0
        })
    filas.append({'tipo':'TOTAL','total':total,'correctos':correctos,'sin_cobertura':sum(1 for r in resultados if r['tipo_detectado']=='sin_cobertura_lexica'),'accuracy_tipo':round(correctos/total,3)})
    return filas


def seleccionar_muestra_revision(resultados):
    # 18 casos: 3 por tipo esperado, priorizando errores/sin cobertura.
    salida=[]
    tipos=sorted(set(r['tipo_esperado'] for r in resultados))
    for tipo in tipos:
        candidatos=[r for r in resultados if r['tipo_esperado']==tipo]
        candidatos=sorted(candidatos, key=lambda x: (x['coincide_con_esperado']=='si', x['tipo_detectado']!='sin_cobertura_lexica', x['id']))
        salida.extend(candidatos[:3])
    return salida


def main():
    preprocesar_lexico()
    rows = leer_dataset(RUTA_DATASET)
    resultados = analizar(rows)
    campos_res=['id','termino_1','termino_2','dominio','tipo_esperado','tipo_detectado','similitud_jaccard_caracteres','coincide_con_esperado','observacion']
    escribir_csv('01_resultados_relaciones_semanticas.csv', resultados, campos_res)
    mc, campos_mc = matriz_confusion(resultados)
    escribir_csv('02_matriz_confusion_relaciones.csv', mc, campos_mc)
    escribir_csv('03_casos_sin_cobertura.csv', [r for r in resultados if r['tipo_detectado']=='sin_cobertura_lexica'], campos_res)
    escribir_csv('04_resumen_por_tipo.csv', resumen(resultados), ['tipo','total','correctos','sin_cobertura','accuracy_tipo'])
    escribir_csv('05_muestra_revision.csv', seleccionar_muestra_revision(resultados), campos_res)
    print('Análisis de relaciones semánticas completado.')
    print(f'Registros procesados: {len(resultados)}')
    print('Salidas generadas en:', CARPETA_SALIDA)
    print('Caso realista: revisar 03_casos_sin_cobertura.csv y ampliar LEXICO_RELACIONES cuando proceda.')

if __name__ == '__main__':
    main()
