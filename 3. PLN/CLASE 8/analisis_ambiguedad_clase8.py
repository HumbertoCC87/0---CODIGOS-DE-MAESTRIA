# Clase 8 - Procesamiento de Lenguaje Natural
# Autor: M.C. Pablo Ricardo Sánchez Gómez


import csv, re, unicodedata
from pathlib import Path
from collections import Counter
ENTRADA=Path(__file__).parent / 'datos' / 'oraciones_clase8.csv'; SALIDA=Path(__file__).parent / 'salidas_clase8'; SALIDA.mkdir(exist_ok=True)
PALABRAS_AMBIGUAS={'banco','planta','cura','capital','raton','gato','sierra','vela','curso','red','copa','columna','radio','clave','linea','cabo'}
PREPOSICIONES={'con','en','de','por','para','sobre','entre'}
SENALES_SEMANTICAS={'todos','no','pocos','muchas','su','sus','ella','ellos','lo','la','porque','cuando','aunque','antes','despues'}
def limpiar(texto):
    texto=texto.strip().lower(); texto=''.join(c for c in unicodedata.normalize('NFD',texto) if unicodedata.category(c)!='Mn'); texto=re.sub(r'[^a-zñ0-9\s-]',' ',texto); return re.sub(r'\s+',' ',texto).strip()
def tokenizar(texto): return [t for t in limpiar(texto).split() if t]
def escribir(path, rows, fns):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fns); w.writeheader(); w.writerows(rows)
def contexto(toks,i,v=3): return ' '.join(toks[max(0,i-v):min(len(toks),i+v+1)])
with open(ENTRADA,encoding='utf-8-sig',newline='') as f: data=list(csv.DictReader(f))
valid=[]; lex=[]; sint=[]; sem=[]; muestra=[]; resumen=Counter()
for r in data:
    toks=tokenizar(r['texto']); tipo=r['tipo_ambiguedad']; resumen[tipo]+=1
    valid.append({'id':r['id'],'texto_original':r['texto'],'texto_limpio':limpiar(r['texto']),'tokens':' | '.join(toks),'tipo_ambiguedad':tipo,'fenomeno':r['fenomeno'],'dominio':r['dominio']})
    preps=[t for t in toks if t in PREPOSICIONES]; sems=[t for t in toks if t in SENALES_SEMANTICAS]
    for i,t in enumerate(toks):
        if t in PALABRAS_AMBIGUAS: lex.append({'id':r['id'],'texto':r['texto'],'palabra_ambigua':t,'contexto_local':contexto(toks,i),'tipo_esperado':tipo,'interpretacion_requerida':'Seleccionar sentido con base en contexto y dominio.'})
    if preps: sint.append({'id':r['id'],'texto':r['texto'],'preposiciones_detectadas':' | '.join(preps),'num_preposiciones':len(preps),'tipo_esperado':tipo,'posible_problema':'Adjunción de sintagma preposicional o complemento con más de una lectura.'})
    if sems: sem.append({'id':r['id'],'texto':r['texto'],'senales_semanticas':' | '.join(sems),'tipo_esperado':tipo,'posible_problema':'Revisar referencia, alcance, cuantificador, causalidad o roles.'})
    score=(2 if any(t in PALABRAS_AMBIGUAS for t in toks) else 0)+(1 if preps else 0)+(1 if sems else 0)+1
    muestra.append({'id':r['id'],'texto':r['texto'],'tipo_ambiguedad':tipo,'fenomeno':r['fenomeno'],'puntaje_revision':score,'uso_sugerido':'Seleccionar para muestra si representa claramente el tipo de ambigüedad.'})
escribir(SALIDA/'01_dataset_validado.csv',valid,['id','texto_original','texto_limpio','tokens','tipo_ambiguedad','fenomeno','dominio'])
escribir(SALIDA/'02_indicios_lexicos.csv',lex,['id','texto','palabra_ambigua','contexto_local','tipo_esperado','interpretacion_requerida'])
escribir(SALIDA/'03_indicios_sintacticos.csv',sint,['id','texto','preposiciones_detectadas','num_preposiciones','tipo_esperado','posible_problema'])
escribir(SALIDA/'04_indicios_semanticos.csv',sem,['id','texto','senales_semanticas','tipo_esperado','posible_problema'])
escribir(SALIDA/'05_resumen_por_tipo.csv',[{'tipo_ambiguedad':k,'total':v} for k,v in sorted(resumen.items())],['tipo_ambiguedad','total'])
escribir(SALIDA/'06_muestra_sugerida.csv',sorted(muestra,key=lambda x:(x['tipo_ambiguedad'],-int(x['puntaje_revision']),int(x['id']))),['id','texto','tipo_ambiguedad','fenomeno','puntaje_revision','uso_sugerido'])
print('Clase 8 - análisis de ambigüedad terminado'); print('Oraciones procesadas:',len(data)); print('Salidas generadas en:',SALIDA.resolve())
