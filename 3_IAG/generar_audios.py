import asyncio
from pathlib import Path
import edge_tts

# Voces en español de México (HD Neuronales)
VOZ_CHICA = "es-MX-DaliaNeural"  
VOZ_CHICO = "es-MX-JorgeNeural"  


BLOQUES = [
    {
        "filename": "bloque_01_chica.mp3",
        "voice": VOZ_CHICA,
        "text": """Estimados compañeros. En la  distribución de última milla, el mantenimiento reactivo no es simplemente un costo financiero imprevisto; representa un golpe directo a nuestro nivel de servicio y, sobre todo, a la seguridad de nuestra gente
            . Cuando una de nuestras unidades de reparto directo al punto de venta, o rutas DSD, sufre una avería mayor en ruta, se desencadena un efecto dominó sumamente crítico
            . Perdemos la entrega del día, incumplimos la promesa de servicio con el cliente de Sabritas, incurrimos en costosos arrastres de grúa y rescates mecánicos de emergencia, y lo más grave: dejamos a nuestro vendedor varado en carretera, expuesto a riesgos viales y de seguridad física que son completamente evitables si actuamos a tiempo
            .
            Actualmente, la flota de transporte en el centro de distribución de Pachuca opera bajo un modelo de mantenimiento preventivo calendarizado o, en el peor de los casos, reactivo, careciendo de un monitoreo predictivo en tiempo real
            . Para romper este paradigma operativo, hemos diseñado un Producto Mínimo Viable enfocado en una muestra estratégica de ochenta y ocho vehículos para Sabritas
            .
            La viabilidad de este proyecto es contundente gracias a su arquitectura con inversión inicial cero en software
            . Al maximizar el uso de los dispositivos y las licencias de Geotab con las que ya cuenta nuestra flota, eliminamos por completo los costos de licenciamiento de plataformas externas o intermediarios
            . Esta innovación propia e in-house nos permite dar el salto hacia una operación de transporte inteligente, enfocada en la reducción de costos por reparaciones mayores y en la consolidación de una flota completamente sana para nuestros vendedores""",
    },
    {
        "filename": "bloque_02_chica.mp3",
        "voice": VOZ_CHICA,
        "text": """"La solidez de este desarrollo radica en una infraestructura de datos robusta y escalable
            . El pipeline de ingeniería de datos comienza con la extracción dinámica de telemetría a través del script cliente de Geotab en VS Code, el cual realiza peticiones automatizadas a la API REST del proveedor utilizando una ventana móvil de datos de treinta días
            .
            Una vez extraída la información cruda de los sensores, el script de preprocesamiento de motor entra en acción para consolidar los viajes mecánicos
            . Este proceso incluye una fase crítica de limpieza de ruido: el algoritmo analiza miles de trayectos y descarta de forma automática los denominados microviajes, es decir, aquellos traslados extremadamente cortos que no permiten estabilizar las curvas térmicas y eléctricas de los componentes, evitando sesgar el modelo predictivo. Posteriormente, el sistema realiza un mapeo dinámico vinculando los identificadores de hardware de Geotab con los números económicos reales de la flota para una perfecta trazabilidad en el taller.
            Toda la gobernanza de este proyecto, desde las dependencias del código hasta la arquitectura del flujo, está documentada de forma modular en una bóveda de Obsidian. Esto nos garantiza un sistema ligero, independiente de proveedores y cien por ciento preparado para escalabilidad nacional. Si el negocio lo requiere, el código está optimizado para cómputo en paralelo utilizando arquitecturas GPU y automatización de tareas en servidores centrales como podria ser Azure""",
    },
    {
        "filename": "bloque_03_chica.mp3",
        "voice": VOZ_CHICA,
        "text": """Para diagnosticar fallas invisibles a los escáneres convencionales, implementamos una red neuronal profunda Autoencoder de aprendizaje no supervisado mediante la librería PyOD con backend en PyTorch
            . La validez científica de este modelo está respaldada por un análisis riguroso sobre catorce mil setecientos dos viajes reales evaluados de nuestra flota piloto
            .
            Al analizar la distribución de los scores de riesgo, definimos un umbral de corte operativo con una tasa de contaminación del cinco por ciento, equivalente a un valor de cero punto setenta y cuatro setenta y ocho
            . Bajo este criterio, clasificamos con total precisión trece mil novecientos sesenta y seis viajes como completamente saludables, aislando únicamente setecientos treinta y seis viajes con anomalías mecánicas críticas
            .
            Para validar que el modelo aísla fallas reales y no asigna alertas al azar, calculamos la métrica estadística del Índice d de Cohen sobre el Score de Riesgo
            . Mientras el grupo saludable mantuvo una media de score de apenas cero punto veintiséis setenta y siete, el grupo clasificado en riesgo disparó su media a cero punto noventa y ocho cincuenta y ocho, arrojando un d de Cohen de tres punto cero cero cuarenta y dos
            . En la ciencia de datos, un valor de d de Cohen mayor a cero punto ocho indica un impacto grande; obtener un tres punto cero representa una separación extrema y limpia entre unidades sanas y enfermas, erradicando los falsos positivos
            .
            Este rigor analítico se comprueba al desglosar los errores de reconstrucción de los sensores cuando el modelo detecta una anomalía
            . El error de reconstrucción en la temperatura máxima del refrigerante aumenta drásticamente cuatro punto dos veces en comparación con un viaje normal, elevándose de cero punto doce noventa y cuatro a cero punto cincuenta y cuatro veinticinco
            . Asimismo, el error en el voltaje mínimo de arranque del motor se incrementa dos punto nueve veces, delatando fallas inminentes de batería, y la intensidad de vibración medida por el acelerómetro aumenta tres punto cuatro veces
            . Estas desviaciones masivas no son ruido estocástico; son patrones de deformación física en los componentes que nos permiten programar el taller antes de que ocurra la avería en ruta""",
    },
    {
        "filename": "bloque_04_chico.mp3",
        "voice": VOZ_CHICO,
        "text": """La verdadera transformación digital ocurre cuando la ciencia de datos se traduce en una herramienta accionable para el día a día de la operación. Para lograrlo, consolidamos este modelo en una interfaz web interactiva que funciona como nuestra Torre de Control de Flota, desarrollada enteramente sobre Streamlit
            . Al ingresar al dashboard, los facilitadores y auxiliares de transporte se encuentran con un mapa de geolocalización de unidades en tiempo real y tarjetas dinámicas que resumen los indicadores clave, como el volumen de viajes analizados, las alertas rojas por componentes críticos y los semáforos generales de salud de las unidades [Prototipo pagina 1.png].
            Sin embargo, la inteligencia artificial en la industria logística exige un equilibrio operativo. Si enviamos alertas sin un filtro riguroso, corremos el riesgo de provocar fatiga por alarmas y de saturar la capacidad instalada de nuestros talleres mecánicos
            . Es aquí donde entran en juego nuestras reglas de negocio y el script motor de alertas punto pe ye
            .
            En primer lugar, aplicamos un umbral de corte estricto basado en una tasa de contaminación del cinco por ciento, lo que equivale a un score de riesgo de cero punto setenta y cuatro setenta y ocho, para aislar únicamente al grupo de unidades que representa el peligro real más inminente
            .
            En segundo lugar, nuestro algoritmo realiza una limpieza de trayectos donde se descartan automáticamente los microviajes, garantizando que el análisis térmico y eléctrico de los componentes se base en rutas operativas estándar y no en maniobras cortas de patio
            .
            Finalmente, implementamos la regla de persistencia: el motor de alertas exige que una anomalía sea detectada y se mantenga de manera consecutiva a lo largo de diez viajes antes de generar una orden de inspección física en el taller
            . De esta manera, aseguramos que cada intervención preventiva esté plenamente justificada por un patrón de fallo físico sostenido, permitiendo al equipo consultar el análisis de causa raíz detallado por sensor para actuar de inmediato""",
    },
    {
        "filename": "bloque_05_chico.mp3",
        "voice": VOZ_CHICO,
        "text": """Viabilidad financiera, ahorros comparativos (ocho a doce por ciento versus preventivo, cuarenta por ciento versus reactivo), reducción de averías mayores, escalabilidad en hardware (GPU) e infraestructura, gestión del cambio con mecánicos, cierre de alto impacto y agradecimiento al comité directivo.
            Pasemos ahora al análisis de viabilidad financiera y al retorno de inversión esperado para el negocio. De acuerdo con el estado del arte en la ingeniería de transportes y la logística de última milla, la transición de un esquema calendarizado a uno basado en mantenimiento predictivo genera un beneficio directo e inmediato. Un plan de mantenimiento predictivo bien ejecutado es capaz de ahorrar entre un ocho y un doce por ciento en costos globales frente al mantenimiento preventivo tradicional, y reduce de manera sobresaliente hasta un cuarenta por ciento el gasto operativo frente al costoso e imprevisto mantenimiento reactivo
            .
            Evitar averías mayores no solo optimiza nuestros presupuestos de reparaciones en talleres; previene de manera directa fallas en el servicio de entrega de Sabritas, protege nuestro nivel de servicio de cara a los clientes y evita incidentes críticos donde nuestros vendedores queden varados en ruta por alguna falla mecánica, disminuyendo su exposición a riesgos viales en el camino
            .
            Este proyecto destaca por tener un costo de desarrollo inicial de cero
            . Al utilizar y maximizar las licencias y dispositivos de Geotab con los que ya cuenta la flota de la compañía, no requerimos inversión de capital en software externo
            . La única inversión marginal requerida para un despliegue masivo a nivel nacional será la adquisición de capacidad en un servidor centralizado o espacio en azure para alojar el motor de optimización y el dashboard de monitoreo preventivo
            .
            El código está diseñado bajo estándares de alta escalabilidad; tanto el backend analítico como el motor de optimización están preparados para realizar cómputo en paralelo mediante arquitecturas de procesamiento gráfico o GPU y la programación automatizada de tareas de forma integrada
            .
            .
            Les invito a respaldar esta iniciativa. Transformemos nuestra logística en un motor inteligente que se adelanta a los problemas antes de que sucedan.
            A nombre del equipo de innovación logística y arquitectura de inteligencia artificial, agradezco profundamente su atención y el tiempo dedicado el día de hoy para conocer el futuro de nuestra operación de flota. Muchas gracias""",
    },
]


async def generar_todos_los_audios():
    # Obtiene la ruta del directorio donde se encuentra este script
    directorio_script = Path(__file__).parent
    print(f"🎙️ Iniciando generación de audios con Edge-TTS...")
    print(f"Los archivos se guardarán en: {directorio_script}\n")

    for bloque in BLOQUES:
        # Construye la ruta de guardado completa
        ruta_guardado = directorio_script / bloque["filename"]
        print(f"Generando: {ruta_guardado}...")
        communicate = edge_tts.Communicate(bloque["text"], bloque["voice"])
        await communicate.save(ruta_guardado)
        print(f"✓ Guardado con éxito: {ruta_guardado}\n")
    print("✨ ¡Todos los audios se han generado correctamente!")


if __name__ == "__main__":
    asyncio.run(generar_todos_los_audios())