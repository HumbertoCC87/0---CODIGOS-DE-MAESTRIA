"""
Actividad 01 - Operaciones con pixeles usando OpenCV.

El programa carga dos imagenes, valida que tengan las mismas dimensiones y
realiza suma, resta, AND y OR pixel a pixel. Cada resultado se guarda en la
carpeta Salidas_act01.
"""

from pathlib import Path

import cv2


# Rutas de entrada y salida.
CARPETA_TRABAJO = Path(__file__).resolve().parent
RUTA_MUJER = CARPETA_TRABAJO / "Mujer_SVA_02.jpg"
RUTA_PAISAJE = CARPETA_TRABAJO / "Paisaje_SVA_01.jpg"
CARPETA_SALIDAS = CARPETA_TRABAJO / "Salidas_act01"


def cargar_imagen(ruta: Path):
	imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
	if imagen is None:
		raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta}")
	return imagen


imagen_mujer = cargar_imagen(RUTA_MUJER)
imagen_paisaje = cargar_imagen(RUTA_PAISAJE)

print("=" * 60)
print("VALIDACION DE DIMENSIONES")
print("=" * 60)
print(f"Mujer_SVA_02.jpg:    {imagen_mujer.shape[1]} x {imagen_mujer.shape[0]} px")
print(f"Paisaje_SVA_01.jpg:  {imagen_paisaje.shape[1]} x {imagen_paisaje.shape[0]} px")

if imagen_mujer.shape != imagen_paisaje.shape:
	raise ValueError(
		"Las imagenes deben tener las mismas dimensiones y canales "
		"para realizar las operaciones pixel a pixel."
	)

CARPETA_SALIDAS.mkdir(exist_ok=True)


# ============================== SUMA ===============================
resultado_suma = cv2.add(imagen_mujer, imagen_paisaje)
cv2.imwrite(str(CARPETA_SALIDAS / "resultado_suma.jpg"), resultado_suma)
print("Resultado guardado: resultado_suma.jpg")


# ============================== RESTA ==============================
resultado_resta = cv2.subtract(imagen_mujer, imagen_paisaje)
cv2.imwrite(str(CARPETA_SALIDAS / "resultado_resta.jpg"), resultado_resta)
print("Resultado guardado: resultado_resta.jpg")


# ============================== AND ================================
resultado_and = cv2.bitwise_and(imagen_mujer, imagen_paisaje)
cv2.imwrite(str(CARPETA_SALIDAS / "resultado_and.jpg"), resultado_and)
print("Resultado guardado: resultado_and.jpg")


# ============================== OR =================================
resultado_or = cv2.bitwise_or(imagen_mujer, imagen_paisaje)
cv2.imwrite(str(CARPETA_SALIDAS / "resultado_or.jpg"), resultado_or)
print("Resultado guardado: resultado_or.jpg")

print(f"\nTodas las salidas se guardaron en: {CARPETA_SALIDAS}")
