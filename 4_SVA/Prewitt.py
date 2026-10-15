#Dettección de bordes con el operador Prewitt horizontal y vertical
import os
import cv2
import numpy as np

img = cv2.imread(r'E:\Carpeta compartidad Workgroup\INTELIGENCIA ARTIFICIAL\0 - CODIGOS DE MAESTRIA\4_SVA\Lena.png', 0)
alto, ancho = img.shape

#Crear matrices para almacenar los resultados
img_nueva_x = np.zeros((alto, ancho), dtype=np.float32)
img_nueva_y = np.zeros((alto, ancho), dtype=np.float32)

gx = np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1]
])

gy = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

for y in range(1, alto - 1):
    for x in range(1, ancho - 1):

        #Obtener los valores de los nueve píxeles vecinos
        arriba_izq = int(img[y - 1, x - 1])
        arriba = int(img[y - 1, x])
        arriba_der = int(img[y - 1, x + 1])

        izquierda = int(img[y, x - 1])
        centro = int(img[y, x])
        derecha = int(img[y, x + 1])

        abajo_izq = int(img[y + 1, x - 1])
        abajo = int(img[y + 1, x])
        abajo_der = int(img[y + 1, x + 1])
                
        #Calcular Prewitt X
        suma_x = (
            arriba_izq * gx[0, 0] + arriba * gx[0, 1] + arriba_der * gx[0, 2] +
            izquierda * gx[1, 0] + centro * gx[1, 1] + derecha * gx[1, 2] + 
            abajo_izq * gx[2, 0] + abajo * gx[2, 1] + abajo_der * gx[2, 2])

        #Calcular Prewitt Y
        suma_y = (
            arriba_izq * gy[0, 0] + arriba * gy[0, 1] + arriba_der * gy[0, 2] +
            izquierda * gy[1, 0] + centro * gy[1, 1] + derecha * gy[1, 2] +
            abajo_izq * gy[2, 0] + abajo * gy[2, 1] + abajo_der * gy[2, 2])

        #Guardar los resultados de cada componente
        img_nueva_x[y, x] = suma_x
        img_nueva_y[y, x] = suma_y

#Combinar los bordes verticales y horizontales
img_nueva = np.sqrt(img_nueva_x**2 + img_nueva_y**2)

img_nueva = cv2.normalize(
    img_nueva, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imshow('Imagen original', img)
cv2.imshow('Prewitt combinado (verticales y horizontales)', img_nueva)

cv2.waitKey(0)
cv2.destroyAllWindows()