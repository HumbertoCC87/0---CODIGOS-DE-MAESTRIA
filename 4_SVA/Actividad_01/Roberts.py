#combinación de filtros Sobel horizontal y vertical
import cv2
import numpy as np

img = cv2.imread(r'E:\Carpeta compartidad Workgroup\INTELIGENCIA ARTIFICIAL\0 - CODIGOS DE MAESTRIA\4_SVA\Lena.png', 0)
alto, ancho = img.shape
img_nueva = np.zeros((alto, ancho), dtype=np.float32)

#Kernel 3x3
k00, k01, k02 =   0,  1,  0
k10, k11, k12 =  -1,  0,  0
k20, k21, k22 =   0,  0,  0


#Recorremos cada píxel de la imagen
#No se evaluan los píxeles del perímetro
for y in range(1, alto - 1):
    for x in range(1, ancho - 1):

        #Píxel central o de interés
        centro = int(img[y, x])

        # Vecinos adyacentes
        arriba    = int(img[y - 1, x])
        abajo     = int(img[y + 1, x])
        izquierda = int(img[y, x - 1])
        derecha   = int(img[y, x + 1])
        #Vecinos de los 4 vértices
        arriba_izq = int(img[y - 1, x - 1])
        arriba_der = int(img[y - 1, x + 1])
        abajo_izq  = int(img[y + 1, x - 1])
        abajo_der  = int(img[y + 1, x + 1])

        #Multiplicación posición a posición con el kernel
        suma = (arriba_izq * k00) + (arriba * k01) + (arriba_der * k02) + \
               (izquierda  * k10) + (centro * k11) + (derecha    * k12) + \
               (abajo_izq  * k20) + (abajo  * k21) + (abajo_der  * k22)
        
        img_nueva[y, x] = abs(suma)

img_nueva = cv2.normalize(
    img_nueva, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)#trunca decimales 

cv2.imshow('Original', img)
cv2.imshow('Filtro Kernel 3x3', img_nueva)
cv2.waitKey(0)
cv2.destroyAllWindows()