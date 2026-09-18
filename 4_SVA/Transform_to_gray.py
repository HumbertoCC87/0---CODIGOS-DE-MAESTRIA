import cv2
import numpy as np

img = cv2.imread('/Users/ernestogarciaamaro/Documents/Python/Lena_Gray.png', 0)
alto, ancho = img.shape #Obtenemos dimensiones
img_nueva = np.zeros((alto, ancho), dtype=np.uint8)#Imagen resultante

#Recorremos cada píxel de la imagen
for y in range(1, alto - 1): #Filas
    for x in range(1, ancho - 1): #Columnas

        arriba_izq = int(img[y - 1, x - 1])
        arriba_der = int(img[y - 1, x + 1])
        abajo_izq  = int(img[y + 1, x - 1])
        abajo_der  = int(img[y + 1, x + 1])
        
        media = (arriba_izq + arriba_der + abajo_izq + abajo_der) / 4
        img_nueva[y, x] = media

cv2.imshow('Original', img)
cv2.imshow('4 vecinos (vertices)', img_nueva)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('/Users/ernestogarciaamaro/Documents/Python/Lena_Gray_vecinos.png', img_nueva)