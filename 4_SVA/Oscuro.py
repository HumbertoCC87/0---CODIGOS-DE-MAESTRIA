import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('paisaje.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: No se pudo cargar la imagen. Verifica que el archivo existe.")
    exit()

# Obtener dimensiones (alto y ancho)
if len(img.shape) == 3:
    # Si es color, convertir a escala de grises
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

alto, ancho = img.shape
histograma = np.zeros(256, dtype=int)

for y in range(alto): #Filas
    for x in range(ancho): #Columnas
        tono = img[y, x]
        histograma[tono] += 1

histograma_acumulado = np.cumsum(histograma)

plt.figure('Histograma')
plt.title('Histograma de Escala de Grises')
plt.xlabel('Tono de gris (0 = Negro, 255 = Blanco)')
plt.ylabel('Cantidad de píxeles')
plt.plot(histograma_acumulado, color='Blue')
plt.xlim([0, 256])
plt.grid(True)
plt.show()

cv2.imshow('Imagen', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 