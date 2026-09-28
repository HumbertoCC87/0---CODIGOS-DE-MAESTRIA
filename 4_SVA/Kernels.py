import os
import cv2
import numpy as np

def aplicar_kernel(img, kernel, divisor=1, offset=0):
    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 == 0:
        raise ValueError("El kernel debe ser cuadrado y tener un tamaño impar.")
    if divisor == 0:
        raise ValueError("El divisor no puede ser cero.")

    resultado = cv2.filter2D(img, cv2.CV_32F, kernel)
    resultado = np.clip(np.rint(resultado / divisor + offset), 0, 255).astype(np.uint8)

    # Se conserva el comportamiento original: el perímetro no se procesa.
    radio = kernel.shape[0] // 2
    if radio:
        resultado[:radio, :] = 0
        resultado[-radio:, :] = 0
        resultado[:, :radio] = 0
        resultado[:, -radio:] = 0

    return resultado

# Leer imagen
img = cv2.imread(r'E:\Carpeta compartidad Workgroup\INTELIGENCIA ARTIFICIAL\0 - CODIGOS DE MAESTRIA\4_SVA\Lena.png', 0)
if img is None:
    raise FileNotFoundError("No se pudo cargar Lena.png. Verifica la ruta de la imagen.")

# Kernel 1: Filtro Media 3x3
kernel1 = [[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]]
img_kernel1 = aplicar_kernel(img, kernel1, divisor=9)

# Kernel 2: Filtro Gaussiano 3x3
kernel2 = [[1, 2, 1],
           [2, 4, 2],
           [1, 2, 1]]
img_kernel2 = aplicar_kernel(img, kernel2, divisor=16)

# Kernel 3: Filtro Gaussiano 5x5 (suma de coeficientes = 273)
kernel3 = [[1, 4, 7, 4, 1],
           [4, 16, 26, 16, 4],
           [7, 26, 41, 26, 7],
           [4, 16, 26, 16, 4],
           [1, 4, 7, 4, 1]]
img_kernel3 = aplicar_kernel(img, kernel3, divisor=273)

# Kernel 4: Realce
kernel4 = [[-1,  2, -1],
           [ 2, -4,  2],
           [-1,  2, -1]]
img_kernel4 = aplicar_kernel(img, kernel4, offset=128)

# Kernel 5: Paso Alto
kernel5 = [[-1, -1, -1],
           [-1,  8, -1],
           [-1, -1, -1]]
img_kernel5 = aplicar_kernel(img, kernel5)

def agregar_titulo(imagen, titulo):
    panel = cv2.copyMakeBorder(imagen, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
    cv2.putText(panel, titulo, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 255, 1, cv2.LINE_AA)
    return panel


# Comparación en una sola imagen: original y los tres filtros (media 1/9, Gausiano 1/16, y Gausiano 1/273) 
paneles = [
    agregar_titulo(img, 'Original'),
    agregar_titulo(img_kernel1, 'Media 3x3 (1/9)'),
    agregar_titulo(img_kernel2, 'Gaussiano 3x3 (1/16)'),
    agregar_titulo(img_kernel3, 'Gaussiano 5x5 (1/273)'),
]
fila_superior = cv2.hconcat(paneles[:2])
fila_inferior = cv2.hconcat(paneles[2:])
comparativo = cv2.vconcat([fila_superior, fila_inferior])

ruta_salida = os.path.join(os.path.dirname(__file__), 'Comparativo_filtros.png')
cv2.imwrite(ruta_salida, comparativo)
print(f'Comparativo guardado en: {ruta_salida}')
cv2.imshow('Comparativo de filtros', comparativo)

cv2.waitKey(0)
cv2.destroyAllWindows()