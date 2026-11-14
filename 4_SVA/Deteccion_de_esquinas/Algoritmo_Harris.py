import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calcular_matriz_harris(img):
    alto, ancho = img.shape
    kernel_media = np.ones((3, 3), np.float32) / 9.0
    img_nueva_filtro = cv2.filter2D(img.astype(np.float32), -1, kernel_media)
    
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    img_nueva_x = cv2.filter2D(img_nueva_filtro, -1, gx)
    img_nueva_y = cv2.filter2D(img_nueva_filtro, -1, gy)
    
    Ixx = cv2.GaussianBlur(img_nueva_x ** 2, (3, 3), 0)
    Iyy = cv2.GaussianBlur(img_nueva_y ** 2, (3, 3), 0)
    Ixy = cv2.GaussianBlur(img_nueva_x * img_nueva_y, (3, 3), 0)
    
    alpha = 0.04
    V = (Ixx * Iyy - Ixy**2) - alpha * (Ixx + Iyy)**2
    return V

def aplicar_umbral_delta(V, delta):
    V_max = cv2.dilate(V, np.ones((3, 3), np.float32))
    U = ((V == V_max) & (V >= delta)).astype(np.uint8)
    return cv2.dilate(U, np.ones((5, 5), np.uint8)) * 255

base_dir = r"E:\Carpeta compartidad Workgroup\INTELIGENCIA ARTIFICIAL\0 - CODIGOS DE MAESTRIA\4_SVA\Deteccion_de_esquinas"


archivos = [
    "Lena_Gray.png",
    "Libros.jpeg",
    "Estrella.jpeg",
    "Cuadricula.jpeg", 
    "Figuras.jpeg",
    "Cuadrado.jpeg"
]

rutas_imagenes = [os.path.join(base_dir, f) for f in archivos]
deltas = np.linspace(900, 10000, 10, dtype=int)

# Proporción ajustada para reducir huecos
fig, axes = plt.subplots(nrows=len(rutas_imagenes), ncols=len(deltas) + 1, figsize=(20, 9))
fig.suptitle("Experimentación Detección de Esquinas (Algoritmo de Harris)", fontsize=16)

for i, ruta in enumerate(rutas_imagenes):
    nombre_archivo = os.path.basename(ruta)
    
    if not os.path.exists(ruta):
        print(f"¡ADVERTENCIA! No se encontró la imagen: {ruta}")
        # Apaga los ejes vacíos para que no se vean recuadros blancos si falla una ruta
        for ax in axes[i]:
            ax.axis('off')
        continue
        
    print(f"\n--- Procesando imagen: {nombre_archivo} ---")
    img = cv2.imread(ruta, 0)
    
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].axis('off')
    axes[i, 0].set_title("Original", fontsize=9)
        
    V = calcular_matriz_harris(img)
    
    for j, delta in enumerate(deltas):
        col_idx = j + 1
        print(f"  -> Calculando para delta = {delta}")
        esquinas_detectadas = aplicar_umbral_delta(V, delta)
        
        axes[i, col_idx].imshow(esquinas_detectadas, cmap='gray')
        axes[i, col_idx].axis('off')
        axes[i, col_idx].set_title(f"$\delta$={delta}", fontsize=9)

# OPTIMIZACIÓN DE ESPACIOS
plt.tight_layout(pad=1.0)
# Ajusta wspace (espacio horizontal) y hspace (espacio vertical) entre recuadros
plt.subplots_adjust(top=0.92, wspace=0.05, hspace=0.2) 
plt.show()