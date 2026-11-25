#Maestria en Inteligencia Artificial
#Univeridad Politecnica Metropolitana de Hidalgo
#Alumno: Humberto Cruz

#El siguiente código implementa un sistema de análisis y detección de baches en pavimento utilizando técnicas de procesamiento de imágenes. 
#Se aplican filtros de suavizado, operadores de detección de bordes y algoritmos de detección de esquinas para identificar irregularidades en la superficie del pavimento.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def procesar_bache(ruta_imagen):
    # 1. Cargar la imagen original
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {ruta_imagen}")
    
    # Convertir de BGR (formato por defecto de OpenCV) a RGB para visualización
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Estandarizar las dimensiones de la imagen (ej. 500x500 píxeles)
    dimensiones = (500, 500)
    img_resized_rgb = cv2.resize(img_rgb, dimensiones, interpolation=cv2.INTER_AREA)

    # 3. Cambiar el espacio de color (RGB a escala de grises)
    img_gray = cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2GRAY)

    # 4. Integrar técnicas de suavizado de imágenes
    # Aplicamos Filtro Gaussiano (ideal para operadores de gradiente) y Filtro de Mediana (reduce ruido sal y pimienta del asfalto)
    img_gaussian = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_median = cv2.medianBlur(img_gray, 5)

    # Usaremos la imagen suavizada con Gauss para la detección de bordes
    img_suavizada = img_gaussian

    # 5. Detectar bordes con tres operadores (Sobel, Prewitt y Roberts)
    
    # --- A) Operador Sobel ---
    sobel_x = cv2.Sobel(img_suavizada, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_suavizada, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = cv2.convertScaleAbs(sobel_mag)

    # --- B) Operador Prewitt ---
    kernel_prewitt_x = np.array([[-1, 0, 1], 
                                 [-1, 0, 1], 
                                 [-1, 0, 1]], dtype=np.float32)
    kernel_prewitt_y = np.array([[-1, -1, -1], 
                                 [0,  0,  0], 
                                 [1,  1,  1]], dtype=np.float32)
    prewitt_x = cv2.filter2D(img_suavizada, cv2.CV_64F, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(img_suavizada, cv2.CV_64F, kernel_prewitt_y)
    prewitt_mag = cv2.magnitude(prewitt_x, prewitt_y)
    prewitt_edges = cv2.convertScaleAbs(prewitt_mag)

    # --- C) Operador Roberts ---
    kernel_roberts_x = np.array([[1,  0], 
                                 [0, -1]], dtype=np.float32)
    kernel_roberts_y = np.array([[0,  1], 
                                 [-1, 0]], dtype=np.float32)
    roberts_x = cv2.filter2D(img_suavizada, cv2.CV_64F, kernel_roberts_x)
    roberts_y = cv2.filter2D(img_suavizada, cv2.CV_64F, kernel_roberts_y)
    roberts_mag = cv2.magnitude(roberts_x, roberts_y)
    roberts_edges = cv2.convertScaleAbs(roberts_mag)

    # 6. Detectar esquinas y puntos de interés (Algoritmo de Harris y Shi-Tomasi)
    img_esquinas = img_resized_rgb.copy()
    
    # Detección de esquinas con Shi-Tomasi (Good Features to Track) - Muy efectivo en grietas y bordes irregulares
    esquinas = cv2.goodFeaturesToTrack(img_suavizada, maxCorners=150, qualityLevel=0.02, minDistance=10)
    esquinas = np.int64(esquinas)
    for i in esquinas:
        x, y = i.ravel()
        cv2.circle(img_esquinas, (x, y), 4, (255, 0, 0), -1) # Puntos rojos en esquinas

    # 7. Binarizar la imagen
    # Aplicamos el método de Otsu sobre la magnitud de bordes de Sobel y sobre la imagen en grises
    _, binaria_otsu_grises = cv2.threshold(img_suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, binaria_bordes = cv2.threshold(sobel_edges, 50, 255, cv2.THRESH_BINARY)

    # --- Visualización de Resultados ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    fig.suptitle(
        f"Sistema de Análisis y Detección de Baches - {Path(ruta_imagen).name}",
        fontsize=16,
        fontweight='bold'
    )

    axes[0, 0].imshow(img_resized_rgb)
    axes[0, 0].set_title("1. Imagen RGB Estandarizada (500x500)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_gray, cmap='gray')
    axes[0, 1].set_title("2. Escala de Grises")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(img_suavizada, cmap='gray')
    axes[0, 2].set_title("3. Suavizado Gaussiano (5x5)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(sobel_edges, cmap='gray')
    axes[1, 0].set_title("4. Bordes: Operador Sobel")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(prewitt_edges, cmap='gray')
    axes[1, 1].set_title("5. Bordes: Operador Prewitt")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(roberts_edges, cmap='gray')
    axes[1, 2].set_title("6. Bordes: Operador Roberts")
    axes[1, 2].axis('off')

    axes[2, 0].imshow(img_esquinas)
    axes[2, 0].set_title("7. Esquinas y Puntos de Interés")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(binaria_otsu_grises, cmap='gray')
    axes[2, 1].set_title("8. Binarización (Otsu en Grises)")
    axes[2, 1].axis('off')

    axes[2, 2].imshow(binaria_bordes, cmap='gray')
    axes[2, 2].set_title("9. Binarización de Bordes (Sobel)")
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

# ==========================================
# Bloque principal de ejecución
# ==========================================
if __name__ == "__main__":
    carpeta_imagenes = Path(__file__).resolve().parent / "baches"
    extensiones_validas = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    imagenes = sorted(
        (ruta for ruta in carpeta_imagenes.iterdir()
         if ruta.is_file() and ruta.suffix.lower() in extensiones_validas),
        key=lambda ruta: ruta.name.lower()
    ) if carpeta_imagenes.is_dir() else []

    if not imagenes:
        raise FileNotFoundError(
            f"No se encontraron imágenes compatibles en la carpeta: {carpeta_imagenes}"
        )

    print(f"Se encontraron {len(imagenes)} imágenes en '{carpeta_imagenes}'.")
    for indice, ruta_imagen in enumerate(imagenes, start=1):
        print(f"Procesando imagen {indice}/{len(imagenes)}: {ruta_imagen.name}")
        procesar_bache(str(ruta_imagen))