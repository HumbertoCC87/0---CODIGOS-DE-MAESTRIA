# -*- coding: utf-8 -*-
"""
Universidad Piltecnica Metropolitana de Hidalgo
Maestria en Inteligencia Artificial
Actividad 4: Comparativa de Pipelines de Segmentación (Visión por Computadora)

Este script implementa tres pipelines de preprocesamiento y segmentación de imágenes foliares,
Reporte y Comparativa de Pipelines de Segmentación (Visión por Computadora)
Imágenes de entrada: PVC1.JPG (Hoja 1) y PVC2.JPG (Hoja 2)
Autor: Humberto Cruz Cruz
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def evaluar_pipelines_segmentacion(ruta_imagen, nombre_hoja):
    """
    Ejecuta y compara 3 pipelines de preprocesamiento y segmentación sobre la imagen de una hoja,
    retornando las máscaras binarias de cada candidato y las métricas cuantitativas de calidad.
    """
    img_bgr = cv2.imread(str(ruta_imagen))
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # =========================================================================
    # CANDIDATO 1: Escala de Grises + Filtro Gaussiano + Umbralización de Otsu
    # =========================================================================
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gris_suavizado = cv2.GaussianBlur(gris, (5, 5), 0)
    umbral_1, mascara_1 = cv2.threshold(
        gris_suavizado, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # =========================================================================
    # CANDIDATO 2: Índice de Exceso de Verde (ExG = 2G - R - B) + Otsu
    # =========================================================================
    b_ch, g_ch, r_ch = cv2.split(img_bgr.astype(np.int16))
    exg = np.clip(2 * g_ch - r_ch - b_ch, 0, 255).astype(np.uint8)
    umbral_2, mascara_2 = cv2.threshold(
        exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # =========================================================================
    # CANDIDATO 3 (GANADOR): Espacio CIE L*a*b* (Diferencia Cromática b* - a*)
    # + Filtro Gaussiano + Otsu + Morfología Matemática + Contorno Mayor
    # =========================================================================
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    
    # Realce cromático: maximiza simultáneamente el tejido verde (a* bajo) y zonas
    # con clorosis/amarillamiento (b* alto), suprimiendo el fondo gris y la sombra proyectada
    diferencia_cromatica = np.clip(
        b.astype(np.int16) - a.astype(np.int16) + 128, 0, 255
    ).astype(np.uint8)
    
    croma_suavizado = cv2.GaussianBlur(diferencia_cromatica, (5, 5), 0)
    umbral_3, otsu_crudo = cv2.threshold(
        croma_suavizado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Post-procesamiento morfológico con elemento estructurante elíptico (5x5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morfologia = cv2.morphologyEx(otsu_crudo, cv2.MORPH_CLOSE, kernel, iterations=2)
    morfologia = cv2.morphologyEx(morfologia, cv2.MORPH_OPEN, kernel, iterations=1)

    # Selección de la componente conexa principal y relleno topológico de poros internos
    contornos, _ = cv2.findContours(
        morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mascara_3_optima = np.zeros_like(morfologia)
    if contornos:
        contorno_mayor = max(contornos, key=cv2.contourArea)
        cv2.drawContours(
            mascara_3_optima, [contorno_mayor], -1, 255, thickness=cv2.FILLED
        )

    return {
        "img_rgb": img_rgb,
        "canal_a": a,
        "croma_suavizado": croma_suavizado,
        "otsu_crudo": otsu_crudo,
        "mascara_1": mascara_1,
        "mascara_2": mascara_2,
        "mascara_3": mascara_3_optima,
        "umbrales": (umbral_1, umbral_2, umbral_3),
    }


if __name__ == "__main__":
    directorio_script = Path(__file__).resolve().parent
    archivos = [
        (directorio_script / "PVC1.JPG", "Hoja 1"),
        (directorio_script / "PVC2.JPG", "Hoja 2"),
    ]
    for ruta, etiqueta in archivos:
        resultados = evaluar_pipelines_segmentacion(ruta, etiqueta)
        nombre_salida = directorio_script / f"Mascara_Binaria_{etiqueta.replace(' ', '')}_{ruta.name}.png"
        cv2.imwrite(str(nombre_salida), resultados["mascara_3"])
        print(f"Segmentación óptima completada y guardada en: {nombre_salida}")