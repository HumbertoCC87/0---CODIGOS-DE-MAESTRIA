import time
from pathlib import Path
import cv2
import kagglehub
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------
# 1. Descarga y carga de rutas de imágenes
# ---------------------------------------------------------
# Si ya tienes el dataset descargado localmente, reemplaza la siguiente línea
# por: DATASET_DIR = Path("ruta/a/tu/carpeta")
DATASET_DIR = Path(kagglehub.dataset_download("naivedatamodel/soil-dataset"))

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cargar_rutas_y_etiquetas(root_dir: Path):
    rutas, etiquetas = [], []
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            # Toma el nombre de la carpeta inmediata como clase (ej. Alluvial, Black, Clay, Red)
            clase = path.parent.name.strip()
            rutas.append(path)
            etiquetas.append(clase)
    return rutas, etiquetas


rutas, etiquetas = cargar_rutas_y_etiquetas(DATASET_DIR)
le = LabelEncoder()
y = le.fit_transform(etiquetas)

print(f"Total de imágenes encontradas: {len(rutas)}")
print("Distribución por clase:", dict(zip(*np.unique(etiquetas, return_counts=True))))

# Misma partición estratificada (80/20) para ambos experimentos
idx_all = np.arange(len(rutas))
idx_train, idx_test = train_test_split(
    idx_all, test_size=0.20, stratify=y, random_state=42
)

# ---------------------------------------------------------
# 2. Funciones de extracción de características
# ---------------------------------------------------------
def extraer_haralick(path: Path, size=(128, 128), levels=32):
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    # Cuantización de 256 a 32 niveles de gris para estabilizar la GLCM
    gray_q = (gray // (256 // levels)).astype(np.uint8)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        gray_q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    features = []
    for prop in props:
        # Promedio en los 4 ángulos para invarianza a rotación (3 distancias x 6 props = 18 features)
        vals = graycoprops(glcm, prop).mean(axis=1)
        features.extend(vals)
    return np.array(features, dtype=np.float32)


def extraer_pixeles(path: Path, size=(64, 64)):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    # Normalización [0, 1] y aplanamiento (64 x 64 x 3 = 12,288 features)
    return (img_resized.astype(np.float32) / 255.0).flatten()


def construir_matriz(indices, extractor):
    X_list, y_list = [], []
    for i in indices:
        feat = extractor(rutas[i])
        if feat is not None:
            X_list.append(feat)
            y_list.append(y[i])
    return np.vstack(X_list), np.array(y_list)


# ---------------------------------------------------------
# 3. Función de entrenamiento y evaluación
# ---------------------------------------------------------
def evaluar_experimento(nombre, extractor, usar_scaler=True):
    t0 = time.perf_counter()
    X_train, y_train = construir_matriz(idx_train, extractor)
    X_test, y_test = construir_matriz(idx_test, extractor)

    pasos = []
    if usar_scaler:
        pasos.append(("scaler", StandardScaler()))
    pasos.append(("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)))

    modelo = Pipeline(pasos)
    modelo.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'=' * 55}\nEXPERIMENTO: {nombre}\n{'=' * 55}")
    print(f"Dimensión del vector: {X_train.shape[1]}")
    print(f"Exactitud (Accuracy): {acc:.4f}")
    print(f"F1-Score (Macro):     {f1_macro:.4f}")
    print(f"Tiempo total:         {elapsed:.2f} s\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    return {
        "Experimento": nombre,
        "Dimensión": X_train.shape[1],
        "Accuracy (%)": round(acc * 100, 2),
        "F1-Macro (%)": round(f1_macro * 100, 2),
        "Tiempo (s)": round(elapsed, 2),
    }


# ---------------------------------------------------------
# 4. Ejecución comparativa
# ---------------------------------------------------------
res_haralick = evaluar_experimento("Con Haralick (GLCM)", extraer_haralick, usar_scaler=True)
res_pixeles = evaluar_experimento("Sin Haralick (Píxeles RGB)", extraer_pixeles, usar_scaler=False)

df_resumen = pd.DataFrame([res_haralick, res_pixeles])
print("\nRESUMEN COMPARATIVO:")
print(df_resumen.to_string(index=False))