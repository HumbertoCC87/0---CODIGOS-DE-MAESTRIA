# Clase 11 · Clasificación supervisada de textos
# Autor académico: M.C. Pablo Ricardo Sánchez Gómez
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RUTA = Path("dataset_intenciones_clase11.csv")
SALIDA = Path("salidas_clase11")
SALIDA.mkdir(exist_ok=True)

# Irregularidad intencional: el catálogo contiene una variante con guion.
# El estudiante debe detectar su efecto y completar la normalización.
MAPA_ETIQUETAS = {
    "consulta_academica": "consulta_academica",
    "tramite_escolar": "tramite_escolar",
    "pago": "pago",
    "soporte_tecnico": "soporte_tecnico",
    "soporte-tecnico": "soporte_tecnico",  # Normaliza la variante con guion
    "cancelacion": "cancelacion",
}

def normalizar_etiqueta(valor: str) -> str:
    valor = str(valor).strip().lower()
    return MAPA_ETIQUETAS.get(valor, valor)

def cargar_datos(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    requeridas = {"id", "texto", "etiqueta"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes: {sorted(faltantes)}")
    if df[["texto", "etiqueta"]].isna().any().any():
        raise ValueError("Existen textos o etiquetas vacías")
    df["etiqueta_original"] = df["etiqueta"]
    return df

def entrenar_y_evaluar(df: pd.DataFrame, nombre: str):
    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["etiqueta"], test_size=0.30,
        random_state=42, stratify=df["etiqueta"]
    )
    modelo = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1200, class_weight="balanced"))
    ])
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    etiquetas = sorted(df["etiqueta"].unique())
    reporte = pd.DataFrame(classification_report(
        y_test, pred, labels=etiquetas, output_dict=True, zero_division=0
    )).T
    matriz = pd.DataFrame(confusion_matrix(y_test, pred, labels=etiquetas),
                        index=[f"real_{e}" for e in etiquetas],
                        columns=[f"pred_{e}" for e in etiquetas])
    errores = pd.DataFrame({"texto": X_test.values, "real": y_test.values, "prediccion": pred})
    errores["correcto"] = errores["real"] == errores["prediccion"]
    reporte.to_csv(SALIDA / f"reporte_{nombre}.csv", encoding="utf-8-sig")
    matriz.to_csv(SALIDA / f"matriz_{nombre}.csv", encoding="utf-8-sig")
    errores.to_csv(SALIDA / f"errores_{nombre}.csv", index=False, encoding="utf-8-sig")
    print(f"{nombre}: clases={len(etiquetas)} accuracy={accuracy_score(y_test,pred):.3f}")
    print(matriz)

if __name__ == "__main__":
    df = cargar_datos(RUTA)
    print("Catálogo original:")
    print(df["etiqueta"].value_counts())
    entrenar_y_evaluar(df.copy(), "antes")

    df["etiqueta"] = df["etiqueta"].apply(normalizar_etiqueta)
    df[["id", "etiqueta_original", "etiqueta"]].to_csv(
        SALIDA / "trazabilidad_etiquetas.csv", index=False, encoding="utf-8-sig"
    )
    print("\nCatálogo normalizado:")
    print(df["etiqueta"].value_counts())
    entrenar_y_evaluar(df, "despues")
