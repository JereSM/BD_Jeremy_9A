### This was for smt else, los reportes se generan desde train.py ###
import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIGURACIÓN ===
DATASET_PATH = "data/survey_lung_cancer.csv"  
REPORTS_DIR = "reports_2" # Carpeta para guardar reportes_2 (para que no choquen con los de train.py :^) )

# Crear carpeta si no existeeeeeeeeeeeeee
os.makedirs(REPORTS_DIR, exist_ok=True)

print("Cargando dataset...")
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset cargado correctamente con {df.shape[0]} filas y {df.shape[1]} columnas.")
except FileNotFoundError:
    print("Error: No se encontró el archivo del dataset. Verifica el nombre y la ruta.")
    exit()

# === ANÁLISIS DESCRIPTIVO ===
desc = df.describe(include='all')
desc.to_csv(os.path.join(REPORTS_DIR, "dataset_description.csv"))
print("Archivo 'dataset_description.csv' generado.")

# === GRÁFICAS ===

# Distribución de edadesss
plt.figure(figsize=(8, 5))
df["AGE"].hist(bins=20, color='skyblue', edgecolor='black')
plt.title("Distribución de edades")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.savefig(os.path.join(REPORTS_DIR, "age_distribution.png"))
plt.close()

# Distribución por género
plt.figure(figsize=(6, 4))
df["GENDER"].value_counts().plot(kind="bar", color=['lightcoral', 'lightblue'])
plt.title("Distribución por género")
plt.xlabel("Género")
plt.ylabel("Cantidad")
plt.savefig(os.path.join(REPORTS_DIR, "gender_distribution.png"))
plt.close()

# Conteo de fumadores
plt.figure(figsize=(6, 4))
df["SMOKING"].value_counts().plot(kind="bar", color=['orange', 'green'])
plt.title("Personas fumadoras vs no fumadoras")
plt.xlabel("2 = Sí, 1 = No")
plt.ylabel("Cantidad")
plt.savefig(os.path.join(REPORTS_DIR, "smoking_status.png"))
plt.close()

# Correlación entre variables
plt.figure(figsize=(12, 10))
corr = df.corr(numeric_only=True)
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="Correlación")
plt.title("Mapa de correlaciones entre variables")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "correlation_matrix.png"))
plt.close()

# Conteo de casos positivos vs negativos (si existe columna 'LUNG_CANCER')
if "LUNG_CANCER" in df.columns:
    plt.figure(figsize=(6, 4))
    df["LUNG_CANCER"].value_counts().plot(kind="bar", color=['red', 'green'])
    plt.title("Distribución de diagnóstico de cáncer de pulmón")
    plt.xlabel("LUNG_CANCER")
    plt.ylabel("Cantidad")
    plt.savefig(os.path.join(REPORTS_DIR, "lung_cancer_distribution.png"))
    plt.close()

print("Gráficas generadas correctamente en la carpeta 'reports/'.")
