# train.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from   sklearn.model_selection import train_test_split, GridSearchCV
from   sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from   sklearn.compose import ColumnTransformer
from   sklearn.pipeline import Pipeline
from   sklearn.impute import SimpleImputer
from   sklearn.linear_model import LogisticRegression
from   sklearn.neighbors import KNeighborsClassifier
from   sklearn.ensemble import RandomForestClassifier
from   sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
import joblib

# CONFIG
DATA_PATH    = "data/survey_lung_cancer.csv"
OUTPUT_MODEL = "models/best_pipeline.joblib"
REPORTS_DIR  = "reports"

os.makedirs("models", exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1) Cargar dataset
df = pd.read_csv(DATA_PATH)

# Limpieza de nombres de columnas (eliminar espacios extra, el excel los tiene por alguna razon)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

print("Columns:", df.columns.tolist())
print("Primeras filas:")
print(df.head())

# 2) Transformar target
# LUNG_CANCER es "YES"/"NO" o similar. Convertir a 1/0
if df['LUNG_CANCER'].dtype == object:
    df['LUNG_CANCER'] = df['LUNG_CANCER'].str.strip().str.upper().map({'YES':1, 'NO':0})
    # Si existen otros valores, revisa:
    if df['LUNG_CANCER'].isnull().any():
        print("Atención: hay valores no mapeados en LUNG_CANCER:", df['LUNG_CANCER'].unique())

# 3) EDA basica
print("\n--- Descripción estadistica ---")
print(df.describe(include='all').transpose())

# Detección de nulos
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# Guardar heatmap de missing
plt.figure(figsize=(8,4))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Mapa de valores faltantes")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/missing_heatmap.png")
plt.close()

# Histograma de variables numéricas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'LUNG_CANCER' in num_cols:
    num_cols.remove('LUNG_CANCER')

# Para cada columna numérica, crea histogramas
for col in num_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribución: {col}")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/hist_{col}.png")
    plt.close()

# Conteo de variables categóricas
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
# Para cada columna categórica, crea gráficos de barras
for col in cat_cols:
    plt.figure(figsize=(6,4))
    df[col].value_counts(dropna=False).plot(kind='bar')
    plt.title(f"Conteo: {col}")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/count_{col}.png")
    plt.close()

# Matriz de correlación
plt.figure(figsize=(12,10))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag")
plt.title("Matriz de correlación (numéricas)")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/correlation_matrix.png")
plt.close()

# 4) Preparación de features y target
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# Convertir booleanos codificados como texto (ej. 'M','F') a categorías
# Identificar columnas numéricas vs categóricas
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numéricos:", numeric_features)
print("Categóricos:", categorical_features)

# Si hay columnas categóricas estas siendo >>> '1','2' codificadas como string, convertir:
for c in categorical_features:
    # if values are numeric strings, convert to numeric
    if X[c].dropna().apply(lambda v: str(v).isdigit()).all():
        X[c] = pd.to_numeric(X[c])
        numeric_features.append(c)
    else:
        # keep as categorical
        pass
# reeval categorical list
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
print("Final Numéricos:", numeric_features)
print("Final Categóricos:", categorical_features)

# 5) Preprocesador
# Imputación: para num -> median, para cat -> most_frequent (modo)
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
], remainder='drop')

# 6) Modelado: probamos LogisticRegression, KNN, RandomForest
models = {
    'logreg': LogisticRegression(max_iter=1000),
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(random_state=42)
}

results = {}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline común (preprocessor + model)
for name, estimator in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('clf', estimator)])
    print(f"\nEntrenando {name}...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        'pipeline': pipe,
        'accuracy': acc,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'confusion_matrix': cm
    }
    print(f"{name} -> acc: {acc:.4f}, recall: {rec:.4f}, f1: {f1:.4f}, roc_auc: {roc}")

# 7) Ajuste de hiperparámetros para RandomForest (GridSearchCV)
print("\nGridSearchCV para RandomForest (Pipeline completo)...")
param_grid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [None, 5, 10],
    'clf__min_samples_split': [2, 5]
}
rf_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                          ('clf', RandomForestClassifier(random_state=42))])
grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print("Mejores parámetros RF:", grid.best_params_)
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
results['rf_grid'] = {
    'pipeline': best_rf,
    'accuracy': acc,
    'recall': rec,
    'f1': f1,
    'roc_auc': roc,
    'confusion_matrix': cm,
    'best_params': grid.best_params_
}
print(f"rf_grid -> acc: {acc:.4f}, recall: {rec:.4f}, f1: {f1:.4f}, roc_auc: {roc:.4f}")

# 8) Comparación y selección del mejor modelo (por F1 o ROC_AUC)
print("\nComparación de modelos:")
for k,v in results.items():
    print(k, "-> accuracy:", round(v['accuracy'],4), "f1:", round(v['f1'],4), "recall:", round(v['recall'],4), "roc_auc:", v['roc_auc'])

# Elegimos el que tenga mejor F1 (puedes cambiar criterio)
best_key = max(results.keys(), key=lambda k: results[k]['f1'] if results[k]['f1'] is not None else -1)
best_pipeline = results[best_key]['pipeline']
print("\nMejor modelo seleccionado:", best_key)

# Guardar pipeline completo
joblib.dump(best_pipeline, OUTPUT_MODEL)
print("Pipeline guardado en:", OUTPUT_MODEL)

# Guardar métricas y matrices
import json
summary = {k: { 'accuracy':float(v['accuracy']), 'recall':float(v['recall']), 'f1':float(v['f1']), 'roc_auc': (float(v['roc_auc']) if v['roc_auc'] is not None else None)} for k,v in results.items()}
with open(f"{REPORTS_DIR}/metrics_summary.json","w") as f:
    json.dump(summary,f,indent=2)

# Guardar matriz de confusión y ROC plot para el mejor
best = results[best_key]

# Confusion matrix
cm = best['confusion_matrix']
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Matriz de confusión - {best_key}")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/confusion_{best_key}.png")
plt.close()

# ROC curve
if best.get('roc_auc') is not None:
    y_prob = best['pipeline'].predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {best_key} (AUC={best['roc_auc']:.3f})")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/roc_{best_key}.png")
    plt.close()

print("Reportes y gráficas guardadas en", REPORTS_DIR)
