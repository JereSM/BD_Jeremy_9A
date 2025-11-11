import pandas as pd
from   sklearn.model_selection import train_test_split
from   sklearn.preprocessing import LabelEncoder
from   sklearn.linear_model import LogisticRegression
from   sklearn.metrics import classification_report, confusion_matrix
import joblib

# Cargar el dataset
data = pd.read_csv('data/survey_lung_cancer.csv') 

# Convertir texto a números
le = LabelEncoder()
data['GENDER'] = le.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])

# Separar variables independientes (X) y dependiente (y)
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

#Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Crear modelo balanceado
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# Guardar el modelo
joblib.dump(model, 'lung_cancer_model.pkl')
print("Modelo guardado correctamente como lung_cancer_model.pkl")
