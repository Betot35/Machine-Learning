#librerias

import os
import json
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import joblib

#Descrompimir el dataset

zip_file_path = "/content/IRND.zip"
extracted_dir = "/content/IRND"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

#preparacion dataset

import os
import json

# Ruta del directorio que contiene los archivos JSON
data_dir = "/content/IRND/outputs_2"

# Directorio de salida para los archivos modificados
output_dir = "/content/IRND/outputs_2_modified"
os.makedirs(output_dir, exist_ok=True)

# Número máximo de registros deseados
max_num_records = 30

for file_name in os.listdir(data_dir):
    if file_name.endswith('.json'):
        input_file_path = os.path.join(data_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)

        with open(input_file_path, 'r') as input_file:
            data = json.load(input_file)

        # Limitar a 30 registros
        data['data'] = data['data'][:max_num_records]

        # Ajustar longitud de 'angles', 'dists', 'counts_left', 'counts_right', y 'horn' a 8
        for record in data['data']:
            record['angles'] = record.get('angles', [])[:8]
            record['dists'] = record.get('dists', [])[:8]
            record['counts_left'] = record.get('counts_left', 0)
            record['counts_right'] = record.get('counts_right', 0)
            record['horn'] = record.get('horn', 0)

        # Rellenar 'angles', 'dists' con datos similares si es necesario
        for record in data['data']:
            record['angles'] += [record['angles'][0]] * (8 - len(record['angles']))
            record['dists'] += [record['dists'][0]] * (8 - len(record['dists']))

        # Rellenar con registros vacíos si es necesario
        while len(data['data']) < max_num_records:
            data['data'].append({'direction': '', 'pose': {}, 'brake': 0, 'angles': [0.0] * 8, 'dists': [0.0] * 8, 'counts_left': 0, 'horn': 0, 'counts_right': 0})

        # Guardar el archivo modificado
        with open(output_file_path, 'w') as output_file:
            json.dump(data, output_file, indent=2)

print("Archivos modificados guardados en:", output_dir)

#Visualizacion

import os

# Ruta al directorio que contiene los archivos outputs_2
outputs_2_dir = "/content/IRND/outputs_2_modified"

# Leer algunos ejemplos de archivos JSON y mostrar información
for i in range(1, 6):  # Visualizar información de los primeros 5 archivos
    file_path = os.path.join(outputs_2_dir, f"{i}.json")
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(f"Información del archivo {i}:")
        print(f"Keys presentes: {list(data.keys())}")
        print(f"Número de elementos en 'data': {len(data['data'])}")
        print(f"Ejemplo de un elemento en 'data': {data['data'][0]}")
        print("\n" + "-"*40 + "\n")

#Lectura de archivos JSON y preparación de datos

# Preparar la lectura de outputs
data_dir = "/content/IRND"
output_dir = os.path.join(data_dir, 'outputs_2_modified')

data_outputs_2 = []
etiquetas_reales = []

for file_name in os.listdir(output_dir):
    if file_name.endswith('.json'):
        file_path = os.path.join(output_dir, file_name)
        try:
            with open(file_path, 'r') as f:
                current_data = json.load(f)
                # Verificar la longitud de 'data' y rellenar si es necesario
                while len(current_data['data']) < 30:
                    current_data['data'].append(current_data['data'][0])
                data_outputs_2.append(current_data)
                etiquetas_reales.extend([file_name.split('_')[0]] * len(current_data['data']))
        except json.JSONDecodeError as e:
            print(f"Error al leer {file_name}: {e}")

print("Número de archivos en outputs_2:", len(data_outputs_2))

#Características y etiquetas

# Características (X): Usar distancias como características
X = [item['dists'] for data_output in data_outputs_2 for item in data_output['data']]

# Codificación de etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(etiquetas_reales)

#Aprendizaje no supervisado (K-Means)

# Aprendizaje No Supervisado (Ejemplo con K-Means)
X_unsupervised = [item['dists'] for data_output in data_outputs_2 for item in data_output['data']]
kmeans = KMeans(n_clusters=2, random_state=42)
y_unsupervised = kmeans.fit_predict(X_unsupervised)

#División en conjuntos de entrenamiento y prueba para aprendizaje supervisado

# División en Conjuntos de Entrenamiento y Prueba para aprendizaje supervisado
X_train_supervised, X_test_supervised, y_train_encoded_supervised, y_test_encoded_supervised = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

#Entrenamiento del modelo supervisado y persistencia del modelo

# Entrenar el modelo supervisado
model_supervised = DecisionTreeClassifier(random_state=42)
model_supervised.fit(X_train_supervised, y_train_encoded_supervised)

# Guardar el modelo entrenado
joblib.dump(model_supervised, model_path)
print("Modelo entrenado guardado con éxito.")

#Evaluación del modelo supervisado

# Evaluación del Modelo Supervisado
y_pred_supervised = model_supervised.predict(X_test_supervised)
accuracy_supervised = accuracy_score(y_test_encoded_supervised, y_pred_supervised)
conf_matrix_supervised = confusion_matrix(y_test_encoded_supervised, y_pred_supervised, labels=le.transform(etiquetas_reales))

#Acciones según la categoría del aprendizaje no supervisado

# Acciones según Categoría del Aprendizaje No Supervisado
points_unsupervised = 0
choca_repetidamente = 0

for features, true_label_unsupervised, true_label_supervised in zip(X_unsupervised, y_unsupervised, y_test_encoded_supervised):
    decision_unsupervised = kmeans.predict([features])[0]

    if decision_unsupervised == 0:  # RODEAR
        print("RODEAR - Acción de rodear")
        if choca_repetidamente > 2:
            decision_unsupervised = 1  # Cambiar a RETROCEDER
            choca_repetidamente = 0
        else:
            choca_repetidamente += 1
    else:  # RETROCEDER
        print("RETROCEDER - Acción de retroceder y doblar a la derecha")
        choca_repetidamente = 0

    decision_supervised = model_supervised.predict([features])[0]
    if decision_supervised == true_label_supervised:
        points_unsupervised += 10
    else:
        points_unsupervised -= 20

#Resultados finales

print("\nResultados del Modelo Supervisado:")

print(f'Accuracy (Supervised): {accuracy_supervised}')

print("Matriz de Confusión (Supervised):")

print(conf_matrix_supervised)

print(f'Puntos finales (Supervised): {points_unsupervised}')
