from sklearn.naive_bayes import GaussianNB
import numpy as np

# Datos de entrenamiento
X_train = np.array([[3, 1500], [4, 2000], [2, 1000], [3, 1200], [4, 1800]])
y_train = np.array([200000, 250000, 150000, 180000, 220000])

# Crear y entrenar el modelo Naive Bayes para regresión
model = GaussianNB()
model.fit(X_train, y_train)

# Predicción para una nueva casa con 3 habitaciones y 1600 m²
new_data = np.array([[3, 1600]])
predicted_price = model.predict(new_data)

print(f"Predicción del precio de la casa: ${predicted_price[0]}")