# Importowanie potrzebnych bibliotek i modułów
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import keras
from keras import layers

# Tworzenie zbioru danych
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

# Podział danych na zbiory uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie modelu sieci neuronowej
inputs = keras.Input(shape=(2,))
hidden1 = layers.Dense(64, activation='relu')(inputs)
hidden2 = layers.Dense(64, activation='relu')(hidden1)
outputs = layers.Dense(1, activation='sigmoid')(hidden2)
model = keras.Model(inputs=inputs, outputs=outputs)

# Wizualizacja modelu
model.summary()
keras.utils.plot_model(model, "my_first_model.png", show_shapes=True)

# Kompilacja modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Ocena modelu na danych testowych
_, test_accuracy = model.evaluate(X_test, y_test)
_, train_accuracy = model.evaluate(X_train, y_train)

# Rysowanie granicy decyzyjnej dla sieci neuronowej
plt.figure(figsize=(8, 6))
# Rysowanie punktów danych treningowych
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')
# Rysowanie punktów danych testowych
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')

# Tworzenie siatki punktów dla wykresu konturu decyzji
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))

# Obliczanie wyników dla wszystkich punktów siatki jednocześnie
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid_points)
Z = np.round(Z).reshape(xx.shape)

# Rysowanie granicy decyzyjnej na wykresie konturu
plt.contourf(xx, yy, Z, alpha=0.3)

# Ustawienie tytułu wykresu
plt.title(f"Neural Network\nTest accuracy: {test_accuracy:.3f} and train accuracy: {train_accuracy:.3f}")
# Dodanie legendy
plt.legend()
# Wyświetlenie wykresu
plt.tight_layout()
plt.show()
