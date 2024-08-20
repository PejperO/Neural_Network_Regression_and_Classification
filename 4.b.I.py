import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from keras import Model
from keras.src.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


# Załadowanie danych
XX = 9
with open(f"Dane/dane{XX}.txt", "r") as file:
    data = file.readlines()

x_data, y_data = zip(*[map(float, line.split()) for line in data])

# Normalizacja danych
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_data_normalized = scaler_x.fit_transform(np.array(x_data).reshape(-1, 1))
y_data_normalized = scaler_y.fit_transform(np.array(y_data).reshape(-1, 1))

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_data_normalized, y_data_normalized, test_size=0.2, random_state=42)

# Definiowanie modelu z użyciem Keras Functional API
input_size = 1
hidden_size = 100  # Możesz zmienić liczbę neuronów, aby skrócić czas treningu
hidden_size2 = 50  # Dodanie drugiej warstwy ukrytej
output_size = 1

inputs = Input(shape=(input_size,))
hidden1 = Dense(hidden_size, activation='relu')(inputs)
hidden2 = Dense(hidden_size2, activation='relu')(hidden1)
outputs = Dense(output_size, activation='linear')(hidden2)
model = Model(inputs=inputs, outputs=outputs)

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Wczesne zatrzymywanie
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

# Obliczenie przewidywań
output_train = model.predict(X_train)
output_test = model.predict(X_test)

# Obliczenie wartości MSE i R^2
mse_train = mean_squared_error(y_train, output_train)
r2_train = r2_score(y_train, output_train)
mse_test = mean_squared_error(y_test, output_test)
r2_test = r2_score(y_test, output_test)

# Wykresy
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'MSE vs. Epochs, Final MSE Train:{round(mse_train, 3)}, Final MSE Test:{round(mse_test, 3)}')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'R^2 Train:{round(r2_train, 2)}, R^2 Test:{round(r2_test, 2)}')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.scatter(X_train, output_train, color='green', label='Predictions', marker='s')
plt.scatter(X_test, output_test, color='green', label='Predictions', marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Neurons={hidden_size}, Epochs={len(history.history["loss"])}, learn_r=0.001')
plt.legend()

# Zapisywanie wykresu
now = datetime.now()
nazwa_pliku = now.strftime(f'DataSet_{XX}_%Y%m%d_%H%M%S_neu_{hidden_size}_epch_{len(history.history["loss"])}.png')
plt.savefig(nazwa_pliku)
plt.show()
