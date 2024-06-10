import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pylab as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn import metrics
from tensorflow.python.client import device_lib 

# Wyświetlenie dostępnych urządzeń lokalnych
print(device_lib.list_local_devices())

# Generowanie losowych kolorów dla klas
COLORS = np.random.uniform(0, 255, size=(2, 3))

# Wyświetlenie wersji TensorFlow i informacji o dostępności GPU
print("Wersja TensorFlow:", tf.__version__)
print("GPU jest", "dostępne" if tf.config.list_physical_devices('GPU') else "NIEDOSTĘPNE")

# Zdefiniowanie ścieżek do katalogów z danymi walidacyjnymi i treningowymi
validation_dir = 'archive/dataset/test_set'
train_dir = 'archive/dataset/training_set'

# Sprawdzenie, czy katalogi istnieją
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Nie znaleziono katalogu treningowego: {train_dir}")
if not os.path.exists(validation_dir):
    raise FileNotFoundError(f"Nie znaleziono katalogu walidacyjnego: {validation_dir}")

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# Wczytanie zbioru danych treningowych
train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# Wczytanie zbioru danych walidacyjnych
validation_dataset = image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# Pobranie nazw klas ze zbioru danych treningowych
class_names = train_dataset.class_names

# Liczba batchy w zbiorze walidacyjnym
val_batches = tf.data.experimental.cardinality(validation_dataset)
print(val_batches)

# Wyodrębnienie części zbioru walidacyjnego jako zbioru testowego
test_dataset = validation_dataset.take(val_batches // 5)
metrics_dataset = validation_dataset
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Liczba batchy w zbiorze walidacyjnym: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Liczba batchy w zbiorze testowym: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE

# Ulepszenie wydajności poprzez prefetching
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
metrics_dataset = metrics_dataset.prefetch(buffer_size=AUTOTUNE)

# Definicja augmentacji danych
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

# Definicja funkcji przetwarzania wstępnego dla MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Definicja skalowania obrazów
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

IMG_SHAPE = IMG_SIZE + (3,)

# Wczytanie wytrenowanego modelu
model = tf.keras.models.load_model('trained_model.keras')

# Pobranie batcha obrazów ze zbioru testowego
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
image_metrics, label_metrics = metrics_dataset.as_numpy_iterator().next()

# Predykcje na batchu obrazów
predictions = model.predict_on_batch(image_batch).flatten()
metrics_predictions = model.predict_on_batch(image_metrics).flatten()

# Zastosowanie funkcji sigmoid do predykcji, ponieważ model zwraca logity
predictions = tf.nn.sigmoid(predictions)
metrics_predictions = tf.nn.sigmoid(metrics_predictions)
predictions = tf.where(predictions < 0.5, 0, 1)
metrics_predictions = tf.where(metrics_predictions < 0.5, 0, 1)

# Wyświetlenie predykcji i rzeczywistych etykiet
print('Predykcje:\n', predictions.numpy())
print('Etykiety:\n', label_batch)

# Obliczenie i wyświetlenie metryk dla jednego batcha
print("Metryki dla jednego batcha:")
print("Dokładność: {:.3f}".format(metrics.accuracy_score(label_batch, predictions)))
print("Zrównoważona dokładność: {:.3f}".format(metrics.balanced_accuracy_score(label_batch, predictions)))
print("Średnia precyzja: {:.3f}".format(metrics.average_precision_score(label_batch, predictions)))
print('\n')

# Obliczenie i wyświetlenie metryk dla zbioru testowego
print("Metryki dla zbioru testowego:")
print("Dokładność: {:.3f}".format(metrics.accuracy_score(label_metrics, metrics_predictions)))
print("Zrównoważona dokładność: {:.3f}".format(metrics.balanced_accuracy_score(label_metrics, metrics_predictions)))
print("Średnia precyzja: {:.3f}".format(metrics.average_precision_score(label_metrics, metrics_predictions)))

# Wyświetlenie obrazów z predykcjami
plt.figure(figsize=(10, 10))
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")
plt.show()
