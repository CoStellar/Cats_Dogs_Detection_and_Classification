import pandas as pd
import numpy as np
import tensorflow as tf
import os 
import cv2
import matplotlib.pylab as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Wyświetlenie wersji TensorFlow i informacji o dostępności GPU
print("Wersja TensorFlow:", tf.__version__)
print("GPU jest", "dostępne" if tf.config.list_physical_devices('GPU') else "NIEDOSTĘPNE")

# Zdefiniowanie ścieżek do katalogów z danymi treningowymi i walidacyjnymi
validation_dir = 'archive/dataset/test_set'
train_dir = 'archive/dataset/training_set'

# Ustawienie wielkości partii danych i rozmiaru obrazów
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# Ładowanie zestawu danych treningowych
train_dataset = image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

# Ładowanie zestawu danych walidacyjnych
validation_dataset = image_dataset_from_directory(
    validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

# Pobieranie nazw klas z zestawu danych treningowych
class_names = train_dataset.class_names

# Wyświetlanie kilku obrazów z zestawu danych treningowych z etykietami klas
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Dzielenie zestawu walidacyjnego na walidacyjny i testowy
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
print('Liczba partii walidacyjnych: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Liczba partii testowych: %d' % tf.data.experimental.cardinality(test_dataset))

# Przygotowanie danych do wydajnego ładowania
AUTOTUNE = tf.data.AUTOTUNE

# Prefetchowanie zestawów danych
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Definiowanie augmentacji danych - losowe odwracanie i obracanie obrazów
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

# Wizualizacja przykładowych augmentacji danych
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
plt.show()

# Definiowanie funkcji przetwarzania wstępnego dla modelu MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Normalizacja obrazów (rescale) w zakresie od -1 do 1
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

# Pobranie bieżącego katalogu roboczego
current_directory = os.getcwd()

# Utworzenie ścieżki do pliku modelu w bieżącym katalogu
model_path = os.path.join(current_directory, 'base_model')  # Upewnij się, że wskazuje na odpowiednią ścieżkę

# Definiowanie kształtu obrazów dla modelu MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# Przetworzenie jednej partii obrazów przez model bazowy
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# Zablokowanie warstw modelu bazowego, aby nie były trenowane
base_model.trainable = False

# Wyświetlenie podsumowania modelu bazowego
base_model.summary()

# Definiowanie warstwy globalnego uśredniania
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# Definiowanie warstwy predykcyjnej
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# Budowanie pełnego modelu z warstwą augmentacji, przetwarzania wstępnego, modelem bazowym i warstwą predykcyjną
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)  # Dropout to avoid overfitting
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# Kompilowanie modelu z optymalizatorem Adam i funkcją straty binarnej entropii krzyżowej
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Wyświetlenie podsumowania modelu
model.summary()

# Wyświetlenie liczby trenowalnych zmiennych (powinno być 0, ponieważ model bazowy jest zamrożony)
len(model.trainable_variables)

# Ustawienie liczby epok treningu na 30
initial_epochs = 30  
loss0, accuracy0 = model.evaluate(validation_dataset)

# Wyświetlenie początkowych wartości straty i dokładności
print("Początkowa strata: {:.2f}".format(loss0))
print("Początkowa dokładność: {:.2f}".format(accuracy0))

# Trening modelu z zapisaniem najlepszego modelu na podstawie walidacji i wczesnym zatrzymaniem
history = model.fit(
    train_dataset,
    epochs=initial_epochs,
    validation_data=validation_dataset,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
)

# Rysowanie wykresów dokładności i straty w trakcie treningu
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Dokładność treningowa')
plt.plot(val_acc, label='Dokładność walidacyjna')
plt.legend(loc='lower right')
plt.ylabel('Dokładność')
plt.ylim([min(plt.ylim()), 1])
plt.title('Dokładność treningu i walidacji')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Strata treningowa')
plt.plot(val_loss, label='Strata walidacyjna')
plt.legend(loc='upper right')
plt.ylabel('Entropia krzyżowa')
plt.ylim([0, 1.0])
plt.title('Strata treningu i walidacji')
plt.xlabel('epoka')
plt.show()

# Wyświetlenie liczby warstw w modelu bazowym
print("Liczba warstw w modelu bazowym: ", len(base_model.layers))

# Wyświetlenie podsumowania modelu
model.summary()

# Wyświetlenie liczby trenowalnych zmiennych (powinno być 0)
len(model.trainable_variables)

# Ocena modelu na zestawie testowym
loss, accuracy = model.evaluate(test_dataset)
print('Dokładność testowa:', accuracy)

# Pobranie jednej partii obrazów z zestawu testowego
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Aplikacja funkcji sigmoidalnej do wyników modelu, ponieważ model zwraca logity
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

# Wyświetlenie przewidywań i rzeczywistych etykiet dla partii danych testowych
print('Przewidywania:\n', predictions.numpy())
print('Etykiety:\n', label_batch)

# Wizualizacja wyników predykcji
plt.figure(figsize=(10, 10))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")

# Zapisanie wytrenowanego modelu do pliku
model.save('trained_model.keras')
