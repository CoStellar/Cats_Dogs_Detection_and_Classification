import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from tensorflow.keras.preprocessing import image as keras_image
import tkinter as tk
from tkinter import filedialog, messagebox

# Wyłączanie opcji OneDNN w TensorFlow dla lepszej wydajności
TF_ENABLE_ONEDNN_OPTS = 0
# Tłumienie ostrzeżeń TensorFlow
tf.get_logger().setLevel('ERROR')

# Ładowanie modelu do detekcji obiektów z TensorFlow Hub
model_handle = r'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1'
detection_model = hub.load(model_handle)

# Ładowanie modelu klasyfikacyjnego (upewnij się, że to jest ścieżka do twojego wytrenowanego modelu)
classification_model = tf.keras.models.load_model(r'./trained_model.keras')

# Ładowanie mapy etykiet
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Funkcja do ładowania obrazu z pliku lokalnego
def load_image_from_local_file(file_path):
    try:
        image = Image.open(file_path).convert('RGB')
        return np.array(image)
    except Exception as e:
        raise IOError(f"Nie można otworzyć pliku obrazu: {e}")

# Definiowanie nazw klas (upewnij się, że są zgodne z etykietami użytymi podczas trenowania modelu)
class_names = ['Kot', 'Pies']  # Zmodyfikuj tę listę zgodnie z etykietami w twoim zestawie danych
icon = Image.open(r'./kotvspies.png')

# Inicjalizacja interfejsu graficznego (GUI)
root = tk.Tk()
root.title("Detekcja Psów i Kotów")

# Ustawienie ikony aplikacji
icon_photo = ImageTk.PhotoImage(icon)
root.iconphoto(True, icon_photo)

# Etykieta dla liczby kotów
label_koty = tk.Label(root, text="Liczba kotów: 0")
label_koty.pack()

# Etykieta dla liczby psów
label_psy = tk.Label(root, text="Liczba psów: 0")
label_psy.pack()

# Tworzenie ramki do umieszczenia płótna i pasków przewijania
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Tworzenie widgetu płótna
canvas = tk.Canvas(frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Dodanie pionowego i poziomego paska przewijania do płótna
v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

# Konfiguracja płótna do używania pasków przewijania
canvas.configure(yscrollcommand=v_scrollbar.set)
canvas.configure(xscrollcommand=h_scrollbar.set)

# Tworzenie wewnętrznej ramki na płótnie do umieszczenia obrazu
img_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=img_frame, anchor='nw')

# Konfiguracja regionu przewijania płótna, gdy zmienia się rozmiar ramki
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

img_frame.bind("<Configure>", on_frame_configure)

# Funkcja do aktualizacji etykiety obrazu z obsługą przewijania
def update_image_label(image):
    global img_label, final_image
    final_image = ImageTk.PhotoImage(image)
    
    if 'img_label' in globals():
        img_label.destroy()
    
    img_label = tk.Label(img_frame, image=final_image)
    img_label.pack()

# Funkcja do przetwarzania obrazu wybranego przez użytkownika
def process_image():
    global img_label, final_image
    file_path = filedialog.askopenfilename(filetypes=[("Pliki obrazów", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    try:
        image_np = load_image_from_local_file(file_path)
        koty = 0
        psy = 0

        # Przeprowadzanie detekcji obiektów
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detection_model(input_tensor)

        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        num_detections = int(detections['num_detections'])

        # Filtracja wyników dla kotów i psów
        category_ids = [17, 18]  # ID z COCO dla kota i psa
        min_confidence_threshold = 0.4

        filtered_indices = np.where((scores > min_confidence_threshold) & (np.isin(classes, category_ids)))
        filtered_boxes = boxes[filtered_indices]
        filtered_classes = classes[filtered_indices]
        filtered_scores = scores[filtered_indices]

        # Pobieranie wymiarów obrazu
        img_height, img_width, img_channel = image_np.shape

        # Funkcja do nakładania filtra kolorów
        def apply_color_filter(image, color):
            filter = np.zeros_like(image)
            filter[:, :, 0] = color[0]
            filter[:, :, 1] = color[1]
            filter[:, :, 2] = color[2]
            filtered_image = cv2.addWeighted(image, 0.7, filter, 0.2, 0)
            return filtered_image

        color_map = {'Kot': [153, 60, 54], 'Pies': [102, 177, 227]}

        for i in range(len(filtered_boxes)):
            box = filtered_boxes[i]
            ymin, xmin, ymax, xmax = box
            
            x_up = int(xmin * img_width)
            y_up = int(ymin * img_height)
            x_down = int(xmax * img_width)
            y_down = int(ymax * img_height)
            
            cropped_img = image_np[y_up:y_down, x_up:x_down]

            img_resized = cv2.resize(cropped_img, (160, 160))
            img_array = keras_image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = classification_model.predict(img_array).flatten()
            prediction = tf.nn.sigmoid(prediction)

            if prediction.numpy()[0] < 0.5:
                predicted_class = 'Kot'
                confidence_score = (prediction.numpy()[0] - 0.5) / (-0.5)
            else:
                predicted_class = 'Pies'
                confidence_score = (prediction.numpy()[0] - 0.5) / 0.5

            # Aktualizujemy liczbę kotów lub psów tylko jeśli confidence score jest większe niż 0.5
            if confidence_score > 0.8:
                if predicted_class == 'Kot':
                    koty += 1
                elif predicted_class == 'Pies':
                    psy += 1

                filtered_img = apply_color_filter(cropped_img, color_map[predicted_class])

                image_np[y_up:y_down, x_up:x_down] = filtered_img

                text_label = f'{predicted_class.capitalize()}: {confidence_score * 100:.2f}%'
                text_size = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x_up + (x_down - x_up - text_size[0]) // 2
                text_y = y_up + 10
                text_x = max(0, min(text_x, img_width - text_size[0]))
                text_y = max(text_size[1], text_y)

                cv2.putText(image_np, text_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Aktualizacja etykiety dla liczby kotów
        label_koty.configure(text=f"Liczba kotów: {koty}")

        # Aktualizacja etykiety dla liczby psów
        label_psy.configure(text=f"Liczba psów: {psy}")

        final_image = Image.fromarray(image_np)
        
        # Aktualizacja etykiety obrazu nowym obrazem
        update_image_label(final_image)

        # Dostosowanie rozmiaru wewnętrznej ramki do rozmiaru obrazu
        img_frame.update_idletasks()
        img_frame.config(width=final_image.width(), height=final_image.height())
        canvas.config(scrollregion=canvas.bbox("all"))

    except Exception as e:
        messagebox.showerror("Błąd", str(e))

# Przycisk do wybrania i przetworzenia obrazu
btn_select_image = tk.Button(root, text="Wybierz obraz", command=process_image)
btn_select_image.pack()

# Uruchomienie pętli głównej interfejsu graficznego
root.mainloop()
