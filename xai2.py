# 1. Import Library
import os
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 2. Load model utuh dari .h5
model = load_model("/kaggle/working/xai_calorify/best_calorify.h5")
print("✅ Model loaded successfully!")
model.summary()  # Opsional: lihat struktur model

# 3. Fungsi untuk memuat dan memproses gambar dari folder
def load_images_from_folder(folder_path, target_size=(224, 224)):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img) / 255.0  # Normalisasi
            images.append(img_array)
            filenames.append(filename)
    return np.array(images), filenames

# 4. Load gambar dari folder 'test/'
images, filenames = load_images_from_folder("/kaggle/working/xai_calorify/test")

print(f"📷 Loaded {len(images)} images from 'test/'")

# 5. Pilih background dan gambar yang ingin dijelaskan
background = images[:10]           # SHAP background (misal 10 gambar pertama)
to_explain = images[10:11]         # Gambar ke-11 akan dijelaskan

# 6. Inisialisasi SHAP explainer
explainer = shap.DeepExplainer(model, background)

# 7. Hitung SHAP values
shap_values = explainer.shap_values(to_explain)

# 8. Visualisasi hasil SHAP
shap.image_plot(shap_values, to_explain)
