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
model.summary()

# 3. Fungsi untuk memuat dan memproses gambar dari folder
def load_images_from_folder(folder_path, target_size=(224, 224)):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            images.append(img_array)
            filenames.append(filename)
    return np.array(images), filenames

# 4. Load gambar dari folder 'test/'
images, filenames = load_images_from_folder("/kaggle/working/xai_calorify/test")
print(f"📷 Loaded {len(images)} images from 'test/'")

# 5. Validasi jumlah gambar
if len(images) < 2:
    raise ValueError("❌ Setidaknya butuh 2 gambar: 1 untuk background, 1 untuk dijelaskan.")

# 6. Pilih background dan gambar yang ingin dijelaskan
background = images[:10]
to_explain = images[10:11]

# 7. Gunakan SHAP DeepExplainer
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(to_explain)

# 8. Prediksi kelas
preds = model.predict(to_explain)
predicted_class = np.argmax(preds[0])
print(f"📊 Predicted class index: {predicted_class}")

# 9. Visualisasi SHAP untuk kelas tersebut
shap.image_plot([shap_values[predicted_class]], to_explain)
