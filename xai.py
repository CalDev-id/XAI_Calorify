import tensorflow as tf
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# =============================
# 1. Load Trained Model
# =============================
model = tf.keras.models.load_model('best_calorify.h5')  # ubah jika nama model berbeda
model.summary()

# =============================
# 2. Load Test Dataset
# =============================
# Asumsi struktur: dataset/test/<class_name>/*.jpg
img_size = (224, 224)
batch_size = 32

test_dataset = image_dataset_from_directory(
    'test',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True  # agar gambar yang diambil acak
)

class_names = test_dataset.class_names
print("Class Names:", class_names)

# Normalisasi data seperti saat training
def preprocess(images, labels):
    return preprocess_input(images), labels

test_dataset = test_dataset.map(preprocess)

# Ambil sample batch (1 batch = 32 gambar)
for test_images, test_labels in test_dataset.take(1):
    break

# =============================
# 3. Ambil Background & Test Data untuk SHAP
# =============================
# Ambil 100 gambar background dan 5 gambar test untuk dijelaskan
background = test_images[:100]
test_to_explain = test_images[100:105]

# =============================
# 4. Jalankan SHAP GradientExplainer
# =============================
explainer = shap.GradientExplainer(model, background)

# Dapatkan SHAP values
shap_values = explainer.shap_values(test_to_explain)

# =============================
# 5. Visualisasi SHAP
# =============================
# Visualisasi untuk setiap gambar
for i in range(len(test_to_explain)):
    print(f"\nPrediksi untuk gambar ke-{i+1}:")
    pred = model.predict(np.expand_dims(test_to_explain[i], axis=0))
    top_class = np.argmax(pred[0])
    print(f"  Kelas Prediksi: {class_names[top_class]} (Confidence: {pred[0][top_class]:.2f})")

    # Visualisasi SHAP
    shap.image_plot([s[i] for s in shap_values], [test_to_explain[i].numpy()])
