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
model = tf.keras.models.load_model('best_calorify.h5')  # Ganti jika nama model berbeda
model.summary()

# =============================
# 2. Load Test Dataset
# =============================
img_size = (224, 224)
batch_size = 32

test_dataset = image_dataset_from_directory(
    'test',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
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
# Konversi dari Tensor ke NumPy array
test_images_np = test_images.numpy()

# Pastikan dataset cukup besar
if test_images_np.shape[0] < 105:
    raise ValueError(f"Dataset hanya memiliki {test_images_np.shape[0]} gambar. SHAP memerlukan minimal 105 gambar.")

background = test_images_np[:100]
test_to_explain = test_images_np[100:105]

# =============================
# 4. Jalankan SHAP GradientExplainer
# =============================
explainer = shap.GradientExplainer(model, background)

# Dapatkan SHAP values
shap_values = explainer.shap_values(test_to_explain)

# =============================
# 5. Visualisasi SHAP
# =============================
for i in range(len(test_to_explain)):
    img = test_to_explain[i]
    pred = model.predict(np.expand_dims(img, axis=0))
    predicted_class = np.argmax(pred)
    confidence = pred[0][predicted_class]

    print(f"Gambar ke-{i+1} diprediksi sebagai: {class_names[predicted_class]} (Confidence: {confidence:.2f})")

    shap.image_plot([s[i] for s in shap_values], [img])
