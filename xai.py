import shap
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# Load model
model = load_model("/kaggle/working/xai_calorify/best_calorify.h5")

# Load image
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

images, filenames = load_images_from_folder("/kaggle/working/xai_calorify/test")
print(f"Loaded {len(images)} images.")

background = images[:10]
to_explain = images[10:11]

# Gunakan masker image baru
masker = shap.maskers.Image("inpaint_telea", images[0].shape)

# Buat explainer dengan masker image
explainer = shap.Explainer(model, masker)

# Hitung SHAP values
shap_values = explainer(to_explain)

# Plot hasil
shap.image_plot(shap_values.values, to_explain)
