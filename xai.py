# import tensorflow as tf
# import shap
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# model = tf.keras.models.load_model('best_calorify.h5')
# model.summary()

# img_size = (224, 224)
# batch_size = 32

# test_dataset = image_dataset_from_directory(
#     'test',
#     image_size=img_size,
#     batch_size=batch_size,
#     shuffle=True
# )

# class_names = test_dataset.class_names
# print("Class Names:", class_names)

# def preprocess(images, labels):
#     return preprocess_input(images), labels

# test_dataset = test_dataset.map(preprocess)

# # Ambil 1 batch (32 gambar)
# for test_images, test_labels in test_dataset.take(1):
#     break

# test_images_np = test_images.numpy()

# if test_images_np.shape[0] < 32:
#     raise ValueError(f"Hanya ada {test_images_np.shape[0]} gambar. Dibutuhkan minimal 32.")

# background = test_images_np[:27]
# test_to_explain = test_images_np[27:32]


# explainer = shap.GradientExplainer(model, background)
# shap_values = explainer.shap_values(test_to_explain)

# #print
# for i in range(len(test_to_explain)):
#     img = test_to_explain[i]
#     pred = model.predict(np.expand_dims(img, axis=0))
#     predicted_class = np.argmax(pred)
#     confidence = pred[0][predicted_class]

#     print(f"Gambar ke-{i+1} diprediksi sebagai: {class_names[predicted_class]} (Confidence: {confidence:.2f})")
#     shap.image_plot([s[i] for s in shap_values], [img])


# 1. Import library
import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image

# 2. Load dan bangun model CNN + MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(3, activation='softmax')(x)  # Ganti sesuai jumlah kelas

model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights("best_calorify.h5")

# 3. Fungsi untuk load gambar dari folder
def load_images_from_folder(folder, target_size=(224, 224)):
    images_list = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0  # Normalisasi
            images_list.append(img_array)
            filenames.append(filename)
    return np.array(images_list), filenames

# 4. Load data dari folder test/
test_folder = "test"
images, filenames = load_images_from_folder(test_folder)

# 5. Ambil background dan sampel untuk explain
background = images[:10]          # SHAP background
to_explain = images[10:11]        # Gambar yang ingin dijelaskan

# 6. Inisialisasi SHAP DeepExplainer
explainer = shap.DeepExplainer(model, background)

# 7. Hitung SHAP values
shap_values = explainer.shap_values(to_explain)

# 8. Visualisasi hasil SHAP
shap.image_plot(shap_values, to_explain)
