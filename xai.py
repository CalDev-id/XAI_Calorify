import tensorflow as tf
import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


model = tf.keras.models.load_model('best_calorify.h5')
model.summary()

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

def preprocess(images, labels):
    return preprocess_input(images), labels

test_dataset = test_dataset.map(preprocess)

# Ambil 1 batch (32 gambar)
for test_images, test_labels in test_dataset.take(1):
    break

test_images_np = test_images.numpy()

if test_images_np.shape[0] < 32:
    raise ValueError(f"Hanya ada {test_images_np.shape[0]} gambar. Dibutuhkan minimal 32.")

background = test_images_np[:27]
test_to_explain = test_images_np[27:32]


explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(test_to_explain)

#print
for i in range(len(test_to_explain)):
    img = test_to_explain[i]
    pred = model.predict(np.expand_dims(img, axis=0))
    predicted_class = np.argmax(pred)
    confidence = pred[0][predicted_class]

    print(f"Gambar ke-{i+1} diprediksi sebagai: {class_names[predicted_class]} (Confidence: {confidence:.2f})")
    shap.image_plot([s[i] for s in shap_values], [img])
