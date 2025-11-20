import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained global model
model = tf.keras.models.load_model("global_rice_leaf_model.h5")

# Path to the test image
img_path = "test_leaf_2.jpg"  # Replace with your image name/path
img = image.load_img(img_path, target_size=(128, 128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0  # Normalize the image

# Predict
pred = model.predict(x)
classes = ['Bacterial_leaf_blight', 'Brown_spot', 'Leaf_smut']
print("Predicted Disease:", classes[np.argmax(pred)])
