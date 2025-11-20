# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import layers, models
# import os

# # --------------------------
# # Path Setup
# # --------------------------
# base_dir = 'data/rice_leaf_diseases'
# IMG_SIZE = (128, 128)
# BATCH_SIZE = 32

# # --------------------------
# # Data Preprocessing
# # --------------------------
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,  # 80% train, 20% validation
#     rotation_range=20,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# train_gen = train_datagen.flow_from_directory(
#     base_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     subset='training',
#     class_mode='categorical'
# )

# val_gen = train_datagen.flow_from_directory(
#     base_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     subset='validation',
#     class_mode='categorical'
# )

# # --------------------------
# # Model Definition
# # --------------------------
# model = models.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
#     layers.MaxPooling2D(2,2),

#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),

#     layers.Conv2D(128, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),

#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(train_gen.num_classes, activation='softmax')
# ])

# # --------------------------
# # Compile Model
# # --------------------------
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # --------------------------
# # Train Model
# # --------------------------
# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=10
# )

# # --------------------------
# # Save Model
# # --------------------------
# model.save('model.h5')
# print("\n✅ Model saved successfully as model.h5")

# # Print classes for reference
# print("\nClass indices:", train_gen.class_indices)



# model_training.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import tensorflow as tf

DATA_DIR = "data/disease/rice/all"  # prepare full dataset here if using single-node training
IMG_SIZE = (128,128)
BATCH = 8
EPOCHS = 10

def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    train = datagen.flow_from_directory(DATA_DIR, subset="training", target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
    val = datagen.flow_from_directory(DATA_DIR, subset="validation", target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
    model = build_model(len(train.class_indices))
    model.fit(train, validation_data=val, epochs=EPOCHS)
    model.save("local_trained_model.h5")
    print("Saved local_trained_model.h5")
