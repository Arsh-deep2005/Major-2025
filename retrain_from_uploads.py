# retrain_from_uploads.py
import os, json, shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import models
import numpy as np

UPLOAD_DIR = "data/incoming_uploads"
DEST_DIR = "uploads_for_training"
MODEL_PATH = "global_model.h5"
IMG_SIZE = (128,128)
BATCH = 8
EPOCHS = 5

os.makedirs(DEST_DIR, exist_ok=True)

def collect_labeled():
    moved = 0
    for fname in os.listdir(UPLOAD_DIR):
        if not (fname.lower().endswith((".jpg",".jpeg",".png"))):
            continue
        meta_path = os.path.join(UPLOAD_DIR, fname + ".json")
        src = os.path.join(UPLOAD_DIR, fname)
        label = None
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                label = meta.get("suggested_label", None)
        if not label:
            label = "unlabeled"
        dest_folder = os.path.join(DEST_DIR, label)
        os.makedirs(dest_folder, exist_ok=True)
        shutil.move(src, os.path.join(dest_folder, fname))
        if os.path.exists(meta_path):
            shutil.move(meta_path, os.path.join(dest_folder, fname + ".json"))
        moved += 1
    print(f"[retrain] moved {moved} files to {DEST_DIR}")

def fine_tune():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH+" not found")
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0/255,
                                 rotation_range=15, horizontal_flip=True)
    train = datagen.flow_from_directory(
        DEST_DIR, subset="training", target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical"
    )
    val = datagen.flow_from_directory(
        DEST_DIR, subset="validation", target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical"
    )
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=EPOCHS)
    model.save("global_model_finetuned.h5")
    print("[retrain] saved global_model_finetuned.h5")

if __name__ == "__main__":
    collect_labeled()
    fine_tune()
