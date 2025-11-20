
from flask import Flask, render_template, request, flash, url_for,jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import uuid
import shutil
import threading
import traceback
import flwr as fl

app = Flask(__name__)
app.secret_key = "secret"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "global_model.h5"
IMG_SIZE = (128,128)

# ------------ Load Model ------------ #
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Loaded", MODEL_PATH)
else:
    print("❌ Model not found:", MODEL_PATH)

# ------------ Allowed Extensions ------------ #
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"jpg","jpeg","png","bmp"}
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "jpg", "jpeg", "png"
    }


def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def make_generators():
    print("[Gen] Creating data generators...")

    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1/255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        CLIENT_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        CLIENT_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    print("[Gen] Classes:", train_gen.class_indices)
    return train_gen, val_gen


def local_training_thread():
    try:
        print("\n======================")
        print("⏳ LOCAL TRAINING STARTED")
        print("======================")

        train_gen, val_gen = make_generators()
        num_classes = len(train_gen.class_indices)

        model_local = build_model(num_classes)

        model_local.fit(train_gen, epochs=LOCAL_EPOCHS, verbose=1)

        print("✅ LOCAL TRAINING DONE")

        # Save model weights for FL
        _shared["model"] = model_local

    except Exception as e:
        print("❌ Error in local training:", e)
        traceback.print_exc()


# ==========================================
# FEDERATED LEARNING CLIENT
# ==========================================
class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print("[FL] get_parameters called.")
        return _shared["model"].get_weights()

    def fit(self, parameters, config):
        print("[FL] fit round triggered by server.")
        _shared["model"].set_weights(parameters)

        train_gen, _ = make_generators()
        _shared["model"].fit(train_gen, epochs=1, verbose=1)

        print("[FL] Sending updated weights to server")
        return _shared["model"].get_weights(), len(train_gen), {}

    def evaluate(self, parameters, config):
        print("[FL] evaluation called.")
        _shared["model"].set_weights(parameters)

        _, val_gen = make_generators()
        loss, acc = _shared["model"].evaluate(val_gen, verbose=0)

        return loss, len(val_gen), {"accuracy": float(acc)}


def start_fl_client_thread():
    def run():
        print("🚀 Starting FL client thread")
        fl.client.start_numpy_client(SERVER_ADDRESS, client=FLClient())

    threading.Thread(target=run, daemon=True).start()

# ------------ ROUTES ------------ #

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded")
            return render_template("predict.html")

        file = request.files["file"]

        if file.filename == "":
            flash("Please select an image")
            return render_template("predict.html")

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            if model is None:
                flash("❌ Model could not be loaded.")
                return render_template("predict.html")

            # preprocess
            img = tf.keras.preprocessing.image.load_img(save_path, target_size=IMG_SIZE)
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, 0) / 255.0

            preds = model.predict(x)
            idx = int(np.argmax(preds[0]))
            conf = float(np.max(preds[0]) * 100)

            class_labels = [
                "Bacterial Leaf Blight",
                "Brown_spot",
                "Healthy Rice Leaf",
                "Leaf Blast",
                "Leaf scald",
                "Sheath Blight"
            ]

            label = class_labels[idx] if idx < len(class_labels) else f"Class_{idx}"

            return render_template(
                "result.html",
                image_url=url_for("static", filename=f"uploads/{filename}"),
                prediction=label,
                confidence=round(conf, 2)
            )

    return render_template("predict.html")

CLIENT_DATA_DIR = "data/disease/rice/farm_1"
os.makedirs(CLIENT_DATA_DIR, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Image not found!"}), 400

        image = request.files["file"]
        label = request.form.get("label")

        if not label:
            return jsonify({"error": "Label is required"}), 400

        label_dir = os.path.join(CLIENT_DATA_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        ext = image.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"

        save_path = os.path.join(label_dir, filename)
        image.save(save_path)

        print("Saved:", save_path)

        threading.Thread(target=local_training_thread, daemon=True).start()
        start_fl_client_thread()

        return jsonify({"message": "Image saved", "path": save_path})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



# ------------ Run App ------------ #
if __name__ == "__main__":
    app.run(debug=True)
