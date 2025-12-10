
from flask import Flask, render_template, request, flash, url_for, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import uuid
import shutil
import threading
import traceback

app = Flask(__name__)
app.secret_key = "secret"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "global_model.h5"
IMG_SIZE = (128,128)

# ------------ Load Model ------------ #
model = None
print("\n================= FLASK SERVER STARTED =================")

if os.path.exists(MODEL_PATH):
    print(f"[MODEL] Loading initial model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[MODEL] ✔ Model loaded successfully.")
else:
    print(f"❌ [MODEL] Model file not found: {MODEL_PATH}")

# ------------ Allowed Extensions ------------ #
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"jpg","jpeg","png","bmp"}

# ------------ ROUTES ------------ #

@app.route("/")
def home():
    print("\n[ROUTE] / - Home Page")
    return render_template("index.html")

@app.route("/about")
def about():
    print("\n[ROUTE] /about - About Page")
    return render_template("about.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    global model
    print("\n================= /predict REQUEST =================")

    # reload latest FL model dynamically
    if os.path.exists("global_model.h5"):
        print("[MODEL] Reloading latest FL Global Model...")
        model = tf.keras.models.load_model("global_model.h5")
        print("[MODEL] ✔ Latest model loaded.")
    else:
        print("❌ [MODEL] global_model.h5 not found during predict")

    if request.method == "POST":
        print("[PREDICT] POST request received.")

        if "file" not in request.files:
            print("❌ [PREDICT] No file in request.")
            flash("No file uploaded")
            return render_template("predict.html")

        file = request.files["file"]

        if file.filename == "":
            print("❌ [PREDICT] No file selected by user.")
            flash("Please select an image")
            return render_template("predict.html")

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            print(f"[PREDICT] Saving uploaded image to: {save_path}")
            file.save(save_path)

            if model is None:
                print("❌ [PREDICT] Model not loaded, cannot predict.")
                flash("❌ Model could not be loaded.")
                return render_template("predict.html")

            # preprocess
            print("[PREDICT] Preprocessing image...")
            img = tf.keras.preprocessing.image.load_img(save_path, target_size=IMG_SIZE)
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, 0) / 255.0
            print("[PREDICT] Preprocessing complete.")

            print("[PREDICT] Running model prediction...")
            preds = model.predict(x)
            print("[PREDICT] Raw prediction output:", preds)

            idx = int(np.argmax(preds[0]))
            conf = float(np.max(preds[0]) * 100)
            print(f"[PREDICT] Predicted class index: {idx}, confidence: {conf:.2f}%")

            class_labels = [
                "Bacterial Leaf Blight",
                "Brown_spot",
                "Healthy Rice Leaf",
                "Leaf Blast",
                "Leaf scald",
                "Sheath Blight"
            ]

            label = class_labels[idx] if idx < len(class_labels) else f"Class_{idx}"
            print("[PREDICT] Final label:", label)

            return render_template(
                "result.html",
                image_url=url_for("static", filename=f"uploads/{filename}"),
                prediction=label,
                confidence=round(conf, 2)
            )

    print("[PREDICT] GET request - loading page.")
    return render_template("predict.html")

# ------------------------- UPLOAD ROUTE ---------------------------- #

CLIENT_DATA_DIR = "data/disease/rice/farm_1"
os.makedirs(CLIENT_DATA_DIR, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    print("\n================= /upload REQUEST =================")

    try:
        if "file" not in request.files:
            print("❌ [UPLOAD] No image found in request.")
            return jsonify({"error": "Image not found!"}), 400

        image = request.files["file"]
        label = request.form.get("label")

        print(f"[UPLOAD] Received label: {label}")

        if not label:
            print("❌ [UPLOAD] No label provided.")
            return jsonify({"error": "Label is required"}), 400

        ALLOWED_LABELS = [
            "Bacterial Leaf Blight",
            "Brown Spot",
            "Healthy Rice Leaf",
            "Leaf Blast",
            "Leaf scald",
            "Sheath Blight"
        ]

        if label not in ALLOWED_LABELS:
            print(f"❌ [UPLOAD] Invalid label: {label}")
            return jsonify({"error": f"Invalid label. Choose from: {ALLOWED_LABELS}"}), 400

        label_dir = os.path.join(CLIENT_DATA_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        print(f"[UPLOAD] Saving under directory: {label_dir}")

        ext = image.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"

        save_path = os.path.join(label_dir, filename)
        image.save(save_path)

        print(f"[UPLOAD] ✔ Saved image at: {save_path}")

        return jsonify({"message": "Image saved", "path": save_path})

    except Exception as e:
        print("❌ [UPLOAD] ERROR OCCURRED:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------ Run App ------------ #
if __name__ == "__main__":
    print("\n================= FLASK RUNNING =================\n")
    app.run(debug=True)
