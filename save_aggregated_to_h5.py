# # save_aggregated_to_h5.py
# import pickle
# import numpy as np
# from tensorflow.keras import layers, models

# # Set correct number of classes or derive from your clients
# CLASS_NAMES = ["Bacterial Leaf Blight", "Brown_spot", "Healthy Rice Leaf", "Leaf Blast", "Leaf scald", "Sheath Blight"]  # replace with your classes
# NUM_CLASSES = len(CLASS_NAMES)

# def build_model(num_classes):
#     model = models.Sequential([
#         layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
#         layers.MaxPooling2D(2,2),
#         layers.Conv2D(64, (3,3), activation='relu'),
#         layers.MaxPooling2D(2,2),
#         layers.Conv2D(128, (3,3), activation='relu'),
#         layers.MaxPooling2D(2,2),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.4),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# if __name__ == "__main__":
#     with open("aggregated_weights.pkl", "rb") as f:
#         weights = pickle.load(f)

#     print("[save] loaded aggregated weights. Number of arrays:", len(weights))

#     model = build_model(NUM_CLASSES)
#     try:
#         model.set_weights(weights)
#     except Exception as e:
#         print("Error setting weights:", e)
#         raise

#     model.save("global_model.h5")
#     print("[save] Saved global_model.h5")


# save_aggregated_to_h5.py
"""
Adds full debug printing to track:
- Weight loading
- Conversion to ndarray
- Class detection
- Model reconstruction
- Weight assignment
- H5 save confirmation
"""

import pickle
import sys
from pathlib import Path

try:
    from flwr.common import parameters_to_ndarrays
except Exception:
    parameters_to_ndarrays = None

import numpy as np
from tensorflow.keras import layers, models

OUT_H5 = "global_model.h5"
AGG_PICKLE = "aggregated_weights.pkl"
CLASS_COUNT_JSON = "global_class_count.txt"

def build_model(num_classes, input_shape=(128,128,3)):
    print(f"[build] Building model with num_classes={num_classes}")
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("[build] Model created and compiled.")
    return model

def load_parameters_pickle(path: Path):
    print(f"[load] Opening pickle file: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)
    print("[load] Pickle loaded successfully.")
    return obj

def to_ndarrays(maybe_parameters):
    print("[convert] Converting Flower parameters to numpy arrays...")
    if isinstance(maybe_parameters, list):
        print("[convert] Already list of ndarrays.")
        return maybe_parameters

    if parameters_to_ndarrays is not None:
        try:
            nds = parameters_to_ndarrays(maybe_parameters)
            print("[convert] Converted using parameters_to_ndarrays.")
            return nds
        except Exception as e:
            print("[convert] parameters_to_ndarrays failed:", e)

    if hasattr(maybe_parameters, "tensors"):
        print("[convert] Found .tensors attribute")
        return list(maybe_parameters.tensors)

    if hasattr(maybe_parameters, "ndarrays"):
        print("[convert] Found .ndarrays attribute")
        return list(maybe_parameters.ndarrays)

    if hasattr(maybe_parameters, "weights"):
        print("[convert] Found .weights attribute")
        return list(maybe_parameters.weights)

    raise ValueError("[convert] Unsupported parameter type. Cannot extract arrays.")

def detect_num_classes_from_weights(weights):
    print("[detect] Detecting number of output classes from last bias vector...")
    if not weights:
        raise ValueError("[detect] ERROR: Weight list empty.")

    last = weights[-1]
    print("[detect] Last tensor shape:", last.shape)

    if isinstance(last, np.ndarray) and last.ndim == 1:
        print("[detect] Last bias vector found. Num classes =", last.shape[0])
        return int(last.shape[0])

    print("[detect] Searching last 6 tensors for bias...")
    for arr in reversed(weights[-6:]):
        if isinstance(arr, np.ndarray) and arr.ndim == 1:
            print("[detect] Bias detected. Num classes =", arr.shape[0])
            return int(arr.shape[0])

    raise ValueError("[detect] ERROR: Could not detect num_classes from any bias vector!")

def main():
    p = Path(AGG_PICKLE)
    if not p.exists():
        print(f"❌ ERROR: {AGG_PICKLE} not found in folder: {Path.cwd()}")
        sys.exit(1)

    print(f"\n================ LOAD AGGREGATED WEIGHTS ================")
    params = load_parameters_pickle(p)

    try:
        weights = to_ndarrays(params)
    except Exception as e:
        print("❌ Conversion failed:", e)
        sys.exit(1)

    weights = [np.array(w) for w in weights]
    print(f"[save] Total tensors received = {len(weights)}")

    print(f"[save] First tensor shape: {weights[0].shape}")

    print("\n================ DETECT NUM CLASSES ================")
    try:
        num_classes = detect_num_classes_from_weights(weights)
    except Exception as e:
        print("❌ Detection Error:", e)
        sys.exit(1)

    print(f"[save] ✔ Number of Classes Detected = {num_classes}")

    print("\n================ BUILD MODEL ================")
    model = build_model(num_classes=num_classes)

    model_weights_template = model.get_weights()
    print(f"[save] Model expects {len(model_weights_template)} tensors.")

    if len(model_weights_template) != len(weights):
        print("⚠️ WARNING: Mismatch in tensor count!")
        print("⚠️ Template expects:", len(model_weights_template))
        print("⚠️ Aggregated:", len(weights))

    print("\n================ SET MODEL WEIGHTS ================")
    try:
        model.set_weights(weights)
        print("[save] ✔ Weights successfully loaded into model.")
    except Exception as e:
        print("❌ ERROR setting weights:", e)
        print("💡 Hint: Clients must have SAME model architecture + SAME class count.")
        raise

    print("\n================ SAVE GLOBAL MODEL ================")
    model.save(OUT_H5)
    print(f"✅ Saved updated FL global model → {OUT_H5}")

    with open(CLASS_COUNT_JSON, "w") as f:
        f.write(str(num_classes))
    print(f"📄 Saved number of classes → {CLASS_COUNT_JSON}")

    print("\n🎉 All steps completed successfully.")

if __name__ == "__main__":
    main()
