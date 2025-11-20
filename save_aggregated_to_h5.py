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
Load aggregated_weights.pkl (Flower Parameters or list), convert to numpy arrays,
detect number of classes from final bias vector, build matching Keras model,
set weights and save global_model.h5 (and class_count.json).
"""
import pickle
import sys
from pathlib import Path

# flwr conversion helper (handles different flwr versions)
try:
    from flwr.common import parameters_to_ndarrays
except Exception:
    parameters_to_ndarrays = None

import numpy as np
from tensorflow.keras import layers, models

OUT_H5 = "global_model.h5"
AGG_PICKLE = "aggregated_weights.pkl"
CLASS_COUNT_JSON = "global_class_count.txt"  # simple text file with class count

def build_model(num_classes, input_shape=(128,128,3)):
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
    return model

def load_parameters_pickle(path: Path):
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj

def to_ndarrays(maybe_parameters):
    """
    Convert Flower Parameters to list of ndarrays if necessary,
    or pass through if already ndarray list.
    """
    if isinstance(maybe_parameters, list):
        return maybe_parameters
    # If it's flwr Parameters object and parameters_to_ndarrays available:
    if parameters_to_ndarrays is not None:
        try:
            nds = parameters_to_ndarrays(maybe_parameters)
            return nds
        except Exception:
            pass
    # Last resort: try attribute names often present
    # Try to detect .tensors or .weights or .ndarrays
    if hasattr(maybe_parameters, "tensors"):
        return list(maybe_parameters.tensors)
    if hasattr(maybe_parameters, "ndarrays"):
        return list(maybe_parameters.ndarrays)
    if hasattr(maybe_parameters, "weights"):
        return list(maybe_parameters.weights)
    raise ValueError("Unsupported aggregated object type. It is not a list and conversion failed.")

def detect_num_classes_from_weights(weights):
    """Assume last weight array is bias of final Dense -> 1-D length == num_classes"""
    if not weights:
        raise ValueError("Empty weights list")
    last = weights[-1]
    if isinstance(last, np.ndarray) and last.ndim == 1:
        return int(last.shape[0])
    # Some save orders might place bias earlier; search for 1D array in last few entries
    for arr in reversed(weights[-6:]):  # check last up to 6 arrays
        if isinstance(arr, np.ndarray) and arr.ndim == 1:
            return int(arr.shape[0])
    raise ValueError("Could not infer num_classes from aggregated weights (no 1D bias found).")

def main():
    p = Path(AGG_PICKLE)
    if not p.exists():
        print(f"ERROR: {AGG_PICKLE} not found in current folder: {Path.cwd()}")
        sys.exit(1)

    print("[save] Loading aggregated weights from", AGG_PICKLE)
    params = load_parameters_pickle(p)

    try:
        weights = to_ndarrays(params)
    except Exception as e:
        print("❌ Conversion to ndarrays failed:", e)
        sys.exit(1)

    # ensure numpy arrays
    weights = [np.array(w) for w in weights]
    print(f"[save] Converted aggregated parameters -> {len(weights)} tensors.")

    # detect num_classes
    try:
        num_classes = detect_num_classes_from_weights(weights)
    except Exception as e:
        print("❌ Could not detect num_classes:", e)
        sys.exit(1)

    print(f"[save] Detected num_classes = {num_classes}")

    # build model with detected classes
    model = build_model(num_classes=num_classes)

    # quick check: number of arrays from model.get_weights()
    model_weights_template = model.get_weights()
    print("[save] Model expects", len(model_weights_template), "weight arrays.")

    if len(model_weights_template) != len(weights):
        print("⚠️ Warning: number of aggregated tensors != model template tensors.")
        print("Template lengths: expected", len(model_weights_template), "got", len(weights))
        # Try a flexible approach: if lengths match after removing optimizer tensors etc.
        # But Keras get_weights for Sequential normally matches. We'll attempt set_weights and catch error.
    try:
        model.set_weights(weights)
    except Exception as e:
        print("\n❌ ERROR setting weights! This usually means the model architecture (layer shapes) does not match aggregated weights.")
        print("Error:", e)
        print("Suggestion: ensure clients used exactly the same model architecture and same input size.")
        raise

    # Save final model
    model.save(OUT_H5)
    print(f"\n✅ Saved global model to: {OUT_H5}")

    # Save class count for Flask (simple)
    with open(CLASS_COUNT_JSON, "w") as f:
        f.write(str(num_classes))
    print(f"[save] Wrote {CLASS_COUNT_JSON} with value {num_classes}")

if __name__ == "__main__":
    main()
