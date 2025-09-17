import os
import io
import traceback
from typing import Tuple, Optional, List, Dict
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model#type:ignore
DEFAULT_MODEL_REL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "tomato_final_demo.h5")
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_REL_PATH)
_MODEL = None
_INPUT_SIZE = None
_CLASS_NAMES = None
_CLASS_NAMES = ["Tomato___Early_blight", "Tomato___healthy"]
def _get_size_from_model(m) -> Tuple[int, int]:
    """
    Derive expected (height, width) from model.input_shape.
    Fallback to (224, 224).
    """
    try:
        shape = m.input_shape
        if not shape:
            return (224, 224)
        if len(shape) == 4:
            h, w = int(shape[1]) if shape[1] is not None else None, int(shape[2]) if shape[2] is not None else None
            if h and w:
                return (h, w)
        # fallback
    except Exception:
        pass
    return (224, 224)


def load_my_model(path: Optional[str] = None) -> None:
    """
    Load the Keras model into the global _MODEL variable.
    This function is safe to call multiple times.
    """
    global _MODEL, _INPUT_SIZE
    if path is None:
        path = MODEL_PATH
    if _MODEL is not None:
        return

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    try:
        m = load_model(path)
        _MODEL = m
        _INPUT_SIZE = _get_size_from_model(m)
        print(f"✅ Model loaded from: {path}")
        print(f"   model.input_shape = {m.input_shape}; inferred input size = {_INPUT_SIZE}")
    except Exception as e:
        tb = traceback.format_exc()
        print("Failed to load model:", e)
        print(tb)
        raise


def get_model_status() -> Dict:
    """Return basic status for health checks and debugging."""
    return {
        "model_path": MODEL_PATH,
        "loaded": _MODEL is not None,
        "input_size": _INPUT_SIZE,
        "class_names_provided": bool(_CLASS_NAMES)
    }


def _preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes into a model-ready numpy array:
      - convert to RGB (3 channels)
      - resize to model input size (or 224x224 fallback)
      - scale to [0,1]
      - return shape (1, H, W, C)
    """
    if _INPUT_SIZE is None:
        target_size = (224, 224)
    else:
        target_size = _INPUT_SIZE

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # PIL resize expects (width, height)
    img = img.resize((target_size[1], target_size[0]), resample=Image.Resampling.BILINEAR)
    arr = np.array(img).astype("float32") / 255.0
    # ensure shape (H,W,3)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        # drop alpha if present
        arr = arr[..., :3]
    return np.expand_dims(arr, axis=0)


def predict_from_bytes(image_bytes: bytes, top_k: int = 1) -> List[Dict]:
    """
    Ensure model is loaded (attempt to load if not), preprocess bytes, run predict,
    and return list of {"class": name_or_index, "prob": float}.
    Raises RuntimeError if model can't be loaded.
    """
    global _MODEL
    if _MODEL is None:
        try:
            load_my_model()
        except Exception as e:
            raise RuntimeError(f"Model not loaded and failed to load now: {e}")

    x = _preprocess_bytes(image_bytes)
    preds = _MODEL.predict(x, verbose=0)[0]#type:ignore
    top_k = max(1, int(top_k))
    idxs = preds.argsort()[-top_k:][::-1]
    out = []
    for i in idxs:
        label = _CLASS_NAMES[i] if _CLASS_NAMES and i < len(_CLASS_NAMES) else str(int(i))
        out.append({"class": label, "prob": float(preds[int(i)])})
    return out