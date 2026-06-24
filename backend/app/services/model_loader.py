import json
from pathlib import Path

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"

BEST_MODEL_CONFIG_PATH = MODELS_DIR / "best_model_config.json"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"


_model_cache = None
_class_names_cache = None
_best_model_config_cache = None


def load_best_model_config():
    global _best_model_config_cache

    if _best_model_config_cache is None:
        if not BEST_MODEL_CONFIG_PATH.exists():
            raise FileNotFoundError(f"Missing best model config: {BEST_MODEL_CONFIG_PATH}")

        with open(BEST_MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
            _best_model_config_cache = json.load(f)

    return _best_model_config_cache


def load_class_names():
    global _class_names_cache

    if _class_names_cache is None:
        if not CLASS_INDICES_PATH.exists():
            raise FileNotFoundError(f"Missing class indices file: {CLASS_INDICES_PATH}")

        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)

        _class_names_cache = [
            class_indices[str(i)]
            for i in range(len(class_indices))
        ]

    return _class_names_cache


def load_model():
    global _model_cache

    if _model_cache is None:
        config = load_best_model_config()
        model_path = MODELS_DIR / config["model_file"]

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        _model_cache = tf.keras.models.load_model(model_path, compile=False)

    return _model_cache