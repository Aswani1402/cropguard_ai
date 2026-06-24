import json
from pathlib import Path

import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOGS_DIR = PROJECT_ROOT / "outputs" / "training_logs"
ERROR_DIR = PROJECT_ROOT / "outputs" / "error_analysis"

MODEL_PATH = MODELS_DIR / "custom_cnn_baseline.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
CLASS_WEIGHTS_PATH = MODELS_DIR / "class_weights.json"
METADATA_PATH = MODELS_DIR / "custom_cnn_model_metadata.json"

RESULTS_PATH = METRICS_DIR / "custom_cnn_results.csv"
REPORT_PATH = METRICS_DIR / "custom_cnn_classification_report.csv"
HISTORY_PATH = LOGS_DIR / "custom_cnn_history.csv"
ERROR_PATH = ERROR_DIR / "custom_cnn_wrong_predictions_summary.csv"


def check_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    print(f"FOUND: {path}")


def main():
    print("\nChecking Custom CNN output files...\n")

    required_files = [
        MODEL_PATH,
        CLASS_INDICES_PATH,
        CLASS_WEIGHTS_PATH,
        RESULTS_PATH,
        REPORT_PATH,
        HISTORY_PATH,
    ]

    optional_files = [
        METADATA_PATH,
        ERROR_PATH,
    ]

    for path in required_files:
        check_file(path)

    for path in optional_files:
        if path.exists():
            check_file(path)
        else:
            print(f"OPTIONAL MISSING: {path}")

    print("\nLoading results CSV...")
    results_df = pd.read_csv(RESULTS_PATH)
    print(results_df.to_string(index=False))

    print("\nLoading class indices...")
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    print("Number of classes:", len(class_indices))
    print("First 5 classes:")
    for key in list(class_indices.keys())[:5]:
        print(key, "->", class_indices[key])

    print("\nLoading classification report...")
    report_df = pd.read_csv(REPORT_PATH)
    print(report_df.head(10).to_string(index=False))

    print("\nLoading trained Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nModel loaded successfully.")
    print("Model name:", model.name)
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    if model.output_shape[-1] != len(class_indices):
        raise ValueError(
            f"Model output classes {model.output_shape[-1]} does not match "
            f"class index count {len(class_indices)}"
        )

    print("\nCustom CNN verification completed successfully.")


if __name__ == "__main__":
    main()