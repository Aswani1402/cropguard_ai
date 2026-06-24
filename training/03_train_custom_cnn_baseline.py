import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from config import (
    DATA_ROOT,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    MODELS_DIR,
    OUTPUT_DIR,
    METRICS_DIR,
)


EXPERIMENT_NAME = "CropGuard-Custom-CNN-Baseline"


def find_dataset_split_dirs(data_root: Path):
    train_dirs = [p for p in data_root.rglob("train") if p.is_dir()]

    for train_dir in train_dirs:
        parent = train_dir.parent
        valid_dir = parent / "valid"

        if valid_dir.exists() and valid_dir.is_dir():
            return train_dir, valid_dir

    raise FileNotFoundError(
        "Could not find train/valid folders inside data/raw/kaggle."
    )


def create_datasets(train_dir: Path, valid_dir: Path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=SEED,
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="int",
        class_names=train_ds.class_names,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
    )

    return train_ds, valid_ds


def optimize_dataset(ds):
    return ds.prefetch(tf.data.AUTOTUNE)


def load_class_weights():
    class_weights_path = MODELS_DIR / "class_weights.json"

    if not class_weights_path.exists():
        raise FileNotFoundError(
            "models/class_weights.json not found. "
            "Run training/02_prepare_data_pipeline.py first."
        )

    with open(class_weights_path, "r", encoding="utf-8") as f:
        raw_weights = json.load(f)

    return {int(k): float(v) for k, v in raw_weights.items()}


def build_custom_cnn(num_classes: int):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="custom_cnn_baseline")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )

    return model


def plot_training_curves(history, output_path: Path):
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(10, 6))
    plt.plot(history_df["accuracy"], label="train_accuracy")
    plt.plot(history_df["val_accuracy"], label="val_accuracy")
    plt.plot(history_df["loss"], label="train_loss")
    plt.plot(history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Custom CNN Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use small batches to test training pipeline quickly."
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "training_logs").mkdir(parents=True, exist_ok=True)

    train_dir, valid_dir = find_dataset_split_dirs(DATA_ROOT)

    print(f"Train directory: {train_dir}")
    print(f"Valid directory: {valid_dir}")

    train_ds, valid_ds = create_datasets(train_dir, valid_dir)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")

    if args.smoke_test:
        print("Running smoke test with limited batches...")
        train_ds = train_ds.take(20)

        # Shuffle validation batches during smoke test so evaluation is not taken
        # only from the first few sorted class folders.
        valid_ds = valid_ds.shuffle(
            buffer_size=200,
            seed=SEED,
            reshuffle_each_iteration=False
        ).take(10)

    train_ds = optimize_dataset(train_ds)
    valid_ds = optimize_dataset(valid_ds)

    class_weights = load_class_weights()

    model = build_custom_cnn(num_classes)
    model.summary()

    model_path = MODELS_DIR / "custom_cnn_baseline.keras"
    history_csv_path = OUTPUT_DIR / "training_logs" / "custom_cnn_history.csv"
    curve_path = METRICS_DIR / "custom_cnn_training_curves.png"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=history_csv_path,
            append=False,
        ),
    ]

    mlflow.set_tracking_uri(f"sqlite:///{Path.cwd() / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="custom_cnn_smoke" if args.smoke_test else "custom_cnn_full"):
        mlflow.log_param("model_name", "custom_cnn_baseline")
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 1e-3)
        mlflow.log_param("loss", "sparse_categorical_crossentropy")
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("smoke_test", args.smoke_test)

        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=callbacks,
        )

        final_metrics = model.evaluate(valid_ds, verbose=1, return_dict=True)

        print("\nValidation metrics:")
        for name, value in final_metrics.items():
            print(f"{name}: {value:.4f}")
            mlflow.log_metric(f"valid_{name}", float(value))

        plot_training_curves(history, curve_path)

        mlflow.log_artifact(str(history_csv_path))
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(model_path))

    print("\nCustom CNN baseline training completed.")
    print(f"Saved model: {model_path}")
    print(f"Saved history CSV: {history_csv_path}")
    print(f"Saved training curve: {curve_path}")


if __name__ == "__main__":
    main()