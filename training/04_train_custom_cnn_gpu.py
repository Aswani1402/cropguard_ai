import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 10


def find_dataset_root():
    possible_roots = [
        Path("/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"),
        Path("data/raw/kaggle/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"),
    ]

    for root in possible_roots:
        train_dir = root / "train"
        valid_dir = root / "valid"

        if train_dir.exists() and valid_dir.exists():
            return root, train_dir, valid_dir

    raise FileNotFoundError(
        "Could not find dataset train/valid folders. "
        "Check Kaggle dataset path or local data/raw/kaggle folder."
    )


def create_datasets(train_dir, valid_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_names=train_ds.class_names,
    )

    return train_ds, valid_ds


def optimize_dataset(ds):
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def compute_class_weights(train_dir, class_names):
    counts = {}

    for class_name in class_names:
        class_dir = train_dir / class_name
        image_count = len(list(class_dir.glob("*")))
        counts[class_name] = image_count

    total_images = sum(counts.values())
    num_classes = len(class_names)

    class_weights = {}

    for index, class_name in enumerate(class_names):
        class_weights[index] = total_images / (num_classes * counts[class_name])

    return class_weights, counts


def build_custom_cnn(num_classes):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(224, 224, 3))

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


def plot_curves(history, output_path):
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(10, 6))
    plt.plot(history_df["accuracy"], label="train_accuracy")
    plt.plot(history_df["val_accuracy"], label="val_accuracy")
    plt.plot(history_df["loss"], label="train_loss")
    plt.plot(history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Custom CNN Baseline Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    output_root = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    models_dir = output_root / "models"
    metrics_dir = output_root / "outputs" / "metrics"
    logs_dir = output_root / "outputs" / "training_logs"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_root, train_dir, valid_dir = find_dataset_root()

    print("Dataset root:", dataset_root)
    print("Train dir:", train_dir)
    print("Valid dir:", valid_dir)

    train_ds, valid_ds = create_datasets(train_dir, valid_dir)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("Number of classes:", num_classes)

    class_indices = {str(i): name for i, name in enumerate(class_names)}

    with open(models_dir / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=4)

    class_weights, class_counts = compute_class_weights(train_dir, class_names)

    with open(models_dir / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in class_weights.items()}, f, indent=4)

    class_df = pd.DataFrame(
        [{"class_name": k, "image_count": v} for k, v in class_counts.items()]
    )
    class_df.to_csv(metrics_dir / "custom_cnn_class_distribution.csv", index=False)

    train_ds = optimize_dataset(train_ds)
    valid_ds = optimize_dataset(valid_ds)

    model = build_custom_cnn(num_classes)
    model.summary()

    model_path = models_dir / "custom_cnn_baseline.keras"
    history_path = logs_dir / "custom_cnn_history.csv"
    curves_path = metrics_dir / "custom_cnn_training_curves.png"

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
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=history_path,
            append=False,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    results = model.evaluate(valid_ds, return_dict=True)

    print("\nFinal validation results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    results_df = pd.DataFrame([{
        "model": "custom_cnn_baseline",
        "accuracy": results["accuracy"],
        "loss": results["loss"],
        "top_3_accuracy": results["top_3_accuracy"],
        "epochs": EPOCHS,
        "image_size": str(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
    }])

    results_df.to_csv(metrics_dir / "custom_cnn_results.csv", index=False)

    plot_curves(history, curves_path)

    print("\nSaved files:")
    print(model_path)
    print(history_path)
    print(curves_path)
    print(metrics_dir / "custom_cnn_results.csv")


if __name__ == "__main__":
    main()