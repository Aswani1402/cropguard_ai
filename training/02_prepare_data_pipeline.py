import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from config import (
    DATA_ROOT,
    METRICS_DIR,
    MODELS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
)


def find_dataset_split_dirs(data_root: Path):
    train_dirs = [p for p in data_root.rglob("train") if p.is_dir()]

    for train_dir in train_dirs:
        parent = train_dir.parent
        valid_dir = parent / "valid"

        if valid_dir.exists() and valid_dir.is_dir():
            return train_dir, valid_dir

    raise FileNotFoundError(
        "Could not find train/valid folders inside data/raw/kaggle. "
        "Run the Kaggle extraction step again."
    )


def load_class_distribution():
    csv_path = METRICS_DIR / "dataset_class_distribution.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            "dataset_class_distribution.csv not found. "
            "Run training/01_dataset_audit_kaggle.py first."
        )

    return pd.read_csv(csv_path)


def calculate_class_weights(df: pd.DataFrame, class_names):
    train_df = df[df["split"] == "train"].copy()

    counts = {
        row["class_name"]: int(row["image_count"])
        for _, row in train_df.iterrows()
    }

    total_images = sum(counts.values())
    num_classes = len(class_names)

    class_weights = {}

    for class_index, class_name in enumerate(class_names):
        class_count = counts[class_name]
        weight = total_images / (num_classes * class_count)
        class_weights[class_index] = round(float(weight), 6)

    return class_weights


def create_datasets(train_dir: Path, valid_dir: Path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=SEED,
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=train_ds.class_names,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
    )

    return train_ds, valid_ds


def build_augmentation_layer():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def save_augmented_sample(train_ds, augmentation_layer):
    for images, labels in train_ds.take(1):
        augmented_images = augmentation_layer(images, training=True)

        plt.figure(figsize=(12, 8))

        for i in range(min(9, augmented_images.shape[0])):
            plt.subplot(3, 3, i + 1)
            image = augmented_images[i].numpy().astype("uint8")
            plt.imshow(image)
            plt.axis("off")

        plt.tight_layout()
        sample_path = METRICS_DIR / "augmentation_sample.png"
        plt.savefig(sample_path, dpi=200)
        plt.close()

        print(f"Saved augmentation sample: {sample_path}")
        break


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Finding dataset folders...")
    train_dir, valid_dir = find_dataset_split_dirs(DATA_ROOT)

    print(f"Train folder: {train_dir}")
    print(f"Valid folder: {valid_dir}")

    print("\nCreating TensorFlow datasets...")
    train_ds, valid_ds = create_datasets(train_dir, valid_dir)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print(f"\nNumber of classes: {num_classes}")
    print("First 5 classes:")
    for name in class_names[:5]:
        print("-", name)

    print("\nTesting one batch...")
    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Image dtype:", images.dtype)
        print("Label dtype:", labels.dtype)
        break

    print("\nCalculating class weights...")
    df = load_class_distribution()
    class_weights = calculate_class_weights(df, class_names)

    class_weights_path = MODELS_DIR / "class_weights.json"
    with open(class_weights_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in class_weights.items()}, f, indent=4)

    print(f"Saved class weights: {class_weights_path}")

    class_indices = {str(index): class_name for index, class_name in enumerate(class_names)}
    class_indices_path = MODELS_DIR / "class_indices.json"

    with open(class_indices_path, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=4)

    print(f"Saved class indices: {class_indices_path}")

    data_config = {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_classes": num_classes,
        "train_dir": str(train_dir),
        "valid_dir": str(valid_dir),
        "class_indices_file": str(class_indices_path),
        "class_weights_file": str(class_weights_path),
        "augmentation": [
            "RandomFlip(horizontal)",
            "RandomRotation(0.08)",
            "RandomZoom(0.10)",
            "RandomContrast(0.10)",
        ],
        "normalization": "Model-specific preprocessing will be applied inside training scripts.",
    }

    data_config_path = MODELS_DIR / "data_config.json"
    with open(data_config_path, "w", encoding="utf-8") as f:
        json.dump(data_config, f, indent=4)

    print(f"Saved data config: {data_config_path}")

    print("\nTesting augmentation layer...")
    augmentation_layer = build_augmentation_layer()
    save_augmented_sample(train_ds, augmentation_layer)

    print("\nData pipeline preparation completed successfully.")


if __name__ == "__main__":
    main()