import json
import random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "kaggle"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_dataset_split_dirs(data_root: Path):
    train_dirs = [p for p in data_root.rglob("train") if p.is_dir()]

    for train_dir in train_dirs:
        parent = train_dir.parent
        valid_dir = parent / "valid"

        if valid_dir.exists() and valid_dir.is_dir():
            return train_dir, valid_dir

    raise FileNotFoundError(
        "Could not find train/valid folders inside data/raw/kaggle. "
        "Please check whether the Kaggle dataset was extracted correctly."
    )


def count_images(split_dir: Path):
    rows = []

    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])

    for class_dir in class_dirs:
        image_files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        rows.append({
            "split": split_dir.name,
            "class_name": class_dir.name,
            "image_count": len(image_files)
        })

    return rows


def save_sample_images(train_dir: Path, class_names, output_path: Path, max_samples: int = 12):
    selected_classes = random.sample(class_names, min(max_samples, len(class_names)))

    plt.figure(figsize=(14, 10))

    for idx, class_name in enumerate(selected_classes):
        class_dir = train_dir / class_name
        image_files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not image_files:
            continue

        image_path = random.choice(image_files)
        image = Image.open(image_path).convert("RGB")

        plt.subplot(3, 4, idx + 1)
        plt.imshow(image)
        plt.title(class_name[:35], fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_distribution_chart(df: pd.DataFrame, output_path: Path):
    train_df = df[df["split"] == "train"].sort_values("image_count", ascending=True)

    plt.figure(figsize=(12, 14))
    plt.barh(train_df["class_name"], train_df["image_count"])
    plt.xlabel("Number of Images")
    plt.ylabel("Class Name")
    plt.title("Training Set Class Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    print("Finding Kaggle Plant Disease dataset folders...")

    train_dir, valid_dir = find_dataset_split_dirs(DATA_ROOT)

    print(f"Train folder found: {train_dir}")
    print(f"Valid folder found: {valid_dir}")

    train_class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    valid_class_names = sorted([p.name for p in valid_dir.iterdir() if p.is_dir()])

    if train_class_names != valid_class_names:
        raise ValueError(
            "Train and valid class folders do not match. "
            "Please inspect the dataset folder structure."
        )

    class_names = train_class_names
    class_indices = {str(index): class_name for index, class_name in enumerate(class_names)}

    with open(MODELS_DIR / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=4)

    rows = []
    rows.extend(count_images(train_dir))
    rows.extend(count_images(valid_dir))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "dataset_class_distribution.csv", index=False)

    train_total = df[df["split"] == "train"]["image_count"].sum()
    valid_total = df[df["split"] == "valid"]["image_count"].sum()

    print("\nDataset audit completed.")
    print(f"Number of classes: {len(class_names)}")
    print(f"Total train images: {train_total}")
    print(f"Total valid images: {valid_total}")

    print("\nTop 10 largest training classes:")
    print(
        df[df["split"] == "train"]
        .sort_values("image_count", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nBottom 10 smallest training classes:")
    print(
        df[df["split"] == "train"]
        .sort_values("image_count", ascending=True)
        .head(10)
        .to_string(index=False)
    )

    save_distribution_chart(
        df,
        OUTPUT_DIR / "class_distribution.png"
    )

    save_sample_images(
        train_dir,
        class_names,
        OUTPUT_DIR / "sample_images.png"
    )

    print("\nSaved files:")
    print(OUTPUT_DIR / "dataset_class_distribution.csv")
    print(OUTPUT_DIR / "class_distribution.png")
    print(OUTPUT_DIR / "sample_images.png")
    print(MODELS_DIR / "class_indices.json")


if __name__ == "__main__":
    main()