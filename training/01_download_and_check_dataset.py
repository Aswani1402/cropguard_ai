import json
from pathlib import Path

import pandas as pd
import tensorflow_datasets as tfds


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("Loading PlantVillage dataset from TensorFlow Datasets...")

dataset, info = tfds.load(
    "plant_village",
    split="train",
    with_info=True,
    as_supervised=True,
    data_dir="data/raw/tfds"
)
num_examples = info.splits["train"].num_examples
num_classes = info.features["label"].num_classes
class_names = info.features["label"].names

print("\nDataset loaded successfully.")
print(f"Total images: {num_examples}")
print(f"Number of classes: {num_classes}")

print("\nClass names:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# Save class index mapping
class_indices = {str(i): name for i, name in enumerate(class_names)}

with open(MODELS_DIR / "class_indices.json", "w", encoding="utf-8") as f:
    json.dump(class_indices, f, indent=4)

print("\nSaved class index mapping to:")
print(MODELS_DIR / "class_indices.json")

# Count class distribution
label_counts = {name: 0 for name in class_names}

print("\nCounting class distribution...")

for _, label in tfds.as_numpy(dataset):
    class_name = class_names[int(label)]
    label_counts[class_name] += 1

df = pd.DataFrame(
    [{"class_name": k, "image_count": v} for k, v in label_counts.items()]
)

df = df.sort_values("image_count", ascending=False)
df.to_csv(OUTPUT_DIR / "dataset_class_distribution.csv", index=False)

print("\nSaved dataset class distribution to:")
print(OUTPUT_DIR / "dataset_class_distribution.csv")

print("\nTop 10 classes by image count:")
print(df.head(10))

print("\nBottom 10 classes by image count:")
(print
 (df.tail(10)))