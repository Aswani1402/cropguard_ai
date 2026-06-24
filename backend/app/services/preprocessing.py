from pathlib import Path

import numpy as np
from PIL import Image


def load_image_as_array(image_path: str | Path, image_size=(224, 224)) -> np.ndarray:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)

    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array