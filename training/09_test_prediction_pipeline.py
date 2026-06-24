import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from backend.app.services.inference import predict_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "kaggle"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prediction_tests"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_sample_image():
    valid_dirs = [p for p in DATA_ROOT.rglob("valid") if p.is_dir()]

    if not valid_dirs:
        raise FileNotFoundError("Could not find valid folder inside data/raw/kaggle.")

    valid_dir = valid_dirs[0]

    image_extensions = {".jpg", ".jpeg", ".png"}

    for image_path in valid_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            return image_path

    raise FileNotFoundError("No sample image found in validation dataset.")


def main():
    sample_image = find_sample_image()

    print("Testing prediction pipeline with:")
    print(sample_image)

    result = predict_image(sample_image, top_k=3)

    print("\nPrediction result:")
    print(json.dumps(result, indent=4))

    output_path = OUTPUT_DIR / "sample_prediction_result.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_image": str(sample_image),
                "prediction": result,
            },
            f,
            indent=4,
        )

    print("\nSaved prediction result:")
    print(output_path)


if __name__ == "__main__":
    main()