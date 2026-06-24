import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from backend.app.services.inference import predict_image
from backend.app.services.gradcam import save_gradcam_overlay


DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "kaggle"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam_tests"

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

    print("Testing Grad-CAM with image:")
    print(sample_image)

    prediction = predict_image(sample_image, top_k=3)

    print("\nPrediction:")
    print(json.dumps(prediction, indent=4))

    gradcam_output_path = OUTPUT_DIR / "sample_gradcam_overlay.jpg"

    gradcam_result = save_gradcam_overlay(
        image_path=sample_image,
        output_path=gradcam_output_path,
        alpha=0.4,
    )

    result = {
        "sample_image": str(sample_image),
        "prediction": prediction,
        "gradcam": gradcam_result,
    }

    output_json_path = OUTPUT_DIR / "sample_gradcam_result.json"

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("\nGrad-CAM result:")
    print(json.dumps(gradcam_result, indent=4))

    print("\nSaved files:")
    print(gradcam_output_path)
    print(output_json_path)


if __name__ == "__main__":
    main()