import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from backend.app.main import app


client = TestClient(app)

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "kaggle"


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


def find_apple_scab_image():
    image_extensions = {".jpg", ".jpeg", ".png"}

    for valid_dir in DATA_ROOT.rglob("valid"):
        if not valid_dir.is_dir():
            continue

        apple_scab_dir = valid_dir / "Apple___Apple_scab"

        if not apple_scab_dir.is_dir():
            continue

        for image_path in apple_scab_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                return image_path

    raise FileNotFoundError("No Apple___Apple_scab image found in validation dataset.")


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "CropGuard AI"


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "CropGuard AI API is running"
    assert data["model"] == "EfficientNetV2B0"


def test_predict_endpoint_with_valid_image():
    image_path = find_sample_image()

    with open(image_path, "rb") as image_file:
        response = client.post(
            "/predict",
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "prediction" in data
    assert "explanation" in data
    assert "uploaded_image" in data

    prediction = data["prediction"]

    assert prediction["model_name"] == "EfficientNetV2B0"
    assert isinstance(prediction["predicted_class"], str)
    assert len(prediction["predicted_class"]) > 0
    assert isinstance(prediction["display_class"], str)
    assert isinstance(prediction["predicted_crop"], str)
    assert isinstance(prediction["confidence"], float)
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["confidence_info"]["level"] in {"high", "medium", "low"}
    assert prediction["validation"]["status"] in {
        "crop_mismatch",
        "uncertain_confidence",
        "accepted",
    }
    assert isinstance(prediction["validation"]["is_accepted"], bool)
    assert len(prediction["top_predictions"]) == 3

    explanation = data["explanation"]

    assert explanation["method"] == "Grad-CAM"
    assert explanation["gradcam_url"].endswith(".jpg")


def test_predict_endpoint_with_crop_match():
    image_path = find_apple_scab_image()

    with open(image_path, "rb") as image_file:
        response = client.post(
            "/predict",
            data={"selected_crop": "Apple"},
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    prediction = data["prediction"]

    assert prediction["predicted_crop"] == "Apple"
    assert prediction["confidence"] >= 0.80
    assert prediction["validation"]["status"] == "accepted"
    assert prediction["validation"]["is_accepted"] is True


def test_predict_endpoint_with_crop_mismatch():
    image_path = find_apple_scab_image()

    with open(image_path, "rb") as image_file:
        response = client.post(
            "/predict",
            data={"selected_crop": "Pepper bell"},
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    prediction = response.json()["prediction"]
    validation = prediction["validation"]

    assert prediction["predicted_crop"] == "Apple"
    assert validation["status"] == "crop_mismatch"
    assert validation["is_accepted"] is False
    assert validation["selected_crop"] == "Pepper bell"
    assert validation["predicted_crop"] == "Apple"
    assert "Crop mismatch" in validation["message"]


def test_predict_endpoint_without_selected_crop():
    image_path = find_apple_scab_image()

    with open(image_path, "rb") as image_file:
        response = client.post(
            "/predict",
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    prediction = response.json()["prediction"]

    assert "display_class" in prediction
    assert "predicted_crop" in prediction
    assert "confidence_info" in prediction
    assert "validation" in prediction
    assert "top_predictions" in prediction
    assert len(prediction["top_predictions"]) == 3


def test_predict_endpoint_rejects_invalid_file_type():
    response = client.post(
        "/predict",
        files={
            "file": (
                "not_an_image.txt",
                b"This is not an image.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert "Invalid file type" in data["detail"]
    
