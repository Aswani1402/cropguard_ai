from pathlib import Path
import re

import numpy as np

from backend.app.services.model_loader import (
    load_model,
    load_class_names,
    load_best_model_config,
)
from backend.app.services.preprocessing import load_image_as_array


CONFIDENCE_REJECTION_THRESHOLD = 0.80


def clean_text(text: str) -> str:
    cleaned = text.replace("__", " ")
    cleaned = cleaned.replace("_", " ")
    cleaned = cleaned.replace(",", "")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def clean_class_name(class_name: str) -> str:
    if "___" in class_name:
        crop, condition = class_name.split("___", 1)
        return f"{clean_text(crop)} — {clean_text(condition)}"

    return clean_text(class_name)


def extract_crop_name(class_name: str) -> str:
    if "___" in class_name:
        crop = class_name.split("___", 1)[0]
        return clean_text(crop)

    return clean_text(class_name)


def normalize_for_match(text: str) -> str:
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def get_confidence_message(confidence: float) -> dict:
    if confidence >= 0.85:
        return {
            "level": "high",
            "message": "High-confidence prediction within the trained class set.",
        }

    if confidence >= 0.60:
        return {
            "level": "medium",
            "message": (
                "Medium-confidence prediction. The image may be similar to this class, "
                "but it should be reviewed carefully."
            ),
        }

    return {
        "level": "low",
        "message": (
            "Low-confidence prediction. This image may be outside the supported dataset classes."
        ),
    }


def build_validation_status(
    selected_crop: str | None,
    predicted_crop: str,
    confidence: float,
) -> dict:
    selected_crop = selected_crop.strip() if selected_crop else None

    if selected_crop and normalize_for_match(selected_crop) in {
        "unknown",
        "notsure",
        "unknownnotsure",
    }:
        selected_crop = None

    if selected_crop:
        selected_norm = normalize_for_match(selected_crop)
        predicted_norm = normalize_for_match(predicted_crop)

        if selected_norm != predicted_norm:
            return {
                "status": "crop_mismatch",
                "is_accepted": False,
                "selected_crop": selected_crop,
                "predicted_crop": predicted_crop,
                "message": (
                    f"Crop mismatch detected. You selected {selected_crop}, "
                    f"but the model's closest prediction is {predicted_crop}. "
                    "Please upload a clearer image of the selected crop or review manually."
                ),
            }

    if confidence < CONFIDENCE_REJECTION_THRESHOLD:
        return {
            "status": "uncertain_confidence",
            "is_accepted": False,
            "selected_crop": selected_crop,
            "predicted_crop": predicted_crop,
            "message": (
                "Prediction is uncertain because confidence is below the acceptance threshold. "
                "The image may be outside the supported dataset classes or too different from training images."
            ),
        }

    return {
        "status": "accepted",
        "is_accepted": True,
        "selected_crop": selected_crop,
        "predicted_crop": predicted_crop,
        "message": "Prediction accepted within the trained class set.",
    }


def predict_image(image_path: str | Path, top_k: int = 3, selected_crop: str | None = None):
    config = load_best_model_config()
    image_size = tuple(config["image_size"])

    model = load_model()
    class_names = load_class_names()

    image_array = load_image_as_array(image_path, image_size=image_size)

    probabilities = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:top_k]

    top_predictions = []

    for rank, index in enumerate(top_indices):
        raw_class = class_names[index]
        confidence = round(float(probabilities[index]), 6)

        top_predictions.append(
            {
                "rank": rank + 1,
                "class_name": raw_class,
                "display_name": clean_class_name(raw_class),
                "crop_name": extract_crop_name(raw_class),
                "confidence": confidence,
            }
        )

    predicted_index = int(top_indices[0])
    predicted_class = class_names[predicted_index]
    confidence = round(float(probabilities[predicted_index]), 6)
    predicted_crop = extract_crop_name(predicted_class)

    validation = build_validation_status(
        selected_crop=selected_crop,
        predicted_crop=predicted_crop,
        confidence=confidence,
    )

    result = {
        "model_name": config["best_model_name"],
        "predicted_class": predicted_class,
        "display_class": clean_class_name(predicted_class),
        "predicted_crop": predicted_crop,
        "confidence": confidence,
        "confidence_info": get_confidence_message(confidence),
        "validation": validation,
        "top_predictions": top_predictions,
        "note": (
            "This is an educational AI prototype trained only on supported dataset classes. "
            "It is not an expert agricultural diagnosis."
        ),
    }

    return result
