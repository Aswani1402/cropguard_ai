import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.services.inference import predict_image
from backend.app.services.gradcam import save_gradcam_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "api_uploads"
GRADCAM_DIR = OUTPUT_DIR / "api_gradcam"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


app = FastAPI(
    title="CropGuard AI API",
    description="Plant disease prediction API using EfficientNetV2B0 and Grad-CAM explanation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # okay for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/")
def root():
    return {
        "message": "CropGuard AI API is running",
        "model": "EfficientNetV2B0",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CropGuard AI",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str | None = Form(None),
):
    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPG, JPEG, or PNG image.",
        )

    unique_id = uuid.uuid4().hex

    saved_image_path = UPLOAD_DIR / f"{unique_id}{suffix}"
    gradcam_output_path = GRADCAM_DIR / f"{unique_id}_gradcam.jpg"

    try:
        with open(saved_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        prediction = predict_image(
            saved_image_path,
            top_k=3,
            selected_crop=selected_crop,
        )

        gradcam_result = save_gradcam_overlay(
            image_path=saved_image_path,
            output_path=gradcam_output_path,
            alpha=0.4,
        )

        image_url = f"/outputs/api_uploads/{saved_image_path.name}"
        gradcam_url = f"/outputs/api_gradcam/{gradcam_output_path.name}"

        return {
            "success": True,
            "prediction": prediction,
            "explanation": {
                "method": "Grad-CAM",
                "gradcam_url": gradcam_url,
                "note": (
                    "Grad-CAM highlights image regions that influenced the model prediction. "
                    "It should not be interpreted as confirmed biological disease localization."
                ),
            },
            "uploaded_image": {
                "filename": file.filename,
                "saved_path": str(saved_image_path),
                "image_url": image_url,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
