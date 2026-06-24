# CropGuard AI Backend

This folder contains the FastAPI backend for CropGuard AI.

The backend loads the trained EfficientNetV2B0 model, accepts plant leaf image uploads, performs image preprocessing, returns top-3 disease predictions, applies confidence and crop-aware validation, and generates Grad-CAM visual explanations.

---

## Backend Features

* FastAPI REST API
* Image upload through `/predict`
* TensorFlow/Keras model loading
* EfficientNetV2B0 inference
* Top-3 prediction output
* Confidence-based uncertainty handling
* Crop-aware validation
* Grad-CAM explanation generation
* Static serving of uploaded images and Grad-CAM outputs
* pytest backend tests

---

## Folder Structure

```text
backend/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   └── services/
│       ├── __init__.py
│       ├── model_loader.py
│       ├── preprocessing.py
│       ├── inference.py
│       └── gradcam.py
```

---

## Main Files

### `app/main.py`

Defines the FastAPI application and API endpoints.

Main endpoints:

```text
GET /
GET /health
POST /predict
```

---

### `app/services/model_loader.py`

Loads:

```text
models/efficientnetv2b0_best.keras
models/class_indices.json
models/best_model_config.json
```

The model is cached after first load to avoid reloading it on every request.

---

### `app/services/preprocessing.py`

Handles image loading, RGB conversion, resizing, and batch dimension creation.

Input image shape:

```text
224 × 224 × 3
```

---

### `app/services/inference.py`

Runs model prediction and returns:

```text
predicted class
display class
predicted crop
confidence score
top-3 predictions
confidence information
crop validation status
```

---

### `app/services/gradcam.py`

Generates Grad-CAM overlay images using the final convolution layer of the trained model.

Grad-CAM is used as a visual explanation method. It highlights model-influential regions, but it should not be interpreted as confirmed biological disease localization.

---

## Required Model Files

Large trained model files are not committed to GitHub.

Place these files inside the root `models/` folder:

```text
models/efficientnetv2b0_best.keras
models/class_indices.json
models/best_model_config.json
```

The `.keras` model file is excluded from GitHub because of file size.

---

## Run Backend Locally

From the project root:

```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Root Endpoint

```http
GET /
```

Example response:

```json
{
  "message": "CropGuard AI API is running",
  "model": "EfficientNetV2B0",
  "docs": "/docs"
}
```

---

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "service": "CropGuard AI"
}
```

---

### Prediction Endpoint

```http
POST /predict
```

Form fields:

```text
file: JPG, JPEG, or PNG leaf image
selected_crop: optional crop name
```

Example response:

```json
{
  "success": true,
  "prediction": {
    "model_name": "EfficientNetV2B0",
    "predicted_class": "Apple___Apple_scab",
    "display_class": "Apple — Apple scab",
    "predicted_crop": "Apple",
    "confidence": 0.996033,
    "validation": {
      "status": "accepted",
      "is_accepted": true,
      "selected_crop": "Apple",
      "predicted_crop": "Apple",
      "message": "Prediction accepted within the trained class set."
    }
  },
  "explanation": {
    "method": "Grad-CAM",
    "gradcam_url": "/outputs/api_gradcam/sample_gradcam.jpg"
  }
}
```

---

## Crop-Aware Validation

The backend supports optional crop validation.

Example:

```text
Selected crop: Pepper bell
Model predicted crop: Corn (maize)
Validation status: crop_mismatch
```

This prevents the system from blindly accepting predictions when the uploaded crop type does not match the model prediction.

---

## Confidence-Based Rejection

The backend uses a confidence threshold to reduce over-trusting uncertain predictions.

Current threshold:

```text
0.80
```

Validation logic:

```text
If selected crop does not match predicted crop:
    status = crop_mismatch

Else if confidence < 0.80:
    status = uncertain_confidence

Else:
    status = accepted
```

---

## Run Backend Tests

From the project root:

```bash
pytest -q
```

Current test result:

```text
7 passed, 1 warning
```

Backend tests cover:

```text
Root endpoint
Health endpoint
Valid image prediction
Invalid file type rejection
Crop match validation
Crop mismatch validation
Prediction without selected crop
```

---

## Important Notes

* This backend is part of an educational AI prototype.
* The model supports only the trained dataset classes.
* Unknown crops may still be mapped to the closest known class.
* Grad-CAM is an explanation aid, not biological proof.
* Predictions should not replace expert agricultural diagnosis.
