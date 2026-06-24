# CropGuard AI Frontend

This folder contains the React frontend for CropGuard AI.

The frontend allows users to select a crop type, upload a plant leaf image, send the image to the FastAPI backend, and view the predicted disease class, confidence score, top-3 predictions, validation warning, and Grad-CAM explanation.

---

## Frontend Features

* React + Vite frontend
* Leaf image upload
* Crop type dropdown
* Prediction result display
* Top-3 prediction display
* Confidence warning display
* Crop mismatch warning display
* Grad-CAM image display
* Responsive UI layout

---

## Folder Structure

```text
frontend/
│
├── public/
│
├── src/
│   ├── App.jsx
│   ├── App.css
│   ├── main.jsx
│   └── index.css
│
├── .env
├── package.json
├── package-lock.json
└── README.md
```

---

## Tech Stack

```text
React
Vite
JavaScript
CSS
Fetch API
```

---

## Environment Variable

Create a `.env` file inside the `frontend/` folder:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

This tells the frontend where the FastAPI backend is running.

---

## Install Dependencies

From the `frontend/` folder:

```bash
npm install
```

---

## Run Frontend Locally

From the `frontend/` folder:

```bash
npm run dev
```

The frontend usually runs at:

```text
http://localhost:5173
```

If port `5173` is already used, Vite may use another port such as:

```text
http://localhost:5174
```

Use the exact URL shown in the terminal.

---

## Backend Requirement

The frontend needs the backend running before prediction can work.

Start backend from the project root:

```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Then start frontend:

```bash
cd frontend
npm run dev
```

---

## User Flow

```text
Open frontend
        ↓
Select crop type
        ↓
Upload leaf image
        ↓
Click Predict Disease
        ↓
Frontend sends image to FastAPI backend
        ↓
Backend returns prediction and Grad-CAM URL
        ↓
Frontend displays result
```

---

## Supported Crop Dropdown

The frontend includes crop options such as:

```text
Apple
Blueberry
Cherry (including sour)
Corn (maize)
Grape
Orange
Peach
Pepper bell
Potato
Raspberry
Soybean
Squash
Strawberry
Tomato
Unknown / Not sure
```

If the user selects a crop and the model predicts a different crop, the frontend displays a crop mismatch warning.

---

## Prediction Display

The frontend displays:

```text
Predicted disease class
Confidence score
Confidence warning
Crop validation message
Top-3 predictions
Grad-CAM explanation image
Educational disclaimer
```

Example:

```text
Predicted Class: Apple — Apple scab
Confidence: 99.60%
Validation: Prediction accepted within the trained class set.
```

---

## Crop Mismatch Display

Example:

```text
Selected crop: Pepper bell
Model prediction: Corn (maize) — healthy
```

The frontend shows:

```text
Crop mismatch detected. You selected Pepper bell, but the model's closest prediction is Corn (maize). Please upload a clearer image of the selected crop or review manually.
```

This prevents users from blindly trusting a wrong crop prediction.

---

## Confidence Warning

The frontend displays backend confidence information.

Example statuses:

```text
High confidence
Medium confidence
Low confidence
Uncertain confidence
Crop mismatch
Accepted
```

---

## Grad-CAM Display

The frontend displays the Grad-CAM image returned by the backend.

Grad-CAM highlights image regions that influenced the model prediction. It should not be interpreted as confirmed biological disease localization.

---

## Build for Production

From the `frontend/` folder:

```bash
npm run build
```

The production build is generated inside:

```text
frontend/dist/
```

The `dist/` folder is excluded from GitHub.

---

## Important Notes

* The frontend does not run the machine learning model directly.
* Prediction happens in the FastAPI backend.
* The frontend only sends the image and displays the response.
* If the backend is not running, prediction requests will fail.
* The app is an educational prototype and not an expert agricultural diagnosis tool.
