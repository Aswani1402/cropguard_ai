import { useState } from "react";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    setResult(null);
    setError("");

    if (!file) {
      setSelectedFile(null);
      setPreviewUrl("");
      return;
    }

    const allowedTypes = ["image/jpeg", "image/jpg", "image/png"];

    if (!allowedTypes.includes(file.type)) {
      setError("Please upload a JPG, JPEG, or PNG image.");
      setSelectedFile(null);
      setPreviewUrl("");
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError("Please select a leaf image first.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  };

  const getImageUrl = (path) => {
    if (!path) return "";
    return `${API_BASE_URL}${path}`;
  };

  return (
    <main className="page">
      <section className="hero">
        <p className="eyebrow">CropGuard AI</p>
        <h1>Explainable Plant Disease Detection</h1>
        <p className="subtitle">
          Upload a plant leaf image to predict the disease class using the trained
          EfficientNetV2B0 model and view Grad-CAM explanation.
        </p>
      </section>

      <section className="layout">
        <div className="card upload-card">
          <h2>Upload Leaf Image</h2>

          <label className="upload-box">
            <input
              type="file"
              accept="image/png, image/jpeg, image/jpg"
              onChange={handleFileChange}
            />
            <span>Choose JPG or PNG image</span>
          </label>

          {previewUrl && (
            <div className="preview-block">
              <p className="section-label">Selected Image</p>
              <img src={previewUrl} alt="Selected leaf preview" className="preview-image" />
            </div>
          )}

          <button
            className="predict-button"
            onClick={handlePredict}
            disabled={isLoading || !selectedFile}
          >
            {isLoading ? "Analyzing..." : "Predict Disease"}
          </button>

          {error && <p className="error-message">{error}</p>}
        </div>

        <div className="card result-card">
          <h2>Prediction Result</h2>

          {!result && !isLoading && (
            <p className="empty-state">
              Prediction result will appear here after uploading an image.
            </p>
          )}

          {isLoading && <p className="empty-state">Model is analyzing the image...</p>}

          {result && (
            <>
              <div className="prediction-main">
                <p className="section-label">Predicted Class</p>
                <h3>{result.prediction.predicted_class}</h3>
                <p className="confidence">
                  Confidence: {(result.prediction.confidence * 100).toFixed(2)}%
                </p>
              </div>

              <div className="top-list">
                <p className="section-label">Top 3 Predictions</p>

                {result.prediction.top_predictions.map((item) => (
                  <div className="top-item" key={item.rank}>
                    <span>
                      #{item.rank} {item.class_name}
                    </span>
                    <strong>{(item.confidence * 100).toFixed(2)}%</strong>
                  </div>
                ))}
              </div>

              <div className="gradcam-block">
                <p className="section-label">Grad-CAM Explanation</p>
                <img
                  src={getImageUrl(result.explanation.gradcam_url)}
                  alt="Grad-CAM explanation"
                  className="gradcam-image"
                />
                <p className="note">{result.explanation.note}</p>
              </div>

              <p className="disclaimer">{result.prediction.note}</p>
            </>
          )}
        </div>
      </section>
    </main>
  );
}

export default App;