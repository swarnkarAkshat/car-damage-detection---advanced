import React, { useState } from 'react';
import './index.css';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      // Reset state for new image
      setPrediction(null);
      setConfidence(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!image) return;
    
    setLoading(true);
    setError(null);
    
    // Create form data to send file correctly
    const formData = new FormData();
    formData.append("file", image);

    try {
      // Send POST request strictly to the local FastAPI port
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error("Failed to get prediction from server");
      }
      
      const data = await response.json();
      setPrediction(data.prediction);
      
      // Convert confidence decimal into a clean percentage
      if (data.confidence !== undefined) {
        setConfidence((data.confidence * 100).toFixed(1));
      }
    } catch (err) {
      setError(err.message || "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="glass-card">
        <h1 className="title">Car Damage Detection</h1>
        <p className="subtitle">Upload an image to assess vehicle condition</p>
        
        <div className="upload-section">
          <input 
            type="file" 
            id="file-upload" 
            accept="image/*" 
            onChange={handleImageChange} 
            className="hidden-input"
          />
          <label htmlFor="file-upload" className="upload-button">
            {preview ? 'Change Image' : 'Choose Image'}
          </label>
        </div>

        {preview && (
          <div className="preview-container">
            <img src={preview} alt="Upload Preview" className="image-preview" />
          </div>
        )}

        <button 
          onClick={handlePredict} 
          disabled={!image || loading} 
          className={`predict-button ${(loading || !image) ? 'disabled' : ''}`}
        >
          {loading ? 'Analyzing...' : 'Predict Damage'}
        </button>

        {error && <div className="error-message">{error}</div>}

        {prediction && (
          <div className="result-container">
            <div className="prediction-box">
              <h3>Detection Result</h3>
              {/* Replace underscores with spaces for cleaner UI */}
              <p className="prediction-text">{prediction.replace(/_/g, ' ')}</p>
              
              {confidence && (
                <div className="confidence-meter">
                  <div className="confidence-fill" style={{ width: `${confidence}%` }}></div>
                  <span className="confidence-text">{confidence}% Confidence</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
