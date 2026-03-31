from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import os

# Initialize the FastAPI app
app = FastAPI(title="Car Damage Classifier API")

# 3. Add CORS middleware to allow all origins safely for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Define the exact ResNet50 model architecture used
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights=None)
        
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["F_Breakage", "F_Crushed", "F_Normal", "R_Breakage", "R_Crushed", "R_Normal"]

# 4. Make sure file paths are secure and accurate relative to the environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pth")

# Load the model only once when the server starts
try:
    print(f"Loading model into {DEVICE} from: {MODEL_PATH}")
    model = CarClassifierResNet(num_classes=6)
    
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Warning: Failed to load model weights on startup: {e}")

# Preprocessing Pipeline 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----- API Endpoints -----

# Root endpoint (Requested)
@app.get("/")
def read_root():
    return {"message": "API is running"}

# Health check
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# Classes array
@app.get("/classes")
def get_classes():
    return CLASS_NAMES

# 5. Prediction endpoint for image testing
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The uploaded file must be an image.")

    try:
        # Load File
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            output = model(input_batch)
            
        # Confidence logic
        probabilities = F.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        prediction_name = CLASS_NAMES[predicted_idx.item()]
        
        return {
            "prediction": prediction_name,
            "confidence": round(confidence.item(), 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed processing the image: {str(e)}")

# 2. Add native startup block handling 0.0.0.0 bindings for port mappings in Render
if __name__ == "__main__":
    import uvicorn
    # If Render passes a $PORT variable, it will use it dynamically, otherwise fallback to 10000!
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
