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

# 3. Ensure the FastAPI app is defined cleanly for deployment
app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the ResNet50 model architecture
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

# Setup Environment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["F_Breakage", "F_Crushed", "F_Normal", "R_Breakage", "R_Crushed", "R_Normal"]

# Absolute path to model - immune to cloud directory shifting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pth")

try:
    print(f"Loading model from: {MODEL_PATH}")
    model = CarClassifierResNet(num_classes=6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Failed to load model weights: {e}")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----- API Endpoints -----

# 4. Ensure there is a root endpoint named home
@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.get("/classes")
def get_classes():
    return CLASS_NAMES

# 6. Do NOT break existing /predict endpoint
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The uploaded file must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_batch)
            
        probabilities = F.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        return {
            "prediction": CLASS_NAMES[predicted_idx.item()],
            "confidence": round(confidence.item(), 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed processing the image: {str(e)}")

# 5. Make sure the app can run properly using uvicorn main:app
if __name__ == "__main__":
    import uvicorn
    # Secure bindings defaulting strictly to 0.0.0.0 and port 10000 for Render integration
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
