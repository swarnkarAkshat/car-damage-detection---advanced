import os
import io
import logging
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Initialize FastAPI
app = FastAPI(title="Car Damage Classification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Path Configuration
# This ensures it always uses the exact same folder as main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check multiple common paths for the model
potential_paths = [
    os.path.join(BASE_DIR, "saved_model.pth"),
    os.path.join(BASE_DIR, "saved_model", "saved_model.pth")
]

MODEL_PATH = potential_paths[0]
for path in potential_paths:
    if os.path.exists(path):
        MODEL_PATH = path
        break

# 4. Global Variables
model = None
model_loaded = False  # Boolean variable tracking success
device = torch.device("cpu")

CLASSES = [
    "F_Breakage", 
    "F_Crushed", 
    "F_Normal", 
    "R_Breakage", 
    "R_Crushed", 
    "R_Normal"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# 5. Model Loading Logic
def load_app_model():
    global model, model_loaded
    
    # Print debug logs
    logger.info("=== STARTING MODEL LOAD ===")
    logger.info(f"Current Directory: {os.getcwd()}")
    try:
        logger.info(f"Files in directory: {os.listdir(BASE_DIR)}")
    except Exception as e:
        logger.error(f"Could not list files: {e}")
    logger.info(f"Using Model Path: {MODEL_PATH}")
    logger.info("===========================")

    if not os.path.exists(MODEL_PATH):
        logger.warning(f"{MODEL_PATH} not found! Predictions will not work.")
        model_loaded = False
        return

    try:
        # Load the CarClassifierResNet framework
        logger.info("Setting up CarClassifierResNet...")
        base_model = CarClassifierResNet(num_classes=len(CLASSES))
        
        # Load weights from file
        logger.info("Loading weights from disk...")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # Clean state_dict if it is wrapped in a checkpoint dictionary
        if isinstance(state_dict, dict) and ('state_dict' in state_dict or 'model_state_dict' in state_dict):
            state_dict = state_dict.get('state_dict', state_dict.get('model_state_dict'))

        # Load weights into the architecture
        logger.info("Applying state_dict...")
        base_model.load_state_dict(state_dict)
        
        # Assign to global model & finalize prep
        model = base_model.to(device)
        model.eval()
        
        model_loaded = True
        logger.info("✅ Model loaded successfully!")
        
    except Exception as e:
        logger.exception(f"❌ Error loading model: {e}")
        model_loaded = False
        model = None

# Attempt to load right away
load_app_model()

# 6. API Endpoints
@app.get("/")
def read_root():
    return {"message": "Car Damage Classification API is running!"}

@app.get("/health")
def health_check():
    return {
        "status": "Healthy",
        "model_loaded": model_loaded,
        "model_path_checked": MODEL_PATH
    }

@app.get("/classes")
def get_classes():
    return {"classes": CLASSES}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Must upload an image file.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        max_prob, predicted_idx = torch.max(probabilities, 0)
        
        return {
            "prediction": CLASSES[predicted_idx.item()],
            "confidence": float(max_prob.item())
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
