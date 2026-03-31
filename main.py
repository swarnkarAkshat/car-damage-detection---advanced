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

# 5. Ensure FastAPI app is defined cleanly
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define the model class directly inside main.py using ResNet50
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

CLASS_NAMES = ["F_Breakage", "F_Crushed", "F_Normal", "R_Breakage", "R_Crushed", "R_Normal"]
model = CarClassifierResNet(num_classes=6)

# 4. Load model weights using file path ONLY, defaulting map_location to CPU
try:
    # Explicitly using the user's requested path exactly.
    # Note: If running locally inside the saved_model folder, this path resolving as "saved_model/saved_model.pth" 
    # might fail if it's actually "./saved_model.pth", but this perfectly mirrors their requested Render structure!
    model.load_state_dict(torch.load("saved_model/saved_model.pth", map_location="cpu"))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Failed to load model weights: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 6. Add a root endpoint
@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.get("/classes")
def get_classes():
    return CLASS_NAMES

# 7. Keep existing endpoints like /predict working
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The uploaded file must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        input_tensor = transform(image)
        # Squeeze batch dimension, already explicitly forced to CPU above
        input_batch = input_tensor.unsqueeze(0).to("cpu")
        
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

# 8. Ensure the app runs with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
