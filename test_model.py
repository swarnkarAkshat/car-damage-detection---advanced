import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import sys

# 1. Replace the current model with this exact CarClassifierResNet class
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 2. Use models.resnet50(weights=None) while loading
        self.model = models.resnet50(weights=None)

        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def predict_image(image_path, model_path="saved_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Initialize model with num_classes=6
    # The architecture exactly matches how it was trained
    model = CarClassifierResNet(num_classes=6)
    
    # 4. Load weights
    try:
        # map_location=device ensures it works perfectly even if tested on a CPU
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Model weights loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model.to(device)
    
    # 5. Set model to evaluation mode
    model.eval()

    # 6. Add image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image)
        # PyTorch expects a batch dimension, so we unsqueeze the tensor
        input_batch = input_tensor.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading or processing image '{image_path}': {e}")
        return

    # Run prediction
    with torch.no_grad():
        output = model(input_batch)
    
    # 8. Predict class index using argmax
    _, predicted_idx = torch.max(output, 1)
    
    # 9. Map prediction to class names
    class_names = ["F_Breakage", "F_Crushed", "F_Normal", "R_Breakage", "R_Crushed", "R_Normal"]
    prediction = class_names[predicted_idx.item()]
    
    # 10. Print the predicted class name
    print("\n" + "="*35)
    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")
    print("="*35 + "\n")

if __name__ == "__main__":
    # 7. Accept image path as command line argument
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <path_to_image>")
        sys.exit(1)
        
    test_image = sys.argv[1]
    predict_image(test_image)
