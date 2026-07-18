import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image
from gradcam_utils import generate_gradcam
from class_names import classes

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = resnet18(weights="DEFAULT")

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)

model.load_state_dict(
    torch.load("brain_tumor_resnet18.pth", map_location=device)
)


model.to(device)
model.eval()

# Same transform as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def predict_image(image):

    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

        probs = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    prediction = classes[predicted.item()]
    confidence = confidence.item() * 100

    print(f"[DEBUG] Prediction: {prediction}, Confidence: {confidence}")

    # Generate Grad-CAM
    heatmap = None
    try:
        print("[DEBUG] Starting GradCAM generation...")
        heatmap = generate_gradcam(
            model,
            image,
            transform
        )
        print(f"[DEBUG] GradCAM generated successfully, type: {type(heatmap)}")
    except Exception as e:
        print(f"[ERROR] GradCAM generation failed: {str(e)}")
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        heatmap = {"error": str(e), "traceback": error_msg}

    print(f"[DEBUG] About to return: prediction={prediction}, confidence={confidence}, heatmap={type(heatmap)}")
    result = (prediction, confidence, heatmap)
    print(f"[DEBUG] Returning tuple of length {len(result)}: {result}")
    return result
