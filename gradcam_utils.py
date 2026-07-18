import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(model, image, transform):
    """Generate Grad-CAM visualization for the model."""
    model.eval()

    # Get device from model
    device = next(model.parameters()).device
    print(f"[GradCAM] Model device: {device}")

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)
    print(f"[GradCAM] Input tensor shape: {input_tensor.shape}")

    # Last convolution layer of ResNet18
    target_layers = [model.layer4[-1]]

    # Create GradCAM object - only these two parameters
    cam = GradCAM(model=model, target_layers=target_layers)
    print("[GradCAM] Object created successfully")

    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    print(f"[GradCAM] CAM generated - shape: {grayscale_cam.shape}")

    # Original image - resize to match input
    rgb_img = np.array(image.resize((224, 224)))
    print(f"[GradCAM] Image shape: {rgb_img.shape}")

    # Normalize to 0-1 range
    rgb_img = rgb_img.astype(np.float32) / 255.0

    # Convert grayscale to RGB if needed
    if len(rgb_img.shape) == 2:
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)

    # Overlay heatmap on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    print(f"[GradCAM] Visualization complete")

    return visualization
