# import torch
# import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models, transforms
# from PIL import Image
# import argparse
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# Script to load the trained Mediscan model, evaluate a chest X-ray image, and highlight abnormal regions using Grad-CAM.
# Usage: python evaluate_with_gradcam.py --model_path models/mediscan_model.pth --image_path path/to/your/image.jpg
# Outputs: Prediction, probabilities, and saves 'output_gradcam.jpg' with heatmap overlay if abnormality detected.
# Assumptions:
# - Model is a fine-tuned ResNet18.
# - For binary (normal/tb), use class_names = ['normal', 'tb']
# - For multi-class, update class_names accordingly, e.g., ['bronchiolitis', 'bronchitis', 'normal', 'pneumonia', 'tb']
# - Heatmap is generated only if prediction is not 'normal'.

# Define class names (update based on your trained model)
# class_names = ['normal', 'tb']  # For binary TB detection; change for multi-class

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Image transformation (matches validation transform in training script)
# eval_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_heatmap(self, input_tensor, class_idx):
        # Forward pass
        self.model.eval()
        logit = self.model(input_tensor)
        
        # Backward pass for the target class
        score = logit[:, class_idx].squeeze()
        self.model.zero_grad()
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.data
        activations = self.activations.data
        b, k, u, v = gradients.size()
        
        # Global average pooling on gradients
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        
        # Weighted combination
        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min + 1e-5).data
        
        return saliency_map

# def load_model(model_path):
#     model = models.resnet18(weights=None)
#     num_features = model.fc.in_features
#     model.fc = nn.Linear(num_features, len(class_names))
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     model = model.to(device)
#     model.eval()
#     return model

# def evaluate_and_visualize(model, image_path, output_image_path='output_gradcam.jpg'):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image not found: {image_path}")
    
#     # Load and preprocess image
#     original_image = Image.open(image_path).convert('RGB')
#     input_tensor = eval_transform(original_image)
#     input_tensor = input_tensor.unsqueeze(0).to(device)
    
#     # Run inference
#     with torch.no_grad():
#         outputs = model(input_tensor)
#         probabilities = torch.softmax(outputs, dim=1)[0]
#         predicted_idx = torch.argmax(outputs, dim=1).item()
#         predicted_class = class_names[predicted_idx]
    
#     # Output prediction
#     print(f"Predicted Class: {predicted_class}")
#     print("Class Probabilities:")
#     for i, cls in enumerate(class_names):
#         print(f"{cls}: {probabilities[i].item():.4f}")
    
#     # Generate Grad-CAM if abnormality detected (not 'normal')
#     if predicted_class == 'normal':
#         # Target the last conv layer in ResNet18 (layer4[-1].conv2)
#         target_layer = model.layer4[-1].conv2
#         gradcam = GradCAM(model, target_layer)
#         heatmap = gradcam.generate_heatmap(input_tensor, predicted_idx)
        
#         # Convert heatmap to numpy
#         heatmap = heatmap.squeeze().cpu().numpy()
        
#         # Overlay heatmap on original image
#         original_cv = cv2.cvtColor(np.array(original_image.resize((224, 224))), cv2.COLOR_RGB2BGR)
#         heatmap_cv = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
#         overlay = cv2.addWeighted(original_cv, 0.6, heatmap_cv, 0.4, 0)
        
#         # Save the visualized image
#         cv2.imwrite(output_image_path, overlay)
#         print(f"Visualized image with abnormality highlights saved as: {output_image_path}")
        
#         # Optionally display (uncomment if needed)
#         # plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#         # plt.title(f"Predicted: {predicted_class}")
#         # plt.show()
#     else:
#         print("No abnormality detected; no heatmap generated.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate and visualize abnormality in a chest X-ray using Grad-CAM.")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
#     parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    
#     args = parser.parse_args()
    
#     model = load_model(args.model_path)
#     evaluate_and_visualize(model, args.image_path)