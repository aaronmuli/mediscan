from io import BytesIO
import torch
from torchvision import models, transforms
from utils.gradcam import GradCAM
import cv2
import numpy as np

# Define class names
definite_class_names = ['Normal', 'Abnormal']
compare_class_names = ['Adults', 'Peads']
heatmap_image_path = './static/uploads/gradcam.jpg'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_confidence_margin(probabilities):
    if len(probabilities) < 2:
        return "Error: List must contain at least two probabilities.", None

    sorted_probs = sorted(probabilities.tolist(), reverse=True)
    top_score = sorted_probs[0]
    second_score = sorted_probs[1]
    margin = top_score - second_score
    confidence_percentage = margin * 100
    return f"{confidence_percentage:.1f}"

def load_model(model_path, class_names):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def differentiate_image(model, original_image, class_names):
    tensor_orig = eval_transform(original_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_orig = model(tensor_orig)
        prob_orig = torch.softmax(output_orig, dim=1)[0]

    predicted_idx = torch.argmax(prob_orig).item()
    predicted_class = class_names[predicted_idx]

    return predicted_class

def evaluate_image(model, x_ray, class_names):
    original_image = x_ray
    
    # === Original inference ===
    tensor_orig = eval_transform(original_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_orig = model(tensor_orig)
        prob_orig = torch.softmax(output_orig, dim=1)[0]
    
    predicted_idx = torch.argmax(prob_orig).item()
    predicted_class = class_names[predicted_idx]

    # === ALWAYS generate Grad-CAM heatmap ===
    target_layer = model.layer3[0].conv2  # Your chosen layer
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate_heatmap(tensor_orig, predicted_idx)
    heatmap_np = heatmap.squeeze().cpu().numpy()
    
    # Overlay on ORIGINAL image (224x224)
    original_resized = original_image.resize((224, 224))
    original_cv = cv2.cvtColor(np.array(original_resized), cv2.COLOR_RGB2BGR)
    heatmap_cv = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_cv, 0.6, heatmap_cv, 0.4, 0)

    _ , buffer = cv2.imencode(".jpg", overlay)
    io_buf = BytesIO(buffer)

    return predicted_class, prob_orig, io_buf

def mediscan(x_ray):
    compare_model_path = "./models/adults_vs_peads_cxrs_model.pth"
    compare_model = load_model(compare_model_path, compare_class_names)
    age_class = differentiate_image(compare_model, x_ray, compare_class_names)

    if age_class == "Adults":
        model_path = "./models/adults_cxrs_model.pth"
    elif age_class == "Peads":
        model_path = "./models/peadiatrics_cxrs_model.pth"
    else:
        raise ValueError("Unexpected age classification")

    definite_model = load_model(model_path, definite_class_names)
    predicted_class, probabilities, heatmap = evaluate_image(definite_model, x_ray, definite_class_names)
    confidence_level = calculate_confidence_margin(probabilities)

    return predicted_class, probabilities, confidence_level, heatmap