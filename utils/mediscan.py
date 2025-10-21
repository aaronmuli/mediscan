import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import os

# Script to load the trained Mediscan model and evaluate a single chest X-ray image.

# Define class names (update based on your trained model)
definite_class_names = ['Normal', 'Abnormal']
compare_class_names = ['Adults', 'Peads']

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation (matches validation transform in training script)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_confidence_margin(probabilities):
    if len(probabilities) < 2:
        return "Error: List must contain at least two probabilities.", None

    sorted_probs = sorted(probabilities, reverse=True)
    top_score = sorted_probs[0]
    second_score = sorted_probs[1]
    margin = top_score - second_score
    confidence_percentage = margin * 100
    return f"{confidence_percentage:.1f}"

def load_model(model_path, class_names):
    # Load ResNet18 architecture (no pre-trained weights needed since we're loading custom state_dict)
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(class_names))  # Match output to number of classes
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model

def evaluate_image(model, image_path, class_names):
    # Load and preprocess the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')  # Ensure RGB format
    image = eval_transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]  # Softmax for probabilities
        predicted_idx = torch.argmax(outputs, dim=1).item()
        predicted_class = class_names[predicted_idx]
    
    return predicted_class, probabilities

def mediscan(image_path):
    compare_model_path = "./models/adults_vs_peads_cxrs_model.pth"
    model = load_model(compare_model_path, compare_class_names)
    predicted_class, _ = evaluate_image(model, image_path, compare_class_names)

    if(predicted_class == "Adults"):
        model_path = "./models/adults_cxrs_model.pth"
        model = load_model(model_path, definite_class_names)
        predicted_class, probabilities = evaluate_image(model, image_path, definite_class_names)
        confidence_level = calculate_confidence_margin(probabilities)
        return predicted_class, probabilities, confidence_level
    
    elif(predicted_class == "Peads"):
        model_path = "./models/peadiatrics_cxrs_model.pth"
        model = load_model(model_path, definite_class_names)
        predicted_class, probabilities = evaluate_image(model, image_path, definite_class_names)
        confidence_level = calculate_confidence_margin(probabilities)
        return predicted_class, probabilities, confidence_level
    