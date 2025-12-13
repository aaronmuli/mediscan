# import torch
# from torchvision import models, transforms
# from torchvision.models import ResNet18_Weights
# from PIL import Image
# import os
# from utils.gradcam import GradCAM
# import cv2
# import numpy as np

# # Script to load the trained Mediscan model and evaluate a single chest X-ray image.

# # Define class names (update based on your trained model)
# definite_class_names = ['Normal', 'Abnormal']
# compare_class_names = ['Adults', 'Peads']
# heatmap_image_path = './static/uploads/gradcam.jpg'

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Image transformation (matches validation transform in training script)
# eval_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def calculate_confidence_margin(probabilities):
#     if len(probabilities) < 2:
#         return "Error: List must contain at least two probabilities.", None

#     sorted_probs = sorted(probabilities, reverse=True)
#     top_score = sorted_probs[0]
#     second_score = sorted_probs[1]
#     margin = top_score - second_score
#     confidence_percentage = margin * 100
#     return f"{confidence_percentage:.1f}"

# def load_model(model_path, class_names):
#     # Load ResNet18 architecture (no pre-trained weights needed since we're loading custom state_dict)
#     model = models.resnet18(weights=None)
#     num_features = model.fc.in_features
#     model.fc = torch.nn.Linear(num_features, len(class_names))  # Match output to number of classes
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     model = model.to(device)
#     model.eval()  # Set to evaluation mode
#     return model

# def evaluate_image(model, image_path, class_names):
#     # Load and preprocess the image
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image not found: {image_path}")
    
#     original_image = Image.open(image_path).convert('RGB')  # Ensure RGB format
#     image_tensor = eval_transform(original_image)
#     image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
#     image_tensor = image_tensor.to(device)
    
#     # Run inference
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         probabilities = torch.softmax(outputs, dim=1)[0]  # Softmax for probabilities
#         predicted_idx = torch.argmax(outputs, dim=1).item()
#         predicted_class = class_names[predicted_idx]

#     # Generate Grad-CAM if abnormality detected (not 'normal')
#     if predicted_class == 'Abnormal' or predicted_class == 'Normal':
#         # Target the last conv layer in ResNet18 (layer4[-1].conv2)
#         target_layer = model.layer3[0].conv2
#         gradcam = GradCAM(model, target_layer)
#         heatmap = gradcam.generate_heatmap(image_tensor, predicted_idx)
        
#         # Convert heatmap to numpy
#         heatmap = heatmap.squeeze().cpu().numpy()
        
#         # Overlay heatmap on original image
#         original_cv = cv2.cvtColor(np.array(original_image.resize((224, 224))), cv2.COLOR_RGB2BGR)
#         heatmap_cv = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
#         overlay = cv2.addWeighted(original_cv, 0.6, heatmap_cv, 0.4, 0)
        
#         # Save the visualized image
#         cv2.imwrite(heatmap_image_path, overlay)
    
#     return predicted_class, probabilities

# def mediscan(image_path):
#     compare_model_path = "./models/adults_vs_peads_cxrs_model.pth"
#     model = load_model(compare_model_path, compare_class_names)
#     predicted_class, _ = evaluate_image(model, image_path, compare_class_names)

#     if(predicted_class == "Adults"):
#         model_path = "./models/adults_cxrs_model.pth"
#         model = load_model(model_path, definite_class_names)
#         predicted_class, probabilities = evaluate_image(model, image_path, definite_class_names)
#         confidence_level = calculate_confidence_margin(probabilities)
#         return predicted_class, probabilities, confidence_level
    
#     elif(predicted_class == "Peads"):
#         model_path = "./models/peadiatrics_cxrs_model.pth"
#         model = load_model(model_path, definite_class_names)
#         predicted_class, probabilities = evaluate_image(model, image_path, definite_class_names)
#         confidence_level = calculate_confidence_margin(probabilities)
#         return predicted_class, probabilities, confidence_level
    

import torch
from torchvision import models, transforms
from torchvision.transforms import functional as TF  # For hflip
from PIL import Image
import os
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

def evaluate_image(model, image_path, class_names):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original_image = Image.open(image_path).convert('RGB')
    
    # === Original inference ===
    tensor_orig = eval_transform(original_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_orig = model(tensor_orig)
        prob_orig = torch.softmax(output_orig, dim=1)[0]
    
    # === Flipped inference ===
    flipped_image = TF.hflip(original_image)
    tensor_flip = eval_transform(flipped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_flip = model(tensor_flip)
        prob_flip = torch.softmax(output_flip, dim=1)[0]
    
    # === Final prediction: average probabilities ===
    prob_final = (prob_orig + prob_flip) / 2
    predicted_idx = torch.argmax(prob_final).item()
    predicted_class = class_names[predicted_idx]

    # === ALWAYS generate Grad-CAM heatmap ===
    # Use flipped version for better heatmap quality due to bias
    target_layer = model.layer3[0].conv2  # Your chosen layer
    gradcam = GradCAM(model, target_layer)
    heatmap_flip = gradcam.generate_heatmap(tensor_flip, predicted_idx)
    
    heatmap_flip_np = heatmap_flip.squeeze().cpu().numpy()
    
    # Flip heatmap back to original orientation
    heatmap_corrected = np.fliplr(heatmap_flip_np)
    
    # Overlay on ORIGINAL image (224x224)
    original_resized = original_image.resize((224, 224))
    original_cv = cv2.cvtColor(np.array(original_resized), cv2.COLOR_RGB2BGR)
    heatmap_cv = cv2.applyColorMap(np.uint8(255 * heatmap_corrected), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_cv, 0.6, heatmap_cv, 0.4, 0)
    
    # Ensure directory and save
    os.makedirs(os.path.dirname(heatmap_image_path), exist_ok=True)
    cv2.imwrite(heatmap_image_path, overlay)

    return predicted_class, prob_final

def mediscan(image_path):
    compare_model_path = "./models/adults_vs_peads_cxrs_model.pth"
    compare_model = load_model(compare_model_path, compare_class_names)
    age_class, _ = evaluate_image(compare_model, image_path, compare_class_names)

    if age_class == "Adults":
        model_path = "./models/adults_cxrs_model.pth"
    elif age_class == "Peads":
        model_path = "./models/peadiatrics_cxrs_model.pth"
    else:
        raise ValueError("Unexpected age classification")

    definite_model = load_model(model_path, definite_class_names)
    predicted_class, probabilities = evaluate_image(definite_model, image_path, definite_class_names)
    confidence_level = calculate_confidence_margin(probabilities)

    return predicted_class, probabilities, confidence_level