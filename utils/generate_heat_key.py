import cv2
import numpy as np
from PIL import Image, ImageDraw

# Create vertical color bar
height, width = 300, 50
gradient = np.linspace(0, 255, height, dtype=np.uint8)[:, np.newaxis]
gradient = np.repeat(gradient, width, axis=1)

# Apply JET colormap
key_cv = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

# Add text with PIL
img = Image.fromarray(cv2.cvtColor(key_cv, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img)
draw.text((width + 10, 10), "High Importance", fill="red", fontsize=20)
draw.text((width + 10, height // 2), "Model Attention", fill="black", fontsize=24)
draw.text((width + 10, height - 30), "Low Importance", fill="blue", fontsize=20)

img.save('heatmap_key.jpg')
print("Heatmap key saved!")