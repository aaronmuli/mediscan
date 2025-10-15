import os

def remove_image(image_path):
    os.remove(image_path)
    print(f"Removed {image_path}")
