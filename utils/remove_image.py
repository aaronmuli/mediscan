import os

def remove_image(image_paths):
    for image in image_paths:
        os.remove(image)
