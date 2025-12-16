import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv  
import os


load_dotenv()

# Configure with your credentials (recommended: use environment variables for security)
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"), 
    api_key=os.getenv("API_KEY"),        
    api_secret=os.getenv("API_SECRET"),
    secure=True  # Use HTTPS
)

def upload_cloudinary_image(file_path):
    # Upload an image from a local file path
    upload_result = cloudinary.uploader.upload(
        file=file_path, 
        folder="mediscan",          
        overwrite=True,                    
        resource_type="image"              
    )

    return upload_result["secure_url"], upload_result["public_id"]

