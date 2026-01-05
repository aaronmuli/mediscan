from flask import Flask, render_template, request
from dotenv import load_dotenv
from PIL import Image
import os
import datetime
import base64
import io

from utils.mediscan import mediscan


app = Flask(__name__)

load_dotenv()

current_year = datetime.datetime.now().year

# Route for the upload form
@app.route('/', methods=['GET', 'POST'])
def upload_image():

    ALLOWED_IMAGE_MIMETYPES = {
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp',
        'image/bmp',
        'image/tiff',
        'image/svg+xml',
        'image/avif'
    }

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image_file' not in request.files:
            return render_template('404.html', message="No file part in the form.", current_year=current_year), 400

        file = request.files['image_file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('404.html', message="No selected file.", current_year=current_year), 400
        
        if file.mimetype not in ALLOWED_IMAGE_MIMETYPES:
            return render_template('404.html', message="Only Images are allowed", current_year=current_year), 400

        if file:
            original_image = Image.open(file).convert('RGB')
            
            source_original = original_image
            buffered_original = io.BytesIO()
            source_original.save(buffered_original, format='PNG')
            image_bytes = buffered_original.getvalue()
            original_encoded_image = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

            predicted_class, probabilities, confidence_level, source = mediscan(x_ray=original_image)
            image_bytes = source.getvalue()
            heatmap_encoded_image = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

            class_names = ['Normal', 'Abnormal']
            list_probabilities = []

            for i, cls in enumerate(class_names):
                probs = dict()
                probs["class"] = cls
                probs["value"] = f"{probabilities[i].item():.4f}"
                list_probabilities.append(probs)
            
            return render_template(
                'index.html',
                heatmap_image=heatmap_encoded_image,
                image_url=original_encoded_image,
                predicted_class=predicted_class,
                probabilities=list_probabilities,
                confidence_level=confidence_level,
                current_year=str(current_year),
            )

    return render_template('index.html', current_year=str(current_year))

@app.errorhandler(404)
def onError(err):
    return render_template('404.html', message="Page Not Found", current_year=str(current_year)), 404

if __name__ == '__main__':
    if os.getenv("FLASK_ENV") == "development":
        app.run(debug=True, port=5000)
    else:
        app.run(host='0.0.0.0', port=7860)
