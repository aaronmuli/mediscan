from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import datetime
import threading

from utils.mediscan import mediscan
from utils.upload_cloud import upload_cloudinary_image
from utils.remove_image import remove_image

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
        # heatmap_file_path = '' # to display on the frontend
        # heatmap_file = '' # to remove after display

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('404.html', message="No selected file.", current_year=current_year), 400
        
        if file.mimetype not in ALLOWED_IMAGE_MIMETYPES:
            return render_template('404.html', message="Only Images are allowed", current_year=current_year), 400

        if file:
            secure_url_og, public_id_og = upload_cloudinary_image(file)
            
            predicted_class, probabilities, confidence_level, secure_url_heatmap = mediscan(image_path=secure_url_og)

            class_names = ['Normal', 'Abnormal']
            list_probabilities = []

            for i, cls in enumerate(class_names):
                probs = dict()
                probs["class"] = cls
                probs["value"] = f"{probabilities[i].item():.4f}"
                list_probabilities.append(probs)
            
            
            # heatmap_file_path = url_for('static', filename='uploads/gradcam.jpg')
            # heatmap_file = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam.jpg')

            # secure_url_heatmap, public_id_heatmap = upload_cloudinary_image(heatmap_file)  

            # timer = threading.Timer(10, remove_image, args=[[file_path, heatmap_file]])
            # timer.start()

            return render_template(
                'index.html',
                heatmap_image=secure_url_heatmap,
                image_url=secure_url_og,
                predicted_class=predicted_class,
                probabilities=list_probabilities,
                confidence_level=confidence_level,
                current_year=str(current_year)
            )
    return render_template('index.html', current_year=str(current_year))

@app.errorhandler(404)
def onError(err):
    return render_template('404.html', message="Page Not Found", current_year=str(current_year)), 404

if __name__ == '__main__':
    if os.getenv("FLASK_ENV") == "development":
        app.run(debug=True, port=5000)
    else:
        app.run(port=5000)
