from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from utils.mediscan import mediscan
from dotenv import load_dotenv
import os
import datetime
import time
import threading

from utils.remove_image import remove_image

app = Flask(__name__)

load_dotenv()

# Configure the upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
current_year = datetime.datetime.now().year


# Route for the upload form
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    image_url = None

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
            # Secure the filename to prevent malicious attacks
            filename = secure_filename(file.filename)
            
            # Create the full server path for saving the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(file_path)
            
            # Get the URL that can be used in the template to display the image
            image_url = url_for('static', filename=f'uploads/{filename}')
            
            predicted_class, probabilities = mediscan(image_path=file_path)

            class_names = ['Normal', 'Abnormal']
            list_probabilities = []

            for i, cls in enumerate(class_names):
                probs = dict()
                probs["class"] = cls
                probs["value"] = f"{probabilities[i].item():.4f}"
                list_probabilities.append(probs)

            timer = threading.Timer(10, remove_image, args=[file_path])
            print(image_url)
            timer.start()

            return render_template(
                'index.html',
                image_url=image_url,
                predicted_class=predicted_class,
                probabilities=list_probabilities,
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
