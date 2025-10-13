from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from mediscan import mediscan
import os
import datetime

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for the upload form
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    image_url = None

    current_year = datetime.datetime.now().year

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image_file' not in request.files:
            return "No file part in the form.", 400

        file = request.files['image_file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file.", 400

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

            return render_template(
                'index.html',
                image_url=image_url,
                predicted_class=predicted_class,
                probabilities=list_probabilities,
                current_year=str(current_year)
            )
    
    return render_template('index.html', current_year=str(current_year))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
