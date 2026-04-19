from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
import numpy as np
import sys

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import your custom scripts
from scripts.feature_extraction import extract_color_features, extract_texture_features
from image_clustering import cluster_images
from grad_cam import generate_grad_cam
from metadata_annotation import annotate_metadata

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained model
MODEL_PATH = 'models/tomato_disease_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the image upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image for the model
        image = cv2.imread(file_path)
        image_resized = cv2.resize(image, (224, 224))  # Resize to match model input
        image_normalized = image_resized / 255.0  # Normalize
        image_for_model = image_normalized.reshape((1, 224, 224, 3))  # Reshape for the model

        # Make predictions using the CNN
        prediction = model.predict(image_for_model)
        predicted_class = prediction.argmax()  # Assuming you have 10 classes
        disease_name = get_disease_name(predicted_class)  # Map class to disease name

        # Feature Extraction (Color and Texture)
        color_features = extract_color_features(image)  # Custom function from feature_extraction.py
        texture_features = extract_texture_features(image)  # Custom function from feature_extraction.py

        # Cluster Image Based on Features (e.g., K-Means Clustering)
        cluster_id = cluster_images(image)  # Custom function from image_clustering.py

        # Grad-CAM: Visualize model activation map for the given image
        grad_cam_image = generate_grad_cam(model, image_for_model, predicted_class)

        # Metadata Annotation (e.g., "yellow spots on edges" for early blight)
        metadata = annotate_metadata(disease_name, color_features, texture_features)

        # Save Grad-CAM result to static folder for displaying
        grad_cam_filename = f"grad_cam_{filename}"
        grad_cam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], grad_cam_filename)
        cv2.imwrite(grad_cam_filepath, grad_cam_image)

        # Render the result page with disease information and Grad-CAM image
        return render_template(
            'result.html', 
            result=disease_name, 
            image_url=file_path, 
            grad_cam_url=grad_cam_filepath,
            metadata=metadata,
            cluster=cluster_id
        )

def get_disease_name(predicted_class):
    disease_dict = {
        0: "Bacterial_spot",
        1: "Early_blight",
        2: "Late_blight",
        3: "Leaf_Mold",
        4: "Powdery_mildew",
        5: "Septoria_leaf_spot",
        6: "Spider_mites",
        7: "Target_Spot",
        8: "Tomato_mosaic_virus",
        9: "Tomato_Yellow_Leaf_Curl_Virus"
    }
    return disease_dict.get(predicted_class, "Unknown Disease")

if __name__ == '__main__':
    app.run(debug=True)
