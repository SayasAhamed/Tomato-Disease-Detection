from flask import Flask, render_template, request, redirect, send_from_directory
import os
import numpy as np
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename
from utils.disease_info import disease_info

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔥 LOAD BOTH MODELS
models = {
    "resnet": tf.keras.models.load_model('model/resnet50_best.h5'),  
    "mobilenet": tf.keras.models.load_model('model/mobilenet_model.h5')
}

# Class labels
classes = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Powdery_mildew",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect('/')

    file = request.files['file']

    if file.filename == '':
        return redirect('/')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 🔥 GET SELECTED MODEL
        selected_model = request.form.get('model')

        if selected_model not in models:
            selected_model = "resnet"

        model = models[selected_model]

        # Preprocess
        img = cv2.imread(filepath)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.reshape(img, (1, 224, 224, 3))

        # Prediction
        prediction = model.predict(img)
        confidence = np.max(prediction) * 100
        predicted_class = classes[np.argmax(prediction)]

        info = disease_info.get(predicted_class, "No info available")

        return render_template(
            'result.html',
            prediction=predicted_class,
            confidence=round(confidence, 2),
            image=filename,
            info=info,
            model_used=selected_model
        )

    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)