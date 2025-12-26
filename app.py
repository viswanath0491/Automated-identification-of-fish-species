from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import os
import ast

app = Flask(__name__)
model = None
class_names = {}

# Load model and class names
def load_assets():
    global model, class_names
    # Load the trained model
    model = load_model('fish_model.h5')
    print("Model loaded.")
    
    # Load class dictionary and reverse it to get {0: 'Sea Bass'}
    with open('class_names.txt', 'r') as f:
        class_indices = ast.literal_eval(f.read())
        class_names = {v: k for k, v in class_indices.items()}

# Ensure uploads folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_url = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded"
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        if file:
            # Save the file
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            img_url = filepath

            # Preprocess image for MobileNetV3
            img = image.load_img(filepath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Predict
            preds = model.predict(x)
            predicted_class_index = np.argmax(preds)
            confidence = np.max(preds) * 100
            
            result_text = class_names[predicted_class_index]
            prediction = f"{result_text} ({confidence:.2f}%)"

    return render_template('index.html', prediction=prediction, img_url=img_url)

if __name__ == '__main__':
    load_assets()
    app.run(debug=True)
