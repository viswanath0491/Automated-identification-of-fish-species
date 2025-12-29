import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'fish_model.keras'
LABELS_PATH = 'class_names.txt'
IMG_SIZE = (150, 150)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- GLOBAL VARIABLES ---
model = None
class_names = []

def load_model_and_labels():
    global model, class_names
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(LABELS_PATH, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            print("✅ Model and Labels loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("⚠️ Model or Labels file not found. Please run train_model.py first.")
        return False

# Initial attempt to load
load_model_and_labels()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    user_image = None
    error_msg = None

    # Check if model is loaded every time a POST is made
    if not model:
        load_model_and_labels()

    if request.method == 'POST':
        if not model:
            error_msg = "Model not found! Please run 'python train_model.py' and restart the app."
        elif 'file' not in request.files:
            error_msg = "No file uploaded."
        else:
            file = request.files['file']
            if file.filename == '':
                error_msg = "No file selected."
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                user_image = filepath

                try:
                    # 1. Process Image
                    img = tf.keras.utils.load_img(filepath, target_size=IMG_SIZE)
                    img_array = tf.keras.utils.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0) # Create batch axis

                    # 2. Predict
                    preds = model.predict(img_array)
                    # Use softmax if your model output is raw logits, 
                    # or just argmax if it already has a softmax layer
                    score = tf.nn.softmax(preds[0]) 
                    
                    class_idx = np.argmax(score)
                    prediction = class_names[class_idx]
                    confidence = f"{100 * np.max(score):.2f}%"
                except Exception as e:
                    error_msg = f"Prediction error: {e}"

    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence, 
                           user_image=user_image,
                           error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True, port=5000)