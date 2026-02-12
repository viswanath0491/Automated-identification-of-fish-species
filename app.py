import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, Response
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'fish_model.keras'
LABELS_PATH = 'class_names.txt'
IMG_SIZE = (150, 150)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- GLOBAL MODEL LOADER ---
# We load the model ONCE at the start to prevent lag
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    class_names = []

# --- CAMERA HANDLING ---
class VideoCamera(object):
    def __init__(self):
        # Open the webcam (0 is usually the default)
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        
        # 1. Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # 2. Define Scanning Region (Center Box)
        height, width, _ = frame.shape
        start_point = (int(width/2 - 110), int(height/2 - 110))
        end_point = (int(width/2 + 110), int(height/2 + 110))
        
        # 3. Extract ROI (Region of Interest) for Prediction
        roi = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        
        # 4. Run Prediction Logic (Only if model is loaded)
        label_text = "Scanning..."
        color = (0, 255, 255) # Yellow

        if model is not None and roi.size != 0:
            try:
                # Resize to 150x150 for the model
                roi_resized = cv2.resize(roi, IMG_SIZE)
                roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                roi_array = tf.keras.utils.img_to_array(roi_rgb)
                roi_array = tf.expand_dims(roi_array, 0)

                preds = model.predict(roi_array, verbose=0)
                score = tf.nn.softmax(preds[0])
                class_idx = np.argmax(score)
                confidence = 100 * np.max(score)

                if confidence > 70:
                    label_text = f"{class_names[class_idx]} ({confidence:.1f}%)"
                    color = (0, 255, 0) # Green
                else:
                    label_text = "Unknown Species"
                    color = (0, 0, 255) # Red
            except Exception as e:
                print(f"Prediction Error: {e}")

        # 5. Draw UI on Frame
        cv2.rectangle(frame, start_point, end_point, color, 2)
        cv2.putText(frame, label_text, (start_point[0], start_point[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 6. Encode for Web
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    # (Your existing Static Image Logic goes here - unchanged)
    prediction = None
    confidence = None
    user_image = None
    
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                user_image = filepath
                
                # Static Image Prediction Logic
                if model:
                    img = tf.keras.utils.load_img(filepath, target_size=IMG_SIZE)
                    img_array = tf.keras.utils.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)
                    preds = model.predict(img_array)
                    score = tf.nn.softmax(preds[0])
                    class_idx = np.argmax(score)
                    prediction = class_names[class_idx]
                    confidence = f"{100 * np.max(score):.2f}%"

    return render_template('index.html', prediction=prediction, confidence=confidence, user_image=user_image)

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # threaded=True is important for handling multiple requests (webcam + page load)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    