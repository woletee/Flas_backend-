import os
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from PIL import Image
from flask_cors import CORS  # Import CORS

# Disable GPU if not needed
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# S3 public URL
model_url = 'https://eyerus123.s3.us-east-2.amazonaws.com/Final_food_saved_model.h5'
local_model_path = 'Final_food_saved_model.h5'

# Download model from S3 URL
try:
    print(f"Downloading model from {model_url}")
    response = requests.get(model_url)
    response.raise_for_status()  # Check if the request was successful
    with open(local_model_path, 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully")
except Exception as e:
    print(f"Error downloading model: {e}")
    raise FileNotFoundError(f"Model file could not be downloaded from {model_url}")

# Load the trained model
try:
    print(f"Loading model from {local_model_path}")
    model = load_model(local_model_path, compile=False)
    print("Model loaded successfully")
    # Recompile the model with a standard optimizer
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled successfully with Adam optimizer")
except Exception as e:
    print(f"Error loading or compiling model: {e}")
    raise FileNotFoundError(f"Model file not found or could not be loaded from {local_model_path}")

# Define class names
class_names = [
    "bibimbap", "bulgogi", "godeungeogui", "jjambbong", "ramyun",
    "yangnyumchicken", "duinjangjjigae", "gamjatang", "gimbap", "jeyukbokkeum",
    "jjajangmyeon", "kalguksu", "kimchijjigae", "mandu", "pajeon",
    "samgyetang", "samgyeopsal", "sundaegukbap", "tteokbokki", "tteokguk"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Food Recognition API!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img = Image.open(file_path)
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            predicted_label = class_names[predicted_class]

            return jsonify({'foodName': predicted_label}), 200
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Error processing the image'}), 500
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)