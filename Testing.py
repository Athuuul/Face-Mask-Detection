from flask import Flask, render_template, request, jsonify
import cv2
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import base64

app = Flask(_name_)

# Define IMAGE_SIZE constant
IMAGE_SIZE = 224

# Function to capture image and save it
def capture_image(name):
    # Create a folder if it doesn't exist
    folder_name = "captured_images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return jsonify({'error': 'Unable to access the camera'}), 500

    while True:
        ret, frame = camera.read()
        if ret:
            cv2.imshow('Capture Image - Press Space to Capture', frame)
            key = cv2.waitKey(1)
            if key == ord(' '):  # Press space to capture
                img_name = f"{folder_name}/{name}.png"
                cv2.imwrite(img_name, frame)
                print(f"{name} image has been saved!")
                camera.release()
                cv2.destroyAllWindows()
                return True
            elif key == ord('q'):  # Press q to quit without capturing
                camera.release()
                cv2.destroyAllWindows()
                return False
        else:
            print("Failed to read frame from camera")
            break

# Function to train the recognizer
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    image_files = os.listdir("captured_images/")
    faces = []
    face_ids = []
    id_map = {}  # Dictionary to map names to integer IDs
    current_id = 0

    accuracies = []  # List to store accuracy values during training

    # Your existing code for training recognizer here

# Function to recognize faces
def recognize_faces():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_recognizer.yml')  # Load trained recognizer
    id_map = {}  # Dictionary to map names to integer IDs
    with open("id_map.txt", "r") as f:
        for line in f:
            name, face_id = line.strip().split(":")
            id_map[name] = int(face_id)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return jsonify({'error': 'Unable to access the camera'}), 500

    model = load_model('model.keras')
    MODEL_THRESHOLD = 0.75

    while True:
        ret, frame = camera.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # Detecting mask
                face = cv2.resize(roi_color, (IMAGE_SIZE, IMAGE_SIZE))
                face = np.expand_dims(face, axis=0)
                pred = model.predict(face)
                if np.max(pred) > MODEL_THRESHOLD:
                    label = 'Mask Found'
                else:
                    label = 'No Mask Found'

                # Recognizing face
                face_id, confidence = recognizer.predict(roi_gray)
                for name, id_ in id_map.items():
                    if id_ == face_id:
                        recognized_name = name
                        break
                else:
                    recognized_name = "Unknown"

                label_text = f"Name: {recognized_name}\nMask: {label}"
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow('Face Recognition - Press q to Quit', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to read frame from camera")
            break

    camera.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    if not name:
        return jsonify({'error': 'Name is required'}), 400
    capture_image(name)
    recognizer = train_recognizer()
    if recognizer:
        recognizer.save('trained_recognizer.yml')
        return jsonify({'message': 'Registration successful'})
    else:
        return jsonify({'error': 'Failed to train recognizer'}), 500

@app.route('/scan', methods=['POST'])
def scan():
    recognize_faces()
    return jsonify({'message': 'Scanning successful'})

if _name_ == "_main_":
    app.run()
