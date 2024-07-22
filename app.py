import cv2
import mediapipe as mp
import numpy as np
import pickle
from flask import Flask, render_template, Response, jsonify
#Initializes a Flask application.
app = Flask(__name__)

# Load the trained model from the pickle file
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# Initialize the camera (Change the index as necessary)
cap = cv2.VideoCapture(0)  # Change to 1 or other indices if using a different camera

# Initialize MediaPipe Hands for handmark
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Define the labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'VICTORY'}

#Defining Routes:
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    return jsonify(predict_hand_sign())

#Reads frames from the webcam, processes them, and encodes them as JPEG images to stream to the web page.
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#Converts the frame to RGB, processes it with MediaPipe Hands, and draws the landmarks on the frame.
def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    return frame

#Reading and Processing the Image:
def predict_hand_sign():
    success, frame = cap.read()
    if not success:
        return "No frame"
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    #Processing Hand Landmarks:
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)

            if len(x_) > 0 and len(y_) > 0:
                x_min = min(x_)
                y_min = min(y_)
                data_aux = [(x - x_min) for x in x_] + [(y - y_min) for y in y_]

                if len(data_aux) == 42:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_class = int(prediction[0])
                    predicted_character = labels_dict.get(predicted_class, '?')
                    return predicted_character
                else:
                    return "Error in data length"
    return "No hand detected"

if __name__ == '__main__':
    app.run(debug=True)

#This script creates a Flask web application that performs real-time hand sign language detection using a webcam and a pre-trained model. 
# It streams the video feed to a web page and predicts the hand sign using MediaPipe Hands.