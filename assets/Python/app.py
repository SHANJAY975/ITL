from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import os

# Set up Flask app
app = Flask(__name__, template_folder='../../', static_folder='../../assets')

# Construct dynamic paths for the model and video
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../Model/best.pt')
VIDEO_PATH = os.path.join(os.path.dirname(__file__), '../Model/video.mp4')

# Load the YOLO model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
else:
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

# Function to generate video frames and process them with YOLO
def gen():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found at {VIDEO_PATH}")
    
    cap = cv2.VideoCapture(VIDEO_PATH)  # Use the video file path or 0 for live webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame using YOLO
        results = model(frame)
        annotated_frame = results[0].plot()

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    # Serve the main HTML file (index.html)
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Route to stream the video feed
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
