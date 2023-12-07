import cv2
import io
import os
from PIL import Image
import argparse
import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Find the model inside your directory automatically - works only if there is one model
def find_model():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    print("Please place a model file in this directory!")

model_name = find_model()
model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)
model.eval()

# Function to get predictions from the model
def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # Batched list of images
    # Inference
    results = model(imgs, size=640)  # Includes NMS
    return results

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()
        results = get_prediction(img_bytes)

        # Retrieve predicted animal name (adjust this based on your model's output format)
        animal_name = results.names[int(results.xyxy[0][0][-1])] if results.xyxy[0].shape[0] > 0 else "Not detected"

        # Save the uploaded image
        filename = 'uploaded_image.jpg'
        file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        with open(file_path, 'wb') as f:
            f.write(img_bytes)

        return render_template('result.html', result_image=filename, model_name=model_name, animal_name=animal_name)

    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def handle_video():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Process video file using OpenCV
        video_file = os.path.join(app.config['RESULT_FOLDER'], file.filename)
        file.save(video_file)

        # Your video processing code using OpenCV goes here

    return render_template('video.html')

@app.route('/webcam', methods=['GET', 'POST'])
def web_cam():
    if request.method == 'POST':
        # Access the webcam using OpenCV
        cap = cv2.VideoCapture(0)  # Webcam index (0 for default webcam)

        # Your webcam processing code using OpenCV goes here
        while True:
            ret, frame = cap.read()
            # Process the frame (perform inference, etc.)
            # Display the frame in the browser using Flask (convert it to bytes)
            ret, buffer = cv2.imencode('.jpg', frame)
            img_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

    return app.response_class(web_cam(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov7 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
