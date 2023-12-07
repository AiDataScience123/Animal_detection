import torch
import cv2
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog

# Function to find the YOLOv7 model file in the directory
def find_model():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    print("Please place a model file in this directory!")

# Load the YOLOv7 model
def load_yolov7_model():
    model_name = find_model()
    if model_name is None:
        print("Model file not found.")
        return None
    model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)
    model.eval()
    return model

# Function to perform object detection on an image
def detect_objects(image, model):
    results = model(image)
    return results

# Process the video file with increased speed
def process_video(video_path, output_path, model):
    if model is None:
        return

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))

    frame_counter = 0
    processing_interval = 2  # Process every 3rd frame (adjust this value as needed)

    while True:
        ret, frame = cap.read()
        frame_counter += 1

        if not ret:
            break

        if frame_counter % processing_interval == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = detect_objects(pil_img, model)

            for result in results.pred:
                box = result[:, :4]
                cls = result[:, 5]
                conf = result[:, 4]

                for idx in range(len(conf)):
                    if conf[idx] > 0.5:
                        x1, y1, x2, y2 = map(int, box[idx])
                        label = f"{model.names[int(cls[idx])]}: {conf[idx]:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                                    cv2.LINE_AA)

            out.write(frame)
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Function to handle video upload
def upload_video():
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi")])
    if file_path:
        output_file_path = 'output_video_with_objects.avi'  # Output file path
        model = load_yolov7_model()  # Load YOLOv7 model
        process_video(file_path, output_file_path, model)

# Create a simple tkinter GUI for uploading a video file
root = tk.Tk()
root.title("YOLOv7 Object Detection on Video")

upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(pady=20)

root.mainloop()
