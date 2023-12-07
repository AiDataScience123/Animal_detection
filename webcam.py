import os
import torch
from PIL import Image
import cv2
import io

# Find the model inside your directory automatically - works only if there is one model
def find_model():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    print("Please place a model file in this directory!")

model_name = find_model()
model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)
model.eval()

# Function to perform object detection on an image
def detect_objects(image):
    # Perform object detection
    results = model(image)

    return results


def activate_webcam():
# Open Webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, change accordingly

while True:
    ret, frame = cap.read()

    # Convert the OpenCV frame to a PIL image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform object detection on the PIL image
    results = detect_objects(pil_img)

    # Process and visualize the results
    for result in results.pred:
        # Extract bounding box coordinates, class, and confidence
        box = result[:, :4]
        cls = result[:, 5]
        conf = result[:, 4]

        # Draw bounding box with class name and confidence
        for idx in range(len(conf)):
            if conf[idx] > 0.5:  # Display if confidence is greater than 0.5
                x1, y1, x2, y2 = map(int, box[idx])
                label = f"{model.names[int(cls[idx])]}: {conf[idx]:.2f}"  # Get class name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                            cv2.LINE_AA)  # Show class and confidence

    # Display the video with bounding boxes
    cv2.imshow('YOLOv7 Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()