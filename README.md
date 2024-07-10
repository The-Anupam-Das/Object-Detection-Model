# OpenCV-Object-Detection <img src="https://skillicons.dev/icons?i=python"/>
# Introduction
This project demonstrates an Object Detection Model using Python, OpenCV, and YOLO (You Only Look Once). YOLO is a real-time object detection system capable of processing images and detecting objects in one pass through a neural network.

Setup
Install Dependencies:

bash
Copy code
pip install opencv-python opencv-python-headless
Download YOLO Files:

Download the YOLOv3 or YOLOv4 weights from the official YOLO website.
Download the corresponding configuration file (.cfg) and the COCO names file (coco.names).
Usage
Step 1: Load YOLO
Load the YOLO network using OpenCV:

python
Copy code
import cv2

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
Step 2: Load and Preprocess Image
Read and preprocess the image:

python
Copy code
# Load image
img = cv2.imread("image.jpg")
height, width, channels = img.shape

# Preprocessing
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
Step 3: Object Detection
Perform object detection:

python
Copy code
# Run detection
outs = net.forward(output_layers)

# Analyze detections
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
Step 4: Draw Bounding Boxes
Draw bounding boxes around detected objects:

python
Copy code
# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Non-max suppression to remove overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

# Show image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
Performance and Optimization
YOLO is optimized for real-time object detection and can run efficiently on a GPU.
For better performance, ensure that OpenCV is built with GPU support.
Applications
Surveillance and security systems
Autonomous vehicles
Robotics and automation
Further Enhancements
Fine-tune YOLO on custom datasets for specific object detection tasks.
Experiment with different YOLO versions (e.g., YOLOv3, YOLOv4) for improved accuracy or speed.
