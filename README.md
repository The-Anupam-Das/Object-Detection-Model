# OpenCV-Object-Detection <img src="https://skillicons.dev/icons?i=python"/>
# Introduction
This project demonstrates an Object Detection Model using Python, OpenCV, and YOLO (You Only Look Once). YOLO is a real-time object detection system capable of processing images and detecting objects in one pass through a neural network.

## Step 1: Load YOLO
Load the YOLO network using OpenCV:

import cv2
from random import randint
dnn = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

## Step 2: Load and Preprocess
with open('classes.txt') as f:
    classes = f.read().strip().splitlines()

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

## Step 3: Object Detection
Perform object detection:

color_map = {}

while True:
    
    _, frame = capture.read() 
    frame = cv2.flip(frame,1)

    
    class_ids, confidences, boxes = model.detect(frame)
    for id, confidence, box in zip(class_ids, confidences, boxes):
        x, y, w, h = box
        obj_class = classes[id]

        if obj_class not in color_map:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            color_map[obj_class] = color
        else:
            color = color_map[obj_class]

        cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
      
      
    cv2.imshow('Video Capture', frame)
    key = cv2.waitKey(1) 

    match(key):
        case 27: 
            capture.release()
            cv2.destroyAllWindows()

        case 13: 
            color_map = {}
			
# Performance and Optimization
YOLO is optimized for real-time object detection and can run efficiently on a GPU.
For better performance, ensure that OpenCV is built with GPU support.

# Applications
Surveillance and security systems
Autonomous vehicles
Robotics and automation

# Further Enhancements
Fine-tune YOLO on custom datasets for specific object detection tasks.
Experiment with different YOLO versions (e.g., YOLOv3, YOLOv4) for improved accuracy or speed.
