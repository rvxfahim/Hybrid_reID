"""
YOLO-based human detection module
"""

import os
import cv2
import numpy as np
import time
from profiling import profile_function

class YOLODetector:
    def __init__(self, model_path=None, conf_threshold=0.25, device='cuda'):
        """
        Initialize YOLO detector specialized for human detection
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = device
        
        try:
            from ultralytics import YOLO
            
            # Change the default model to YOLOv11
            if model_path is None:
                self.model = YOLO("yolo11n.pt").to(self.device)  # Use YOLOv11 nano model with device
            else:
                self.model = YOLO(model_path).to(self.device)
                
            self.using_ultralytics = True
            print(f"Using YOLOv11 from ultralytics (human detection only)")
            
        except ImportError:
            print("Ultralytics YOLO not available, using OpenCV DNN module")
            # Fall back to OpenCV DNN (note: this fallback won't support YOLOv11)
            self.model = cv2.dnn.readNetFromDarknet(
                os.path.join("yolo", "yolov4.cfg"),
                os.path.join("yolo", "yolov4.weights")
            )
            self.using_ultralytics = False
            
            # Set backend and target
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Load COCO class names
            with open(os.path.join("yolo", "coco.names"), "r") as f:
                self.classes = f.read().strip().split("\n")
            
            print("Warning: Fallback mode does not support YOLOv11")
        
        self.conf_threshold = conf_threshold
        # COCO dataset: class 0 is 'person'
        self.person_class_id = 0  # Human class ID in COCO dataset
    
    @profile_function
    def detect(self, frame):
        """
        Detect humans in the frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of human detections as [x1, y1, x2, y2, confidence, class_id]
        """
        # Start timing specifically the model inference part
        inference_start = time.time()
        
        if self.using_ultralytics:
            # Use YOLO from ultralytics - only detect humans (class 0)
            # Add class filtering to the model prediction to reduce computation
            results = self.model(frame, device=self.device, classes=[self.person_class_id])
            
            # Calculate and print inference time
            inference_time = time.time() - inference_start
            print(f"YOLO model inference: {inference_time*1000:.1f}ms")
            
            # Process detections (should only be humans due to class filtering)
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = box.cls[0].item()
                    
                    # Skip if conf is below threshold 
                    # (class check is redundant here since we filtered in the model call)
                    if conf < self.conf_threshold:
                        continue
                    
                    detections.append([x1, y1, x2, y2, conf, cls])
            
            return detections
        else:
            # Use OpenCV DNN
            height, width = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.model.setInput(blob)
            
            # Get output layer names
            out_layer_names = self.model.getLayerNames()
            out_layer_names = [out_layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
            
            # Run forward pass (inference)
            outputs = self.model.forward(out_layer_names)
            
            # Calculate and print inference time
            inference_time = time.time() - inference_start
            print(f"YOLO model inference: {inference_time*1000:.1f}ms")
            
            # Process detections - filter for humans only
            detections = []
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Only include human detections (class 0 in COCO)
                    if class_id == self.person_class_id and confidence > self.conf_threshold:
                        # Scale bounding box coordinates to original image size
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x1 = int(center_x - w/2)
                        y1 = int(center_y - h/2)
                        x2 = x1 + w
                        y2 = y1 + h
                        
                        detections.append([x1, y1, x2, y2, confidence, class_id])
            
            return detections