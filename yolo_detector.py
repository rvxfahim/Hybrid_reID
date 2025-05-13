"""
YOLO-based human detection module
"""

import os
import cv2
import numpy as np
import time
from profiling import profile_function

class YOLODetector:
    def __init__(self, model_path=None, conf_threshold=0.25, device='cpu'):
        """
        Initialize YOLO detector specialized for human detection
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.person_class_id = 0  # Human class ID in COCO dataset
        
        try:
            from ultralytics import YOLO
            
            # Change the default model to a YOLOv8 segmentation model
            if model_path is None:
                # User should ensure 'yolov8n-seg.pt' or their desired segmentation model is available
                print("Attempting to load default segmentation model: yolov8n-seg.pt")
                self.model = YOLO("yolov8n-seg.pt").to(self.device)
            else:
                print(f"Loading specified model: {model_path}")
                self.model = YOLO(model_path).to(self.device)
                
            self.using_ultralytics = True
            # Check if the loaded model has segmentation capabilities
            if not hasattr(self.model, 'predict') or not callable(getattr(self.model, 'predict')) or not any(hasattr(res, 'masks') for res in self.model(np.zeros((224,224,3), dtype=np.uint8))):
                 print(f"Warning: Model {model_path or 'yolo11n-seg.pt'} might not be a segmentation model or is not behaving as expected.")
            else:
                 print(f"Using YOLO segmentation model from ultralytics (human detection only)")

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
    
    @profile_function
    def detect(self, frame):
        """
        Detect humans in the frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of human detections as [x1, y1, x2, y2, confidence, class_id, mask_tensor_or_None]
        """
        inference_start = time.time()
        
        if self.using_ultralytics:
            results = self.model(frame, device=self.device, classes=[self.person_class_id], verbose=False) # Added verbose=False
            
            inference_time = time.time() - inference_start
            # print(f"YOLO model inference: {inference_time*1000:.1f}ms") # Optional
            
            detections = []
            for result in results: # Iterates over images in batch (usually 1)
                boxes = result.boxes
                masks = result.masks  # Get masks object

                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes[i]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = box.cls[0].item()
                        
                        current_mask_tensor = None
                        if masks is not None and masks.data is not None and i < len(masks.data):
                            # masks.data contains the mask tensors [H, W]
                            # It's important that these masks are correctly aligned with the boxes
                            current_mask_tensor = masks.data[i] 
                        
                        if conf < self.conf_threshold:
                            continue
                        
                        detections.append([x1, y1, x2, y2, conf, cls, current_mask_tensor])
            return detections
        else:
            # OpenCV DNN fallback (does not support segmentation masks)
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
                        
                        detections.append([x1, y1, x2, y2, confidence, class_id, None]) # Add None for mask
            
            return detections