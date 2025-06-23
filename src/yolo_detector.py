"""
YOLO-based human detection module
"""

import os
import cv2
import numpy as np
import time
from profiling import profile_function

class YOLODetector:
    def __init__(self, model_path=None, conf_threshold=0.25, device='cpu', use_tensorrt=False):
        """
        Initialize YOLO detector specialized for human detection
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold
            device: Computing device ('cuda' or 'cpu')
            use_tensorrt: Whether to use TensorRT for acceleration (requires CUDA device)
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.person_class_id = 0  # Human class ID in COCO dataset
        self.use_tensorrt = use_tensorrt
        
        try:
            from ultralytics import YOLO
            
            # If TensorRT is requested but device is not CUDA, warn and fall back
            if self.use_tensorrt and self.device != 'cuda':
                print("Warning: TensorRT acceleration requires CUDA device. Falling back to standard inference.")
                self.use_tensorrt = False
                
            # Handle model loading based on TensorRT preference
            if model_path is None:
                # Default model path for standard or TensorRT
                if self.use_tensorrt:
                    # Check if TensorRT engine exists for default model
                    engine_path = "yolov8n-seg.engine"
                    if os.path.exists(engine_path):
                        print(f"Loading TensorRT engine: {engine_path}")
                        self.model = YOLO(engine_path)
                    else:
                        print(f"TensorRT engine not found at {engine_path}. Loading standard model and exporting to TensorRT...")
                        base_model = YOLO("yolov8n.pt").to(self.device)
                        # Export to TensorRT
                        base_model.export(format="engine")
                        # Load the exported TensorRT model
                        self.model = YOLO(engine_path)
                else:
                    # Standard model loading
                    print("Attempting to load default segmentation model: yolov8n-seg.pt")
                    self.model = YOLO("yolov8n-seg.pt").to(self.device)
            else:
                # Custom model path
                if self.use_tensorrt:
                    # Check if this is already a TensorRT engine
                    if model_path.endswith('.engine'):
                        print(f"Loading TensorRT engine: {model_path}")
                        self.model = YOLO(model_path)
                    else:
                        # Check if corresponding engine exists
                        engine_path = os.path.splitext(model_path)[0] + ".engine"
                        if os.path.exists(engine_path):
                            print(f"Loading TensorRT engine: {engine_path}")
                            self.model = YOLO(engine_path)
                        else:
                            print(f"TensorRT engine not found. Loading standard model and exporting to TensorRT...")
                            base_model = YOLO(model_path).to(self.device)
                            # Export to TensorRT
                            base_model.export(format="engine")
                            # Load the exported engine
                            self.model = YOLO(engine_path)
                else:
                    # Standard model loading
                    print(f"Loading specified model: {model_path}")
                    self.model = YOLO(model_path).to(self.device)
                
            self.using_ultralytics = True
            # Check if the loaded model has segmentation capabilities
            if not hasattr(self.model, 'predict') or not callable(getattr(self.model, 'predict')) or not any(hasattr(res, 'masks') for res in self.model(np.zeros((224,224,3), dtype=np.uint8))):
                 print(f"Warning: Model {model_path or 'yolov8n-seg.pt'} might not be a segmentation model or is not behaving as expected.")
            else:
                 if self.use_tensorrt:
                     print(f"Using YOLO segmentation model with TensorRT acceleration (human detection only)")
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
            # When using TensorRT, the device is already configured in the engine
            # so we don't need to pass it explicitly
            if self.use_tensorrt:
                results = self.model(frame, classes=[self.person_class_id], verbose=False)
            else:
                results = self.model(frame, device=self.device, classes=[self.person_class_id], verbose=False)
            
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
    
    def export_to_tensorrt(self, model_path=None, save_path=None):
        """
        Export the current or specified YOLO model to TensorRT format
        
        Args:
            model_path: Path to the model to export (if None, uses the current model)
            save_path: Path to save the exported model (if None, creates a .engine file next to the model)
            
        Returns:
            Path to the exported TensorRT engine file or None if export failed
        """
        if not self.using_ultralytics:
            print("TensorRT export requires Ultralytics YOLO.")
            return None
            
        try:
            from ultralytics import YOLO
            
            # Check if CUDA is available (required for TensorRT)
            if self.device != 'cuda':
                print("Warning: TensorRT export requires CUDA. Current device is:", self.device)
                return None
                
            # Determine which model to export
            if model_path is None:
                # Export the currently loaded model
                if hasattr(self.model, 'export'):
                    source_model = self.model
                    # If save_path is not specified, create one based on the model's ckpt path
                    if save_path is None and hasattr(self.model, 'ckpt_path'):
                        model_name = os.path.basename(self.model.ckpt_path)
                        base_name = os.path.splitext(model_name)[0]
                        save_path = f"{base_name}.engine"
                else:
                    print("Current model doesn't support export.")
                    return None
            else:
                # Load the specified model
                print(f"Loading model for export: {model_path}")
                source_model = YOLO(model_path)
                
                # If save_path is not specified, create one based on the input model path
                if save_path is None:
                    base_name = os.path.splitext(model_path)[0]
                    save_path = f"{base_name}.engine"
            
            # Perform the export
            print(f"Exporting model to TensorRT format at: {save_path}")
            result = source_model.export(format="engine", device=self.device)
            
            print(f"Export completed. Engine file created at: {result}")
            return result
            
        except Exception as e:
            print(f"Error during TensorRT export: {e}")
            return None
        
    def toggle_tensorrt(self, use_tensorrt=None):
        """
        Toggle or set TensorRT acceleration mode
        
        Args:
            use_tensorrt: If provided, sets TensorRT mode to this value. If None, toggles current state.
            
        Returns:
            Current TensorRT state after toggling/setting
        """
        if not self.using_ultralytics:
            print("TensorRT acceleration requires Ultralytics YOLO.")
            return False
            
        # Determine the new TensorRT state
        new_tensorrt_state = not self.use_tensorrt if use_tensorrt is None else use_tensorrt
        
        # If no change needed, return current state
        if new_tensorrt_state == self.use_tensorrt:
            print(f"TensorRT acceleration is already {'enabled' if self.use_tensorrt else 'disabled'}")
            return self.use_tensorrt
            
        # Check if CUDA is available for TensorRT
        if new_tensorrt_state and self.device != 'cuda':
            print("Warning: TensorRT acceleration requires CUDA device. Cannot enable.")
            return False
            
        try:
            from ultralytics import YOLO
            
            # Store the current model path if possible
            current_model_path = None
            if hasattr(self.model, 'ckpt_path'):
                current_model_path = self.model.ckpt_path
            
            if new_tensorrt_state:
                # Switching to TensorRT mode
                if current_model_path:
                    # Check if engine already exists
                    engine_path = os.path.splitext(current_model_path)[0] + ".engine"
                    if os.path.exists(engine_path):
                        print(f"Loading existing TensorRT engine: {engine_path}")
                        self.model = YOLO(engine_path)
                        self.use_tensorrt = True
                    else:
                        # Export and load TensorRT model
                        print(f"Exporting model to TensorRT format: {current_model_path}")
                        self.model.export(format="engine", device=self.device)
                        self.model = YOLO(engine_path)
                        self.use_tensorrt = True
                else:
                    print("Cannot enable TensorRT: Unable to determine current model path")
                    return False
            else:
                # Switching back to standard mode
                if current_model_path:
                    # Check if current model is an engine
                    if current_model_path.endswith('.engine'):
                        # Load the original model
                        original_model_path = os.path.splitext(current_model_path)[0] + ".pt"
                        if os.path.exists(original_model_path):
                            print(f"Loading standard model: {original_model_path}")
                            self.model = YOLO(original_model_path).to(self.device)
                            self.use_tensorrt = False
                        else:
                            print(f"Cannot find original model: {original_model_path}")
                            return self.use_tensorrt
                    else:
                        # Already using a standard model
                        self.use_tensorrt = False
                else:
                    print("Cannot disable TensorRT: Unable to determine current model path")
                    return self.use_tensorrt
                    
            print(f"TensorRT acceleration is now {'enabled' if self.use_tensorrt else 'disabled'}")
            return self.use_tensorrt
            
        except Exception as e:
            print(f"Error toggling TensorRT mode: {e}")
            return self.use_tensorrt