"""
Long-Term Object Tracking System with DeepSORT and Offline Re-identification

This implementation demonstrates a hybrid tracking approach that:
1. Uses DeepSORT for short-term, frame-to-frame tracking
2. Incorporates an offline re-identification system to maintain consistent IDs
3. Implements temporal association refinements for improved tracking across occlusions
4. Uses a ResNet50-based feature extractor for appearance embedding
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
import time
import os
from collections import deque
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from PIL import Image

# Deep SORT and feature extraction imports
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection


class FeatureExtractor:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the feature extractor with a pre-trained model
        
        Args:
            model_path: Path to the pre-trained model (if None, use a pre-trained ResNet50)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize model - ResNet50 is commonly used for ReID tasks
        if model_path is None or not os.path.exists(model_path):
            print("Loading pre-trained ResNet50 model")
            self.model = torchvision.models.resnet50(pretrained=True)
            # Remove the final FC layer to get feature vectors
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        else:
            print(f"Loading model from {model_path}")
            self.model = torch.load(model_path, map_location=self.device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),  # Standard size for ReID
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.feature_dim = 2048  # ResNet50 feature dimension
    
    def extract_features(self, frame, bbox):
        """
        Extract features from the given bounding box in the frame
        
        Args:
            frame: Current video frame (BGR format)
            bbox: Bounding box as [x1, y1, x2, y2]
            
        Returns:
            Feature vector (numpy array)
        """
        try:
            # Convert bbox to integers
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame boundaries
            height, width = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(self.feature_dim, dtype=np.float32)
            
            # Crop the object from the frame
            crop = frame[y1:y2, x1:x2]
            
            # Convert from BGR to RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and apply transforms
            img = Image.fromarray(crop)
            img = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img)
                features = features.squeeze().cpu().numpy()
            
            # Normalize the feature vector
            features = features / np.linalg.norm(features)
            
            return features.astype(np.float32)
        
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)


class HybridTracker:
    def __init__(self, max_cosine_distance=0.4, nn_budget=None, max_age=30, min_confidence=0.3,
                 re_id_interval=50, gallery_size=100, iou_threshold=0.3, model_path=None):
        """
        Initialize the hybrid tracker with DeepSORT and Re-ID components
        
        Args:
            max_cosine_distance: Threshold for feature distance in DeepSORT
            nn_budget: Maximum size of the appearance descriptors gallery
            max_age: Maximum number of missed misses before a track is deleted
            min_confidence: Detection confidence threshold
            re_id_interval: How often to run the offline re-identification (in frames)
            gallery_size: Maximum number of object appearances to store in the gallery
            iou_threshold: IoU threshold for association
            model_path: Path to pre-trained model for feature extraction
        """
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(model_path)
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            max_age=max_age,
            nms_max_overlap=1.0
        )
        
        # Parameters
        self.min_confidence = min_confidence
        self.feature_dim = self.feature_extractor.feature_dim
        self.re_id_interval = re_id_interval
        self.gallery_size = gallery_size
        self.iou_threshold = iou_threshold
        
        # Storage for ID management
        self.frame_count = 0
        self.next_id = 1
        self.id_mapping = {}  # Maps DeepSORT temporary IDs to our consistent IDs
        
        # Feature gallery for re-identification
        self.feature_gallery = {}  # {consistent_id: deque of features}
        self.last_seen_frame = {}  # {consistent_id: last frame the object was seen}
        self.inactive_ids = set()  # IDs that left the frame
        
        # Track history for visualization and temporal association
        self.track_history = defaultdict(lambda: deque(maxlen=50))  # {consistent_id: deque of positions}
        
        # For motion prediction
        self.kalman_predictions = {}  # {consistent_id: predicted next position}
        
        print("Hybrid tracker initialized with DeepSORT and Re-ID components")
    
    def update(self, frame, detections):
        """
        Update the tracker with new detections

        Args:
            frame: Current video frame
            detections: List of detections as [x1, y1, x2, y2, confidence, class_id]

        Returns:
            List of tracks as [x1, y1, x2, y2, consistent_id, class_id]
        """
        self.frame_count += 1

        # 1. Format detections for DeepSORT (LTWH format expected)
        formatted_detections = []
        original_bboxes_ltrb = [] # Keep original LTRB for potential feature extraction later if needed

        for i, det in enumerate(detections):
            if len(det) >= 6:
                bbox_ltrb = det[:4]  # [x1, y1, x2, y2]
                confidence = det[4]
                class_id = int(det[5]) # You might need to map this to a class name string depending on DeepSORT setup

                if confidence < self.min_confidence:
                    continue

                # Convert LTRB to LTWH (Left, Top, Width, Height)
                x1, y1, x2, y2 = bbox_ltrb
                w = x2 - x1
                h = y2 - y1

                # Ensure width and height are positive
                if w <= 0 or h <= 0:
                    # print(f"Skipping detection with non-positive width/height: {bbox_ltrb}")
                    continue

                bbox_ltwh = [x1, y1, w, h]

                # Append in the format expected by deep_sort_realtime: (bbox_ltwh, confidence, class_id)
                # Note: We are NOT passing our custom features here. DeepSORT will use its own internal
                # feature extractor based on its configuration.
                formatted_detections.append((bbox_ltwh, confidence, class_id))
                original_bboxes_ltrb.append(bbox_ltrb) # Store original format if needed
            else:
                 print(f"Warning: Detection has unexpected format: {det}")


        # 2. Update DeepSORT tracker
        # It expects [( [x,y,w,h], conf, class_id ), ... ]
        # It returns track objects.
        # Note: The `frame` argument might be used internally by DeepSORT for feature extraction.
        deepsort_tracks_returned = self.tracker.update_tracks(formatted_detections, frame=frame)

        # 3. Process DeepSORT tracks and map to consistent IDs
        current_tracks = []
        current_active_ids = set()

        # Iterate through the tracks returned by DeepSORT
        # These are usually Track objects from the library.
        for track_object in deepsort_tracks_returned:
            # Check if the track object is valid and confirmed
            # Note: The exact conditions might vary slightly based on deep_sort_realtime version.
            # Common checks: is_confirmed(), time_since_update
            if not track_object.is_confirmed() or track_object.time_since_update > 1:
                continue

            track_id = track_object.track_id # This is the temporary DeepSORT ID
            bbox_ltrb = track_object.to_ltrb() # Get LTRB bbox [x1, y1, x2, y2]

            # Retrieve the *most recent feature* associated with this track by DeepSORT's internal extractor
            current_feature = None
            if track_object.features:
                 # Ensure features list is not empty and contains numpy arrays
                 if isinstance(track_object.features[-1], np.ndarray):
                    current_feature = track_object.features[-1]

            # If DeepSORT didn't provide a feature (e.g., config issue, or just Kalman update),
            # we might need to extract it ourselves using the final bbox.
            # However, mixing features from different extractors can be problematic.
            # It's best to rely on DeepSORT's features if possible.
            if current_feature is None:
                 # print(f"Warning: Track {track_id} from DeepSORT has no valid features. Attempting manual extraction.")
                 # Ensure bbox is valid before extraction
                 if bbox_ltrb[2] > bbox_ltrb[0] and bbox_ltrb[3] > bbox_ltrb[1]:
                     current_feature = self.feature_extractor.extract_features(frame, bbox_ltrb)
                     if np.all(current_feature == 0): # Check if extraction failed
                         # print(f"Manual feature extraction failed for track {track_id}.")
                         current_feature = None # Reset to None if failed
                 else:
                     # print(f"Skipping feature extraction for track {track_id} due to invalid bbox: {bbox_ltrb}")
                     current_feature = None

            if current_feature is None:
                # print(f"Skipping track {track_id} due to missing feature.")
                continue # Skip this track if we couldn't get a feature


            # --- ID mapping and Re-ID logic (using the feature obtained above) ---
            if track_id not in self.id_mapping:
                # Pass the feature obtained (either from DeepSORT or manually extracted)
                matched_id = self._re_identify_object(frame, bbox_ltrb, current_feature)

                if matched_id is not None:
                    self.id_mapping[track_id] = matched_id
                    # print(f"Re-identified object with ID {matched_id}") # Optional logging
                else:
                    self.id_mapping[track_id] = self.next_id
                    self.next_id += 1

            consistent_id = self.id_mapping[track_id]
            current_active_ids.add(consistent_id)

            # Add position to history
            center_x = (bbox_ltrb[0] + bbox_ltrb[2]) / 2
            center_y = (bbox_ltrb[1] + bbox_ltrb[3]) / 2
            self.track_history[consistent_id].append((center_x, center_y))

            # Update feature gallery and last seen frame using the obtained feature
            # Check feature dimension consistency before adding
            if current_feature.shape[0] == self.feature_dim:
                if consistent_id not in self.feature_gallery:
                    self.feature_gallery[consistent_id] = deque(maxlen=self.gallery_size)
                # Only append valid features
                if not np.all(current_feature == 0):
                   self.feature_gallery[consistent_id].append(current_feature)
            # else:
            #      print(f"Warning: Feature dimension mismatch for track {consistent_id}. Expected {self.feature_dim}, got {current_feature.shape[0]}. Not adding to gallery.")


            self.last_seen_frame[consistent_id] = self.frame_count

            if consistent_id in self.inactive_ids:
                self.inactive_ids.remove(consistent_id)

            # Retrieve class_id associated with the track by DeepSORT
            # The attribute name might be `det_class`, `cls`, etc. Check library source/docs if needed.
            # Default to 0 or a placeholder if not available.
            track_class_id = track_object.get_det_class() if hasattr(track_object, 'get_det_class') else (track_object.det_class if hasattr(track_object, 'det_class') else 0)


            current_tracks.append([*bbox_ltrb, consistent_id, track_class_id]) # Use LTRB and consistent ID

        # Run periodic re-identification for long-term tracking
        if self.frame_count % self.re_id_interval == 0:
            self._perform_offline_reid(frame) # Pass frame if needed by re-id logic, otherwise remove

        # Update inactive IDs
        # Make a copy of keys to avoid modifying dict while iterating
        all_known_ids = list(self.last_seen_frame.keys())
        for consistent_id in all_known_ids:
            if consistent_id not in current_active_ids:
                # Only mark as inactive if it was previously active or seen recently
                 if consistent_id not in self.inactive_ids:
                    # Add a grace period? E.g., if last seen > N frames ago.
                    # For now, mark inactive immediately if not currently seen.
                    self.inactive_ids.add(consistent_id)
                    # print(f"Object with ID {consistent_id} marked as potentially inactive.") # Optional logging

        # Update motion predictions for temporal association refinement
        self._update_motion_predictions()

        return current_tracks

    # --- Make sure _re_identify_object uses the features correctly ---
    def _re_identify_object(self, frame, bbox, current_features):
        """
        Try to re-identify an object with any of the inactive IDs
        Args:
            frame: Current video frame (may not be needed if only using features)
            bbox: Bounding box as [x1, y1, x2, y2] (may not be needed if only using features)
            current_features: Features of the current detection (numpy array)
        Returns:
            Matched ID or None
        """
        # Ensure current_features is a valid numpy array and not all zeros
        if not isinstance(current_features, np.ndarray) or np.all(current_features == 0) or len(self.inactive_ids) == 0:
            return None

        best_match_id = None
        # Use a lower threshold for re-id matching (higher similarity needed)
        best_match_score = 0.6 # Max allowed cosine distance for re-id (adjust as needed)

        for inactive_id in self.inactive_ids:
            if inactive_id not in self.feature_gallery:
                continue

            gallery_features = self.feature_gallery[inactive_id]
            if not gallery_features: # Check if deque is empty
                continue

            # Calculate feature similarity (cosine distance)
            # Ensure gallery features are also valid numpy arrays
            valid_gallery_features = [f for f in gallery_features if isinstance(f, np.ndarray) and f.shape == current_features.shape]
            if not valid_gallery_features:
                continue

            # Calculate distances only against valid features
            distances = nn_matching.distance_metric('cosine', best_match_score, None).distance(
                 current_features.reshape(1, -1), # Reshape current features
                 np.asarray(valid_gallery_features) # Convert gallery features to numpy array
            ) # This returns a (1, N) matrix of distances

            if distances.size == 0: # Handle case where no valid gallery features matched shape
                 continue

            min_distance = np.min(distances)

            # Check if this is the best match so far and below the threshold
            if min_distance < best_match_score:
                 # If considering multiple matches, store them and decide later
                 # For now, take the first one that's good enough (greedy)
                 # Or find the absolute best:
                 # if min_distance < best_match_score: # Compare with current best threshold
                 #    best_match_score = min_distance
                 #    best_match_id = inactive_id

                 # Let's take the first sufficiently good match for simplicity now
                 best_match_id = inactive_id
                 # print(f"Potential Re-ID match: current detection with inactive ID {inactive_id}, distance: {min_distance:.4f}")
                 break # Found a good enough match

        # Optional: Add motion prediction check as a secondary factor if no appearance match found
        # (Your original code had this, keep it if desired, but appearance should be primary)
        if best_match_id is None:
             # Check motion only if appearance failed
             center_x = (bbox[0] + bbox[2]) / 2
             center_y = (bbox[1] + bbox[3]) / 2
             motion_match_id = None
             min_motion_dist = 100 # Pixel threshold for motion matching

             for inactive_id in self.inactive_ids:
                  if inactive_id in self.kalman_predictions:
                       pred_x, pred_y = self.kalman_predictions[inactive_id]
                       distance = np.sqrt((pred_x - center_x)**2 + (pred_y - center_y)**2)
                       if distance < min_motion_dist:
                            # print(f"Potential motion match: current detection with inactive ID {inactive_id}, distance: {distance:.2f} pixels")
                            # This is a weaker match, maybe only use if appearance fails completely?
                            # For now, let's not override appearance match result.
                            # motion_match_id = inactive_id # Store potential motion match
                            pass # Don't assign yet, prioritize appearance


        return best_match_id # Return appearance match primarily


    # --- Make sure _perform_offline_reid uses features correctly ---
    def _perform_offline_reid(self, frame):
        """
        Perform offline re-identification to merge IDs potentially belonging to the same object.
        This focuses on merging an active track with a recently inactive one if they look similar.
        Args:
            frame: Current video frame (may not be needed)
        """
        active_ids = set(self.id_mapping.values()) - self.inactive_ids
        inactive_ids_list = list(self.inactive_ids)

        if not active_ids or not inactive_ids_list:
            return

        # Compare each active ID with each inactive ID
        ids1 = list(active_ids)
        ids2 = inactive_ids_list
        merge_candidates = []

        for i, id1 in enumerate(ids1): # Active IDs
             features1 = self.feature_gallery.get(id1)
             if not features1: continue
             # Get the most recent feature(s) for the active track
             recent_features1 = [f for f in list(features1)[-5:] if isinstance(f, np.ndarray)] # Take last 5 valid features
             if not recent_features1: continue

             for j, id2 in enumerate(ids2): # Inactive IDs
                 if id1 == id2: continue # Should not happen based on sets, but good check

                 features2 = self.feature_gallery.get(id2)
                 if not features2: continue
                 # Get representative features for the inactive track
                 gallery_features2 = [f for f in features2 if isinstance(f, np.ndarray)] # Take all valid features
                 if not gallery_features2: continue

                 # Ensure feature dimensions match before calculating distance
                 if recent_features1[0].shape != gallery_features2[0].shape:
                      continue

                 # Calculate cosine distance between the recent features of active track
                 # and all features of the inactive track.
                 distances = nn_matching.distance_metric('cosine', 1.0, None).distance(
                     np.asarray(recent_features1),
                     np.asarray(gallery_features2)
                 ) # Returns (N_active, M_inactive) matrix

                 if distances.size == 0: continue

                 # Use min distance as a measure of similarity potential
                 min_distance = np.min(distances)

                 # Use a strict threshold for merging (e.g., 0.3 means high similarity)
                 merge_threshold = 0.3
                 if min_distance < merge_threshold:
                     # Store potential merge pair (active, inactive, score)
                     merge_candidates.append((id1, id2, min_distance))
                     # print(f"Offline ReID Candidate: Merge inactive {id2} into active {id1}? Dist: {min_distance:.4f}")


        # Resolve merge candidates (e.g., using Hungarian algorithm or simply best match)
        # Simple greedy approach: Sort by distance and merge non-conflicting pairs
        merge_candidates.sort(key=lambda x: x[2]) # Sort by distance (ascending)
        merged_inactive = set()
        final_merges = {} # {inactive_id_to_remove: active_id_to_keep}

        for active_id, inactive_id, score in merge_candidates:
            if inactive_id not in merged_inactive and active_id not in final_merges.values():
                 # Ensure the inactive track hasn't been seen *too* long ago? Optional.
                 # frames_missing = self.frame_count - self.last_seen_frame.get(inactive_id, self.frame_count)
                 # if frames_missing > 100: continue # Example: Don't merge very old tracks

                 print(f"Offline ReID: Merging inactive ID {inactive_id} into active ID {active_id} (distance: {score:.4f})")
                 final_merges[inactive_id] = active_id
                 merged_inactive.add(inactive_id)

        # Apply the merges
        for remove_id, keep_id in final_merges.items():
            # Merge feature galleries
            if remove_id in self.feature_gallery:
                features_to_add = list(self.feature_gallery[remove_id])
                for feature in features_to_add:
                     if isinstance(feature, np.ndarray) and feature.shape == self.feature_gallery[keep_id][0].shape:
                         self.feature_gallery[keep_id].append(feature) # Add features to the kept ID

                # Update last seen frame to the most recent one
                self.last_seen_frame[keep_id] = max(
                    self.last_seen_frame.get(keep_id, 0),
                    self.last_seen_frame.get(remove_id, 0)
                )

                # Remove the merged ID data
                del self.feature_gallery[remove_id]
                if remove_id in self.last_seen_frame: del self.last_seen_frame[remove_id]
                if remove_id in self.inactive_ids: self.inactive_ids.remove(remove_id)
                if remove_id in self.track_history: del self.track_history[remove_id]
                if remove_id in self.kalman_predictions: del self.kalman_predictions[remove_id]


            # Update ID mapping for any DeepSORT tracks still mapped to the removed ID
            # This handles cases where a track reappears just before the merge happens
            for track_id, consistent_id in list(self.id_mapping.items()):
                if consistent_id == remove_id:
                    self.id_mapping[track_id] = keep_id
    
    def _update_motion_predictions(self):
        """
        Update motion predictions for all tracks
        This helps with re-identifying objects after they reappear
        """
        for consistent_id, positions in self.track_history.items():
            if len(positions) < 2:
                continue
            
            # Get the last few positions
            recent_positions = list(positions)[-5:]
            
            if len(recent_positions) < 2:
                continue
            
            # Calculate velocity based on recent positions
            velocities = []
            for i in range(1, len(recent_positions)):
                prev_x, prev_y = recent_positions[i-1]
                curr_x, curr_y = recent_positions[i]
                velocities.append((curr_x - prev_x, curr_y - prev_y))
            
            # Average velocity
            avg_vx = np.mean([v[0] for v in velocities])
            avg_vy = np.mean([v[1] for v in velocities])
            
            # Predict next position
            last_x, last_y = recent_positions[-1]
            pred_x = last_x + avg_vx
            pred_y = last_y + avg_vy
            
            # Store prediction
            self.kalman_predictions[consistent_id] = (pred_x, pred_y)


# Example integration with YOLO object detector
class YOLODetector:
    def __init__(self, model_path=None, conf_threshold=0.25, classes=None, device='cuda'):
        self.device = device
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold
            classes: List of classes to detect
        """
        try:
            from ultralytics import YOLO
            
            # Change the default model to YOLOv11
            if model_path is None:
                self.model = YOLO("yolo11s.pt").to(self.device)  # Use YOLOv11 nano model with device
            else:
                self.model = YOLO(model_path).to(self.device)
                
            self.using_ultralytics = True
            print(f"Using YOLOv11 from ultralytics")
            
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
        self.classes = classes  # None means detect all classes
    
    def detect(self, frame):
        """
        Detect objects in the frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections as [x1, y1, x2, y2, confidence, class_id]
        """
        if self.using_ultralytics:
            # Use YOLO from ultralytics
            results = self.model(frame, device=self.device)  # Explicitly set device
            
            # Process detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = box.cls[0].item()
                    
                    # Skip if conf is below threshold or class is not in classes
                    if conf < self.conf_threshold:
                        continue
                    if self.classes is not None and int(cls) not in self.classes:
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
            
            # Run forward pass
            outputs = self.model.forward(out_layer_names)
            
            # Process detections
            detections = []
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.conf_threshold:
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
                        
                        # Skip if class is not in classes
                        if self.classes is not None and class_id not in self.classes:
                            continue
                        
                        detections.append([x1, y1, x2, y2, confidence, class_id])
            
            return detections

def resize_for_display(image, max_width=1280, max_height=720):
    """
    Resize image to fit within specified dimensions while preserving aspect ratio
    
    Args:
        image: Input image
        max_width: Maximum display width
        max_height: Maximum display height
        
    Returns:
        Resized image for display
    """
    h, w = image.shape[:2]
    
    # Calculate scale factor to fit within max dimensions
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    # Only resize if image is larger than max dimensions
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    # Return original if already smaller than max dimensions
    return image

def main():
    """
    Main function to demonstrate the hybrid tracker
    """
    # Initialize video capture
    video_path = "video.mp4"  # Change to your video path
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist. Using webcam instead.")
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: # Handle case where FPS is not available (e.g., some webcams)
            print("Warning: Could not determine video FPS. Assuming 30 FPS for max_age calculation.")
            fps = 30
    except Exception:
        print("Warning: Error getting video FPS. Assuming 30 FPS for max_age calculation.")
        fps = 30
    
    print(f"Video FPS: {fps:.2f}")
     
    # Initialize object detector
    try:
        detector = YOLODetector(conf_threshold=0.3, classes=[0], device='cuda')
        # Option 2: If you have a specific YOLOv11 model file, specify it
        # detector = YOLODetector(model_path="path/to/yolov11n.pt", conf_threshold=0.3, classes=[0])
    except Exception as e:
        print(f"Error initializing YOLO detector: {e}")
        print("Falling back to mock detector")
        
        # Mock detector as fallback
        def mock_detector(frame):
            """
            Mock object detector that simulates detections
            """
            height, width = frame.shape[:2]
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            detections = []
            
            # Create a detection that stays in the frame
            if 0 <= frame_count < 100:
                # Object 1: Moves from left to right
                x = int(width * (0.1 + 0.8 * (frame_count % 100) / 100))
                y = int(height * 0.3)
                w, h = 50, 50
                detections.append([x, y, x+w, y+h, 0.9, 0])
            
            # Create a detection that leaves and re-enters the frame
            if 0 <= frame_count < 30: 
                # First appearance
                x = int(width * 0.7)
                y = int(height * (0.2 + 0.2 * (frame_count % 30) / 30))
                w, h = 60, 60
                detections.append([x, y, x+w, y+h, 0.85, 1])
            elif 70 <= frame_count < 100:
                # Second appearance
                x = int(width * 0.7)
                y = int(height * (0.4 + 0.2 * ((frame_count - 70) % 30) / 30))
                w, h = 60, 60
                detections.append([x, y, x+w, y+h, 0.85, 1])
            
            # Create another detection that crosses paths with others
            if 20 <= frame_count < 80:
                # Object 3: Moves diagonally
                progress = (frame_count - 20) / 60.0
                x = int(width * (0.8 - 0.6 * progress))
                y = int(height * (0.2 + 0.6 * progress))
                w, h = 55, 55
                detections.append([x, y, x+w, y+h, 0.95, 2])
                
            return detections
        
        detector = type('', (), {'detect': mock_detector})()
    
     # Calculate max_age for 5 seconds
    target_occlusion_seconds = 5
    calculated_max_age = int(target_occlusion_seconds * fps)
    # Add a small buffer (e.g., 10-20% or a fixed amount)
    buffer_frames = int(0.1 * calculated_max_age) # 10% buffer
    final_max_age = calculated_max_age + buffer_frames
    # Ensure max_age is reasonably bounded if FPS is very low/high, e.g., min 60, max 500
    final_max_age = max(60, min(final_max_age, 500))


    print(f"Setting DeepSORT max_age to {final_max_age} frames for ~{target_occlusion_seconds}s occlusion at {fps:.2f} FPS.")
    
    tracker = HybridTracker(
        max_cosine_distance=0.6,  # Keep or slightly increase (e.g., 0.5) if needed
        nn_budget=1000,            # Keep or increase if memory allows and needed
        max_age=final_max_age,    # <--- Key change: Increased max_age
        min_confidence=0.3,
        re_id_interval=5,        # Adjust interval if needed (e.g., 30 frames)
        gallery_size=1000          # Keep or increase if needed
    )
    
    # Colors for visualization
    colors = {}
    
    # Initialize video writer if needed
    save_video = False
    video_writer = None
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get detections
        detections = detector.detect(frame)
        
        # Update tracker
        tracks = tracker.update(frame, detections)
        
        # Visualize tracks
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id = track
            
            # Assign a color to this ID
            if track_id not in colors:
                colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
            color = colors[track_id]
            
            # Draw bounding box - increased line width from 2 to 3
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
            # Draw ID - increased font scale from 0.5 to 0.9 and text position adjusted
            text = f"ID: {int(track_id)}"
            cv2.putText(frame, text, (int(x1), int(y1)-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Draw track trail - increased line width from 2 to 3
            if track_id in tracker.track_history:
                points = list(tracker.track_history[track_id])
                for i in range(1, len(points)):
                    cv2.line(frame, (int(points[i-1][0]), int(points[i-1][1])),
                             (int(points[i][0]), int(points[i][1])), color, 3)
        
        # Draw frame count - increased font size from 0.7 to 0.9
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Initialize video writer on first frame if saving
        if save_video and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter('output.avi', fourcc, 30.0, 
                                          (frame.shape[1], frame.shape[0]))
        
        # Write frame if saving
        if save_video and video_writer is not None:
            video_writer.write(frame)
        
        # Resize frame for display only (processing still uses original resolution)
        display_frame = resize_for_display(frame, max_width=1280, max_height=720)

        # Display the resized frame
        cv2.imshow("Hybrid Tracking", display_frame)

        # Add this line to make the window resizable by the user if needed
        cv2.namedWindow("Hybrid Tracking", cv2.WINDOW_NORMAL)
        
        # Break on 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Toggle saving
            save_video = not save_video
            print(f"Video saving: {'ON' if save_video else 'OFF'}")
    
    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()