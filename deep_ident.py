"""
Long-Term Object Tracking System with DeepSORT and Offline Re-identification

This implementation demonstrates a hybrid tracking approach that:
1. Uses DeepSORT for short-term, frame-to-frame tracking
2. Incorporates an offline re-identification system to maintain consistent IDs
3. Implements temporal association refinements for improved tracking across occlusions
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F
import time
import os
import signal
from collections import deque
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from PIL import Image
import cProfile
import pstats
import datetime
from functools import wraps
import sys

# Deep SORT and feature extraction imports
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection

# Global profiler
profiler = cProfile.Profile()

profiling_stats = {}

# Performance profiling decorator
def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize function stats in the global dictionary if not present
        if func.__qualname__ not in profiling_stats:
            profiling_stats[func.__qualname__] = {
                'call_count': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0
            }
            
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Update stats in the global dictionary
        stats = profiling_stats[func.__qualname__]
        stats['call_count'] += 1
        stats['total_time'] += execution_time
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        
        return result
    return wrapper

# Function to collect and display profiling stats
def display_profiling_stats():
    """Collects and displays profiling statistics for key functions"""
    if not profiling_stats:
        print("\nNo profiling statistics collected. Make sure functions are decorated with @profile_function.")
        return
        
    print("\n===== PERFORMANCE PROFILING RESULTS =====")
    print("{:<40} {:<10} {:<15} {:<15} {:<15}".format(
        "Function", "Calls", "Total Time (s)", "Avg Time (s)", "Max Time (s)"))
    print("="*95)
    
    # Sort functions by total execution time (descending)
    sorted_stats = sorted(profiling_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
    
    # Display results
    for func_name, stats in sorted_stats:
        avg_time = stats['total_time'] / stats['call_count'] if stats['call_count'] > 0 else 0
        print("{:<40} {:<10} {:<15.4f} {:<15.4f} {:<15.4f}".format(
            func_name, 
            stats['call_count'], 
            stats['total_time'], 
            avg_time,
            stats['max_time']
        ))
    
    print("\n===== BOTTLENECK ANALYSIS =====")
    
    # Find the bottleneck (function with highest total time)
    if sorted_stats:
        bottleneck = sorted_stats[0]
        func_name = bottleneck[0]
        stats = bottleneck[1]
        avg_time = stats['total_time'] / stats['call_count'] if stats['call_count'] > 0 else 0
        
        print(f"Main bottleneck: {func_name} - {stats['total_time']:.4f}s total ({avg_time:.4f}s avg)")
        
        # Function-specific recommendations
        if "YOLODetector.detect" in func_name:
            print("Recommendations for YOLO detection bottleneck:")
            print("1. Consider using a smaller/faster YOLO model (nano instead of small)")
            print("2. Reduce input resolution for detection")
            print("3. Implement frame skipping (detect every N frames)")
            print("4. Ensure you're using GPU acceleration properly")
        
        elif "FeatureExtractor.extract_features_batch" in func_name:
            print("Recommendations for feature extraction bottleneck:")
            print("1. Use a smaller feature extractor model")
            print("2. Reduce input resolution for the feature extractor")
            print("3. Extract features less frequently")
            print("4. Optimize batch processing to minimize GPU transfers")
        
        elif "HybridTracker.update" in func_name:
            print("Recommendations for tracker update bottleneck:")
            print("1. Simplify feature matching logic")
            print("2. Reduce frequency of re-identification")
            print("3. Optimize data structures for faster lookup")
        
        elif "HybridTracker._perform_offline_reid" in func_name:
            print("Recommendations for re-identification bottleneck:")
            print("1. Reduce re_id_interval (perform less frequently)")
            print("2. Reduce gallery size to compare fewer features")
            print("3. Implement more efficient feature matching")
    
    print("\n===== END OF PROFILING REPORT =====\n")

class FeatureExtractor:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the feature extractor with DINOv2 model

        Args:
            model_path: Path to a custom model (if None, use pre-trained DINOv2)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        print(f"Using device: {self.device}")

        # Initialize DINOv2 model
        try:
            if (model_path is None or not os.path.exists(model_path)):
                print("Loading pre-trained DINOv2 ViT-S/14 model")
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
                self.feature_dim = 768   # ViT-S/14 output dimension
            else:
                print(f"Loading custom model from {model_path}")
                self.model = torch.load(model_path, map_location=self.device)
                # Attempt to determine feature dim from model if not specified
                if hasattr(self.model, 'embed_dim'):
                    self.feature_dim = self.model.embed_dim
                else:
                    self.feature_dim = 384  # Default to ViT-S/14 dimension
        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")
            # Fall back to a simpler ResNet model if DINOv2 fails
            print("Falling back to ResNet50 model")
            self.model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove classification layer
            self.feature_dim = 2048  # ResNet50 feature dimension
        
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define image transforms for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # DINOv2 expects 224x224 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Feature extractor initialized with feature dimension: {self.feature_dim}")
        
    @profile_function
    def extract_features_batch(self, frame, bboxes):
        """
        Extract features for multiple bounding boxes in a single GPU operation
        
        Args:
            frame: Current video frame (BGR format)
            bboxes: List of bounding boxes as [x1, y1, x2, y2]
            
        Returns:
            Batch of feature vectors (numpy array)
        """
        if not bboxes:
            return []
            
        crops = []
        valid_indices = []
        
        # Prepare crops for all valid bounding boxes
        for i, bbox in enumerate(bboxes):
            try:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Ensure coordinates are within frame boundaries
                height, width = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop = frame[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(crop)
                crops.append(self.transform(img))
                valid_indices.append(i)
                
            except Exception as e:
                continue
                
        # Process all crops in a single batch
        if not crops:
            return [np.zeros(self.feature_dim, dtype=np.float32)] * len(bboxes)
            
        # Stack crops into a batch tensor
        batch = torch.stack(crops).to(self.device)
        
        # Extract features with DINOv2 in a single forward pass
        with torch.no_grad():
            features_batch = self.model(batch)
            features_batch = F.normalize(features_batch, p=2, dim=1).cpu().numpy()
            
        # If only one crop, ensure we have correct dimensions
        if len(crops) == 1:
            features_batch = features_batch.reshape(1, -1)
            
        # Create result array with zeros for invalid bboxes
        result = [np.zeros(self.feature_dim, dtype=np.float32)] * len(bboxes)
        for i, valid_idx in enumerate(valid_indices):
            result[valid_idx] = features_batch[i].astype(np.float32)
            
        return result

def compute_cosine_distance_gpu(features1, features2, threshold=1.0):
    """
    Compute cosine distance between two sets of features on GPU
    Args:
        features1: First set of feature vectors (numpy array)
        features2: Second set of feature vectors (numpy array)
        threshold: Maximum distance threshold
    Returns:
        Distance matrix (numpy array)
    """
    if len(features1) == 0 or len(features2) == 0:
        return np.array([])
        
    # Convert to PyTorch tensors and move to GPU
    features1_tensor = torch.tensor(features1, dtype=torch.float32).cuda()
    features2_tensor = torch.tensor(features2, dtype=torch.float32).cuda()
    
    # Ensure features are normalized (DINOv2 features should already be normalized)
    features1_norm = F.normalize(features1_tensor, p=2, dim=1)
    features2_norm = F.normalize(features2_tensor, p=2, dim=1)
    
    # Calculate cosine similarity matrix: (a·b)/(|a|·|b|)
    similarity = torch.mm(features1_norm, features2_norm.t())
    
    # Convert to distance: 1 - similarity
    distance = 1.0 - similarity
    
    # Apply threshold if needed
    if threshold < 1.0:
        distance = torch.clamp(distance, 0.0, threshold)
        
    # Return as numpy array
    return distance.cpu().numpy()
    
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
        
        # Add a flag to identify the primary object of interest
        self.primary_object_id = None  # Will be set to 1 after first frame
        self.primary_object_features = deque(maxlen=gallery_size)  # Store features of primary object only
        self.primary_object_last_seen = 0  # Last frame where primary object was seen
        self.primary_object_active = False  # Whether the primary object is currently being tracked
    
    def update_feature_galleries_batch(self, frame, tracks):
        """Update feature galleries for multiple tracks in a single GPU operation"""
        if not tracks:
            return

        # Extract bboxes for feature extraction
        bboxes = [track[:4] for track in tracks]  # [x1, y1, x2, y2]
        track_ids = [int(track[4]) for track in tracks]  # consistent_ids

        # Extract features in a batch
        features_batch = self.feature_extractor.extract_features_batch(frame, bboxes)

        # Update galleries with the new features
        for i, (track_id, feature) in enumerate(zip(track_ids, features_batch)):
            if np.all(feature == 0):
                continue  # Skip invalid features
            
            if track_id not in self.feature_gallery:
                self.feature_gallery[track_id] = deque(maxlen=self.gallery_size)

            self.feature_gallery[track_id].append(feature)
            self.last_seen_frame[track_id] = self.frame_count
        
    @profile_function
    def update(self, frame, detections):
        """Optimized update method focusing on primary object (ID1)"""
        self.frame_count += 1

        # Format detections for DeepSORT
        deepsort_detections = []

        for det in detections:
            if len(det) >= 6:
                bbox, confidence, class_id = det[:4], det[4], det[5]
                if confidence < self.min_confidence:
                    continue

                # Convert to [x1, y1, w, h] format for DeepSORT
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                # Extract features
                feature = self.feature_extractor.extract_features_batch(frame, [bbox])[0]

                # Create detection tuple in the format expected by deep_sort_realtime
                deepsort_detection = ([x1, y1, w, h], confidence, feature)
                deepsort_detections.append(deepsort_detection)

        # Update DeepSORT tracker with correctly formatted detections
        deepsort_tracks_returned = self.tracker.update_tracks(deepsort_detections, frame=frame)

        # Process DeepSORT tracks
        current_tracks = []
        primary_object_seen = False

        # Initialize primary_object_bbox if not already set
        if not hasattr(self, 'primary_object_bbox'):
            self.primary_object_bbox = None

        for track_object in deepsort_tracks_returned:
            if not track_object.is_confirmed() or track_object.time_since_update > 1:
                continue

            track_id = track_object.track_id  # DeepSORT temporary ID
            bbox_ltrb = track_object.to_ltrb()

            # Get feature from track (either from DeepSORT or extract it)
            current_feature = self._get_feature_from_track(frame, track_object, bbox_ltrb)
            if current_feature is None:
                continue

            # Handle ID mapping
            if track_id not in self.id_mapping:
                # First frame - assign ID1 to first detected object 
                if self.primary_object_id is None:
                    self.primary_object_id = 1
                    self.id_mapping[track_id] = self.primary_object_id
                    self.primary_object_active = True
                    print(f"Initialized primary object with ID {self.primary_object_id}")
                else:
                    # For subsequent new objects, check if this could be the primary object returning
                    if not self.primary_object_active and self.primary_object_features:
                        # Only try to re-identify against the primary object
                        if self._is_primary_object(current_feature, bbox_ltrb):  # Add bbox parameter here
                            self.id_mapping[track_id] = self.primary_object_id
                            self.primary_object_active = True
                            print(f"Re-identified primary object with ID {self.primary_object_id}")
                        else:
                            # Assign a new ID for other objects
                            self.id_mapping[track_id] = self.next_id
                            self.next_id += 1
                    else:
                        # Assign a new ID for other objects
                        self.id_mapping[track_id] = self.next_id
                        self.next_id += 1

            consistent_id = self.id_mapping[track_id]

            # Update track history and tracking info
            center_x = (bbox_ltrb[0] + bbox_ltrb[2]) / 2
            center_y = (bbox_ltrb[1] + bbox_ltrb[3]) / 2
            self.track_history[consistent_id].append((center_x, center_y))

            # For the primary object, update its features and perform additional IoU check
            if consistent_id == self.primary_object_id:
                # If we have a previous bbox for the primary object, check IoU to ensure consistency
                if self.primary_object_active and self.primary_object_bbox is not None:
                    iou = self._calculate_iou(bbox_ltrb, self.primary_object_bbox)
                    # If IoU is too low, either this is not the primary object or it moved very fast
                    # Only accept if the IoU is good enough OR we haven't seen the primary object for a while
                    frames_since_last_seen = self.frame_count - self.primary_object_last_seen
                    if iou < self.iou_threshold and frames_since_last_seen <= 5:  # Only check for recent frames
                        print(f"Warning: Rejecting assignment to primary object due to low IoU: {iou:.3f}")
                        # Create a new ID instead
                        new_id = self.next_id
                        self.next_id += 1
                        self.id_mapping[track_id] = new_id
                        consistent_id = new_id
                    else:
                        # Update primary object information
                        if not hasattr(self, 'primary_object_features'):
                            self.primary_object_features = deque(maxlen=self.gallery_size)
                        self.primary_object_features.append(current_feature)
                        self.primary_object_last_seen = self.frame_count
                        self.primary_object_active = True
                        primary_object_seen = True

                        # Store this bbox for the primary object
                        self.primary_object_bbox = bbox_ltrb
                else:
                    # No previous bbox, update primary object information
                    if not hasattr(self, 'primary_object_features'):
                        self.primary_object_features = deque(maxlen=self.gallery_size)
                    self.primary_object_features.append(current_feature)
                    self.primary_object_last_seen = self.frame_count
                    self.primary_object_active = True
                    primary_object_seen = True

                    # Store this bbox for the primary object
                    self.primary_object_bbox = bbox_ltrb

            # Get class ID from track
            track_class_id = track_object.get_det_class() if hasattr(track_object, 'get_det_class') else \
                            (track_object.det_class if hasattr(track_object, 'det_class') else 0)

            current_tracks.append([*bbox_ltrb, consistent_id, track_class_id])

        # Update primary object status if not seen in this frame
        if not primary_object_seen:
            self.primary_object_active = False

        # Only update motion predictions for the primary object
        if self.primary_object_id in self.track_history:
            self._update_primary_motion_prediction()

        # Update feature galleries
        self.update_feature_galleries_batch(frame, current_tracks)

        # Perform offline re-identification at regular intervals
        if self.frame_count % self.re_id_interval == 0:
            self._perform_offline_reid(frame)

        return current_tracks

    # --- Make sure _re_identify_object uses the features correctly ---
    def _re_identify_object(self, frame, bbox, current_features):
        """Re-identify object with additional spatial constraint"""
        if not isinstance(current_features, np.ndarray) or np.all(current_features == 0) or len(self.inactive_ids) == 0:
            return None
        
        # Get current bbox center
        current_center_x = (bbox[0] + bbox[2]) / 2
        current_center_y = (bbox[1] + bbox[3]) / 2
        
        best_match_id = None
        best_match_score = 0.6  # Threshold for feature distance
        max_spatial_distance = 200  # Maximum allowed spatial distance in pixels
        
        # Rest of your existing feature matching code...
        
        # Add spatial constraint check before returning the match
        if best_match_id is not None:
            # Get last known position of the matched ID
            if best_match_id in self.track_history and len(self.track_history[best_match_id]) > 0:
                last_pos = self.track_history[best_match_id][-1]
                spatial_dist = np.sqrt((last_pos[0] - current_center_x)**2 + 
                                      (last_pos[1] - current_center_y)**2)
                
                # Reject match if too far away
                if spatial_dist > max_spatial_distance:
                    print(f"Rejecting match with ID {best_match_id} due to large spatial distance: {spatial_dist:.1f}px")
                    return None
        
        return best_match_id

    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes
        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]
        Returns:
            IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas of both boxes
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    # --- Make sure _perform_offline_reid uses features correctly ---
    @profile_function
    def _perform_offline_reid(self, frame):
        """
        Perform offline re-identification using GPU-batched distance calculation.
        Merges an active track with a recently inactive one if they look similar.
        Args:
            frame: Current video frame (may not be needed directly here)
        """
        active_ids = set(self.id_mapping.values()) - self.inactive_ids
        inactive_ids_list = list(self.inactive_ids)

        if not active_ids or not inactive_ids_list:
            return

        # --- 1. Gather Features and Create Index Maps ---
        all_active_features = []
        active_feature_indices = {}  # Map: active_id -> [list of indices in all_active_features]

        all_inactive_features = []
        inactive_feature_indices = {} # Map: inactive_id -> [list of indices in all_inactive_features]

        valid_active_ids = []
        # Collect features for active IDs
        for id1 in active_ids:
            features1 = self.feature_gallery.get(id1)
            if not features1: continue
            recent_features1 = [f for f in list(features1)[-5:] if isinstance(f, np.ndarray)] # Take last 5 valid features
            if not recent_features1: continue

            start_idx = len(all_active_features)
            all_active_features.extend(recent_features1)
            end_idx = len(all_active_features)
            active_feature_indices[id1] = list(range(start_idx, end_idx))
            valid_active_ids.append(id1)

        valid_inactive_ids = []
        # Collect features for inactive IDs
        for id2 in inactive_ids_list:
            features2 = self.feature_gallery.get(id2)
            if not features2: continue
            gallery_features2 = [f for f in features2 if isinstance(f, np.ndarray)] # Take all valid features
            if not gallery_features2: continue

            start_idx = len(all_inactive_features)
            all_inactive_features.extend(gallery_features2)
            end_idx = len(all_inactive_features)
            inactive_feature_indices[id2] = list(range(start_idx, end_idx))
            valid_inactive_ids.append(id2)

        if not all_active_features or not all_inactive_features:
            # print("Offline ReID: No features to compare.")
            return # Nothing to compare

        # Convert lists to NumPy arrays just before GPU call
        # Check for shape consistency (assuming all features should have the same dim)
        try:
            active_features_np = np.asarray(all_active_features, dtype=np.float32)
            inactive_features_np = np.asarray(all_inactive_features, dtype=np.float32)
        except ValueError as e:
             print(f"Offline ReID Error: Could not create numpy arrays from features. Possible shape mismatch? Error: {e}")
             # Example: Check shapes if possible
             # if all_active_features: print(f"First active feature shape: {all_active_features[0].shape}")
             # if all_inactive_features: print(f"First inactive feature shape: {all_inactive_features[0].shape}")
             return # Cannot proceed

        if active_features_np.shape[0] == 0 or inactive_features_np.shape[0] == 0:
             print("Offline ReID: Feature arrays are empty after conversion.")
             return

        # --- 2. Batch Distance Calculation using GPU ---
        print(f"Offline ReID: Calculating distances between {active_features_np.shape[0]} active and {inactive_features_np.shape[0]} inactive features using GPU.")

        # <<< Integration Point >>>
        # Call your GPU function here. Pass threshold=1.0 consistent with nn_matching's usual max_distance.
        full_distance_matrix = compute_cosine_distance_gpu(
            active_features_np,
            inactive_features_np,
            threshold=1.0  # Clamp distances > 1.0 (low similarity)
        )

        # Check if GPU calculation failed (e.g., returned empty)
        if full_distance_matrix is None or full_distance_matrix.size == 0 or full_distance_matrix.shape != (active_features_np.shape[0], inactive_features_np.shape[0]):
            print("Offline ReID: GPU distance calculation failed or returned unexpected result. Skipping merge for this frame.")
            return # Abort merge if distance calculation failed

        print("Offline ReID: GPU distance calculation complete.")

        # --- 3. Extract Minimums and Build Merge Candidates ---
        merge_candidates = []
        merge_threshold = 0.2 # Your similarity threshold (applied AFTER distance calculation)

        for id1 in valid_active_ids: # Iterate through ACTIVE track IDs that had features
            indices1 = active_feature_indices.get(id1) # Use .get for safety
            if not indices1: continue

            # Get latest position of active track for spatial check
            active_track_pos = None
            if id1 in self.track_history and len(self.track_history[id1]) > 0:
                active_track_pos = self.track_history[id1][-1]  # Last position (x, y)

            for id2 in valid_inactive_ids: # Iterate through INACTIVE track IDs that had features
                if id1 == id2: continue # Cannot merge with self

                indices2 = inactive_feature_indices.get(id2) # Use .get for safety
                if not indices2: continue

                # Spatial check - compare last track positions (if available)
                if active_track_pos and id2 in self.track_history and len(self.track_history[id2]) > 0:
                    inactive_track_pos = self.track_history[id2][-1]  # Last position
                    
                    # Get bounding boxes if available
                    inactive_bbox = None
                    for track in self.track_history[id2][-1:]:
                        if isinstance(track, list) and len(track) >= 4:
                            inactive_bbox = track[:4]  # [x1, y1, x2, y2]
                            break
                    
                    active_bbox = None
                    for track in self.track_history[id1][-1:]:
                        if isinstance(track, list) and len(track) >= 4:
                            active_bbox = track[:4]  # [x1, y1, x2, y2]
                            break
                    
                    # If we have bounding boxes, calculate IoU
                    skip_this_pair = False
                    if inactive_bbox and active_bbox:
                        iou = self._calculate_iou(inactive_bbox, active_bbox)
                        # If IoU is too low, skip this pair (objects are spatially too different)
                        if iou < self.iou_threshold:
                            skip_this_pair = True
                    
                    if skip_this_pair:
                        continue

                # Efficiently select the sub-matrix corresponding to this pair of IDs
                try:
                    sub_matrix = full_distance_matrix[np.ix_(indices1, indices2)]
                except IndexError as e:
                    print(f"Offline ReID Error: Indexing failed for id1={id1}, id2={id2}. Indices1={indices1}, Indices2={indices2}, MatrixShape={full_distance_matrix.shape}. Error: {e}")
                    continue # Skip this pair

                if sub_matrix.size == 0: continue # No valid feature pairs between these IDs

                # Find the minimum distance within this specific ID-pair's features
                min_distance = np.min(sub_matrix)

                # Check against the merge threshold
                if min_distance < merge_threshold:
                    merge_candidates.append((id1, id2, min_distance))
                    # print(f"Offline ReID Candidate: Merge inactive {id2} into active {id1}? Dist: {min_distance:.4f}")


        # --- 4. Resolve Merge Candidates (Same as your original code) ---
        merge_candidates.sort(key=lambda x: x[2]) # Sort by distance (ascending)
        merged_inactive = set()
        final_merges = {} # {inactive_id_to_remove: active_id_to_keep}

        for active_id, inactive_id, score in merge_candidates:
            if inactive_id not in merged_inactive and active_id not in final_merges.values():
                 # Optional frame gap check (uncomment if needed)
                 # frames_missing = self.frame_count - self.last_seen_frame.get(inactive_id, self.frame_count)
                 # max_allowed_missing = 100
                 # if frames_missing > max_allowed_missing: continue

                 print(f"Offline ReID: Merging inactive ID {inactive_id} into active ID {active_id} (distance: {score:.4f})")
                 final_merges[inactive_id] = active_id
                 merged_inactive.add(inactive_id)


        # --- 5. Apply the Merges (Same as your original code, with minor robustness additions) ---
        if final_merges:
            print(f"Offline ReID: Applying {len(final_merges)} merges.")
        for remove_id, keep_id in final_merges.items():
            # Ensure both IDs still exist in relevant structures before proceeding
            if remove_id not in self.feature_gallery or keep_id not in self.feature_gallery:
                 print(f"Offline ReID Warning: Cannot merge {remove_id} into {keep_id}. One or both galleries missing (perhaps already merged?).")
                 continue

            # Merge feature galleries
            features_to_add = self.feature_gallery[remove_id] # deque
            target_gallery = self.feature_gallery[keep_id] # deque

            target_feature_shape = None
            for f in target_gallery:
                 if isinstance(f, np.ndarray):
                     target_feature_shape = f.shape
                     break
            if target_feature_shape is None and features_to_add:
                 # If target is empty, try to get shape from source
                 for f in features_to_add:
                      if isinstance(f, np.ndarray):
                           target_feature_shape = f.shape
                           break

            added_count = 0
            for feature in list(features_to_add): # Iterate over a copy
                 if isinstance(feature, np.ndarray) and (target_feature_shape is None or feature.shape == target_feature_shape):
                     target_gallery.append(feature)
                     added_count += 1
            # print(f"Merged {added_count} features from {remove_id} to {keep_id}")

            # Update last seen frame
            self.last_seen_frame[keep_id] = max(
                self.last_seen_frame.get(keep_id, 0),
                self.last_seen_frame.get(remove_id, 0)
            )

            # Remove the merged ID data cleanly
            del self.feature_gallery[remove_id]
            if remove_id in self.last_seen_frame: del self.last_seen_frame[remove_id]
            # Use discard() for sets, it doesn't raise an error if the element is not present
            self.inactive_ids.discard(remove_id)
            if remove_id in self.track_history: del self.track_history[remove_id]
            if remove_id in self.kalman_predictions: del self.kalman_predictions[remove_id]

            # Update ID mapping
            updated_mapping_count = 0
            for track_id, consistent_id in list(self.id_mapping.items()):
                if consistent_id == remove_id:
                    self.id_mapping[track_id] = keep_id
                    updated_mapping_count += 1
            # if updated_mapping_count > 0: print(f"Updated id_mapping for {updated_mapping_count} track(s) from {remove_id} to {keep_id}")

            # Ensure kept ID is not marked inactive
            self.inactive_ids.discard(keep_id)
            # print(f"Offline ReID: Ensured {keep_id} is not in inactive_ids.")

        # print("Offline ReID: Finished.")
    
    def _update_motion_predictions(self):
        """
        Update motion predictions for all tracks
        This helps with re-identifying objects after they reappear
        """
        # Group all position histories
        all_tracks = []
        track_ids = []

        for consistent_id, positions in self.track_history.items():
            if len(positions) >= 2:
                all_tracks.append(list(positions)[-5:])  # Take last 5 positions
                track_ids.append(consistent_id)

        if not all_tracks:
            return

        # Convert to tensors
        track_tensors = [torch.tensor(track, dtype=torch.float32).cuda() for track in all_tracks]

        # Process each track in parallel using GPU
        for i, (track_tensor, consistent_id) in enumerate(zip(track_tensors, track_ids)):
            if len(track_tensor) < 2:
                continue

            # Calculate velocity using tensor operations
            velocity = track_tensor[1:] - track_tensor[:-1]
            avg_velocity = torch.mean(velocity, dim=0)

            # Predict next position
            last_pos = track_tensor[-1]
            pred_pos = last_pos + avg_velocity

            # Store prediction
            self.kalman_predictions[consistent_id] = (
                pred_pos[0].item(), 
                pred_pos[1].item()
            )

    def _is_primary_object(self, current_feature, current_bbox=None):
        """
        Check if the current feature matches the primary object

        Args:
            current_feature: Feature vector to check
            current_bbox: Current bounding box to check spatial consistency

        Returns:
            True if this is likely the primary object, False otherwise
        """
        if not self.primary_object_features:
            return False

        # Add IoU check if both bboxes are available
        if current_bbox is not None and self.primary_object_bbox is not None:
            iou = self._calculate_iou(current_bbox, self.primary_object_bbox)

            # If IoU is extremely low and not much time has passed, reject as primary object
            frames_since_last_seen = self.frame_count - self.primary_object_last_seen
            if iou < 0.1 and frames_since_last_seen < 30:  # Adjust thresholds as needed
                print(f"Rejecting primary object candidate due to very low IoU: {iou:.3f}")
                return False

        # Convert features to tensors
        features1 = [current_feature]
        features2 = list(self.primary_object_features)

        # Use GPU for faster computation
        features1_tensor = torch.tensor(features1, dtype=torch.float32).cuda()
        features2_tensor = torch.tensor(features2, dtype=torch.float32).cuda()

        # Normalize features
        features1_norm = F.normalize(features1_tensor, p=2, dim=1)
        features2_norm = F.normalize(features2_tensor, p=2, dim=1)

        # Calculate similarity
        similarity = torch.mm(features1_norm, features2_norm.t())

        # Get max similarity
        max_sim = torch.max(similarity).item()
        distance = 1.0 - max_sim

        # Define a stricter threshold for primary object
        reid_threshold = 0.2  # Lower threshold = more strict matching

        # Check spatial constraint if we have motion prediction
        if self.primary_object_id in self.kalman_predictions and self.primary_object_bbox is not None:
            pred_x, pred_y = self.kalman_predictions[self.primary_object_id]

            # Get current bbox center
            current_x = (current_bbox[0] + current_bbox[2]) / 2
            current_y = (current_bbox[1] + current_bbox[3]) / 2

            # Calculate spatial distance
            spatial_dist = np.sqrt((pred_x - current_x)**2 + (pred_y - current_y)**2)

            # If too far away, increase matching threshold
            if spatial_dist > 200:  # pixels
                reid_threshold *= 0.75  # Make matching harder if spatially distant

        return distance < reid_threshold

    def _update_primary_motion_prediction(self):
        """Update motion prediction only for the primary object"""
        if self.primary_object_id not in self.track_history:
            return
            
        positions = self.track_history[self.primary_object_id]
        if len(positions) < 2:
            return
            
        # Take last 5 positions
        recent_positions = list(positions)[-5:]
        
        # Convert to tensor
        track_tensor = torch.tensor(recent_positions, dtype=torch.float32).cuda()
        
        # Calculate velocity
        velocity = track_tensor[1:] - track_tensor[:-1]
        avg_velocity = torch.mean(velocity, dim=0)
        
        # Predict next position
        last_pos = track_tensor[-1]
        pred_pos = last_pos + avg_velocity
        
        # Store prediction
        self.kalman_predictions[self.primary_object_id] = (
            pred_pos[0].item(),
            pred_pos[1].item()
        )

    def _get_feature_from_track(self, frame, track_object, bbox_ltrb):
        """Extract features from track object"""
        # Try to get feature from DeepSORT first
        if track_object.features and isinstance(track_object.features[-1], np.ndarray):
            return track_object.features[-1]
        
        # If not available, extract manually
        if bbox_ltrb[2] > bbox_ltrb[0] and bbox_ltrb[3] > bbox_ltrb[1]:
            # Extract single feature
            x1, y1, x2, y2 = map(int, bbox_ltrb)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
                
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(crop)
            img_tensor = self.feature_extractor.transform(img).unsqueeze(0).to(self.feature_extractor.device)
            
            with torch.no_grad():
                feature = self.feature_extractor.model(img_tensor)
                feature = F.normalize(feature, p=2, dim=1).cpu().numpy()[0]
                
            return feature
        
        return None


class YOLODetector:
    def __init__(self, model_path=None, conf_threshold=0.25, device='cuda'):
        self.device = device
        """
        Initialize YOLO detector specialized for human detection
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold
            device: Computing device ('cuda' or 'cpu')
        """
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
        if self.using_ultralytics:
            # Use YOLO from ultralytics - only detect humans (class 0)
            # Add class filtering to the model prediction to reduce computation
            results = self.model(frame, device=self.device, classes=[self.person_class_id])
            
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
            
            # Run forward pass
            outputs = self.model.forward(out_layer_names)
            
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
    # Initialize the profiler at the top of main
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Initialize video capture
    video_path = "left_view.mp4"  # Change to your video path
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
        detector = YOLODetector(conf_threshold=0.3, device='cuda')
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
    target_occlusion_seconds = 10
    calculated_max_age = int(target_occlusion_seconds * fps)
    # Add a small buffer (e.g., 10-20% or a fixed amount)
    buffer_frames = int(0.1 * calculated_max_age) # 10% buffer
    final_max_age = calculated_max_age + buffer_frames
    # Ensure max_age is reasonably bounded if FPS is very low/high, e.g., min 60, max 500
    final_max_age = max(60, min(final_max_age, 500))


    print(f"Setting DeepSORT max_age to {final_max_age} frames for ~{target_occlusion_seconds}s occlusion at {fps:.2f} FPS.")
    
    tracker = HybridTracker(
        max_cosine_distance=0.15,      # Reduced threshold for DINOv2 features
        nn_budget=2000,                # Keep or increase if memory allows
        max_age=final_max_age,         # Keep dynamically calculated max_age
        min_confidence=0.3,
        re_id_interval=1,              # Set to run re-ID frequently since DINOv2 is powerful
        gallery_size=3000,             # Keep or increase if needed
        iou_threshold=0.1              # You might need to adjust this based on testing
    )
    
    # Define color for ID1 (primary object)
    id1_color = (0, 255, 0)  # Green color for primary object
    
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
        
        # Create a copy for visualization
        display_frame = frame.copy()
        
        # Visualize tracks - only for ID1
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id = track
            
            # Only display ID 1 (primary object)
            if track_id == 1:
                # Draw bounding box with increased line width
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), id1_color, 3)
                
                # Draw ID
                text = f"ID: {int(track_id)}"
                cv2.putText(display_frame, text, (int(x1), int(y1)-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, id1_color, 3)
                
                # Draw track trail
                if track_id in tracker.track_history:
                    points = list(tracker.track_history[track_id])
                    for i in range(1, len(points)):
                        cv2.line(display_frame, (int(points[i-1][0]), int(points[i-1][1])),
                                 (int(points[i][0]), int(points[i][1])), id1_color, 3)
        
        # Draw frame count
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display status about primary object tracking
        primary_status = "Primary Object: "
        if tracker.primary_object_active:
            primary_status += "TRACKING"
            cv2.putText(display_frame, primary_status, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            primary_status += "LOST"
            cv2.putText(display_frame, primary_status, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Initialize video writer on first frame if saving
        if save_video and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter('output.avi', fourcc, 30.0, 
                                          (frame.shape[1], frame.shape[0]))
        
        # Write frame if saving
        if save_video and video_writer is not None:
            video_writer.write(display_frame)
        
        # Resize frame for display only (processing still uses original resolution)
        display_frame = resize_for_display(display_frame, max_width=1280, max_height=720)

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
    
    # Stop profiling and display results
    profiler.disable()
    
    # Save detailed profiling stats to a file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = f"profiling_stats_{timestamp}.prof"
    profiler.dump_stats(stats_file)
    print(f"\nDetailed profiling stats saved to: {stats_file}")
    print("You can analyze this file with tools like snakeviz or using Python's pstats module")
    
    # Create pstats.Stats object from the file, not directly from the profiler
    stats = pstats.Stats(stats_file).sort_stats('cumtime')
    print("\n===== TOP 20 TIME-CONSUMING FUNCTIONS =====")
    stats.print_stats(20)
    
    # Display our custom performance metrics
    display_profiling_stats()

# Add this new function specifically for resizing input frames while preserving aspect ratio
def resize_with_aspect_ratio(image, width=None, height=None):
    """
    Resize image to target width or height while preserving aspect ratio
    
    Args:
        image: Input image
        width: Target width (if None, calculate from height)
        height: Target height (if None, calculate from width)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Both width and height are None, return original image
    if width is None and height is None:
        return image
    
    # Both are specified, determine which one to follow based on aspect ratio
    if width is not None and height is not None:
        # Calculate target aspect ratio
        target_ratio = width / height
        # Calculate current aspect ratio
        current_ratio = w / h
        
        # If current ratio is wider than target, use width as the limiting factor
        if current_ratio > target_ratio:
            height = None
        # Otherwise use height as the limiting factor
        else:
            width = None
    
    # Calculate new dimensions
    if width is None:
        # Target height is specified, calculate width to maintain aspect ratio
        r = height / h
        new_width = int(w * r)
        new_height = height
    else:
        # Target width is specified, calculate height to maintain aspect ratio
        r = width / w
        new_width = width
        new_height = int(h * r)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Exiting gracefully...")
    # Use the global profiling stats instead
    display_profiling_stats()
    exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    main()