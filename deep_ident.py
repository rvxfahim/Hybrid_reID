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
from torchvision.models import ResNet50_Weights
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
            self.model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
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
        
        # Extract features in a single forward pass
        with torch.no_grad():
            features_batch = self.model(batch)
            features_batch = features_batch.squeeze().cpu().numpy()
            
        # If only one crop, ensure we have correct dimensions
        if len(crops) == 1:
            features_batch = features_batch.reshape(1, -1)
            
        # Normalize feature vectors
        features_batch = features_batch / np.linalg.norm(features_batch, axis=1, keepdims=True)
        
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
    # Normalize on GPU if needed
    f1_norm = torch.norm(features1_tensor, dim=1, keepdim=True)
    f2_norm = torch.norm(features2_tensor, dim=1, keepdim=True)
    # Avoid division by zero
    f1_norm = torch.where(f1_norm == 0, torch.ones_like(f1_norm), f1_norm)
    f2_norm = torch.where(f2_norm == 0, torch.ones_like(f2_norm), f2_norm)
    features1_normalized = features1_tensor / f1_norm
    features2_normalized = features2_tensor / f2_norm
    # Calculate cosine similarity matrix: (a·b)/(|a|·|b|)
    similarity = torch.mm(features1_normalized, features2_normalized.t())
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

            self.last_seen_frame[consistent_id] = self.frame_count

            if consistent_id in self.inactive_ids:
                self.inactive_ids.remove(consistent_id)

            # Retrieve class_id associated with the track by DeepSORT
            # The attribute name might be `det_class`, `cls`, etc. Check library source/docs if needed.
            # Default to 0 or a placeholder if not available.
            track_class_id = track_object.get_det_class() if hasattr(track_object, 'get_det_class') else (track_object.det_class if hasattr(track_object, 'det_class') else 0)


            current_tracks.append([*bbox_ltrb, consistent_id, track_class_id]) # Use LTRB and consistent ID

        # Add batch update right after tracks are processed, before re-ID
        self.update_feature_galleries_batch(frame, current_tracks)
    
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
    
        # Check feature dimension consistency
        expected_dim = self.feature_extractor.feature_dim  # Usually 2048 for ResNet50
        if current_features.shape[-1] != expected_dim:
            print(f"Warning: Feature dimension mismatch. Expected {expected_dim}, got {current_features.shape[-1]}.")
            # Extract features ourselves instead of using potentially incompatible ones
            current_features = self.feature_extractor.extract_features_batch(frame, [bbox])[0]
            if np.all(current_features == 0):
                return None
        
        best_match_id = None
        # Use a lower threshold for re-id matching (higher similarity needed)
        best_match_score = 0.6 # Max allowed cosine distance for re-id (adjust as needed)
        
        all_gallery_features = []
        gallery_id_map = []
        
        for inactive_id in self.inactive_ids:
            if inactive_id in self.feature_gallery:
                gallery = self.feature_gallery[inactive_id]
                # Add dimension check to filter incompatible features
                valid_features = [f for f in gallery if isinstance(f, np.ndarray) and f.shape[-1] == expected_dim]
                
                if valid_features:
                    all_gallery_features.extend(valid_features)
                    gallery_id_map.extend([inactive_id] * len(valid_features))
        
        if not all_gallery_features:
            return None
            
        # Compute distances in one GPU operation
        distances = compute_cosine_distance_gpu(
            current_features.reshape(1, -1),
            np.array(all_gallery_features),
            threshold=0.6
        )
        
        if distances.size == 0:
            return None
            
        # Find best match
        min_idx = np.argmin(distances[0])
        min_distance = distances[0][min_idx]
        
        if min_distance < 0.6:  # threshold
            best_match_id = gallery_id_map[min_idx]
            return best_match_id
        
        return None


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
        merge_threshold = 0.3 # Your similarity threshold (applied AFTER distance calculation)

        for id1 in valid_active_ids: # Iterate through ACTIVE track IDs that had features
            indices1 = active_feature_indices.get(id1) # Use .get for safety
            if not indices1: continue

            for id2 in valid_inactive_ids: # Iterate through INACTIVE track IDs that had features
                if id1 == id2: continue # Cannot merge with self

                indices2 = inactive_feature_indices.get(id2) # Use .get for safety
                if not indices2: continue

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
    
    @profile_function
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
        max_cosine_distance=0.3,  # Keep or slightly increase (e.g., 0.5) if needed
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
    
    # Stop profiling and display results
    profiler.disable()
    
    # Save detailed profiling stats to a file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = f"profiling_stats_{timestamp}.prof"
    profiler.dump_stats(stats_file)
    print(f"\nDetailed profiling stats saved to: {stats_file}")
    print("You can analyze this file with tools like snakeviz or using Python's pstats module")
    
    # Display basic profiling stats
    stats = pstats.Stats(profiler).sort_stats('cumtime')
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