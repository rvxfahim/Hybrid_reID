"""
Hybrid tracking system combining DeepSORT with offline re-identification
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque, defaultdict
from PIL import Image
import cv2

from profiling import profile_function
from feature_extractor import FeatureExtractor, compute_cosine_distance_gpu
from numba_optimizations import NUMBA_AVAILABLE, calculate_iou, calculate_iou_numba, calculate_spatial_distance_numba

# Import Deep SORT components
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection

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
        self.primary_object_bbox = None  # Keep track of the primary object's bounding box
    
    @profile_function
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

    @profile_function
    def _re_identify_object(self, frame, bbox, current_features):
        """Re-identify object with additional spatial constraint"""
        if not isinstance(current_features, np.ndarray) or np.all(current_features == 0) or len(self.inactive_ids) == 0:
            return None
        
        # Get current bbox center
        current_center_x = (bbox[0] + bbox[2]) / 2
        current_center_y = (bbox[1] + bbox[3]) / 2
        current_center = np.array([current_center_x, current_center_y])
        
        best_match_id = None
        best_match_score = 0.6  # Threshold for feature distance
        max_spatial_distance = 200  # Maximum allowed spatial distance in pixels
        
        # Get all inactive IDs with valid feature galleries
        valid_inactive_ids = []
        all_inactive_features = []
        id_to_feature_indices = {}
        
        # Collect feature galleries for inactive IDs
        for inactive_id in self.inactive_ids:
            if inactive_id in self.feature_gallery and self.feature_gallery[inactive_id]:
                valid_features = []
                for feature in self.feature_gallery[inactive_id]:
                    if isinstance(feature, np.ndarray) and not np.all(feature == 0):
                        valid_features.append(feature)
                
                if valid_features:
                    start_idx = len(all_inactive_features)
                    all_inactive_features.extend(valid_features)
                    id_to_feature_indices[inactive_id] = (start_idx, len(all_inactive_features))
                    valid_inactive_ids.append(inactive_id)
        
        if not valid_inactive_ids:
            return None
        
        # Convert to numpy arrays for batch processing
        all_inactive_features_np = np.array(all_inactive_features)
        current_feature_np = current_features.reshape(1, -1)
        
        # Calculate distances between current feature and all inactive features
        distances = compute_cosine_distance_gpu(current_feature_np, all_inactive_features_np)
        
        # Find the best match by minimum distance per ID
        for inactive_id in valid_inactive_ids:
            start_idx, end_idx = id_to_feature_indices[inactive_id]
            id_distances = distances[0, start_idx:end_idx]
            min_distance = np.min(id_distances)
            
            if min_distance < best_match_score:
                # Check spatial constraint
                if inactive_id in self.track_history and len(self.track_history[inactive_id]) > 0:
                    last_pos = self.track_history[inactive_id][-1]
                    last_pos_array = np.array([last_pos[0], last_pos[1]])
                    
                    # Use Numba-accelerated function if available
                    if NUMBA_AVAILABLE:
                        spatial_dist = calculate_spatial_distance_numba(current_center, last_pos_array)
                    else:
                        spatial_dist = np.sqrt(np.sum((current_center - last_pos_array)**2))
                    
                    # Reject match if too far away
                    if spatial_dist > max_spatial_distance:
                        print(f"Rejecting match with ID {inactive_id} due to large spatial distance: {spatial_dist:.1f}px")
                        continue
                
                best_match_score = min_distance
                best_match_id = inactive_id
        
        return best_match_id

    @profile_function
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes
        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]
        Returns:
            IoU value
        """
        if NUMBA_AVAILABLE:
            return calculate_iou_numba(box1, box2)
        else:
            return calculate_iou(box1, box2)
    
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
             return # Cannot proceed

        if active_features_np.shape[0] == 0 or inactive_features_np.shape[0] == 0:
             print("Offline ReID: Feature arrays are empty after conversion.")
             return

        # --- 2. Batch Distance Calculation using GPU ---
        print(f"Offline ReID: Calculating distances between {active_features_np.shape[0]} active and {inactive_features_np.shape[0]} inactive features using GPU.")

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

        # --- 4. Resolve Merge Candidates (Same as your original code) ---
        merge_candidates.sort(key=lambda x: x[2]) # Sort by distance (ascending)
        merged_inactive = set()
        final_merges = {} # {inactive_id_to_remove: active_id_to_keep}

        for active_id, inactive_id, score in merge_candidates:
            if inactive_id not in merged_inactive and active_id not in final_merges.values():
                 print(f"Offline ReID: Merging inactive ID {inactive_id} into active ID {active_id} (distance: {score:.4f})")
                 final_merges[inactive_id] = active_id
                 merged_inactive.add(inactive_id)

        # --- 5. Apply the Merges ---
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

            # Ensure kept ID is not marked inactive
            self.inactive_ids.discard(keep_id)
    
    @profile_function
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

    @profile_function
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
            
            # Calculate spatial distance - use Numba if available
            if NUMBA_AVAILABLE:
                spatial_dist = calculate_spatial_distance_numba(
                    np.array([current_x, current_y]), 
                    np.array([pred_x, pred_y])
                )
            else:
                spatial_dist = np.sqrt((pred_x - current_x)**2 + (pred_y - current_y)**2)

            # If too far away, increase matching threshold
            if spatial_dist > 200:  # pixels
                reid_threshold *= 0.75  # Make matching harder if spatially distant

        return distance < reid_threshold

    @profile_function
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

    @profile_function
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