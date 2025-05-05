"""
Feature extraction module for person re-identification
"""

import os
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2
import time
from PIL import Image
from torchvision.transforms import transforms
from torchvision.models import ResNet50_Weights

from profiling import profile_function

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
        
        # Measure specifically the model inference time
        inference_start = time.time()
        
        # Extract features with DINOv2 in a single forward pass
        with torch.no_grad():
            features_batch = self.model(batch)
            features_batch = F.normalize(features_batch, p=2, dim=1).cpu().numpy()
            
        # Calculate inference time
        inference_time = time.time() - inference_start
        print(f"DINO feature extraction: {inference_time*1000:.1f}ms for {len(crops)} objects")
            
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
        
    # Track computation time
    start_time = time.time()
    
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
    
    # Calculate computation time for large batches
    compute_time = time.time() - start_time
    num_comparisons = features1.shape[0] * features2.shape[0]
    
    # Only print for significant computations (more than 1000 comparisons)
    if num_comparisons > 1000:
        print(f"Feature distance computation: {compute_time*1000:.1f}ms for {num_comparisons} comparisons")
        
    # Return as numpy array
    return distance.cpu().numpy()