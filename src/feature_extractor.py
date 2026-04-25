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

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry: (backend, hub_repo_or_hf_id, entry_or_none, feature_dim)
#   backend 'torch_hub' : torch.hub.load(repo, entry)
#   backend 'hf'        : transformers AutoModel.from_pretrained(hf_id)
# ---------------------------------------------------------------------------
DINO_MODEL_REGISTRY = {
    # DINOv2 – loaded via torch.hub (no local clone required)
    "dinov2_vits14":     ("torch_hub", "facebookresearch/dinov2", "dinov2_vits14",     384),
    "dinov2_vitb14":     ("torch_hub", "facebookresearch/dinov2", "dinov2_vitb14",     768),
    "dinov2_vitb14_reg": ("torch_hub", "facebookresearch/dinov2", "dinov2_vitb14_reg", 768),
    "dinov2_vitl14":     ("torch_hub", "facebookresearch/dinov2", "dinov2_vitl14",    1024),
    "dinov2_vitg14":     ("torch_hub", "facebookresearch/dinov2", "dinov2_vitg14",    1536),
    # DINOv3 – loaded via Hugging Face Transformers (no local clone required)
    # https://github.com/facebookresearch/dinov3
    "dinov3_vits16": ("hf", "facebook/dinov3-vits16-pretrain-lvd1689m", None, 384),
    "dinov3_vitb16": ("hf", "facebook/dinov3-vitb16-pretrain-lvd1689m", None, 768),
    "dinov3_vitl16": ("hf", "facebook/dinov3-vitl16-pretrain-lvd1689m", None, 1024),
}

DEFAULT_DINO_MODEL = "dinov2_vitb14_reg"


class FeatureExtractor:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu',
                 dino_model: str = DEFAULT_DINO_MODEL):
        """
        Initialize the feature extractor.

        Args:
            model_path: Path to a fully custom saved model file (overrides dino_model).
            device: Device to run the model on ('cuda' or 'cpu').
            dino_model: Short name from DINO_MODEL_REGISTRY, e.g. 'dinov2_vitb14_reg'
                        or 'dinov3_vitb16'.  Ignored when model_path is provided.
        """
        self.device = device
        self._hf_processor = None  # set for HuggingFace-backend models
        print(f"Using device: {self.device}")

        # ---- Custom model file takes highest priority ----------------
        if model_path is not None and os.path.exists(model_path):
            print(f"Loading custom model from {model_path}")
            try:
                self.model = torch.load(model_path, map_location=self.device)
                self.feature_dim = self.model.embed_dim if hasattr(self.model, 'embed_dim') else 384
                self.model = self.model.to(self.device)
                self.model.eval()
                self._build_standard_transform()
                print(f"Feature extractor initialized with feature dimension: {self.feature_dim}")
                return
            except Exception as e:
                print(f"Error loading custom model: {e}. Falling back to registry model.")

        # ---- Registry lookup ----------------------------------------
        if dino_model not in DINO_MODEL_REGISTRY:
            print(f"Unknown dino_model '{dino_model}'. Available: {list(DINO_MODEL_REGISTRY)}. "
                  f"Falling back to '{DEFAULT_DINO_MODEL}'.")
            dino_model = DEFAULT_DINO_MODEL

        backend, repo_or_id, entry, feature_dim = DINO_MODEL_REGISTRY[dino_model]
        self.feature_dim = feature_dim

        try:
            if backend == "torch_hub":
                print(f"Loading {dino_model} via torch.hub ({repo_or_id} :: {entry})")
                self.model = torch.hub.load(repo_or_id, entry)
                self._build_standard_transform()
            else:  # 'hf'
                print(f"Loading {dino_model} via Hugging Face Transformers ({repo_or_id})")
                from transformers import AutoImageProcessor, AutoModel, ViTImageProcessor
                try:
                    self._hf_processor = AutoImageProcessor.from_pretrained(repo_or_id)
                except Exception as proc_err:
                    # DINOv3 uses 'DINOv3ViTImageProcessorFast' which may not be registered
                    # in the installed transformers version; fall back to ViTImageProcessor
                    # with the same standard ImageNet parameters.
                    print(f"AutoImageProcessor failed ({proc_err.__class__.__name__}), "
                          f"using ViTImageProcessor with standard ImageNet settings")
                    self._hf_processor = ViTImageProcessor(
                        size={"height": 224, "width": 224},
                        image_mean=[0.485, 0.456, 0.406],
                        image_std=[0.229, 0.224, 0.225],
                        do_resize=True,
                        do_normalize=True,
                        do_rescale=True,
                    )
                self.model = AutoModel.from_pretrained(repo_or_id)
                # HF models ship with their own preprocessor; no manual transform needed.
                self.transform = None

            self.model = self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"Error loading {dino_model}: {e}")
            print("Falling back to ResNet50 model")
            self.model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
            self.model = self.model.to(self.device)
            self.model.eval()
            self._build_standard_transform()

        print(f"Feature extractor initialized with feature dimension: {self.feature_dim}")

    def _build_standard_transform(self):
        """Standard ImageNet transform used by DINOv2 and the ResNet fallback."""
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    @profile_function
    def extract_features_batch(self, frame, bboxes, masks=None):
        """
        Extract features for multiple bounding boxes in a single GPU operation, applying masks if provided.
        
        Args:
            frame: Current video frame (BGR format)
            bboxes: List of bounding boxes as [x1, y1, x2, y2]
            masks: Optional list of mask tensors corresponding to bboxes. Mask is applied if not None.
            
        Returns:
            Batch of feature vectors (numpy array)
        """
        if not bboxes:
            return []
            
        crops_transformed = []
        valid_indices = []
        
        # Ensure masks list has the same length as bboxes if provided, padding with None if necessary
        if masks is None:
            masks = [None] * len(bboxes)
        elif len(masks) < len(bboxes):
            masks.extend([None] * (len(bboxes) - len(masks)))

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
                if crop.size == 0:
                    continue

                current_mask = masks[i]
                if current_mask is not None:
                    try:
                        mask_np = current_mask.cpu().numpy().astype(np.uint8) # Assuming mask is a PyTorch tensor
                        # Resize mask to crop dimensions
                        mask_resized = cv2.resize(mask_np, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        # Ensure mask is binary (0 or 1) then scale to 0 or 255 for bitwise_and
                        binary_mask = (mask_resized > 0).astype(np.uint8) * 255

                        if len(crop.shape) == 3: # Color image
                            # Convert single channel mask to 3 channels
                            mask_3channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                            crop = cv2.bitwise_and(crop, mask_3channel)
                        elif len(crop.shape) == 2: # Grayscale image (should not happen with BGR frame)
                             crop = cv2.bitwise_and(crop, binary_mask)
                    except Exception as e:
                        print(f"Error applying mask: {e}. Proceeding without mask for this crop.")
                
                # Convert crop to RGB and then to PIL Image
                img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if self._hf_processor is not None:
                    # HF backend: store PIL images; will batch-process below
                    crops_transformed.append(img_pil)
                else:
                    crops_transformed.append(self.transform(img_pil))
                valid_indices.append(i)
                
            except Exception as e:
                # print(f"Error processing bbox {bbox}: {e}") # Optional: for debugging
                continue
                
        # Process all crops in a single batch
        if not crops_transformed:
            return [np.zeros(self.feature_dim, dtype=np.float32)] * len(bboxes)

        inference_start = time.time()

        if self._hf_processor is not None:
            # HuggingFace backend: use AutoImageProcessor to build pixel_values tensor
            inputs = self._hf_processor(images=crops_transformed, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                # Use pooler_output (CLS token) as the embedding
                features_batch = F.normalize(outputs.pooler_output, p=2, dim=1).cpu().numpy()
        else:
            # torch.hub / standard backend
            batch = torch.stack(crops_transformed).to(self.device)
            with torch.no_grad():
                features_batch = self.model(batch)
                features_batch = F.normalize(features_batch, p=2, dim=1).cpu().numpy()
            
        # Calculate inference time
        inference_time = time.time() - inference_start
        # print(f"DINO feature extraction: {inference_time*1000:.1f}ms for {len(crops_transformed)} objects") # Optional
            
        # If only one crop, ensure we have correct dimensions
        if len(crops_transformed) == 1 and len(features_batch.shape) == 2 : # check if features_batch is not already (1, dim)
             pass # features_batch is already (1, dim)
        elif len(crops_transformed) == 1 and len(features_batch.shape) == 1: # if it was (dim,)
             features_batch = features_batch.reshape(1, -1)


        # Create result array with zeros for invalid bboxes
        result = [np.zeros(self.feature_dim, dtype=np.float32) for _ in range(len(bboxes))]
        for i, valid_idx in enumerate(valid_indices):
            if i < len(features_batch):
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