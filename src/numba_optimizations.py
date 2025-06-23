"""
Numba-accelerated optimizations for various computations used in object tracking
"""

import numpy as np

# Check if Numba is available
try:
    import numba
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
    print("Numba available - using JIT compilation for performance optimization")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - performance optimizations disabled. Consider installing with: pip install numba")

# Define Numba-accelerated functions only if Numba is available
if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def calculate_iou_numba(box1, box2):
        """
        Numba-accelerated IoU calculation between two bounding boxes
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

    @njit(fastmath=True)
    def calculate_spatial_distance_numba(point1, point2):
        """
        Numba-accelerated calculation of Euclidean distance between two points
        Args:
            point1: First point as (x, y)
            point2: Second point as (x, y)
        Returns:
            Euclidean distance
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    @njit(fastmath=True)
    def find_min_value_index_numba(matrix):
        """
        Numba-accelerated function to find minimum value and its indices in matrix
        Args:
            matrix: 2D numpy array
        Returns:
            (min_value, i, j): tuple of minimum value and its indices
        """
        min_value = float('inf')
        min_i, min_j = -1, -1
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] < min_value:
                    min_value = matrix[i, j]
                    min_i, min_j = i, j
        
        return min_value, min_i, min_j
    
    @njit(fastmath=True, parallel=True)
    def compute_cosine_distance_cpu_numba(features1, features2, threshold=1.0):
        """
        Numba-accelerated version for computing cosine distance on CPU
        Args:
            features1: First set of feature vectors (numpy array)
            features2: Second set of feature vectors (numpy array)
            threshold: Maximum distance threshold
        Returns:
            Distance matrix (numpy array)
        """
        n1 = features1.shape[0]
        n2 = features2.shape[0]
        result = np.zeros((n1, n2), dtype=np.float32)
        
        # Ensure features are normalized
        features1_norm = np.zeros_like(features1)
        features2_norm = np.zeros_like(features2)
        
        # Normalize features1
        for i in prange(n1):
            norm1 = 0.0
            for k in range(features1.shape[1]):
                norm1 += features1[i, k] * features1[i, k]
            norm1 = np.sqrt(norm1)
            if norm1 > 1e-6:  # Avoid division by zero
                for k in range(features1.shape[1]):
                    features1_norm[i, k] = features1[i, k] / norm1
            else:
                for k in range(features1.shape[1]):
                    features1_norm[i, k] = 0.0
        
        # Normalize features2
        for j in prange(n2):
            norm2 = 0.0
            for k in range(features2.shape[1]):
                norm2 += features2[j, k] * features2[j, k]
            norm2 = np.sqrt(norm2)
            if norm2 > 1e-6:  # Avoid division by zero
                for k in range(features2.shape[1]):
                    features2_norm[j, k] = features2[j, k] / norm2
            else:
                for k in range(features2.shape[1]):
                    features2_norm[j, k] = 0.0
        
        # Calculate cosine similarity and convert to distance
        for i in prange(n1):
            for j in range(n2):
                dot_product = 0.0
                for k in range(features1.shape[1]):
                    dot_product += features1_norm[i, k] * features2_norm[j, k]
                
                # Convert similarity to distance
                distance = 1.0 - dot_product
                
                # Apply threshold
                if threshold < 1.0 and distance > threshold:
                    distance = threshold
                
                result[i, j] = distance
                
        return result

# Fallback implementations for when Numba is not available
def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes (fallback for when Numba is not available)
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

def calculate_spatial_distance(point1, point2):
    """
    Calculate Euclidean distance between two points (fallback for when Numba is not available)
    Args:
        point1: First point as (x, y)
        point2: Second point as (x, y)
    Returns:
        Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)