"""
Utility functions for object tracking and visualization
"""

import os
import cv2
import signal
from profiling import display_profiling_stats

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

def setup_signal_handler():
    """Register signal handler for graceful termination"""
    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Exiting gracefully...")
        # Display profiling stats on exit
        display_profiling_stats()
        exit(0)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

def calculate_max_age_from_fps(fps, target_occlusion_seconds=10):
    """
    Calculate appropriate max_age parameter for tracking based on video FPS and desired occlusion time
    
    Args:
        fps: Frames per second of the video
        target_occlusion_seconds: Desired occlusion handling time in seconds
        
    Returns:
        Appropriate max_age value for the tracker
    """
    # Calculate frames for the target occlusion time
    calculated_max_age = int(target_occlusion_seconds * fps)
    
    # Add a buffer (10% or so)
    buffer_frames = int(0.1 * calculated_max_age)
    final_max_age = calculated_max_age + buffer_frames
    
    # Bounds checking for very low/high FPS
    final_max_age = max(60, min(final_max_age, 500))
    
    print(f"Setting DeepSORT max_age to {final_max_age} frames for ~{target_occlusion_seconds}s occlusion at {fps:.2f} FPS.")
    return final_max_age

def create_mock_detector(cap):
    """
    Create a mock detector that simulates detections (for testing without a real detector)
    
    Args:
        cap: Video capture object to get frame dimensions and position
        
    Returns:
        Mock detector object with a detect method
    """
    def mock_detect(frame):
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
    
    # Create a mock detector object with the detect method
    mock_detector = type('', (), {'detect': mock_detect})()
    return mock_detector