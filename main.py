"""
Main application for long-term object tracking system
"""

import os
import cv2
import cProfile
import datetime
import platform
import time

# Import from our modules
from profiling import profiler, save_profiling_data, display_profiling_stats, set_print_per_frame_stats
from feature_extractor import FeatureExtractor
from hybrid_tracker import HybridTracker
from yolo_detector import YOLODetector
from utils import (
    resize_for_display, 
    setup_signal_handler, 
    calculate_max_age_from_fps,
    create_mock_detector
)

def main():
    """
    Main function to demonstrate the hybrid tracker
    """
    # Initialize the profiler at the top of main
    profiler.enable()
    
    # Set up signal handler for graceful exit
    setup_signal_handler()
    
    # Initialize video capture - handle WSL path issues
    video_path = "./MOT_edited.mp4"  # Default path
    
    # If still not found, use webcam
    if not os.path.exists(video_path):
        print(f"Video file could not be found. Using webcam instead.")
        cap = cv2.VideoCapture(0)
    else:
        print(f"Opening video from: {video_path}")
        cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    # Get video properties
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # Handle case where FPS is not available (e.g., some webcams)
            print("Warning: Could not determine video FPS. Assuming 30 FPS for max_age calculation.")
            fps = 30
    except Exception:
        print("Warning: Error getting video FPS. Assuming 30 FPS for max_age calculation.")
        fps = 30
    
    print(f"Video FPS: {fps:.2f}")
    
    # Initialize object detector
    try:
        detector = YOLODetector(conf_threshold=0.3, device='cuda')
    except Exception as e:
        print(f"Error initializing YOLO detector: {e}")
        print("Falling back to mock detector")
        
        # Create mock detector as fallback - fix the calling issue
        detector = create_mock_detector(cap)
    
    # Calculate max_age for occlusion handling
    final_max_age = calculate_max_age_from_fps(fps, target_occlusion_seconds=10)
    
    # Initialize tracker
    tracker = HybridTracker(
        max_cosine_distance=0.15,      # Reduced threshold for DINOv2 features
        nn_budget=2000,                # Keep or increase if memory allows
        max_age=final_max_age,         # Use dynamically calculated max_age
        min_confidence=0.3,
        re_id_interval=1,              # Set to run re-ID frequently since DINOv2 is powerful
        gallery_size=50,             # Keep or increase if needed
        iou_threshold=0.25              # Adjust based on testing
    )
    
    # Define color for ID1 (primary object)
    id1_color = (0, 255, 0)  # Green color for primary object
    
    # Initialize video writer if needed
    save_video = True
    video_writer = None
    
    # Initialize performance tracking
    frame_times = []
    max_frame_times = 30  # Store last 30 frames for moving average
    frame_count = 0
    fps_display = 0
    profiling_enabled = True  # Per-frame profiling starts enabled
    
    # Set the terminal printing state
    set_print_per_frame_stats(profiling_enabled)
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get detections - ensure proper method signature for mock detector
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
        
        # Calculate fps
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        # Keep only last max_frame_times
        if len(frame_times) > max_frame_times:
            frame_times.pop(0)
        
        # Update fps every 5 frames
        if frame_count % 5 == 0:
            if frame_times:
                avg_frame_time = sum(frame_times) / len(frame_times)
                fps_display = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Draw frame count and processing rate
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
        cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display profiling status
        profiling_text = f"Profiling: {'ON' if profiling_enabled else 'OFF'}"
        cv2.putText(display_frame, profiling_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if profiling_enabled else (0, 0, 255), 2)
        
        # Display status about primary object tracking
        primary_status = "Primary Object: "
        if tracker.primary_object_active:
            primary_status += "TRACKING"
            cv2.putText(display_frame, primary_status, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            primary_status += "LOST"
            cv2.putText(display_frame, primary_status, (10, 120), 
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

        # Make the window resizable by the user if needed
        cv2.namedWindow("Hybrid Tracking", cv2.WINDOW_NORMAL)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Toggle saving
            save_video = not save_video
            print(f"Video saving: {'ON' if save_video else 'OFF'}")
        elif key == ord('p'):  # Toggle profiling output
            profiling_enabled = not profiling_enabled
            set_print_per_frame_stats(profiling_enabled)
            print(f"Per-frame profiling output: {'ENABLED' if profiling_enabled else 'DISABLED'}")
    
    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Save and display profiling results
    save_profiling_data(profiler)

if __name__ == "__main__":
    main()