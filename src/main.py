"""
Main application for long-term object tracking system
"""

import os
import cv2
import cProfile
import datetime
import platform
import time
import argparse
import torch # Add torch import
import numpy as np

# Video scaling constants
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
MAINTAIN_ASPECT_RATIO = True  # Set to False to stretch/squash to exact dimensions

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

def scale_frame(frame, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT, maintain_aspect_ratio=MAINTAIN_ASPECT_RATIO):
    """
    Scale frame to target resolution with optional aspect ratio preservation.
    
    Args:
        frame: Input frame to scale
        target_width: Target width in pixels
        target_height: Target height in pixels  
        maintain_aspect_ratio: If True, preserves aspect ratio with padding/cropping
                              If False, stretches/squashes to exact dimensions
    
    Returns:
        Scaled frame
    """
    if maintain_aspect_ratio:
        # Calculate scaling factor to fit within target dimensions
        h, w = frame.shape[:2]
        scale = min(target_width / w, target_height / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create black canvas of target size
        scaled_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate centering offsets
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        # Place resized frame in center of canvas
        scaled_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return scaled_frame
    else:
        # Simply resize to exact dimensions (may distort aspect ratio)
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

def select_tracking_target(cap, detector, is_live):
    """
    Interactive target-selection phase shown before tracking starts.

    The user is presented with the YOLO detection boxes drawn on the frame and can:
      - Click a detection box with the mouse to select that person (preferred).
      - Press a number key 1-9 to select by the index shown on each box.
      - Press 'n' (video files only) to advance to the next frame.
      - Press 'q' to quit the application (returns None).

    For video files the frame is paused until the user acts.
    For live cameras the feed updates in real time until the user clicks/presses a key.

    Returns:
        list [x1, y1, x2, y2] of the selected detection, or None if the user quit.
    """
    WIN = "Hybrid Tracking"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # Shared mutable state for the mouse callback (list so the nested closure can write to it)
    click_pos: list = [None]  # click_pos[0] is (x, y) after a left-click, else None

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)

    cv2.setMouseCallback(WIN, _on_mouse)

    print("\n=== Target Selection ===")
    print("Click on the person you want to track.")
    if not is_live:
        print("Press 'n' to advance to the next frame.")
    print("Press 1-9 to select by index number shown on each box.")
    print("Press 'q' to quit.")
    print("========================\n")

    current_detections = []  # detections on the currently displayed frame
    current_frame = None

    while True:
        # For live cameras always grab a fresh frame; for video files only advance when
        # the user presses 'n' (or we don't have a frame yet).
        if is_live or current_frame is None:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached during target selection. Quitting.")
                return None
            frame = scale_frame(frame)
            current_frame = frame
            current_detections = detector.detect(frame)

        display = current_frame.copy()

        # Draw detection boxes with index labels
        for idx, det in enumerate(current_detections):
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, _ = det[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"{idx + 1}: {conf:.2f}"
            cv2.putText(display, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Instruction overlay
        instruction = "Click or press 1-9 to select target"
        if not is_live:
            instruction += "  |  n = next frame"
        cv2.putText(display, instruction, (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        display = resize_for_display(display, max_width=1280, max_height=720)
        cv2.imshow(WIN, display)

        # Compute display-to-frame scale factors for click mapping
        dh, dw = display.shape[:2]
        fh, fw = current_frame.shape[:2]
        sx = fw / dw
        sy = fh / dh

        # Check for a pending mouse click
        if click_pos[0] is not None:
            cx, cy = click_pos[0]
            click_pos[0] = None
            # Map click back to original frame coordinates
            fx, fy = cx * sx, cy * sy
            for det in current_detections:
                if len(det) < 6:
                    continue
                x1, y1, x2, y2 = det[:4]
                if x1 <= fx <= x2 and y1 <= fy <= y2:
                    print(f"Selected detection at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                    return [x1, y1, x2, y2]
            print("Click did not land on any detection. Try again.")

        # Keyboard: video pauses (waitKey(0)); live streams (waitKey(30))
        wait_ms = 30 if is_live else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):
            return None
        elif key == ord('n') and not is_live:
            # Advance to next frame
            ret, frame = cap.read()
            if not ret:
                print("End of video reached. Quitting.")
                return None
            current_frame = scale_frame(frame)
            current_detections = detector.detect(current_frame)
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(current_detections):
                det = current_detections[idx]
                x1, y1, x2, y2 = det[:4]
                print(f"Selected detection {idx + 1} at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                return [x1, y1, x2, y2]
            else:
                print(f"No detection at index {idx + 1}. Only {len(current_detections)} detected.")


def main():
    """
    Main function to demonstrate the hybrid tracker
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Long-term object tracking system')
    parser.add_argument('--use-tensorrt', action='store_true', help='Use TensorRT for accelerated inference')
    parser.add_argument('--model-path', type=str, default=None, help='Path to YOLO model (default: uses yolov8n-seg.pt)')
    parser.add_argument('--video-path', type=str, default='./left_view.mp4', help='Path to video file (default: ./left_view.mp4)')
    parser.add_argument('--device', type=str, default='cuda', help='Computing device (cuda or cpu, default: cuda)')
    parser.add_argument('--conf-threshold', type=float, default=0.3, help='Confidence threshold for detection (default: 0.3)')
    parser.add_argument('--dino-model', type=str, default='dinov2_vitb14_reg',
                        help='Feature extractor model. DINOv2: dinov2_vits14, dinov2_vitb14, '
                             'dinov2_vitb14_reg, dinov2_vitl14, dinov2_vitg14. '
                             'DINOv3: dinov3_vits16, dinov3_vitb16, dinov3_vitl16. '
                             '(default: dinov2_vitb14_reg)')
    args = parser.parse_args()
    
    # Initialize the profiler at the top of main
    profiler.enable()
    
    # Set up signal handler for graceful exit
    setup_signal_handler()
    
    # Initialize video capture - handle WSL path issues
    video_path = args.video_path  # Use command line argument

    # If still not found, use webcam
    is_live = not os.path.exists(video_path)
    if is_live:
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
    
    print("\n=== Controls ===")
    print("q - Quit application")
    print("s - Toggle video saving")
    print("p - Toggle profiling output")
    print("y - Toggle YOLO detection debugging (shows raw detections)")
    print("i - Save screenshot")
    print("--- Target selection (shown before tracking) ---")
    print("Mouse click - Select target by clicking on a detection box")
    print("1-9         - Select target by index number shown on box")
    if not is_live:
        print("n           - Advance to next frame during selection")
    print("================\n")
    
    # Initialize object detector
    try:
        # Check if CUDA is available when using cuda device
        if args.device == 'cuda':
            if torch.cuda.is_available():
                print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
                device = 'cuda'
            else:
                print("Warning: CUDA requested but not available via torch. Falling back to CPU.")
                device = 'cpu'
        else:
            device = args.device
        # device = 'cuda'    
        # Initialize detector with command line arguments
        detector = YOLODetector(
            model_path=args.model_path,
            conf_threshold=args.conf_threshold, 
            device=device, 
            use_tensorrt=args.use_tensorrt
        )
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
        nn_budget=1000,                # Keep or increase if memory allows
        max_age=final_max_age,         # Use dynamically calculated max_age
        min_confidence=0.5,
        re_id_interval=2,              # Set to run re-ID frequently since DINOv2 is powerful
        gallery_size=5000,             # Keep or increase if needed
        iou_threshold=0.3,             # Adjust based on testing
        dino_model=args.dino_model,
    )

    # --- Target selection phase ---
    # Let the user choose which person to track before the main loop starts.
    selected_bbox = select_tracking_target(cap, detector, is_live)
    if selected_bbox is None:
        # User pressed 'q' during selection – exit gracefully
        cap.release()
        cv2.destroyAllWindows()
        return
    tracker.set_primary_object_by_bbox(selected_bbox)
    # ------------------------------

    # Define color for ID1 (primary object)
    id1_color = (0, 255, 0)  # Green color for primary object
    
    # Initialize video writer if needed
    save_video = True
    video_writer = None
    
    # Initialize debug settings
    show_yolo_debug = True  # Enable YOLO detection debugging by default
    
    # Initialize performance tracking
    frame_times = []
    max_frame_times = 30  # Store last 30 frames for moving average
    frame_count = 0
    fps_display = 0
    profiling_enabled = True  # Per-frame profiling starts enabled
    
    # Set the terminal printing state
    set_print_per_frame_stats(profiling_enabled)
    
    # Record overall start time for FPS calculation
    overall_start_time = time.time()
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Scale the input frame to target resolution
        frame = scale_frame(frame)
        
        frame_count += 1
        
        # Get detections - ensure proper method signature for mock detector
        detections = detector.detect(frame)
        
        # Update tracker
        tracks = tracker.update(frame, detections)
        
        # Create a copy for visualization
        display_frame = frame.copy()
        
        # Visualize raw YOLO detections for debugging (in blue)
        if show_yolo_debug:
            yolo_debug_color = (255, 0, 0)  # Blue color for YOLO detections
            for detection in detections:
                # Detection format: [x1, y1, x2, y2, confidence, class_id, mask_tensor_or_None]
                x1, y1, x2, y2, conf, class_id = detection[:6]
                
                # Draw YOLO detection bounding box
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), yolo_debug_color, 2)
                
                # Draw confidence score
                text = f"YOLO: {conf:.2f}"
                cv2.putText(display_frame, text, (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, yolo_debug_color, 2)
        
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
        
        # Display YOLO debug status
        yolo_debug_text = f"YOLO Debug: {'ON' if show_yolo_debug else 'OFF'}"
        cv2.putText(display_frame, yolo_debug_text, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if show_yolo_debug else (0, 0, 255), 2)
        
        # Display status about primary object tracking
        primary_status = "Primary Object: "
        if tracker.primary_object_active:
            primary_status += "TRACKING"
            cv2.putText(display_frame, primary_status, (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            primary_status += "LOST"
            cv2.putText(display_frame, primary_status, (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
          # Initialize video writer on first frame if saving
        if save_video and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
            output_path = 'output.mp4'  # Change extension to mp4
            video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, 
                                          (frame.shape[1], frame.shape[0]))
            print(f"Saving video to: {output_path}")
        
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
        elif key == ord('y'):  # Toggle YOLO debug visualization
            show_yolo_debug = not show_yolo_debug
            print(f"YOLO debug visualization: {'ENABLED' if show_yolo_debug else 'DISABLED'}")
        elif key == ord('i'):  # Save screenshot
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            screenshot_path = os.path.join(output_dir, f"screenshot_{timestamp}.png")
            cv2.imwrite(screenshot_path, display_frame)
            print(f"Screenshot saved to {screenshot_path}")
    
    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Display overall FPS summary
    if frame_count > 0:
        overall_end_time = time.time()
        total_processing_time = overall_end_time - overall_start_time
        avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total frames processed: {frame_count}")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print("="*50)
    
    # Save and display profiling results
    save_profiling_data(profiler)

if __name__ == "__main__":
    main()