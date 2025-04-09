# Hybrid Long-Term Multi-Object Tracking (HLT-MOT)

## Overview

This project implements a robust multi-object tracking system designed to maintain consistent object identities over extended periods, particularly addressing challenges like occlusions and re-appearance. It combines the efficiency of short-term tracking with the discriminative power of deep learning-based re-identification (Re-ID).

**Core Functionality:**

1.  **Short-Term Tracking:** Utilizes DeepSORT (`deep_sort_realtime`) for frame-to-frame association based on motion (Kalman filter) and appearance similarity (DeepSORT's internal metric).
2.  **Appearance Embedding:** Employs a ResNet50 model (via PyTorch) to extract robust visual appearance features (2048-dim embeddings) for each detected object.
3.  **Online Re-Identification:** When a new track appears, its features are compared against a gallery of features from recently *inactive* tracks. If a strong match (low cosine distance) is found, the existing ID is reassigned, enabling tracking through short occlusions or detection failures.
4.  **Offline Re-Identification Refinement:** Periodically (every `re_id_interval` frames), the system compares active tracks with inactive ones. If strong appearance similarity suggests a fragmented track (same object assigned multiple IDs), the older ID is merged into the current active one, consolidating identity history.
5.  **Temporal Association:** Maintains track history and simple motion prediction (linear velocity) to aid in association, especially as a secondary cue for Re-ID.

## System Architecture Flow

For each frame:

1.  **Detect:** Objects are detected using an upstream detector (e.g., YOLO via `YOLODetector`).
2.  **Update DeepSORT:** Detections are passed to `deep_sort_realtime` for Kalman prediction and association based on motion/appearance.
3.  **Process Tracks:** Confirmed tracks from DeepSORT are processed.
4.  **Extract Features:** For relevant tracks, appearance features are extracted using the `FeatureExtractor` (ResNet50).
5.  **Map IDs & Re-ID:**
    *   If a track ID is new from DeepSORT, attempt **Online Re-ID** against inactive track feature galleries.
    *   Assign a consistent system ID (either re-identified or new).
6.  **Update State:** Add current features to the track's gallery (`feature_gallery`), update last seen frame, and manage active/inactive status.
7.  **Offline Refinement (Periodic):** If `frame_count % re_id_interval == 0`, perform **Offline Re-ID Refinement** to merge fragmented tracks.
8.  **Predict Motion:** Update simple motion predictions for tracks.
9.  **Output:** Return list of active tracks: `[x1, y1, x2, y2, consistent_id, class_id]`.

## Key Components

*   **`HybridTracker`:** Main class managing the overall tracking logic, integrating DeepSORT and the custom Re-ID mechanisms.
*   **`FeatureExtractor`:** Handles loading the ResNet50 model, image preprocessing (resize, normalize), and extracting L2-normalized feature embeddings using PyTorch.
*   **`YOLODetector`:** Wrapper for object detection (supports `ultralytics` YOLOv8+ or OpenCV DNN YOLOv4).
*   **`deep_sort_realtime`:** External library providing the underlying DeepSORT algorithm.

## Dependencies

*   Python 3.7+
*   OpenCV (`opencv-python`)
*   NumPy
*   PyTorch (`torch`, `torchvision`)
*   SciPy
*   Pillow (`PIL`)
*   `deep_sort_realtime`
*   `ultralytics` (Optional, recommended for easy YOLOv8 usage)

Install using pip:

```bash
pip install opencv-python numpy torch torchvision scipy Pillow deep_sort_realtime ultralytics
```

## Configuration

Key parameters for `HybridTracker`:

*   `max_cosine_distance` (float, default: 0.4): DeepSORT's appearance threshold (lower = stricter match).
*   `max_age` (int, default: 30): Max frames DeepSORT keeps a track without association. *Crucial for occlusion handling.* (Dynamically calculated in `main()` based on FPS and target occlusion time).
*   `min_confidence` (float, default: 0.3): Minimum detection confidence threshold.
*   `re_id_interval` (int, default: 50): How often (in frames) to run offline Re-ID refinement.
*   `gallery_size` (int, default: 100): Max features stored per track for custom Re-ID.
*   `model_path` (str | None): Optional path to a custom Re-ID model (ResNet50 architecture assumed). Uses torchvision's default if `None`.

## Usage

1.  **Clone:** `git clone <repository-url> && cd <repository-directory>`
2.  **Install:** `pip install -r requirements.txt` (if provided, otherwise use the command above)
3.  **Setup:**
    *   Ensure YOLO model weights are available (e.g., `yolov8n.pt` will be downloaded by `ultralytics`) or configure paths in `YOLODetector`.
    *   Place your input video as `video.mp4` or modify `video_path` in `main()`. Webcam 0 is used if the file isn't found.
4.  **Run:** `python your_script_name.py` (replace with the actual script file name)
5.  **Interact:**
    *   Press 'q' to quit the display window.
    *   Press 's' to toggle saving the output video to `output.avi`.

## Limitations

*   **Computational Load:** Detection + DeepSORT + ResNet50 feature extraction is demanding.
*   **Parameter Tuning:** Performance depends on careful tuning of `max_age`, distance thresholds, etc.
*   **Detector Reliant:** Tracking quality is capped by the upstream detector's performance.
*   **Appearance Similarity:** Very similar-looking objects can still cause ID switches.

## Potential Future Enhancements

*   Fine-tune the ResNet50 Re-ID model on domain-specific data (e.g., person or vehicle datasets).
*   Implement more advanced motion models.
*   Explore graph-based optimization for offline association.