"""
Entry point for the experimental DINOv2-only reID pipeline.

Differences from main.py:
  - Uses SimpleReIDTracker instead of HybridTracker (no DeepSORT)
  - All detected objects are tracked and displayed (not just ID 1)
  - Output video saved to output_simple_reid.mp4
  - No target-selection phase (all objects tracked from the first frame)
"""

import os
import cv2
import datetime
import time
import argparse
import torch
import numpy as np

# Video scaling constants (same as main.py)
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
MAINTAIN_ASPECT_RATIO = True

from profiling import profiler, save_profiling_data, display_profiling_stats, set_print_per_frame_stats
from simple_reid_tracker import SimpleReIDTracker
from yolo_detector import YOLODetector
from utils import resize_for_display, setup_signal_handler


# Colour palette for track IDs (cycles through a fixed set)
_PALETTE = [
    (0, 255, 0),    # green
    (0, 128, 255),  # orange
    (255, 0, 128),  # pink
    (255, 255, 0),  # cyan
    (128, 0, 255),  # purple
    (0, 255, 255),  # yellow
    (255, 128, 0),  # light blue
    (128, 255, 0),  # lime
]


def _id_colour(track_id: int):
    return _PALETTE[(track_id - 1) % len(_PALETTE)]


def scale_frame(frame, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT,
                maintain_aspect_ratio=MAINTAIN_ASPECT_RATIO):
    if maintain_aspect_ratio:
        h, w = frame.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scaled = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_off = (target_height - new_h) // 2
        x_off = (target_width - new_w) // 2
        scaled[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return scaled
    else:
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description='Experimental DINOv2-only reID pipeline')
    parser.add_argument('--use-tensorrt', action='store_true')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to YOLO model (default: yolov8n-seg.pt)')
    parser.add_argument('--video-path', type=str, default='./left_view.mp4')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--conf-threshold', type=float, default=0.3)
    parser.add_argument('--match-threshold', type=float, default=0.35,
                        help='Cosine distance threshold for reID matching (default: 0.35)')
    parser.add_argument('--gallery-size', type=int, default=10,
                        help='Number of feature vectors stored per track (default: 10)')
    args = parser.parse_args()

    profiler.enable()
    setup_signal_handler()

    # ---- Video source ------------------------------------------------
    video_path = args.video_path
    is_live = not os.path.exists(video_path)
    if is_live:
        print("Video file not found – using webcam.")
        cap = cv2.VideoCapture(0)
    else:
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
    except Exception:
        fps = 30
    print(f"Video FPS: {fps:.2f}")

    # ---- Controls ----------------------------------------------------
    print("\n=== Controls ===")
    print("q - Quit")
    print("s - Toggle video saving")
    print("p - Toggle profiling output")
    print("y - Toggle YOLO detection overlay")
    print("i - Save screenshot")
    print("================\n")

    # ---- Detector ----------------------------------------------------
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = 'cpu'
    else:
        device = args.device

    try:
        detector = YOLODetector(
            model_path=args.model_path,
            conf_threshold=args.conf_threshold,
            device=device,
            use_tensorrt=args.use_tensorrt,
        )
    except Exception as e:
        print(f"Error initialising YOLO detector: {e}")
        return

    # ---- Tracker (no DeepSORT) --------------------------------------
    tracker = SimpleReIDTracker(
        match_threshold=args.match_threshold,
        max_gallery_size=args.gallery_size,
        min_confidence=args.conf_threshold,
    )

    # ---- Main loop ---------------------------------------------------
    save_video = True
    video_writer = None
    show_yolo_debug = True
    profiling_enabled = True
    set_print_per_frame_stats(profiling_enabled)

    frame_times = []
    frame_count = 0
    fps_display = 0
    overall_start = time.time()

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = scale_frame(frame)
        frame_count += 1

        detections = detector.detect(frame)
        tracks = tracker.update(frame, detections)

        display_frame = frame.copy()

        # YOLO raw detections (blue)
        if show_yolo_debug:
            for det in detections:
                x1, y1, x2, y2, conf, *_ = det
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                cv2.putText(display_frame, f"YOLO:{conf:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Tracked objects
        for track in tracks:
            x1, y1, x2, y2, track_id, cls_id = track
            colour = _id_colour(track_id)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(display_frame, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

            # Trail
            if track_id in tracker.track_history:
                pts = list(tracker.track_history[track_id])
                for i in range(1, len(pts)):
                    cv2.line(display_frame,
                             (int(pts[i-1][0]), int(pts[i-1][1])),
                             (int(pts[i][0]),   int(pts[i][1])),
                             colour, 2)

        # FPS
        frame_times.append(time.time() - start_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        if frame_count % 5 == 0 and frame_times:
            fps_display = 1.0 / (sum(frame_times) / len(frame_times))

        # HUD
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Tracks: {len(tracks)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Profiling: {'ON' if profiling_enabled else 'OFF'}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if profiling_enabled else (0, 0, 255), 2)
        cv2.putText(display_frame,
                    f"[SimpleReID | thr={args.match_threshold}]",
                    (10, display_frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Video writer
        if save_video and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = 'output_simple_reid.mp4'
            video_writer = cv2.VideoWriter(
                output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0])
            )
            print(f"Saving video to: {output_path}")
        if save_video and video_writer is not None:
            video_writer.write(display_frame)

        display_frame = resize_for_display(display_frame, max_width=1280, max_height=720)
        cv2.namedWindow("SimpleReID Tracking", cv2.WINDOW_NORMAL)
        cv2.imshow("SimpleReID Tracking", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_video = not save_video
            print(f"Video saving: {'ON' if save_video else 'OFF'}")
        elif key == ord('p'):
            profiling_enabled = not profiling_enabled
            set_print_per_frame_stats(profiling_enabled)
            print(f"Profiling: {'ON' if profiling_enabled else 'OFF'}")
        elif key == ord('y'):
            show_yolo_debug = not show_yolo_debug
            print(f"YOLO debug: {'ON' if show_yolo_debug else 'OFF'}")
        elif key == ord('i'):
            os.makedirs("output", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join("output", f"screenshot_{ts}.png")
            cv2.imwrite(path, display_frame)
            print(f"Screenshot saved: {path}")

    # ---- Cleanup -----------------------------------------------------
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        total = time.time() - overall_start
        print(f"\nProcessed {frame_count} frames in {total:.2f}s "
              f"({frame_count / total:.2f} avg FPS)")

    save_profiling_data(profiler)


if __name__ == "__main__":
    main()
